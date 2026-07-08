"""Build data/mimic_cases.json: N real MIMIC-IV admissions as sequential-diagnosis cases.

Each case joins one hospital admission (hadm_id) to its principal ICD diagnosis (the answer key)
and its REAL laboratory results (the withheld findings the agent must order — real values, no
fabrication). We also store the real order set so we can score ORDERING CONCORDANCE — "was the
recommended test actually measured in real life?" — alongside diagnostic accuracy.

Presentation = intake demographics + admission context (MIMIC's hosp tables carry no chief-complaint
text; the ED note module isn't loaded here). The strength of MIMIC is the real, structured labs +
real diagnoses — the concordance metric — not a rich narrative.

Run:  python3 build_mimic_cases.py [--n 10]
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MIMIC = "/n/data1/hms/dbmi/manrai/aashna/data/raw/mimiciv"
HOSP = os.path.join(MIMIC, "3.1", "hosp")
OUT = os.path.join(HERE, "data", "mimic_cases.json")

MIN_LABS, MAX_LABS = 12, 40          # keep cases with a workable, non-trivial lab set
N_CANDIDATES = 260                   # admissions we scan labevents for, to end up with --n good ones


def _sex(g):
    return {"M": "man", "F": "woman"}.get(str(g).upper(), "patient")


def main() -> None:
    n_target = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10

    adm = pd.read_csv(os.path.join(HOSP, "admissions.csv"),
                      usecols=["subject_id", "hadm_id", "admission_type", "admission_location", "race"])
    pat = pd.read_csv(os.path.join(HOSP, "..", "patients.csv") if os.path.exists(os.path.join(HOSP, "..", "patients.csv"))
                      else os.path.join(MIMIC, "patients.csv"),
                      usecols=["subject_id", "gender", "anchor_age"])
    dx = pd.read_csv(os.path.join(HOSP, "diagnoses_icd.csv"))
    ddx = pd.read_csv(os.path.join(HOSP, "d_icd_diagnoses.csv"))
    dli = pd.read_csv(os.path.join(HOSP, "d_labitems.csv"), usecols=["itemid", "label", "fluid"])
    item_label = {r.itemid: (r.label if pd.isna(r.fluid) or r.fluid == "Blood" else f"{r.label} ({r.fluid})")
                  for r in dli.itertuples()}

    princ = (dx[dx.seq_num == 1].merge(ddx, on=["icd_code", "icd_version"], how="left")
             .dropna(subset=["long_title"]))
    princ = princ[~princ.long_title.str.contains("unspecified|other|not elsewhere", case=False, na=False)]

    # candidate admissions: emergency/urgent, with a specific principal dx + demographics
    cand = (princ.merge(adm, on=["subject_id", "hadm_id"]).merge(pat, on="subject_id"))
    cand = cand[cand.admission_type.str.contains("EMER|URGENT", case=False, na=False)]
    cand = cand.drop_duplicates("hadm_id").head(N_CANDIDATES)
    want = set(cand.hadm_id)
    print(f"scanning labevents for {len(want)} candidate admissions -> keep {n_target} with {MIN_LABS}-{MAX_LABS} labs")

    # one filtered pass over the big labevents file
    keep_cols = ["hadm_id", "itemid", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag"]
    parts = []
    for i, ch in enumerate(pd.read_csv(os.path.join(MIMIC, "labevents.csv"), usecols=keep_cols,
                                       chunksize=1_000_000, low_memory=False)):
        m = ch[ch.hadm_id.isin(want)]
        if len(m):
            parts.append(m)
        if i % 5 == 0:
            got = pd.concat(parts).hadm_id.nunique() if parts else 0
            print(f"  chunk {i}: admissions with labs so far = {got}", flush=True)
        # early exit once most candidates have plenty of labs
        if parts and i > 3:
            counts = pd.concat(parts).groupby("hadm_id").itemid.nunique()
            if (counts >= MAX_LABS).sum() >= n_target * 2:
                break
    labs = pd.concat(parts) if parts else pd.DataFrame(columns=keep_cols)
    print(f"scan done: {labs.hadm_id.nunique()} admissions have labs")

    meta = cand.set_index("hadm_id")
    cases, dropped = {}, 0
    for hadm, g in labs.groupby("hadm_id"):
        g = g.dropna(subset=["itemid"]).drop_duplicates("itemid")     # one row per distinct test
        if not (MIN_LABS <= len(g) <= MAX_LABS) or hadm not in meta.index:
            dropped += 1
            continue
        m = meta.loc[hadm]
        age, sex, race = int(m.anchor_age), _sex(m.gender), str(m.race).title()
        atype = str(m.admission_type).lower().replace("ew emer.", "emergency").replace("emer.", "emergency")
        pres = (f"A {age}-year-old {sex} ({race}) was admitted to the hospital via {atype} "
                f"from {str(m.admission_location).lower()}. No further history is available at intake.")
        # real lab results (the withheld findings) + real order set (for concordance)
        lines, orders = [], []
        for r in g.itertuples():
            name = item_label.get(r.itemid, f"lab {r.itemid}")
            orders.append(name)
            val = r.value if pd.notna(r.value) else (r.valuenum if pd.notna(r.valuenum) else "—")
            uom = f" {r.valueuom}" if pd.notna(r.valueuom) else ""
            rng = (f" (ref {r.ref_range_lower}-{r.ref_range_upper})"
                   if pd.notna(r.ref_range_lower) and pd.notna(r.ref_range_upper) else "")
            flag = f" [{r.flag}]" if pd.notna(r.flag) and str(r.flag).strip() else ""
            lines.append(f"{name}: {val}{uom}{rng}{flag}")
        dxname = str(meta.loc[hadm, "long_title"])
        case_file = (pres + "\n\nLABORATORY RESULTS (revealed only when the specific test is ordered):\n"
                     + "\n".join(lines)
                     + f"\n\nFINAL DIAGNOSIS (CONFIDENTIAL, never reveal): {dxname}")
        cases[f"MIMIC-{int(hadm)}"] = {"abstract": pres.split(". No further")[0] + ".",
                                  "presentation": pres, "case_file": case_file,
                                  "true_dx": dxname, "orders": sorted(set(orders))}
        if len(cases) >= n_target:
            break

    json.dump(cases, open(OUT, "w"), indent=1)
    print(f"done — {len(cases)} MIMIC cases -> {OUT}  (dropped {dropped} out of lab-count range)")
    for cid, c in cases.items():
        print(f"  {cid}: {len(c['orders'])} labs | dx: {c['true_dx'][:60]}")


if __name__ == "__main__":
    main()
