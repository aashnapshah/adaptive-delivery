"""Real MIMIC-IV admissions as labs-on-order cases (stage 01, source="mimic").

A MIMIC case is a hospital admission, not a vignette. The turn-0 `presentation` is the ED intake -
demographics plus the chief complaint and triage vitals from the MIMIC-IV-ED module (set MIMIC_ED_ROOT;
the hosp tables alone carry no chief-complaint text, and without it the agent has nothing to aim at and
just orders a broad screen). The substance is the admission's REAL structured lab results, withheld
until the matching test is ordered. Each case also stores the REAL order set so the recommender can be
scored on ordering CONCORDANCE ("was the recommended test actually measured in this admission?").

Two entry points:
  - `build(...)`  : cluster-side. Reads MIMIC-IV CSVs (needs PhysioNet credential + DUA; path via
                    MIMIC_ROOT) and writes data/mimic_cases.json in the structured schema below.
  - `load_mimic()`: reads data/mimic_cases.json into Case(source="mimic") objects. Backward-compatible
                    with the older narrative fixture (parses its LABORATORY RESULTS block into findings).

Structured schema written by build():
  { "MIMIC-<hadm>": { "abstract", "presentation", "true_dx",
                      "findings": { "<test name>": "<result display>", ... },
                      "orders": ["<test name>", ...] } }
"""

from __future__ import annotations

import json
import os
import re

from .schema import Case

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_MIMIC_CASES = os.path.join(DATA_DIR, "mimic_cases.json")

# Cluster location of the raw MIMIC-IV CSVs (PhysioNet credentialed; never redistributed).
MIMIC_ROOT = os.environ.get("MIMIC_ROOT", "/n/data1/hms/dbmi/manrai/aashna/data/raw/mimiciv")

MIN_LABS, MAX_LABS = 12, 40          # keep cases with a workable, non-trivial lab set
N_CANDIDATES = 260                   # admissions we scan labevents for, to end up with --n good ones


# ------------------------------------------------------------------ loader ---

# a LABORATORY RESULTS line in the older narrative fixture: "<name>: <value...>"
_LAB_LINE = re.compile(r"^(?P<name>[^:]+?):\s*(?P<value>.+)$")


def _findings_from_case_file(case_file: str) -> dict[str, str]:
    """Back-compat: pull findings out of an old fixture's 'LABORATORY RESULTS' block."""
    findings: dict[str, str] = {}
    in_labs = False
    for line in (case_file or "").splitlines():
        s = line.strip()
        if s.upper().startswith("LABORATORY RESULTS"):
            in_labs = True
            continue
        if not in_labs:
            continue
        if not s or s.upper().startswith("FINAL DIAGNOSIS"):
            break
        m = _LAB_LINE.match(s)
        if m:
            findings[m.group("name").strip()] = m.group("value").strip()
    return findings


def _to_case(cid: str, rec: dict) -> Case:
    case_file = rec.get("case_file") or ""
    findings = rec.get("findings") or (_findings_from_case_file(case_file) if case_file else {})
    return Case(
        case_id=cid,
        source="mimic",
        true_diagnosis=rec.get("true_dx") or rec.get("true_diagnosis") or "",
        presentation=(rec.get("presentation") or "").strip(),
        abstract=(rec.get("abstract") or "").strip(),
        case_file=case_file,           # the gatekeeper reads this (or the structured findings, if empty)
        findings=dict(findings),
        orders=tuple(rec.get("orders") or findings),   # preserve findings' (charttime) order, don't alphabetize
    )


def load_mimic(path: str = None, limit: int = None) -> list[Case]:
    """Load MIMIC cases from data/mimic_cases.json (new structured OR old narrative fixture)."""
    path = path or DEFAULT_MIMIC_CASES
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    out = [_to_case(cid, rec) for cid, rec in data.items()]
    return out[:limit] if limit else out


# The live registry (real cases if the fixture is present, else empty). MIMIC_LIMIT caps the count.
try:
    _limit = int(os.environ.get("MIMIC_LIMIT", "0")) or None
    MIMIC_CASES: list[Case] = load_mimic(limit=_limit)
except Exception:
    MIMIC_CASES = []

MIMIC_BY_ID: dict[str, Case] = {c.case_id: c for c in MIMIC_CASES}


# ------------------------------------------------------------------ builder --

def _sex(g):
    return {"M": "man", "F": "woman"}.get(str(g).upper(), "patient")


def _ed_intake(pd, want):
    """hadm_id -> "<chief complaint>. Triage: <vitals>", from the MIMIC-IV-ED module.

    Without this the presentation is demographics only ("admitted via emergency, no further history"),
    which gives the recommender nothing to aim at - so it orders a broad screen and ~75% of what it
    orders was never drawn. The chief complaint + triage vitals are what a clinician actually has at
    intake, and they don't leak the diagnosis. Returns {} if the ED module isn't present.
    """
    ed = os.environ.get("MIMIC_ED_ROOT") or os.path.join(MIMIC_ROOT, "..", "mimic-iv-ed", "2.2", "ed")
    stays_p, triage_p = os.path.join(ed, "edstays.csv"), os.path.join(ed, "triage.csv")
    if not (os.path.exists(stays_p) and os.path.exists(triage_p)):
        print(f"  (no MIMIC-IV-ED at {ed} - falling back to a demographics-only presentation)")
        return {}
    stays = pd.read_csv(stays_p, usecols=["hadm_id", "stay_id"]).dropna(subset=["hadm_id"])
    stays = stays[stays.hadm_id.isin(want)]
    triage = pd.read_csv(triage_p, usecols=["stay_id", "chiefcomplaint", "temperature", "heartrate",
                                            "resprate", "o2sat", "sbp", "dbp", "pain"])
    m = stays.merge(triage, on="stay_id", how="left").drop_duplicates("hadm_id")

    def line(r):
        cc = str(r.chiefcomplaint).strip()
        if not cc or cc.lower() == "nan":
            return None
        vit = [(f"T {r.temperature:.1f}", pd.notna(r.temperature)), (f"HR {r.heartrate:.0f}", pd.notna(r.heartrate)),
               (f"RR {r.resprate:.0f}", pd.notna(r.resprate)), (f"SpO2 {r.o2sat:.0f}%", pd.notna(r.o2sat)),
               (f"BP {r.sbp:.0f}/{r.dbp:.0f}", pd.notna(r.sbp) and pd.notna(r.dbp))]
        vit = ", ".join(v for v, ok in vit if ok)
        return f"{cc.lower()}" + (f". Triage vitals: {vit}." if vit else ".")
    return {int(r.hadm_id): line(r) for r in m.itertuples() if line(r)}


def build(n_target: int = 10, out: str = None) -> dict:
    """Cluster-side: build data/mimic_cases.json from raw MIMIC-IV CSVs. Needs pandas + MIMIC_ROOT.

    Each kept admission -> one structured case: thin intake presentation, real labs as withheld
    `findings` (revealed only when ordered), and the real `orders` set for concordance.
    """
    import pandas as pd

    out = out or DEFAULT_MIMIC_CASES
    hosp = os.path.join(MIMIC_ROOT, "3.1", "hosp")
    if not os.path.isdir(hosp):
        raise FileNotFoundError(
            f"MIMIC-IV not found at {hosp}. Set MIMIC_ROOT to your credentialed extract "
            "(run this on the cluster, not on a laptop).")

    adm = pd.read_csv(os.path.join(hosp, "admissions.csv"),
                      usecols=["subject_id", "hadm_id", "admission_type", "admission_location", "race"])
    pat_path = os.path.join(hosp, "..", "patients.csv")
    pat = pd.read_csv(pat_path if os.path.exists(pat_path) else os.path.join(MIMIC_ROOT, "patients.csv"),
                      usecols=["subject_id", "gender", "anchor_age"])
    dx = pd.read_csv(os.path.join(hosp, "diagnoses_icd.csv"))
    ddx = pd.read_csv(os.path.join(hosp, "d_icd_diagnoses.csv"))
    dli = pd.read_csv(os.path.join(hosp, "d_labitems.csv"), usecols=["itemid", "label", "fluid"])
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
    keep_cols = ["hadm_id", "itemid", "charttime", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag"]
    parts = []
    for i, ch in enumerate(pd.read_csv(os.path.join(MIMIC_ROOT, "labevents.csv"), usecols=keep_cols,
                                       chunksize=1_000_000, low_memory=False)):
        m = ch[ch.hadm_id.isin(want)]
        if len(m):
            parts.append(m)
        if i % 5 == 0:
            got = pd.concat(parts).hadm_id.nunique() if parts else 0
            print(f"  chunk {i}: admissions with labs so far = {got}", flush=True)
        if parts and i > 3:                       # early exit once most candidates have plenty of labs
            counts = pd.concat(parts).groupby("hadm_id").itemid.nunique()
            if (counts >= MAX_LABS).sum() >= n_target * 2:
                break
    labs = pd.concat(parts) if parts else pd.DataFrame(columns=keep_cols)
    print(f"scan done: {labs.hadm_id.nunique()} admissions have labs")

    ed_intake = _ed_intake(pd, want)      # chief complaint + triage vitals, if MIMIC-IV-ED is present
    meta = cand.set_index("hadm_id")
    cases, dropped = {}, 0
    for hadm, g in labs.groupby("hadm_id"):
        # keep each test's FIRST draw, in charttime order, so findings/orders preserve the real ordering sequence
        g = g.dropna(subset=["itemid"]).sort_values("charttime").drop_duplicates("itemid", keep="first")
        if not (MIN_LABS <= len(g) <= MAX_LABS) or hadm not in meta.index:
            dropped += 1
            continue
        m = meta.loc[hadm]
        age, sex, race = int(m.anchor_age), _sex(m.gender), str(m.race).title()
        atype = str(m.admission_type).lower().replace("ew emer.", "emergency").replace("emer.", "emergency")
        intake = ed_intake.get(int(hadm))
        if intake:
            pres = f"A {age}-year-old {sex} ({race}) presented to the emergency department with {intake}"
        else:
            pres = (f"A {age}-year-old {sex} ({race}) was admitted to the hospital via {atype} "
                    f"from {str(m.admission_location).lower()}. No further history is available at intake.")
        # real lab results (the withheld findings) keyed by canonical test name + the real order set
        findings, orders = {}, []
        for r in g.itertuples():
            name = item_label.get(r.itemid, f"lab {r.itemid}")
            orders.append(name)
            val = r.value if pd.notna(r.value) else (r.valuenum if pd.notna(r.valuenum) else "—")
            uom = f" {r.valueuom}" if pd.notna(r.valueuom) else ""
            rng = (f" (ref {r.ref_range_lower}-{r.ref_range_upper})"
                   if pd.notna(r.ref_range_lower) and pd.notna(r.ref_range_upper) else "")
            flag = f" [{r.flag}]" if pd.notna(r.flag) and str(r.flag).strip() else ""
            findings[name] = f"{val}{uom}{rng}{flag}".strip()
        cases[f"MIMIC-{int(hadm)}"] = {
            # first sentence, whichever presentation shape we built
            "abstract": pres.split(". No further")[0].split(". Triage vitals")[0].rstrip(".") + ".",
            "presentation": pres,
            "true_dx": str(meta.loc[hadm, "long_title"]),
            "findings": findings,
            "orders": list(dict.fromkeys(orders)),   # charttime order (first draw of each test), not alphabetized
        }
        if len(cases) >= n_target:
            break

    json.dump(cases, open(out, "w"), indent=1)
    print(f"done - {len(cases)} MIMIC cases -> {out}  (dropped {dropped} out of lab-count range)")
    for cid, c in cases.items():
        print(f"  {cid}: {len(c['findings'])} labs | dx: {c['true_dx'][:60]}")
    return cases


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 10
        build(n_target=n)
    else:
        cases = load_mimic(limit=5)
        print(f"loaded {len(MIMIC_CASES)} MIMIC cases from fixture (showing {len(cases)}):")
        for c in cases:
            print(f"  {c.case_id}: {len(c.findings)} findings | dx: {c.true_diagnosis[:60]}")
