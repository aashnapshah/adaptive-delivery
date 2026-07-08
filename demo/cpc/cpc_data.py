"""Load CPC (clinicopathological conference) cases into the sequential environment.

A CPC case is a free-text narrative plus a ground-truth diagnosis. The LLM gatekeeper
(demo/01_environment) reads the narrative and reveals findings only on request, so a CPC
drops straight into the same loop the recommenders already use.

Real data: download the CPC-Bench 100-case set (cpcbench.com, institutional login) and point
`load_cpc(path)` at it. Until then, `SAMPLE_CPCS` provides a few synthetic narratives so the
playground runs immediately. The schema is unknown ahead of the download, so `load_cpc` is
deliberately flexible (JSON / JSONL / CSV / a folder of .txt) with configurable field names.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
from dataclasses import dataclass, field

# CPC presentations sometimes begin with a discussant attribution we must strip, e.g.
# "Dr. Robert B. Den (Medicine): A 33-year-old man ..." -> "A 33-year-old man ..."
_DISCUSSANT = re.compile(r"^\s*Drs?\.\s[^:\n]{1,90}:\s*")


def _strip_discussant(text: str) -> str:
    return _DISCUSSANT.sub("", (text or "").strip(), count=1)


@dataclass
class CPCCase:
    """A narrative case the gatekeeper can serve (drop-in for the toy EHRCase)."""
    case_id: str
    abstract: str          # short opening vignette (one-liner, for headers/labels)
    case_file: str         # full narrative the gatekeeper holds (diagnosis withheld)
    true_diagnosis: str    # ground truth for the Judge
    presentation: str = ""  # history + physical/neuro exam given to the agent at turn 0
                            # (labs/imaging/path stay behind the gatekeeper). Falls back to abstract.
    # empty stand-ins so the existing env/gatekeeper code paths just work
    panels: dict = field(default_factory=dict)
    labevents: dict = field(default_factory=dict)
    micro: dict = field(default_factory=dict)
    radiology: dict = field(default_factory=dict)
    hpi: dict = field(default_factory=dict)
    exam: str = ""
    patient: object = None
    diagnoses_icd: tuple = ()

    def confirmable_by(self, ordered) -> bool:
        return False


def _first_sentences(text: str, n: int = 2) -> str:
    parts = text.replace("\n", " ").split(". ")
    return ". ".join(parts[:n]).strip().rstrip(".") + "."


# Cache of pre-split presentations (history + exam) keyed by case_id, built once by
# split_presentations.py so the agent starts from the full clinical picture, not one sentence.
_PRES_CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "data", "cpc_presentations.json")
try:
    _PRESENTATIONS = json.load(open(_PRES_CACHE_PATH))
except Exception:
    _PRESENTATIONS = {}


def make_cpc(case_id: str, text: str, diagnosis: str, abstract: str | None = None) -> CPCCase:
    text = _strip_discussant(text.strip())                 # drop any "Dr. X (Specialty):" prefix
    case_file = text + f"\n\nFINAL DIAGNOSIS (CONFIDENTIAL, never reveal): {diagnosis}"
    abstract = _strip_discussant(abstract) if abstract else _first_sentences(text)
    return CPCCase(case_id=str(case_id), abstract=abstract.strip(),
                   case_file=case_file, true_diagnosis=diagnosis,
                   presentation=(_PRESENTATIONS.get(str(case_id)) or "").strip())


# ----------------------------------------------------------------- loader ----

# common field names seen in clinical-case datasets; adjust once you see the real schema
_ID = ("id", "case_id", "caseid", "uid")
_TEXT = ("case", "text", "case_text", "presentation", "vignette", "body", "case_presentation")
_DX = ("diagnosis", "final_diagnosis", "dx", "ground_truth", "answer", "label")
_ABS = ("abstract", "summary", "opening")


def _pick(d: dict, names) -> str | None:
    low = {k.lower(): k for k in d}
    for n in names:
        if n in low:
            return d[low[n]]
    return None


def load_cpc(path: str, id_field=None, text_field=None, dx_field=None, abstract_field=None) -> list[CPCCase]:
    """Load CPC cases from a JSON list, JSONL, CSV, or a folder of .json/.txt files.

    Pass explicit *_field names if auto-detection misses the schema. A .txt file is treated
    as one case whose filename stem is the id and whose first line (after 'DIAGNOSIS:') is the dx.
    """
    records: list[dict] = []
    if os.path.isdir(path):
        for fp in sorted(glob.glob(os.path.join(path, "*"))):
            if fp.endswith(".json"):
                records.append(json.load(open(fp)))
            elif fp.endswith(".txt"):
                body = open(fp).read()
                dx = ""
                for line in body.splitlines():
                    if line.lower().startswith("diagnosis:"):
                        dx = line.split(":", 1)[1].strip()
                records.append({"id": os.path.splitext(os.path.basename(fp))[0], "text": body, "diagnosis": dx})
    elif path.endswith(".jsonl"):
        records = [json.loads(l) for l in open(path) if l.strip()]
    elif path.endswith(".json"):
        data = json.load(open(path))
        records = data if isinstance(data, list) else data.get("cases", [data])
    elif path.endswith(".csv"):
        records = list(csv.DictReader(open(path)))
    else:
        raise ValueError(f"Unrecognized CPC path: {path}")

    cases = []
    for i, r in enumerate(records):
        cid = (r.get(id_field) if id_field else _pick(r, _ID)) or f"CPC-{i+1:03d}"
        text = (r.get(text_field) if text_field else _pick(r, _TEXT))
        dx = (r.get(dx_field) if dx_field else _pick(r, _DX))
        ab = (r.get(abstract_field) if abstract_field else _pick(r, _ABS))
        if not text or not dx:
            continue
        cases.append(make_cpc(cid, text, dx, ab))
    return cases


# ----------------------------------------------------- bundled sample cases ---

SAMPLE_CPCS = [
    make_cpc("SAMPLE-Hodgkin",
        "A 29-year-old previously healthy man presented with six weeks of intermittent fevers, "
        "drenching night sweats, and a 7 kg weight loss. He reported a painless, enlarging swelling "
        "in his neck. On examination he had firm, non-tender cervical and axillary lymphadenopathy "
        "and splenomegaly. Initial laboratory studies showed a mild normocytic anemia and an elevated "
        "lactate dehydrogenase. A chest radiograph demonstrated mediastinal widening. An excisional "
        "biopsy of a cervical lymph node revealed Reed-Sternberg cells.",
        "Hodgkin lymphoma"),
    make_cpc("SAMPLE-Endocarditis",
        "A 41-year-old man with a history of injection drug use presented with two weeks of fevers, "
        "fatigue, and night sweats. On examination he was febrile with a new holosystolic murmur, "
        "splinter hemorrhages, and tender nodules on the finger pads. Blood cultures grew "
        "Staphylococcus aureus. A transthoracic echocardiogram showed a vegetation on the tricuspid "
        "valve with moderate regurgitation.",
        "Infective endocarditis"),
    make_cpc("SAMPLE-PE",
        "A 52-year-old woman presented with the sudden onset of pleuritic chest pain and dyspnea "
        "three days after a long-haul flight. She was tachycardic with an oxygen saturation of 91% on "
        "room air; her right calf was mildly tender. A D-dimer was markedly elevated, and a CT "
        "pulmonary angiogram showed a filling defect in the right lower-lobe segmental artery.",
        "Pulmonary embolism"),
]


def list_cases(cases: list[CPCCase]) -> None:
    for i, c in enumerate(cases):
        print(f"  [{i}] {c.case_id}: {c.abstract[:80]}")


# ----------------------------------------------- CPC-Bench (NEJM) loader ------
# data/cpc_bench.json = the NEJM CPC-Bench `automated_annotations.json` (100 cases). Each case has
# `presentation_of_case` (the narrative the gatekeeper holds; the diagnosis is NOT in it) and a
# `final_diagnosis` dict. We pick the canonical pathological/clinical diagnosis as the answer key.

DEFAULT_CPC_BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "cpc_bench.json")
# preference order for the gold diagnosis (impersonal canonical labels before any "Dr. X's Diagnosis")
_DX_PRIORITY = ("Anatomical Diagnosis", "Anatomical Diagnoses", "Final Pathological Diagnosis",
                "Pathological Diagnosis", "Final Diagnosis", "Clinical Diagnosis", "Clinical Diagnoses",
                "Laboratory Diagnosis", "Microbiologic Diagnosis", "Microbiologic and Immunologic Diagnosis",
                "Diagnosis")


def _clean_dx(v) -> str:
    s = str(v).strip().lstrip("?").strip()
    return s.split("\n")[0].strip().rstrip(".")          # first line, no trailing period


def _final_dx(fd) -> str:
    if isinstance(fd, str):
        return _clean_dx(fd)
    if not isinstance(fd, dict):
        return ""
    for k in _DX_PRIORITY:
        if fd.get(k) and str(fd[k]).strip():
            return _clean_dx(fd[k])
    for k, v in fd.items():                               # else first non-discussant entry
        if not str(k).startswith("Dr.") and v and str(v).strip():
            return _clean_dx(v)
    for v in fd.values():
        if v and str(v).strip():
            return _clean_dx(v)
    return ""


def load_cpc_bench(path: str = None, limit: int = None) -> list[CPCCase]:
    """Load CPC-Bench cases from the annotations JSON into CPCCase objects (LLM-gatekeeper ready)."""
    path = path or DEFAULT_CPC_BENCH
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    out = []
    for c in data:
        pres = (c.get("presentation_of_case") or "").strip()
        dx = _final_dx(c.get("final_diagnosis"))
        if not pres or not dx:
            continue
        # the dataset's clean sentence split gives a far better opening vignette than naive splitting
        sent1 = (c.get("presentation_of_case_sent") or {}).get("1")
        out.append(make_cpc(c.get("id") or f"CPC-{len(out)+1:03d}", pres, dx, abstract=sent1))
        if limit and len(out) >= limit:
            break
    return out


# The live registry: real CPC-Bench cases if present, else the bundled samples. CPC_LIMIT caps the
# count (handy while iterating / to bound a recording run).
try:
    _limit = int(os.environ.get("CPC_LIMIT", "0")) or None
    CPCS = load_cpc_bench(limit=_limit) or SAMPLE_CPCS
except Exception:
    CPCS = SAMPLE_CPCS


if __name__ == "__main__":
    print(f"loaded {len(CPCS)} CPC cases")
    list_cases(CPCS[:5])
