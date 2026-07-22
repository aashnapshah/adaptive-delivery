"""Load NEJM CPC (clinicopathological conference) narratives as cases (stage 01, source="cpc").

A CPC case is a free-text vignette plus a ground-truth diagnosis. Unlike a MIMIC admission, the
whole case exists as prose: the LLM gatekeeper (stage 02) reads the narrative and reveals findings
only on request. The agent starts from the presenting history + vitals (`presentation`), split out
of the narrative and cached by `build_cpc_presentations.py`; the exam and every test result stay
behind the gatekeeper.

Real data: the NEJM CPC-Bench 100-case set (`data/cpc_bench.json`, the `automated_annotations.json`),
loaded by `load_cpc_bench`. Until it's present, `SAMPLE_CPCS` provides a few synthetic narratives so
the pipeline runs offline.
"""

from __future__ import annotations

import json
import os
import re

from .schema import Case

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# CPC presentations sometimes begin with a discussant attribution we must strip, e.g.
# "Dr. Robert B. Den (Medicine): A 33-year-old man ..." -> "A 33-year-old man ..."
_DISCUSSANT = re.compile(r"^\s*Drs?\.\s[^:\n]{1,90}:\s*")


def _strip_discussant(text: str) -> str:
    return _DISCUSSANT.sub("", (text or "").strip(), count=1)


def _first_sentences(text: str, n: int = 2) -> str:
    parts = text.replace("\n", " ").split(". ")
    return ". ".join(parts[:n]).strip().rstrip(".") + "."


# Cache of pre-split presentations (history + vitals) keyed by case_id, from
# data/cpc_presentations.json, so the agent starts from the full clinical picture, not one sentence.
_PRES_CACHE_PATH = os.path.join(DATA_DIR, "cpc_presentations.json")
try:
    _PRESENTATIONS = json.load(open(_PRES_CACHE_PATH))
except Exception:
    _PRESENTATIONS = {}

# The real work-up: tests actually performed in the narrative, in order, extracted and cached by
# build_cpc_workup.py. This is CPC's human order set (mirrors MIMIC's `orders`), used by the
# ordering-concordance and head-to-head benchmarks.
_WORKUP_CACHE_PATH = os.path.join(DATA_DIR, "cpc_workup.json")
try:
    _WORKUP = json.load(open(_WORKUP_CACHE_PATH))
except Exception:
    _WORKUP = {}


def make_cpc(case_id: str, text: str, diagnosis: str, abstract: str | None = None) -> Case:
    text = _strip_discussant(text.strip())                 # drop any "Dr. X (Specialty):" prefix
    case_file = text + f"\n\nFINAL DIAGNOSIS (CONFIDENTIAL, never reveal): {diagnosis}"
    abstract = _strip_discussant(abstract) if abstract else _first_sentences(text)
    return Case(case_id=str(case_id), source="cpc", true_diagnosis=diagnosis,
                abstract=abstract.strip(), case_file=case_file,
                presentation=(_PRESENTATIONS.get(str(case_id)) or "").strip(),
                orders=tuple(_WORKUP.get(str(case_id)) or ()))


# ----------------------------------------- CPC-Bench (NEJM) canonical loader ---
# data/cpc_bench.json = the NEJM CPC-Bench `automated_annotations.json` (100 cases). Each case has
# `presentation_of_case` (the narrative the gatekeeper holds; the diagnosis is NOT in it) and a
# `final_diagnosis` dict. We pick the canonical pathological/clinical diagnosis as the answer key.

DEFAULT_CPC_BENCH = os.path.join(DATA_DIR, "cpc_bench.json")
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


def load_cpc_bench(path: str = None, limit: int = None) -> list[Case]:
    """Load CPC-Bench cases from the annotations JSON into Case objects (LLM-gatekeeper ready)."""
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


# The live registry: real CPC-Bench cases if present, else the bundled samples. CPC_LIMIT caps the
# count (handy while iterating / to bound a recording run).
try:
    _limit = int(os.environ.get("CPC_LIMIT", "0")) or None
    CPCS: list[Case] = load_cpc_bench(limit=_limit) or SAMPLE_CPCS
except Exception:
    CPCS = SAMPLE_CPCS


def list_cases(cases: list[Case]) -> None:
    for i, c in enumerate(cases):
        print(f"  [{i}] {c.case_id}: {c.abstract[:80]}")


if __name__ == "__main__":
    print(f"loaded {len(CPCS)} CPC cases")
    list_cases(CPCS[:5])
