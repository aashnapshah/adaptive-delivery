"""Stage 01 - case presentations.

Loads every case source into the unified `Case` (see `schema.py`) and exposes one registry:

    from harness.cases import get_case, case_options, Case

Sources and their registries:
    toy    TOY_CASES     synthetic MIMIC-shaped records (offline fixtures, rich ground truth)
    cpc    CPCS          NEJM CPC narratives (real CPC-Bench if present, else samples)
    mimic  MIMIC_CASES   real MIMIC-IV admissions (labs-on-order; empty until the fixture is built)
"""

from __future__ import annotations

import json
import os

from .schema import (
    Case, LabEvent, MicroEvent, Patient, RadiologyStudy, TestPanel,
)
from .toy import TOY_CASES, all_cases, load_case
from .cpc import CPCS, load_cpc_bench, make_cpc
from .mimic import MIMIC_BY_ID, MIMIC_CASES, load_mimic

CPC_BY_ID: dict[str, Case] = {c.case_id: c for c in CPCS}

__all__ = [
    "Case", "LabEvent", "MicroEvent", "Patient", "RadiologyStudy", "TestPanel",
    "TOY_CASES", "CPCS", "CPC_BY_ID", "MIMIC_CASES", "MIMIC_BY_ID",
    "load_case", "all_cases", "load_cpc_bench", "make_cpc", "load_mimic",
    "registry", "get_case", "case_options", "refined_case_ids",
]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def refined_case_ids() -> set[str]:
    """The cases we've refined: all MIMIC admissions + every CPC case with a split presentation.
    Empty set = nothing refined yet, so callers fall back to showing everything."""
    ids = set(MIMIC_BY_ID)
    try:
        ids |= set(json.load(open(os.path.join(_DATA_DIR, "cpc_presentations.json"))))
    except Exception:
        pass
    return ids


def registry() -> dict[str, Case]:
    """Every known case by id, across all sources (toy + cpc + mimic)."""
    reg: dict[str, Case] = {}
    reg.update(TOY_CASES)
    reg.update(CPC_BY_ID)
    reg.update(MIMIC_BY_ID)
    return reg


def get_case(case_id: str) -> Case:
    """Look up a case by id across all sources. Raises KeyError if unknown."""
    return registry()[case_id]


def case_options() -> list[dict]:
    """UI-friendly list of every case: {id, source, label}."""
    return [{"id": c.case_id, "source": c.source, "label": f"{c.case_id} — {c.abstract[:70]}"}
            for c in registry().values()]
