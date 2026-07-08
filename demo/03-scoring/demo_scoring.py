"""Demo 03 - Appropriateness and Harm Scoring.

Maps to Methods section: "Appropriateness and Harm Scoring".

Generation (Demo 02) produces a stream of test recommendations. Before we decide
whether to *deliver* one as an alert (Demo 05), we score each recommendation on two
axes the literature treats as central to diagnostic stewardship:

  - appropriateness: is this the right test given the case state, or is it redundant /
    off-protocol / low-yield? (Park 2022; Muskens 2022 overuse; Baron 2021 duplicate alerts)
  - harm/burden: radiation, invasiveness, and cost of the test itself. (Wu 2025 NOHARM)

Each recommendation gets a structured score used downstream as the bandit's context.
A transparent rule-based scorer runs offline; an optional LLM rubric judge mirrors the
appropriateness ratings used in CDS studies.

Run:  python3 demo_scoring.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared import llm  # noqa: E402
from shared.toy_data import EHRCase, load_case  # noqa: E402

# Relative harm/burden by test kind, nudged for radiation / contrast / invasiveness.
_BASE_HARM = {"lab_panel": 0.10, "micro": 0.15, "imaging": 0.45}
_HARM_BUMPS = {"ct": 0.15, "contrast": 0.10, "angiogram": 0.10, "biopsy": 0.35}
_MAX_REASONABLE_COST = 600.0  # for normalizing cost into [0, 1]


@dataclass
class Score:
    test: str
    appropriateness: float   # 0 (inappropriate) .. 1 (clearly indicated)
    harm: float              # 0 (benign) .. 1 (high burden/risk)
    cost: float              # USD
    redundant: bool
    rationale: str

    def as_dict(self) -> dict:
        return {"test": self.test, "appropriateness": round(self.appropriateness, 2),
                "harm": round(self.harm, 2), "cost": self.cost,
                "redundant": self.redundant, "rationale": self.rationale}


def harm_score(case: EHRCase, test: str) -> tuple[float, float]:
    """Return (harm in [0,1], cost in USD) for a named test."""
    panel = case.panels.get(test)
    if panel is None:
        return 0.4, 50.0  # unknown/off-protocol: assume moderate burden
    harm = _BASE_HARM.get(panel.kind, 0.3)
    name = f"{panel.order_name} {panel.imaging_exam or ''}".lower()
    for kw, bump in _HARM_BUMPS.items():
        if kw in name:
            harm += bump
    # cost itself is part of burden
    harm += 0.2 * min(panel.cost / _MAX_REASONABLE_COST, 1.0)
    return min(harm, 1.0), panel.cost


def appropriateness_score(case: EHRCase, test: str, ordered: set[str]) -> tuple[float, bool, str]:
    """Rule-based appropriateness in [0,1] plus a redundancy flag and rationale."""
    panel = case.panels.get(test)
    if panel is None:
        return 0.30, False, "Off-protocol / not a recognized test for this presentation."
    if test in ordered:
        return 0.05, True, "Duplicate: this test was already ordered (Baron 2021)."
    if panel.informative:
        return 0.88, False, "Discriminates the leading differential; high diagnostic yield."
    return 0.40, False, "Plausible but low-yield given the current findings (overuse risk)."


def score_recommendation(case: EHRCase, test: str, ordered: set[str] | None = None) -> Score:
    ordered = ordered or set()
    appr, redundant, why = appropriateness_score(case, test, ordered)
    harm, cost = harm_score(case, test)
    return Score(test, appr, harm, cost, redundant, why)


# --- Optional: LLM appropriateness judge (1-5, like CDS rubric studies) ------

JUDGE_SYSTEM = (
    "You are a diagnostic stewardship reviewer. Rate how APPROPRIATE it is to order the "
    "named test for this patient right now on a 1-5 scale (5 = clearly indicated, "
    "1 = inappropriate/redundant). Reply with ONLY the integer."
)


def llm_appropriateness(case: EHRCase, test: str, findings: str, backend: str | None = None) -> int | None:
    """Ask a model for a 1-5 appropriateness rating (None if unavailable)."""
    msg = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": f"Patient: {case.abstract}\nFindings so far: {findings}\n"
         f"Proposed test: {test}\nRating (1-5):"},
    ]
    try:
        raw = llm.chat(msg, backend=backend, max_tokens=4)
    except Exception:
        return None
    digits = [c for c in raw if c.isdigit()]
    return int(digits[0]) if digits else None


def _bar(x: float, width: int = 20) -> str:
    return "#" * int(round(x * width)) + "-" * (width - int(round(x * width)))


def main() -> None:
    case = load_case("PE-2180")
    ordered: set[str] = {"D-Dimer"}  # pretend D-Dimer already ordered, to show redundancy

    print(f"CASE {case.case_id}: {case.abstract}\n(D-Dimer already ordered)\n")
    print(f"{'test':<26}{'appr':>6} {'harm':>6} {'cost':>7}  rationale")
    print("-" * 92)
    for test in ["CT pulmonary angiogram", "D-Dimer", "Chest radiograph", "Troponin", "MRI brain"]:
        s = score_recommendation(case, test, ordered)
        print(f"{test:<26}{s.appropriateness:>6.2f} {s.harm:>6.2f} ${s.cost:>6.0f}  {s.rationale}")

    print("\nVisual (appropriateness vs harm) for the confirmatory test:")
    s = score_recommendation(case, "CT pulmonary angiogram", ordered)
    print(f"  appropriateness [{_bar(s.appropriateness)}] {s.appropriateness:.2f}")
    print(f"  harm/burden     [{_bar(s.harm)}] {s.harm:.2f}")


if __name__ == "__main__":
    main()
