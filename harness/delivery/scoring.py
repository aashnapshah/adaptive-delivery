"""Appropriateness + harm scoring over the structured toy cases (delivery layer, WIP).

Part of the parked delivery direction: before deciding whether to DELIVER a recommendation as an alert
(interface + policy), score each test on two stewardship axes -

  - appropriateness: is this the right test given the case state, or is it redundant / off-protocol /
    low-yield? (Park 2022; Muskens 2022 overuse; Baron 2021 duplicate alerts)
  - harm/burden: radiation, invasiveness, and cost of the test itself. (Wu 2025 NOHARM)

This is the rich, panel-based scorer that uses the toy cases' structured `panels`. The live pipeline's
harm benchmark (name-based, works on CPC/MIMIC) lives in harness/harm.py instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cases import Case, load_case
from ..evaluation.cost import test_cost

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


def harm_score(case: Case, test: str) -> tuple[float, float]:
    """Return (harm in [0,1], cost in USD) for a named test. Cost comes from harness/evaluation/cost.py."""
    cost = test_cost(case, test)
    panel = case.panels.get(test)
    if panel is None:
        return 0.4, cost  # unknown/off-protocol: assume moderate burden
    harm = _BASE_HARM.get(panel.kind, 0.3)
    name = f"{panel.order_name} {panel.imaging_exam or ''}".lower()
    for kw, bump in _HARM_BUMPS.items():
        if kw in name:
            harm += bump
    harm += 0.2 * min(cost / _MAX_REASONABLE_COST, 1.0)   # cost itself is part of burden
    return min(harm, 1.0), cost


def appropriateness_score(case: Case, test: str, ordered: set[str]) -> tuple[float, bool, str]:
    """Rule-based appropriateness in [0,1] plus a redundancy flag and rationale."""
    panel = case.panels.get(test)
    if panel is None:
        return 0.30, False, "Off-protocol / not a recognized test for this presentation."
    if test in ordered:
        return 0.05, True, "Duplicate: this test was already ordered (Baron 2021)."
    if panel.informative:
        return 0.88, False, "Discriminates the leading differential; high diagnostic yield."
    return 0.40, False, "Plausible but low-yield given the current findings (overuse risk)."


def score_recommendation(case: Case, test: str, ordered: set[str] | None = None) -> Score:
    ordered = ordered or set()
    appr, redundant, why = appropriateness_score(case, test, ordered)
    harm, cost = harm_score(case, test)
    return Score(test, appr, harm, cost, redundant, why)


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
