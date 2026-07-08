"""Demo 04 - Delivery Interface and Clinician Response.

Maps to Methods section: "Delivery Interface and Clinician Response".

A scored recommendation only creates value if it is delivered well and the clinician acts
on it. This stage simulates the human in the loop across the SAME five presentation actions
the plan uses - interrupt, passive, delay, bundle, suppress - and encodes alert fatigue:
acceptance falls as recent alert burden grows, and each action adds to that burden in
proportion to how intrusive it is (AHRQ 2019; Park 2022; Goh 2024).

The five actions trade salience against burden:
  interrupt - most likely to be acted on, but adds the most burden
  bundle    - groups related recs into one alert: high salience, moderate burden
  delay      - re-surfaced later: moderate salience and burden
  passive   - non-blocking note: low salience, low burden
  suppress  - withheld: no salience, burden decays

Run:  python3 demo_interface.py
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared.toy_data import load_case  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "03_scoring")))
from demo_scoring import score_recommendation  # noqa: E402

ACTIONS = ["interrupt", "passive", "delay", "bundle", "suppress"]

# Probability the clinician even engages with the alert, given they would otherwise accept.
SALIENCE = {"interrupt": 1.0, "passive": 0.45, "delay": 0.70, "bundle": 0.85, "suppress": 0.0}
# How much each delivered alert adds to the running alert burden (intrusiveness).
BURDEN_ADD = {"interrupt": 1.0, "passive": 0.30, "delay": 0.50, "bundle": 0.50, "suppress": 0.0}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class Decision:
    action: str
    delivered: bool
    accepted: bool
    p_accept: float
    burden: float


@dataclass
class ClinicianSimulator:
    """Seeded model of a clinician reacting to an alert delivered via a chosen action.

    Acceptance rises with appropriateness, falls with the test's harm/burden, and - the key
    alert-fatigue effect - falls as the running `burden` grows. Salience scales acceptance by
    how attention-grabbing the action is; intrusive actions also add more burden.
    """

    seed: int = 0
    w_appr: float = 4.0
    w_harm: float = 1.5
    w_burden: float = 1.0
    bias: float = -1.0
    burden: float = 0.0
    decay: float = 0.8        # fatigue recovers over time, every step (not only when suppressing)
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def respond(self, appropriateness: float, harm: float, action: str) -> Decision:
        self.burden *= self.decay                       # time-based recovery each step
        if action == "suppress":
            return Decision(action, False, False, 0.0, self.burden)
        base = _sigmoid(self.w_appr * appropriateness - self.w_harm * harm
                        - self.w_burden * self.burden + self.bias)
        p = SALIENCE[action] * base
        accepted = self._rng.random() < p
        self.burden += BURDEN_ADD[action]               # delivering adds to the burden
        return Decision(action, True, accepted, p, self.burden)


def main() -> None:
    case = load_case("PE-2180")
    good = score_recommendation(case, "CT pulmonary angiogram")

    print(f"CASE {case.case_id}: one appropriate alert, delivered different ways (fresh clinician each):\n")
    print(f"{'action':<10}{'p(accept)':>10}{'burden_add':>12}")
    for action in ACTIONS:
        clin = ClinicianSimulator(seed=1)
        d = clin.respond(good.appropriateness, good.harm, action)
        print(f"{action:<10}{d.p_accept:>10.2f}{BURDEN_ADD[action]:>12.2f}")

    print("\nAlert fatigue: interrupt the same appropriate alert repeatedly (one clinician):\n")
    clin = ClinicianSimulator(seed=1)
    print(f"{'#':>2}{'burden':>8}{'p(accept)':>11}  outcome")
    for i in range(1, 7):
        d = clin.respond(good.appropriateness, good.harm, "interrupt")
        print(f"{i:>2}{d.burden:>8.1f}{d.p_accept:>11.2f}  {'ACCEPT' if d.accepted else 'DISMISS'}")
    print("\nThe policy (Demo 05) chooses among the five actions to keep useful alerts accepted "
          "without driving burden up - which is why it must reason about burden as a state.")


if __name__ == "__main__":
    main()
