"""Adaptive delivery policy: a contextual bandit, learned online (stage 06).

Decide HOW to deliver each scored recommendation - interrupt / passive / delay / bundle / suppress -
to realize clinical value without overloading the clinician.

Framing: a contextual bandit. Each recommendation is a largely local choice: context = (estimated
appropriateness, harm, alert burden); arms = the five actions; reward from the clinician's response.
We start with a contextual bandit (LinUCB) over the five actions, with fixed baselines (always / never
/ threshold) for comparison (Varatharajah 2022). Richer sequential modeling is a later extension.

Reward is decoupled from adherence: value is set by the appropriateness/harm SCORE, not by whether the
clinician happens to agree, so withholding a high-value recommendation is penalized whether or not the
clinician would have acted. The policy sees a NOISY estimated appropriateness, never the latent truth
(which is used only to measure realized outcome and safety).
"""

from __future__ import annotations

import numpy as np

from ..cases import all_cases
from .interface import ACTIONS, ClinicianSimulator
from .scoring import score_recommendation

# Attention cost charged in the reward for delivering at all (separate from clinical value).
ATTENTION_COST = {"interrupt": 0.30, "passive": 0.10, "delay": 0.15, "bundle": 0.15, "suppress": 0.0}


# --- Recommendation stream: latent truth vs noisy scored estimate ------------

def build_stream(n: int = 400, seed: int = 0, scorer_noise: float = 0.12) -> list[dict]:
    """Reproducible stream of recommendations with a LATENT value and a NOISY estimate.

    `appr_est` (noisy) is what the policy sees; `v_true` (latent) is used only for outcome and safety
    metrics. Signed value maps appropriateness to [-1, 1].
    """
    rng = np.random.default_rng(seed)
    cases = all_cases()
    stream = []
    for _ in range(n):
        case = cases[rng.integers(len(cases))]
        useful = rng.random() < 0.5
        if useful:
            test = case.key_tests[rng.integers(len(case.key_tests))]
        else:
            low = [p for p in case.panels if not case.panels[p].informative]
            test = low[rng.integers(len(low))] if low else list(case.panels)[0]
        s = score_recommendation(case, test)
        appr_true = s.appropriateness
        appr_est = float(np.clip(appr_true + rng.normal(0, scorer_noise), 0.0, 1.0))
        stream.append({
            "case_id": case.case_id, "test": test, "harm": s.harm,
            "appr_true": appr_true, "appr_est": appr_est,
            "v_true": 2.0 * (appr_true - 0.5),   # latent signed value (outcome/safety only)
            "v_hat": 2.0 * (appr_est - 0.5),     # rubric value the reward uses
        })
    return stream


def make_context(appr_est: float, harm: float, burden: float) -> np.ndarray:
    return np.array([appr_est, harm, min(burden / 5.0, 1.0), 1.0])


def reward(action: str, item: dict, accepted: bool) -> float:
    """Value-based reward, decoupled from adherence (see module docstring)."""
    v = item["v_hat"]
    if action == "suppress":
        return -max(v, 0.0)                      # harm of omission
    r = -ATTENTION_COST[action]                  # cost of delivering at all
    if accepted:
        r += v                                   # realize signed value (harmful -> negative)
    return r


# --- Policies ----------------------------------------------------------------

class FixedPolicy:
    """Baselines: always-interrupt, never-alert, threshold on estimated appropriateness."""

    def __init__(self, kind: str):
        self.kind = kind

    def act(self, item: dict, burden: float) -> str:
        if self.kind == "always":
            return "interrupt"
        if self.kind == "never":
            return "suppress"
        return "interrupt" if item["appr_est"] >= 0.6 else "suppress"   # threshold

    def learn(self, *args) -> None:
        pass


class LinUCB:
    """Disjoint LinUCB over the five actions - a MYOPIC contextual bandit (ignores burden dynamics)."""

    def __init__(self, d: int = 4, alpha: float = 0.15):
        self.alpha = alpha
        self.A = {a: np.eye(d) for a in ACTIONS}
        self.b = {a: np.zeros(d) for a in ACTIONS}
        self._x = None
        self._a = None

    def _ucb(self, a: str, x: np.ndarray) -> float:
        A_inv = np.linalg.inv(self.A[a])
        return float((A_inv @ self.b[a]) @ x + self.alpha * np.sqrt(x @ A_inv @ x))

    def act(self, item: dict, burden: float) -> str:
        x = make_context(item["appr_est"], item["harm"], burden)
        a = max(ACTIONS, key=lambda a: self._ucb(a, x))
        self._x, self._a = x, a
        return a

    def learn(self, r: float, next_item: dict, next_burden: float) -> None:
        self.A[self._a] += np.outer(self._x, self._x)
        self.b[self._a] += r * self._x


# --- Simulation + metrics ----------------------------------------------------

def run_policy(policy, stream: list[dict], seed: int = 7) -> dict:
    clin = ClinicianSimulator(seed=seed)
    cum, curve = 0.0, []
    delivered = accepted = interrupts = 0
    realized, safety_miss, commission = 0.0, 0, 0
    for i, item in enumerate(stream):
        burden = clin.burden
        action = policy.act(item, burden)
        d = clin.respond(item["appr_est"], item["harm"], action)
        r = reward(action, item, d.accepted)
        cum += r
        curve.append(cum)
        nxt = stream[i + 1] if i + 1 < len(stream) else item
        policy.learn(r, nxt, clin.burden)
        if d.delivered:
            delivered += 1
            interrupts += int(action == "interrupt")
            accepted += int(d.accepted)
        acted = d.delivered and d.accepted
        if acted:
            realized += item["v_true"]
        if item["v_true"] > 0.4 and not acted:
            safety_miss += 1                       # a truly useful rec not acted on
        if item["v_true"] < -0.3 and acted:
            commission += 1                        # a truly harmful test acted on
    n = len(stream)
    return {"total_reward": cum, "cum_curve": curve, "n": n,
            "deliver_rate": delivered / n, "interrupt_rate": interrupts / n,
            "accept_rate": (accepted / delivered) if delivered else 0.0,
            "realized_value": realized, "safety_miss": safety_miss, "commission": commission}


def second_half_reward(curve: list[float]) -> float:
    half = len(curve) // 2
    return curve[-1] - curve[half]


def all_policies() -> dict:
    return {
        "always-interrupt": FixedPolicy("always"),
        "never-alert": FixedPolicy("never"),
        "threshold": FixedPolicy("threshold"),
        "LinUCB (bandit)": LinUCB(alpha=0.15),
    }


def main() -> None:
    stream = build_stream(n=400, seed=0)
    print(f"Stream of {len(stream)} recommendations (~50% useful); policy sees NOISY estimates only.\n")
    print(f"{'policy':<20}{'reward':>8}{'2nd-half':>9}{'realized':>10}{'safety-miss':>12}{'interrupt%':>11}")
    print("-" * 70)
    for name, pol in all_policies().items():
        r = run_policy(pol, stream)
        print(f"{name:<20}{r['total_reward']:>8.1f}{second_half_reward(r['cum_curve']):>9.1f}"
              f"{r['realized_value']:>10.1f}{r['safety_miss']:>12d}{r['interrupt_rate']*100:>10.0f}%")
    print("\nIllustrative on synthetic data (no real data yet). The robust point is the mechanism: "
          "always-interrupt collapses under fatigue and never-alert wastes value, while an adaptive "
          "contextual bandit (LinUCB) recovers most of the value at far fewer interrupts.")


if __name__ == "__main__":
    main()
