"""Delivery layer (parked, WIP) - the "how do we deliver a recommendation" direction.

Not wired into the benchmark pipeline. Three parts over the structured toy cases:
    scoring    appropriateness + harm scoring of each recommendation (panel-based)
    interface  simulated clinician + alert fatigue across five presentation actions
    policy     adaptive delivery policy (a contextual bandit) vs fixed baselines

The live pipeline's harm benchmark (name-based, CPC/MIMIC) is harness/harm.py, separate from this.
"""

from __future__ import annotations

from .scoring import Score, appropriateness_score, harm_score, score_recommendation

__all__ = ["Score", "score_recommendation", "appropriateness_score", "harm_score"]
