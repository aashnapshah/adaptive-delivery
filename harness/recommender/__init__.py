"""The recommender agent - reads the presentation and recommends a work-up (orders tests via the
gatekeeper, stops when done). Diagnosis is a separate readout.

    from harness.recommender import recommend, diagnose
"""

from __future__ import annotations

from .recommender import diagnose, recommend

__all__ = ["recommend", "diagnose"]
