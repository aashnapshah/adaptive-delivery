"""Stage 02 - the Gatekeeper agent.

    from harness.gatekeeper import Gatekeeper

It holds a case, gives the presentation, and answers the recommender's requests by reading the case.
The loop that drives it (the recommender's session) is built separately.
"""

from __future__ import annotations

from .gatekeeper import GATEKEEPER_SYSTEM, Gatekeeper, case_text

__all__ = ["Gatekeeper", "case_text", "GATEKEEPER_SYSTEM"]
