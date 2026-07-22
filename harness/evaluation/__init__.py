"""Evaluation: the scoring driver (evaluate.py) and the deterministic side of scoring.

The one deterministic input, test cost, lives here (cost.py) and is shared with the live delivery stage.
Every LLM-computed metric is a Judge method in harness/judge.
"""

from .cost import DEFAULT_TEST_COST, estimate_test_cost, test_cost

__all__ = ["test_cost", "estimate_test_cost", "DEFAULT_TEST_COST"]
