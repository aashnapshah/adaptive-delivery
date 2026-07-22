"""The Judge - the harness's single LLM evaluator, one method per judgment, each with its own prompt.

  judge_diagnosis(reference, candidate)                 score a diagnosis vs the reference (CPC).
  judge_workup(presentation, llm_orders, human_orders)  blinded head-to-head: is the recommender's
                                                        work-up a better pathway than the human's?
  judge_concordance(ordered, reference)                 flexible match of the ordered tests against the
                                                        tests actually done -> recall / precision.
  judge_harm(ordered)                                   per-test burden rating -> harm/cost of the work-up.

Diagnosis and work-up return a graceful sentinel if the LLM is unavailable (they are always scorable).
Concordance and harm RAISE on LLM failure, so the driver skips that (run, benchmark) and retries it later
rather than persisting a fake score. Cost stays deterministic (harness/evaluation/cost.py); only burden is judged.

Prompts: prompts/diagnosis.txt, prompts/workup.txt, prompts/concordance.txt, prompts/harm.txt.
"""

import random

from ..evaluation.cost import estimate_test_cost
from ..shared import llm
from ..shared.parsing import extract_json
from ..shared.prompts import load

DIAGNOSIS_SYSTEM = load(__file__, "diagnosis")
WORKUP_SYSTEM = load(__file__, "workup")
CONCORDANCE_SYSTEM = load(__file__, "concordance")
HARM_SYSTEM = load(__file__, "harm")


def _ask(system, user, max_tokens, model, backend):
    """One judge LLM call: system + user prompt -> parsed JSON. Deterministic (temperature 0), no thinking."""
    return extract_json(llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 model=model or llm.light_model(backend), backend=backend,
                                 max_tokens=max_tokens, think=False))


def _numbered(seq):
    """A 1-based numbered list, one item per line (order is legible to the judge)."""
    return "\n".join(f"{i}. {t}" for i, t in enumerate(seq, 1)) or "(none)"


def _indices(raw, n):
    """The valid 1..n integers in a model-returned index list (tolerates strings / floats / junk)."""
    out = set()
    for x in raw or []:
        try:
            i = int(x)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n:
            out.add(i)
    return out


def judge_diagnosis(reference, candidate, context="", model=None, backend=None):
    """Score `candidate` against `reference` (context optional). Returns {score, reason, correct}."""
    user = f"REFERENCE: {reference}\nCANDIDATE: {candidate}"
    if context:
        user += f"\nCase context: {context}"
    try:
        rec = _ask(DIAGNOSIS_SYSTEM, user + "\nScore:", 160, model, backend)
        score = max(1, min(5, int(rec["score"])))
        reason = str(rec.get("reason", ""))[:160]
    except Exception as exc:
        return {"score": 0, "reason": f"(judge unavailable: {exc})", "correct": False}
    return {"score": score, "reason": reason, "correct": score >= 4}


def judge_workup(presentation, llm_orders, human_orders, model=None, backend=None):
    """Blinded head-to-head: is the recommender's work-up a better diagnostic PATHWAY than the human's
    actual ordering? Both are ordered sequences (order matters), shown in the same numbered format so the
    judge can't tell which is the agent. Returns {better: 'llm'|'human'|'tie', reason}."""
    llm_is_1 = random.random() < 0.5                              # blind the judge to which list is the LLM
    a, b = (llm_orders, human_orders) if llm_is_1 else (human_orders, llm_orders)
    user = (f"PRESENTATION:\n{presentation}\n\nWORK-UP 1 (in order):\n{_numbered(a)}"
            f"\n\nWORK-UP 2 (in order):\n{_numbered(b)}")
    try:
        rec = _ask(WORKUP_SYSTEM, user, 200, model, backend)
        pick = int(rec.get("better", 0))
        reason = str(rec.get("reason", ""))[:200]
    except Exception as exc:
        return {"better": None, "reason": f"(workup judge unavailable: {exc})"}
    if pick == 0:
        better = "tie"
    else:
        better = "llm" if (pick == 1) == llm_is_1 else "human"    # map the picked position back to who it was
    return {"better": better, "reason": reason}


def judge_concordance(ordered, reference, model=None, backend=None):
    """Flexibly match everything the recommender `ordered` against the `reference` tests actually done -
    by clinical meaning, so synonyms, abbreviations, and a panel covering its components all count. The
    reference is built in case-building (CPC: case.orders from the narrative; MIMIC: case.findings keys).
    Reports the confusion cells - overlap (in both, TP), missed (human-only, FN), extra (work-up-only, FP) -
    and the two standard set-agreement summaries: F1 (= Dice = 2TP/(2TP+FP+FN)) and Jaccard (TP/(TP+FP+FN)).
    Empty input -> a blank record; LLM failure -> raises, so the driver skips and retries."""
    ordered = list(dict.fromkeys(ordered))
    reference = list(dict.fromkeys(reference))
    if not ordered or not reference:
        return {"overlap": 0, "missed": len(reference), "extra": len(ordered), "f1": None, "jaccard": None}
    user = f"REFERENCE:\n{_numbered(reference)}\n\nORDERED:\n{_numbered(ordered)}"
    rec = _ask(CONCORDANCE_SYSTEM, user, 500, model, backend)
    covered = len(_indices(rec.get("covered_reference"), len(reference)))     # human tests the work-up covered
    matched = len(_indices(rec.get("matched_ordered"), len(ordered)))         # work-up tests that hit a human test
    overlap, missed, extra = covered, len(reference) - covered, len(ordered) - matched
    union, dice = overlap + missed + extra, 2 * overlap + missed + extra
    return {"overlap": overlap, "missed": missed, "extra": extra,
            "f1": round(2 * overlap / dice, 3) if dice else 0.0,
            "jaccard": round(overlap / union, 3) if union else 0.0}


def judge_harm(ordered, model=None, backend=None):
    """Total cost and harm burden of a whole work-up. The LLM rates each ordered test's burden in [0,1]
    (invasiveness + radiation); cost stays deterministic (harness/cost.py). Returns
    {n_tests, cost_total, harm_mean, harm_max, n_high_harm}. LLM failure (or a count mismatch) raises,
    so the driver skips and retries rather than saving a partial score."""
    costs = [estimate_test_cost(t) for t in ordered]
    if not ordered:
        return {"n_tests": 0, "cost_total": 0, "harm_mean": 0.0, "harm_max": 0.0, "n_high_harm": 0}
    rec = _ask(HARM_SYSTEM, _numbered(ordered), 500, model, backend)
    harms = [max(0.0, min(1.0, float(h))) for h in rec.get("harms", [])]
    if len(harms) != len(ordered):
        raise ValueError(f"harm judge returned {len(harms)} scores for {len(ordered)} tests")
    return {"n_tests": len(ordered), "cost_total": round(sum(costs)),
            "harm_mean": round(sum(harms) / len(harms), 3), "harm_max": round(max(harms), 3),
            "n_high_harm": sum(1 for h in harms if h >= 0.5)}
