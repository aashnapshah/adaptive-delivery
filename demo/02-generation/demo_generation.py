"""Demo 02 - Recommendation Generation: baseline prompt, then MAI-DxO panel.

Maps to Methods section: "Recommendation Generation".

We replicate two of the diagnostic agents from Nori et al. (2025) on top of the
SDBench-style EHR environment from Demo 01:

  1. BASELINE  - the paper's minimal prompt (their Figure 4): the model emits
     <question>, <test>, and <diagnosis> tags. A fair "out-of-the-box" control.

  2. MAI-DxO PANEL - one model call role-plays a five-physician virtual panel
     (Dr. Hypothesis / Test-Chooser / Challenger / Stewardship / Checklist),
     runs a short "Chain of Debate", and commits to one action per turn.

Running both on the same cases shows the accuracy/cost lift the panel buys.

Backends auto-detected by ../shared/llm.py (OpenRouter -> Ollama -> offline stub).

Run:  python3 demo_generation.py
"""

from __future__ import annotations

import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "01_environment")))
from shared import llm  # noqa: E402
from shared.toy_data import load_case  # noqa: E402
from demo_environment import SequentialDiagnosisEnv  # noqa: E402

MAX_TURNS = 8


def _resolve(case_id):
    """Accept a case_id (toy cases) or a case object (e.g. a CPC narrative)."""
    return load_case(case_id) if isinstance(case_id, str) else case_id


def _orderable(env):
    t = env.orderable_tests()
    return f"Orderable tests: {t}" if t else "Order any diagnostic test by name (free text)."


def _make_env(case, backend):
    """LLM gatekeeper for narrative cases (with a case_file), rule-based otherwise."""
    return SequentialDiagnosisEnv(case=case, backend=backend,
                                  llm_gatekeeper=getattr(case, "case_file", None) is not None)


# ============================================================================
# 1. BASELINE agent  (paper Figure 4)
# ============================================================================

BASELINE_SYSTEM = (
    "You are the clinician deciding the work-up for a patient who has just presented. You are given "
    "the presenting history and vital signs; reach the correct diagnosis by ordering a focused "
    "SERIES of diagnostic tests. Each turn, take ONE action using XML tags:\n"
    "  <question>ask one specific follow-up about history or exam</question>\n"
    "  <test>order one named diagnostic test (lab, imaging, microbiology, or pathology)</test>\n"
    "  <diagnosis>commit your single final diagnosis</diagnosis>\n"
    "Order tests by their exact name. Do not mix tag types in one turn. Keep "
    "testing while the picture is ambiguous; commit only when a result pins the diagnosis. When you "
    "diagnose, name the SPECIFIC disease, etiology, or organism the evidence supports — the precise "
    "entity (the causative organism or exact disease), not a broad syndrome or category (e.g. not "
    "'viral infection', 'meningitis', 'pneumonia', or 'cancer')."
)

_TAG = re.compile(r"<(question|test|diagnosis)>(.*?)</\1>", re.DOTALL | re.IGNORECASE)


def parse_baseline(text: str) -> tuple[str, str] | None:
    m = _TAG.search(text)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).strip()


def make_baseline_stub(env: SequentialDiagnosisEnv):
    def handler(_messages):
        case = env.case
        if case.confirmable_by(env.ordered):
            return f"<diagnosis>{case.true_diagnosis}</diagnosis>"
        if not env.asked:
            return "<question>What is the history of present illness and key risk factors?</question>"
        pool = [p for p in env.orderable_tests() if case.panels[p].informative] or env.orderable_tests()
        if not pool:
            return f"<diagnosis>{case.true_diagnosis}</diagnosis>"
        choice = min(pool, key=lambda p: case.panels[p].cost)
        return f"<test>{choice}</test>"
    return handler


def run_baseline(case_id: str = "PE-2180", backend: str | None = None) -> dict:
    case = _resolve(case_id)
    env = _make_env(case, backend)
    llm.set_stub_handler(make_baseline_stub(env))
    print("-" * 80)
    print(f"BASELINE | case {case.case_id} | model = {llm.active_model(backend)}")
    print(f"case: {case.abstract}")
    print("-" * 80)

    prev = None
    for turn in range(MAX_TURNS):
        final = turn == MAX_TURNS - 1
        sys_prompt = BASELINE_SYSTEM + (
            "\nThis is your final turn: you MUST output a <diagnosis> now." if final else "")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": env.observation() + "\n\n" + _orderable(env) + "\nTake one action."},
        ]
        parsed = parse_baseline(llm.chat(messages, backend=backend))
        if parsed is None:
            print("  [no valid tag; forcing final diagnosis]")
            break
        action, content = parsed
        if action == "diagnosis":
            print(f"  diagnose -> {content}")
            print("   ", env.diagnose(content).response)
            return _summary(env)
        if (action, content) == prev:        # loop guard: model is stuck repeating
            print("  [repeated action; forcing final diagnosis]")
            break
        prev = (action, content)
        step = env.ask_question(content) if action == "question" else env.order_test(content)
        finding = step.response.strip().splitlines()[-1][:50]
        print(f"  turn {turn+1}: {action:<8} {content[:28]:<29} +${step.cost:<4.0f} -> {finding}")

    if not env.done:                         # never committed -> ask once for a final dx
        _force_diagnosis(env, backend)
    return _summary(env)


# ============================================================================
# 2. MAI-DxO virtual panel  (paper Section 3.2)
# ============================================================================

# All-roles: the SAME single call and output format as the baseline, but the model is asked to
# CONSIDER several clinical perspectives before it acts. Single vs All-roles therefore differ in
# exactly one thing — this "consider these" instruction — so any effect (and its token cost) is
# attributable to it, not to a heavier output format.
PANEL_SYSTEM = (
    "You are the clinician deciding the work-up for a patient who has just presented. You are given "
    "the presenting history and vital signs; reach the correct diagnosis by ordering a focused SERIES "
    "of diagnostic tests.\n"
    "Before each action, briefly CONSIDER these five expert perspectives together when choosing what to "
    "order (you do not need to write them out): (1) Hypothesis — keep a probability-ranked differential "
    "of the three most likely conditions, updated in a Bayesian way as findings emerge; (2) Test-Chooser "
    "— pick the test that best DISCRIMINATES among the leading hypotheses; (3) Challenger — play devil's "
    "advocate: watch for anchoring bias, weigh contradictory evidence, and consider a test that could "
    "FALSIFY the leading diagnosis; (4) Stewardship — prefer a lower-cost alternative when diagnostically "
    "equivalent and avoid expensive low-yield tests; (5) Checklist — confirm the test name is valid and "
    "the reasoning is internally consistent.\n"
    "Then take ONE action using XML tags:\n"
    "  <question>ask one specific follow-up about history or exam</question>\n"
    "  <test>order one named diagnostic test (lab, imaging, microbiology, or pathology)</test>\n"
    "  <diagnosis>commit your single final diagnosis</diagnosis>\n"
    "Order tests by their exact name. Do not mix tag types in one turn. Keep testing while the picture "
    "is ambiguous; commit only when a result pins the diagnosis. When you diagnose, name the SPECIFIC "
    "disease, etiology, or organism the evidence supports — the precise entity (the causative organism "
    "or exact disease), not a broad syndrome or category (e.g. not 'viral infection', 'meningitis', "
    "'pneumonia', or 'cancer')."
)


def _extract_json(text: str) -> dict:
    if not text or not text.strip():
        raise ValueError("empty model output")
    cleaned = re.sub(r"```(?:json)?|```", "", text)   # strip code fences
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in: {text[:160]}")
    return json.loads(m.group(0))


def _force_diagnosis(env: SequentialDiagnosisEnv, backend: str | None) -> None:
    """Ask the model for a single final diagnosis string and commit it.

    Used when an agent exhausts its turns without diagnosing, so every episode
    terminates with a verdict (honest: a wrong guess still scores incorrect).
    """
    messages = [
        {"role": "system", "content": "Give your single best final diagnosis. "
         "Reply with ONLY the diagnosis name, no tags, no JSON, no explanation."},
        {"role": "user", "content": env.observation()},
    ]
    try:
        raw = llm.chat(messages, backend=backend, max_tokens=40)
    except Exception:
        raw = ""
    dx = re.sub(r"</?\w+>|[{}\"]", "", raw).strip().splitlines()[0][:80] if raw.strip() else "undetermined"
    print(f"  forced diagnose -> {dx}")
    print("   ", env.diagnose(dx or "undetermined").response)


def make_panel_stub(env: SequentialDiagnosisEnv):
    def handler(_messages):
        case = env.case
        diff = [{"dx": case.true_diagnosis, "p": 0.6}, {"dx": "alternative", "p": 0.2}]
        if case.confirmable_by(env.ordered):
            return json.dumps({"differential": diff, "debate": "Confirmatory tests are in; panel concurs.",
                               "action": "diagnose", "query": None, "diagnosis": case.true_diagnosis})
        if not env.asked:
            return json.dumps({"differential": diff,
                               "debate": "Dr. Hypothesis wants the history before committing tests.",
                               "action": "ask", "query": "History of present illness and risk factors?",
                               "diagnosis": None})
        pool = [p for p in env.orderable_tests() if case.panels[p].informative] or env.orderable_tests()
        choice = min(pool, key=lambda p: case.panels[p].cost) if pool else None
        if choice is None:
            return json.dumps({"differential": diff, "debate": "No further tests warranted.",
                               "action": "diagnose", "query": None, "diagnosis": case.true_diagnosis})
        return json.dumps({"differential": diff,
                           "debate": f"Test-Chooser favors {choice}; Stewardship approves the cost.",
                           "action": "test", "query": choice, "diagnosis": None})
    return handler


def run_panel(case_id: str = "PE-2180", backend: str | None = None,
              system: str = PANEL_SYSTEM, label: str = "MAI-DxO PANEL") -> dict:
    case = _resolve(case_id)
    env = _make_env(case, backend)
    llm.set_stub_handler(make_panel_stub(env))
    print("-" * 80)
    print(f"{label} | case {case.case_id} | model = {llm.active_model(backend)}")
    print(f"case: {case.abstract}")
    print("-" * 80)

    for turn in range(MAX_TURNS):
        final = turn == MAX_TURNS - 1
        sys_prompt = system + (
            "\nThis is your final turn: action MUST be 'diagnose'." if final else "")
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": env.observation() + "\n\n" + _orderable(env) + "\nDeliberate, then act."},
        ]
        try:
            rec = _extract_json(llm.chat(messages, backend=backend))
        except Exception:                    # one stricter retry, then force a diagnosis
            try:
                rec = _extract_json(llm.chat(
                    messages + [{"role": "user", "content": "Return ONLY the JSON object."}],
                    backend=backend))
            except Exception:
                print("  [panel returned no parseable JSON; forcing final diagnosis]")
                break
        diff = ", ".join(f"{d.get('dx')} ({d.get('p')})" for d in rec.get("differential", [])[:3])
        print(f"  differential: {diff}")
        print(f"  debate: {rec.get('debate', '')[:90]}")
        action = rec.get("action")
        if action == "diagnose":
            dx = rec.get("diagnosis") or "(none)"
            print(f"  -> DIAGNOSE: {dx}")
            print("   ", env.diagnose(dx).response, "\n")
            return _summary(env)
        query = rec.get("query") or ""
        step = env.ask_question(query) if action == "ask" else env.order_test(query)
        finding = step.response.strip().splitlines()[-1][:50]
        print(f"  turn {turn+1}: {str(action).upper():<8} {query[:28]:<29} +${step.cost:<4.0f} -> {finding}\n")

    if not env.done:
        _force_diagnosis(env, backend)
    return _summary(env)


# ============================================================================

def _summary(env: SequentialDiagnosisEnv) -> dict:
    correct = env.final_diagnosis and "CORRECT" in env.transcript[-1]
    tests = [ln.split("Test [")[1].split("]")[0] for ln in env.transcript if ln.startswith("Test [")]
    print(f"  SUMMARY: turns={env.turn}  cost=${env.total_cost:.0f}  "
          f"dx={env.final_diagnosis!r}  correct={bool(correct)}\n")
    return {"turns": env.turn, "cost": env.total_cost, "tests": tests,
            "diagnosis": env.final_diagnosis, "correct": bool(correct)}


# ============================================================================

if __name__ == "__main__":
    print("\n### Same case, two single-call recommenders ###\n")
    print("(The genuine multi-agent 'debate' design lives in app.py — it issues\n"
          " separate advocate + coordinator LLM calls, not a single role-play prompt.)\n")
    run_baseline("PE-2180")
    run_panel("PE-2180")
