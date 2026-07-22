"""The recommender agent: reads the presentation and recommends a work-up. It talks to the gatekeeper in
one running conversation - order tests, read the results, order more, and STOP when done. Its job is the
ordering; it does not diagnose in the loop.

A design is just a prompt in prompts/<design>.txt: single (one call), roles (one call, several
perspectives), debate (five advocates consult, a coordinator decides). `diagnose()` is a separate
one-shot readout used to score CPC cases.
"""

import re

from ..gatekeeper import Gatekeeper
from ..shared import llm
from ..shared.prompts import load

SAFETY_CAP = 40   # runaway guard; the recommender ends by choosing STOP
ADVOCATES = ["hypothesis", "testchooser", "challenger", "stewardship", "checklist"]


def _parse(reply):
    """A reply -> ('order', [tests]) | ('stop', []) | ('none', [])."""
    m = re.search(r"ORDER:\s*(.+)", reply, re.I)
    tests = [t.strip() for t in re.split(r"[;\n]+", m.group(1)) if t.strip()][:5] if m else []
    if tests:
        return "order", tests
    return ("stop" if re.search(r"\bSTOP\b", reply, re.I) else "none"), []


def _advocates(presentation, transcript, model, backend):
    """debate: each advocate proposes the next step (one-shot, from the current state)."""
    results = "\n".join(f"- {n}: {r}" for n, r in transcript) or "(no tests yet)"
    state = f"{presentation}\n\nRESULTS SO FAR:\n{results}"
    fmt = load(__file__, "debate/format")
    votes = [llm.chat([{"role": "system", "content": load(__file__, f"debate/{a}") + "\n" + fmt},
                       {"role": "user", "content": state}], model=model, backend=backend, max_tokens=120)
             for a in ADVOCATES]
    return "\n".join(f"- {a}: {v.strip().splitlines()[0]}" for a, v in zip(ADVOCATES, votes))


def _decide(thread, design, presentation, transcript, model, backend):
    """The recommender's deliberation this turn: append its reply to `thread` and return
    (action, tests, telemetry). single/roles = one call; debate = advocates + a coordinator call.
    The gatekeeper is not called here, so `cap` is recommender-only tokens/latency."""
    with llm.capture() as cap:
        if design == "debate":
            ballot = _advocates(presentation, transcript, model, backend)
            thread.append({"role": "user", "content": f"SPECIALIST PROPOSALS:\n{ballot}\n\nReconcile into one action."})
        reply = llm.chat(thread, model=model, backend=backend)
    thread.append({"role": "assistant", "content": reply})
    action, tests = _parse(reply)
    return action, tests, cap


def recommend(case, design="single", model=None, backend=None):
    """Run the recommender as one running conversation with the gatekeeper.
    Returns {ordered, transcript, stopped, conversation, meta}."""
    gk = Gatekeeper(case, backend)
    thread = [{"role": "system", "content": load(__file__, "debate/coord" if design == "debate" else design)},
              {"role": "user", "content": f"PRESENTATION:\n{gk.presentation}\n\nOrder tests, or STOP."}]
    transcript, convo, stopped = [], [("gatekeeper", gk.presentation)], False
    meta = {"turns": 0, "calls": 0, "tokens": 0, "deliberate_s": 0.0}   # recommender-side only

    for _ in range(SAFETY_CAP):
        action, tests, cap = _decide(thread, design, gk.presentation, transcript, model, backend)
        meta["turns"] += 1
        meta["calls"] += cap["calls"]
        meta["tokens"] += cap["total_tokens"]
        meta["deliberate_s"] += cap["latency_s"]

        if action == "stop":
            convo.append(("recommender", "STOP - work-up complete."))
            stopped = True
            break
        if action == "none":                                    # unparseable: nudge and retry
            thread.append({"role": "user", "content": "Reply with 'ORDER: <tests>' or 'STOP'."})
            continue

        convo.append(("recommender", "Order: " + "; ".join(tests)))
        new = [(name, gk.ask(f"Order this test: {name}")) for name in tests]
        transcript += new
        convo += [("gatekeeper", f"{n} -> {r}") for n, r in new]
        thread.append({"role": "user", "content":
                       "RESULTS:\n" + "\n".join(f"{n}: {r}" for n, r in new) + "\n\nOrder more tests, or STOP."})

    meta["deliberate_s"] = round(meta["deliberate_s"], 2)
    return {"ordered": [n for n, _ in transcript], "transcript": transcript,
            "stopped": stopped, "conversation": convo, "meta": meta}


def diagnose(case, transcript, model=None, backend=None):
    """One-shot diagnosis readout from a completed work-up (for CPC scoring). Not part of the loop."""
    results = "\n".join(f"- {n}: {r}" for n, r in transcript) or "(no tests ordered)"
    presentation = (case.presentation or case.abstract or "").strip()
    messages = [
        {"role": "system", "content": "Given the presentation and the test results, state the single "
         "most specific diagnosis (disease, etiology, or organism). Reply with ONLY the diagnosis."},
        {"role": "user", "content": f"PRESENTATION:\n{presentation}\n\nRESULTS:\n{results}\n\nDiagnosis:"},
    ]
    return llm.chat(messages, model=model or llm.light_model(backend), backend=backend,
                    think=False).strip().splitlines()[0]
