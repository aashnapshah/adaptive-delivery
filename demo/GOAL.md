# GOAL — `demo/` — a Flask localhost app you click through

## What it is
Run `python3 app.py`, open `localhost`, click through **tabs that each showcase one part of the
pipeline.** The real LLM runs server-side (the key in `demo/.env` works → Llama-3.3-70B). The
heart of it is the **sequential diagnostic encounter**: you watch a model work a case turn by turn.

## Design — clean, minimalist, aesthetic (top priority)
Not busy. Whitespace is a feature. One quiet top bar, one slim patient line, a tab strip. One hero
per tab; everything else small and aside. Small neutral palette + one accent. A simple visual beats
a dense table. Few words; detail hides behind an expandable "?". If a tab feels full, remove
something.

## Tabs (each = a pipeline part; merge the v1 redundancy)
The gatekeeper, the environment, and "generation" were the same mechanism shown 3 times — merge.

1. **Diagnose** (the centrepiece). Pick a case (EHR or CPC). A live **transcript** is the hero:
   each turn shows the agent's brief reasoning, one action (ask / order / diagnose), the
   gatekeeper's revealed finding, and running cost — streamed from the server. Two controls:
   *drive (you ↔ agent)* and *design (Single LLM · MAI-DxO panel · Ours)*. Switching design on the
   same case changes the workup, cost, and dx — that contrast is the comparison.
2. **Deliver.** The tests the agent ordered → scored (appropriateness/harm); deliver one as an
   alert via the five actions; a simulated clinician accepts/dismisses with alert fatigue.
3. **Policy.** Adaptive delivery policy over a stream; naive rules collapse, adaptive beats them.
   One learning-curve chart + a small metrics table. Numbers reported honestly.

## How
`app.py` (Flask) reuses the existing modules — `SequentialDiagnosisEnv` (01), the agents (02),
`score_recommendation` (03), `ClinicianSimulator` (04), the policies (05) — and `shared/llm.py`
for the live model. In-memory per-session env. Key stays server-side; the browser only sees turns.
No key → deterministic stub, labelled offline. `templates/` + one small `static/` CSS+JS.

## Done =
`python3 app.py` → open localhost → pick a case → watch a real LLM diagnose it sequentially, switch
the design and see it change → Deliver scores + alerts those recs → Policy shows adaptive beating
naive rules. Three calm tabs, live where it matters.
