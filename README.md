# Adaptive Delivery of LLM-Generated Diagnostic Test Recommendations

An open, reproducible harness that reproduces and extends the sequential test-ordering benchmark of
**Nori et al. (2025)** — *Sequential Diagnosis with Language Models* (SDBench + MAI-DxO) — and studies
the **delivery** problem: not just whether an LLM's diagnostic recommendation is good, but whether it is
acted upon.

## Headline finding (full 100-case CPC run, Llama-3.1-8B)

Comparing three recommender designs of increasing orchestration complexity — a single LLM call, a single
call reasoning through the five MAI-DxO roles, and those five roles split across separate agents — **added
orchestration does not buy accuracy**. The two single-call designs are statistically indistinguishable
(both 23%), and the genuine multi-agent panel is the weakest (13%) while spending ~1.9× the tokens and
~4.3× the wall-clock time. See [`reports/2026-07-07-report.md`](reports/2026-07-07-report.md).

## Layout

```
harness/     the importable pipeline (see harness/README.md for the stage map)
  cases/       01  case presentations — MIMIC admissions, NEJM CPC narratives, toy EHR
  gatekeeper/  02  the sequential-diagnosis environment + the oracle that reveals findings
  generation/  03  recommender designs (single / maidxo / debate)
  scoring/     04  appropriateness + harm scoring
  interface/   05  delivery interface + simulated clinician (alert fatigue)
  policy/      06  adaptive delivery policy (contextual bandit)
  evaluation/  07  run the benchmark (record / sweeps / export) AND measure it; recordings + results live here
  shared/         the LLM backend (Gemini by default) + parsing helpers
app/         a localhost Flask study/demo UI over the recorded runs
reports/     the progress report (markdown + styled HTML) and figures
slides/      a short figure-forward deck
```

The MIMIC vs CPC distinction is deliberate: **CPC is a static vignette** (an LLM reads the narrative and
discloses on request), while **MIMIC is a hospital admission** whose real structured labs are revealed
only when the matching test is ordered (labs-on-order). See `harness/README.md`.

## Running

```bash
pip install -r requirements.txt
cp .env.example .env          # add a GEMINI_API_KEY (or leave empty for the offline stub)

python -m harness.cases.toy               # offline: list the toy cases
python -m harness.scoring.scoring         # offline: a stage demo
python -m app.app                          # the localhost UI (http://127.0.0.1:5050)
python -m harness.evaluation.record        # record the benchmark (Gemini by default)
```

Backend auto-detects Gemini first, then OpenRouter, Ollama, or an offline stub; force one with
`DEMO_LLM_BACKEND=gemini|openrouter|ollama|stub`.

## Data note

**No patient or copyrighted data is included in this repository.** The benchmark runs over MIMIC-IV
(PhysioNet credentialed, under a Data Use Agreement) and NEJM clinicopathological-conference cases
(copyrighted); neither may be redistributed, and both `harness/cases/data/` and the recordings/results
under `harness/evaluation/` are gitignored. The code runs offline on the synthetic MIMIC-shaped stand-ins in `harness/cases/toy.py`, and
you supply your own credentialed data locally (`harness/cases/mimic.py --build`) to reproduce the full runs.
MIMIC cases may only be sent to a **local** model (Ollama) — the recorder refuses a hosted API for them.
