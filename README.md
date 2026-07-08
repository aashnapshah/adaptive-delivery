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
demo/        the pipeline: environment, generation, scoring, interface, policy, evaluation
  shared/    synthetic MIMIC-shaped toy data so the demo runs fully offline
reports/     the progress report (markdown + styled HTML) and figures
slides/      a short figure-forward deck (deck.pdf)
```

## Running

```bash
pip install -r demo/requirements.txt
python3 demo/01-environment/demo_environment.py     # offline stub, no setup
DEMO_LLM_BACKEND=openrouter python3 demo/02-generation/demo_generation.py   # with a real model
```

## Data note

**No patient or copyrighted data is included in this repository.** The benchmark runs over MIMIC-IV
(PhysioNet credentialed, under a Data Use Agreement) and NEJM clinicopathological-conference cases
(copyrighted); neither may be redistributed. The code runs offline on the synthetic MIMIC-shaped stand-ins
in `demo/shared/toy_data.py`, and you supply your own credentialed data locally to reproduce the full runs.
