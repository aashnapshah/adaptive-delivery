# Demos

Runnable, tutorial-style walkthroughs of each stage of the research plan
*"Learning When to Alert: Adaptive Delivery of LLM-Generated Diagnostic Test
Recommendations."* Each stage has a **script** (`demo_*.py`, run end-to-end) and a
**notebook** (`*.ipynb`) that follows the same layout: the section title, a short
description, the pipeline diagram with the current stage highlighted, a "how it works"
figure, then the worked code.

The environment and generation stages replicate Nori et al. (2025), *Sequential Diagnosis
with Language Models* (SDBench + MAI-DxO), adapted to run over **structured, MIMIC-IV-shaped
EHR records** (`shared/toy_data.py`) instead of NEJM narratives. Records are small synthetic
stand-ins shaped like real MIMIC tables - real MIMIC-IV needs PhysioNet credentialing + a
DUA, and swapping it in is a loader change. Stages that use a model call a real LLM when
configured (OpenRouter / Ollama) and fall back to a deterministic offline stub otherwise.

## Stages (map 1:1 to the Methods sections)

| Folder | Methods section | What it shows | Uses a model? |
|--------|-----------------|---------------|---------------|
| `01_environment/` | Datasets & Sequential Environment | SDBench-style Gatekeeper loop over a MIMIC-shaped record | optional |
| `02_generation/`  | Recommendation Generation | Paper's baseline prompt, then the MAI-DxO 5-persona panel | **yes** |
| `03_scoring/`     | Appropriateness & Harm Scoring | Score each recommendation: appropriateness + harm/cost | optional |
| `04_interface/`   | Delivery Interface & Clinician Response | Clinician accept/dismiss model with alert fatigue | optional |
| `05_policy/`      | Adaptive Delivery Policy | LinUCB contextual bandit for when-to-alert (+ background) | no |
| `06_evaluation/`  | Evaluation | Metrics + learning curves + value-vs-burden Pareto | no |

Each notebook follows the same layout: **title -> goal -> pipeline diagram (this stage
highlighted) -> relevant work -> what we're building & how it improves on prior work ->
how it works (schematic + expected input/output) -> worked code**.

## Running

```bash
# offline, no setup (deterministic stub):
python3 01_environment/demo_environment.py
python3 02_generation/demo_generation.py

# with a real model (any model via OpenRouter):
cp .env.example .env        # then add OPENROUTER_API_KEY
pip install -r requirements.txt
DEMO_LLM_BACKEND=openrouter python3 02_generation/demo_generation.py

# notebooks:
jupyter lab        # open any *.ipynb
```

## LLM backend (`shared/llm.py`)

One interface, three backends, auto-detected in this order:

1. **OpenRouter** - any model, set `OPENROUTER_API_KEY` (+ optional `OPENROUTER_MODEL`).
2. **Ollama** - local model, requires `ollama serve` (+ optional `OLLAMA_MODEL`).
3. **stub** - deterministic offline fallback so notebooks never hard-fail.

Force one with `DEMO_LLM_BACKEND=openrouter|ollama|stub`. See `.env.example`.

## Layout

```
demo/
├── shared/
│   ├── toy_data.py     # 3 synthetic cases reused everywhere
│   └── llm.py          # OpenRouter / Ollama / stub interface
├── 01_environment/     # demo_environment.py + environment.ipynb
├── 02_generation/      # demo_generation.py + generation.ipynb
└── ...                 # 03-06 follow the same shape
```
