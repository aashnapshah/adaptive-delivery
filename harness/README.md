# harness

The pipeline, as an importable package.
Cross-stage imports are plain (`from harness.gatekeeper import SequentialDiagnosisEnv`).

## Three agents

Each is its own script, with its prompt(s) in a `prompts/` folder next to the code so you can edit a prompt without touching Python.

| Agent | Package | Role |
|-------|---------|------|
| **Gatekeeper** | `gatekeeper` | The oracle: presents the case at turn 0, then releases findings on request. One script dispatches by source - CPC (LLM over the vignette), MIMIC (labs-on-order), toy (rule-based panels). |
| **Lab generation** | `generation` | The diagnostic agent: asks / orders tests / commits a diagnosis. Three designs: `single` / `maidxo` / `debate`. |
| **Judge** | `judge` | Scores the committed diagnosis against the ground truth on a 5-point rubric (>= 4 correct). |

## Supporting modules

| Package | Role |
|---------|------|
| `cases` | Build case presentations from three sources into one `Case` (see `cases/schema.py`). |
| `cost` | Estimate the cost of a test / visit. A separate concern - the gatekeeper never prices; the environment tallies running spend and the scoring/eval stages price the workup, all through here. |
| `scoring` | Appropriateness + harm scoring of a recommendation. |
| `interface` | Delivery interface + simulated clinician (alert fatigue). |
| `policy` | Adaptive delivery policy (contextual bandit) vs fixed baselines. |
| `evaluation` | Run the benchmark (`record` / `budget` / `ablation` / `export`) AND measure it (accuracy-vs-cost, value-vs-burden). Recordings + results live here. |
| `shared` | The LLM backend (`llm.py`, Gemini by default), the prompt-file loader (`prompts.py`), and parsing helpers. |

## The case-source split

Every source loads into one `Case`, tagged by `source`, and the gatekeeper matches:

- **CPC** (`source="cpc"`) - a static NEJM vignette. The whole case is prose in `case_file`; the gatekeeper hands it to an LLM that discloses findings on request and answers history/exam questions.
- **MIMIC** (`source="mimic"`) - a real hospital admission. Thin intake `presentation`; real structured labs live in `findings` and are revealed only when the matching test is ordered (labs-on-order): panel expansion → analyte match → a cheap LLM name-match fallback; off-record synthesizes a normal, never paraphrasing. No history to ask for beyond intake. `orders` is the real order set, for the concordance metric.
- **toy** (`source="toy"`) - synthetic MIMIC-shaped records with rich `panels` and ground-truth `informative` / `key_tests` flags; served by the rule-based path. The offline fixtures for the whole pipeline.

## Editing prompts

Prompts live as `.txt` files under each agent's `prompts/` folder and load via `harness.shared.prompts.load`:

```
harness/gatekeeper/prompts/gatekeeper.txt
harness/judge/prompts/{diagnosis,workup,concordance,harm}.txt
harness/generation/prompts/{single,maidxo,debate_coord,advocate_*}.txt
harness/cases/prompts/cpc_split.txt
```

Editing a generation design's prompt changes its fingerprint, which forks a new recording file (old runs are kept).

## Running

```bash
python -m harness.cases.toy                 # list the toy cases
python -m harness.scoring.scoring           # a stage demo
python -m harness.evaluation.record         # record the benchmark (Gemini by default)
python -m harness.evaluation.export         # aggregate recordings -> CSVs + figures
python -m harness.evaluation.evaluation     # the offline measures + figures

# data builders (need a model / credentialed data)
DEMO_LLM_BACKEND=gemini python -m harness.cases.build_cpc_presentations --limit 5
MIMIC_ROOT=/path/to/mimiciv python -m harness.cases.mimic --build --n 100
```

## Backend

`shared/llm.py` auto-detects Gemini first (the current default; key in the repo-root `.env`), then OpenRouter, Ollama, or an offline stub. Force one with `DEMO_LLM_BACKEND=gemini|openrouter|ollama|stub`.
