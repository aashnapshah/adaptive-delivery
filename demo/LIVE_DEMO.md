# Live demo run order (advisor talk)

The deck embeds figures for everything below, so a live run is optional. If you go live,
this is the order, the one-line narration, and what should appear. Total ~6-8 min.

## One-time setup
```bash
cd demo
pip install -r requirements.txt          # numpy, matplotlib, jupyter, nbclient
cp .env.example .env                      # then paste OPENROUTER_API_KEY (for the real-model step)
```
Everything except the generation step runs offline; with no key it uses a deterministic stub.

## Terminal run order
| # | Command | Say | You should see |
|---|---------|-----|----------------|
| 1 | `python3 01_environment/demo_environment.py` | "A case as a sequential encounter; the gatekeeper reveals findings only when asked, with cost." | A trace: ask -> D-Dimer -> CT angiogram -> CORRECT, total cost $840 |
| 2 | `python3 02_generation/demo_generation.py` | "Same case, two agents: the paper's baseline prompt, then the 5-physician MAI-DxO panel." | Both reach PE; panel shows a Bayesian differential and comes in cheaper |
| 3 | `python3 03_scoring/demo_scoring.py` | "Each candidate scored for appropriateness and harm before delivery." | The CT scores high value AND high harm; a duplicate is flagged |
| 4 | `python3 04_interface/demo_interface.py` | "The clinician side: five delivery actions, and alert fatigue." | p(accept) decays from 0.77 as the same alert is repeated |
| 5 | `python3 05_policy/demo_policy.py` | "When/how to alert, learned. Naive rules collapse; adaptive policies recover value." | always-interrupt and never-alert deeply negative; adaptive policies positive |
| 6 | `python3 06_evaluation/demo_evaluation.py` | "Two axes: accuracy vs cost, and realized value vs alert burden." | A metrics table + three PNGs written |

> For step 2 to differentiate the agents you need the real model (the key in `.env`).
> Under the stub both agents are identical.

## Notebooks (if you prefer to scroll through with narration)
```bash
cd demo && jupyter lab        # open 01_environment/environment.ipynb ... 06_evaluation/evaluation.ipynb
```
Each notebook is the same content, top to bottom: goal -> pipeline diagram -> relevant work ->
what we're building -> how it works -> worked code.

## Clinician mockups (the visual crowd-pleaser)
```bash
cd demo/ui && python3 -m http.server        # then open the printed URL + /index.html
```
- `index.html` -> the three touchpoints.
- Delivery: pick an action, click **Deliver alert**, watch the burden meter; try interrupt vs passive.
- Expert review console: rate a couple items, see agreement with the automated score.
- Recommendation review: keep/drop tests from the Quadruple-Aim list.

## If live fails
Every screen above is already a slide in `research_plan/postdoc_plan_talk.pptx`. Just keep going.
```
