# UI / UX Master List

Legend: `[x]` done · `[ ]` pending · **P1** high-impact / **P2** nice-to-have

---

## Global (cross-cutting)
- [x] Title Case for all headers (Pipeline, Pareto Frontier, …)
- [x] Prettified model names (Llama 3.1 8B, not `llama3.1:8b`)
- [x] Clean status chip (● Live, not raw `openrouter:…` string)
- [x] Removed dead agent option (Multi-agent debate) from pickers
- [x] Centered empty-state cards (Benchmark, Simulation)
- [x] Friendly case labels (CPC Case 1) with raw ID muted
- [x] Removed "Synthetic data… not a clinical system" footer
- [x] References section (SDBench + NOHARM) in footer
- [ ] **P2** Standardize picker label casing/style ("Agent:") across tabs
- [ ] **P2** Trim every opening instructional hint to the shortest clear form
- [ ] **P1** Progressive disclosure everywhere: summary first, detail on expand

## Recommend
- [ ] **P1** Outcome-first: comparison strip at TOP — per agent: diagnosis, ✓/✗ vs gold, cost, # tests
- [ ] **P1** Collapse the transcript behind per-agent "View workup ▾" (default = verdict + table)
- [ ] **P1** Make Quad-Aim table the hero: highlight best cell per column + a "best overall" marker
- [ ] **P2** Collapse long gatekeeper lab dumps (CSF/EEG) to a one-line expandable
- [ ] **P2** Quad-Aim headers: move the Quadruple-Aim sub-labels into one legend line, not under every column
- [ ] **P2** Think-time chips (⏱ 1.2s): show once / on hover, not every bubble
- [ ] **P2** Case header: raw NEJM ID → hover tooltip

## Benchmark
- [ ] **P1** Action bar `[Ask] [Order test] [Diagnose]` instead of the dropdown
- [ ] **P1** Live scorecard pinned: running cost · # questions · # tests · elapsed
- [ ] **P1** Diagnose = distinct commit → ends case, reveals how you compared to the agents
- [ ] **P2** Tighten intro-card body copy

## Grading
- [x] Removed repeated "Grade this recommendation —" (just the test name)
- [x] Clean ✓ done-state (no "Recorded — appropriate" text)
- [x] Deduplicate re-ordered tests (grade each unique test once)
- [x] Faithful two-axis NOHARM (Appropriateness → Harm if inappropriate)
- [x] Honest rubric wording ("adapted from NOHARM"; omission out of scope)
- [ ] **P1** Progress indicator ("Graded 3 / 7")
- [ ] **P1** End-of-pass summary (appropriate vs inappropriate, harm severities, agreement w/ auto-scorer)
- [ ] **P2** Rubric as a "?" side reference instead of an inline block
- [ ] **P2** Instruction → "Score each test to reveal its result and continue."
- [ ] **P2** Trim the rubric intro paragraph

## Simulation
- [ ] **P1** Shift header bar: clock · handled/total · pending count
- [ ] **P2** Urgency cue — patient tabs redden as unhandled alerts pile up
- [ ] **P1** End-of-shift report: ordered/deferred/rejected + one line on alert burden
- [ ] **P2** Tighten intro-card body copy

## Results
- [x] Method/Result/Interpretation bubbles per section
- [x] Legend chips + dashed Pareto swatch
- [x] Capped breakdown bar width + subtle border
- [ ] **P1** Stat tiles at the very top: best accuracy · its cost · n cases
- [ ] **P2** Tighter plot y-axis (points sit squashed at the bottom of 0–100)
- [ ] **P2** Drop "We…" preamble on Method rows where redundant
- [ ] **P2** Make Interpretation row optional when it adds little

---

### Suggested build order
1. Recommend outcome-first + collapsible workup (biggest visual payoff)
2. Benchmark live scorecard + action bar
3. Grading progress + end summary
4. Simulation shift header + report
5. Results stat tiles
6. Global wording/label trims (fast cleanup pass)
