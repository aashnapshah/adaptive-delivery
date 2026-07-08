"""Generate the benchmark.

For every (model x dataset-case x agent design) run the agent to completion through the
gatekeeper and save the turn-by-turn transcript. The Flask app then replays these instantly;
only the clinician-driven Benchmark path stays live.

Models : OpenRouter ids — from $BENCHMARK_MODELS (comma-separated) or data/benchmark_models.txt
         (one per line, '#' comments allowed); defaults to the .env model.
Datasets/cases : whatever the registry exposes (all_case_options) — MIMIC-IV / CPC / Clalit
         once their loaders are registered.
Designs : the agent designs in app.DESIGNS (single LLM, all-roles prompt, multi-agent).

RESUMABLE: each (model, case, design) is checkpointed after it finishes; re-running skips
combos already recorded, so a run interrupted by rate limits / crashes just continues.

Output : data/recordings/<model-slug>.json   (case -> design -> run)
         data/recordings.json                 kept in sync with PRIMARY_MODEL so the live UI works.

Run:  python3 record.py
      BENCHMARK_MODELS="openai/gpt-4o, anthropic/claude-3.5-sonnet, meta-llama/llama-3.3-70b-instruct" python3 record.py
"""

from __future__ import annotations

import datetime
import json
import os
import re
import time

import app  # reuses the same env / agents / judge as the live app
from app import (DESIGNS, DEFAULT_MODEL, MAX_TURNS, _make_env, agent_turn,
                 all_case_options, full_vignette, load_any, prompt_fp, run_key)
from shared import llm

HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "data", "recordings")
INDEX_PATH = os.path.join(REC_DIR, "index.jsonl")     # append-only run log (time + tokens per run)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_STOP = {"test", "panel", "blood", "serum", "plasma", "level", "levels", "of", "and", "the", "a",
         "count", "for", "study", "studies", "total"}


def _toks(s):
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 1}


def _token_concordance(recommended, real):
    """Fallback: approximate token-overlap match (misses panel->analyte, e.g. CBC->hematocrit)."""
    rec = [_toks(r) for r in recommended]
    covered = sum(1 for lab in real if any(_toks(lab) & rt for rt in rec))
    hit = sum(1 for rt in rec if any(rt & _toks(lab) for lab in real))
    return {"recommended_n": len(recommended), "real_n": len(real), "covered": covered,
            "recall": round(covered / len(real), 3) if real else None,
            "precision": round(hit / len(rec), 3) if rec else None, "matcher": "token"}


_CONCORD_SYS = (
    "You match a clinician's ORDERED tests against the labs ACTUALLY MEASURED for a patient. A panel "
    "covers its component analytes: 'CBC'/'complete blood count' -> hemoglobin, hematocrit, white blood "
    "cells, platelets, RBC, MCV, RDW; 'BMP'/'basic metabolic panel' -> sodium, potassium, chloride, "
    "bicarbonate, BUN/urea, creatinine, glucose, calcium; 'CMP' -> BMP plus AST, ALT, alkaline "
    "phosphatase, bilirubin, albumin, total protein; 'LFTs' -> AST, ALT, alk phos, bilirubin; "
    "'coags' -> PT, INR, PTT. A MEASURED lab is COVERED if the clinician ordered it directly OR via a "
    "panel that includes it. Reply ONLY JSON: {\"covered\": [<the MEASURED labs, copied verbatim, that "
    "were ordered>], \"matched\": [<the ORDERED tests that correspond to at least one measured lab>]}."
)


def concordance(recommended, real, model=None):
    """Ordering concordance: did the recommended tests match the labs actually measured? An LLM does
    the matching (knows panels cover analytes); falls back to token overlap if it fails."""
    real, recommended = sorted(real), sorted(recommended)
    if not real or not recommended:
        return {"recommended_n": len(recommended), "real_n": len(real), "covered": 0,
                "recall": None, "precision": None, "matcher": "empty"}
    user = ("ORDERED (by the agent):\n" + "\n".join(f"- {r}" for r in recommended)
            + "\n\nMEASURED (in real life):\n" + "\n".join(f"- {r}" for r in real))
    try:
        raw = llm.chat([{"role": "system", "content": _CONCORD_SYS}, {"role": "user", "content": user}],
                       model=model, think=False, max_tokens=900)
        m = re.search(r"\{.*\}", re.sub(r"```(?:json)?|```", "", raw), re.DOTALL)
        rec = json.loads(m.group(0))
        cov = len({c for c in rec.get("covered", []) if c})
        mat = len({c for c in rec.get("matched", []) if c})
    except Exception:
        return _token_concordance(recommended, real)
    cov, mat = min(cov, len(real)), min(mat, len(recommended))
    return {"recommended_n": len(recommended), "real_n": len(real), "covered": cov,
            "recall": round(cov / len(real), 3), "precision": round(mat / len(recommended), 3),
            "matcher": "llm"}


def models() -> list[str]:
    env = os.environ.get("BENCHMARK_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    cfg = os.path.join(HERE, "data", "benchmark_models.txt")
    if os.path.exists(cfg):
        out = [ln.strip() for ln in open(cfg) if ln.strip() and not ln.lstrip().startswith("#")]
        if out:
            return out
    return [DEFAULT_MODEL]


def slug(model: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", model.lower()).strip("-")


def record_one(case_id: str, design: str, model: str, budget: float | None = None) -> dict:
    case, mode = load_any(case_id)
    if mode == "mimic" and llm.detect_backend() not in ("ollama", "stub"):
        raise RuntimeError("MIMIC cases contain real credentialed patient data — run on a LOCAL "
                           "backend only (DEMO_LLM_BACKEND=ollama). Refusing to send to a hosted API (DUA).")
    env = _make_env(case)
    turns = []
    run_t0 = time.time()
    for _ in range(10):
        # budget knob: once spend reaches the cap, force the agent to commit a diagnosis (SDBench cost sweep)
        final = env.turn >= MAX_TURNS - 1 or (budget is not None and env.total_cost >= budget)
        turn_t0 = time.time()
        with llm.capture() as cap:                  # tokens + latency for every model call this turn
            try:
                t = agent_turn(env, design, model=model, final=final)
            except Exception as e:  # keep going; record a terminal error turn (so resume can retry it)
                t = {"action": "diagnose", "query": "(unavailable)", "finding": f"error: {e}",
                     "done": True, "correct": False, "cost": 0, "total_cost": round(env.total_cost),
                     "turn": env.turn + 1, "reasoning": "", "differential": [],
                     "ordered": sorted(env.ordered), "n_questions": len(env.asked),
                     "n_tests": len(env.ordered), "error": True}
        t["meta"] = {"duration_s": round(time.time() - turn_t0, 3), "llm_calls": cap["calls"],
                     "prompt_tokens": cap["prompt_tokens"], "completion_tokens": cap["completion_tokens"],
                     "total_tokens": cap["total_tokens"]}
        turns.append(t)
        if t.get("done"):
            break

    tot = lambda k: sum((tt.get("meta") or {}).get(k, 0) for tt in turns)
    meta = {"run_key": run_key(model, design, case_id), "prompt_fp": prompt_fp(design),
            "model": model, "design": design, "case_id": case_id, "mode": mode,
            "gatekeeper": "llm" if env.llm_gatekeeper else "rule", "recorded_at": _now_iso(),
            "duration_s": round(time.time() - run_t0, 2), "llm_calls": tot("llm_calls"),
            "prompt_tokens": tot("prompt_tokens"), "completion_tokens": tot("completion_tokens"),
            "total_tokens": tot("total_tokens")}
    if mode == "mimic":                        # ordering concordance vs the real order set (LLM-matched)
        meta["concordance"] = concordance(sorted(env.ordered), app.MIMIC_ORDERS.get(case_id, set()), model=model)
    # presentation = exactly what the agent saw at turn 0 (history + exam), for the UI to display
    presentation = (getattr(case, "presentation", "") or "").strip() or case.abstract
    return {"abstract": case.abstract, "presentation": presentation,
            "full_case": full_vignette(case), "true_dx": case.true_diagnosis,
            "mode": mode, "model": model, "turns": turns, "meta": meta}


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _is_done(run: dict) -> bool:
    """A recorded run counts as done only if it has turns and didn't end on an error turn."""
    turns = (run or {}).get("turns") or []
    return bool(turns) and not turns[-1].get("error")


def _append_index(run: dict) -> None:
    """Append one row per completed run to index.jsonl — the queryable time/token log."""
    row = dict(run.get("meta") or {})
    turns = run.get("turns") or []
    dx = next((t for t in turns if t.get("action") == "diagnose"), None)
    row["correct"] = bool(dx.get("correct")) if dx else None
    row["judge_score"] = dx.get("judge_score") if dx else None
    row["total_cost"] = turns[-1].get("total_cost") if turns else None
    row["n_turns"] = len(turns)
    with open(INDEX_PATH, "a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    os.makedirs(REC_DIR, exist_ok=True)
    cases = all_case_options()
    # Optional scoping knobs (handy for a quick token/latency pass):
    #   BENCHMARK_MODE=cpc|ehr   restrict to one dataset
    #   BENCHMARK_MAX_CASES=15   take only the first N cases
    mode_f = os.environ.get("BENCHMARK_MODE", "").strip().lower()
    if mode_f:                                # cpc | ehr | mimic
        cases = [c for c in cases if c.get("mode") == mode_f]
    maxc = os.environ.get("BENCHMARK_MAX_CASES")
    if maxc and maxc.strip().isdigit():
        cases = cases[:int(maxc)]
    ms = models()
    sel = [x.strip() for x in os.environ.get("BENCHMARK_DESIGNS", "").split(",") if x.strip()]
    designs = sel or list(DESIGNS)
    total = len(ms) * len(cases) * len(designs)
    n = done = 0
    print(f"benchmark: {len(ms)} models x {len(cases)} cases x {len(designs)} designs = {total} runs")
    for model in ms:
        # one content-keyed file per design; open them all up front
        files = {}
        for d in designs:
            fp = prompt_fp(d)                                  # prompt fingerprint -> its own file
            path = os.path.join(REC_DIR, slug(model), f"{d}@{fp}.json")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            files[d] = (path, _load(path))
        for o in cases:                                        # CASE-major: every case does all designs,
            cid = o["id"]                                      # so the design comparison fills in immediately
            for d in designs:
                path, out = files[d]
                n += 1
                if _is_done(out.get(cid)):                     # content-keyed resume: an edited prompt
                    done += 1                                  # gets a new fp -> new file -> re-records
                    print(f"[{n}/{total}] skip {model} / {cid} / {d}")
                    continue
                print(f"[{n}/{total}] record {model} / {cid} / {d} ...", flush=True)
                run = record_one(cid, d, model)
                out[cid] = run
                with open(path, "w") as f:                     # checkpoint after every run
                    json.dump(out, f, indent=1)
                _append_index(run)
    print(f"done — {total - done} newly recorded, {done} already present. "
          f"per-(model,prompt) files + index.jsonl under {REC_DIR}/")


if __name__ == "__main__":
    main()
