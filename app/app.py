"""Flask app: a localhost UI that walks one patient through the pipeline.

Run:  python -m app.app   then open http://127.0.0.1:5050

Tabs:
  1. Diagnose - replay a recorded LLM working a case turn by turn through the gatekeeper; switch the
     agent design and case source and see the workup change.
  2. Deliver  - score the ordered tests, deliver one as an alert, simulated clinician with fatigue.
  3. Results  - the recorded benchmark: accuracy/cost, concordance, budget + ablation sweeps.

The LLM runs server-side via harness.shared.llm (Gemini by default; key in the repo-root .env). Agent
runs are static replays (recorded by benchmark/record.py); only the clinician-driven path calls a model
live. All pipeline logic is imported from the harness package - there is no copy of it here.
"""

from __future__ import annotations

import json
import os
import uuid

from flask import Flask, jsonify, render_template, request, session

from harness.cases import (  # noqa: F401
    CPC_BY_ID, MIMIC_BY_ID, TOY_CASES, case_options, get_case, load_case, refined_case_ids,
)
from harness.gatekeeper import SequentialDiagnosisEnv, full_vignette
from harness.generation import DESIGNS, apply_turn, prompt_fp  # noqa: F401
from harness.scoring import score_recommendation
from harness.shared import llm

from harness.evaluation.store import REC_DIR, iter_runs, run_file  # noqa: F401  recording storage layer

from . import store   # SQLite study store (app/store.py)

HERE = os.path.dirname(os.path.abspath(__file__))          # app/
REPO = os.path.dirname(HERE)                               # repo root
CASES_DATA = os.path.join(REPO, "harness", "cases", "data")
BENCH_RESULTS = os.path.join(REPO, "harness", "evaluation", "results")

app = Flask(__name__)
app.secret_key = os.environ.get("STUDY_SECRET", "demo-not-secret")
app.config["TEMPLATES_AUTO_RELOAD"] = True      # re-read templates on edit even in prod mode
store.init()

# Default agent model + the UI model picker. Gemini is the current default (key in .env); other models
# are added for benchmarking. Any OpenRouter id the key can reach also works.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
MODELS = [
    {"id": DEFAULT_MODEL, "label": DEFAULT_MODEL},
    {"id": "gemini-2.5-pro", "label": "gemini-2.5-pro"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "label": "llama-3.3-70b (openrouter)"},
]
_seen = set()
MODELS = [m for m in MODELS if not (m["id"] in _seen or _seen.add(m["id"]))]

# MIMIC real order sets, keyed by case id (for the ordering-concordance metric).
MIMIC_ORDERS = {cid: set(c.orders) for cid, c in MIMIC_BY_ID.items()}

# Map a case source to the app/UI mode vocabulary the templates + recordings use.
_SOURCE_TO_MODE = {"toy": "ehr", "cpc": "cpc", "mimic": "mimic"}
_MODE_TAG = {"ehr": "", "cpc": " (CPC)", "mimic": " (MIMIC)"}


def all_case_options():
    """UI case list with the app's mode vocabulary (ehr / cpc / mimic)."""
    opts = []
    for o in case_options():
        mode = _SOURCE_TO_MODE.get(o["source"], o["source"])
        opts.append({"id": o["id"], "mode": mode,
                     "label": f"{o['id']}{_MODE_TAG.get(mode, '')} - {o['label'][:60]}"})
    return opts


def load_any(case_id):
    """(case, mode) for any case id; mode in {ehr, cpc, mimic}."""
    case = get_case(case_id)
    return case, _SOURCE_TO_MODE.get(case.source, case.source)


# ---- per-session env store (in memory; the Benchmark turns are durably logged in SQLite) ----
SESSIONS: dict[str, dict] = {}


def _sid():
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def _pid():
    return session.get("pid")


def _attribution():
    """The (session_id, pid) every study event is tagged with."""
    return {"session_id": _sid(), "pid": _pid()}


def _state():
    return SESSIONS.get(_sid())


def _make_env(case, gatekeeper="auto"):
    """Build the environment. The gatekeeper is always the LLM oracle now; the `gatekeeper` arg is kept
    for API compatibility but no longer selects a rule-based path."""
    return SequentialDiagnosisEnv(case=case)

# ---- routes ----------------------------------------------------------------


@app.get("/")
def index():
    # Participant link: /?pid=ABC attributes this browser to a participant. No pid -> generate
    # an anonymous one so nothing is ever unattributed.
    pid = request.args.get("pid") or session.get("pid") or ("anon-" + uuid.uuid4().hex[:8])
    session["pid"] = pid
    store.upsert_participant(pid)
    store.ensure_session(_sid(), pid, request.headers.get("User-Agent"))
    refined = refined_case_ids()                  # show only the refined cases (if any refined yet)
    recorded_modes = {}                           # cases that actually have a recording to replay
    for _model, _d, _fp, cid, run in iter_runs():
        if refined and cid not in refined:
            continue
        recorded_modes.setdefault(cid, run.get("mode") or "cpc")
    recorded = sorted(recorded_modes)
    cases = [c for c in all_case_options() if not refined or c["id"] in refined]
    return render_template("index.html",
                           cases=cases,
                           recorded=recorded,
                           recorded_modes=recorded_modes,
                           backend=llm.active_model(),
                           live=llm.detect_backend() != "stub",
                           models=MODELS,
                           pid=pid,
                           consented=True,  # consent gate disabled (advisor/demo share)
                           designs=[{"id": k, "label": v[0]} for k, v in DESIGNS.items()])


@app.get("/api/models")
def api_models():
    """Models that have replayable recordings, for the UI model picker. One entry per model
    slug with its recorded case ids + modes. Backups (_*), SAMPLE padding, and error-only
    models are hidden. Sorted by coverage (most cases first)."""
    refined = refined_case_ids()                         # UI shows only the refined cases
    by = {}                                              # slug -> {cases:{cid:mode}, designs:set, ok:bool}
    for model, design, fp, cid, run in iter_runs():
        if model.startswith("_") or cid.startswith("SAMPLE") or (refined and cid not in refined):
            continue
        m = by.setdefault(model, {"cases": {}, "designs": set(), "ok": False})
        m["cases"][cid] = run.get("mode") or "cpc"
        m["designs"].add(design)
        turns = run.get("turns") or []
        if turns and not any(t.get("error") for t in turns):
            m["ok"] = True
    out = []
    for slug, m in by.items():
        if not m["ok"]:                                  # skip models whose runs are all errors
            continue
        out.append({"slug": slug, "n_cases": len(m["cases"]), "designs": sorted(m["designs"]),
                    "cases": sorted(m["cases"]), "modes": m["cases"]})
    out.sort(key=lambda x: -x["n_cases"])
    return jsonify({"models": out})


@app.get("/api/case")
def api_case():
    """What the model is GIVEN for a case: the turn-0 presentation (history + exam), plus the
    full vignette for reviewers. Independent of recordings, so the UI can show it immediately."""
    cid = request.args.get("case")
    try:
        case, mode = load_any(cid)
    except Exception:
        return jsonify({"error": f"unknown case {cid}"}), 404
    presentation = (getattr(case, "presentation", "") or "").strip() or case.abstract
    return jsonify({"case_id": cid, "mode": mode, "abstract": case.abstract,
                    "presentation": presentation, "full_case": full_vignette(case)})


@app.get("/api/recording")
def api_recording():
    """Return a pre-recorded agent run for (case, design[, model]) — static replay, no live calls."""
    cid = request.args.get("case")
    design = request.args.get("design", "single")
    model = request.args.get("model") or DEFAULT_MODEL
    rec = None
    nf = run_file(model, design)                  # data/recordings/<model>/<design>@<fp>.json
    if os.path.exists(nf):
        try:
            rec = json.load(open(nf)).get(cid)
        except Exception:
            rec = None
    if not rec:
        return jsonify({"error": f"no recording for {cid} / {design}"}), 404
    return jsonify(rec)


def _auto_noharm(appr, harm):
    """Server-side NOHARM auto-score (1-9) — stored for human-vs-auto agreement, never shown."""
    if appr >= 0.6:
        return 9 if appr >= 0.85 else 8 if appr >= 0.70 else 7
    if appr <= 0.4:
        return 1 if harm >= 0.70 else 2 if harm >= 0.45 else 3
    return 6 if appr >= 0.55 else 4 if appr <= 0.45 else 5


def _score_features(case_id, test):
    """(appropriateness, harm, cost, auto_noharm) for a test in a case — or Nones if unscorable."""
    try:
        case, _ = load_any(case_id)
        s = score_recommendation(case, test)
        return round(s.appropriateness, 3), round(s.harm, 3), s.cost, _auto_noharm(s.appropriateness, s.harm)
    except Exception:
        return None, None, None, None


@app.post("/api/consent")
def api_consent():
    """Record that this participant consented. Required before any task data is meaningful."""
    pid = _pid()
    if not pid:
        return jsonify({"error": "no participant"}), 400
    store.set_consent(pid)
    return jsonify({"ok": True, "pid": pid})


@app.post("/api/grade")
def api_grade():
    """Persist one clinician grade. The auto-scorer is computed and stored (for human-vs-auto
    agreement) but NEVER returned to the client, so it can't bias grading. Idempotent."""
    rec = request.json or {}
    appr, harm, cost, auto = _score_features(rec.get("case_id"), rec.get("test"))
    try:
        store.log_grade(client_event_id=rec.get("event_id"), **_attribution(),
                        case_id=rec.get("case_id"), design=rec.get("design"), test=rec.get("test"),
                        grade=rec.get("grade"), appropriateness=rec.get("appropriateness"),
                        harm=rec.get("harm"), latency_ms=rec.get("latency_ms"),
                        auto_appropriateness=appr, auto_harm=harm, auto_cost=cost, auto_noharm=auto)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


@app.post("/api/sim_event")
def api_sim_event():
    """Log one Simulation alert decision with the RL state/action/features: recommendation
    appropriateness/harm/cost, clinician action, latency, time-in-shift, the running
    alert-fatigue (interruption count) and sequence index. Idempotent."""
    rec = request.json or {}
    appr, harm, cost, auto = _score_features(rec.get("case_id"), rec.get("test"))
    try:
        store.log_sim(client_event_id=rec.get("event_id"), **_attribution(),
                      case_id=rec.get("case_id"), test=rec.get("test"), choice=rec.get("choice"),
                      latency_ms=rec.get("latency_ms"), shift_min=rec.get("shift_min"),
                      seq=rec.get("seq"), fatigue=rec.get("fatigue"),
                      appropriateness=appr, harm=harm, cost=cost, auto_noharm=auto)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


@app.post("/api/clin_event")
def api_clin_event():
    """Log one clinician benchmark turn: action, decision-time, running cost, outcome. Idempotent."""
    rec = request.json or {}
    try:
        store.log_bench(client_event_id=rec.get("event_id"), **_attribution(),
                        case_id=rec.get("case_id"), action=rec.get("action"), query=rec.get("query"),
                        latency_ms=rec.get("latency_ms"), turn=rec.get("turn"),
                        total_cost=rec.get("total_cost"), correct=rec.get("correct"),
                        judge_score=rec.get("judge_score"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return jsonify({"ok": True})


ADMIN_TOKEN = os.environ.get("STUDY_ADMIN_TOKEN")


def _admin_ok():
    """Researcher-only endpoints. If an admin token is set (e.g. when exposed via a tunnel),
    REQUIRE it — don't trust apparent-localhost, since a tunnel makes every request look local.
    With no token set, fall back to localhost-only (fine for local dev)."""
    if ADMIN_TOKEN:
        return (request.args.get("token") == ADMIN_TOKEN
                or request.headers.get("X-Admin-Token") == ADMIN_TOKEN)
    return request.remote_addr in ("127.0.0.1", "::1", None)


_ABBR = {"single": "LLM", "maidxo": "LLM-Roles", "debate": "LLM-Multi"}


@app.get("/api/stats")
def api_stats():
    """Benchmark points for the Pareto plot: one point per (model x agent design), aggregated
    over cases, split by dataset mode (cpc / ehr / all). Scans every per-model recording file."""
    refined = refined_case_ids()                         # Results reflect only the refined cases
    by = {}                                              # (model, design, mode) -> rows
    for model, d, fp, cid, run in iter_runs():
        if cid.startswith("SAMPLE") or model.startswith("_"):   # skip demo padding + backup/error dumps
            continue
        if d not in DESIGNS or (refined and cid not in refined):
            continue
        turns = run.get("turns") or []
        dx = next((t for t in turns if t.get("action") == "diagnose"), None)
        if not dx:
            continue
        last = turns[-1]
        meta = run.get("meta") or {}
        conc = meta.get("concordance") or {}
        by.setdefault((model, d, run.get("mode", "ehr")), []).append(
            {"correct": bool(dx.get("correct")), "score": dx.get("judge_score"),
             "cost": last.get("total_cost"), "turns": len(turns), "tests": last.get("n_tests"),
             "tokens": meta.get("total_tokens"), "secs": meta.get("duration_s"),
             "recall": conc.get("recall"), "precision": conc.get("precision")})

    def summ(rows):
        n = len(rows)
        if not n:
            return {"n": 0, "accuracy": 0, "accuracy_ci": 0, "judge": 0}

        def stat(k, dp):                                # mean + standard error over present values
            vals = [r[k] for r in rows if r.get(k) is not None]
            if not vals:
                return None, None
            m = sum(vals) / len(vals)
            sd = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
            return round(m, dp), round(sd / len(vals) ** 0.5, max(dp, 1))

        p = sum(1 for r in rows if r["correct"]) / n
        out = {"n": n, "accuracy": round(100 * p),
               "accuracy_ci": round(100 * 1.96 * (p * (1 - p) / n) ** 0.5),   # 95% CI half-width (pts)
               "judge": stat("score", 2)[0]}
        for k, dp in (("cost", 0), ("turns", 1), ("tests", 1), ("tokens", 0), ("secs", 1),
                      ("recall", 3), ("precision", 3)):
            out[k], out[k + "_sem"] = stat(k, dp)
        return out

    md = {}                                              # (model, design) -> {mode: rows}
    for (model, d, mode), rows in by.items():
        md.setdefault((model, d), {})[mode] = rows
    modes = {"cpc": [], "mimic": [], "all": []}
    for (model, d), mm in sorted(md.items()):
        allrows = [r for rows in mm.values() for r in rows]
        for mode, rows in (("cpc", mm.get("cpc", [])), ("mimic", mm.get("mimic", [])), ("all", allrows)):
            if not rows:
                continue
            ms = model.replace("-instruct", "")
            for pref in ("meta-llama-", "openai-", "google-", "anthropic-", "mistralai-", "qwen-", "deepseek-"):
                if ms.startswith(pref):
                    ms = ms[len(pref):]
            modes[mode].append({"model": model, "model_short": ms[:18],
                                "design": d, "label": DESIGNS.get(d, (d,))[0], "abbr": _ABBR.get(d, d),
                                **summ(rows)})
    return jsonify({"modes": modes, "n_models": len({m for m, _ in md})})


_SPECIALTY = [
    ("Infectious", r"infect|sepsis|abscess|meningitis|encephalitis|tuberculos|pneumon|endocarditis|"
                   r"virus|viral|bacteri|fungal|cryptococc|malaria|\bhiv\b|hepatitis|cellulitis"),
    ("Onc / Heme", r"lymphoma|leukemia|carcinoma|sarcoma|cancer|tumor|myeloma|metasta|neoplas|melanoma|"
                   r"blastoma|malignan|histiocytosis|lymphoprolifer"),
    ("Cardiovascular", r"cardi|myocard|\bheart\b|aortic|vascul|embol|infarct|arrhythm|valv|pericard"),
    ("Neurology", r"encephal|neuro|stroke|seizure|sclerosis|myelo|cerebr|spinal|parkinson|dementia|angiitis"),
    ("Renal", r"renal|nephr|kidney|glomerul"),
    ("GI / Hepatic", r"hepat|liver|pancreat|colitis|bowel|gastr|crohn|biliary|cirrhos|esophag"),
    ("Pulmonary", r"pulmonar|\blung\b|pleural|respiratory|asthma|copd|sarcoidosis"),
    ("Endocrine", r"diabet|thyroid|adrenal|pituitary|endocr|prolactin|ketoacidosis"),
    ("Rheum / Immune", r"vasculitis|lupus|arthritis|autoimmune|amyloid|rheum|immune|granulomato"),
]
_COSTCAT = [
    ("Imaging", r"\bct\b|\bmri\b|x[\s-]?ray|radiograph|ultrasound|echo|angiogra|\bpet\b|scan|mammogram"),
    ("Procedure", r"biopsy|bronchoscopy|endoscopy|colonoscopy|lumbar puncture|aspirat|marrow|\blp\b"),
    ("Genetic/Path", r"flow cytometry|cytogenet|genetic|\bpcr\b|sequenc|immunohist|patholog|molecular|\bfish\b"),
    ("Micro/Serol", r"culture|gram stain|serolog|antibody|antigen|\btiter\b|microbiolog"),
]


def _specialty(dx):
    d = (dx or "").lower()
    for name, pat in _SPECIALTY:
        if re.search(pat, d):
            return name
    return "Other"


def _costcat(name):
    n = (name or "").lower()
    for cat, pat in _COSTCAT:
        if re.search(pat, n):
            return cat
    return "Labs"


@app.get("/api/analysis")
def api_analysis():
    """Extra SDBench-style breakdowns from the recorded benchmark (CPC, one model):
    error-type distribution (Judge 1-5), cost composition by test category, accuracy by specialty.
    Defaults to the model with the most CPC runs; override with ?model=<slug>."""
    rows = [(m, d, cid, run) for m, d, fp, cid, run in iter_runs()
            if run.get("mode") == "cpc" and not cid.startswith("SAMPLE")]
    target = request.args.get("model")
    if not target and rows:                          # pick the model with the most CPC runs
        from collections import Counter
        target = Counter(m for m, *_ in rows).most_common(1)[0][0]
    rows = [r for r in rows if r[0] == target]
    agg = {}
    for _m, d, cid, run in rows:
        ts = run.get("turns") or []
        dx = next((t for t in ts if t.get("action") == "diagnose"), None)
        if not dx:
            continue
        D = agg.setdefault(d, {"n": 0, "err": [0, 0, 0, 0, 0], "spec": {}, "cost": {}})
        D["n"] += 1
        js = min(5, max(1, dx.get("judge_score") or 1))
        D["err"][js - 1] += 1
        sp = _specialty(run.get("true_dx"))
        s = D["spec"].setdefault(sp, [0, 0]); s[1] += 1; s[0] += 1 if dx.get("correct") else 0
        for t in ts:
            if t.get("action") == "ask" and (t.get("cost") or 0) > 0:
                D["cost"]["Visits"] = D["cost"].get("Visits", 0) + (t.get("cost") or 0)
            if t.get("action") == "order":
                for o in (t.get("orders") or []):
                    c = _costcat(o.get("name", ""))
                    D["cost"][c] = D["cost"].get(c, 0) + (o.get("cost") or 0)

    out = {"designs": [], "model": target}
    for d in DESIGNS:                                    # the three current designs, in order
        if d not in agg:
            continue
        D = agg[d]; n = D["n"] or 1
        out["designs"].append({
            "id": d, "label": DESIGNS[d][0], "n": D["n"],
            "errors": [round(100 * x / n) for x in D["err"]],
            "specialty": [{"name": k, "acc": round(100 * v[0] / v[1]) if v[1] else 0, "n": v[1]}
                          for k, v in sorted(D["spec"].items(), key=lambda kv: -kv[1][1])],
            "cost": {k: round(v / n) for k, v in sorted(D["cost"].items(), key=lambda kv: -kv[1])},
        })
    return jsonify(out)


@app.get("/api/budget")
def api_budget():
    """Cost-budget sweep -> per (model, design, cap): accuracy and avg actual spend (the SDBench curve)."""
    path = os.path.join(BENCH_RESULTS, "budget_sweep.json")
    try:
        data = json.load(open(path))
    except Exception:
        return jsonify({"models": []})
    out = {"models": []}
    for model, caps in data.items():
        designs = {}
        for cap, dd in caps.items():
            for d, cases in dd.items():
                rows = []
                for cid, run in cases.items():
                    ts = (run or {}).get("turns") or []
                    dx = next((t for t in ts if t.get("action") == "diagnose"), None)
                    if not dx:
                        continue
                    rows.append((bool(dx.get("correct")), ts[-1].get("total_cost") or 0))
                if not rows:
                    continue
                n = len(rows)
                designs.setdefault(d, []).append(
                    {"cap": cap, "n": n,
                     "accuracy": round(100 * sum(1 for c, _ in rows if c) / n),
                     "cost": round(sum(c for _, c in rows) / n)})
        for d in designs:
            designs[d].sort(key=lambda p: p["cost"])
        if designs:
            out["models"].append({"model": model, "model_short": model.replace("-instruct", "")[:18],
                                   "designs": [{"id": d, "label": DESIGNS.get(d, (d,))[0], "points": p}
                                               for d, p in designs.items()]})
    return jsonify(out)


@app.get("/api/ablation")
def api_ablation():
    """Panel role ablation -> per removed role: accuracy and delta vs the full panel."""
    path = os.path.join(BENCH_RESULTS, "ablation.json")
    try:
        data = json.load(open(path))
    except Exception:
        return jsonify({"models": []})
    out = {"models": []}
    for model, variants in data.items():
        rows = []
        for key, v in variants.items():
            accs = []
            for cid, r in (v.get("runs") or {}).items():
                ts = (r or {}).get("turns") or []
                dx = next((t for t in ts if t.get("action") == "diagnose"), None)
                if dx:
                    accs.append(bool(dx.get("correct")))
            if not accs:
                continue
            rows.append({"key": key, "label": v.get("label", key), "n": len(accs),
                         "accuracy": round(100 * sum(accs) / len(accs))})
        if not rows:
            continue
        base = next((r["accuracy"] for r in rows if r["key"] == "full"), 0)
        for r in rows:
            r["delta"] = r["accuracy"] - base
        rows = ([r for r in rows if r["key"] == "full"] +
                sorted([r for r in rows if r["key"] != "full"], key=lambda r: r["accuracy"]))
        out["models"].append({"model": model, "model_short": model.replace("-instruct", "")[:18],
                              "base": base, "variants": rows})
    return jsonify(out)


@app.get("/api/counts")
def api_counts():
    """Live row counts per table — for verifying data is being captured during a session."""
    if not _admin_ok():
        return jsonify({"error": "unauthorized"}), 401
    return jsonify(store.counts())


@app.get("/api/export")
def api_export():
    """Download a table as CSV (grade | sim_event | bench_turn | participant | session)."""
    if not _admin_ok():
        return jsonify({"error": "unauthorized"}), 401
    table = request.args.get("table", "grade")
    try:
        csv_text = store.export_csv(table)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    from flask import Response
    return Response(csv_text, mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={table}.csv"})


@app.post("/api/new")
def api_new():
    case_id = request.json["case_id"]
    gatekeeper = request.json.get("gatekeeper", "auto")
    case, mode = load_any(case_id)
    env = _make_env(case, gatekeeper)
    SESSIONS[_sid()] = {"case_id": case_id, "mode": mode, "env": env,
                        "design": request.json.get("design", "single"),
                        "model": request.json.get("model") or DEFAULT_MODEL,
                        "gatekeeper": gatekeeper}
    return jsonify({"abstract": case.abstract, "true_dx": case.true_diagnosis,
                    "full_case": full_vignette(case),
                    "gatekeeper": "llm",
                    "mode": mode, "orderable": env.orderable_tests(),
                    "case_id": case_id})


@app.post("/api/manual")
def api_manual():
    st = _state()
    if not st:
        return jsonify({"error": "no session; pick a case"}), 400
    env = st["env"]
    if env.done:
        return jsonify({"done": True, "noop": True})
    action = request.json["action"]            # ask | order | diagnose
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    try:
        return jsonify(apply_turn(env, action, query, model=st.get("model")))
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5050"))          # 5000 collides with macOS AirPlay Receiver
    prod = os.environ.get("STUDY_PROD", "").lower() in ("1", "true", "yes")
    print(f"\n  LLM backend: {llm.active_model()}  (live={llm.detect_backend() != 'stub'})")
    print(f"  open http://127.0.0.1:{port}\n")
    if prod:                                             # study mode: stable server, no debugger page
        if app.secret_key == "demo-not-secret":
            raise SystemExit("Refusing to run prod with the default secret. Set STUDY_SECRET=<random>.")
        host = os.environ.get("STUDY_BIND", "127.0.0.1")  # localhost by default; opt in to expose
        if host != "127.0.0.1":
            print(f"  ⚠ binding {host} — put HTTPS + auth (reverse proxy) in front; set STUDY_ADMIN_TOKEN.")
        from waitress import serve as _serve
        print("  production server (waitress, debug off) — STUDY_PROD=1\n")
        _serve(app, host=host, port=port, threads=8)
    else:
        app.run(debug=True, port=port, use_reloader=False)
