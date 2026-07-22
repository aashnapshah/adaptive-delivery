"""Export saved conversations to a readable folder: results/processed/chats/<source>/<design>-<case_id>.md

    python harness/report/chats.py                   # export every saved run
    python harness/report/chats.py --design strict-roles    # just one design
    python harness/report/chats.py --source cpc --limit 10

Reads results/raw/runs.jsonl (which holds the full recommender <-> gatekeeper conversation per run) and
writes one markdown file per run plus an index. Nothing is re-run and no LLM is called - this is purely
a readable view over what generate.py already saved.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from harness.store.transcripts import as_bool, load_prompts, load_runs, load_scores, select_runs   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHATDIR = os.path.join(ROOT, "results", "processed", "chats")


def _facts(run, scores):
    """The header table rows for one run: what it was, and how it scored (from its runs.csv score columns)."""
    cfg, meta = run["config"], run["meta"]
    s = scores.get(run["run_id"], {})
    rows = [("Case", run["case_id"]), ("Source", run["source"]), ("Design", cfg["design"]),
            ("Model", cfg["model"]), ("Prompt", cfg["prompt_fp"]),
            ("Turns", meta["turns"]), ("Tests ordered", len(run["ordered"])),
            ("Deliberation", f"{meta['deliberate_s']}s"), ("Recommender tokens", meta["tokens"]),
            ("Stopped on its own", run.get("stopped", ""))]
    if s.get("concordance_recall"):
        rows.append(("Concordance (recall)", s["concordance_recall"]))
    if s.get("cost_total"):
        rows.append(("Work-up cost", f"${s['cost_total']}"))
    if s.get("workup_better"):
        rows.append(("Better work-up", s["workup_better"]))
    return rows


def render(run, scores, prompts):
    """One run -> a readable markdown page: header facts, the prompts, the chat, then the outcome."""
    cfg = run["config"]
    out = [f"# {run['case_id']} — {cfg['design']}", ""]
    out += ["| | |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in _facts(run, scores)]

    # the exact prompts behind this run, recovered by fingerprint (collapsed so the chat stays readable)
    texts = prompts.get(cfg["prompt_fp"], {})
    if texts:
        out += ["", "## Prompts", "", f"_Fingerprint `{cfg['prompt_fp']}` - the exact text that "
                "produced this run._", ""]
        for name, text in texts.items():
            out += [f"<details><summary><code>{name}</code></summary>", "",
                    "```", text.strip(), "```", "", "</details>", ""]

    out += ["", "## Conversation", ""]
    if run.get("conversation"):
        for speaker, msg in run["conversation"]:
            label = "GATEKEEPER" if speaker == "gatekeeper" else "RECOMMENDER"
            out += [f"**{label}**", "", "```", msg.strip(), "```", ""]
    else:
        # Runs saved before the conversation was persisted: rebuild what we can from the
        # test -> result transcript. The recommender's deliberation text is gone for these.
        out += ["_Full chat not saved for this run (predates conversation logging) - "
                "showing the ordered tests and what the gatekeeper returned._", ""]
        for name, res in run.get("transcript", []):
            out += [f"**RECOMMENDER** — ordered `{name}`", "", "```", str(res).strip(), "```", ""]

    out += ["## Tests ordered (in sequence)", ""]
    out += [f"{i}. {t}" for i, t in enumerate(run["ordered"], 1)] or ["_(none)_"]
    out += ["", "## Outcome", ""]
    if run.get("diagnosis"):
        out += [f"- **Diagnosis readout:** {run['diagnosis']}"]
    out += [f"- **True diagnosis:** {run.get('true_dx') or '(n/a)'}"]
    s = scores.get(run["run_id"], {})
    if s.get("judge_score"):
        out += [f"- **Judge:** {s['judge_score']}/5 — {'correct' if as_bool(s.get('judge_correct')) else 'incorrect'}"
                + (f" ({s['judge_reason']})" if s.get("judge_reason") else "")]
    return "\n".join(out) + "\n"


def export(runs, scores, prompts):
    """Write one markdown page per run under results/processed/chats/<source>/, plus an index. Returns the paths."""
    written = []
    for run in runs:
        d = os.path.join(CHATDIR, run["source"])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{run['config']['design']}-{run['case_id']}.md")
        with open(path, "w") as f:
            f.write(render(run, scores, prompts))
        written.append(path)

    lines = ["# Chats", "",
             "Readable exports of every saved recommender <-> gatekeeper conversation.",
             "Regenerate with `python harness/report/chats.py`.", ""]
    for source in sorted({r["source"] for r in runs}):
        lines += [f"## {source.upper()}", ""]
        for run in sorted((r for r in runs if r["source"] == source),
                          key=lambda r: (r["config"]["design"], r["case_id"])):
            rel = f"{source}/{run['config']['design']}-{run['case_id']}.md"
            lines.append(f"- [{run['case_id']} — {run['config']['design']}]({rel}) "
                         f"· {len(run['ordered'])} tests · {run['meta']['turns']} turns")
        lines.append("")
    with open(os.path.join(CHATDIR, "README.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="")      # cpc | mimic; empty = all
    ap.add_argument("--designs", default="")     # comma-separated; empty = every design
    args = ap.parse_args()

    runs = [r for _, r in select_runs(load_runs(), args.source, args.designs)]
    if not runs:
        print("no matching runs - generate some first.")
        return
    os.makedirs(CHATDIR, exist_ok=True)
    written = export(runs, load_scores(), load_prompts())
    print(f"wrote {len(written)} chats to {CHATDIR}/  (index: results/processed/chats/README.md)")


if __name__ == "__main__":
    main()
