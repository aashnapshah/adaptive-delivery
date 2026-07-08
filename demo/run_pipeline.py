"""Run the WHOLE benchmark end-to-end for a small N and refresh everything the UI shows.

    python run_pipeline.py [N] [--model qwen3:32b] [--designs single,maidxo] [--sweeps] [--no-mimic]

Steps, in order (all local, all resumable — re-running continues where it left off):
    1. CPC benchmark     (N cases × designs)
    2. MIMIC benchmark   (N cases × designs)          [skip with --no-mimic]
    3. budget sweep      (N cases)                    [only with --sweeps]
    4. role ablation     (N cases)                    [only with --sweeps]
    5. export results    -> results/ CSVs + figures

The Flask app reads the recordings and figures live, so the Results tab updates AS this runs
(auto-refreshes every 30 s) and is fully current once it finishes. Meant as a quick smoke test of
the entire pipeline. NOTE: don't run this at the same time as a big background benchmark on the same
model — they'd share recording files and the GPU. Stop that first.

Examples:
    python run_pipeline.py 5 --sweeps          # 5 cases through EVERYTHING (CPC+MIMIC+budget+ablation)
    python run_pipeline.py 5 --no-mimic        # 5 CPC cases only, no sweeps
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

args = sys.argv[1:]
N = next((a for a in args if a.isdigit()), "5")
model = args[args.index("--model") + 1] if "--model" in args else "qwen3:32b"
designs = args[args.index("--designs") + 1] if "--designs" in args else "single,maidxo"
do_sweeps = "--sweeps" in args
do_mimic = "--no-mimic" not in args

base = {**os.environ, "OLLAMA_HOST": os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"),
        "DEMO_LLM_BACKEND": "ollama", "OLLAMA_MODEL": model}
bench = {**base, "BENCHMARK_MODELS": model, "BENCHMARK_DESIGNS": designs, "BENCHMARK_MAX_CASES": N}


def step(title, cmd, env):
    print(f"\n{'='*70}\n▶ {title}\n{'='*70}", flush=True)
    subprocess.run([sys.executable, *cmd], cwd=HERE, env=env, check=False)


def main():
    print(f"end-to-end pipeline · {N} cases · model={model} · designs={designs} · "
          f"mimic={do_mimic} · sweeps={do_sweeps}")
    step("[1] CPC benchmark", ["record.py"], {**bench, "BENCHMARK_MODE": "cpc"})
    if do_mimic:
        step("[2] MIMIC benchmark", ["record.py"], {**bench, "BENCHMARK_MODE": "mimic"})
    if do_sweeps:
        sweep_n = str(min(int(N), 25))    # sweeps are secondary trends — cap cases so big runs stay tractable
        step(f"[3] budget sweep ({sweep_n} cases)", ["record_budget.py"],
             {**base, "BUDGET_MODEL": model, "BUDGET_DESIGNS": designs, "BUDGET_CASES": sweep_n})
        step(f"[4] role ablation ({sweep_n} cases)", ["record_ablation.py"],
             {**base, "ABLATION_MODEL": model, "ABLATION_CASES": sweep_n})
    step("[5] export results (CSVs + figures)", ["export_results.py"], base)
    print(f"\n✅ done — {N} cases through the full pipeline. Open/refresh the Results tab; "
          f"tables + figures are in results/.")


if __name__ == "__main__":
    main()
