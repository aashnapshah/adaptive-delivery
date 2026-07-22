#!/usr/bin/env bash
# Run the whole harness pipeline on a small batch: extract -> generate -> evaluate -> plots.
#
#   ./run.sh                          # 5 cases per source, ALL prompts, prints the chats
#   ./run.sh 10 single,strict-roles   # 10 cases per source, just those two designs
#   SHOW=0 ./run.sh                   # don't print the conversations
#   FORCE=1 ./run.sh                  # re-generate cases already saved (else they're skipped as done)
#
# Designs already run on a case are skipped for free, so re-running this only generates what's missing.
#
# Generation (step 2) is the slow, token-expensive part - roughly 30-90s per case. Steps 3-4 read the
# saved transcripts, so they're cheap and safe to re-run as often as you like without touching Gemini
# for anything but the two judges.

set -eo pipefail
cd "$(dirname "$0")"

N=${1:-5}                                  # cases per source
DESIGNS=${2:-single,roles,strict-roles}    # comma-separated prompt files in recommender/prompts/
SHOW=${SHOW:-1}                            # 1 = print each recommender <-> gatekeeper conversation
FORCE=${FORCE:-0}                          # 1 = re-generate cells already in results/raw/
TAG=${TAG:-}                               # names this experiment; forks a new config_id
NOTE=${NOTE:-}                             # what changed vs the last experiment

gen_flags=""
[ "$SHOW" = "1" ] && gen_flags="$gen_flags --show"
[ "$FORCE" = "1" ] && gen_flags="$gen_flags --force"
[ -n "$TAG" ] && gen_flags="$gen_flags --tag $TAG"


echo "==> pipeline: $N cases/source | designs=$DESIGNS | show=$SHOW force=$FORCE"

echo
echo "==> 1/5  extract CPC human work-ups (cached + resumable; gives CPC its ordering sequence)"
python -m harness.cases.build_cpc_workup --limit "$N"

echo
echo "==> 2/5  generate: recommender <-> gatekeeper, saving transcripts (no scoring here)"
python harness/generate.py --source cpc   --designs "$DESIGNS" --limit "$N" $gen_flags --note "$NOTE"
python harness/generate.py --source mimic --designs "$DESIGNS" --limit "$N" $gen_flags --note "$NOTE"

echo
echo "==> 3/5  score: LLM judges + deterministic metrics -> results/processed/scores.csv (skips already-scored)"
python -m harness.judge --designs "$DESIGNS"
python -m harness.evaluation.eval --designs "$DESIGNS"

echo
echo "==> 4/5  summary stats: every table (csv) + figure (pdf/png)"
python -m harness.report.summ_stats

echo
echo "==> 5/5  export the chats to a readable folder"
python -m harness.report.chats --designs "$DESIGNS"

echo
echo "done."
echo "  chats   : results/processed/chats/README.md  (one readable page per run, per source)"
echo "  results : results/processed/tables/*.csv  (python -m harness.report.summ_stats)"
echo "  figures : results/processed/figures/pdf/ + png/"
