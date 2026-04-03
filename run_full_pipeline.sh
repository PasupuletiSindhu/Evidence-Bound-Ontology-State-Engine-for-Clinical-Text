#!/usr/bin/env bash
# Run baselines → aligned QA → state engine → baseline QA → comparison → HTML/PNG report.
# Usage:
#   ./run_full_pipeline.sh
#   ./run_full_pipeline.sh --no-plots
#   ./run_full_pipeline.sh --baselines 1,2
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NO_PLOTS=""
BASELINES="1,2,3"
PARAPHRASE_JSON="baselines/data/paraphrases/paraphrase_sets_50.json"
QA_COUNT="100"
QA_ALIGNED_JSON="results/qa_aligned_100.json"
RELATION_TRAIN_JSON="results/relation_train.json"
RELATION_MAP_JSON="results/relation_map.json"
RELATION_MIN_TOP_RATIO="0.0"
REL_DIR="baselines/out/relation"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-plots) NO_PLOTS="--no_plots"; shift ;;
    --baselines)
      BASELINES="$2"
      shift 2
      ;;
    --paraphrases)
      PARAPHRASE_JSON="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--no-plots] [--baselines 1,2,3] [--paraphrases PATH] [--qa-count N] [--relation-train PATH] [--relation-map PATH] [--relation-min-top-ratio R]"
      echo "  Default paraphrases: ${PARAPHRASE_JSON}"
      echo "  Default qa-count: ${QA_COUNT}"
      exit 0
      ;;
    --qa-count)
      QA_COUNT="$2"
      QA_ALIGNED_JSON="results/qa_aligned_${QA_COUNT}.json"
      shift 2
      ;;
    --relation-train)
      RELATION_TRAIN_JSON="$2"
      shift 2
      ;;
    --relation-map)
      RELATION_MAP_JSON="$2"
      shift 2
      ;;
    --relation-min-top-ratio)
      RELATION_MIN_TOP_RATIO="$2"
      shift 2
      ;;
    --rel-dir)
      REL_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1 (try --help)"
      exit 1
      ;;
  esac
done

# Baselines 1 & 2 need a real relation checkpoint; otherwise run_graph_eval exits after a cryptic torch load error.
REL_CHECK_DIR="$ROOT/$REL_DIR"
_need_rel=false
case ",${BASELINES}," in
  *,1,*|*,2,*) _need_rel=true ;;
esac
if [[ "${_need_rel}" == true ]]; then
  if [[ ! -d "${REL_CHECK_DIR}" ]]; then
    echo "ERROR: You asked for Baselines 1 and/or 2, but the relation checkpoint directory is missing:"
    echo "       ${REL_CHECK_DIR}"
    echo "Train it from baselines/:"
    echo "  cd baselines && python3 train.py --dataset bc5cdr --model_name dmis-lab/biobert-base-cased-v1.1 --skip_ner_training"
    echo "Or run only the LLM baseline (no local NER/RE):"
    echo "  ./run_full_pipeline.sh --baselines 3"
    exit 1
  fi
  if ! find "${REL_CHECK_DIR}" -maxdepth 2 \( -name 'pytorch_model.bin' -o -name 'model.safetensors' \) -print -quit | grep -q .; then
    echo "ERROR: No pytorch_model.bin or model.safetensors under ${REL_CHECK_DIR} (including checkpoint-* subfolders)."
    echo "Re-train or copy the full relation output directory from your training machine."
    exit 1
  fi
fi

mkdir -p results/.matplotlib
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/results/.matplotlib}"

echo "==> [1/7] Paraphrase + baselines → results/paraphrase_results.json"
(
  cd "$ROOT/baselines"
  python3 run_graph_eval.py \
    --paraphrases_file "$ROOT/$PARAPHRASE_JSON" \
    --output "$ROOT/results/paraphrase_results.json" \
    --rel_dir "$ROOT/$REL_DIR" \
    --baselines "$BASELINES" \
    $NO_PLOTS
)

echo "==> [2/7] Build learned relation map → ${RELATION_MAP_JSON}"
if [[ -f "$ROOT/$RELATION_TRAIN_JSON" ]]; then
  python3 -m state_engine.build_relation_map \
    --input "$ROOT/$RELATION_TRAIN_JSON" \
    --output "$ROOT/$RELATION_MAP_JSON" \
    --min_top_ratio "$RELATION_MIN_TOP_RATIO"
else
  echo "Warning: relation training data not found at $ROOT/$RELATION_TRAIN_JSON; using empty map."
  mkdir -p "$(dirname "$ROOT/$RELATION_MAP_JSON")"
  printf '{}\n' > "$ROOT/$RELATION_MAP_JSON"
fi

echo "==> [3/7] Aligned QA → ${QA_ALIGNED_JSON}"
python3 -m state_engine.generate_aligned_qa \
  --paraphrase_sets "$PARAPHRASE_JSON" \
  --qa_count "$QA_COUNT" \
  --relation_map "$RELATION_MAP_JSON" \
  --output "$QA_ALIGNED_JSON"

echo "==> [4/7] State engine → results/state_engine_results.{json,md}"
python3 -m state_engine.run_state_engine \
  --paraphrase_results results/paraphrase_results.json \
  --qa_file "$QA_ALIGNED_JSON" \
  --relation_map "$RELATION_MAP_JSON" \
  --output_json results/state_engine_results.json \
  --output_md results/state_engine_results.md

echo "==> [5/7] Baseline QA (same questions) → results/qa_results.json"
(
  cd "$ROOT/baselines"
  python3 run_qa_eval.py \
    --paraphrase_results "$ROOT/results/paraphrase_results.json" \
    --qa_file "$ROOT/$QA_ALIGNED_JSON" \
    --output "$ROOT/results/qa_results.json"
)

echo "==> [6/7] Method comparison → results/method_comparison.{md,csv}"
python3 -m state_engine.compare_to_baselines \
  --paraphrase_results results/paraphrase_results.json \
  --qa_results results/qa_results.json \
  --state_engine_results results/state_engine_results.json \
  --out_md results/method_comparison.md \
  --out_csv results/method_comparison.csv

echo "==> [7/7] HTML report + sample graph PNG"
python3 -m state_engine.visualize_results \
  --json results/state_engine_results.json \
  --html_out results/state_engine_report.html \
  --graph_png results/state_engine_graph_set0.png \
  --set_id 0

echo ""
echo "Done. Key outputs:"
echo "  results/paraphrase_results.json"
echo "  ${QA_ALIGNED_JSON}"
echo "  ${RELATION_MAP_JSON}"
echo "  relation checkpoint: ${REL_DIR}"
echo "  results/state_engine_results.json"
echo "  results/state_engine_results.md"
echo "  results/qa_results.json"
echo "  results/method_comparison.md"
echo "  results/state_engine_report.html"
echo "  results/state_engine_graph_set0.png"
