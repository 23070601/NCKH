#!/bin/bash

# Vietnamese FDI Stock Volatility Prediction - Pipeline Runner
# Supports caching for fast re-runs, --force flag for full recomputation

set -e

FORCE_FLAG=""
if [[ "$1" == "--force" || "$1" == "-f" ]]; then
    FORCE_FLAG="--force"
    echo "🔄 FORCE MODE: Recomputing all steps"
    echo ""
fi

source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null

echo "════════════════════════════════════════════════════════════════════"
echo "🎯 NCKH PIPELINE RUNNER"
echo "════════════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)

echo "📍 STEP 1: Collect values.csv + adj.npy"
echo "────────────────────────────────────────────────────────────────────"
if [[ -f "data/raw/values.csv" && -f "data/raw/adj.npy" && -z "$FORCE_FLAG" ]]; then
    echo "✓ Found raw values + adjacency (skip)"
else
    python pipeline/01_collect_values.py
fi
echo ""

echo "📍 STEP 2: Export full feature dataset"
echo "────────────────────────────────────────────────────────────────────"
if [[ -f "data/features/all_features_raw.csv" && -f "data/features/all_features_processed.csv" && -z "$FORCE_FLAG" ]]; then
    echo "✓ Found full feature dataset (skip)"
else
    python pipeline/02b_export_full_features.py
fi
echo ""

echo "📍 STEP 3: Build timestep tensors"
echo "────────────────────────────────────────────────────────────────────"
if [[ -f "data/processed/timestep_0.pt" && -z "$FORCE_FLAG" ]]; then
    echo "✓ Found timestep tensors (skip)"
else
    python pipeline/03_build_tensors.py $FORCE_FLAG
fi
echo ""

echo "📍 STEP 4: Training Models"
echo "────────────────────────────────────────────────────────────────────"
python pipeline/04_train_models.py $FORCE_FLAG
echo ""

echo "📍 STEP 5: Creating Base Predictions"
echo "────────────────────────────────────────────────────────────────────"
python pipeline/05_base_predictions.py $FORCE_FLAG
echo ""

echo "📍 STEP 6: Generating Improved Predictions"
echo "────────────────────────────────────────────────────────────────────"
python pipeline/06_generate_predictions.py $FORCE_FLAG
echo ""

echo "📍 STEP 7: Evaluating Results"
echo "────────────────────────────────────────────────────────────────────"
python pipeline/07_evaluate.py $FORCE_FLAG
echo ""

echo "📍 STEP 8: Export CSV Tables"
echo "────────────────────────────────────────────────────────────────────"
python pipeline/08_export_tables.py --all
echo ""

echo "📍 STEP 9: Risk Metrics + Portfolio Optimization"
echo "────────────────────────────────────────────────────────────────────"
python pipeline/09_risk_portfolio.py
echo ""

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo "════════════════════════════════════════════════════════════════════"
echo "✅ PIPELINE COMPLETE"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "⏱️  Total time: ${DURATION}s"
echo ""
echo "📊 Results location:"
echo "   • Models:      data/results/models/"
echo "   • Predictions: data/results/predictions/predictions_improved_lag_*.csv"
echo "   • Evaluation:  data/results/evaluation/"
echo ""
echo "💡 Usage:"
echo "   • Cached run:  ./run_pipeline.sh"
echo "   • Full rerun:  ./run_pipeline.sh --force"
echo ""
