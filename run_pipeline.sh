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

echo "📍 STEP 1: Training Models"
echo "────────────────────────────────────────────────────────────────────"
python train_models.py $FORCE_FLAG
echo ""

echo "📍 STEP 2: Creating Base Predictions"
echo "────────────────────────────────────────────────────────────────────"
python create_base_predictions.py $FORCE_FLAG
echo ""

echo "📍 STEP 3: Generating Improved Predictions"
echo "────────────────────────────────────────────────────────────────────"
python generate_predictions.py $FORCE_FLAG
echo ""

echo "📍 STEP 4: Evaluating Results"
echo "────────────────────────────────────────────────────────────────────"
python evaluate_models.py $FORCE_FLAG
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
echo "   • Models:      data/analysis/quick_improvement/"
echo "   • Predictions: data/analysis/predictions_improved_lag_*.csv"
echo "   • Evaluation:  data/analysis/evaluation_improved_lag/"
echo ""
echo "💡 Usage:"
echo "   • Cached run:  ./run_pipeline.sh"
echo "   • Full rerun:  ./run_pipeline.sh --force"
echo ""
