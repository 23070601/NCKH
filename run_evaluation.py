#!/usr/bin/env python3
"""
Evaluate prediction accuracy and calibration.
"""

import json
from pathlib import Path
import importlib.util
import os
import sys
import numpy as np

# Load evaluation module directly to avoid heavy package imports
eval_path = os.path.join(os.path.dirname(__file__), "src", "utils", "evaluate_predictions.py")
spec = importlib.util.spec_from_file_location("evaluate_predictions", eval_path)
evaluate_predictions = importlib.util.module_from_spec(spec)
sys.modules["evaluate_predictions"] = evaluate_predictions
spec.loader.exec_module(evaluate_predictions)

classification_report_text = evaluate_predictions.classification_report_text
evaluate = evaluate_predictions.evaluate
load_latest_predictions = evaluate_predictions.load_latest_predictions
plot_calibration = evaluate_predictions.plot_calibration
plot_confusion_matrix = evaluate_predictions.plot_confusion_matrix


def main() -> None:
    analysis_dir = Path("data/analysis")
    output_dir = analysis_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_latest_predictions(analysis_dir)
    results = evaluate(df)

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save classification report if available
    if "Risk_Label" in df.columns and "Predicted_Risk" in df.columns:
        report = classification_report_text(df)
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write(report)

        cm = np.array(results["risk_classification"]["confusion_matrix"])
        plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
        plot_calibration(df, output_dir / "calibration.png")

    print(f"✓ Evaluation complete. Results in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
