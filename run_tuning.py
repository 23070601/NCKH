#!/usr/bin/env python3
"""
Run model tuning for regression and classification.
"""

import json
from pathlib import Path
import importlib.util
import os
import sys

# Load model_tuning module directly to avoid heavy package imports
module_path = os.path.join(os.path.dirname(__file__), "src", "utils", "model_tuning.py")
spec = importlib.util.spec_from_file_location("model_tuning", module_path)
model_tuning = importlib.util.module_from_spec(spec)
sys.modules["model_tuning"] = model_tuning
spec.loader.exec_module(model_tuning)


def main() -> None:
    analysis_dir = Path("data/analysis")
    output_dir = analysis_dir / "tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = model_tuning.TuningConfig()
    df = model_tuning.load_latest_predictions(analysis_dir)

    reg_results, rf_reg, hgb_reg = model_tuning.tune_regression(df, cfg)
    cls_results, rf_cls, hgb_cls = model_tuning.tune_classification(df, cfg)

    results = {
        "regression": reg_results,
        "classification": cls_results,
    }

    with open(output_dir / "tuning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save best models
    import joblib
    joblib.dump(rf_reg, output_dir / "best_rf_reg.pkl")
    joblib.dump(hgb_reg, output_dir / "best_hgb_reg.pkl")
    if rf_cls is not None:
        joblib.dump(rf_cls, output_dir / "best_rf_cls.pkl")
    if hgb_cls is not None:
        joblib.dump(hgb_cls, output_dir / "best_hgb_cls.pkl")

    print(f"✓ Tuning complete. Results in {output_dir.resolve()}")


if __name__ == "__main__":
    main()
