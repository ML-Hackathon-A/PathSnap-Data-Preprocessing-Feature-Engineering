"""Copy the chosen run's model to artifacts/final_model.joblib and write a manifest.

    python -m src.finalize --run-id lgbm_tuned
"""
import argparse
import json
import platform
import shutil

import sklearn

from .config import ARTIFACT_DIR, MODEL_DIR, RESULTS_DIR
from .features import FEATURES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    a = ap.parse_args()
    shutil.copy(MODEL_DIR / f"{a.run_id}.joblib", ARTIFACT_DIR / "final_model.joblib")
    result = json.load(open(RESULTS_DIR / f"result_{a.run_id}.json"))
    manifest = {"run_id": a.run_id, "model": result["model"], "params": result["params"],
                "features": FEATURES, "sklearn_version": sklearn.__version__, "python": platform.python_version()}
    json.dump(manifest, open(ARTIFACT_DIR / "final_manifest.json", "w"), indent=2)
    hmm_cfg = ARTIFACT_DIR / "hmm_config.json"
    if not hmm_cfg.exists():
        json.dump({"beta": 5.0, "emission_weight": 1.0, "selected_on": "default"}, open(hmm_cfg, "w"), indent=2)
    print("final model ->", ARTIFACT_DIR / "final_model.joblib")
