"""Collect results/result_*.json into the report's comparison table + comparison plots.

    python -m src.compare_models
"""
import json

import matplotlib
import numpy as np
import pandas as pd

from .config import PLOTS_DIR, RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    rows = []
    for f in sorted(RESULTS_DIR.glob("result_*.json")):
        r = json.load(open(f))
        st = r["student_test"]
        rows.append({
            "run_id": r["run_id"], "family": r["family"],
            "key_config": ", ".join(f"{k}={v}" for k, v in r["params"].items()),
            "cv_f1_macro": r["cv_f1_macro_mean"], "cv_f1_macro_std": r["cv_f1_macro_std"],
            "cv_top1": r["cv_top1_mean"], "cv_top1_labelled": r.get("cv_top1_labelled_mean"),
            "cv_train_f1_macro": r["cv_train_f1_macro_mean"],
            "test_top1": st["top1_accuracy"], "test_f1_macro": st["f1_macro"],
            "test_precision_macro": st["precision_macro"], "test_recall_macro": st["recall_macro"],
            "test_log_loss": st["log_loss"], "test_route_continuity": st["route_continuity"],
            "candidate_miss_rate": st["candidate_miss_rate"],
            "train_time_s": r["train_time_s"], "inference_ms_per_obs": r["inference_ms_per_obs"],
        })
    df = pd.DataFrame(rows).sort_values("test_top1", ascending=False)
    df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(df.drop(columns="key_config").round(4).to_string(index=False))

    for cols, title, fname in [
        (["cv_f1_macro", "cv_top1_labelled"], "Cross-validation (5-fold GroupKFold on train): macro F1 and Top-1 (labelled obs)", "cmp_cv_f1.png"),
        (["test_top1", "test_f1_macro"], "Student-test Top-1 accuracy and macro F1", "cmp_test_top1_f1.png"),
        (["test_log_loss"], "Student-test log loss (lower is better)", "cmp_logloss.png"),
        (["inference_ms_per_obs"], "Inference time per GPS observation (ms)", "cmp_inference.png"),
        (["test_route_continuity"], "Route continuity of classifier-only Top-1 path", "cmp_continuity.png"),
    ]:
        data = df.set_index("run_id")[cols]
        yerr = None
        if cols[0] == "cv_f1_macro":
            yerr = pd.DataFrame({"cv_f1_macro": df.set_index("run_id")["cv_f1_macro_std"], "cv_top1_labelled": 0.0})
        ax = data.plot.bar(figsize=(8, 4), rot=30, yerr=yerr, capsize=3)
        lo = float(np.nanmin(data.values))
        if fname not in ("cmp_inference.png", "cmp_logloss.png"):
            ax.set_ylim(max(0.0, lo - 0.08), 1.0)
        for c in ax.containers:
            if hasattr(c, "datavalues"):
                ax.bar_label(c, fmt="%.3f" if fname != "cmp_inference.png" else "%.4f", fontsize=7, padding=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / fname, dpi=130)
        plt.close()
    print("plots ->", PLOTS_DIR)


if __name__ == "__main__":
    main()
