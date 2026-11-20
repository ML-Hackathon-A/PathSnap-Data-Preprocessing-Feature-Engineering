"""Plots + paired comparison for the evaluator_test results (results/evaluator_test/*.csv).

    python -m src.plot_holdout
"""
import json

import matplotlib
import numpy as np
import pandas as pd

from .config import PLOTS_DIR, RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = RESULTS_DIR / "evaluator_test"


def main():
    S = pd.read_csv(D / "summary.csv")
    T = pd.read_csv(D / "per_trip.csv", dtype={"trip_id": str})
    S = S.sort_values("top1_accuracy", ascending=False)
    S["rank_top1"] = np.arange(1, len(S) + 1)
    S.to_csv(D / "ranking.csv", index=False)

    x = np.arange(len(S))
    fig, ax = plt.subplots(figsize=(9, 4))
    b1 = ax.bar(x - 0.2, S.top1_accuracy, 0.4, label="classifier only (before HMM)")
    b2 = ax.bar(x + 0.2, S.hmm_top1_accuracy, 0.4, label="classifier + HMM/Viterbi (after)")
    ax.bar_label(b1, fmt="%.3f", fontsize=7)
    ax.bar_label(b2, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, S.run_id)
    ax.set_ylim(0.85, 0.94)
    ax.set_title("Evaluator-test Top-1 accuracy per model: before vs after HMM decoding")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eval_top1_before_after_hmm.png", dpi=130)
    plt.close()

    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, (c, t) in zip(axs, [("f1_macro", "Macro F1"), ("log_loss", "Log loss (lower = better)"),
                               ("hmm_route_continuity", "Route continuity after HMM")]):
        bars = ax.bar(S.run_id, S[c])
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
        lo, hi = S[c].min(), S[c].max()
        ax.set_ylim(max(0, lo - (hi - lo) * 1.5 - 0.005), hi + (hi - lo) * 0.6 + 0.002)
        ax.set_title(f"Evaluator-test: {t}", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eval_f1_logloss_cont.png", dpi=130)
    plt.close()

    piv = T.pivot(index="trip_id", columns="run_id", values="clf_top1_accuracy")
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for r in S.run_id:
        ax.plot(piv.index, piv[r], marker="o", label=r)
    ax.set_title("Evaluator-test Top-1 per trip (classifier only)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(ncol=6, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eval_per_trip_top1.png", dpi=130)
    plt.close()

    # paired comparison of the two best models across trips (is the gap bigger than trip-to-trip noise?)
    a, b = S.run_id.iloc[0], S.run_id.iloc[1]
    d = (piv[a] - piv[b]).dropna()
    rng = np.random.default_rng(42)
    boots = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
    paired = {"model_a": a, "model_b": b, "trips_a_better": int((d > 0).sum()), "trips_b_better": int((d < 0).sum()),
              "mean_per_trip_diff": float(d.mean()), "ci95_low": float(np.percentile(boots, 2.5)),
              "ci95_high": float(np.percentile(boots, 97.5)),
              "per_trip_top1_std": {r: float(piv[r].std()) for r in S.run_id}}
    json.dump(paired, open(D / "paired_top2.json", "w"), indent=2)
    print(S[["rank_top1", "run_id", "top1_accuracy", "hmm_top1_accuracy", "f1_macro", "log_loss"]].round(4).to_string(index=False))
    print(paired)


if __name__ == "__main__":
    main()
