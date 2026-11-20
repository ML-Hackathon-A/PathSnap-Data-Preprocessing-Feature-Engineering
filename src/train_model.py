"""Train + evaluate one model: group-aware CV on the 80 training trips, then refit and score student_test.

    python -m src.train_model --model lgbm
    python -m src.train_model --model lgbm --params-json artifacts/best_params_lgbm.json --tag tuned
"""
import argparse
import json
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import FEATURE_DIR, MODEL_DIR, N_FOLDS, PLOTS_DIR, RESULTS_DIR, SEED, max_train_rows
from .features import FEATURES
from .metrics import candidate_metrics, full_report, matching_metrics
from .models import DEFAULT_PARAMS, FAMILIES, make_model

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_features():
    tr = pd.read_pickle(FEATURE_DIR / "train.pkl")
    te = pd.read_pickle(FEATURE_DIR / "student_test.pkl")
    return tr, te


def X_of(df):
    return df[FEATURES].to_numpy(dtype=np.float32)


def train_rows(idx, df, cap, rng):
    """Training rows: only observations with a reliable label, optionally subsampled."""
    idx = idx[df["label_ok"].to_numpy()[idx] == 1]
    if cap and len(idx) > cap:
        idx = np.sort(rng.choice(idx, cap, replace=False))
    return idx


def cross_validate(name, params, tr, n_folds=N_FOLDS, verbose=True):
    rng = np.random.default_rng(SEED)
    X, y, groups = X_of(tr), tr["y"].to_numpy(), tr["trip_id"].to_numpy()
    cap = max_train_rows(name)
    oof = np.zeros(len(tr))
    folds = []
    for f, (a, b) in enumerate(GroupKFold(n_splits=n_folds).split(X, y, groups)):
        a = train_rows(a, tr, cap, rng)
        m = make_model(name, params)
        t0 = time.time()
        m.fit(X[a], y[a])
        fit_s = time.time() - t0
        oof[b] = m.predict_proba(X[b])[:, 1]
        val = tr.iloc[b]
        cm = candidate_metrics(val["y"].to_numpy(), oof[b])
        mm = matching_metrics(val, oof[b])
        tr_f1 = candidate_metrics(y[a], m.predict_proba(X[a])[:, 1])["f1_macro"]
        row = {"fold": f, "train_rows": len(a), "fit_s": fit_s, "train_f1_macro": tr_f1,
               "f1_macro": cm["f1_macro"], "log_loss": cm["log_loss"], "top1_accuracy": mm["top1_accuracy"],
               "top1_accuracy_labelled": mm["top1_accuracy_labelled"]}
        folds.append(row)
        if verbose:
            print(f"  fold {f}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                          for k, v in row.items()))
    return pd.DataFrame(folds), oof


def save_confusion(cm, title, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(name, params=None, tag="base"):
    run_id = name if tag == "base" else f"{name}_{tag}"
    params = {**DEFAULT_PARAMS[name], **(params or {})}
    tr, te = load_features()
    print(f"=== {run_id} | train rows={len(tr):,} test rows={len(te):,} | params={params}")

    folds, oof = cross_validate(name, params, tr)
    folds.to_csv(RESULTS_DIR / f"cv_folds_{run_id}.csv", index=False)

    rng = np.random.default_rng(SEED)
    idx = train_rows(np.arange(len(tr)), tr, max_train_rows(name), rng)
    model = make_model(name, params)
    t0 = time.time()
    model.fit(X_of(tr)[idx], tr["y"].to_numpy()[idx])
    train_s = time.time() - t0

    Xte = X_of(te)
    t0 = time.time()
    p = model.predict_proba(Xte)[:, 1]
    infer_s = time.time() - t0
    n_obs = te.groupby(["trip_id", "obs_idx"]).ngroups

    rep = full_report(te, p)
    result = {
        "run_id": run_id, "model": name, "family": FAMILIES[name], "tag": tag,
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "seed": SEED, "features": FEATURES, "train_rows_used": int(len(idx)),
        "cv_f1_macro_mean": float(folds["f1_macro"].mean()), "cv_f1_macro_std": float(folds["f1_macro"].std()),
        "cv_top1_mean": float(folds["top1_accuracy"].mean()),
        "cv_top1_labelled_mean": float(folds["top1_accuracy_labelled"].mean()),
        "cv_log_loss_mean": float(folds["log_loss"].mean()),
        "cv_train_f1_macro_mean": float(folds["train_f1_macro"].mean()),
        "train_time_s": train_s, "inference_time_s": infer_s,
        "inference_ms_per_obs": 1000 * infer_s / n_obs,
        "student_test": rep,
    }
    with open(RESULTS_DIR / f"result_{run_id}.json", "w") as f:
        json.dump(result, f, indent=2)
    joblib.dump(model, MODEL_DIR / f"{run_id}.joblib")
    pd.DataFrame({"trip_id": te["trip_id"], "obs_idx": te["obs_idx"], "arc_id": te["arc_id"], "p": p}) \
        .to_pickle(RESULTS_DIR / f"student_test_probs_{run_id}.pkl")
    save_confusion(rep["confusion_matrix"], f"{run_id} (student_test)", PLOTS_DIR / f"confusion_{run_id}.png")

    # Epoch-wise curves ONLY for models that really produce them (MLP)
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "loss_curve_"):
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(est.loss_curve_, label="training loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("log loss")
        if getattr(est, "validation_scores_", None) is not None:
            ax2 = ax1.twinx()
            ax2.plot(est.validation_scores_, color="tab:orange", label="validation accuracy (early-stopping split)")
            ax2.set_ylabel("validation accuracy")
            ax2.legend(loc="center right")
        ax1.legend(loc="upper right")
        ax1.set_title(f"{run_id} training history")
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f"history_{run_id}.png", dpi=130)
        plt.close(fig)

    print(json.dumps({k: v for k, v in result.items() if k not in ("features", "student_test")}, indent=2))
    print("student_test:", json.dumps({k: v for k, v in rep.items() if k != "confusion_matrix"}, indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(FAMILIES))
    ap.add_argument("--params-json", default=None)
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()
    prm = json.load(open(a.params_json)) if a.params_json else None
    run(a.model, prm, a.tag)
