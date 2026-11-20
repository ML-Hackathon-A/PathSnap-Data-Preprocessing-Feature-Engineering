"""Evaluation metrics: candidate-level classification + observation/route-level map matching."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score)

KEYS = ["trip_id", "obs_idx"]


def candidate_metrics(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision_macro": precision_score(y, yhat, average="macro", zero_division=0),
        "recall_macro": recall_score(y, yhat, average="macro", zero_division=0),
        "f1_macro": f1_score(y, yhat, average="macro", zero_division=0),
        "f1_positive": f1_score(y, yhat, zero_division=0),
        "log_loss": log_loss(y, pc, labels=[0, 1]),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }


def top1_predictions(df, score):
    """Pick the highest-scoring candidate per observation. Returns one row per observation."""
    cols = KEYS + ["arc_id", "u", "v", "y"] + (["label_ok"] if "label_ok" in df else [])
    tmp = df[cols].copy()
    tmp["score"] = score
    idx = tmp.groupby(KEYS, sort=False)["score"].idxmax()
    return tmp.loc[idx].sort_values(KEYS).reset_index(drop=True)


def route_continuity(pred):
    """Share of consecutive predicted arcs that are identical or directly connected (v_prev == u_next)."""
    ok, tot = 0, 0
    for _, g in pred.sort_values(KEYS).groupby("trip_id", sort=False):
        a, u, v = g["arc_id"].to_numpy(), g["u"].to_numpy(), g["v"].to_numpy()
        same = a[1:] == a[:-1]
        conn = v[:-1] == u[1:]
        ok += int((same | conn).sum())
        tot += max(len(a) - 1, 0)
    return ok / tot if tot else float("nan")


def arc_set_f1(pred_arcs, ref_arcs):
    """Sequence-level: precision/recall/F1 between the set of predicted arcs and the reference route arcs."""
    p, r = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    if not p or not r:
        return 0.0, 0.0, 0.0
    tp = len(p & r)
    prec, rec = tp / len(p), tp / len(r)
    return prec, rec, (2 * prec * rec / (prec + rec) if tp else 0.0)


def matching_metrics(df, score):
    pred = top1_predictions(df, score)
    hit = df.groupby(KEYS)["y"].max()
    # Top-1 restricted to observations whose aligned reference arc is within the candidate radius.
    # In trip 00000054 the vehicle is parked off the road network for most of the recording and in 00000033
    # the reference route covers only part of the driven path; plain Top-1 of a fold containing them is
    # dominated by these unlabelled points, so we report both numbers.
    lab = pred[pred["label_ok"] == 1] if "label_ok" in pred else pred
    return {
        "top1_accuracy": float(pred["y"].mean()),
        "top1_accuracy_labelled": float(lab["y"].mean()) if len(lab) else float("nan"),
        "labelled_fraction": float(len(lab) / len(pred)) if len(pred) else float("nan"),
        "candidate_miss_rate": float(1 - hit.mean()),
        "route_continuity": route_continuity(pred),
        "n_observations": int(len(pred)),
    }


def full_report(df, p):
    out = candidate_metrics(df["y"].to_numpy(), p)
    out.update(matching_metrics(df, p))
    return out
