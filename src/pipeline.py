"""End-to-end inference for one trip: raw files -> candidates/features -> classifier -> HMM/Viterbi.
Ground truth (.route) is only used AFTER prediction, in evaluate_prediction(), on the server."""
import json
import time

import joblib
import numpy as np

from .config import ARTIFACT_DIR
from .features import FEATURES, attach_labels, build_trip_frame, prepare_trip
from .hmm import ShortestPaths, viterbi_trip
from .metrics import arc_set_f1


def load_artifacts():
    model = joblib.load(ARTIFACT_DIR / "final_model.joblib")
    hmm_cfg = json.load(open(ARTIFACT_DIR / "hmm_config.json"))
    return model, hmm_cfg


def predict_trip(trip, model, hmm_cfg):
    t0 = time.time()
    trip_nolabel = {**trip, "route": None}  # the prediction path never sees the route
    df = build_trip_frame(trip_nolabel, with_labels=False)
    df["p"] = model.predict_proba(df[FEATURES].to_numpy(dtype=np.float32))[:, 1]
    proj, net, gx, gy, times = prepare_trip(trip_nolabel)
    clf_pred = df.loc[df.groupby("obs_idx")["p"].idxmax()].set_index("obs_idx")["arc_id"].to_dict()
    hmm_pred, n_breaks = viterbi_trip(df, np.c_[gx, gy], times, net, ShortestPaths(net),
                                      beta=hmm_cfg["beta"], w_em=hmm_cfg["emission_weight"])
    return {"df": df, "net": net, "gx": gx, "gy": gy, "clf_pred": clf_pred, "hmm_pred": hmm_pred,
            "n_breaks": n_breaks, "seconds": time.time() - t0}


def path_metrics(pred, net, labels_by_obs, route):
    obs = np.array(sorted(pred))
    arcs = np.array([pred[o] for o in obs])
    acc = float(np.mean(arcs == labels_by_obs[obs]))
    same = arcs[1:] == arcs[:-1]
    conn = net.v[arcs[:-1]] == net.u[arcs[1:]]
    cont = float(np.mean(same | conn)) if len(arcs) > 1 else float("nan")
    seq = [a for i, a in enumerate(arcs) if i == 0 or a != arcs[i - 1]]
    prec, rec, f1 = arc_set_f1(seq, route)
    return {"top1_accuracy": acc, "route_continuity": cont, "arc_precision": prec, "arc_recall": rec, "arc_f1": f1}


def evaluate_prediction(trip, out):
    """Server-side only: compare predictions with the aligned reference labels."""
    if trip.get("route") is None:
        return None
    df = attach_labels(out["df"].copy(), out["net"], out["gx"], out["gy"], trip["route"])
    labels = df.groupby("obs_idx")["label_arc"].first()
    lab = np.full(int(labels.index.max()) + 1, -1)
    lab[labels.index.to_numpy()] = labels.to_numpy()
    return {"classifier_only": path_metrics(out["clf_pred"], out["net"], lab, trip["route"]),
            "classifier_hmm": path_metrics(out["hmm_pred"], out["net"], lab, trip["route"])}


def route_coordinates(pred, trip):
    """Lat/lon polyline of the predicted arc sequence (consecutive duplicates removed)."""
    nodes, arcs = trip["nodes"], trip["arcs"]
    seq = []
    for o in sorted(pred):
        a = pred[o]
        if not seq or seq[-1] != a:
            seq.append(a)
    coords = []
    for a in seq:
        u, v = arcs.iloc[a]["u"], arcs.iloc[a]["v"]
        for n in (u, v):
            pt = [float(nodes.iloc[n]["lat"]), float(nodes.iloc[n]["lon"])]
            if not coords or coords[-1] != pt:
                coords.append(pt)
    return seq, coords
