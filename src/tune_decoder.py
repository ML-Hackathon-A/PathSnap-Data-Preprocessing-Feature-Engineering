"""Tune HMM decoder parameters (beta, emission weight) and compare classifier-only vs classifier+HMM.
Uses ONLY student_test trips (allowed for decoder tuning). Requires artifacts/final_model.joblib.

    python -m src.tune_decoder
"""
import itertools
import json
import time

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, RESULTS_DIR
from .data_io import list_trips, load_trip
from .features import FEATURES, build_trip_frame, prepare_trip
from .hmm import ShortestPaths, viterbi_trip
from .pipeline import path_metrics

BETAS = [2.0, 5.0, 10.0, 20.0]
EMISSION_WEIGHTS = [0.5, 1.0, 2.0]


def main():
    model = joblib.load(ARTIFACT_DIR / "final_model.joblib")
    prepared = []
    for tid in list_trips("student_test"):
        trip = load_trip("student_test", tid)
        df = build_trip_frame(trip, with_labels=True)
        df["p"] = model.predict_proba(df[FEATURES].to_numpy(dtype=np.float32))[:, 1]
        _, net, gx, gy, times = prepare_trip(trip)
        labels = df.groupby("obs_idx")["label_arc"].first()
        lab = np.full(int(labels.index.max()) + 1, -1)
        lab[labels.index.to_numpy()] = labels.to_numpy()
        clf = df.loc[df.groupby("obs_idx")["p"].idxmax()].set_index("obs_idx")["arc_id"].to_dict()
        prepared.append((tid, trip, df, net, np.c_[gx, gy], times, lab, clf, ShortestPaths(net)))

    rows = []
    for beta, w in itertools.product(BETAS, EMISSION_WEIGHTS):
        for tid, trip, df, net, xy, times, lab, clf, sp in prepared:
            t0 = time.time()
            path, nb = viterbi_trip(df, xy, times, net, sp, beta=beta, w_em=w)
            m_h = path_metrics(path, net, lab, trip["route"])
            m_c = path_metrics(clf, net, lab, trip["route"])
            rows.append({"beta": beta, "emission_weight": w, "trip_id": tid, "breaks": nb,
                         "decode_s": time.time() - t0,
                         **{f"hmm_{k}": v for k, v in m_h.items()}, **{f"clf_{k}": v for k, v in m_c.items()}})
        print(f"beta={beta} w={w} done")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "hmm_grid_per_trip.csv", index=False)
    summary = res.groupby(["beta", "emission_weight"]).mean(numeric_only=True).reset_index()
    summary.to_csv(RESULTS_DIR / "hmm_grid_summary.csv", index=False)
    best = summary.sort_values(["hmm_top1_accuracy", "hmm_route_continuity"], ascending=False).iloc[0]
    cfg = {"beta": float(best["beta"]), "emission_weight": float(best["emission_weight"]),
           "selected_on": "student_test mean top1 accuracy"}
    json.dump(cfg, open(ARTIFACT_DIR / "hmm_config.json", "w"), indent=2)
    cols = ["clf_top1_accuracy", "hmm_top1_accuracy", "clf_route_continuity", "hmm_route_continuity",
            "clf_arc_f1", "hmm_arc_f1", "breaks"]
    print(summary[["beta", "emission_weight"] + cols].round(4).to_string(index=False))
    print("best decoder config:", cfg)
    per_trip = res[(res.beta == cfg["beta"]) & (res.emission_weight == cfg["emission_weight"])]
    print(per_trip[["trip_id"] + cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
