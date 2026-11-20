"""Final evaluation of the FROZEN models on a hold-out partition (no refit, no tuning).

    python -m src.evaluate_holdout --partition eval           # evaluator_test (PATHSNAP_EVAL_DIR)
    python -m src.evaluate_holdout --partition student_test

For every saved model (5 base models + tuned LightGBM) this computes the Deliverables 3.1 metrics
on the partition, then decodes every trip with the HMM/Viterbi decoder using that model's
probabilities. The decoder parameters come from artifacts/hmm_config.json, which was tuned on
student_test, so the result is classifier-only vs classifier + HMM for each model.

Nothing here feeds back into model or hyperparameter choice: the final model stays the one selected
on student_test (artifacts/final_manifest.json). Evaluator results are reported, not used for selection.
"""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd

from .config import ARTIFACT_DIR, MODEL_DIR, RESULTS_DIR
from .data_io import list_trips, load_trip
from .features import FEATURES, build_trip_frame, prepare_trip
from .hmm import ShortestPaths, viterbi_trip
from .metrics import full_report
from .models import FAMILIES
from .pipeline import path_metrics

RUNS = ["logreg", "knn", "rf", "lgbm", "mlp", "lgbm_tuned"]


def main(partition):
    out_dir = RESULTS_DIR / ("evaluator_test" if partition == "eval" else partition)
    out_dir.mkdir(parents=True, exist_ok=True)
    hcfg = json.load(open(ARTIFACT_DIR / "hmm_config.json"))
    final_run = json.load(open(ARTIFACT_DIR / "final_manifest.json"))["run_id"]

    t0 = time.time()
    trips = []
    for tid in list_trips(partition):
        trip = load_trip(partition, tid)
        df = build_trip_frame(trip, with_labels=True)
        _, net, gx, gy, times = prepare_trip(trip)
        labels = df.groupby("obs_idx")["label_arc"].first()
        lab = np.full(int(labels.index.max()) + 1, -1)
        lab[labels.index.to_numpy()] = labels.to_numpy()
        trips.append(dict(tid=tid, trip=trip, df=df, net=net, xy=np.c_[gx, gy], times=times, lab=lab,
                          sp=ShortestPaths(net)))
    allf = pd.concat([t["df"] for t in trips], ignore_index=True)
    print(f"[{partition}] {len(trips)} trips, {allf.groupby(['trip_id', 'obs_idx']).ngroups:,} observations, "
          f"{len(allf):,} candidate rows, features in {time.time() - t0:.1f}s")

    summary, per_trip = [], []
    for run in RUNS:
        path = MODEL_DIR / f"{run}.joblib"
        if not path.exists():
            print("skip (no model):", run)
            continue
        model = joblib.load(path)
        X = allf[FEATURES].to_numpy(dtype=np.float32)
        t1 = time.time()
        p = model.predict_proba(X)[:, 1]
        infer_s = time.time() - t1
        rep = full_report(allf, p)
        allf["p"] = p
        hmm_rows, t_dec = [], 0.0
        for t in trips:
            d = allf[allf.trip_id == t["tid"]]
            clf = d.loc[d.groupby("obs_idx")["p"].idxmax()].set_index("obs_idx")["arc_id"].to_dict()
            t2 = time.time()
            hpath, nb = viterbi_trip(d, t["xy"], t["times"], t["net"], t["sp"], beta=hcfg["beta"], w_em=hcfg["emission_weight"])
            t_dec += time.time() - t2
            mc = path_metrics(clf, t["net"], t["lab"], t["trip"]["route"])
            mh = path_metrics(hpath, t["net"], t["lab"], t["trip"]["route"])
            n = len(hpath)
            row = {"run_id": run, "trip_id": t["tid"], "n_obs": n, "breaks": nb,
                   **{f"clf_{k}": v for k, v in mc.items()}, **{f"hmm_{k}": v for k, v in mh.items()}}
            per_trip.append(row)
            hmm_rows.append(row)
        h = pd.DataFrame(hmm_rows)
        w = h["n_obs"] / h["n_obs"].sum()
        summary.append({
            "run_id": run, "family": FAMILIES[run.split("_")[0]], "is_final_model": run == final_run,
            "top1_accuracy": rep["top1_accuracy"], "top1_accuracy_labelled": rep["top1_accuracy_labelled"],
            "f1_macro": rep["f1_macro"], "precision_macro": rep["precision_macro"], "recall_macro": rep["recall_macro"],
            "accuracy": rep["accuracy"], "log_loss": rep["log_loss"], "route_continuity": rep["route_continuity"],
            "candidate_miss_rate": rep["candidate_miss_rate"],
            "inference_ms_per_obs": 1000 * infer_s / rep["n_observations"],
            "hmm_top1_accuracy": float((h["hmm_top1_accuracy"] * w).sum()),
            "hmm_route_continuity": float((h["hmm_route_continuity"] * w).sum()),
            "clf_arc_f1_mean": float(h["clf_arc_f1"].mean()), "hmm_arc_f1_mean": float(h["hmm_arc_f1"].mean()),
            "hmm_decode_s_total": t_dec, "confusion_matrix": json.dumps(rep["confusion_matrix"]),
        })
        print(f"  {run:11s} top1={rep['top1_accuracy']:.4f} -> HMM {summary[-1]['hmm_top1_accuracy']:.4f}  "
              f"F1={rep['f1_macro']:.4f} logloss={rep['log_loss']:.4f} cont={rep['route_continuity']:.4f} -> "
              f"{summary[-1]['hmm_route_continuity']:.4f}")
    S = pd.DataFrame(summary)
    S.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(per_trip).to_csv(out_dir / "per_trip.csv", index=False)
    json.dump({"partition": partition, "hmm_config": hcfg, "final_model": final_run,
               "n_trips": len(trips), "note": "frozen models, no refit/tuning on this partition"},
              open(out_dir / "meta.json", "w"), indent=2)
    print("->", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="eval", choices=["eval", "student_test"])
    main(ap.parse_args().partition)
