"""Build candidate/feature tables for train and student_test.

    python -m src.build_dataset            # both partitions
    python -m src.build_dataset --limit 3  # quick smoke test on 3 trips per partition
"""
import argparse
import time

import pandas as pd
from joblib import Parallel, delayed

from .config import FEATURE_DIR, N_JOBS, TRAIN_OBS_STEP
from .data_io import list_trips, load_trip
from .features import build_trip_frame


def _one(partition, trip_id):
    return build_trip_frame(load_trip(partition, trip_id), with_labels=True)


def build(partition, limit=None):
    trips = list_trips(partition)[:limit]
    t0 = time.time()
    frames = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(_one)(partition, tid) for tid in trips)
    df = pd.concat(frames, ignore_index=True)
    if partition == "train" and TRAIN_OBS_STEP > 1:
        df = df[df["obs_idx"] % TRAIN_OBS_STEP == 0].reset_index(drop=True)
    out = FEATURE_DIR / f"{partition}.pkl"
    df.to_pickle(out)
    per_obs = df.groupby(["trip_id", "obs_idx"])["y"].max()
    print(f"[{partition}] trips={len(trips)} rows={len(df):,} observations={len(per_obs):,} "
          f"positive_rate={df['y'].mean():.3f} candidate_miss_rate={1 - per_obs.mean():.4f} "
          f"time={time.time() - t0:.1f}s -> {out}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", choices=["train", "student_test", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    for p in (["train", "student_test"] if a.partition == "both" else [a.partition]):
        build(p, a.limit)
