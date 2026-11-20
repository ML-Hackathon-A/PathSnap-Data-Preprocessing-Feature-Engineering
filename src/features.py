"""Candidate generation, label alignment and feature engineering.

One row = (GPS observation, candidate road arc). Target y = 1 if the candidate is the reference arc
the vehicle was on at that observation. The .route file has no timestamps, so labels come from a
monotonic dynamic-programming alignment of GPS points to the ordered route arcs (see align_route).
Labels are ONLY used for y / evaluation, never as features.
"""
import numpy as np
import pandas as pd

from .config import CANDIDATE_RADIUS_M, INDEX_STEP_M, MAX_CANDIDATES
from .geo import LocalProjection, RoadNetwork

FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min", "dist_ratio", "n_cands",
    "dist_prev_m", "dist_next_m", "dist_window_mean",
    "cos_heading", "abs_heading_diff", "cos_heading_long", "speed_mps", "heading_valid",
    "arc_len", "proj_frac", "overhang_m", "out_deg_end", "in_deg_start", "has_twin",
    "dt_prev", "is_fallback",
]

ALIGN_COST_CAP_M = 100.0


def prepare_trip(trip):
    track = trip["track"]
    proj = LocalProjection(track["lon"].mean(), track["lat"].mean())
    net = RoadNetwork(trip["nodes"], trip["arcs"], proj, INDEX_STEP_M)
    gx, gy = proj(track["lon"].to_numpy(), track["lat"].to_numpy())
    return proj, net, gx, gy, track["t"].to_numpy()


def kinematics(gx, gy, t, half_window=1):
    n = len(gx)
    i0 = np.clip(np.arange(n) - half_window, 0, n - 1)
    i1 = np.clip(np.arange(n) + half_window, 0, n - 1)
    dx, dy = gx[i1] - gx[i0], gy[i1] - gy[i0]
    dt = np.maximum(t[i1] - t[i0], 1e-6)
    heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    speed = np.hypot(dx, dy) / dt
    return heading, speed


def angle_diff(a, b):
    d = np.abs(a - b) % 360.0
    return np.minimum(d, 360.0 - d)


def generate_candidates(net, gx, gy, radius=CANDIDATE_RADIUS_M, k=MAX_CANDIDATES):
    hits = net.tree.query_ball_point(np.c_[gx, gy], r=radius + net.index_step)
    obs_l, arc_l, rank_l, fb_l = [], [], [], []
    for i, h in enumerate(hits):
        arcs = np.unique(net.sample_arc[h]) if h else np.empty(0, dtype=np.int64)
        fallback = False
        if len(arcs):
            d, _, _ = net.project(gx[i], gy[i], arcs)
            m = d <= radius
            arcs, d = arcs[m], d[m]
        if len(arcs) == 0:  # nothing within radius: take the nearest arcs anywhere
            fallback = True
            _, idx = net.tree.query([gx[i], gy[i]], k=50)
            arcs = np.unique(net.sample_arc[np.atleast_1d(idx)])
            d, _, _ = net.project(gx[i], gy[i], arcs)
        order = np.argsort(d, kind="stable")[:k]
        arcs = arcs[order]
        obs_l.append(np.full(len(arcs), i))
        arc_l.append(arcs)
        rank_l.append(np.arange(len(arcs)))
        fb_l.append(np.full(len(arcs), fallback))
    return np.concatenate(obs_l), np.concatenate(arc_l), np.concatenate(rank_l), np.concatenate(fb_l)


def align_route(net, gx, gy, route):
    """Monotonic DP: assign each GPS point a route position j (non-decreasing in time) minimising the
    sum of point-to-arc distances (capped at 100 m for robustness). Returns route index per point."""
    n, m = len(gx), len(route)
    back = np.empty((n, m), dtype=np.int32)
    idx = np.arange(m)

    def cost(i):
        d, _, _ = net.project(gx[i], gy[i], route)
        return np.minimum(d, ALIGN_COST_CAP_M)

    D = cost(0)
    back[0] = idx
    for i in range(1, n):
        pm_val = np.minimum.accumulate(D)
        pm_idx = np.maximum.accumulate(np.where(D <= pm_val, idx, 0))
        D = cost(i) + pm_val
        back[i] = pm_idx
    path = np.empty(n, dtype=np.int64)
    j = int(np.argmin(D))
    for i in range(n - 1, -1, -1):
        path[i] = j
        j = back[i, j]
    return path


def build_trip_frame(trip, with_labels=True):
    """Return the candidate/feature frame for one trip (plus labels if a route is available)."""
    proj, net, gx, gy, t = prepare_trip(trip)
    n = len(gx)
    obs, arc, rank, fb = generate_candidates(net, gx, gy)

    dist, frac, traw = net.project(gx[obs], gy[obs], arc)
    heading, speed = kinematics(gx, gy, t, 1)
    heading_long, _ = kinematics(gx, gy, t, 5)

    def dist_to(offset):
        j = np.clip(obs + offset, 0, n - 1)
        return net.project(gx[j], gy[j], arc)[0]

    d_prev, d_next = dist_to(-1), dist_to(1)
    d_window = (dist_to(-2) + d_prev + dist + d_next + dist_to(2)) / 5.0

    df = pd.DataFrame({"trip_id": trip["trip_id"], "obs_idx": obs, "arc_id": arc})
    df["t"] = t[obs]
    df["u"] = net.u[arc]
    df["v"] = net.v[arc]
    df["dist_m"] = dist
    df["dist_rank"] = rank
    dmin = df.groupby("obs_idx")["dist_m"].transform("min").to_numpy()
    df["dist_minus_min"] = dist - dmin
    df["dist_ratio"] = dist / (dmin + 1.0)
    df["n_cands"] = df.groupby("obs_idx")["arc_id"].transform("size").to_numpy()
    df["dist_prev_m"] = d_prev
    df["dist_next_m"] = d_next
    df["dist_window_mean"] = d_window
    valid = speed[obs] >= 1.5
    hd = angle_diff(heading[obs], net.bearing[arc])
    hd_long = angle_diff(heading_long[obs], net.bearing[arc])
    df["cos_heading"] = np.where(valid, np.cos(np.radians(hd)), 0.0)
    df["abs_heading_diff"] = np.where(valid, hd, 90.0)
    df["cos_heading_long"] = np.where(valid, np.cos(np.radians(hd_long)), 0.0)
    df["speed_mps"] = speed[obs]
    df["heading_valid"] = valid.astype(np.int8)
    df["arc_len"] = net.length[arc]
    df["proj_frac"] = frac
    df["overhang_m"] = np.maximum(np.maximum(-traw, traw - 1.0), 0.0) * net.length[arc]
    df["out_deg_end"] = net.out_deg[net.v[arc]]
    df["in_deg_start"] = net.in_deg[net.u[arc]]
    df["has_twin"] = net.has_twin[arc].astype(np.int8)
    dt_prev = np.r_[1.0, np.diff(t)]
    df["dt_prev"] = dt_prev[obs]
    df["is_fallback"] = fb.astype(np.int8)

    if with_labels and trip.get("route") is not None:
        df = attach_labels(df, net, gx, gy, trip["route"])
    return df


def attach_labels(df, net, gx, gy, route):
    path = align_route(net, gx, gy, route)
    label_arc = route[path]
    label_dist, _, _ = net.project(gx, gy, label_arc)
    obs = df["obs_idx"].to_numpy()
    df["label_arc"] = label_arc[obs]
    df["label_dist_m"] = label_dist[obs]
    df["y"] = (df["arc_id"].to_numpy() == df["label_arc"].to_numpy()).astype(np.int8)
    # Where the reference route has holes (see the disconnected transitions in notebook 01) the aligned
    # label can be hundreds of metres from the GPS point. Such labels are unreliable: we do not TRAIN on
    # them, but we still EVALUATE on all observations (they count as candidate misses).
    df["label_ok"] = (df["label_dist_m"] <= CANDIDATE_RADIUS_M).astype(np.int8)
    return df
