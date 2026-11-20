"""HMM + Viterbi route reconstruction.

Hidden state at time t      : one candidate road arc (with its projected position) for GPS observation t.
Emission  log P(z_t | s)    : w * log(classifier probability of candidate s)   (w = emission weight)
Transition log P(s_t | s_t-1): -|network_distance - gps_distance| / (beta * max(dt, 1))
                               (Newson & Krumm style). Network distance is computed on the DIRECTED graph
                               with Dijkstra, so impossible moves (wrong way, unconnected) get -inf.
Breaks                       : if no transition is feasible (route hole / GPS jump) the chain is restarted
                               from emissions only, so decoding never fails; the break count is reported.
GPS gaps                     : dt > gap threshold widens the Dijkstra search radius and beta scales with dt.
"""
import math

import networkx as nx
import numpy as np

from .config import HMM_BETA, HMM_EMISSION_WEIGHT, HMM_GAP_SECONDS, HMM_SEARCH_CUTOFF_M

NEG_INF = -1e18


class ShortestPaths:
    def __init__(self, net):
        self.net = net
        self.G = nx.DiGraph()
        self.G.add_weighted_edges_from(zip(net.u.tolist(), net.v.tolist(), net.length.tolist()))
        self.cache = {}

    def from_node(self, node, cutoff):
        key = (node, cutoff)
        if key not in self.cache:
            if node not in self.G:
                self.cache[key] = {node: 0.0}
            else:
                self.cache[key] = nx.single_source_dijkstra_path_length(self.G, node, cutoff=cutoff, weight="weight")
        return self.cache[key]


def network_distance(sp, a, fa, b, fb, cutoff):
    net = sp.net
    la, lb = net.length[a], net.length[b]
    if a == b and fb >= fa - 0.05:  # same arc, moving forward (small backward jitter tolerated)
        return max(fb - fa, 0.0) * la
    d = sp.from_node(int(net.v[a]), cutoff).get(int(net.u[b]))
    if d is None:
        return math.inf
    return (1.0 - fa) * la + d + fb * lb


def viterbi_trip(cands, gps_xy, times, net, sp=None, beta=HMM_BETA, w_em=HMM_EMISSION_WEIGHT,
                 gap_s=HMM_GAP_SECONDS, cutoff=HMM_SEARCH_CUTOFF_M):
    """cands: DataFrame for ONE trip with obs_idx, arc_id, proj_frac, p. Returns (dict obs_idx->arc_id, n_breaks)."""
    sp = sp or ShortestPaths(net)
    groups = [(o, g) for o, g in cands.groupby("obs_idx", sort=True)]
    obs_ids = [o for o, _ in groups]
    arcs = [g["arc_id"].to_numpy() for _, g in groups]
    fracs = [g["proj_frac"].to_numpy() for _, g in groups]
    emis = [w_em * np.log(np.clip(g["p"].to_numpy(), 1e-6, 1.0)) for _, g in groups]

    delta = emis[0].copy()
    backs = [np.full(len(arcs[0]), -1)]
    n_breaks = 0
    for t in range(1, len(groups)):
        o_prev, o_cur = obs_ids[t - 1], obs_ids[t]
        dt = max(float(times[o_cur] - times[o_prev]), 1.0)
        gc = float(np.hypot(*(gps_xy[o_cur] - gps_xy[o_prev])))
        cut = cutoff if dt <= gap_s else max(cutoff, 3.0 * gc + 200.0)
        scale = beta * (dt if dt > gap_s else 1.0)
        A, FA, B, FB = arcs[t - 1], fracs[t - 1], arcs[t], fracs[t]
        trans = np.full((len(A), len(B)), NEG_INF)
        for i in range(len(A)):
            if delta[i] <= NEG_INF / 2:
                continue
            for j in range(len(B)):
                nd = network_distance(sp, A[i], FA[i], B[j], FB[j], cut)
                if nd < math.inf:
                    trans[i, j] = -abs(nd - gc) / scale
        score = delta[:, None] + trans
        best = score.argmax(axis=0)
        new_delta = score[best, np.arange(len(B))] + emis[t]
        if np.all(new_delta <= NEG_INF / 2):  # no feasible transition -> close segment, restart chain
            n_breaks += 1
            best = np.full(len(B), int(np.argmax(delta)))  # previous segment ends in its own best state
            new_delta = emis[t].copy()
        delta = new_delta - new_delta.max()  # normalise to avoid drift
        backs.append(best)

    path = {}
    j = int(np.argmax(delta))
    for t in range(len(groups) - 1, -1, -1):
        path[obs_ids[t]] = int(arcs[t][j])
        j = int(backs[t][j])
    return path, n_breaks
