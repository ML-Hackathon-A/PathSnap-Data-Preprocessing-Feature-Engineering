"""Geometry: local metric projection + road network with a spatial index.

Lon/lat are angles, so we project each trip onto a local tangent plane (equirectangular around the
trip centroid). At city scale (< 50 km) the distortion is far below GPS noise, and it avoids a pyproj
dependency. Distances are therefore in metres.
"""
import numpy as np
from scipy.spatial import cKDTree

EARTH_R = 6371008.8


class LocalProjection:
    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.kx = EARTH_R * np.cos(np.radians(lat0)) * np.pi / 180.0
        self.ky = EARTH_R * np.pi / 180.0

    def __call__(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky


class RoadNetwork:
    def __init__(self, nodes, arcs, proj: LocalProjection, index_step: float = 10.0):
        x, y = proj(nodes["lon"].to_numpy(), nodes["lat"].to_numpy())
        self.node_x, self.node_y = x, y
        self.n_nodes = len(nodes)
        self.u = arcs["u"].to_numpy()
        self.v = arcs["v"].to_numpy()
        self.ax, self.ay = x[self.u], y[self.u]
        self.dx, self.dy = x[self.v] - self.ax, y[self.v] - self.ay
        self.length = np.hypot(self.dx, self.dy)
        self.bearing = np.degrees(np.arctan2(self.dx, self.dy)) % 360.0  # 0 = north, clockwise
        self.out_deg = np.bincount(self.u, minlength=self.n_nodes)
        self.in_deg = np.bincount(self.v, minlength=self.n_nodes)
        key = self.u * self.n_nodes + self.v
        self.has_twin = np.isin(self.v * self.n_nodes + self.u, key)  # two-way street indicator
        self.index_step = index_step
        self._build_index(index_step)

    def _build_index(self, step):
        n_samp = np.maximum(1, np.ceil(self.length / step).astype(np.int64)) + 1
        self.sample_arc = np.repeat(np.arange(len(self.u)), n_samp)
        starts = np.repeat(np.cumsum(n_samp) - n_samp, n_samp)
        t = (np.arange(len(self.sample_arc)) - starts) / np.repeat(n_samp - 1, n_samp)
        px = self.ax[self.sample_arc] + t * self.dx[self.sample_arc]
        py = self.ay[self.sample_arc] + t * self.dy[self.sample_arc]
        self.tree = cKDTree(np.c_[px, py])

    def project(self, px, py, arc_ids):
        """Element-wise projection of points onto arcs. Returns (distance_m, t_clipped, t_raw)."""
        dx, dy = self.dx[arc_ids], self.dy[arc_ids]
        l2 = np.maximum(dx * dx + dy * dy, 1e-9)
        t_raw = ((px - self.ax[arc_ids]) * dx + (py - self.ay[arc_ids]) * dy) / l2
        t = np.clip(t_raw, 0.0, 1.0)
        qx, qy = self.ax[arc_ids] + t * dx, self.ay[arc_ids] + t * dy
        return np.hypot(px - qx, py - qy), t, t_raw

    def point_on_arc(self, arc_ids, t):
        return self.ax[arc_ids] + t * self.dx[arc_ids], self.ay[arc_ids] + t * self.dy[arc_ids]
