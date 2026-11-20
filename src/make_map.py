"""Folium map: raw GPS, nearby road network, reference route, classifier-only route, HMM route.

    python -m src.make_map --partition student_test --trip 00000076
"""
import argparse

import folium
import numpy as np

from .config import MAPS_DIR
from .data_io import load_trip
from .pipeline import load_artifacts, predict_trip, route_coordinates


def arcs_polyline(arc_ids, trip):
    nodes, arcs = trip["nodes"], trip["arcs"]
    return [[[nodes.lat[arcs.u[a]], nodes.lon[arcs.u[a]]], [nodes.lat[arcs.v[a]], nodes.lon[arcs.v[a]]]]
            for a in arc_ids]


def build_map(trip, out, zoom_window=None):
    """zoom_window=(i0, i1): fit the view to GPS points i0..i1 (for close-up screenshots)."""
    tr = trip["track"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=15, tiles="OpenStreetMap")

    # Road network near the trajectory (bounding box + margin, capped for file size)
    pad = 0.002
    nodes, arcs = trip["nodes"], trip["arcs"]
    inbox = nodes.lon.between(tr.lon.min() - pad, tr.lon.max() + pad) & nodes.lat.between(tr.lat.min() - pad, tr.lat.max() + pad)
    near = np.flatnonzero(inbox.to_numpy()[arcs.u] & inbox.to_numpy()[arcs.v])[:20000]
    fg_net = folium.FeatureGroup(name="Road network (nearby)", show=False)
    # one multi-segment PolyLine per layer keeps the HTML small
    folium.PolyLine(arcs_polyline(near, trip), color="#999999", weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS (blue)")
    for lat, lon in zip(tr.lat.to_numpy()[::2], tr.lon.to_numpy()[::2]):
        folium.CircleMarker([lat, lon], radius=2, color="blue", weight=1, fill=True, fill_opacity=0.6, opacity=0.6).add_to(fg_gps)
    fg_gps.add_to(m)

    if trip.get("route") is not None:
        fg_ref = folium.FeatureGroup(name="Reference route (green)")
        folium.PolyLine(arcs_polyline(trip["route"], trip), color="green", weight=7, opacity=0.5).add_to(fg_ref)
        fg_ref.add_to(m)

    for name, key, color in [("Classifier-only route (orange)", "clf_pred", "orange"),
                             ("Classifier + HMM route (red)", "hmm_pred", "red")]:
        _, coords = route_coordinates(out[key], trip)
        fg = folium.FeatureGroup(name=name, show=True)
        folium.PolyLine(coords, color=color, weight=7 if key == "clf_pred" else 3, opacity=0.9).add_to(fg)
        fg.add_to(m)

    folium.Marker([tr.lat.iloc[0], tr.lon.iloc[0]], tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr.lat.iloc[-1], tr.lon.iloc[-1]], tooltip="End", icon=folium.Icon(color="red")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    z = tr.iloc[zoom_window[0]:zoom_window[1]] if zoom_window else tr
    m.fit_bounds([[z.lat.min(), z.lon.min()], [z.lat.max(), z.lon.max()]])
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="student_test")
    ap.add_argument("--trip", required=True)
    ap.add_argument("--zoom", type=float, default=None, help="also write a close-up map at this fraction of the trip")
    a = ap.parse_args()
    trip = load_trip(a.partition, a.trip)
    model, cfg = load_artifacts()
    out = predict_trip(trip, model, cfg)
    path = MAPS_DIR / f"map_{a.partition}_{a.trip}.html"
    build_map(trip, out).save(str(path))
    print("map ->", path)
    if a.zoom:  # close-up: 90 s window starting at the given fraction of the trip
        i0 = int(a.zoom * len(trip["track"]))
        zpath = MAPS_DIR / f"map_{a.partition}_{a.trip}_zoom.html"
        build_map(trip, out, (i0, i0 + 90)).save(str(zpath))
        print("map ->", zpath)
