"""Exploratory data analysis for report section 2.2 (train + student_test partitions only).

    python -m src.eda
Outputs: results/eda/*.csv, results/plots/eda_*.png, results/maps/eda_*.html
"""
import folium
import matplotlib
import networkx as nx
import numpy as np
import pandas as pd

from .config import MAPS_DIR, PLOTS_DIR, RESULTS_DIR
from .data_io import list_trips, load_trip, partition_dir
from .geo import LocalProjection

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EDA_DIR = RESULTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)


def trip_stats(partition, tid):
    trip = load_trip(partition, tid)
    tr, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    proj = LocalProjection(tr.lon.mean(), tr.lat.mean())
    x, y = proj(tr.lon.to_numpy(), tr.lat.to_numpy())
    step = np.hypot(np.diff(x), np.diff(y))
    dt = np.diff(tr.t.to_numpy())
    G = nx.DiGraph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(zip(arcs.u, arcs.v))
    wcc = nx.number_weakly_connected_components(G)
    isolated = int(sum(1 for n in G.nodes if G.degree(n) == 0))
    ra = arcs.iloc[route].to_numpy()
    files = {ext: (partition_dir(partition) / tid / f"{tid}.{ext}").stat().st_size for ext in ("track", "nodes", "arcs", "route")}
    return {
        "partition": partition, "trip_id": tid, "gps_points": len(tr),
        "duration_s": float(tr.t.iloc[-1] - tr.t.iloc[0]), "length_km": float(step.sum() / 1000),
        "dt_1s_share": float(np.mean(dt == 1)), "max_gap_s": float(dt.max()), "gaps_over_10s": int((dt > 10).sum()),
        "max_jump_m": float(step.max()), "lon_min": tr.lon.min(), "lon_max": tr.lon.max(),
        "lat_min": tr.lat.min(), "lat_max": tr.lat.max(), "nodes": len(nodes), "arcs": len(arcs),
        "weak_components": wcc, "isolated_nodes": isolated, "route_arcs": len(route),
        "route_disconnections": int((ra[:-1, 1] != ra[1:, 0]).sum()),
        "missing_values": int(tr.isna().sum().sum() + nodes.isna().sum().sum() + arcs.isna().sum().sum()),
        **{f"bytes_{k}": v for k, v in files.items()},
    }


def plot_trip(ax, partition, tid, with_network=True):
    trip = load_trip(partition, tid)
    tr, nodes, arcs, route = trip["track"], trip["nodes"], trip["arcs"], trip["route"]
    pad = 0.003
    if with_network:
        inb = (nodes.lon.between(tr.lon.min() - pad, tr.lon.max() + pad)
               & nodes.lat.between(tr.lat.min() - pad, tr.lat.max() + pad)).to_numpy()
        sel = np.flatnonzero(inb[arcs.u] & inb[arcs.v])
        for a in sel:
            u, v = arcs.u[a], arcs.v[a]
            ax.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]], color="#bbbbbb", lw=0.5, zorder=1)
    for a in route:
        u, v = arcs.u[a], arcs.v[a]
        ax.plot([nodes.lon[u], nodes.lon[v]], [nodes.lat[u], nodes.lat[v]], color="limegreen", lw=3, alpha=0.7, zorder=3)
    ax.scatter(tr.lon, tr.lat, s=1, c="blue", zorder=5)
    ax.set_title(f"{partition} {tid}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect(1 / np.cos(np.radians(tr.lat.mean())))


def main():
    rows = [trip_stats(p, t) for p in ("train", "student_test") for t in list_trips(p)]
    df = pd.DataFrame(rows)
    df.to_csv(EDA_DIR / "trip_inventory.csv", index=False)
    summ = df.groupby("partition")[["gps_points", "duration_s", "length_km", "dt_1s_share", "max_gap_s",
                                    "gaps_over_10s", "nodes", "arcs", "weak_components", "isolated_nodes",
                                    "route_arcs", "route_disconnections", "missing_values"]].describe().T
    summ.to_csv(EDA_DIR / "partition_summary.csv")
    print(df.groupby("partition").agg(trips=("trip_id", "size"), gps=("gps_points", "sum"),
                                      km=("length_km", "sum"), nodes=("nodes", "sum"), arcs=("arcs", "sum"),
                                      missing=("missing_values", "sum"), gaps10=("gaps_over_10s", "sum"),
                                      route_disc=("route_disconnections", "sum")).to_string())

    # sampling-interval distribution (train)
    dts = []
    for t in list_trips("train"):
        dts.append(np.diff(load_trip("train", t)["track"].t.to_numpy()))
    dts = np.concatenate(dts)
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    vals, cnts = np.unique(np.clip(dts, 0, 20), return_counts=True)
    axs[0].bar(vals, cnts)
    axs[0].set_yscale("log")
    axs[0].set_title("Train: sampling interval (s, clipped at 20)")
    axs[0].set_xlabel("seconds between GPS points")
    for p, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        axs[1].hist(df[df.partition == p].gps_points, bins=20, alpha=0.6, label=p, color=c)
        axs[2].hist(df[df.partition == p].length_km, bins=20, alpha=0.6, label=p, color=c)
    axs[1].set_title("GPS points per trip")
    axs[2].set_title("Trip length (km)")
    axs[1].legend()
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_distributions.png", dpi=130)
    plt.close(fig)

    # missing values (all partitions & files)
    fig, ax = plt.subplots(figsize=(6, 3))
    mv = df.groupby("partition")["missing_values"].sum()
    ax.bar(mv.index, mv.values)
    ax.set_ylim(0, 1)
    ax.set_title("Missing values per partition (all four file types)")
    for i, v in enumerate(mv.values):
        ax.text(i, 0.05, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_missing_values.png", dpi=130)
    plt.close(fig)

    # network size vs. trajectory size
    fig, ax = plt.subplots(figsize=(6, 4))
    for p, c in (("train", "tab:blue"), ("student_test", "tab:orange")):
        d = df[df.partition == p]
        ax.scatter(d.arcs, d.gps_points, label=p, c=c, s=18)
    ax.set_xscale("log")
    ax.set_xlabel("road arcs in trip network (log)")
    ax.set_ylabel("GPS points")
    ax.set_title("Road-network size vs trajectory length")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_network_size.png", dpi=130)
    plt.close(fig)

    # GPS trajectories + reference routes + road network
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (p, t) in zip(axs, [("train", "00000005"), ("train", "00000054"), ("student_test", "00000076")]):
        plot_trip(ax, p, t)
    fig.suptitle("GPS points (blue), reference route (green), road network (grey)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_trajectories.png", dpi=130)
    plt.close(fig)

    # Zoomed view: long off-network stationary period in trip 00000054 vs a normal trip
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (p, t) in zip(axs, [("train", "00000005"), ("train", "00000054")]):
        plot_trip(ax, p, t)
        tr = load_trip(p, t)["track"]
        c = tr.iloc[len(tr) // 2]
        ax.set_xlim(c.lon - 0.006, c.lon + 0.006)
        ax.set_ylim(c.lat - 0.004, c.lat + 0.004)
        ax.set_title(f"{p} {t} (zoom ~800 m)")
    fig.suptitle("Zoom: GPS (blue) vs reference route (green). In 00000054 the vehicle is parked off the road network for most of the recording")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "eda_offset_zoom.png", dpi=130)
    plt.close(fig)

    # Folium EDA map: raw GPS + network around trip 00000054 (off-network parking)
    trip = load_trip("train", "00000054")
    tr, nodes, arcs = trip["track"], trip["nodes"], trip["arcs"]
    m = folium.Map(location=[tr.lat.mean(), tr.lon.mean()], zoom_start=13)
    folium.PolyLine([[[nodes.lat[arcs.u[a]], nodes.lon[arcs.u[a]]], [nodes.lat[arcs.v[a]], nodes.lon[arcs.v[a]]]]
                     for a in trip["route"]], color="green", weight=6, opacity=0.6,
                    tooltip="reference route").add_to(folium.FeatureGroup(name="Reference route (green)").add_to(m))
    fg = folium.FeatureGroup(name="Raw GPS (blue)").add_to(m)
    for lat, lon in zip(tr.lat.to_numpy()[::5], tr.lon.to_numpy()[::5]):
        folium.CircleMarker([lat, lon], radius=2, color="blue", fill=True).add_to(fg)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr.lat.min(), tr.lon.min()], [tr.lat.max(), tr.lon.max()]])
    m.save(str(MAPS_DIR / "eda_train_00000054_parked.html"))
    print("EDA outputs ->", EDA_DIR, PLOTS_DIR)


if __name__ == "__main__":
    main()
