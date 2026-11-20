"""Raw-file loading + the cleaning rules validated in the preprocessing notebook."""
import numpy as np
import pandas as pd

from .config import partition_dir


def list_trips(partition: str):
    d = partition_dir(partition)
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def _read(path, names, dtype):
    return pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=dtype)


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Same checks as notebook 01: finite, plausible coordinates, no duplicates, strictly increasing time."""
    ok = (
        np.isfinite(track[["lon", "lat", "t"]]).all(axis=1)
        & track["lon"].between(-180, 180)
        & track["lat"].between(-90, 90)
    )
    track = track[ok].drop_duplicates(subset=["lon", "lat", "t"])
    track = track.sort_values("t", kind="stable")
    track = track[track["t"].diff().fillna(1) > 0]
    return track.reset_index(drop=True)


def load_trip(partition: str, trip_id: str, with_route: bool = True, base_dir=None) -> dict:
    d = (base_dir if base_dir is not None else partition_dir(partition)) / trip_id
    track = clean_track(_read(d / f"{trip_id}.track", ["lon", "lat", "t"], float))
    nodes = _read(d / f"{trip_id}.nodes", ["lon", "lat"], float)
    arcs = _read(d / f"{trip_id}.arcs", ["u", "v"], np.int64)
    route = None
    rp = d / f"{trip_id}.route"
    if with_route and rp.exists():
        route = _read(rp, ["arc_id"], np.int64)["arc_id"].to_numpy()
    return {"trip_id": trip_id, "track": track, "nodes": nodes, "arcs": arcs, "route": route}
