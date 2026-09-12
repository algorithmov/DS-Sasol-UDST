"""
sem_racing.trackmodel
======================
This is the piece that did NOT exist across the three bootcamp sandboxes and
is the main architectural gap this system closes:

  Space-Time Transformation produces real, per-lap ``lap_dist`` telemetry.
  The Knowledge-Graph optimizer needs a ``config`` DataFrame of
  distance -> (vmax, vmin, elevation) waypoints, but the notebook only ever
  showed it hand-typed for one fictional track.

``build_track_config`` derives that config automatically from historical lap
data instead of a human re-typing waypoints for every venue, so the
optimizer stays connected to reality as more races are logged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_track_config(
    df: pd.DataFrame,
    section_length: float = 50.0,
    dist_col: str = "lap_dist",
    speed_col: str = "gps_speed",
    vmax_quantile: float = 0.95,
    vmin_quantile: float = 0.05,
    elevation_col: str | None = None,
    standing_start: bool = False,
) -> pd.DataFrame:
    """Bin historical laps by distance-into-lap and derive a speed envelope
    per section from what the car has actually achieved there before.

    vmax per section = a high quantile of historically observed speed in
    that bin (a data-driven proxy for "how fast a corner/straight allows",
    since the sample telemetry has no separate corner-radius channel).
    vmin is a low quantile, mainly so the graph still has a feasible node at
    every section. Both are intentionally conservative (quantiles, not
    max/min) so a single noisy lap can't create an unsafe target.

    Returns a DataFrame with columns ``s, vmax, vmin, m_above_sea`` shaped
    exactly like the ``config`` object ``optimize.build_speed_graph``
    consumes -- i.e. this replaces the hand-authored ``waypoints`` list in
    the original notebooks.
    """
    if dist_col not in df.columns:
        raise KeyError(f"'{dist_col}' missing -- run laps.ensure_laps() first")

    max_dist = float(np.ceil(df[dist_col].max() / section_length) * section_length)
    bins = np.arange(0, max_dist + section_length, section_length)
    df = df.copy()
    df["_bin"] = pd.cut(df[dist_col], bins=bins, right=False, labels=bins[:-1])

    grouped = df.groupby("_bin", observed=True)[speed_col]
    vmax = grouped.quantile(vmax_quantile)
    vmin = grouped.quantile(vmin_quantile)

    config = pd.DataFrame({"s": vmax.index.astype(float), "vmax": vmax.values, "vmin": vmin.values})
    config = config.sort_values("s").reset_index(drop=True)

    if elevation_col and elevation_col in df.columns:
        elev = df.groupby("_bin", observed=True)[elevation_col].mean()
        config["m_above_sea"] = elev.reindex(config["s"]).values
    else:
        config["m_above_sea"] = 0.0  # flat-track assumption; wire in a real elevation channel if logged

    if standing_start:
        # Point-to-point track (e.g. a drag strip): force a dead stop at the
        # very start/end nodes, same convention the notebooks used, so the
        # graph collapses to a single well-defined source/sink.
        config.loc[0, ["vmax", "vmin"]] = 0.0
        config.loc[config.index[-1], ["vmax", "vmin"]] = 0.0
    # else: closed-loop circuit -- keep the real quantile-derived envelope at
    # both ends, since the car is already moving at race pace when it crosses
    # the start/finish line lap after lap. optimize.py handles this as a
    # multi-source/-target search instead of assuming a fixed v=0 start.

    config = config.ffill().bfill()
    config.loc[config["vmin"] > config["vmax"], "vmin"] = config["vmax"]
    return config


def config_from_waypoints(waypoints: list[dict], section_length: float = 50.0) -> pd.DataFrame:
    """Fallback for tracks with no historical telemetry yet: interpolate a
    config from a short hand-authored waypoint list, same mechanism as the
    original notebooks (useful for pre-season planning on a brand new track)."""
    wdf = pd.DataFrame(waypoints).set_index("s").sort_index()
    s_range = np.arange(wdf.index.min(), wdf.index.max() + 1, section_length)
    config = wdf.reindex(wdf.index.union(s_range)).sort_index().interpolate("index").loc[s_range]
    return config.reset_index()


def attach_elevation(
    df: pd.DataFrame,
    elevation: pd.DataFrame,
    dist_col: str = "lap_dist",
    lat_col: str = "gps_latitude",
    lon_col: str = "gps_longitude",
    out_col: str = "elevation_m",
) -> pd.DataFrame:
    """Merge a user-supplied elevation source onto telemetry, so it can be
    passed as ``elevation_col`` to ``build_track_config``. No fabricated
    default here -- if you don't have elevation data, don't call this and
    the track stays explicitly flat (``m_above_sea = 0``) rather than
    silently wrong.

    ``elevation`` needs one of two shapes:
    - distance-based: columns ``dist_m, elevation_m`` -- a single profile
      along the route, matched to each telemetry row by nearest distance.
      Use this if you measured/looked up elevation once along the route
      rather than per-GPS-fix.
    - coordinate-based: columns ``lat, lon, elevation_m`` -- matched to each
      telemetry row by nearest lat/lon (straight-line, fine at this scale).
      Use this if your elevation source is a set of lat/lon points (e.g.
      pulled from a GPX file or an elevation API).
    """
    df = df.copy()
    if {"dist_m", "elevation_m"}.issubset(elevation.columns):
        if dist_col not in df.columns:
            raise KeyError(f"'{dist_col}' missing -- run laps.ensure_laps() first")
        left = df.sort_values(dist_col)
        right = elevation.sort_values("dist_m")
        merged = pd.merge_asof(
            left, right, left_on=dist_col, right_on="dist_m", direction="nearest"
        )
        merged = merged.rename(columns={"elevation_m": out_col}).sort_index()
        return merged

    if {"lat", "lon", "elevation_m"}.issubset(elevation.columns):
        pts = elevation[["lat", "lon", "elevation_m"]].to_numpy()
        query = df[[lat_col, lon_col]].to_numpy()
        # nearest lat/lon in degrees -- fine for matching within one route's
        # extent; not a geodesic distance, don't reuse this for long-range work.
        idx = np.array([np.argmin(np.sum((pts[:, :2] - q) ** 2, axis=1)) for q in query])
        df[out_col] = pts[idx, 2]
        return df

    raise ValueError(
        "elevation must have either ['dist_m','elevation_m'] or "
        "['lat','lon','elevation_m'] columns."
    )
