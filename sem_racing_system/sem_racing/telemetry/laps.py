"""
sem_racing.laps
================
Lap segmentation. Two situations show up in the club's data and this module
handles both behind one function, ``ensure_laps``:

* Some exports (Sandbox 1 / 2-3 samples) already ship with ``lap_lap`` /
  ``start_lap`` / ``finish_lap`` columns baked in by the logger -- nothing to
  do.
* Raw OBC logs (Space-Time Transformation sample) only have a GPS trace and
  need the start/finish/lap timing lines to be intersected against the
  vehicle's path, exactly like ``JupyterShowcase.ipynb`` does by hand. That
  logic is lifted out of the notebook and turned into ``segment_laps`` so it
  can run on *any* track's line coordinates, not just London 2019.
"""
from __future__ import annotations

from typing import Mapping, Tuple

import numpy as np
import pandas as pd

LineCoords = Tuple[Tuple[float, float], Tuple[float, float]]  # (lon1,lon2),(lat1,lat2)

EARTH_RADIUS_M = 6371000


def add_cumulative_distance(
    df: pd.DataFrame, lat_col: str = "gps_latitude", lon_col: str = "gps_longitude"
) -> pd.DataFrame:
    """Haversine great-circle distance between consecutive GPS fixes,
    accumulated into a running total distance column ``dist``."""
    df = df.copy()
    lat1 = np.radians(df[lat_col])
    lon1 = np.radians(df[lon_col])
    lat2 = np.radians(df[lat_col].shift(1))
    lon2 = np.radians(df[lon_col].shift(1))

    step = 2 * EARTH_RADIUS_M * np.arcsin(
        np.sqrt(
            np.sin((lat2 - lat1) / 2) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        )
    )
    df["dist"] = step.fillna(0).cumsum()
    return df


def segment_laps(
    df: pd.DataFrame,
    lines: Mapping[str, LineCoords],
    lat_col: str = "gps_latitude",
    lon_col: str = "gps_longitude",
) -> pd.DataFrame:
    """Detect crossings of named track lines (typically ``start``, ``finish``,
    ``lap``) via 2D segment intersection between consecutive GPS fixes and the
    line, and derive a running lap counter (``lap_lap``) plus per-line
    crossing counters (``<name>_lap``).

    ``lines`` maps a line name to ``((lon1, lon2), (lat1, lat2))`` -- the same
    shape used in the bootcamp notebook, so track configs can be copy-pasted
    straight out of it.
    """
    df = df.copy()
    x3, y3 = df[lon_col], df[lat_col]
    x4, y4 = df[lon_col].shift(-1), df[lat_col].shift(-1)

    for name, (xs, ys) in lines.items():
        x1, x2 = xs
        y1, y2 = ys
        denom = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
        t = (((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))) / denom
        u = (((x1 - x2) * (y1 - y3)) - ((y1 - y2) * (x1 - x3))) / denom
        crossed = (t.round(1) >= 0) & (t.round(1) <= 1) & (u.round(1) >= 0) & (u.round(1) <= 1)
        df[f"{name}_cross"] = crossed.fillna(False)
        df[f"{name}_lap"] = df[f"{name}_cross"].cumsum()

    if lat_col == "gps_latitude" and lon_col == "gps_longitude":
        df = add_cumulative_distance(df, lat_col, lon_col)

    if "lap" in lines:
        df["lap_lap"] = df["lap_lap"] if "lap_lap" in df.columns else df["lap_cross"].cumsum()
    return df


def trim_to_race_window(df: pd.DataFrame) -> pd.DataFrame:
    """Drop everything before the first start-line crossing and everything
    after the last finish-line crossing, same trimming rule as the notebook."""
    df = df.copy()
    if "start_lap" in df.columns:
        df = df.loc[df["start_lap"] >= 1]
    if "finish_lap" in df.columns and df["finish_lap"].max() > 0:
        df = df.loc[df["finish_lap"] < df["finish_lap"].max()]
    return df.reset_index(drop=True)


def add_lap_relative_columns(
    df: pd.DataFrame,
    lap_col: str = "lap_lap",
    time_col: str = "obc_timestamp",
    dist_col: str = "dist",
    cumulative_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """For each lap, zero out time/distance/energy counters so laps become
    directly comparable (lap_time, lap_dist, lap_<col> for each cumulative
    energy/flow channel requested)."""
    df = df.copy()
    if lap_col not in df.columns:
        raise KeyError(
            f"'{lap_col}' not found -- run segment_laps() first, "
            "or this export already has lap columns under a different name."
        )
    for lap, group in df.groupby(lap_col):
        idx = group.index
        if time_col in df.columns:
            df.loc[idx, "lap_time"] = group[time_col] - group[time_col].min()
        if dist_col in df.columns:
            df.loc[idx, "lap_dist"] = group[dist_col] - group[dist_col].min()
        for col in cumulative_cols:
            if col in df.columns:
                df.loc[idx, f"lap_{col}"] = group[col] - group[col].iloc[0]
    return df


def ensure_laps(df: pd.DataFrame, lines: Mapping[str, LineCoords] | None = None) -> pd.DataFrame:
    """Idempotent entry point used by the rest of the pipeline: if the
    DataFrame already has lap bookkeeping (processed exports), leave it
    alone; otherwise segment it from GPS + line coordinates (raw OBC logs)."""
    if "lap_lap" in df.columns and df["lap_lap"].notna().any():
        return df
    if lines is None:
        raise ValueError(
            "This telemetry has no pre-computed lap columns and no track "
            "start/finish/lap line coordinates were provided to segment it."
        )
    df = segment_laps(df, lines)
    df = trim_to_race_window(df)
    cumulative_candidates = [
        c for c in ("jm3_netjoule", "lfm_integratedcorrflow", "gfs_netgasvolume") if c in df.columns
    ]
    df = add_lap_relative_columns(df, cumulative_cols=tuple(cumulative_candidates))
    return df
