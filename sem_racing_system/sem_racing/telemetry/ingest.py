"""
sem_racing.ingest
==================
Turns the two telemetry file formats the club actually produces into one
consistent, tidy DataFrame:

1. Raw on-board-computer (OBC) logs (``*.log``), semicolon separated, with a
   ~20 line config header before a ``-- DATA --`` marker.
2. Pre-processed CSV exports (``BE_sample_data*.csv``) that already contain
   ``lap_lap`` / ``start_lap`` / ``finish_lap`` bookkeeping columns.

Everything downstream (laps.py, features.py, ...) only has to deal with one
shape of DataFrame, so we do the messy format-sniffing once, here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _find_data_start(path: Path) -> int:
    """Raw OBC logs have a variable-length config header ending in a line of
    dashes (``-- DATA --------``). Detect it instead of hard-coding
    ``skiprows=21`` like the bootcamp notebook did -- header length varies
    between vehicle configs."""
    with open(path, "r", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if line.strip().startswith("--") and "DATA" in line.upper():
                return i + 1
    raise ValueError(f"Could not find '-- DATA --' marker in {path}")


def load_obc_log(path: str | Path) -> pd.DataFrame:
    """Load a raw semicolon-delimited OBC log file."""
    path = Path(path)
    header_row = _find_data_start(path)
    df = pd.read_csv(path, sep=";", skiprows=header_row, low_memory=False)
    return _clean(df)


def load_export_csv(path: str | Path) -> pd.DataFrame:
    """Load an already-processed, comma-delimited CSV export."""
    df = pd.read_csv(path, low_memory=False)
    return _clean(df)


def load(path: str | Path) -> pd.DataFrame:
    """Auto-detect format by extension and dispatch to the right loader."""
    path = Path(path)
    if path.suffix.lower() == ".log":
        return load_obc_log(path)
    return load_export_csv(path)


def load_many(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Load and concatenate multiple session exports, re-basing the
    ``lap_lap`` counter across files so lap numbers stay globally unique.

    This generalises the ad-hoc ``load_data()`` helper buried inside
    ``AI_SpeedProfileOptimisation.ipynb`` into something any pipeline stage
    can call to build a historical training set.
    """
    frames = []
    lap_offset = 0
    for p in paths:
        df = load(p)
        if "lap_lap" in df.columns:
            df["lap_lap"] = df["lap_lap"] + lap_offset
            lap_offset = int(df["lap_lap"].max()) + 1
        df["source_file"] = Path(p).name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    return out


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = "obc_timestamp" if "obc_timestamp" in df.columns else None
    if ts_col:
        df.sort_values(by=ts_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.ffill(inplace=True)
    return df
