"""
sem_racing.features
====================
Derived-channel calculations lifted out of ``JupyterShowcase_BE.ipynb``:
instantaneous power, acceleration, rolling-window consumption. Written as
vectorised, reusable functions instead of one-off notebook cells so the same
logic can run identically in a live dashboard and in offline model training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_instantaneous_power(
    df: pd.DataFrame,
    voltage_col: str = "jm3_voltage",
    current_col: str = "jm3_current",
    out_col: str = "power_w",
) -> pd.DataFrame:
    """Instantaneous power (W) from millivolt/milliamp joulemeter channels."""
    df = df.copy()
    if voltage_col in df.columns and current_col in df.columns:
        df[out_col] = (df[voltage_col] / 1000.0) * (df[current_col] / 1000.0)
    return df


def add_windowed_consumption(
    df: pd.DataFrame,
    time_col: str = "obc_timestamp",
    energy_col: str = "jm3_netjoule",
    window_len: int = 10,
    out_col: str = "consumption_w",
) -> pd.DataFrame:
    """Average power over a rolling window of samples, computed from the
    difference of the accumulated-joules counter (more robust to sensor
    noise than the raw instantaneous reading for strategy purposes)."""
    df = df.copy()
    if energy_col not in df.columns or time_col not in df.columns:
        return df
    dE = df[energy_col].diff(window_len)
    dt = df[time_col].diff(window_len)
    df[out_col] = (dE / dt).replace([np.inf, -np.inf], np.nan)
    return df


def add_acceleration(
    df: pd.DataFrame,
    speed_col: str = "gps_speed",
    time_col: str = "obc_timestamp",
    out_col: str = "acceleration",
    speed_is_kmh: bool = True,
) -> pd.DataFrame:
    """Acceleration (m/s^2) from consecutive speed samples."""
    df = df.copy()
    speed_mps = df[speed_col] / 3.6 if speed_is_kmh else df[speed_col]
    dv = speed_mps.diff()
    dt = df[time_col].diff()
    df[out_col] = (dv / dt).replace([np.inf, -np.inf], np.nan)
    return df


def add_moving_averages(
    df: pd.DataFrame, columns: tuple[str, ...], window: int = 10
) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f"{col}_ma{window}"] = df[col].rolling(window, min_periods=1).mean()
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: apply the standard feature set used everywhere downstream
    (dashboard plots, ML training set)."""
    df = add_instantaneous_power(df)
    df = add_windowed_consumption(df)
    df = add_acceleration(df)
    df = add_moving_averages(df, columns=("acceleration", "consumption_w"))
    return df
