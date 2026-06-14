"""
build_dataset.py
================

Builds the BWSC/Sasol weather dataset from scratch:

  KMZ road polyline  →  60 road-distance-uniform checkpoints
                     →  hourly Open-Meteo ERA5 archive (2017-2025 race windows)
                     →  feature engineering (crosswind, headwind, temporal)
                     →  Weather Prediction/dataset.parquet

Self-contained — only requires `pandas numpy requests pyarrow tenacity tqdm`.

Usage
-----
    python "Weather Prediction/build_dataset.py" \
        --kmz "route/Directions from Darwin NT, Australia to Adelaide SA, Australia.kmz" \
        --out "Weather Prediction/dataset.parquet"

The Open-Meteo cache (~21 MB) is written next to the output as `*.openmeteo_cache.parquet`
so re-runs are instant.
"""
from __future__ import annotations

import argparse
import math
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_KM = 6371.0
N_CHECKPOINTS = 60

# 10 official Sasol Solar Challenge / BWSC control stops (Darwin -> Adelaide).
# Source: https://www.jusolarteam.se/challenge
CONTROL_STOPS: list[tuple[str, float, float]] = [
    ("Darwin",        -12.463732830749025, 130.8444325220543),
    ("Katherine",     -14.451966176114073, 132.26984885979022),
    ("Tennant Creek", -19.645930990244036, 134.1909741394222),
    ("Barrow Creek",  -21.531399876248777, 133.8889459393273),
    ("Alice Springs", -23.697812025639,    133.88055019451048),
    ("Erldunda",      -25.197599865195613, 133.2009779645565),
    ("Coober Pedy",   -29.02152578398911,  134.75685866910896),
    ("Glendambo",     -30.968402753694704, 135.7493552304439),
    ("Port Augusta",  -32.49527644368528,  137.77123220750624),
    ("Adelaide",      -34.92862942314289,  138.59985690123526),
]

# Race window each year: Aug 15 -> Oct 31 covers both BWSC (Aug-Oct windows
# historically) and Sasol Solar Challenge (mid-September Australian rounds).
RACE_YEARS = list(range(2017, 2026))
RACE_WINDOW = ("08-15", "10-31")

OPEN_METEO_HOURLY = (
    "temperature_2m,precipitation,wind_speed_10m,wind_direction_10m,"
    "shortwave_radiation,direct_normal_irradiance,diffuse_radiation"
)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Spherical initial bearing in degrees [0, 360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


# ---------------------------------------------------------------------------
# KMZ parsing  (adapted from DS-Sasol-UDST/SSC_track_parser.ipynb)
# ---------------------------------------------------------------------------

def load_kmz_polyline(kmz_path: Path) -> np.ndarray:
    """Return the road polyline as an (N, 2) array of (lat, lon).

    The KMZ holds one big LineString with ~15k vertices along the actual
    Stuart Highway from Darwin to Adelaide.
    """
    with zipfile.ZipFile(kmz_path) as zf:
        kml_name = next(n for n in zf.namelist() if n.endswith(".kml"))
        kml_text = zf.read(kml_name).decode("utf-8")

    root = ET.fromstring(kml_text)
    ns = root.tag[: root.tag.find("}") + 1] if root.tag.startswith("{") else ""

    longest: list[tuple[float, float]] = []
    for ls in root.iter(f"{ns}LineString"):
        coord_el = ls.find(f"{ns}coordinates")
        if coord_el is None or not coord_el.text:
            continue
        pts: list[tuple[float, float]] = []
        for part in coord_el.text.split():
            pieces = part.split(",")
            if len(pieces) >= 2:
                lon, lat = float(pieces[0]), float(pieces[1])
                pts.append((lat, lon))
        if len(pts) > len(longest):
            longest = pts

    if not longest:
        raise RuntimeError(f"No LineString found in {kmz_path}")
    return np.array(longest, dtype=float)


# ---------------------------------------------------------------------------
# Checkpoint placement along the road polyline
# ---------------------------------------------------------------------------

@dataclass
class Checkpoint:
    idx: int
    km_from_darwin: float
    lat: float
    lon: float
    road_bearing_deg: float
    nearest_stop: str


def _cumulative_km(poly: np.ndarray) -> np.ndarray:
    """Per-vertex cumulative road distance from the start of the polyline."""
    n = len(poly)
    cum = np.zeros(n)
    for i in range(1, n):
        cum[i] = cum[i - 1] + haversine_km(poly[i - 1, 0], poly[i - 1, 1],
                                             poly[i,     0], poly[i,     1])
    return cum


def _snap_stops_to_polyline(poly: np.ndarray, cum_km: np.ndarray) -> list[tuple[str, float]]:
    """For each control stop, return (name, km_from_darwin) at its nearest polyline vertex."""
    snapped: list[tuple[str, float]] = []
    for name, lat, lon in CONTROL_STOPS:
        # Brute-force nearest neighbour; polyline is only ~15k points so this is fine.
        d = np.array([haversine_km(lat, lon, p[0], p[1]) for p in poly])
        idx = int(d.argmin())
        snapped.append((name, float(cum_km[idx])))
    return snapped


def _local_bearing(poly: np.ndarray, vertex_idx: int) -> float:
    """Bearing of the polyline tangent at vertex_idx (uses neighbouring vertex)."""
    i = max(0, min(len(poly) - 2, vertex_idx))
    return initial_bearing_deg(poly[i, 0], poly[i, 1], poly[i + 1, 0], poly[i + 1, 1])


def build_checkpoints(
    poly: np.ndarray, n: int = N_CHECKPOINTS
) -> tuple[list[Checkpoint], list[tuple[str, float]]]:
    cum_km = _cumulative_km(poly)
    total_km = float(cum_km[-1])
    snapped = _snap_stops_to_polyline(poly, cum_km)
    stop_km = np.array([s[1] for s in snapped])
    stop_names = [s[0] for s in snapped]

    checkpoints: list[Checkpoint] = []
    for i in range(n):
        target_km = total_km * (i / (n - 1))
        # Find polyline segment containing target_km.
        vi = int(np.searchsorted(cum_km, target_km, side="right") - 1)
        vi = max(0, min(len(poly) - 2, vi))
        seg_a_km, seg_b_km = cum_km[vi], cum_km[vi + 1]
        seg_len = seg_b_km - seg_a_km
        t = 0.0 if seg_len == 0 else (target_km - seg_a_km) / seg_len
        lat = poly[vi, 0] + t * (poly[vi + 1, 0] - poly[vi, 0])
        lon = poly[vi, 1] + t * (poly[vi + 1, 1] - poly[vi, 1])
        bearing = _local_bearing(poly, vi)
        nearest = stop_names[int(np.abs(stop_km - target_km).argmin())]

        checkpoints.append(Checkpoint(
            idx=i,
            km_from_darwin=round(target_km, 2),
            lat=round(float(lat), 5),
            lon=round(float(lon), 5),
            road_bearing_deg=round(float(bearing), 2),
            nearest_stop=nearest,
        ))

    print(f"  Total road length (KMZ): {total_km:.1f} km")
    print(f"  Snapped control stops:")
    for name, km in snapped:
        print(f"    {name:15s} @ km {km:7.1f}")
    return checkpoints, snapped


# ---------------------------------------------------------------------------
# Open-Meteo client
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=5, min=10, max=300))
def _get_json(url: str, params: dict) -> dict:
    """GET with patient back-off. Open-Meteo's free tier sometimes returns 429 even
    when daily quota is not exhausted; waiting 1-5 minutes between retries clears it.
    """
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


def _hourly_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if not hourly:
        return pd.DataFrame()
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_archive_range(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """One Open-Meteo call covering the entire date range."""
    payload = _get_json(ARCHIVE_URL, {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     OPEN_METEO_HOURLY,
        "timezone":   "Australia/Darwin",
    })
    df = _hourly_to_df(payload)
    if not df.empty:
        df["lat"] = lat
        df["lon"] = lon
    return df


def _filter_race_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows whose date falls within Aug 15 - Oct 31 of any race year."""
    if df.empty:
        return df
    t = df["time"].dt
    mm_dd = t.month * 100 + t.day  # e.g. 0815, 1031
    start = int(RACE_WINDOW[0].replace("-", ""))  # 815
    end   = int(RACE_WINDOW[1].replace("-", ""))  # 1031
    year_ok = t.year.isin(RACE_YEARS)
    window_ok = (mm_dd >= start) & (mm_dd <= end)
    out = df[year_ok & window_ok].copy()
    out["year"] = out["time"].dt.year
    return out


def fetch_all(checkpoints: list[Checkpoint], cache_path: Path) -> pd.DataFrame:
    """One Open-Meteo call per checkpoint covering the entire 2017-2025 window.

    The API returns continuous hourly data across the full range; we filter
    down to the Aug 15 – Oct 31 race windows client-side.

    Progress is checkpointed per-checkpoint to `*_per_checkpoint/`, so partial
    runs aren't lost if the API rate-limits us. Re-running picks up where we
    left off automatically.
    """
    if cache_path.exists():
        print(f"  Loading cached Open-Meteo pull → {cache_path}")
        return pd.read_parquet(cache_path)

    per_cp_dir = cache_path.parent / (cache_path.stem + "_per_checkpoint")
    per_cp_dir.mkdir(parents=True, exist_ok=True)

    start_date = f"{min(RACE_YEARS)}-{RACE_WINDOW[0]}"
    end_date   = f"{max(RACE_YEARS)}-{RACE_WINDOW[1]}"
    already_done = sum(1 for cp in checkpoints if (per_cp_dir / f"cp_{cp.idx:02d}.parquet").exists())
    print(f"  Fetching Open-Meteo: {len(checkpoints)} checkpoints × 1 range "
          f"({start_date} → {end_date})  [{already_done} already cached]")

    frames: list[pd.DataFrame] = []
    for cp in tqdm(checkpoints, desc="checkpoints"):
        cp_file = per_cp_dir / f"cp_{cp.idx:02d}.parquet"
        if cp_file.exists():
            df = pd.read_parquet(cp_file)
        else:
            df = fetch_archive_range(cp.lat, cp.lon, start_date, end_date)
            df = _filter_race_windows(df)
            if not df.empty:
                df["checkpoint_idx"]    = cp.idx
                df["km_from_darwin"]    = cp.km_from_darwin
                df["road_bearing_deg"]  = cp.road_bearing_deg
                df["nearest_stop"]      = cp.nearest_stop
                df.to_parquet(cp_file, index=False)
            time.sleep(0.5)  # be polite to the free tier
        if not df.empty:
            frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(cache_path, index=False)
    print(f"  Cached → {cache_path}  ({len(raw):,} rows)")
    return raw


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _to_utc(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s, errors="coerce")
    if getattr(t.dt, "tz", None) is None:
        # Open-Meteo returned Australia/Darwin local — convert to UTC.
        t = t.dt.tz_localize("Australia/Darwin").dt.tz_convert("UTC")
    else:
        t = t.dt.tz_convert("UTC")
    return t


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time in UTC.
    df["time"] = _to_utc(df["time"])

    # Crosswind / headwind from wind + road bearing.
    theta_rad = np.radians(df["wind_direction_10m"].astype(float) - df["road_bearing_deg"].astype(float))
    df["crosswind_ms"] = df["wind_speed_10m"].astype(float) * np.abs(np.sin(theta_rad))
    df["headwind_ms"]  = df["wind_speed_10m"].astype(float) * np.cos(theta_rad)

    # Temporal features (in Darwin local time, matching the original convention).
    local = df["time"].dt.tz_convert("Australia/Darwin")
    df["hour"]      = local.dt.hour + local.dt.minute / 60.0
    df["dayofyear"] = local.dt.dayofyear.astype("int32")
    df["month"]     = local.dt.month.astype("int32")

    # "Day of race" — index within the race window for that year.
    start_dates = {y: pd.Timestamp(f"{y}-{RACE_WINDOW[0]}", tz="Australia/Darwin") for y in RACE_YEARS}
    starts = local.dt.year.map(start_dates).astype("datetime64[ns, Australia/Darwin]")
    df["day_of_race"] = ((local - starts).dt.days + 1).astype("int32")

    # Solar elevation (closed-form NOAA approximation).
    df["solar_elevation_deg"] = _solar_elevation(df["time"], df["lat"], df["lon"])

    return df


def add_segment_features(df: pd.DataFrame, snapped_stops: list[tuple[str, float]]) -> pd.DataFrame:
    """Add 1-based segment_idx (1..9) and km_to_next_stop using the 10 control stops."""
    stop_km = np.array(sorted(s[1] for s in snapped_stops), dtype=float)
    n_segments = len(stop_km) - 1  # 9 for the canonical 10 stops

    # segment_idx: 1..n_segments. pd.cut returns NaN for values outside the bins;
    # include_lowest=True ensures km=0 (Darwin) lands in segment 1.
    df["segment_idx"] = pd.cut(
        df["km_from_darwin"], bins=stop_km, labels=list(range(1, n_segments + 1)),
        include_lowest=True,
    ).astype("Int8")

    # km_to_next_stop: distance to the next control stop strictly ahead. Clamped
    # to 0 at/past Adelaide.
    km_arr = df["km_from_darwin"].to_numpy(dtype=float)
    idx = np.searchsorted(stop_km, km_arr, side="right")
    idx_clamped = np.minimum(idx, len(stop_km) - 1)
    df["km_to_next_stop"] = np.where(idx < len(stop_km),
                                       stop_km[idx_clamped] - km_arr, 0.0)
    return df


def _solar_elevation(time_utc: pd.Series, lat: pd.Series, lon: pd.Series) -> np.ndarray:
    t = pd.to_datetime(time_utc, utc=True)
    day = t.dt.dayofyear.to_numpy(dtype=float)
    hour_utc = (t.dt.hour + t.dt.minute / 60.0 + t.dt.second / 3600.0).to_numpy(dtype=float)
    decl = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284.0 + day)))
    # Hour angle in degrees, relative to solar noon at the given longitude.
    hour_angle = 15.0 * (hour_utc - 12.0) + lon.to_numpy(dtype=float)
    lat_rad = np.radians(lat.to_numpy(dtype=float))
    decl_rad = np.radians(decl)
    ha_rad = np.radians(hour_angle)
    sin_alt = (np.sin(lat_rad) * np.sin(decl_rad)
               + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(ha_rad))
    return np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def build(kmz: Path, out: Path) -> Path:
    print(f"[1/4] Loading KMZ road polyline: {kmz}")
    poly = load_kmz_polyline(kmz)
    print(f"  Polyline vertices: {len(poly):,}")

    print(f"[2/4] Building {N_CHECKPOINTS} road-uniform checkpoints")
    cps, snapped_stops = build_checkpoints(poly, n=N_CHECKPOINTS)

    cache_path = out.with_suffix("")  # strip .parquet
    cache_path = cache_path.parent / (cache_path.name + ".openmeteo_cache.parquet")
    print(f"[3/4] Fetching Open-Meteo archive")
    raw = fetch_all(cps, cache_path)

    print(f"[4/4] Engineering features and writing parquet")
    df = add_features(raw)
    df = add_segment_features(df, snapped_stops)
    df = df.sort_values(["checkpoint_idx", "time"]).reset_index(drop=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"  Wrote {out}  ({len(df):,} rows, {df.shape[1]} cols)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kmz",
                    default="route/Directions from Darwin NT, Australia to Adelaide SA, Australia.kmz",
                    help="Path to the BWSC/Sasol route KMZ file")
    ap.add_argument("--out",
                    default="Weather Prediction/dataset.parquet",
                    help="Output parquet path")
    args = ap.parse_args()
    build(Path(args.kmz), Path(args.out))


if __name__ == "__main__":
    main()
