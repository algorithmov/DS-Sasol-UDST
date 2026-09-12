"""
sem_racing.route
=================
``laps.py`` assumes a repeating circuit: it detects start/finish/lap line
*crossings*, which only makes sense for a car that comes back around. The
Sasol Solar Challenge (and most real endurance events) is point-to-point --
one stage, ~250-300km, driven once, no lap line to cross a second time.
Reusing lap-crossing logic on a point-to-point stage would silently produce
nonsense (it would just never detect a second crossing and call it one
giant "lap"). This module is the honest, purpose-built alternative.

Two independent things a route gives you here:

1. ``add_stage_distance`` -- turn a point-to-point telemetry log into a
   distance-into-stage column, no crossing detection needed (there's only
   one continuous run).
2. ``build_stage_config`` -- turn a *planned* route (a sequence of
   lat/lon[,elevation] points -- from a GPX-converted CSV, a hand-typed
   waypoint list, whatever) into the same (s, vmax, vmin, m_above_sea)
   shape ``optimize.py`` consumes, using route geometry alone. This is
   what makes "I found the route tomorrow, can I just plug it in" mostly
   true even with zero historical telemetry: corner speed limits are
   estimated from the route's own curvature instead of requiring real laps
   first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .laps import add_cumulative_distance


def load_route(path: str | pd.DataFrame) -> pd.DataFrame:
    """Load a planned route as an ordered sequence of points. Deliberately
    format-agnostic about *how* you got the points -- a GPX track exported
    to CSV, a hand-typed waypoint list, a Google Maps route dumped to
    coordinates -- as long as the file has, in order, ``lat``/``lon``
    columns (and optionally ``elevation_m``). Order in the file IS the
    route order; nothing here re-sorts by anything.
    """
    df = path if isinstance(path, pd.DataFrame) else pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"lat", "lon"}
    if not required.issubset(df.columns):
        raise ValueError(f"Route file needs at least {required}, got {list(df.columns)}")
    return df.reset_index(drop=True)


def add_stage_distance(
    df: pd.DataFrame, lat_col: str = "gps_latitude", lon_col: str = "gps_longitude"
) -> pd.DataFrame:
    """Distance-into-stage for a point-to-point drive: just the running
    haversine total along the vehicle's own GPS trace, no start/finish/lap
    crossing detection. Use this instead of ``laps.ensure_laps`` for
    telemetry from a point-to-point stage (one continuous run, not a
    repeating circuit)."""
    df = add_cumulative_distance(df, lat_col, lon_col)
    return df.rename(columns={"dist": "stage_dist"})


def _three_point_radius(p0, p1, p2) -> float:
    """Circumradius (m, planar approximation -- fine at road-corner scale)
    of the circle through three consecutive route points. Large/undefined
    radius = a straight; small radius = a tight corner."""
    a = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
    b = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
    c = np.hypot(p2[0] - p0[0], p2[1] - p0[1])
    area2 = abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
    if area2 < 1e-9 or a < 1e-9 or b < 1e-9 or c < 1e-9:
        return np.inf  # collinear (or duplicate) points -- treat as a straight
    return (a * b * c) / (2 * area2)


def local_xy_projection(route: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon"):
    """Equirectangular local projection (good enough at route scale) so
    planar geometry in metres can be used instead of lat/lon degrees.
    Shared by ``estimate_corner_speed_limits`` and, cross-pillar, by
    ``strategy.wind``'s heading calculation -- one projection, not two
    slightly-different copies of the same approximation."""
    lat0 = np.radians(route[lat_col].mean())
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(lat0)
    x = route[lon_col].to_numpy() * m_per_deg_lon
    y = route[lat_col].to_numpy() * m_per_deg_lat
    return x, y


def compute_headings_deg(route: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> np.ndarray:
    """Compass heading (degrees, 0=North/+y, 90=East/+x) of travel at each
    route point, from the vector to the next point (last point repeats the
    previous heading). Used by ``strategy.wind`` to resolve a wind vector
    into a headwind/tailwind component that varies as the road turns."""
    x, y = local_xy_projection(route, lat_col, lon_col)
    dx = np.diff(x, append=x[-1])
    dy = np.diff(y, append=y[-1])
    headings = np.degrees(np.arctan2(dx, dy)) % 360.0
    if len(headings) > 1:
        headings[-1] = headings[-2]
    return headings


def estimate_corner_speed_limits(
    route: pd.DataFrame,
    friction_coeff: float = 0.6,
    top_speed_kmh: float = 90.0,
    lat_col: str = "lat",
    lon_col: str = "lon",
    g: float = 9.81,
) -> pd.Series:
    """Per-point safe cornering speed from route curvature alone, via the
    standard v_max = sqrt(mu * g * R) relation (R = local turn radius, mu =
    an assumed tyre/road friction coefficient -- 0.6 is a conservative dry
    tarmac default, lower it for an openly cautious plan). Straights (large
    R) are capped at ``top_speed_kmh`` rather than left unbounded.

    This is a real physics estimate, not a guess -- but it only accounts
    for lateral grip in a corner. It does NOT know about speed limits,
    traffic, road surface quality, or visibility, all of which matter more
    than physics on public roads. Treat this as an upper bound to plan
    against, not a target."""
    x, y = local_xy_projection(route, lat_col, lon_col)

    n = len(route)
    v_limit_ms = np.full(n, top_speed_kmh / 3.6)
    for i in range(1, n - 1):
        r = _three_point_radius((x[i - 1], y[i - 1]), (x[i], y[i]), (x[i + 1], y[i + 1]))
        if np.isfinite(r):
            v_limit_ms[i] = min(np.sqrt(friction_coeff * g * r), top_speed_kmh / 3.6)
    return pd.Series(v_limit_ms * 3.6, index=route.index, name="corner_vmax_kmh")


def build_stage_config(
    route: pd.DataFrame,
    section_length: float = 100.0,
    friction_coeff: float = 0.6,
    top_speed_kmh: float = 90.0,
    vmin_fraction: float = 0.4,
    standing_start: bool = True,
) -> pd.DataFrame:
    """Build an ``optimize.py``-ready config purely from a planned route's
    geometry -- no historical telemetry required. This is the "found the
    route, plug it in" path: distance and corner-speed limits come from
    the route itself; elevation comes along for free if the route file has
    an ``elevation_m`` column.

    Once you've actually driven the stage, prefer
    ``trackmodel.build_track_config()`` on the real telemetry instead --
    real data beats a curvature heuristic. Use this to get a first plan
    before that data exists, per the earlier conversation about pre-season
    planning on a route with zero logged laps.
    """
    route = load_route(route) if not isinstance(route, pd.DataFrame) else route
    route = add_cumulative_distance(route, lat_col="lat", lon_col="lon").rename(columns={"dist": "stage_dist"})

    corner_vmax = estimate_corner_speed_limits(route, friction_coeff, top_speed_kmh)

    max_dist = float(np.ceil(route["stage_dist"].max() / section_length) * section_length)
    bins = np.arange(0, max_dist + section_length, section_length)
    route = route.copy()
    route["_bin"] = pd.cut(route["stage_dist"], bins=bins, right=False, labels=bins[:-1])

    vmax = corner_vmax.groupby(route["_bin"], observed=True).min()
    config = pd.DataFrame({"s": vmax.index.astype(float), "vmax": vmax.values})
    config = config.sort_values("s").reset_index(drop=True)
    config["vmin"] = config["vmax"] * vmin_fraction

    if "elevation_m" in route.columns:
        elev = route.groupby("_bin", observed=True)["elevation_m"].mean()
        config["m_above_sea"] = elev.reindex(config["s"]).values
    else:
        config["m_above_sea"] = 0.0

    config = config.ffill().bfill()

    if standing_start:
        config.loc[0, ["vmax", "vmin"]] = 0.0
        config.loc[config.index[-1], ["vmax", "vmin"]] = 0.0

    return config
