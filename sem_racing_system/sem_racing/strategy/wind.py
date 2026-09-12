"""
sem_racing.strategy.wind
=========================
Before this module: wind did not exist anywhere in this system. Every
aerodynamic drag calculation used ground speed alone, silently assuming a
perfectly still day, always. That's wrong for "route/weather optimisation"
specifically -- a Sasol-style route changes heading constantly (it's a real
road, not a wind-tunnel straight), so the *same* wind forecast produces a
headwind on one section and a tailwind two kilometres later just because
the road turned. That route-dependent variation is real strategic
information ("push harder on the headwind-into-tailwind swap, ease off
where it reverses") and this module is what makes it computable.

Deliberately NOT modeled: crosswind's effect on drag (this treats only the
along-heading wind component; a pure crosswind is assumed to add no drag,
which is optimistic -- real crosswind still increases drag somewhat via
yaw and induced drag on the body, just less than an equivalent headwind);
gusting/turbulence (a single forecast value per section, not a
distribution); and wind that changes over the course of the day at a fixed
location (one ``WindConditions`` per stage plan, not a time-varying one --
same distance-vs-time-of-day approximation limitation already documented
in solar.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..telemetry.route import compute_headings_deg, add_cumulative_distance


@dataclass
class WindConditions:
    """A single wind estimate for a stage: speed and the compass direction
    the wind is blowing FROM (meteorological convention -- '090' means an
    easterly, blowing towards the west)."""

    speed_ms: float
    direction_from_deg: float


def headwind_component(heading_deg: float, wind: WindConditions) -> float:
    """Component of the wind along the direction of travel. Positive =
    headwind (adds to drag), negative = tailwind (reduces it). Crosswind
    contributes zero here -- see the module docstring's caveat on that."""
    # wind blows FROM direction_from_deg, i.e. travels TOWARD direction_from_deg+180.
    # The component opposing a vehicle heading in heading_deg is:
    rel = math.radians(wind.direction_from_deg - heading_deg)
    return wind.speed_ms * math.cos(rel)


class WindProfile:
    """Distance-into-stage -> headwind component (m/s), same shape and
    calling convention as ``solar.IrradianceProfile`` so ``optimize.py``
    can treat both environmental inputs symmetrically."""

    def __init__(self, distances_m: list[float], headwind_ms: list[float]):
        self._d = np.asarray(distances_m, dtype=float)
        self._hw = np.asarray(headwind_ms, dtype=float)
        if len(self._d) != len(self._hw) or len(self._d) < 1:
            raise ValueError("distances_m and headwind_ms must be equal-length, non-empty")

    @classmethod
    def calm(cls, stage_length_m: float = 1.0):
        return cls([0.0, stage_length_m], [0.0, 0.0])

    @classmethod
    def from_route_and_wind(
        cls, route: pd.DataFrame, wind: WindConditions, lat_col: str = "lat", lon_col: str = "lon"
    ) -> "WindProfile":
        """Resolve one wind forecast against the route's actual heading at
        each point -- this is what makes the same wind forecast produce a
        varying headwind/tailwind profile along a real, turning road."""
        headings = compute_headings_deg(route, lat_col, lon_col)
        staged = add_cumulative_distance(route.rename(columns={lat_col: "gps_latitude", lon_col: "gps_longitude"}),
                                          lat_col="gps_latitude", lon_col="gps_longitude")
        distances = staged["dist"].to_numpy()
        components = np.array([headwind_component(h, wind) for h in headings])
        return cls(distances.tolist(), components.tolist())

    def at(self, distance_m: float) -> float:
        return float(np.interp(distance_m, self._d, self._hw))
