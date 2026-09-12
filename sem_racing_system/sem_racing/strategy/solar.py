"""
sem_racing.solar
=================
The piece that was completely missing from the original three bootcamp
notebooks and from this system until now: a model of energy *coming in*
from the panels, not just energy going out through the motor.

Two things this deliberately does NOT do, stated up front rather than
buried in a comment:

1. It does not know your race's actual irradiance -- you supply that, as
   either a constant (quick "what if it's overcast all day" check) or a
   distance-indexed profile (``IrradianceProfile``). Nothing here forecasts
   weather.
2. It approximates time-of-day effects via *distance into the stage*, not
   true elapsed time. A rigorous version would need the optimizer's state
   to include elapsed time (arrival time depends on the very speed profile
   you're solving for -- a circular dependency), which would blow up the
   graph from a 2D (distance, speed) state space to 3D. That's a real
   limitation, not an oversight -- see optimize.py's docstring for the
   practical workaround this system uses instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class ArraySpec:
    """Your solar array's rated output at Standard Test Conditions (STC:
    1000 W/m^2, 25 degC cell temperature). Get ``rated_power_w`` from your
    own array wiring calculation (e.g. the club's array-connections
    diagram), not from multiplying a single cell's datasheet Pmax by cell
    count -- real wiring losses, mismatch, and derating already show up in
    a properly computed rated figure. IMPORTANT: this number is only valid
    for whatever cell count/array size you've actually finalised -- if the
    array gets resized to fit a class limit (see the regulatory note this
    system flagged earlier), recompute this before trusting anything
    downstream of it.
    """

    rated_power_w: float
    temp_coeff_per_c: float = -0.0035  # ~ -0.35%/degC above 25degC, typical for silicon PV


def available_power(array: ArraySpec, irradiance_w_m2: float, cell_temp_c: float = 25.0) -> float:
    """Instantaneous array output (W) at a given irradiance and cell
    temperature. Linear in irradiance (a good approximation near STC; real
    cells are slightly non-linear at very low light, ignored here) and
    derated for temperature above 25 degC -- cells lose efficiency as they
    heat up, which matters on a black car body in direct sun far more than
    it matters on a rooftop panel."""
    irradiance_ratio = max(irradiance_w_m2, 0.0) / 1000.0
    temp_derate = 1.0 + array.temp_coeff_per_c * (cell_temp_c - 25.0)
    return array.rated_power_w * irradiance_ratio * max(temp_derate, 0.0)


class IrradianceProfile:
    """Maps distance-into-stage (m) to expected irradiance (W/m^2). Build
    this from a weather forecast + your planned average pace (turn a
    time-of-day forecast into a distance-of-stage forecast once, before
    optimizing), or from onboard pyranometer logs of a previous attempt at
    the same stage."""

    def __init__(self, distances_m: list[float], irradiance_w_m2: list[float]):
        self._d = np.asarray(distances_m, dtype=float)
        self._irr = np.asarray(irradiance_w_m2, dtype=float)
        if len(self._d) != len(self._irr) or len(self._d) < 1:
            raise ValueError("distances_m and irradiance_w_m2 must be equal-length, non-empty")

    @classmethod
    def constant(cls, irradiance_w_m2: float, stage_length_m: float = 1.0):
        """Quick sanity-check profile: same irradiance everywhere (e.g.
        "what if it's overcast, 200 W/m^2, all day")."""
        return cls([0.0, stage_length_m], [irradiance_w_m2, irradiance_w_m2])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, dist_col: str = "dist_m", irr_col: str = "irradiance_w_m2"):
        return cls(df[dist_col].tolist(), df[irr_col].tolist())

    def at(self, distance_m: float) -> float:
        return float(np.interp(distance_m, self._d, self._irr))


def energy_income(
    array: ArraySpec,
    irradiance: IrradianceProfile,
    distance_m: float,
    time_s: float,
    cell_temp_c: float = 25.0,
) -> float:
    """Energy (J) collected while covering one section, sampled at the
    section's start distance. Slower sections (bigger ``time_s`` for the
    same ``distance_m``) collect more -- this is the real strategic
    tension a solar car has that a plain EV doesn't: easing off doesn't
    just cut consumption, it also buys more time under the sun."""
    power_w = available_power(array, irradiance.at(distance_m), cell_temp_c)
    return power_w * time_s
