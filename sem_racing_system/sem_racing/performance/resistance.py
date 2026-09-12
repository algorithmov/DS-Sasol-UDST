"""
sem_racing.performance.resistance
===================================
"Performance predictions: ... resistance" -- before this module, rolling
resistance and aerodynamic drag only existed buried inside a motor model's
force-balance equations, invisible as standalone numbers. This exposes them
directly: given ``CarConstants`` (ideally calibrated by
``testing.coastdown``, not left at placeholder values), predict the
resistance forces/power at any speed -- useful for a report or sanity check
completely independent of running the full optimizer.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..strategy.carmodel import CarConstants


def resistance_at_speed(
    constants: CarConstants, speed_ms: float, gradient_angle: float = 0.0, headwind_ms: float = 0.0
) -> dict:
    """Breakdown of resistance forces (N) and the power (W) needed to
    overcome them at a steady speed -- no acceleration term, this is a
    cruise-condition snapshot, not a full transition prediction (use
    ``carmodel.DirectDriveMotorModel`` for that)."""
    c = constants
    airspeed = max(speed_ms + headwind_ms, 0.0)
    f_roll = c.car_mass * c.g * math.cos(gradient_angle) * c.coeff_roll_res
    f_slope = c.car_mass * c.g * math.sin(gradient_angle)
    f_aero = 0.5 * c.rho_air * c.a_aero * c.coeff_aero_drag * airspeed ** 2
    f_total = f_roll + f_slope + f_aero
    return {
        "speed_ms": speed_ms,
        "f_rolling_n": f_roll,
        "f_gradient_n": f_slope,
        "f_aero_n": f_aero,
        "f_total_n": f_total,
        "power_to_overcome_w": f_total * speed_ms,
    }


def resistance_curve(
    constants: CarConstants, speed_range_ms: np.ndarray, gradient_angle: float = 0.0, headwind_ms: float = 0.0
) -> pd.DataFrame:
    """Resistance force/power vs. speed, for a report or a quick plot. Also
    useful as a sanity check after calibrating from a coastdown test: the
    rolling term should be flat, the aero term should visibly dominate at
    speed if ``coeff_aero_drag`` is realistic."""
    rows = [resistance_at_speed(constants, v, gradient_angle, headwind_ms) for v in speed_range_ms]
    return pd.DataFrame(rows)
