"""
sem_racing.testing.coastdown
==============================
Answers "After-testing analysis: coastdown, regression analysis" directly,
and is the thing that turns ``CarConstants.coeff_roll_res``/``coeff_aero_drag``
from supplied numbers into *measured* ones.

The physics, following the same road-load model as SAE J2263 (the real
industry-standard coastdown test procedure -- checked against it rather
than assumed): during a coastdown (motor disengaged, no braking, flat
road), deceleration force has three additive terms, not two:

    F(v) = A + B*v + C*v^2
    => a(v) = A0 + A1*v + A2*v^2

A0 (constant) comes from rolling resistance: A0 = g*Crr.
A2 (quadratic) comes from aerodynamic drag: A2 = rho*A*Cd / (2m).
A1 (linear) captures velocity-proportional losses -- bearing/drivetrain
friction -- that J2263 includes but which this system's motor models
(carmodel.py) don't currently have a term for. It's fit and reported for
completeness/diagnostic value, but NOT applied to CarConstants below,
since there's nowhere in the current force-balance equations for it to
go. An earlier version of this module used a simpler 2-term fit (A0 + A2*v^2
only) -- upgraded to 3-term after checking against the actual SAE
methodology rather than assuming the simpler form was sufficient.

This is real physics, but it has real preconditions this module can't
verify for you: truly flat road (any gradient contaminates A0), no wind
(contaminates A2 -- see strategy/wind.py's note on wind not being modeled
elsewhere either; a coastdown test done on a windy day bakes wind into
your "aerodynamic drag" estimate -- SAE J2263 handles this with real-time
anemometry, which this module does not), and a genuinely disengaged
drivetrain. Check ``r_squared`` before trusting a fit -- a poor fit usually
means one of these preconditions was violated, not a code bug.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from ..strategy.carmodel import CarConstants


def smoothed_deceleration(
    df: pd.DataFrame,
    speed_col: str = "gps_speed",
    time_col: str = "obc_timestamp",
    smoothing_window: int = 5,
    min_speed_ms: float = 1.0,
):
    """Shared by ``analyze_coastdown`` (fit) and
    ``testing.validation.validate_resistance`` (predict-and-compare on a
    held-out run) -- one smoothing/differencing implementation, not two
    copies that could quietly drift apart. Returns (v_valid_ms, a_valid_ms2)
    -- speed and deceleration magnitude at each valid (post-smoothing,
    genuinely decelerating) sample.

    ``min_periods=smoothing_window`` (not 1) deliberately produces NaN at
    the array's edges rather than a truncated, biased average there -- a
    centered rolling window with a partial edge sample systematically
    under/over-estimates the true local slope right where boundary
    artifacts matter most for a regression fit. Those edge points are
    dropped rather than kept and allowed to corrupt the result.
    """
    v = df[speed_col].to_numpy(dtype=float) / 3.6  # km/h -> m/s
    t = df[time_col].to_numpy(dtype=float)

    v_smooth_full = pd.Series(v).rolling(smoothing_window, center=True, min_periods=smoothing_window).mean().to_numpy()
    valid = ~np.isnan(v_smooth_full)
    if valid.sum() < 2:
        return np.array([]), np.array([])  # not enough points to even attempt a gradient
    v_smooth = v_smooth_full[valid]
    t_valid = t[valid]
    dv_dt = np.gradient(v_smooth, t_valid)

    mask = (dv_dt < 0) & (v_smooth > min_speed_ms)
    return v_smooth[mask], -dv_dt[mask]  # deceleration magnitude, positive


@dataclass
class CoastdownResult:
    a0: float                      # constant term (m/s^2) -- rolling resistance
    a1: float                      # linear-in-v term (1/s) -- velocity-proportional losses (diagnostic only)
    a2: float                      # v^2 coefficient (1/m) -- aerodynamic drag
    coeff_roll_res: float          # Crr = a0 / g
    drag_area_m2: float            # Cd*A (m^2) -- always computable, doesn't need frontal area
    coeff_aero_drag: float | None  # Cd alone -- only if frontal_area_m2 was supplied
    r_squared: float               # fit quality -- LOW VALUE HERE MEANS DON'T TRUST THIS RESULT
    n_points: int                  # how many (speed, deceleration) samples went into the fit


def analyze_coastdown(
    df: pd.DataFrame,
    mass_kg: float,
    frontal_area_m2: float | None = None,
    speed_col: str = "gps_speed",
    time_col: str = "obc_timestamp",
    rho_air: float = 1.225,
    g: float = 9.81,
    smoothing_window: int = 5,
    min_speed_ms: float = 1.0,
) -> CoastdownResult:
    """Fit ``CoastdownResult`` from one logged coastdown run, via the same
    3-term road-load model SAE J2263 uses. ``df`` should be *just* the
    coastdown segment (motor off, braking off, from top speed down to
    near-stop) -- trim the log to that window before calling this; it
    doesn't detect the coastdown window for you.
    """
    v_fit, a_fit = smoothed_deceleration(df, speed_col, time_col, smoothing_window, min_speed_ms)
    if len(v_fit) < 10:
        raise ValueError(
            f"Only {len(v_fit)} valid decelerating samples found -- need a clean, "
            "clearly-decelerating coastdown segment, not a mixed driving log."
        )

    X = np.column_stack([np.ones_like(v_fit), v_fit, v_fit ** 2])
    coeffs, _, _, _ = np.linalg.lstsq(X, a_fit, rcond=None)
    a0, a1, a2 = coeffs

    predicted = X @ coeffs
    ss_res = np.sum((a_fit - predicted) ** 2)
    ss_tot = np.sum((a_fit - a_fit.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    coeff_roll_res = a0 / g
    drag_area_m2 = a2 * 2 * mass_kg / rho_air
    coeff_aero_drag = drag_area_m2 / frontal_area_m2 if frontal_area_m2 else None

    return CoastdownResult(
        a0=float(a0), a1=float(a1), a2=float(a2), coeff_roll_res=float(coeff_roll_res),
        drag_area_m2=float(drag_area_m2), coeff_aero_drag=coeff_aero_drag,
        r_squared=float(r_squared), n_points=int(len(v_fit)),
    )


def apply_to_car_constants(result: CoastdownResult, base: CarConstants, frontal_area_m2: float | None = None) -> CarConstants:
    """Return a new ``CarConstants`` with ``coeff_roll_res``/``coeff_aero_drag``
    replaced by the measured values -- everything else in ``base`` untouched.
    Refuses to silently apply a bad fit. Note ``result.a1`` (velocity-linear
    losses) is NOT applied here -- CarConstants has no field for it; it's
    folded into the residual error of every prediction until the force
    equations in carmodel.py are extended to include it."""
    if result.r_squared < 0.8:
        raise ValueError(
            f"r_squared={result.r_squared:.2f} is too low to trust -- check for wind, "
            "road gradient, or residual motor/brake drag during the test before applying this."
        )
    updates = {"coeff_roll_res": result.coeff_roll_res}
    if result.coeff_aero_drag is not None:
        updates["coeff_aero_drag"] = result.coeff_aero_drag
    elif frontal_area_m2:
        updates["coeff_aero_drag"] = result.drag_area_m2 / frontal_area_m2
    return replace(base, **updates)
