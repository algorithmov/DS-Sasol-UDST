"""
sem_racing.testing.validation
================================
This is the module that makes "digital twin" an earned label instead of a
claim: for each subsystem the rest of this codebase simulates (motor,
resistance, battery, solar array), this gives you a function that takes
your own real lab/bench/track measurements and tells you, with real error
statistics, how well the simulation matches reality. Every ``validate_*``
function here is independent -- test the motor on its own against your dyno
runs, the battery on its own against a discharge-bench log, resistance
against a coastdown you held back from calibration, the array against an
outdoor panel measurement. None of them need the others to run.

Design principle: validation is not calibration. ``testing.coastdown`` FITS
new coefficients from data. Everything in this module PREDICTS with
whatever coefficients you already have, then reports how wrong it was. If
you calibrated and validated on the *same* run, you're measuring how well
the model fits its own training data, not whether it generalizes -- use a
held-out run for anything you're calling a validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..strategy.carmodel import CarConstants
from ..strategy.solar import ArraySpec, available_power
from ..strategy.battery import BatterySpec
from .coastdown import smoothed_deceleration


@dataclass
class ValidationResult:
    subsystem: str
    n_points: int
    mae: float          # mean absolute error, in the same units as the quantity compared
    rmse: float         # root mean squared error, same units
    mape: float | None  # mean absolute percentage error (NaN if any actual value is ~0)
    r_squared: float
    bias: float          # mean(predicted - actual); positive = model over-predicts on average
    table: pd.DataFrame  # per-point predicted/actual/residual, for your own plotting

    def summary(self) -> str:
        mape_str = f"{self.mape:.1%}" if self.mape is not None and not math.isnan(self.mape) else "n/a"
        return (
            f"[{self.subsystem}] n={self.n_points}  MAE={self.mae:.4g}  RMSE={self.rmse:.4g}  "
            f"MAPE={mape_str}  R\u00b2={self.r_squared:.3f}  bias={self.bias:+.4g}"
        )


def _score(subsystem: str, predicted: np.ndarray, actual: np.ndarray) -> ValidationResult:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    residual = predicted - actual

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    bias = float(np.mean(residual))

    nonzero = np.abs(actual) > 1e-9
    mape = float(np.mean(np.abs(residual[nonzero] / actual[nonzero]))) if nonzero.any() else None

    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    table = pd.DataFrame({"predicted": predicted, "actual": actual, "residual": residual})
    return ValidationResult(subsystem=subsystem, n_points=len(actual), mae=mae, rmse=rmse,
                             mape=mape, r_squared=r_squared, bias=bias, table=table)


# ---------------------------------------------------------------------------
# Motor: two independent tests -- whole-vehicle energy behaviour from real
# driving, and raw bench/dyno efficiency at specific operating points.
# ---------------------------------------------------------------------------

def validate_energy_model(
    car_model, sections_df: pd.DataFrame, gradient_angle: float = 0.0
) -> ValidationResult:
    """Compare a car model's predicted per-section energy against real,
    already-driven telemetry -- ``sections_df`` is the same
    (v1, v2, E_Engine, time) shape ``carmodel.aggregate_sections()``
    produces from real logs. This validates motor + resistance TOGETHER as
    they actually behave on the road, which is a different (and arguably
    more honest) test than a clean bench measurement: it includes whatever
    the physics model is missing in combination, not each piece in
    isolation."""
    predicted = []
    for _, row in sections_df.iterrows():
        try:
            energy, _ = car_model.predict(gradient_angle, row["v1"], row["v2"], section_length=100.0)
        except Exception:
            energy = np.nan
        predicted.append(energy)
    predicted = np.array(predicted)
    valid = ~np.isnan(predicted)
    return _score("energy_model (motor+resistance, from real driving)",
                   predicted[valid], sections_df["E_Engine"].to_numpy()[valid])


def validate_motor_bench(efficiency_map, bench_df: pd.DataFrame) -> ValidationResult:
    """Compare a ``MotorEfficiencyMap``'s predicted efficiency against a
    real dyno/bench log. ``bench_df`` needs ``rpm``, ``torque_nm``,
    ``voltage_v``, ``current_a`` -- actual efficiency is computed from
    those directly (mechanical power / electrical power), not pre-supplied,
    so you're comparing the map against raw bench measurements exactly as
    they came off the equipment."""
    required = {"rpm", "torque_nm", "voltage_v", "current_a"}
    if not required.issubset(bench_df.columns):
        raise ValueError(f"bench_df needs {required}, got {list(bench_df.columns)}")

    omega = bench_df["rpm"] * 2 * math.pi / 60.0
    mech_power = bench_df["torque_nm"] * omega
    elec_power = bench_df["voltage_v"] * bench_df["current_a"]
    actual_eff = (mech_power / elec_power).clip(0, 1.5)  # allow a little headroom to see measurement noise, not hide it

    predicted_eff = np.array([
        efficiency_map.efficiency_at(r, t) if efficiency_map.is_2d else efficiency_map.efficiency_at(r)
        for r, t in zip(bench_df["rpm"], bench_df["torque_nm"])
    ])
    return _score("motor_efficiency_map (bench)", predicted_eff, actual_eff.to_numpy())


# ---------------------------------------------------------------------------
# Battery: capacity delivered vs. spec, from a real constant-current
# discharge test to a cutoff voltage -- the standard bench methodology.
# ---------------------------------------------------------------------------

@dataclass
class CapacityValidationResult:
    delivered_ah: float
    delivered_wh: float
    predicted_usable_wh: float
    percent_error: float          # (predicted - actual) / actual
    suggested_usable_fraction: float  # what usable_fraction WOULD have matched this test
    cutoff_reached: bool
    table: pd.DataFrame

    def summary(self) -> str:
        return (
            f"[battery_capacity] delivered={self.delivered_ah:.2f}Ah / {self.delivered_wh:.0f}Wh, "
            f"predicted usable={self.predicted_usable_wh:.0f}Wh, error={self.percent_error:+.1%}, "
            f"suggested usable_fraction={self.suggested_usable_fraction:.2f}"
        )


def validate_battery_capacity(
    battery: BatterySpec, discharge_df: pd.DataFrame, cutoff_voltage: float,
    time_col: str = "time_s", voltage_col: str = "voltage_v", current_col: str = "current_a",
) -> CapacityValidationResult:
    """Standard constant-current discharge test analysis: integrate
    delivered charge/energy from real logged current+voltage until the
    pack hits its cutoff voltage, and compare against
    ``battery.usable_capacity_wh``. This is what an actual capacity bench
    test looks like -- discharge at a known rate, log voltage sag, stop at
    the cutoff, see how much you actually got.
    """
    df = discharge_df.sort_values(time_col).reset_index(drop=True)
    below_cutoff = df[voltage_col] <= cutoff_voltage
    cutoff_reached = bool(below_cutoff.any())
    if cutoff_reached:
        cutoff_idx = below_cutoff.idxmax()
        df = df.iloc[: cutoff_idx + 1]

    t = df[time_col].to_numpy(dtype=float)
    i = df[current_col].to_numpy(dtype=float)
    v = df[voltage_col].to_numpy(dtype=float)

    delivered_ah = float(np.trapezoid(i, t) / 3600.0)
    delivered_wh = float(np.trapezoid(i * v, t) / 3600.0)

    predicted_usable_wh = battery.usable_capacity_wh
    percent_error = (predicted_usable_wh - delivered_wh) / delivered_wh if delivered_wh > 0 else float("nan")
    nameplate_wh = battery.nominal_voltage_v * battery.capacity_ah
    suggested_fraction = delivered_wh / nameplate_wh if nameplate_wh > 0 else float("nan")

    return CapacityValidationResult(
        delivered_ah=delivered_ah, delivered_wh=delivered_wh, predicted_usable_wh=predicted_usable_wh,
        percent_error=float(percent_error), suggested_usable_fraction=float(suggested_fraction),
        cutoff_reached=cutoff_reached, table=df,
    )


# ---------------------------------------------------------------------------
# Resistance: predict-and-compare on a coastdown run held OUT of fitting.
# ---------------------------------------------------------------------------

def validate_resistance(
    constants: CarConstants, held_out_df: pd.DataFrame, mass_kg: float,
    speed_col: str = "gps_speed", time_col: str = "obc_timestamp",
    smoothing_window: int = 5, min_speed_ms: float = 1.0, g: float = 9.81,
) -> ValidationResult:
    """The correct way to check a coastdown fit: run it on a DIFFERENT
    coastdown than the one used to calibrate ``constants``. Predicts
    deceleration from the already-calibrated Crr/Cd and compares against
    what this new run actually did -- reuses the exact same
    smoothing/differencing as ``coastdown.analyze_coastdown`` via
    ``smoothed_deceleration`` so the comparison is apples-to-apples."""
    v, a_actual = smoothed_deceleration(held_out_df, speed_col, time_col, smoothing_window, min_speed_ms)
    if len(v) < 5:
        raise ValueError(f"Only {len(v)} valid points in the held-out run -- need more to validate against.")

    a_predicted = g * constants.coeff_roll_res + (constants.rho_air * constants.a_aero * constants.coeff_aero_drag) / (2 * mass_kg) * v ** 2
    return _score("resistance (held-out coastdown)", a_predicted, a_actual)


# ---------------------------------------------------------------------------
# Solar array: predicted vs. measured output at known irradiance/temperature.
# ---------------------------------------------------------------------------

def validate_solar_array(array: ArraySpec, lab_df: pd.DataFrame) -> ValidationResult:
    """``lab_df`` needs ``irradiance_w_m2``, ``measured_power_w``, and
    optionally ``cell_temp_c`` (defaults to 25 if not measured -- note that
    skipping real cell temperature will bias this validation on a hot day,
    since the model derates for temperature and the comparison won't know
    the real cells were hotter than 25degC)."""
    required = {"irradiance_w_m2", "measured_power_w"}
    if not required.issubset(lab_df.columns):
        raise ValueError(f"lab_df needs {required}, got {list(lab_df.columns)}")
    temps = lab_df["cell_temp_c"] if "cell_temp_c" in lab_df.columns else pd.Series(25.0, index=lab_df.index)
    predicted = np.array([available_power(array, irr, t) for irr, t in zip(lab_df["irradiance_w_m2"], temps)])
    return _score("solar_array", predicted, lab_df["measured_power_w"].to_numpy())


def print_report(*results) -> None:
    """Convenience: print every validation result's one-line summary
    together, for a quick "how's the whole digital twin doing" check."""
    for r in results:
        print(r.summary())
