"""
sem_racing.carmodel
====================
Both ``BE_SpeedProfileOptimisation.ipynb`` and ``AI_SpeedProfileOptimisation.ipynb``
plug a "car model" into the identical graph-building loop. This module makes
that interface explicit (``.predict(gradient_angle, v1, v2, section_length) ->
(energy_J, time_s)``, raising ``ImpossibleState`` when a transition can't
physically/plausibly happen) so ``optimize.py`` never needs to know or care
whether it's driving a first-principles model or a trained regressor.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class ImpossibleState(Exception):
    """Raised when a requested speed transition can't be achieved (e.g. the
    motor's torque limit would be exceeded)."""


@dataclass
class CarConstants:
    """Vehicle parameters -- pull these from the club's actual car/rules
    rather than the notebook's placeholder numbers before trusting outputs.
    These drive ``PhysicalCarModel`` (geared brushed-DC motor, the original
    bootcamp assumption) -- for the club's real Mitsuba M2096D-III in-wheel
    direct-drive motor, use ``MotorConstants`` + ``DirectDriveMotorModel``
    below instead."""

    driver_mass: float = 71.0          # kg -- still a placeholder, confirm your actual driver's weight
    car_mass_no_driver: float = 300.0  # kg -- real spec. ASSUMPTION: treated as vehicle-only,
                                        # excluding driver -- confirm if "300kg" already includes them,
                                        # in which case set driver_mass=0 here instead of double-counting
    diameter_tire: float = 0.4         # m -- still a placeholder, wheel diameter not yet given
    coeff_roll_res: float = 0.0134     # Michelin Pilot Street 2 (70/90-17), ENGINEERING ESTIMATE not
                                        # a measured value: 0.0150 baseline (concrete, ordinary tire)
                                        # x0.85 (silica compound) x1.05 (5mm tread depth) = 0.0134.
                                        # No manufacturer Crr spec exists for this commuter tire --
                                        # run testing.coastdown on a real coastdown test to check
                                        # this estimate against reality before trusting it fully.
    gear_ratio: float = 5.0
    coeff_aero_drag: float = 0.2       # still a placeholder -- no drag coefficient given yet
    a_aero: float = 0.8                # m^2 frontal area -- still a placeholder
    rho_air: float = 1.225             # kg/m^3
    torque_max: float = 3.6            # N*m
    torque_constant: float = 0.130     # N*m/A
    velocity_constant: float = 70.0    # 1/(min*V)
    anchor_resistance: float = 0.1     # Ohm
    torque_loss: float = 0.5           # N*m
    g: float = 9.81

    @property
    def car_mass(self) -> float:
        return self.car_mass_no_driver + self.driver_mass


class PhysicalCarModel:
    """First-principles series-wound-DC-motor model, refactored 1:1 out of
    ``BE_SpeedProfileOptimisation.ipynb`` (cell 11) with the constants made
    configurable instead of module-level globals."""

    def __init__(self, constants: CarConstants | None = None):
        self.c = constants or CarConstants()

    def predict(self, gradient_angle: float, v1: float, v2: float, section_length: float,
                headwind_ms: float = 0.0):
        c = self.c
        accel = (v2 ** 2 - v1 ** 2) / section_length / 2.0
        if math.isclose(accel, 0.0):
            time = section_length / v2
        else:
            time = (v2 - v1) / accel

        def velocity(t):
            return v1 + accel * t

        def total_force(t):
            f_roll_res = c.car_mass * c.g * math.cos(gradient_angle) * c.coeff_roll_res
            f_slope = c.car_mass * c.g * math.sin(gradient_angle)
            f_accel = accel * c.car_mass
            # NOTE: this legacy model ignores headwind_ms -- kept for interface
            # parity with DirectDriveMotorModel/AICarModel, not honoured here.
            f_aero_drag = 0.5 * c.rho_air * c.a_aero * c.coeff_aero_drag * velocity(t) ** 2
            return f_roll_res + f_slope + f_accel + f_aero_drag

        def total_power_usage(t):
            return total_force(t) * velocity(t)

        def engine_rotation_speed(t):
            return velocity(t) * c.gear_ratio / (c.diameter_tire / 2)

        def engine_torque(t):
            torque = total_power_usage(t) / engine_rotation_speed(t)
            if torque > c.torque_max:
                raise ImpossibleState("Torque exceeds the maximum allowed.")
            return torque

        def engine_current(t):
            return (engine_torque(t) + c.torque_loss) / c.torque_constant

        def engine_power(t):
            if total_power_usage(t) < 0:
                return 0.0
            factor_rpm = 60 / (2 * math.pi)
            voltage = (
                engine_rotation_speed(t) * factor_rpm / c.velocity_constant
                + c.anchor_resistance * engine_torque(t)
            )
            return voltage * engine_current(t)

        energy_motor, _ = integrate.quad(engine_power, 0, time)
        return energy_motor, time


@dataclass
class MotorConstants:
    """Real Mitsuba M2096D-III spec (in-wheel, brushless, direct-drive --
    https://www.mitsuba.co.jp/scr/product/m2096-iii.html). This motor has no
    gearbox and no published torque/winding constants (no KT/KV/anchor
    resistance) -- Mitsuba only publish a performance-curve *image*, not
    numeric equivalent-circuit values -- so it cannot honestly be plugged
    into ``PhysicalCarModel``'s brushed-DC equations. What the datasheet
    does give directly is a power ceiling and an efficiency peak, which is
    exactly what ``DirectDriveMotorModel`` below uses."""

    wheel_diameter: float = 0.558       # m -- computed from real tyre choice, not a guess:
                                         # Michelin Pilot Street 2, front 70/90-17: width 70mm,
                                         # aspect ratio 90% -> sidewall 63mm, + 17in (431.8mm) rim
                                         # -> OD = 431.8 + 2*63 = 557.8mm. The rear tyre (80/90-17)
                                         # works out to 0.576m instead -- CONFIRM which wheel
                                         # actually carries the M2096D-III before trusting this;
                                         # front value used here as the default, not verified.
    rated_power_w: float = 2000.0       # continuous rated output
    peak_power_w: float = 5000.0        # short-burst ceiling ("reference value" per datasheet)
    peak_efficiency: float = 0.95       # at the rated operating point only
    rated_speed_rpm: float = 810.0      # wheel rpm where peak_efficiency applies
    efficiency_floor: float = 0.75      # floor away from the rated point -- a guess, not
                                         # a datasheet value; replace once you have real
                                         # dyno points or a digitised efficiency curve
    efficiency_width_rpm: float = 500.0  # how quickly efficiency falls off away from rated


def mitsuba_efficiency(rpm: float, m: MotorConstants) -> float:
    """Kept only for backward-compat call sites; use
    ``performance.motor_map.MotorEfficiencyMap.from_datasheet_default()``
    directly instead -- ``DirectDriveMotorModel`` no longer calls this
    function, it consumes a proper ``MotorEfficiencyMap`` object."""
    if rpm <= 0:
        return m.efficiency_floor
    delta = (rpm - m.rated_speed_rpm) / m.efficiency_width_rpm
    return m.efficiency_floor + (m.peak_efficiency - m.efficiency_floor) * math.exp(-0.5 * delta ** 2)


class DirectDriveMotorModel:
    """Physical model matching the club's actual motor: brushless, in-wheel,
    direct-drive (no gearbox -- ``gear_ratio`` from ``PhysicalCarModel``
    doesn't apply here, wheel speed *is* motor speed). Mechanical demand is
    computed with the same force balance as ``PhysicalCarModel`` (rolling
    resistance, gradient, acceleration, aero drag -- now against *relative
    airspeed*, not ground speed, see ``headwind_ms`` below); electrical draw
    is mechanical power divided by a real ``MotorEfficiencyMap`` lookup,
    capped at the datasheet's rated/peak power ceiling instead of a torque
    limit.

    This intentionally does NOT reuse ``PhysicalCarModel``'s winding-level
    equations -- this motor's datasheet doesn't publish the constants those
    equations need, and it's a different motor topology (BLDC direct-drive
    vs. brushed-through-a-gearbox) anyway. Forcing one onto the other's
    equations would produce numbers that look precise but describe a motor
    you don't have.
    """

    def __init__(self, car: CarConstants | None = None, motor: MotorConstants | None = None,
                 efficiency_map=None):
        self.c = car or CarConstants()
        self.m = motor or MotorConstants()
        if efficiency_map is None:
            # default is now the digitized real chart (see performance.motor_map's
            # module docstring for the three-tier honesty ladder), not the guessed
            # Gaussian -- strictly better information, still not a bench measurement
            from ..performance.motor_map import MotorEfficiencyMap
            efficiency_map = MotorEfficiencyMap.from_mitsuba_eco_mode_chart()
        self.efficiency_map = efficiency_map

    def predict(self, gradient_angle: float, v1: float, v2: float, section_length: float,
                headwind_ms: float = 0.0, controller_temp_c: float | None = None):
        c, m = self.c, self.m
        accel = (v2 ** 2 - v1 ** 2) / section_length / 2.0
        if math.isclose(accel, 0.0):
            if v2 <= 0:
                raise ImpossibleState("Zero/negative speed with no acceleration.")
            time = section_length / v2
        else:
            time = (v2 - v1) / accel

        # thermal derating straight from the manual's fault table (page 13) --
        # only applied if you actually supply a measured/estimated controller
        # temperature; no thermal model predicts this on its own.
        from ..performance.motor_map import mitsuba_thermal_derate
        peak_power_limit = m.peak_power_w
        if controller_temp_c is not None:
            peak_power_limit = m.peak_power_w * mitsuba_thermal_derate(controller_temp_c)

        def velocity(t):
            return v1 + accel * t

        def airspeed(t):
            # relative airspeed = ground speed + headwind component (headwind
            # positive = blowing against travel, increases drag; tailwind
            # negative = reduces it). Clamped at 0: a tailwind stronger than
            # ground speed can't produce negative drag by this model.
            return max(velocity(t) + headwind_ms, 0.0)

        def mech_power(t):
            v = velocity(t)
            f_roll = c.car_mass * c.g * math.cos(gradient_angle) * c.coeff_roll_res
            f_slope = c.car_mass * c.g * math.sin(gradient_angle)
            f_accel = accel * c.car_mass
            f_drag = 0.5 * c.rho_air * c.a_aero * c.coeff_aero_drag * airspeed(t) ** 2
            return (f_roll + f_slope + f_accel + f_drag) * v

        def wheel_rpm(t):
            v = velocity(t)
            return (v / (math.pi * m.wheel_diameter)) * 60.0  # direct drive: gear_ratio == 1

        def electrical_power(t):
            p_mech = mech_power(t)
            if p_mech <= 0:
                return 0.0  # coasting/braking -- not crediting regen here, see solar.py note
            if p_mech > peak_power_limit:
                if controller_temp_c is not None and peak_power_limit < m.peak_power_w:
                    raise ImpossibleState(
                        f"Demanded {p_mech:.0f} W exceeds the thermally-derated limit of "
                        f"{peak_power_limit:.0f} W at {controller_temp_c:.0f}\u00b0C "
                        f"(nameplate peak is {m.peak_power_w:.0f} W)."
                    )
                raise ImpossibleState(
                    f"Demanded {p_mech:.0f} W exceeds the motor's {m.peak_power_w:.0f} W peak."
                )
            eff = self.efficiency_map.efficiency_at(wheel_rpm(t))
            return p_mech / eff

        energy, _ = integrate.quad(electrical_power, 0, time)
        return energy, time


@dataclass
class AICarModel:
    """Data-driven replacement for ``PhysicalCarModel``, refactored out of
    ``AI_SpeedProfileOptimisation.ipynb``. Trained on real per-section
    (v1, v2) -> (energy, time) samples aggregated from historical telemetry,
    so it captures effects the physics model glosses over (real rolling
    resistance, driver behaviour, wind on the day, etc.) at the cost of
    needing enough logged laps to be trustworthy."""

    sk_pipeline: object
    metrics_: dict = field(default_factory=dict, init=False)

    def fit(self, section_samples: pd.DataFrame):
        """``section_samples`` needs columns v1, v2, E_Engine, time -- see
        ``trackmodel``/notebook's ``aggregate_data`` for how these are built
        from raw telemetry."""
        X = section_samples[["v1", "v2"]]
        y = section_samples[["E_Engine", "time"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        self.sk_pipeline.fit(X_train, y_train)
        y_pred = self.sk_pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        self.metrics_ = {
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "corr_energy": float(np.corrcoef(y_test.iloc[:, 0], y_pred[:, 0])[0, 1]),
            "corr_time": float(np.corrcoef(y_test.iloc[:, 1], y_pred[:, 1])[0, 1]),
        }
        return self

    def predict(self, gradient_angle: float, v1: float, v2: float, section_length: float,
                headwind_ms: float = 0.0):
        # NOTE: trained purely on (v1, v2) -> (energy, time); gradient_angle
        # and headwind_ms are accepted for interface parity but NOT used --
        # this model has no notion of hills or wind at all. If your training
        # laps included both flat and hilly sections indiscriminately, that
        # variation just becomes unexplained noise in what it learned.
        df = pd.DataFrame([[v1, v2]], columns=["v1", "v2"])
        energy_motor, time = self.sk_pipeline.predict(df)[0]
        if time <= 0:
            raise ImpossibleState("Negative time")
        if energy_motor <= 0:
            energy_motor = 0.0
        return energy_motor, time


def aggregate_sections(
    df: pd.DataFrame,
    section_length: float = 50.0,
    dist_col: str = "lap_dist",
    speed_col: str = "gps_speed",
    energy_col: str = "jm3_netjoule",
    time_col: str = "obc_timestamp",
    lap_col: str = "lap_lap",
    n_shifts: int = 4,
) -> pd.DataFrame:
    """Turn raw per-sample telemetry into the (v1, v2, E_Engine, time) rows
    an ``AICarModel`` trains on -- refactored out of the AI notebook's
    ``aggregate_data``. Bins are shifted ``n_shifts`` times to multiply the
    effective training-set size out of a limited number of real laps."""
    max_dist = np.ceil(df[dist_col].max() / section_length) * section_length
    shifts = np.linspace(0, section_length, n_shifts, endpoint=False)
    results = []
    for shift in shifts:
        bins = np.arange(0, max_dist, section_length) + shift
        d = df.copy()
        d["_bin"] = pd.cut(d[dist_col], bins=bins, right=False)
        grouped = d.groupby([lap_col, "_bin"], observed=True).agg(
            E_Engine=(energy_col, lambda x: x.max() - x.min()),
            time=(time_col, lambda x: x.max() - x.min()),
            v1=(speed_col, "first"),
            v2=(speed_col, "last"),
        ).reset_index(drop=True)
        grouped["v1"] /= 3.6
        grouped["v2"] /= 3.6
        results.append(grouped)
    return pd.concat(results, ignore_index=True).dropna()
