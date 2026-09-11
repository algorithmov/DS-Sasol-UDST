"""A lightweight digital twin for the supplied Mitsuba solar-car components.

The model uses SI units internally.  It is deliberately parameterized: values that
were not in the supplied documents are defaults, not asserted specifications.
Run ``python solar_car_twin.py --help`` for the command-line interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import argparse
import json
import math
from pathlib import Path
from typing import Any

G = 9.80665


@dataclass
class MotorConfig:
    """Mitsuba M2096D-III + M2096C configuration.

    Values marked *manual* originate from the supplied M2096 manual.  The
    current/torque and current/speed slopes are digitized approximations of the
    96-V ECO chart; they must be replaced by measured data for design sign-off.
    """
    nominal_voltage_v: float = 96.0  # manual
    min_controller_voltage_v: float = 45.0  # manual
    max_controller_voltage_v: float = 140.0  # manual
    nominal_power_w: float = 2_000.0  # manual
    peak_power_w: float = 5_000.0  # manual: "about 5000 W"
    nominal_speed_rpm: float = 810.0  # manual
    combined_efficiency: float = 0.95  # manual: more than 95%, conservative bound
    eco_no_load_speed_rpm: float = 900.0  # digitized from manual 96-V ECO chart
    # Derived from manual nominal power and speed, not a published torque rating.
    nominal_torque_nm: float = 23.6


@dataclass
class PVConfig:
    """Maxeon Gen-6 cell array; cell data are from the supplied purchase file."""
    cell_pmax_w: float = 6.7
    cell_vmp_v: float = 0.625
    cell_imp_a: float = 10.8
    cell_voc_v: float = 0.73
    cell_isc_a: float = 11.5
    cell_efficiency: float = 0.245
    cells_series: int = 80  # ASSUMPTION: gives a ~50 V MPP string for a 48-V pack
    strings_parallel: int = 1  # INPUT REQUIRED: actual array layout
    mppt_efficiency: float = 0.97  # only documented for one candidate controller
    charge_current_limit_a: float = 60.0  # candidate-controller rating


@dataclass
class BatteryConfig:
    """48-V LFP pack selected in the supplied purchase file.

    The file calls it "48V, 60A" but purchase links say 60 Ah.  This twin treats
    it as 60 Ah; change capacity_ah if the vendor confirms another value.
    """
    nominal_voltage_v: float = 48.0
    capacity_ah: float = 60.0
    internal_resistance_ohm: float = 0.050  # INPUT REQUIRED / provisional
    min_soc: float = 0.10
    max_soc: float = 0.98
    max_discharge_a: float = 60.0  # provisional: a continuous BMS rating is needed
    max_charge_a: float = 60.0


@dataclass
class VehicleConfig:
    """Vehicle and tyre model.

    The radius uses the documented 80/90-17 rear tyre: (17 in * 25.4 mm +
    2 * 80 mm * 0.90) / 2 = 0.288 m.  The source estimates Crr = 0.0134 and
    mentions a 300-kg vehicle.
    """
    mass_kg: float = 300.0
    wheel_radius_m: float = 0.2879
    crr: float = 0.0134
    cd: float = 0.20  # INPUT REQUIRED / provisional
    frontal_area_m2: float = 1.0  # INPUT REQUIRED / provisional
    air_density_kg_m3: float = 1.225


@dataclass
class TwinConfig:
    motor: MotorConfig = field(default_factory=MotorConfig)
    pv: PVConfig = field(default_factory=PVConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)


class MotorTwin:
    def __init__(self, cfg: MotorConfig) -> None:
        self.cfg = cfg

    def performance(self, voltage_v: float, command: float) -> dict[str, float | bool]:
        """Return steady-state torque/speed/power at a 0..1 command.

        At command=1 it uses the published 2.0-kW / 810-rpm nominal point.  The
        low-current 96-V ECO chart supplies the roughly 900-rpm no-load point.
        It is a vehicle-level model, not an FOC model.
        """
        if not self.cfg.min_controller_voltage_v <= voltage_v <= self.cfg.max_controller_voltage_v:
            return {"enabled": False, "torque_nm": 0.0, "speed_rpm": 0.0,
                    "mechanical_power_w": 0.0, "electrical_power_w": 0.0}
        demand = max(0.0, min(1.0, command))
        scale = voltage_v / self.cfg.nominal_voltage_v
        torque = demand * self.cfg.nominal_torque_nm
        speed = max(0.0, (self.cfg.eco_no_load_speed_rpm -
                          demand * (self.cfg.eco_no_load_speed_rpm - self.cfg.nominal_speed_rpm)) * scale)
        mech = torque * speed * 2.0 * math.pi / 60.0
        mech = min(mech, self.cfg.peak_power_w)
        return {"enabled": True, "torque_nm": torque, "speed_rpm": speed,
                "mechanical_power_w": mech,
                "electrical_power_w": mech / self.cfg.combined_efficiency}


class PVBatteryTwin:
    def __init__(self, pv: PVConfig, battery: BatteryConfig) -> None:
        self.pv, self.battery = pv, battery

    def pv_power(self, irradiance_w_m2: float, temperature_c: float = 25.0) -> float:
        """MPP power. Temperature coefficient is unknown, so it is not invented."""
        _ = temperature_c
        cells = self.pv.cells_series * self.pv.strings_parallel
        return max(0.0, irradiance_w_m2) / 1000.0 * cells * self.pv.cell_pmax_w

    def ocv(self, soc: float) -> float:
        """Simple LFP plateau approximation; replace with the supplier OCV-SOC curve."""
        s = max(0.0, min(1.0, soc))
        return 44.0 + 8.0 * s

    def step(self, soc: float, load_power_w: float, irradiance_w_m2: float, dt_s: float) -> dict[str, float]:
        pv_w = self.pv_power(irradiance_w_m2) * self.pv.mppt_efficiency
        battery_w = load_power_w - min(pv_w, self.pv.charge_current_limit_a * self.ocv(soc))
        v_oc = self.ocv(soc)
        # P = (V_oc - I R) I.  Positive current discharges the pack.
        discriminant = max(0.0, v_oc * v_oc - 4.0 * self.battery.internal_resistance_ohm * battery_w)
        current_a = (v_oc - math.sqrt(discriminant)) / (2.0 * self.battery.internal_resistance_ohm)
        current_a = max(-self.battery.max_charge_a, min(self.battery.max_discharge_a, current_a))
        voltage_v = v_oc - current_a * self.battery.internal_resistance_ohm
        next_soc = max(self.battery.min_soc, min(self.battery.max_soc,
                       soc - current_a * dt_s / (3600.0 * self.battery.capacity_ah)))
        return {"soc": next_soc, "battery_current_a": current_a, "battery_voltage_v": voltage_v,
                "pv_power_w": pv_w, "battery_power_w": voltage_v * current_a}


class PowertrainTwin:
    def __init__(self, config: TwinConfig = TwinConfig()) -> None:
        self.config = config
        self.motor = MotorTwin(config.motor)
        self.energy = PVBatteryTwin(config.pv, config.battery)

    def road_load(self, speed_mps: float, grade: float = 0.0) -> dict[str, float]:
        v = max(0.0, speed_mps)
        c = self.config.vehicle
        rolling = c.mass_kg * G * c.crr * math.cos(math.atan(grade))
        grade_force = c.mass_kg * G * math.sin(math.atan(grade))
        aero = 0.5 * c.air_density_kg_m3 * c.cd * c.frontal_area_m2 * v * v
        return {"rolling_n": rolling, "grade_n": grade_force, "aero_n": aero,
                "total_n": rolling + grade_force + aero}

    def cruise(self, speed_kph: float, grade_percent: float, irradiance_w_m2: float,
               soc: float = 0.80, duration_s: float = 60.0) -> dict[str, Any]:
        speed_mps = speed_kph / 3.6
        road = self.road_load(speed_mps, grade_percent / 100.0)
        wheel_power_w = road["total_n"] * speed_mps
        electrical_w = wheel_power_w / self.config.motor.combined_efficiency
        state = self.energy.step(soc, electrical_w, irradiance_w_m2, duration_s)
        rpm = speed_mps / (2.0 * math.pi * self.config.vehicle.wheel_radius_m) * 60.0
        required_torque_nm = road["total_n"] * self.config.vehicle.wheel_radius_m
        nominal_speed_at_bus = self.config.motor.nominal_speed_rpm * state["battery_voltage_v"] / self.config.motor.nominal_voltage_v
        no_load_speed_at_bus = self.config.motor.eco_no_load_speed_rpm * state["battery_voltage_v"] / self.config.motor.nominal_voltage_v
        # Cap at nominal torque because peak-torque/current data were not supplied.
        torque_available_nm = self.config.motor.nominal_torque_nm * max(0.0, min(1.0,
            (no_load_speed_at_bus - rpm) / (no_load_speed_at_bus - nominal_speed_at_bus)))
        return {"speed_kph": speed_kph, "wheel_speed_rpm": rpm, "road_load": road,
                "wheel_power_w": wheel_power_w, "electrical_load_w": electrical_w,
                "state_after_duration": state,
                "required_wheel_torque_nm": required_torque_nm,
                "available_wheel_torque_nm": torque_available_nm,
                "torque_margin_nm": torque_available_nm - required_torque_nm,
                "nominal_speed_at_current_bus_rpm": nominal_speed_at_bus,
                "no_load_speed_at_current_bus_rpm": no_load_speed_at_bus,
                "speed_torque_feasible": required_torque_nm <= torque_available_nm,
                "controller_voltage_in_range": self.config.motor.min_controller_voltage_v <= state["battery_voltage_v"] <= self.config.motor.max_controller_voltage_v}


def _config_from_json(path: Path | None) -> TwinConfig:
    if path is None:
        return TwinConfig()
    data = json.loads(path.read_text(encoding="utf-8"))
    return TwinConfig(**{name: cls(**data.get(name, {})) for name, cls in
                         [("motor", MotorConfig), ("pv", PVConfig), ("battery", BatteryConfig), ("vehicle", VehicleConfig)]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Solar-car component digital twin")
    parser.add_argument("--config", type=Path, help="JSON file overriding the defaults")
    parser.add_argument("--speed-kph", type=float, default=50.0)
    parser.add_argument("--grade-percent", type=float, default=0.0)
    parser.add_argument("--irradiance", type=float, default=1000.0, help="W/m²")
    parser.add_argument("--soc", type=float, default=0.80)
    parser.add_argument("--duration", type=float, default=60.0, help="seconds")
    parser.add_argument("--write-default-config", type=Path)
    args = parser.parse_args()
    if args.write_default_config:
        args.write_default_config.write_text(json.dumps(asdict(TwinConfig()), indent=2), encoding="utf-8")
        return
    twin = PowertrainTwin(_config_from_json(args.config))
    print(json.dumps(twin.cruise(args.speed_kph, args.grade_percent, args.irradiance,
                                  args.soc, args.duration), indent=2))


if __name__ == "__main__":
    main()
