"""
sem_racing.performance.motor_map
==================================
"Performance predictions: motor efficiency maps" -- taken literally. The
version of this that lived in ``carmodel.py`` before this redesign
(``mitsuba_efficiency()``) was a single hand-shaped curve pinned to one
datasheet point -- not a map, an educated guess wearing a map's name. This
module has three tiers of honesty about where efficiency numbers come from:

  1. ``source="fallback_approximation"`` -- a symmetric guessed curve,
     used only when nothing else is available.
  2. ``source="digitized_datasheet_chart"`` -- real points read off the
     manufacturer's actual published performance chart (see
     ``from_mitsuba_eco_mode_chart()``), by eye, not with plot-digitizing
     software -- better than a guess, still not a bench measurement.
  3. ``source="bench_measurement"`` -- the club's own dyno/bench data via
     ``from_measurements()``. The real thing.

``describe()`` and every consumer of this map can tell which tier they're
looking at -- "measured" was previously a binary flag; it's now a named
source, since a digitized chart is neither a guess nor a true measurement
and deserves its own label rather than being lumped into either.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MotorEfficiencyMap:
    points: pd.DataFrame          # columns: rpm, efficiency, and optionally torque_nm
    source: str = "fallback_approximation"  # see module docstring for the three tiers
    floor: float = 0.5            # never return an efficiency below this, however far off-map

    @property
    def is_measured(self) -> bool:
        """Kept for backward compatibility: True for either real tier
        (digitized chart or bench data), False only for the guessed curve."""
        return self.source != "fallback_approximation"

    @classmethod
    def from_datasheet_default(
        cls,
        rated_speed_rpm: float = 810.0,
        peak_efficiency: float = 0.95,
        efficiency_floor: float = 0.75,
        width_rpm: float = 500.0,
        n_points: int = 41,
    ) -> "MotorEfficiencyMap":
        """Last-resort fallback when no chart or bench data exists at all:
        a symmetric peak centred on one published operating point. NOT
        measured, NOT digitized from a real curve -- an invented shape."""
        rpm = np.linspace(max(rated_speed_rpm - 3 * width_rpm, 1.0), rated_speed_rpm + 3 * width_rpm, n_points)
        delta = (rpm - rated_speed_rpm) / width_rpm
        eff = efficiency_floor + (peak_efficiency - efficiency_floor) * np.exp(-0.5 * delta ** 2)
        return cls(points=pd.DataFrame({"rpm": rpm, "efficiency": eff}),
                    source="fallback_approximation", floor=efficiency_floor)

    @classmethod
    def from_mitsuba_eco_mode_chart(cls) -> "MotorEfficiencyMap":
        """Real points read by eye off Mitsuba's own published "Motor
        Chracteristics, 96V ECO Mode" chart (M2096D-III/M2096C instruction
        manual, page 12) -- Speed[rpm], Efficiency[%], and Torque[kgfcm] all
        plotted against Current[A], 1-30A. Digitized visually from the chart
        image, not with plot-digitizing software -- treat these as good
        estimates, not precision data; re-digitize with something like
        WebPlotDigitizer if this ever needs to be more exact than "good
        estimate."

        HONEST DISCREPANCY, not silently resolved: this chart's efficiency
        plateaus around 93-94% out to 30A. The datasheet's headline spec
        claims ">95% (including motor controller efficiency)". This chart is
        explicitly labelled ECO mode -- the discrepancy may be because POWER
        mode achieves the higher figure, or because 95%+ is reached at a
        current beyond this chart's 30A range, or measured under different
        conditions entirely. Not reconciled here -- both numbers are real
        manufacturer claims that don't quite agree, worth asking Mitsuba
        directly about before treating either as exact.
        """
        current_a = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        rpm = [890, 878, 865, 852, 842, 832, 822, 815, 808, 802, 797, 792, 787, 782, 778]
        efficiency = [0.40, 0.75, 0.86, 0.90, 0.91, 0.92, 0.93, 0.935, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94]
        torque_kgfcm = [2, 4.5, 7, 9.5, 12, 14.5, 17, 19.5, 22, 24.5, 27, 29, 31, 32.5, 34]
        torque_nm = [t * 0.0980665 for t in torque_kgfcm]  # 1 kgf.cm = 0.0980665 N.m

        points = pd.DataFrame({
            "current_a": current_a, "rpm": rpm, "efficiency": efficiency, "torque_nm": torque_nm,
        })
        return cls(points=points, source="digitized_datasheet_chart", floor=0.3)

    @classmethod
    def from_measurements(cls, df: pd.DataFrame, floor: float = 0.5, source: str = "bench_measurement") -> "MotorEfficiencyMap":
        """Build a map from real data. ``df`` needs at least
        ``rpm``/``efficiency`` columns; add ``torque_nm`` too for a genuine
        2D (rpm, torque) -> efficiency map instead of an rpm-only curve."""
        required = {"rpm", "efficiency"}
        if not required.issubset(df.columns):
            raise ValueError(f"Measurements need at least {required}, got {list(df.columns)}")
        return cls(points=df.reset_index(drop=True), source=source, floor=floor)

    @property
    def is_2d(self) -> bool:
        return "torque_nm" in self.points.columns

    def efficiency_at(self, rpm: float, torque_nm: float | None = None) -> float:
        """Interpolated efficiency lookup. Extrapolation beyond the map's
        range is clamped to the nearest edge value rather than wildly
        extrapolated -- an interpolated guess just outside real data is far
        more trustworthy than a polynomial's runaway tail."""
        p = self.points
        if self.is_2d and torque_nm is not None:
            d2 = (p["rpm"] - rpm) ** 2 + (p["torque_nm"] - torque_nm) ** 2
            eff = float(p.loc[d2.idxmin(), "efficiency"])
        else:
            # np.interp requires x (rpm) sorted ascending -- real data is NOT
            # guaranteed to arrive that way (this motor's own chart has rpm
            # *decreasing* with current/load), so sort defensively here
            # rather than silently corrupting the interpolation.
            ordered = p.sort_values("rpm")
            rpm_clamped = np.clip(rpm, ordered["rpm"].min(), ordered["rpm"].max())
            eff = float(np.interp(rpm_clamped, ordered["rpm"], ordered["efficiency"]))
        return max(eff, self.floor)

    def describe(self) -> str:
        labels = {
            "fallback_approximation": "FALLBACK APPROXIMATION (not measured, not from a chart)",
            "digitized_datasheet_chart": "digitized from Mitsuba's published chart (by eye, not precision-digitized)",
            "bench_measurement": "real bench/dyno measurement",
        }
        kind = labels.get(self.source, self.source)
        shape = "2D (rpm, torque)" if self.is_2d else "1D (rpm only)"
        return f"{shape} efficiency map, {len(self.points)} points, {kind}"


def mitsuba_thermal_derate(controller_temp_c: float) -> float:
    """Power derating multiplier straight from the M2096C manual's own
    fault table (page 13): 85degC+ -> half power, 95degC+ -> quarter
    power, 105degC+ -> drive stop. This is the controller's real,
    documented protective behaviour, not a modeling assumption -- applying
    it doesn't predict what temperature the controller will reach (no
    thermal model exists for that), it only says what happens to available
    power *once* a given temperature is reached, exactly as Mitsuba
    specify.
    """
    if controller_temp_c >= 105.0:
        return 0.0
    if controller_temp_c >= 95.0:
        return 0.25
    if controller_temp_c >= 85.0:
        return 0.5
    return 1.0
