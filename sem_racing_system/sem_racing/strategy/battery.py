"""
sem_racing.battery
===================
Before this module: nothing in the system tracked battery state at all --
the optimizer computed energy per section with no ceiling. Now that a real
capacity exists (60V, 50Ah NMC), this adds a genuine state-of-charge check:
walk the solved speed profile and confirm the battery never runs out or
overflows above 100%.

What this deliberately does NOT do, because the inputs to do it honestly
don't exist yet:

- No discharge CURRENT limit check. The battery's continuous/peak amp
  ratings haven't been given. Without them this can tell you "you'll run
  out of energy at km 210" but not "you'll blow past the safe discharge
  rate on that climb at km 40" -- those are different failure modes, and
  only the first is covered here.
- No voltage sag / internal resistance. Terminal voltage is treated as
  flat at ``nominal_voltage_v`` regardless of state-of-charge or load --
  real packs sag under load and as they deplete. Fine for an energy-budget
  check, not fine for a voltage-accurate current calculation.
- No temperature effects on capacity.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BatterySpec:
    """Real pack spec -- NMC, 60V nominal, 50Ah."""

    nominal_voltage_v: float = 60.0
    capacity_ah: float = 50.0
    chemistry: str = "NMC"
    weight_kg: float = 17.5  # midpoint of the 15-20kg range given -- narrow this once weighed
    usable_fraction: float = 0.9  # NMC is typically not cycled 0-100% to protect cell life;
                                   # 0.9 is a conservative planning default, not a datasheet value --
                                   # replace with your BMS's actual configured SoC window

    # Left unset deliberately -- no discharge current limits have been supplied yet.
    continuous_current_limit_a: float | None = None
    peak_current_limit_a: float | None = None

    @property
    def usable_capacity_wh(self) -> float:
        return self.nominal_voltage_v * self.capacity_ah * self.usable_fraction

    @property
    def usable_capacity_j(self) -> float:
        return self.usable_capacity_wh * 3600.0


def simulate_soc(
    node_path: list[str], G, battery: BatterySpec, starting_soc_fraction: float = 1.0
) -> pd.DataFrame:
    """Walk a solved speed-profile path and track state of charge section by
    section, using each edge's stored ``energy``/``income`` attributes (set
    by ``optimize.build_speed_graph``). Returns a DataFrame with running
    energy used, income, net, and resulting SoC fraction -- so you can see
    exactly where in the stage the battery would run out, if it does.
    """
    capacity_j = battery.usable_capacity_j
    soc_j = starting_soc_fraction * capacity_j

    rows = [{"step": 0, "soc_fraction": starting_soc_fraction, "soc_wh": soc_j / 3600.0, "net_energy_j": 0.0}]
    for i, (u, v) in enumerate(zip(node_path[:-1], node_path[1:]), start=1):
        edge = G.edges[u, v]
        net = edge.get("energy", 0.0) - edge.get("income", 0.0)
        soc_j -= net
        rows.append({
            "step": i,
            "soc_fraction": soc_j / capacity_j,
            "soc_wh": soc_j / 3600.0,
            "net_energy_j": net,
        })
    return pd.DataFrame(rows)


def check_feasible(soc_trace: pd.DataFrame) -> tuple[bool, str]:
    """Plain-English feasibility verdict from a ``simulate_soc`` trace."""
    min_soc = soc_trace["soc_fraction"].min()
    max_soc = soc_trace["soc_fraction"].max()
    if min_soc < 0:
        step = soc_trace.loc[soc_trace["soc_fraction"].idxmin(), "step"]
        return False, f"Battery would be depleted before the finish (hits {min_soc:.0%} at step {step})."
    if max_soc > 1.0:
        step = soc_trace.loc[soc_trace["soc_fraction"].idxmax(), "step"]
        return False, (
            f"Battery would overflow past 100% at step {step} ({max_soc:.0%}) -- solar income "
            "exceeds usable capacity; excess would need to be dumped or the array/pace reconsidered."
        )
    return True, f"Feasible: SoC stays between {min_soc:.0%} and {max_soc:.0%} across the stage."
