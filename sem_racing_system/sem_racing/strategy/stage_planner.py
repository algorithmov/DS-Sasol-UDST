"""
sem_racing.strategy.stage_planner
===================================
"Route/weather optimisation (Sasol routes require variable strategy)" --
this is what makes that literally true in code. Sasol is not one road: it's
several stages over several days, each with its own route geometry, its own
day's weather (irradiance, wind), and a battery state that carries over
from wherever the previous stage left it (this system does NOT model
overnight recharging -- see the caveat below). A single call to
``optimize.build_speed_graph`` plans one stage in isolation; this module
chains stages together so the strategy for stage 3 correctly reflects
however stage 1 and 2 actually went.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..telemetry import route as route_mod
from . import optimize
from .battery import BatterySpec, simulate_soc, check_feasible
from .solar import ArraySpec, IrradianceProfile
from .wind import WindProfile


@dataclass
class StagePlan:
    """One day's inputs. ``route`` is a lat/lon[,elevation_m] DataFrame (or
    a path -- see ``telemetry.route.load_route``); ``irradiance``/``wind``
    are that day's forecast, already built via
    ``IrradianceProfile``/``WindProfile.from_route_and_wind``."""

    name: str
    route: pd.DataFrame
    irradiance: IrradianceProfile
    wind: WindProfile | None = None
    section_length: float = 100.0
    top_speed_kmh: float = 90.0
    friction_coeff: float = 0.6


@dataclass
class StageResult:
    name: str
    profile: pd.DataFrame
    summary: dict
    soc_trace: pd.DataFrame
    feasible: bool
    message: str
    ending_soc_fraction: float


def plan_stages(
    stages: list[StagePlan],
    car_model,
    array: ArraySpec,
    battery: BatterySpec,
    settings_overrides: dict | None = None,
    starting_soc_fraction: float = 1.0,
    recharge_between_stages_to: float | None = None,
) -> list[StageResult]:
    """Solve each stage in order, carrying battery state of charge forward.

    ``recharge_between_stages_to``: if given (e.g. 1.0), SoC is reset to
    this value at the start of every stage instead of carrying over --
    models a scenario where the rules/schedule guarantee a full overnight
    recharge. Left as ``None`` (the honest default), SoC genuinely carries
    over unchanged from wherever the previous stage's plan left it -- if
    your event *does* allow overnight charging and you don't set this,
    you'll get an artificially pessimistic multi-day plan; if it does NOT
    and you do set it, you'll get an artificially optimistic one. This is a
    real assumption about your event's rules, not a default I can pick
    honestly on your behalf.
    """
    soc = starting_soc_fraction
    results: list[StageResult] = []

    for stage in stages:
        if recharge_between_stages_to is not None:
            soc = recharge_between_stages_to

        route_df = route_mod.load_route(stage.route) if not isinstance(stage.route, pd.DataFrame) else stage.route
        config = route_mod.build_stage_config(
            route_df,
            section_length=stage.section_length,
            friction_coeff=stage.friction_coeff,
            top_speed_kmh=stage.top_speed_kmh,
        )

        overrides = settings_overrides or {}
        settings = optimize.OptimizerSettings(section_length=stage.section_length, **overrides)

        G, coords, start_nodes, end_nodes = optimize.build_speed_graph(
            config, car_model, settings, array=array, irradiance=stage.irradiance, wind=stage.wind
        )
        profile, summary = optimize.solve_optimal_profile(G, config, start_nodes, end_nodes, settings)

        soc_trace = simulate_soc(summary["node_path"], G, battery, starting_soc_fraction=soc)
        feasible, message = check_feasible(soc_trace)
        ending_soc = float(soc_trace["soc_fraction"].iloc[-1])

        results.append(StageResult(
            name=stage.name, profile=profile, summary=summary, soc_trace=soc_trace,
            feasible=feasible, message=message, ending_soc_fraction=ending_soc,
        ))
        soc = ending_soc  # carry forward into the next stage unless recharge_between_stages_to overrides it

    return results
