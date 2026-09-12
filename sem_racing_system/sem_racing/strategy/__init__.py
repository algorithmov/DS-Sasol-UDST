"""Pillar 2: Race strategy simulation -- energy model, route profile, PV vs.
consumption optimisation, pre-race target speed planning, and multi-stage
route/weather variable strategy."""
from . import trackmodel, carmodel, solar, wind, battery, optimize, stage_planner  # noqa: F401
__all__ = ["trackmodel", "carmodel", "solar", "wind", "battery", "optimize", "stage_planner"]
