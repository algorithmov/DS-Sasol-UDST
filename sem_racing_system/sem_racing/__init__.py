"""
sem_racing
==========
Rebuilt around six pillars, and only these six:

1. Telemetry processing & analytics       -> sem_racing.telemetry
2. Race strategy simulation                -> sem_racing.strategy
   (energy model, route profile, PV vs. consumption optimisation,
   pre-race target speed planning, multi-stage route/weather strategy)
3. Performance predictions                  -> sem_racing.performance
   (motor efficiency maps, resistance)
4. After-testing analysis                   -> sem_racing.testing
   (coastdown, regression analysis)

No dashboard, no driver-training module -- deliberately out of scope for
this build.
"""
from . import telemetry, strategy, performance, testing  # noqa: F401

__all__ = ["telemetry", "strategy", "performance", "testing"]
