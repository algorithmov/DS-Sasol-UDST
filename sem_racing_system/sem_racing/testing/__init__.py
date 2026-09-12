"""Pillar 4: After-testing analysis -- coastdown, regression analysis.
Includes sem_racing.testing.validation: compare every simulated subsystem
(motor, resistance, battery, solar array) against real lab/bench data,
independently of the others."""
from . import coastdown, validation  # noqa: F401
__all__ = ["coastdown", "validation"]
