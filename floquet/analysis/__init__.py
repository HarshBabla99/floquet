"""Core orchestration utilities for running Floquet simulations.

This subpackage contains light-weight helpers that break the monolithic
``FloquetAnalysis`` workflow into composable pieces.  The public API of the
package still exposes :class:`~floquet.floquet.FloquetAnalysis` from the legacy
module to avoid breaking downstream imports.
"""

from .range import AmplitudeWindow, partition_amplitudes
from .results import FloquetRunResult, SimulationArrays, WindowResult
from .executor import FrequencySweepExecutor
from .mode_tracking import ModeTracking

__all__ = [
    "AmplitudeWindow",
    "partition_amplitudes",
    "FloquetRunResult",
    "SimulationArrays",
    "WindowResult",
    "FrequencySweepExecutor",
    "ModeTracking",
]
