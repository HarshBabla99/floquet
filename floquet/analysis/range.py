"""Amplitude range helpers for staged Floquet simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, slots=True)
class AmplitudeWindow:
    """Half-open interval of amplitude indices processed in a single batch."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            msg = (
                "AmplitudeWindow boundaries must satisfy 0 <= start < stop; "
                f"received start={self.start}, stop={self.stop}."
            )
            raise ValueError(msg)

    @property
    def slice(self) -> slice:
        """Return the python slice for this amplitude window."""

        return slice(self.start, self.stop)

    @property
    def width(self) -> int:
        """Number of amplitude samples covered by the window."""

        return self.stop - self.start

    def offset(self, delta: int) -> "AmplitudeWindow":
        """Create a new window shifted by ``delta`` amplitudes."""

        return AmplitudeWindow(self.start + delta, self.stop + delta)


def partition_amplitudes(total_points: int, fraction: float) -> List[AmplitudeWindow]:
    """Return amplitude windows honoring the ``fit_range_fraction`` protocol."""

    if total_points <= 0:
        raise ValueError("At least one amplitude point is required for a simulation.")
    if not 0 < fraction <= 1:
        raise ValueError(
            "fit_range_fraction must lie in the interval (0, 1]; "
            f"received {fraction}."
        )

    num_ranges = int((1 / fraction) + 0.999999999)
    points_per_window = max(total_points // num_ranges, 1)

    windows: List[AmplitudeWindow] = []
    start = 0
    for window_idx in range(num_ranges):
        if window_idx == num_ranges - 1:
            stop = total_points
        else:
            stop = min(total_points, start + points_per_window)
        windows.append(AmplitudeWindow(start, stop))
        start = stop
        if start >= total_points:
            break

    if windows[-1].stop != total_points:
        windows[-1] = AmplitudeWindow(windows[-1].start, total_points)

    return windows
