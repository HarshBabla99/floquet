"""Structured containers for Floquet simulation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .range import AmplitudeWindow


@dataclass(slots=True)
class WindowResult:
    """Intermediate arrays generated for a single amplitude window."""

    bare_state_overlaps: np.ndarray
    floquet_modes: np.ndarray
    avg_excitation: np.ndarray
    quasienergies: np.ndarray
    final_modes: np.ndarray


@dataclass(slots=True)
class SimulationArrays:
    """Mutable buffers aggregating results across amplitude windows."""

    num_omega: int
    num_amp: int
    num_states: int
    hilbert_dim: int
    store_modes: bool
    bare_state_overlaps: np.ndarray = field(init=False, repr=False)
    intermediate_displaced_state_overlaps: np.ndarray = field(init=False, repr=False)
    avg_excitation: np.ndarray = field(init=False, repr=False)
    quasienergies: np.ndarray = field(init=False, repr=False)
    floquet_modes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        shape = (self.num_omega, self.num_amp, self.num_states)
        self.bare_state_overlaps = np.zeros(shape)
        self.intermediate_displaced_state_overlaps = np.zeros(shape)
        self.avg_excitation = np.zeros((self.num_omega, self.num_amp, self.hilbert_dim))
        self.quasienergies = np.zeros_like(self.avg_excitation)
        self.floquet_modes = np.zeros(
            (self.num_omega, self.num_amp, self.num_states, self.hilbert_dim), dtype=complex
        )

    def update_window(self, window: AmplitudeWindow, result: WindowResult) -> None:
        """Copy a window result into the aggregated buffers."""

        slc = window.slice
        self.bare_state_overlaps[:, slc] = result.bare_state_overlaps
        self.avg_excitation[:, slc] = result.avg_excitation
        self.quasienergies[:, slc] = result.quasienergies
        self.floquet_modes[:, slc] = result.floquet_modes

    def to_result(
        self,
        fit_data: np.ndarray,
        displaced_state_overlaps: np.ndarray,
    ) -> "FloquetRunResult":
        """Create the immutable :class:`FloquetRunResult`."""

        return FloquetRunResult(
            bare_state_overlaps=self.bare_state_overlaps,
            fit_data=fit_data,
            displaced_state_overlaps=displaced_state_overlaps,
            intermediate_displaced_state_overlaps=self.intermediate_displaced_state_overlaps,
            quasienergies=self.quasienergies,
            avg_excitation=self.avg_excitation,
            floquet_modes=self.floquet_modes if self.store_modes else None,
        )


@dataclass(slots=True)
class FloquetRunResult:
    """Final immutable representation of a Floquet simulation."""

    bare_state_overlaps: np.ndarray
    fit_data: np.ndarray
    displaced_state_overlaps: np.ndarray
    intermediate_displaced_state_overlaps: np.ndarray
    quasienergies: np.ndarray
    avg_excitation: np.ndarray
    floquet_modes: Optional[np.ndarray]

    def to_dict(self, include_modes: bool) -> Dict[str, np.ndarray]:
        """Convert to the legacy dictionary representation."""

        data = {
            "bare_state_overlaps": self.bare_state_overlaps,
            "fit_data": self.fit_data,
            "displaced_state_overlaps": self.displaced_state_overlaps,
            "intermediate_displaced_state_overlaps": self.intermediate_displaced_state_overlaps,
            "quasienergies": self.quasienergies,
            "avg_excitation": self.avg_excitation,
        }
        if include_modes and self.floquet_modes is not None:
            data["floquet_modes"] = self.floquet_modes
        return data
