"""Reusable helpers for selecting and ordering Floquet modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Tuple

import numpy as np
import qutip as qt

from ..displaced_state import DisplacedState


@dataclass(slots=True)
class ModeTracking:
    """Track Floquet modes across amplitude sweeps."""

    hilbert_dim: int
    state_indices: Sequence[int]
    _bare_states: np.ndarray = field(init=False, repr=False)
    _excitation_numbers: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._bare_states = np.stack(
            [
                qt.basis(self.hilbert_dim, idx).data.to_array()[:, 0]
                for idx in range(self.hilbert_dim)
            ],
            axis=0,
        ).astype(complex)
        self._excitation_numbers = np.arange(0, self.hilbert_dim)

    @property
    def bare_states(self) -> np.ndarray:
        """Return cached column vectors for the computational basis."""

        return self._bare_states

    def identify(
        self,
        f_modes_energies: Tuple[np.ndarray, qt.Qobj],
        params_0: Tuple[float, float],
        displaced_state: DisplacedState,
        previous_coefficients: np.ndarray,
    ) -> np.ndarray:
        """Return the dominant Floquet mode for each tracked state."""

        f_modes_0, _ = f_modes_energies
        f_modes_cols = np.array(
            [f_modes_0[idx].data.to_array()[:, 0] for idx in range(self.hilbert_dim)], dtype=complex
        ).T
        ideal_displaced_state_array = np.array(
            [
                displaced_state.displaced_state(
                    *params_0, state_idx, previous_coefficients[array_idx]
                )
                .dag()
                .data.to_array()[0]
                for array_idx, state_idx in enumerate(self.state_indices)
            ],
            dtype=complex,
        )
        overlaps = np.einsum("ij,jk->ik", ideal_displaced_state_array, f_modes_cols)
        f_idxs = np.argmax(np.abs(overlaps), axis=1)
        bare_state_overlaps = overlaps[np.arange(len(self.state_indices)), f_idxs]
        ovlps_and_modes = np.zeros((len(self.state_indices), 1 + self.hilbert_dim), dtype=complex)
        ovlps_and_modes[:, 0] = bare_state_overlaps
        selected_modes = f_modes_cols[:, f_idxs].T
        ovlps_and_modes[:, 1:] = np.sign(bare_state_overlaps)[:, None] * selected_modes
        return ovlps_and_modes

    def step_in_amp(
        self,
        f_modes_energies: Tuple[qt.Qobj, np.ndarray],
        prev_f_modes: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reorder Floquet modes and compute diagnostics for a single step."""

        f_modes_0, f_energies_0 = f_modes_energies
        f_modes_0 = np.squeeze([f_mode.data.to_array() for f_mode in f_modes_0])
        all_overlaps = np.abs(np.einsum("ij,kj->ik", np.conj(prev_f_modes), f_modes_0))
        max_idxs = np.argmax(all_overlaps, axis=1)
        f_modes_ordered = f_modes_0[max_idxs]
        avg_excitation = self._calculate_mean_excitation(f_modes_ordered)
        return avg_excitation, f_energies_0[max_idxs], f_modes_ordered

    def _calculate_mean_excitation(self, f_modes_ordered: np.ndarray) -> np.ndarray:
        overlaps_sq = np.abs(np.einsum("ij,kj->ik", self.bare_states, f_modes_ordered)) ** 2
        return np.einsum("ik,i->k", overlaps_sq, self._excitation_numbers)
