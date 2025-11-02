"""Parallel execution helpers for amplitude windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import numpy as np

from ..utils.parallel import parallel_map
from .mode_tracking import ModeTracking
from .range import AmplitudeWindow
from .results import WindowResult

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from ..displaced_state import DisplacedState
    from ..model import Model


@dataclass(slots=True)
class FrequencyTask:
    """Callable payload executed for a single drive frequency."""

    omega_d: float
    omega_idx: int
    amps_for_frequency: np.ndarray
    params_0: Tuple[float, float]
    run_one_floquet: callable
    mode_tracker: ModeTracking
    displaced_state: "DisplacedState"
    previous_coefficients: np.ndarray
    prev_f_modes_for_frequency: np.ndarray

    def __call__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        num_states = len(self.mode_tracker.state_indices)
        window_length = len(self.amps_for_frequency)
        avg_excitation_arr = np.zeros((window_length, self.mode_tracker.hilbert_dim))
        quasienergies_arr = np.zeros_like(avg_excitation_arr)
        ovlps_and_modes_arr = np.zeros(
            (
                window_length,
                num_states,
                1 + self.mode_tracker.hilbert_dim,
            ),
            dtype=complex,
        )
        prev_f_modes_for_omega_d = self.prev_f_modes_for_frequency
        for amp_idx, amp in enumerate(self.amps_for_frequency):
            params = (self.omega_d, amp)
            f_modes_energies = self.run_one_floquet(params)
            ovlps_and_modes = self.mode_tracker.identify(
                f_modes_energies,
                self.params_0,
                self.displaced_state,
                self.previous_coefficients,
            )
            ovlps_and_modes_arr[amp_idx] = ovlps_and_modes
            avg_excitation, quasi_es, new_f_modes_arr = self.mode_tracker.step_in_amp(
                f_modes_energies, prev_f_modes_for_omega_d
            )
            avg_excitation_arr[amp_idx] = np.real(avg_excitation)
            quasienergies_arr[amp_idx] = np.real(quasi_es)
            prev_f_modes_for_omega_d = new_f_modes_arr
        return ovlps_and_modes_arr, avg_excitation_arr, quasienergies_arr, prev_f_modes_for_omega_d


def _execute_task(task: FrequencyTask):  # pragma: no cover - trivial wrapper
    return task()


@dataclass(slots=True)
class FrequencySweepExecutor:
    """Run Floquet mode extraction for a single amplitude window."""

    mode_tracker: ModeTracking
    model: "Model"
    displaced_state: "DisplacedState"
    run_one_floquet: callable
    num_cpus: int

    def execute(
        self,
        window: AmplitudeWindow,
        previous_coefficients: np.ndarray,
        prev_f_modes_arr: np.ndarray,
    ) -> WindowResult:
        amp_range_vals = self.model.drive_amplitudes[window.slice]
        tasks = [
            FrequencyTask(
                omega_d=omega_d,
                omega_idx=omega_idx,
                amps_for_frequency=amp_range_vals[:, omega_idx],
                params_0=(omega_d, amp_range_vals[0, omega_idx]),
                run_one_floquet=self.run_one_floquet,
                mode_tracker=self.mode_tracker,
                displaced_state=self.displaced_state,
                previous_coefficients=previous_coefficients,
                prev_f_modes_for_frequency=prev_f_modes_arr[omega_idx],
            )
            for omega_idx, omega_d in enumerate(self.model.omega_d_values)
        ]
        floquet_data = list(parallel_map(self.num_cpus, _execute_task, tasks))
        (
            all_modes_quasies_ovlps,
            all_avg_excitation,
            all_quasienergies,
            f_modes_last_amp,
        ) = list(zip(*floquet_data, strict=True))
        floquet_mode_array = np.array(all_modes_quasies_ovlps, dtype=complex).reshape(
            (
                len(self.model.omega_d_values),
                window.width,
                len(self.mode_tracker.state_indices),
                1 + self.mode_tracker.hilbert_dim,
            )
        )
        f_modes_last_amp = np.array(f_modes_last_amp, dtype=complex).reshape(
            (len(self.model.omega_d_values), self.mode_tracker.hilbert_dim, self.mode_tracker.hilbert_dim)
        )
        all_avg_excitation = np.array(all_avg_excitation).reshape(
            (len(self.model.omega_d_values), window.width, self.mode_tracker.hilbert_dim)
        )
        all_quasienergies = np.array(all_quasienergies).reshape(
            (len(self.model.omega_d_values), window.width, self.mode_tracker.hilbert_dim)
        )
        bare_state_overlaps = np.abs(floquet_mode_array[..., 0])
        floquet_modes = floquet_mode_array[..., 1:]
        return WindowResult(
            bare_state_overlaps=bare_state_overlaps,
            floquet_modes=floquet_modes,
            avg_excitation=all_avg_excitation,
            quasienergies=all_quasienergies,
            final_modes=f_modes_last_amp,
        )
