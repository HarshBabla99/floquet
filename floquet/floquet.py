"""Public entry points for running Floquet simulations.

The refactored module keeps the :class:`FloquetAnalysis` facade intact while the
heavy lifting is handled by helpers in :mod:`floquet.analysis`.  The goal is a
cleaner orchestration layer that exposes the same ``run()`` contract used across
docs, notebooks, and tests.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import qutip as qt

from .analysis import FrequencySweepExecutor, ModeTracking, SimulationArrays, partition_amplitudes
from .analysis.results import FloquetRunResult
from .displaced_state import DisplacedState, DisplacedStateFit
from .model import Model
from .options import Options
from .utils.file_io import Serializable


class FloquetAnalysis(Serializable):
    """Perform a floquet analysis to identify nonlinear resonances.

    In most workflows, one needs only to call the ``run`` method which performs
    both the displaced state fit and the Blais branch analysis. For an example
    workflow, see the [transmon](../examples/transmon) tutorial.

    Parameters:
        model: Class specifying the model, including the Hamiltonian, drive
            amplitudes, frequencies.
        state_indices: State indices of interest. Defaults to ``[0, 1]``,
            indicating the two lowest-energy states.
        options: Options for the Floquet analysis.
            ??? info "Detailed `opt_options` API"
                - fit_range_fraction (`float`, default: 1.0): Fraction of the
                  amplitude range to sweep over before changing the definition
                  of the bare state to that of the fitted state from the previous
                  range.
                - floquet_sampling_time_fraction (`float`, default: 0.0): What
                  point of the drive period we want to sample the Floquet modes.
                - fit_cutoff (`int`, default: 4): Cutoff for the fit polynomial
                  of the displaced state.
                - overlap_cutoff (`float`, default: 0.8): Cutoff for fitting
                  overlaps. Modes with overlap with the "bare" state below this
                  cutoff are excluded from the fit.
                - nsteps (`int`, default: 30_000): QuTiP integration parameter,
                  number of steps the solver can take.
                - num_cpus (`int`, default: 1): Number of CPUs to use in parallel
                  computation of Floquet modes over the different values of
                  ``omega_d`` and amplitude.
                - save_floquet_modes (`bool`, default: False): Whether to save
                  the extracted Floquet modes themselves.
        init_data_to_save: Initial parameter metadata to save to file. Defaults
            to ``None``.
    """

    def __init__(
        self,
        model: Model,
        state_indices: list | None = None,
        options: Options = Options(),  # noqa B008
        init_data_to_save: dict | None = None,
    ):
        if state_indices is None:
            state_indices = [0, 1]
        self.model = model
        self.state_indices = state_indices
        self.options = options
        self.init_data_to_save = init_data_to_save
        self.hilbert_dim = model.H0.shape[0]
        self.mode_tracker = ModeTracking(self.hilbert_dim, self.state_indices)

    def __str__(self) -> str:
        return "Running floquet simulation with parameters: \n" + super().__str__()

    def run_one_floquet(
        self, omega_d_amp: Tuple[float, float]
    ) -> Tuple[np.ndarray, qt.Qobj]:
        """Run one instance of the problem for a pair of drive frequency and amp."""

        omega_d, _ = omega_d_amp
        T = 2.0 * np.pi / omega_d
        fbasis = qt.FloquetBasis(
            self.model.hamiltonian(omega_d_amp),
            T,
            options={"nsteps": self.options.nsteps},  # type: ignore[arg-type]
        )
        f_modes_0 = fbasis.mode(0.0)
        f_energies = fbasis.e_quasi
        sampling_time = self.options.floquet_sampling_time_fraction * T % T
        f_modes_t = fbasis.mode(sampling_time) if sampling_time != 0.0 else f_modes_0
        return f_modes_t, f_energies

    def run(self, filepath: str | None = None) -> Dict[str, np.ndarray]:
        """Perform Floquet analysis across drive frequencies and amplitudes.

        The workflow mirrors the original implementation: for each drive-frequency
        slice we fit displaced states (Xiao et al.) while simultaneously tracking
        branch swaps (Blais et al.) across incremental amplitude windows.  The heavy
        lifting now lives in :mod:`floquet.analysis`, but the public contract and
        numerical outputs remain unchanged.
        """

        print(self)
        start_time = time.time()

        num_omega = len(self.model.omega_d_values)
        num_amp = len(self.model.drive_amplitudes)
        arrays = SimulationArrays(
            num_omega=num_omega,
            num_amp=num_amp,
            num_states=len(self.state_indices),
            hilbert_dim=self.hilbert_dim,
            store_modes=self.options.save_floquet_modes,
        )

        prev_f_modes_arr = np.tile(
            self.mode_tracker.bare_states[None, :, :], (num_omega, 1, 1)
        )
        displaced_state = DisplacedStateFit(
            hilbert_dim=self.hilbert_dim,
            model=self.model,
            state_indices=self.state_indices,
            options=self.options,
        )
        previous_coefficients = np.array(
            [
                displaced_state.bare_state_coefficients(state_idx)
                for state_idx in self.state_indices
            ],
            dtype=complex,
        )

        executor = FrequencySweepExecutor(
            mode_tracker=self.mode_tracker,
            model=self.model,
            displaced_state=displaced_state,
            run_one_floquet=self.run_one_floquet,
            num_cpus=self.options.num_cpus,
        )
        windows = partition_amplitudes(num_amp, self.options.fit_range_fraction)

        for window in windows:
            window_result = executor.execute(
                window, previous_coefficients, prev_f_modes_arr
            )
            arrays.update_window(window, window_result)
            prev_f_modes_arr = window_result.final_modes

            overlaps_with_bare = displaced_state.overlap_with_bare_states(
                window.start, previous_coefficients, window_result.floquet_modes
            )
            omega_d_amp_slice = list(
                self.model.omega_d_amp_params([window.start, window.stop])
            )
            new_coefficients = displaced_state.displaced_states_fit(
                omega_d_amp_slice, overlaps_with_bare, window_result.floquet_modes
            )
            overlaps_with_displaced = displaced_state.overlap_with_displaced_states(
                [window.start, window.stop], new_coefficients, arrays.floquet_modes
            )
            arrays.intermediate_displaced_state_overlaps[:, window.slice] = (
                overlaps_with_displaced
            )
            previous_coefficients = new_coefficients

        full_range = [0, num_amp]
        omega_d_amp_slice = list(self.model.omega_d_amp_params(full_range))
        full_displaced_fit = displaced_state.displaced_states_fit(
            omega_d_amp_slice,
            arrays.intermediate_displaced_state_overlaps,
            arrays.floquet_modes,
        )
        true_overlaps = displaced_state.overlap_with_displaced_states(
            full_range, full_displaced_fit, arrays.floquet_modes
        )

        result: FloquetRunResult = arrays.to_result(full_displaced_fit, true_overlaps)
        data_dict = result.to_dict(self.options.save_floquet_modes)

        print(f"finished in {(time.time() - start_time) / 60} minutes")
        if filepath is not None:
            self.write_to_file(filepath, data_dict)
        return data_dict

    def identify_floquet_modes(
        self,
        f_modes_energies: tuple[np.ndarray, qt.Qobj],
        params_0: tuple[float, float],
        displaced_state: DisplacedState,
        previous_coefficients: np.ndarray,
    ) -> np.ndarray:
        """Compatibility wrapper delegating to :class:`ModeTracking`."""

        return self.mode_tracker.identify(
            f_modes_energies, params_0, displaced_state, previous_coefficients
        )
