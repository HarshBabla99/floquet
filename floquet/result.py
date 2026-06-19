from __future__ import annotations

import warnings

import numpy as np

from .model import Model
from .options import Options
from .utils.file_io import read_from_file, Serializable


class Result(Serializable):
    r"""Results collected during a Floquet run.

    Parameters:
        hilbert_dim: Hilbert space dimension
        model: Model instance. Stores the Hamiltonian, drive amplitudes, frequencies
        state_indices: States of interest
        options: Options instance
        init_data_to_save: Initial parameter metadata to save to file. Defaults to None.
    """

    def __init__(
        self,
        model: Model,
        state_indices: list,
        hilbert_dim: int,
        options: Options,
        init_data_to_save: dict | None = None,
    ):
        # Shape dimensions
        n_omega_d = len(model.omega_d_values)
        n_amps = model.drive_amplitudes.shape[0]
        n_states = len(state_indices)

        # Parameters
        self.model = model
        self.state_indices = state_indices
        self.hilbert_dim = hilbert_dim
        self.options = options
        self.init_data_to_save = init_data_to_save

        # Results to be saved
        array_shape = (n_omega_d, n_amps, n_states)
        self.bare_state_overlaps = np.zeros(array_shape)
        self.intermediate_displaced_state_overlaps = np.zeros(array_shape)
        self.quasienergies = np.zeros((n_omega_d, n_amps, hilbert_dim))
        self.avg_excitation = np.zeros((n_omega_d, n_amps, hilbert_dim))

        # Optional results
        self.fit_data = np.zeros((n_omega_d, n_amps, hilbert_dim, hilbert_dim))
        self.displaced_state_overlaps = np.zeros((n_omega_d, n_amps, hilbert_dim))

        # Floquet modes; always stored during computation, but only saved if needed
        self._save_floquet_modes = options.save_floquet_modes
        self.floquet_modes = np.zeros((*array_shape, hilbert_dim), dtype=complex)

    def store_for_amp_range(
        self,
        amp_idxs: list,
        bare_state_overlaps: np.ndarray,
        floquet_modes: np.ndarray,
        avg_excitation: np.ndarray,
        quasienergies: np.ndarray,
    ) -> None:
        """Store result from _floquet_main_for_amp_range for one amplitude range."""
        self.bare_state_overlaps[:, amp_idxs[0] : amp_idxs[1]] = bare_state_overlaps
        self.floquet_modes[:, amp_idxs[0] : amp_idxs[1]] = floquet_modes
        self.avg_excitation[:, amp_idxs[0] : amp_idxs[1]] = avg_excitation
        self.quasienergies[:, amp_idxs[0] : amp_idxs[1]] = quasienergies

    def store_intermediate_overlaps(self, amp_idxs: list, overlaps: np.ndarray) -> None:
        """Store intermediate displaced-state overlaps for a single amplitude range."""
        self.intermediate_displaced_state_overlaps[:, amp_idxs[0] : amp_idxs[1]] = (
            overlaps
        )

    def store_overall_fit(
        self, fit_data: np.ndarray, displaced_state_overlaps: np.ndarray
    ) -> None:
        """Store post-processing results computed over the full amplitude range."""
        self.fit_data = fit_data
        self.displaced_state_overlaps = displaced_state_overlaps

    #####
    # Save and load
    #####
    def save(self, filepath: str | None = None) -> dict:
        """Save this class and the corresponding results."""
        data = {
            "bare_state_overlaps": self.bare_state_overlaps,
            "intermediate_displaced_state_overlaps": (
                self.intermediate_displaced_state_overlaps
            ),
            "quasienergies": self.quasienergies,
            "avg_excitation": self.avg_excitation,
            "fit_data": self.fit_data,
            "displaced_state_overlaps": self.displaced_state_overlaps,
        }
        if self._save_floquet_modes:
            data["floquet_modes"] = self.floquet_modes

        # Note: write_to_file only save init_attr of the class
        # So the results must be saved seperately as a dict
        if filepath is not None:
            self.write_to_file(filepath, data)

        return data

    @classmethod
    def load(cls, filepath: str) -> Result:
        result, data = read_from_file(filepath)
        assert isinstance(result, cls), (
            "The loaded object is not an instance of Result."
        )

        result.bare_state_overlaps = data["bare_state_overlaps"]
        result.intermediate_displaced_state_overlaps = data[
            "intermediate_displaced_state_overlaps"
        ]
        result.quasienergies = data["quasienergies"]
        result.avg_excitation = data["avg_excitation"]
        result.fit_data = data["fit_data"]
        result.displaced_state_overlaps = data["displaced_state_overlaps"]

        if result._save_floquet_modes and ("floquet_modes" in data):  # noqa: SLF001
            result.floquet_modes = data["floquet_modes"]

        return result