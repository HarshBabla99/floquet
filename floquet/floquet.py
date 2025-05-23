from __future__ import annotations

import time

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap
from dynamiqs import floquet as dq_floquet, QArray, TimeQArray
from dynamiqs.method import Tsit5
from dynamiqs.utils import Options as DqOptions

from .displaced_state import DisplacedState, DisplacedStateFit
from .model import Model
from .options import Options
from .utils.file_io import Serializable
from .utils.parallel import parallel_map


class FloquetAnalysis(Serializable):
    """Perform a floquet analysis to identify nonlinear resonances.

    In most workflows, one needs only to call the run() method which performs
    both the displaced state fit and the Blais branch analysis. For an example
    workflow, see the [transmon](../examples/transmon) tutorial.

    Arguments:
        model: Class specifying the model, including the Hamiltonian, drive amplitudes,
            frequencies
        state_indices: State indices of interest. Defaults to [0, 1], indicating the two
            lowest-energy states.
        options: Options for the Floquet analysis.
        init_data_to_save: Initial parameter metadata to save to file. Defaults to None.
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

    def __str__(self) -> str:
        return "Running floquet simulation with parameters: \n" + super().__str__()

    @jit
    def run_one_floquet(self, omega_d_amp: tuple[float, float]) -> tuple[QArray, QArray]:
        """Run one instance of the problem for a pair of drive frequency and amp.

        Returns Floquet modes and quasienergies.

        Parameters:
            omega_d_amp: Pair of drive frequency and amp.
        """
        omega_d, amp = omega_d_amp
        T = 2.0 * jnp.pi / omega_d
        tsave = jnp.linspace(0.0, T, 11)  # Example: 11 time points
        H = self.model.hamiltonian(omega_d, amp)
        result = dq_floquet(H, T, tsave, method=Tsit5(), options=DqOptions())
        return result.modes, result.quasienergies

    @jit
    def identify_floquet_modes(
        self,
        f_modes_energies: tuple[QArray, QArray],
        params_0: tuple[float, float],
        displaced_state: DisplacedState,
        previous_coefficients: QArray,
    ) -> QArray:
        """Return Floquet modes with largest overlap with ideal displaced state.

        Parameters:
            f_modes_energies: Output of self.run_one_floquet(params)
            params_0: (omega_d_0, amp_0) to use for displaced fit
            displaced_state: Instance of DisplacedState
            previous_coefficients: Coefficients from the previous amplitude range
        """
        f_modes_0, _ = f_modes_energies
        overlaps = jnp.einsum("ij,jk->ik", previous_coefficients, f_modes_0)
        max_indices = jnp.argmax(jnp.abs(overlaps), axis=1)
        return f_modes_0[:, max_indices]

    def bare_state_array(self) -> np.ndarray:
        """Return array of bare states.

        Used to specify initial bare states for the Blais branch analysis.
        """
        return np.squeeze(
            np.array(
                [
                    qt.basis(self.hilbert_dim, idx).data.to_array()
                    for idx in range(self.hilbert_dim)
                ]
            )
        )

    def _step_in_amp(
        self, f_modes_energies: tuple[qt.Qobj, np.ndarray], prev_f_modes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform Blais branch analysis.

        Gorgeous in its simplicity. Simply calculate overlaps of new floquet modes with
        those from the previous amplitude step, and order the modes accordingly. So
        ordered, compute the mean excitation number, yielding our branches.
        """
        f_modes_0, f_energies_0 = f_modes_energies
        f_modes_0 = np.squeeze([f_mode.data.to_array() for f_mode in f_modes_0])
        all_overlaps = np.abs(np.einsum("ij,kj->ik", np.conj(prev_f_modes), f_modes_0))
        # assume that prev_f_modes_arr have been previously sorted. Question
        # is which k index has max overlap?
        max_idxs = np.argmax(all_overlaps, axis=1)
        f_modes_ordered = f_modes_0[max_idxs]
        avg_excitation = self._calculate_mean_excitation(f_modes_ordered)
        return avg_excitation, f_energies_0[max_idxs], f_modes_ordered

    def _calculate_mean_excitation(self, f_modes_ordered: np.ndarray) -> np.ndarray:
        """Mean excitation number of ordered floquet modes.

        Based on Blais arXiv:2402.06615, specifically Eq. (12) but going without the
        integral over floquet modes in one period.
        """
        bare_states = self.bare_state_array()
        overlaps_sq = np.abs(np.einsum("ij,kj->ik", bare_states, f_modes_ordered)) ** 2
        # sum over bare excitations weighted by excitation number
        return np.einsum("ik,i->k", overlaps_sq, np.arange(0, self.hilbert_dim))

    def run(self, filepath: str | None = None) -> dict:
        """Perform floquet analysis over range of amplitudes and drive frequencies.

        This function largely performs two calculations. The first is the Xiao analysis
        introduced in https://arxiv.org/abs/2304.13656, fitting the extracted Floquet
        modes to the "ideal" displaced state which does not include resonances by design
        (because we fit to a low order polynomial and ignore any floquet modes with
        overlap with the bare state below a given threshold). This analysis produces the
        "scar" plots. The second is the Blais branch analysis, which tracks the Floquet
        modes by stepping in drive amplitude for a given drive frequency. For this
        reason the code is structured to parallelize over drive frequency, but scans in
        a loop over drive amplitude. This way the two calculations can be performed
        simultaneously.

        A nice bonus is that both of the above mentioned calculations determine
        essentially independently whether a resonance occurs. In the first, it is
        deviation of the Floquet mode from the fitted displaced state. In the second,
        it is branch swapping that indicates a resonance, independent of any fit. Thus
        the two simulations can be used for cross validation of one another.

        We perform these simulations iteratively over the drive amplitudes as specified
        by fit_range_fraction. This is to allow for simulations stretching to large
        drive amplitudes, where the overlap with the bare eigenstate would fall below
        the threshold (due to ac Stark shift) even in the absence of any resonances.
        We thus use the fit from the previous range of drive amplitudes as our new bare
        state.
        """
        print(self)
        start_time = time.time()

        # initialize all arrays that will contain our data
        array_shape = (
            len(self.model.omega_d_values),
            len(self.model.drive_amplitudes),
            len(self.state_indices),
        )
        bare_state_overlaps = np.zeros(array_shape)
        # fit over the full range, using states identified by the fit over
        # intermediate ranges
        intermediate_displaced_state_overlaps = np.zeros(array_shape)
        floquet_modes = np.zeros((*array_shape, self.hilbert_dim), dtype=complex)
        avg_excitation = np.zeros(
            (
                len(self.model.omega_d_values),
                len(self.model.drive_amplitudes),
                self.hilbert_dim,
            )
        )
        quasienergies = np.zeros_like(avg_excitation)

        # for all omega_d, the bare states are identical at zero drive. We define
        # two sets of bare modes (prev_f_modes_arr and disp_coeffs_for_prev_amp)
        # because for the fit calculation, the bare modes are specified as fit
        # coefficients, whereas for the Blais calculation, the bare modes are specified
        # as actual kets.
        prev_f_modes_arr = np.tile(
            self.bare_state_array()[None, :, :], (len(self.model.omega_d_values), 1, 1)
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
            ]
        )
        num_fit_ranges = int(np.ceil(1 / self.options.fit_range_fraction))
        num_amp_pts_per_range = int(
            np.floor(len(self.model.drive_amplitudes) / num_fit_ranges)
        )
        for amp_range_idx in range(num_fit_ranges):
            print(f"calculating for amp_range_idx={amp_range_idx}")
            # edge case if range doesn't fit in neatly
            if amp_range_idx == num_fit_ranges - 1:
                amp_range_idx_final = len(self.model.drive_amplitudes)
            else:
                amp_range_idx_final = (amp_range_idx + 1) * num_amp_pts_per_range
            amp_idxs = [amp_range_idx * num_amp_pts_per_range, amp_range_idx_final]
            # now perform floquet mode calculation for amp_range_idx
            # need to pass forward the floquet modes from the previous amp range
            # which allow us to identify floquet modes that may have been displaced
            # far from the origin
            output = self._floquet_main_for_amp_range(
                amp_idxs, displaced_state, previous_coefficients, prev_f_modes_arr
            )
            (
                bare_state_overlaps_for_range,
                floquet_modes_for_range,
                avg_excitation_for_range,
                quasienergies_for_range,
                prev_f_modes_arr,
            ) = output
            bare_state_overlaps = self._place_into(
                amp_idxs, bare_state_overlaps_for_range, bare_state_overlaps
            )
            floquet_modes = self._place_into(
                amp_idxs, floquet_modes_for_range, floquet_modes
            )
            avg_excitation = self._place_into(
                amp_idxs, avg_excitation_for_range, avg_excitation
            )
            quasienergies = self._place_into(
                amp_idxs, quasienergies_for_range, quasienergies
            )

            # ovlp_with_bare_states is used as a mask for the fit
            ovlp_with_bare_states = displaced_state.overlap_with_bare_states(
                amp_idxs[0], previous_coefficients, floquet_modes_for_range
            )
            omega_d_amp_slice = list(self.model.omega_d_amp_params(amp_idxs))
            # Compute the fitted 'ideal' displaced state, excluding those
            # floquet modes experiencing resonances.
            new_coefficients = displaced_state.displaced_states_fit(
                omega_d_amp_slice, ovlp_with_bare_states, floquet_modes_for_range
            )
            # Compute overlap of floquet modes with ideal displaced state using this
            # new fit. We use this data as the mask for when we compute the coefficients
            # over the whole range. Note that we pass in floquet_modes as
            # opposed to the more restricted floquet_modes_for_range since we
            # use indexing methods inside of overlap_with_displaced_states, so its
            # easier to pass in the whole array.
            overlaps = displaced_state.overlap_with_displaced_states(
                amp_idxs, new_coefficients, floquet_modes
            )
            intermediate_displaced_state_overlaps = self._place_into(
                amp_idxs, overlaps, intermediate_displaced_state_overlaps
            )
            previous_coefficients = new_coefficients
        # The previously extracted coefficients were valid for the amplitude ranges
        # we asked for the fit over. Now armed with with correctly identified floquet
        # modes, we recompute these coefficients over the whole sea of floquet mode data
        # to get a plot that is free from numerical artifacts associated with
        # the fits being slightly different at the boundary of ranges. We utilize the
        # previously computed overlaps of the floquet modes with the displaced states
        # (stored in intermediate_displaced_state_overlaps) to obtain the mask with
        # which we exclude some data from the fit (because we suspect they've hit
        # resonances).
        amp_idxs = [0, len(self.model.drive_amplitudes)]
        omega_d_amp_slice = list(self.model.omega_d_amp_params(amp_idxs))
        full_displaced_fit = displaced_state.displaced_states_fit(
            omega_d_amp_slice, intermediate_displaced_state_overlaps, floquet_modes
        )
        true_overlaps = displaced_state.overlap_with_displaced_states(
            amp_idxs, full_displaced_fit, floquet_modes
        )
        data_dict = {
            "bare_state_overlaps": bare_state_overlaps,
            "fit_data": full_displaced_fit,
            "displaced_state_overlaps": true_overlaps,
            "intermediate_displaced_state_overlaps": intermediate_displaced_state_overlaps,  # noqa E501
            "quasienergies": quasienergies,
            "avg_excitation": avg_excitation,
        }
        if self.options.save_floquet_modes:
            data_dict["floquet_modes"] = floquet_modes
        print(f"finished in {(time.time() - start_time) / 60} minutes")
        if filepath is not None:
            self.write_to_file(filepath, data_dict)
        return data_dict

    @staticmethod
    def _place_into(
        amp_idxs: list, array_for_range: QArray, overall_array: QArray
    ) -> QArray:
        overall_array = overall_array.at[:, amp_idxs[0] : amp_idxs[1]].set(array_for_range)
        return overall_array

    def _floquet_main_for_amp_range(
        self,
        amp_idxs: list,
        displaced_state: DisplacedState,
        previous_coefficients: QArray,
        prev_f_modes_arr: QArray,
    ) -> tuple:
        """Run the Floquet simulation over a specific amplitude range."""
        amp_range_vals = self.model.drive_amplitudes[amp_idxs[0] : amp_idxs[1]]

        def _run_floquet_and_calculate(omega_d):
            amps_for_omega_d = amp_range_vals[:, self.model.omega_d_to_idx(omega_d)]
            return vmap(self.run_one_floquet)((omega_d, amps_for_omega_d))

        floquet_data = vmap(_run_floquet_and_calculate)(self.model.omega_d_values)
        return floquet_data  # Process as needed
