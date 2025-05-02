from __future__ import annotations

import jax
from jax import Array
import jax.numpy as jnp
import jax.tree_util as jtu
import dynamiqs as dq
import optimistix as optx

from jax import vmap, jit

from jaxtyping import ArrayLike, Int, Float, Bool, Complex
from typing import List


from .model import Model
from .options import Options


class DisplacedState:
    """Class providing methods for computing displaced states.

    Parameters:
        hilbert_dim: Hilbert space dimension
        model: Model including the Hamiltonian, drive amplitudes, frequencies,
            state indices
        state_indices: States of interest
        options: Options used
    """
    hilbert_dim: int
    state_indices: list
    exponent_pair: dict
    fit_cutoff : int
    fit_cutoff_omega_d : int
    fit_cutoff_amp : int

    def __init__(
        self, hilbert_dim: int, model: Model, state_indices: list, options: Options
    ):
        self.hilbert_dim = hilbert_dim
        self.state_indices = state_indices

        self.fit_cutoff = self.options.fit_cutoff
        self.fit_cutoff_omega_d = min(model.omega_d_values.shape[-1], self.options.fit_cutoff)
        self.fit_cutoff_amp = min(model.drive_amplitudes.shape[-1], self.options.fit_cutoff)

        self.exponent_pair = self._create_exponent_pair()

    def displaced_states(
        self, 
        omega_ds: Float[ArrayLike, 'num_omega_ds'],,
        amps: Float[ArrayLike, 'num_amps'],
        state_indices: int | List[int],
        coefficients: Float[Array, 'num_state_indices hilbert_dim num_omega_ds num_amps num_fit_terms'],
    ) -> dq.QArray:
        """Construct an approximate displaced state, $\left| \tilde{i}(\omega_d, \xi) \right>$.
        $$
        \left| \tilde{i}(\omega_d, \xi) \right> = \sum_{j} \left| j \right> \left< j \middle| \tilde{i}(\omega_d, \xi) \right>
        $$
        where $\left| j \right>$ are the undriven (bare) states and the coefficients \left< j \middle| \tilde{i}(\omega_d, \xi) \right> are given by a polynomial fit. The polynomial is of the form:
        $$
            \left< j \middle| \tilde{i}(\omega_d, \xi) \right> = \sum_{\substack{k \ge 0, l>0 \\ k+l \leq \text{cutoff}}} C_{i,j,k,l}(\omega_d, \xi) \omega_d^k \xi^l
        $$
        where $C_{i,j,k,l}(\omega_d, \xi)$ are the coefficients to be fitted.

        Note: Careful with the shape of the coefficients vs. that of the output.

        Parameters:
            omega_ds: Drive frequency. Shape: (num_omega_ds,)
            amps: Drive amplitude. Shape: (num_amps,)
            state_indices: Indices of the states, $\left| \tilde{i}(\omega_d, \xi) \right>$ that we are interested in.
            coefficients: Coefficients to expand the displaced state in terms of the undriven states. Shape: (num_state_indices, hilbert_dim, num_omega_ds, num_amps, num_fit_terms).

        Returns:
            result: The displaced state(s). Shape = (num_state_indices, num_omega_ds, num_amps, hilbert_dim, 1). Careful, that the first index is the array index, not the state index! For example, if state_indices = [0, 2], then result[0, ...] is the displaced state for state index 0, and result[1, ...] is the displaced state for state index 2.
        """
        # Vectorized bare_same, is a boolean vector with True when idx == state_idx
        # shape should be (num_state_indices, hilbert_dim,)
        vec_bare_same = (dq.fock(self.hilbert_dim, state_indices).to_jax() == 1)[...,0] 

        # Simply vectorize _compute_polynomial the basis vectors! Isn't that so cool? :)
        _vmap_compute_polynomial = vmap(DisplacedState._compute_polynomial,
                                       in_axes=(None, None, -4, None, -1), out_axes=-1)

        # Vectorized function returns shape: (num_state_indices, hilbert_dim, num_omega_ds, num_amps, hilbert_dim). 
        result = _vmap_compute_polynomial(omega_ds, amps, coefficients, self.exponent_pair, 
                                          vec_bare_same)

        # Reshape to (num_state_indices, hilbert_dim, num_omega_ds, num_amps, hilbert_dim, 1).
        return dq.as_qarray(result[..., None])

    @staticmethod
    @jit
    def _compute_polynomial(
        omega_ds: Float[ArrayLike, 'num_omega_ds'],
        amps: Float[ArrayLike, 'num_amps'],
        coefficients: Float[Array, '... num_omega_ds num_amps num_fit_terms'],
        exponent_pair: Int[Array, '2 num_fit_terms'],
        bare_same: Bool[ArrayLike, '...'],
    ) -> ArrayLike:
        """Evaluate the polynomial function to approximate the overlap. This polynomial is of the form:
        $$
            \left< j \middle| \tilde{i}(\omega_d, \xi) \right> = \sum_{\substack{k \ge 0, l>0 \\ k+l \leq \text{cutoff}}} C_{i,j,k,l}(\omega_d, \xi) \omega_d^k \xi^l
        $$
        where $C_{i,j,k,l}$ are the coefficients to be fitted.

        Parameters:
            omega_d: Drive frequency. Shape: (num_omega_ds,)
            amp: Drive amplitude. Shape: (num_amps,)
            coefficients: Coefficients to be fitted, i.e., $C_{k,l}(\omega_d, \xi)$. Shape: (..., num_omega_ds, num_amps, num_fit_terms)
            bare_same: Boolean indicating if the state is the same as the bare state. If yes, return 1 + result, else return result. Shape: (...,)

        Returns:
            result: The result of the polynomial function. Shape: (..., num_omega_ds, num_amps)
        """ 

        # Helper function to compute one term of the polynomial
        def _poly_term(omega_d: ArrayLike, amp: ArrayLike, exponent_pair: ArrayLike,
                       ) -> ArrayLike:
            return (omega_d ** exponent_pair[0]) * (amp ** exponent_pair[1])

        # Cartesian vmap (vmap in reverse order)
        _cvmap_poly_term = vmap(vmap(vmap(_poly_term,
               in_axes=(None,None,-1), out_axes=0), # vmap over exponents, shape: (2, num_fit_terms)
               in_axes=(None,-1,None), out_axes=0), # vmap over amp      , shape: (num_amps,)
               in_axes=(-1,None,None), out_axes=0)  # vmap over omega_d  , shape: (num_omega_d,)

        # Get all the terms i.e. omega_d^k * amp^l, shape: (num_omega_ds, num_amps, num_fit_terms)
        all_ploy_terms = _cvmap_poly_term(omega_ds, amps, exponent_pair)

        # NOTE: coefficients.shape=(num_omega_ds, num_amps, num_fit_terms).
        # This isn't explicitly validated. 
        result = jnp.sum(coefficients * all_ploy_terms, axis = -1)
        return jax.lax.cond(bare_same, lambda: 1.0 + result, lambda: result)


    def _create_exponent_pair(self) -> dict:
        """Create the pair of exponents for the polynomial to be fitted. The polynomial is of the form:
        $$
            \sum_{\substack{k \ge 0, l>0 \\ k+l \leq \text{cutoff}}} C_{k,l} \omega_d^k \xi^l
        $$
        where $C_{k,l}$ are the coefficients to be fitted. This function creates the (j,l) pairs

        We truncate the fit if e.g. there is only a single frequency value to scan over but the fit is nominally set to order four. We additionally eliminate the constant term that should always be either zero or one.

        Returns:
            exponent_pair: A 2D array of shape (2,num_fit_terms) where the first column
                contains the exponents for omega_d and the second column contains the exponents
                for amplitude. The number of terms is determined by the cutoff value.
        """

        # Generate all combinations of indices. 
        # Remove amplitude-independent terms (i.e. exponent for the amp, l == 0).
        # This is enforced by the fact the states have to agree at zero drive strength.
        idx_exp_map = jnp.stack(jnp.meshgrid(
                                    jnp.arange(self.fit_cutoff_omega_d), 
                                    jnp.arange(1, self.fit_cutoff_amp), 
                                    indexing='ij'), 
                                axis=-1).reshape(-1, 2)
        
        # Sort. Introduce a fudge factor to ensure that the sorting is stable
        # Only keep terms where the sum of the exponents is less than the cutoff
        weighted_vals = 1.01 * idx_exp_map[:, 0] + idx_exp_map[:, 1]
        sorted_idxs = jnp.argsort(weighted_vals[weighted_vals <= 1.01*self.fit_cutoff])
        return idx_exp_map[sorted_idxs].T # shape = (2,num_fit_terms)


class DisplacedStateFit(DisplacedState):
    """Methods for fitting an ideal displaced state to calculated Floquet modes."""

    def displaced_states_fit(
        self,
        omega_d_amp_slice: list,
        ovlp_with_bare_states: Array,
        floquet_modes: Array,
    ) -> Array:
        """Perform a fit for the indicated range, ignoring specified modes.

        We loop over all states in state_indices and perform the fit for a given
        amplitude range. We ignore floquet modes (not included in the fit) where
        the corresponding value in ovlp_with_bare_states is below the threshold
        specified in options.

        Parameters:
            omega_d_amp_slice: Pairs of omega_d, amplitude values at which the
                floquet modes have been computed and which we will use as the
                independent variables to fit the Floquet modes
            ovlp_with_bare_states: Bare state overlaps that has shape (w, a, s) where w
                is drive frequency, a is drive amplitude and s is state_indices
            floquet_modes: Floquet mode array with the same shape as
                ovlp_with_bare_states except with an additional trailing dimension h,
                the Hilbert-space dimension.

        Returns:
            Optimized fit coefficients
        """

        def _fit_for_state_idx(array_state_idx: tuple[int, int]) -> Array:
            array_idx, state_idx = array_state_idx
            floquet_mode_for_state = floquet_modes[:, :, array_idx, :]
            mask = (
                jnp.abs(ovlp_with_bare_states[:, :, array_idx].ravel())
                > self.fit_cutoff
            )

            # only fit states that we think haven't run into a nonlinear transition
            omega_d_amp_masked = jnp.where(mask, omega_d_amp_slice, jnp.nan)
            num_coeffs = len(self.exponent_pair)


            def _fit_for_component(state_idx_component):
                floquet_mode_bare_component = floquet_mode_for_state[
                    :, :, state_idx_component
                ].ravel()
                floquet_mode_bare_component_masked = jnp.where(
                    mask, floquet_mode_bare_component, jnp.nan
                )
                bare_same = (state_idx_component == state_idx)
                return self._fit_coefficients_for_component(
                    omega_d_amp_masked, floquet_mode_bare_component_masked, bare_same
                )

            # Return a fitted result only if there are enough states to fit
            # Otherwise, return zeros 
            # Note, warnings are not jittable (see: https://github.com/dynamiqs/dynamiqs/pull/925)
            num_not_nan = jnp.sum(not jnp.isnan(omega_d_amp_masked))
            return jax.lax.cond(
                num_not_nan < len(self.exponent_pair),
                lambda _: jax.vmap(_fit_for_component)(
                            range(self.hilbert_dim)),   # If success, return the result
                lambda _: jnp.zeros((self.hilbert_dim, num_coeffs), 
                                     dtype=complex),    # If failure, return zeros
                operand=None
            )

        return jax.vmap(_fit_for_state_idx)(enumerate(self.state_indices))

    def _fit_coefficients_for_component(
        self,
        omega_d_amp_masked: list,
        floquet_component_masked: Array,
        bare_same: bool,
    ) -> Array:
        """Fit the floquet modes to an "ideal" displaced state based on a polynomial.

        This is done here over the grid specified by omega_d_amp_slice. We ignore
        floquet mode data indicated by mask, where we suspect by looking at overlaps
        with the bare state that we have hit a resonance.
        """
        p0 = jnp.zeros(len(self.exponent_pair))
        # fit the real and imaginary parts of the overlap separately
        popt_r = self._fit_coefficients_factory(
            omega_d_amp_masked, jnp.real(floquet_component_masked), p0, bare_same
        )
        popt_i = self._fit_coefficients_factory(
            omega_d_amp_masked,
            jnp.imag(floquet_component_masked),
            p0,
            False,  # for the imaginary part, constant term should always be zero
        )
        return popt_r + 1j * popt_i

    def _fit_coefficients_factory(
        self, 
        xy_data: list, 
        z_data: Array, 
        p0: tuple | Array, 
        bare_same: bool,
    ) -> Array:
    
        def _residuals(_p0):
            coefficient_fun = jtu.Partial(self._compute_polynomial, *_p0, bare_same=bare_same)
            pred_z_data = jax.vmap(coefficient_fun)(xy_data)
            return jnp.nansum(jnp.abs(z_data - pred_z_data) ** 2)

        # TODO solver to use here? LM?
        # solver = optx.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        solver = optx.BFGS(rtol=1e-8, atol=1e-8)
        opt_result = optx.minimise(_residuals, solver, p0)

        success = jnp.isfinite(opt_result.params).all()

        # Note, warnings are not jittable (see: https://github.com/dynamiqs/dynamiqs/pull/925)
        opt_result_with_fallback = jax.lax.cond(
            success,
            lambda _: opt_result.params,  # If success, return the result
            lambda _: jnp.zeros(len(p0)),  # If failure, return zeros
            operand=None
        )

        return opt_result_with_fallback
