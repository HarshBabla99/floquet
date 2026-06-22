import pathlib
import warnings

import numpy as np
import pytest
import qutip as qt

from floquet import Model, Options, Result


def _make_model(
    omega_d_values: np.ndarray,
    n_amps: int = 5,
    hilbert_dim: int = 4,
    H0: qt.Qobj | None = None,
    H1: qt.Qobj | None = None,
) -> Model:
    if H0 is None:
        H0 = qt.Qobj(np.diag(np.arange(hilbert_dim, dtype=float)))
    if H1 is None:
        H1 = qt.Qobj(np.eye(hilbert_dim))
    return Model(H0, H1, omega_d_values, np.ones((n_amps, len(omega_d_values))))


@pytest.fixture
def simple_model() -> Model:
    return _make_model(np.linspace(1.0, 3.0, 3))


# ---------------------------------------------------------------------------
# Fixtures shared by __add__ tests
# ---------------------------------------------------------------------------

@pytest.fixture
def shared_hamiltonians():
    H0 = qt.Qobj(np.diag(np.arange(4, dtype=float)))
    H1 = qt.Qobj(np.eye(4))
    return H0, H1


@pytest.fixture
def adjacent_models(shared_hamiltonians):
    """Two models whose omega_d values form a single equally-spaced sequence."""
    H0, H1 = shared_hamiltonians
    model_a = _make_model(np.array([1.0, 2.0, 3.0]), H0=H0, H1=H1)
    model_b = _make_model(np.array([4.0, 5.0]), H0=H0, H1=H1)
    return model_a, model_b


# ---------------------------------------------------------------------------
# store_for_amp_range
# ---------------------------------------------------------------------------


def test_store_for_amp_range_writes_correct_slice(simple_model):
    result = Result(simple_model, [0, 1], 4, Options())
    n_omega_d, n_slice = 3, 3
    amp_idxs = [1, 4]

    rng = np.random.default_rng(0)
    overlaps = rng.random((n_omega_d, n_slice, 2))
    modes = rng.random((n_omega_d, n_slice, 2, 4)) + 0j
    avg_exc = rng.random((n_omega_d, n_slice, 4))
    quasi = rng.random((n_omega_d, n_slice, 4))

    result.store_for_amp_range(amp_idxs, overlaps, modes, avg_exc, quasi)

    assert np.allclose(result.bare_state_overlaps[:, 1:4], overlaps)
    assert np.allclose(result.floquet_modes[:, 1:4], modes)
    assert np.allclose(result.avg_excitation[:, 1:4], avg_exc)
    assert np.allclose(result.quasienergies[:, 1:4], quasi)
    # untouched slices remain zero
    assert np.allclose(result.bare_state_overlaps[:, :1], 0)
    assert np.allclose(result.bare_state_overlaps[:, 4:], 0)


# ---------------------------------------------------------------------------
# store_intermediate_overlaps
# ---------------------------------------------------------------------------


def test_store_intermediate_overlaps_writes_correct_slice(simple_model):
    result = Result(simple_model, [0, 1], 4, Options())
    amp_idxs = [0, 3]
    overlaps = np.random.default_rng(1).random((3, 3, 2))

    result.store_intermediate_overlaps(amp_idxs, overlaps)

    assert np.allclose(result.intermediate_displaced_state_overlaps[:, :3], overlaps)
    assert np.allclose(result.intermediate_displaced_state_overlaps[:, 3:], 0)


# ---------------------------------------------------------------------------
# store_overall_fit
# ---------------------------------------------------------------------------


def test_store_overall_fit(simple_model):
    result = Result(simple_model, [0, 1], 4, Options())
    rng = np.random.default_rng(2)
    fit = rng.random(result.fit_data.shape)
    ovlps = rng.random(result.displaced_state_overlaps.shape)

    result.store_overall_fit(fit, ovlps)

    assert np.allclose(result.fit_data, fit)
    assert np.allclose(result.displaced_state_overlaps, ovlps)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


def test_save_load_round_trip(tmp_path: pathlib.Path, simple_model):
    result = Result(simple_model, [0, 1], 4, Options())
    rng = np.random.default_rng(3)
    result.bare_state_overlaps[:] = rng.random(result.bare_state_overlaps.shape)
    result.quasienergies[:] = rng.random(result.quasienergies.shape)
    result.fit_data[:] = rng.random(result.fit_data.shape)

    fp = tmp_path / "result.h5py"
    result.save(fp)
    loaded = Result.load(fp)

    for attr in ("bare_state_overlaps", "quasienergies", "fit_data"):
        assert np.allclose(getattr(loaded, attr), getattr(result, attr)), attr
    # floquet_modes not included in saved data → stays zero after load
    assert np.allclose(loaded.floquet_modes, 0)


def test_save_load_with_floquet_modes(tmp_path: pathlib.Path, simple_model):
    result = Result(simple_model, [0, 1], 4, Options(save_floquet_modes=True))
    result.floquet_modes[:] = (
        np.random.default_rng(4).random(result.floquet_modes.shape) + 0j
    )

    fp = tmp_path / "result_modes.h5py"
    result.save(fp)
    loaded = Result.load(fp)

    assert loaded._save_floquet_modes
    assert np.allclose(loaded.floquet_modes, result.floquet_modes)


# ---------------------------------------------------------------------------
# __add__ — merge along omega_d axis
# ---------------------------------------------------------------------------


def test_add_concatenates_along_omega_d(adjacent_models):
    model_a, model_b = adjacent_models
    res_a = Result(model_a, [0, 1], 4, Options())
    res_b = Result(model_b, [0, 1], 4, Options())

    rng = np.random.default_rng(5)
    res_a.bare_state_overlaps[:] = rng.random(res_a.bare_state_overlaps.shape)
    res_b.bare_state_overlaps[:] = rng.random(res_b.bare_state_overlaps.shape)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        merged = res_a + res_b

    assert len(w) == 1
    assert "refit" in str(w[0].message).lower()
    assert merged.bare_state_overlaps.shape == (5, 5, 2)
    expected = np.concatenate(
        [res_a.bare_state_overlaps, res_b.bare_state_overlaps], axis=0
    )
    assert np.allclose(merged.bare_state_overlaps, expected)


def test_add_floquet_modes_flag_is_or(adjacent_models):
    """_save_floquet_modes on merged result is OR of the two inputs."""
    model_a, model_b = adjacent_models
    res_a = Result(model_a, [0, 1], 4, Options(save_floquet_modes=False))
    res_b = Result(model_b, [0, 1], 4, Options(save_floquet_modes=True))

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        merged = res_a + res_b

    assert merged._save_floquet_modes is True


def test_add_raises_on_state_indices_mismatch(simple_model):
    res_a = Result(simple_model, [0, 1], 4, Options())
    res_b = Result(simple_model, [0, 2], 4, Options())
    with pytest.raises(AssertionError, match="state_indices"):
        _ = res_a + res_b


def test_add_raises_on_hilbert_dim_mismatch(simple_model):
    res_a = Result(simple_model, [0, 1], 4, Options())
    res_b = Result(simple_model, [0, 1], 6, Options())
    with pytest.raises(AssertionError, match="hilbert_dim"):
        _ = res_a + res_b


def test_add_raises_on_unequal_omega_d_spacing(shared_hamiltonians):
    H0, H1 = shared_hamiltonians
    # gap of 2 between [1,2,3] and [5,6] breaks equal spacing
    res_a = Result(_make_model(np.array([1.0, 2.0, 3.0]), H0=H0, H1=H1), [0, 1], 4, Options())
    res_b = Result(_make_model(np.array([5.0, 6.0]), H0=H0, H1=H1), [0, 1], 4, Options())
    with pytest.raises(AssertionError, match="equally spaced"):
        _ = res_a + res_b


# ---------------------------------------------------------------------------
# __and__ — average displaced state overlaps
# ---------------------------------------------------------------------------


def test_and_averages_overlaps(simple_model):
    res_a = Result(simple_model, [0, 1], 4, Options())
    res_b = Result(simple_model, [0, 1], 4, Options())

    rng = np.random.default_rng(6)
    res_a.bare_state_overlaps[:] = rng.random(res_a.bare_state_overlaps.shape)
    res_b.bare_state_overlaps[:] = rng.random(res_b.bare_state_overlaps.shape)
    res_a.displaced_state_overlaps[:] = rng.random(res_a.displaced_state_overlaps.shape)
    res_b.displaced_state_overlaps[:] = rng.random(res_b.displaced_state_overlaps.shape)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        merged = res_a & res_b

    assert len(w) == 1
    assert np.allclose(
        merged.bare_state_overlaps,
        np.mean([res_a.bare_state_overlaps, res_b.bare_state_overlaps], axis=0),
    )
    assert np.allclose(
        merged.displaced_state_overlaps,
        np.mean([res_a.displaced_state_overlaps, res_b.displaced_state_overlaps], axis=0),
    )
    assert merged._save_floquet_modes is False


def test_and_raises_on_omega_d_mismatch(shared_hamiltonians):
    H0, H1 = shared_hamiltonians
    res_a = Result(_make_model(np.array([1.0, 2.0, 3.0]), H0=H0, H1=H1), [0, 1], 4, Options())
    res_b = Result(_make_model(np.array([4.0, 5.0, 6.0]), H0=H0, H1=H1), [0, 1], 4, Options())
    with pytest.raises(AssertionError, match="omega_d_values"):
        _ = res_a & res_b
