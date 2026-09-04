import numpy as np

from spatial_filtering_rfi_mitigation.haslam_protection import (
    fixed_rank_subspace_null_batch,
    interpolate_covariances,
    psd_safe_subtract,
    shrink_directional_eigenmodes_to_noise,
)


def _steering(lm, locations, frequency=150e6):
    """Unit-modulus steering vectors, matching protected_pipeline.steering_vectors."""
    wavelength = 299792458.0 / frequency
    phase = 2j * np.pi * (lm @ locations[:, :2].T) / wavelength
    return np.exp(phase)


def test_interpolate_covariances_hits_all_three_nodes():
    nodes = np.arange(3, dtype=float)[:, None, None]
    result = interpolate_covariances(nodes, 7)
    np.testing.assert_allclose(result[[0, 3, 6], 0, 0], [0.0, 1.0, 2.0])


def test_psd_safe_subtract_closes_exactly():
    covariance = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    model = 2.0 * covariance
    residual, removed, cap, valid = psd_safe_subtract(covariance, model)
    assert np.all(valid)
    assert np.all(cap < 1.0)
    np.testing.assert_allclose(residual + removed, covariance)
    assert np.min(np.linalg.eigvalsh(residual)) >= -1e-12


def test_fixed_rank_subspace_null_is_hermitian_and_psd():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(6, 8, 20)) + 1j * rng.normal(size=(6, 8, 20))
    covariance = samples @ np.swapaxes(samples.conj(), -1, -2)
    basis = rng.normal(size=(6, 8, 1)) + 1j * rng.normal(size=(6, 8, 1))
    cleaned, information = fixed_rank_subspace_null_batch(covariance, basis, 3)
    assert information["rank"] == 3
    assert information["protected_dimension"] == 1
    np.testing.assert_allclose(
        cleaned, np.swapaxes(cleaned.conj(), -1, -2), atol=1e-12
    )
    assert np.min(np.linalg.eigvalsh(cleaned)) >= -1e-10


def test_directional_shrink_reduces_response_and_stays_psd():
    direction = np.array([1.0, 0.7j, -0.4, 0.2j], dtype=complex)
    covariance = np.eye(4) + 10.0 * np.outer(direction, direction.conj())
    directions = np.broadcast_to(direction, (12, 4))
    shrunk, information = shrink_directional_eigenmodes_to_noise(
        covariance, directions, 512
    )
    assert information["response_after"] < information["response_before"]
    assert information["minimum_eigenvalue_ratio"] >= 0.0
    assert np.min(np.linalg.eigvalsh(shrunk)) >= -1e-12


# --- Point-source amplitude estimators (invert the dirty-image forward model) --


# --- Whitened protected response invariance / hard null ------------------------


# --- Non-negative integrated template fit -------------------------------------


