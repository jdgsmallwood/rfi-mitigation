import numpy as np
from astropy.time import Time

from lambda_rfi_mitigation.low_rank_sky_experiment import (
    _fit_sky_and_sun,
    beam_weighted_haslam_components,
    beam_weighted_haslam_covariance,
)


def _fourier_mode(nant, index):
    antennas = np.arange(nant)
    return np.exp(2j * np.pi * index * antennas / nant) / np.sqrt(nant)


def test_beam_weighted_haslam_covariance_is_finite_hermitian_psd():
    locations = np.column_stack([
        np.linspace(-9, 9, 6), np.linspace(4, -4, 6)
    ])
    covariance, metadata = beam_weighted_haslam_covariance(
        Time("2026-06-11T05:21:00"), locations, 137.5e6, nside=8
    )

    assert covariance.shape == (6, 6)
    assert np.all(np.isfinite(covariance))
    assert np.allclose(covariance, covariance.conj().T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(covariance)) >= -1e-9 * np.linalg.norm(covariance)
    assert metadata["channel_number"] == 176
    assert metadata["visible_pixel_count"] > 0


def test_apparent_haslam_quarter_turn_preserves_weights_and_radius():
    time = Time("2026-06-11T05:21:00")
    lm, weights, metadata = beam_weighted_haslam_components(
        time, 137.5e6, nside=8
    )
    rotated_lm, rotated_weights, rotated_metadata = (
        beam_weighted_haslam_components(
            time, 137.5e6, nside=8,
            apparent_rotation_quarter_turns=1,
        )
    )

    assert np.allclose(rotated_lm[:, 0], -lm[:, 1])
    assert np.allclose(rotated_lm[:, 1], lm[:, 0])
    assert np.allclose(np.sum(rotated_lm**2, axis=1), np.sum(lm**2, axis=1))
    assert np.array_equal(rotated_weights, weights)
    assert metadata["apparent_rotation_quarter_turns"] == 0
    assert rotated_metadata["apparent_rotation_quarter_turns"] == 1


def test_rotated_haslam_covariance_is_psd_with_unchanged_diagonal():
    locations = np.column_stack([
        np.linspace(-9, 9, 6), np.linspace(4, -4, 6)
    ])
    time = Time("2026-06-11T05:21:00")
    covariance, _ = beam_weighted_haslam_covariance(
        time, locations, 137.5e6, nside=8
    )
    rotated, metadata = beam_weighted_haslam_covariance(
        time, locations, 137.5e6, nside=8,
        apparent_rotation_quarter_turns=1,
    )

    assert np.allclose(rotated, rotated.conj().T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(rotated)) >= -1e-9 * np.linalg.norm(rotated)
    assert np.allclose(np.diag(rotated), np.diag(covariance))
    assert metadata["apparent_rotation_quarter_turns"] == 1


def test_fit_sky_and_sun_recovers_known_amplitudes():
    """Two known templates in, the same two amplitudes out.

    This is the fit behind `trained_haslam_scale` in paper-figures.ipynb; see
    paper/fit_haslam_scale.py.
    """
    rng = np.random.default_rng(0)
    nant = 12
    sky_covariance = _fourier_mode(nant, 1)[:, None] * _fourier_mode(nant, 1).conj()
    sky_covariance = sky_covariance + np.eye(nant)
    sun_direction = np.exp(2j * np.pi * rng.random(nant))
    sun_covariance = np.outer(sun_direction, sun_direction.conj())

    sky_amplitude, sun_amplitude = 42.84, 2.0e4
    covariance = sky_amplitude * sky_covariance + sun_amplitude * sun_covariance

    fitted, residual_fraction, correlation = _fit_sky_and_sun(
        covariance, sky_covariance, sun_direction
    )
    assert np.allclose(fitted, [sky_amplitude, sun_amplitude], rtol=1e-6)
    assert residual_fraction < 1e-8
    assert correlation > 1 - 1e-8


def test_fit_sky_and_sun_amplitudes_are_non_negative():
    """A template that is not there gets amplitude zero, never a negative one."""
    nant = 10
    sky_covariance = np.eye(nant) + 0j
    sun_direction = np.ones(nant, dtype=complex)
    # Data containing only the Sun template, with the sky template absent.
    covariance = 5.0 * np.outer(sun_direction, sun_direction.conj())

    fitted, _, _ = _fit_sky_and_sun(covariance, sky_covariance, sun_direction)
    assert np.all(fitted >= 0.0)
    assert fitted[1] > 0.0
