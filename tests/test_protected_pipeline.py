import numpy as np
from astropy.time import Time

from spatial_filtering_rfi_mitigation.protected_pipeline import (
    dirty_image,
    galactic_plane_lm,
    steering_vectors,
)


def test_image_and_overlay_use_the_same_lm_convention():
    rng = np.random.default_rng(61)
    antenna_locations = rng.uniform(-5, 5, size=(20, 2))
    expected = np.array([[0.25, -0.30]])
    frequency = 138e6
    direction = steering_vectors(expected, antenna_locations, frequency)[0]
    image = dirty_image(
        np.outer(direction, direction.conj()), frequency, antenna_locations, npix=128
    )
    row, column = np.unravel_index(np.nanargmax(image), image.shape)
    axis = np.linspace(-1, 1, image.shape[0])

    assert abs(axis[row] - expected[0, 0]) < 2 / 127
    assert abs(axis[column] - expected[0, 1]) < 2 / 127


def test_visible_galactic_plane_is_masked_to_the_lm_horizon():
    plane = galactic_plane_lm(Time("2026-06-11T15:20:00"), npoints=360)
    finite = np.all(np.isfinite(plane), axis=1)

    assert np.any(finite)
    assert np.any(~finite)
    assert np.all(np.sum(plane[finite] ** 2, axis=1) <= 1 + 1e-12)
