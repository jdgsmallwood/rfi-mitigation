"""Satellite-blind protection of independently modelled bright-sky modes."""

from __future__ import annotations

from functools import lru_cache
from importlib import resources

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import AltAz, SkyCoord
from astropy.time import Time
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares, lsq_linear

from spatial_filtering_rfi_mitigation.constants import NARRIBRI
from spatial_filtering_rfi_mitigation.protected_pipeline import steering_vectors


HASLAM_SPECTRAL_INDEX = -2.55
HASLAM_NSIDE = 32


@lru_cache(maxsize=None)
def _haslam_pixels(nside: int):
    """Return a degraded Galactic Haslam map and its fixed pixel coordinates."""
    data_root = resources.files("spatial_filtering_rfi_mitigation.data")
    map_path = data_root.joinpath("haslam408_ds_Remazeilles2014.fits")
    temperature = hp.read_map(str(map_path), dtype=np.float64)
    if hp.get_nside(temperature) != int(nside):
        temperature = hp.ud_grade(
            temperature, nside_out=int(nside), order_in="RING",
            order_out="RING", power=0,
        )
    longitude, latitude = hp.pix2ang(
        int(nside), np.arange(hp.nside2npix(int(nside))), lonlat=True
    )
    coordinates = SkyCoord(
        l=longitude * u.deg, b=latitude * u.deg, frame="galactic"
    )
    return temperature, coordinates, float(hp.nside2pixarea(int(nside)))


@lru_cache(maxsize=None)
def _aep_interpolator(channel_number: int):
    """Load the pseudo-Stokes-I element power pattern for one hardware channel."""
    data_root = resources.files("spatial_filtering_rfi_mitigation.data")
    path = data_root.joinpath("aeps", f"aeps_{int(channel_number)}.npz")
    if not path.exists():
        raise FileNotFoundError(f"missing antenna element pattern: {path}")
    with np.load(path) as data:
        l_values = np.asarray(data["lVec"], dtype=float)
        m_values = np.asarray(data["mVec"], dtype=float)
        beam = np.maximum(
            0.5 * (
                np.asarray(data["aep_XX"], dtype=float)
                + np.asarray(data["aep_YY"], dtype=float)
            ),
            0.0,
        )
        model_frequency = float(np.ravel(data["freq"])[0]) * 1e6
    interpolator = RegularGridInterpolator(
        (m_values, l_values), beam, method="linear", bounds_error=False,
        fill_value=0.0,
    )
    return interpolator, model_frequency, str(path)


def beam_weighted_haslam_covariance(
    time_value,
    antenna_locations,
    frequency,
    nside=HASLAM_NSIDE,
    spectral_index=HASLAM_SPECTRAL_INDEX,
    apparent_rotation_quarter_turns=0,
):
    """Synthesize the apparent whole-sky covariance at one time.

    The returned normalization is deliberately in arbitrary temperature-solid-
    angle units.  A single independent scale is fitted from quiet target data;
    the spatial covariance shape is fixed by Haslam and the channel AEP.
    """
    time_value = Time(time_value)
    if not time_value.isscalar:
        raise ValueError("time_value must be scalar")
    antenna_locations = np.asarray(antenna_locations, dtype=float)
    frequency = float(frequency)
    lm, weights, component_metadata = beam_weighted_haslam_components(
        time_value, frequency, nside=nside, spectral_index=spectral_index,
        apparent_rotation_quarter_turns=apparent_rotation_quarter_turns,
    )
    vectors = steering_vectors(lm, antenna_locations, frequency)
    covariance = np.einsum(
        "s,si,sj->ij", weights, vectors, np.conj(vectors), optimize=True,
    )
    covariance = 0.5 * (covariance + covariance.conj().T)
    return covariance, component_metadata


def beam_weighted_haslam_components(
    time_value,
    frequency,
    nside=HASLAM_NSIDE,
    spectral_index=HASLAM_SPECTRAL_INDEX,
    apparent_rotation_quarter_turns=0,
):
    """Return apparent Haslam directions and temperature-solid-angle weights.

    ``apparent_rotation_quarter_turns`` rotates the already beam-weighted
    topocentric sky about zenith.  The weights are deliberately left unchanged,
    giving a negative-control sky with identical total apparent power but moved
    structure.  One positive quarter turn maps ``(l, m)`` to ``(-m, l)``.
    """
    time_value = Time(time_value)
    if not time_value.isscalar:
        raise ValueError("time_value must be scalar")
    frequency = float(frequency)
    quarter_turns = int(apparent_rotation_quarter_turns)
    if quarter_turns != apparent_rotation_quarter_turns:
        raise ValueError("apparent_rotation_quarter_turns must be an integer")
    quarter_turns %= 4
    channel_number = int(np.rint(frequency / 781250.0))
    beam_interpolator, beam_frequency, beam_path = _aep_interpolator(
        channel_number
    )
    if not np.isclose(beam_frequency, frequency, rtol=0, atol=1.0):
        raise ValueError(
            f"AEP frequency {beam_frequency} does not match {frequency} Hz"
        )
    temperature, galactic_coordinates, pixel_area = _haslam_pixels(int(nside))
    altaz = galactic_coordinates.transform_to(
        AltAz(obstime=time_value, location=NARRIBRI)
    )
    altitude = np.asarray(altaz.alt.rad)
    azimuth = np.asarray(altaz.az.rad)
    visible = altitude > 0
    l_value = np.cos(altitude[visible]) * np.sin(azimuth[visible])
    m_value = np.cos(altitude[visible]) * np.cos(azimuth[visible])
    beam = beam_interpolator(np.column_stack([m_value, l_value]))
    scaled_temperature = np.asarray(temperature[visible], dtype=float) * (
        frequency / 408e6
    ) ** float(spectral_index)
    weights = np.maximum(scaled_temperature, 0.0) * beam * pixel_area
    if quarter_turns == 0:
        rotated_l, rotated_m = l_value, m_value
    elif quarter_turns == 1:
        rotated_l, rotated_m = -m_value, l_value
    elif quarter_turns == 2:
        rotated_l, rotated_m = -l_value, -m_value
    else:
        rotated_l, rotated_m = m_value, -l_value
    lm = np.column_stack([rotated_l, rotated_m])
    finite = np.all(np.isfinite(lm), axis=1) & np.isfinite(weights) & (weights > 0)
    return lm[finite], weights[finite], {
        "channel_number": channel_number,
        "aep_path": beam_path,
        "nside": int(nside),
        "spectral_index": float(spectral_index),
        "apparent_rotation_quarter_turns": quarter_turns,
        "visible_pixel_count": int(np.count_nonzero(finite)),
    }


def beam_weighted_haslam_covariances(
    times, antenna_locations, frequency, **kwargs
):
    """Vector form of :func:`beam_weighted_haslam_covariance`."""
    times = Time(times)
    output = []
    metadata = None
    for time_value in times:
        covariance, metadata = beam_weighted_haslam_covariance(
            time_value, antenna_locations, frequency, **kwargs
        )
        output.append(covariance)
    return np.asarray(output), metadata


def _fit_sky_and_sun(covariance, sky_covariance, sun_direction):
    """Robustly fit non-negative sky and Sun amplitudes to complex cross-visibilities."""
    covariance = np.asarray(covariance, dtype=np.complex128)
    sky_covariance = np.asarray(sky_covariance, dtype=np.complex128)
    sun_direction = np.asarray(sun_direction, dtype=np.complex128)
    nant = len(sun_direction)
    row, column = np.triu_indices(nant, 1)
    sun_covariance = np.outer(sun_direction, sun_direction.conj())
    complex_design = np.column_stack([
        sky_covariance[row, column], sun_covariance[row, column]
    ])
    target = covariance[row, column]
    design = np.vstack([complex_design.real, complex_design.imag])
    values = np.r_[target.real, target.imag]
    initial = lsq_linear(
        design, values, bounds=(0, np.inf), lsmr_tol="auto", max_iter=300
    ).x
    initial_residual = design @ initial - values
    robust_scale = max(
        1.4826 * float(np.median(np.abs(initial_residual))),
        np.finfo(float).eps,
    )
    fitted = least_squares(
        lambda parameters: design @ parameters - values,
        initial, bounds=(0, np.inf), loss="soft_l1", f_scale=robust_scale,
        max_nfev=200,
    ).x
    prediction = design @ fitted
    residual_fraction = float(
        np.linalg.norm(values - prediction)
        / max(np.linalg.norm(values), np.finfo(float).eps)
    )
    correlation = (
        float(np.corrcoef(values, prediction)[0, 1])
        if np.std(values) > 0 and np.std(prediction) > 0 else np.nan
    )
    return fitted, residual_fraction, correlation
