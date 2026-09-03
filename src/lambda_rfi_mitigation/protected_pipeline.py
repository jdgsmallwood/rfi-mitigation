"""Streaming analysis for the direction-protected satellite eigenfilter.

The filter itself knows only the protected Sun direction.  Satellite ephemeris
information enters later, in :func:`evaluate_satellite`, and is used solely to
choose and score validation snapshots.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import astropy.units as u
import h5py as h5
import numpy as np
from astropy.coordinates import (
    AltAz,
    CartesianRepresentation,
    ITRS,
    SkyCoord,
    TEME,
    get_sun,
)
from astropy.time import Time
from loguru import logger

from lambda_rfi_mitigation import utils, vistools
from lambda_rfi_mitigation.constants import NARRIBRI, c
from lambda_rfi_mitigation.modelling import calc_lmn


DEFAULT_TLE = (
    "1 41184U 15081F   26161.08878541  .00000540  00000-0  15048-3 0  9990",
    "2 41184  47.0001 303.8133 0003307 186.7885 173.2956 14.58893590557911",
)


@dataclass(frozen=True)
class ProtectedRunConfig:
    visibility_file: str
    gain_file: str
    start_index: int = 30000
    stop_index: Optional[int] = None
    target_channel: int = 4
    clean_channels: tuple = (2, 5)
    flag_antennas: tuple = (8, 9, 12, 14, 22, 33, 36)
    polarisations: tuple = ("XX", "YY")
    batch_size: int = 5000
    tw_margin: float = 5.0
    leshem_rcond: float = 1e-3
    leshem_operator_samples: int = 256
    snapshot_window: int = 2000
    snapshot_pixels: int = 256
    ephemeris_time_offset_s: float = 0.0
    tle: tuple = DEFAULT_TLE


@dataclass
class Observation:
    config: ProtectedRunConfig
    gains: dict
    time: Time
    frequencies: np.ndarray
    selected_start: int
    selected_stop: int
    file_ntime: int
    n_channels: int
    baseline_valid: np.ndarray
    antenna1: np.ndarray
    antenna2: np.ndarray
    antenna_ids: np.ndarray
    good_antennas: np.ndarray
    antenna_locations_all: np.ndarray
    antenna_locations: np.ndarray
    baseline_gain_i: np.ndarray
    baseline_gain_j: np.ndarray
    snapshots_per_integration: int
    gain_amplitude_xx: np.ndarray
    gain_amplitude_yy: np.ndarray

    @property
    def ntime(self):
        return len(self.time)

    @property
    def target_frequency(self):
        return float(self.frequencies[self.config.target_channel])

    def _selection_to_hdf5(self, selection):
        if isinstance(selection, slice):
            start, stop, step = selection.indices(self.ntime)
            local = np.arange(start, stop, step, dtype=int)
        else:
            local = np.asarray(selection, dtype=int)
            if local.ndim != 1:
                raise ValueError("time selection must be one-dimensional")
            if np.any((local < 0) | (local >= self.ntime)):
                raise IndexError("time selection is outside the selected capture")
        if local.size == 0 or np.any(np.diff(local) <= 0):
            raise ValueError("time indices must be non-empty and increasing")
        absolute = self.selected_start + local
        if np.all(np.diff(absolute) == 1):
            h5_selection = slice(int(absolute[0]), int(absolute[-1]) + 1)
        else:
            h5_selection = absolute
        logger.info(f"returning {local.size} time steps...")
        return h5_selection, local.size

    def load_channel(self, selection, channel):
        """Load one calibrated pseudo-Stokes-I covariance cube."""
        h5_selection, ntime = self._selection_to_hdf5(selection)
        selected_ids = self.antenna_ids[self.good_antennas]
        selected_set = set(selected_ids.tolist())
        baseline_keep = np.array(
            [
                int(a1) in selected_set and int(a2) in selected_set
                for a1, a2 in zip(self.antenna1, self.antenna2)
            ]
        )
        antenna1 = self.antenna1[baseline_keep]
        antenna2 = self.antenna2[baseline_keep]
        local_index = {int(ant): i for i, ant in enumerate(selected_ids)}
        row = np.array([local_index[int(ant)] for ant in antenna1])
        column = np.array([local_index[int(ant)] for ant in antenna2])

        pseudo_i = None
        with h5.File(self.config.visibility_file, "r") as handle:
            dataset = handle["visibilities"]
            for pol_index, pol in enumerate(self.config.polarisations):
                raw = dataset[h5_selection, int(channel), :, pol_index, pol_index, :]
                visibility = raw[..., 0] + 1j * raw[..., 1]
                visibility = visibility[:, self.baseline_valid][:, baseline_keep]
                denominator = self.gains[pol][
                    int(channel), self.baseline_gain_i[baseline_keep]
                ] * np.conj(
                    self.gains[pol][int(channel), self.baseline_gain_j[baseline_keep]]
                )
                np.divide(visibility, denominator[None], out=visibility)
                if pseudo_i is None:
                    pseudo_i = visibility
                else:
                    pseudo_i += visibility
        pseudo_i *= 1.0 / len(self.config.polarisations)

        nant = selected_ids.size
        result = np.zeros((ntime, nant, nant), dtype=np.complex64)
        result[:, row, column] = pseudo_i
        result[:, column, row] = np.conj(pseudo_i)
        return np.ascontiguousarray(result)


def prepare_observation(config, verbose=True):
    """Read metadata, gains and fixed geometry without loading visibility cubes."""
    gains = utils.read_hdf5_gains(config.gain_file, list(config.polarisations))
    index_range = (config.start_index, config.stop_index)
    times, frequencies = utils.read_hdf5_time_freq(
        config.visibility_file, verbose=verbose, index_range=index_range
    )
    with h5.File(config.visibility_file, "r") as handle:
        visibility_shape = handle["visibilities"].shape
        baseline_ids = handle["baseline_ids"][:]
        packets_per_visibility = float(handle["vis_missing_nums"][0][1])
    file_ntime, n_channels = int(visibility_shape[0]), int(visibility_shape[1])
    selected_start = int(config.start_index)
    selected_stop = (
        file_ntime
        if config.stop_index is None
        else min(int(config.stop_index), file_ntime)
    )
    if len(times) != selected_stop - selected_start:
        raise ValueError("time metadata does not match the selected visibility range")

    antenna1_all, antenna2_all = utils.split_baseline(baseline_ids)
    baseline_valid = (
        (baseline_ids > 0)
        & (antenna1_all >= 1)
        & (antenna1_all <= 36)
        & (antenna2_all >= 1)
        & (antenna2_all <= 36)
    )
    antenna1 = antenna1_all[baseline_valid].astype(int)
    antenna2 = antenna2_all[baseline_valid].astype(int)
    antenna_ids = np.unique(np.r_[antenna1, antenna2])
    telescope = utils.make_telescope_model(
        antenna_ids, telescope="LAMBDA36", verbose=verbose
    )
    antenna_locations_all = np.column_stack([telescope.east, telescope.north])
    good_antennas = ~np.isin(antenna_ids, np.asarray(config.flag_antennas))

    gain_index = {int(ant): i for i, ant in enumerate(antenna_ids)}
    baseline_gain_i = np.array([gain_index[int(ant)] for ant in antenna1])
    baseline_gain_j = np.array([gain_index[int(ant)] for ant in antenna2])
    antenna_groups = math.ceil(antenna_ids.size / 10)
    time_packets = packets_per_visibility / n_channels / antenna_groups
    snapshots_per_integration = int(time_packets * 64)

    return Observation(
        config=config,
        gains=gains,
        time=times,
        frequencies=np.asarray(frequencies),
        selected_start=selected_start,
        selected_stop=selected_stop,
        file_ntime=file_ntime,
        n_channels=n_channels,
        baseline_valid=baseline_valid,
        antenna1=antenna1,
        antenna2=antenna2,
        antenna_ids=antenna_ids,
        good_antennas=good_antennas,
        antenna_locations_all=antenna_locations_all,
        antenna_locations=antenna_locations_all[good_antennas],
        baseline_gain_i=baseline_gain_i,
        baseline_gain_j=baseline_gain_j,
        snapshots_per_integration=snapshots_per_integration,
        gain_amplitude_xx=np.nanmedian(np.abs(gains["XX"]), axis=0),
        gain_amplitude_yy=np.nanmedian(np.abs(gains["YY"]), axis=0),
    )


def sun_lm_track(times):
    altaz = get_sun(times).transform_to(AltAz(obstime=times, location=NARRIBRI))
    l_value, m_value, _ = calc_lmn(altaz.alt.deg, altaz.az.deg, degrees=True)
    return np.column_stack(
        [np.asarray(l_value, dtype=float), np.asarray(m_value, dtype=float)]
    )


def galactic_plane_lm(time_value, npoints=1440):
    """Project the visible Galactic equator into topocentric ``(l, m)``."""
    npoints = int(npoints)
    if npoints < 32:
        raise ValueError("npoints must be at least 32")
    longitude = np.linspace(0.0, 360.0, npoints, endpoint=False) * u.deg
    plane = SkyCoord(l=longitude, b=np.zeros(npoints) * u.deg, frame="galactic")
    altaz = plane.transform_to(AltAz(obstime=time_value, location=NARRIBRI))
    l_value, m_value, _ = calc_lmn(altaz.alt.deg, altaz.az.deg, degrees=True)
    lm = np.column_stack(
        [np.asarray(l_value, dtype=float), np.asarray(m_value, dtype=float)]
    )
    lm[np.asarray(altaz.alt.deg) <= 0] = np.nan
    return lm


def steering_vectors(lm_values, antenna_locations, frequency):
    lm_values = np.asarray(lm_values)
    wavelength = c / float(frequency)
    phase = (
        2j
        * np.pi
        * (
            lm_values[:, 0, None] * antenna_locations[None, :, 0]
            + lm_values[:, 1, None] * antenna_locations[None, :, 1]
        )
        / wavelength
    )
    return np.exp(phase)


def dirty_image(covariance, frequency, antenna_locations, npix=256):
    return np.real(
        complex_dirty_image(covariance, frequency, antenna_locations, npix=npix)
    )


def complex_dirty_image(covariance, frequency, antenna_locations, npix=256):
    """Return the complex quadratic-form image before its physical real part."""
    covariance = np.array(covariance, copy=True)
    np.fill_diagonal(covariance, 0)
    return vistools.DFT_image(covariance, npix, antenna_locations, frequency)


def satellite_lm(time_value, satellite, time_offset_s=0.0):
    time_value = time_value + float(time_offset_s) * u.s
    utc = time_value.utc
    error, position, _ = satellite.sgp4(float(utc.jd1), float(utc.jd2))
    if error:
        raise RuntimeError("SGP4 failed with error code {}".format(error))
    teme = TEME(
        CartesianRepresentation(np.asarray(position) * u.km), obstime=time_value
    )
    itrs = teme.transform_to(ITRS(obstime=time_value))
    topocentric = ITRS(
        itrs.cartesian - NARRIBRI.get_itrs(obstime=time_value).cartesian,
        obstime=time_value,
        location=NARRIBRI,
    )
    altaz = topocentric.transform_to(AltAz(obstime=time_value, location=NARRIBRI))
    l_value, m_value, _ = calc_lmn(altaz.alt.deg, altaz.az.deg, degrees=True)
    return float(l_value), float(m_value), float(altaz.alt.deg)


