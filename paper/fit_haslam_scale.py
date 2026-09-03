"""Re-derive the trained Haslam amplitude used by paper-figures.ipynb.

`paper-figures.ipynb` takes `trained_haslam_scale = 42.84043253358354` as a
fixed constant; nothing in the notebook fits it, so no free parameter is tuned
against the results it presents. This script is where that number comes from.

The amplitude is fitted on the satellite-free start of the capture, on the
clean reference channels rather than the target channel -- channel 4 carries
the Orbcomm downlink and is never RFI-free in time. The fitted coefficients are
dimensionless amplitudes on top of physically normalised templates, so they
transfer to the target frequency and are held fixed across the satellite
windows.

Original run (2026-07-22), reproduced by this script:

    Trained on satellite-free abs [2000, 26000) (24 blocks, clean channels (0, 1))
    Trained Haslam scale: 42.84 +/- 4.8   (prior unvalidated model_scale_upper = 36.92)
    Cross-visibility-fitted actual Haslam scale: 42.84043253358354
    Median actual-Haslam cross-vis fit residual: 0.6974359579813743
    Trained Sun power:    2.084e+04 +/- 5.9e+03

Re-running this script today reproduces 42.840431127 against the published
42.840432534: a difference of 1.4e-06, or 3e-8 relative. The two differ only in
the convergence tolerance of the `soft_l1` least-squares refinement under a
newer scipy, and the gap is eight orders of magnitude below the +/- 4.8 MAD of
the fit itself. The notebook keeps the published value so its figures stay
exactly the ones in the paper.

Usage (needs the visibility captures, see README.md):

    python fit_haslam_scale.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from lambda_rfi_mitigation.low_rank_sky_experiment import (
    _fit_sky_and_sun,
    beam_weighted_haslam_covariance,
)
from lambda_rfi_mitigation.protected_pipeline import (
    ProtectedRunConfig,
    prepare_observation,
    steering_vectors,
    sun_lm_track,
)

TARGET_NAME = "visibilities_20260611-1511_172_179_ALVEO0_ALVEO1_ALVEO2_ALVEO3.hdf5"
GAIN_NAME = "visibilities_20260611-1349_172_179_ALVEO0_ALVEO1_ALVEO2_ALVEO3.hdf5"
DATA_DOI = "https://doi.org/10.5281/zenodo.22255363"

# Training stretch: satellite-free start of the capture, chosen from the RFI
# landscape. Absolute HDF5 integration indices.
TRAIN_START_INDEX = 2000
TRAIN_LENGTH = 24000
TRAIN_BLOCK_HALF = 150
TRAIN_BLOCKS = 24
SUN_POWER_CHANNELS = (0, 1)      # per-integration Sun-power references
FLAG_ANTENNAS = (8, 9, 12, 14, 22, 33, 36)

# The value published in paper-figures.ipynb.
PUBLISHED_SCALE = 42.84043253358354


def find_data_root():
    """Locate the directory holding the two captures. Mirrors the notebook."""
    search_roots = [Path(os.environ[name]) for name in ("LAMBDA_DATA_PATH",)
                    if os.environ.get(name)]
    search_roots += [Path(".."), Path(".")]
    for root in search_roots:
        if (root / TARGET_NAME).exists():
            return root
    raise FileNotFoundError(
        "Could not locate the visibility captures\n"
        f"  {TARGET_NAME}\n  {GAIN_NAME}\n"
        "Set $LAMBDA_DATA_PATH to the directory containing them, or place them "
        f"at the repository root. They are archived at {DATA_DOI}\n"
        "Searched: " + ", ".join(str(root.resolve()) for root in search_roots)
    )


def robust_median_mad(values):
    """Median and MAD-derived standard deviation, ignoring non-finite entries."""
    values = np.asarray(values)[np.isfinite(values)]
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    return median, mad


def fit_trained_haslam_scale(data_root, verbose=True):
    """Fit the Haslam and Sun amplitudes on the satellite-free training stretch.

    Returns (haslam_median, haslam_mad, sun_median, sun_mad, residual_median).
    """
    config = ProtectedRunConfig(
        visibility_file=str(data_root / TARGET_NAME),
        gain_file=str(data_root / GAIN_NAME),
        start_index=TRAIN_START_INDEX,
        stop_index=TRAIN_START_INDEX + TRAIN_LENGTH + 10,
        target_channel=4,
        clean_channels=(0,),
        flag_antennas=FLAG_ANTENNAS,
        snapshot_pixels=256,
    )
    observation = prepare_observation(config, verbose=verbose)
    locations = observation.antenna_locations
    centres = np.linspace(
        TRAIN_BLOCK_HALF + 10, TRAIN_LENGTH - TRAIN_BLOCK_HALF - 10, TRAIN_BLOCKS
    ).astype(int)

    haslam_fit, sun_fit, fit_residual = [], [], []
    for centre in centres:
        block = slice(int(centre - TRAIN_BLOCK_HALF), int(centre + TRAIN_BLOCK_HALF))
        block_time = observation.time[int(centre)]
        block_sun_lm = sun_lm_track(np.atleast_1d(block_time))[0]
        per_channel = []
        for channel in SUN_POWER_CHANNELS:
            frequency = float(observation.frequencies[channel])
            cube = observation.load_channel(block, channel)
            diagonal = np.real(np.diagonal(cube, axis1=-2, axis2=-1))
            valid = (
                np.all(np.isfinite(cube), axis=(1, 2)) & np.all(diagonal > 0, axis=1)
            )
            if not np.any(valid):
                continue
            clean_covariance = np.mean(cube[valid], axis=0)
            direction = steering_vectors(
                block_sun_lm[None, :], locations, frequency
            )[0]
            # Only the unrotated template is ever fitted: rotated skies are
            # falsification controls and must not be tuned to the data.
            haslam_template, _ = beam_weighted_haslam_covariance(
                block_time, locations, frequency,
                apparent_rotation_quarter_turns=0,
            )
            fitted, residual, _ = _fit_sky_and_sun(
                clean_covariance, haslam_template, direction
            )
            per_channel.append((fitted[0], fitted[1], residual))
        if per_channel:
            channel_mean = np.mean(np.asarray(per_channel), axis=0)
            haslam_fit.append(channel_mean[0])
            sun_fit.append(channel_mean[1])
            fit_residual.append(channel_mean[2])

    haslam_median, haslam_mad = robust_median_mad(haslam_fit)
    sun_median, sun_mad = robust_median_mad(sun_fit)
    residual_median = float(np.median(fit_residual))

    if verbose:
        print(
            f"Trained on satellite-free abs "
            f"[{TRAIN_START_INDEX}, {TRAIN_START_INDEX + TRAIN_LENGTH}) "
            f"({len(haslam_fit)} blocks, clean channels {SUN_POWER_CHANNELS})"
        )
        print(f"Trained Haslam scale: {haslam_median:.4g} +/- {haslam_mad:.2g}")
        print(f"Cross-visibility-fitted actual Haslam scale: {haslam_median!r}")
        print(f"Median actual-Haslam cross-vis fit residual: {residual_median!r}")
        print(f"Trained Sun power:    {sun_median:.4g} +/- {sun_mad:.2g}")
        print(
            f"Published value in paper-figures.ipynb: {PUBLISHED_SCALE!r} "
            f"(difference {haslam_median - PUBLISHED_SCALE:+.3e})"
        )
    return haslam_median, haslam_mad, sun_median, sun_mad, residual_median


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Directory holding the two captures (default: $LAMBDA_DATA_PATH, "
             "then the repository root, then the working directory).",
    )
    arguments = parser.parse_args()
    data_root = arguments.data_root or find_data_root()
    fit_trained_haslam_scale(data_root)


if __name__ == "__main__":
    main()
