"""Build an animated dirty-image movie with Sun and satellite-ephemeris overlays.

Reads the same visibility/gain captures as paper/paper-figures.ipynb and
writes an MP4 of the raw (unprotected) dirty image over time, per batch of
integrations, with the Sun position and SGP4 satellite track overlaid.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from sgp4.api import Satrec

from lambda_rfi_mitigation.protected_pipeline import (
    ProtectedRunConfig, dirty_image, prepare_observation, satellite_lm, sun_lm_track,
)

TARGET_NAME = "visibilities_20260611-1511_172_179_ALVEO0_ALVEO1_ALVEO2_ALVEO3.hdf5"
GAIN_NAME = "visibilities_20260611-1349_172_179_ALVEO0_ALVEO1_ALVEO2_ALVEO3.hdf5"
FLAG_ANTENNAS = (8, 9, 12, 14, 22, 33, 36)


def find_data_root(target_name):
    for candidate in (Path("."), Path("..")):
        if (candidate / target_name).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {target_name}")


def load_valid(observation, selection, channel):
    """One channel's covariance cube plus the per-integration validity mask."""
    cube = observation.load_channel(selection, channel)
    diagonal = np.real(np.diagonal(cube, axis1=-2, axis2=-1))
    valid = np.all(np.isfinite(cube), axis=(1, 2)) & np.all(diagonal > 0.0, axis=1)
    return cube, valid


def build_movie(
    output_path, channel=4, batch_size=200, fps=6, npix=256, start=None, stop=None,
):
    data_root = find_data_root(TARGET_NAME)
    config = ProtectedRunConfig(
        visibility_file=str(data_root / TARGET_NAME),
        gain_file=str(data_root / GAIN_NAME),
        start_index=30000, target_channel=channel, flag_antennas=FLAG_ANTENNAS,
        snapshot_pixels=npix,
    )
    observation = prepare_observation(config)
    satellite = Satrec.twoline2rv(*config.tle)
    frequency = observation.target_frequency

    start = 0 if start is None else start
    stop = observation.ntime if stop is None else stop

    figure, axis = plt.subplots(figsize=(5, 5))
    image_artist = axis.imshow(
        np.zeros((npix, npix)), extent=[-1, 1, -1, 1], origin="lower", cmap="RdBu_r",
    )
    satellite_point, = axis.plot([], [], "kx", ms=8, mew=1.5, label="Satellite")
    sun_point, = axis.plot(
        [], [], "o", mfc="none", mec="tab:green", ms=8, mew=1.2, label="Sun",
    )
    axis.set(
        xlim=(1, -1), ylim=(-1, 1), aspect="equal", xlabel="l (east left)", ylabel="m",
    )
    axis.legend(loc="upper right", fontsize=6)
    title = axis.set_title("")

    writer = FFMpegWriter(fps=fps)
    with writer.saving(figure, str(output_path), dpi=120):
        for frame_start in range(start, stop, batch_size):
            frame_stop = min(frame_start + batch_size, stop)
            cube, valid = load_valid(observation, slice(frame_start, frame_stop), channel)
            if not np.any(valid):
                continue
            covariance = np.mean(cube[valid], axis=0)
            image = dirty_image(covariance, frequency, observation.antenna_locations, npix)

            mid_time = observation.time[(frame_start + frame_stop - 1) // 2]
            l_sat, m_sat, _ = satellite_lm(mid_time, satellite, config.ephemeris_time_offset_s)
            l_sun, m_sun = sun_lm_track(mid_time.reshape(1))[0]

            image_artist.set_data(image.T)
            limit = float(np.nanpercentile(np.abs(image), 99.5)) or 1.0
            image_artist.set_clim(-limit, limit)
            satellite_point.set_data([l_sat], [m_sat])
            sun_point.set_data([l_sun], [m_sun])
            title.set_text(mid_time.iso)
            writer.grab_frame()
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="eigenfilter-movie.mp4")
    parser.add_argument("--channel", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--npix", type=int, default=256)
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--stop", type=int, default=None)
    args = parser.parse_args()
    build_movie(
        args.output, args.channel, args.batch_size, args.fps, args.npix,
        args.start, args.stop,
    )


if __name__ == "__main__":
    main()
