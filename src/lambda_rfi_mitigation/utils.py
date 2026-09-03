import numpy as np
import h5py as h5
import json
import importlib.resources as resources
from lambda_rfi_mitigation.constants import NARRIBRI, CHANWIDTH
from astropy.time import Time
import astropy.units as u
from loguru import logger

import math


antennaConfigPath = resources.files("lambda_rfi_mitigation.data")
# Antenna mapping dictionary.
mappingFile = "LAMBDA36-antenna-mappings.json"
with antennaConfigPath.joinpath(mappingFile).open("r") as f:
    antennaDict = json.load(f)
    antennaIDs = np.array(list(antennaDict.keys())).astype(int)


def read_hdf5_gains(filePath, stokes=["XX"], verbose=False):
    """
    Reads a gain table from HDF5 file and returns it.

    Parameters
    ----------
    filePath : str
        Path to the HDF5 file.
    stokes : list of str, optional
        List of Stokes parameters to read (e.g., ["XX", "YY", "XY", "YX"]),
        by default ["XX"].
    verbose : bool, optional
        If True, print verbose output, by default False.

    Returns
    -------
    data : dict of numpy arrays per polarization.
        The dataset read from the HDF5 file.

    """

    # Support for reading in multiple files and concatenating them together.
    if isinstance(filePath, str):
        filePaths = [filePath]
    elif isinstance(filePath, list):
        filePaths = filePath

    logger.info(filePaths)
    gainsDict = {}
    for i, filePath in enumerate(filePaths):
        with h5.File(filePath, "r") as hdf_file:
            gains = hdf_file["calibration"]
            for pol in stokes:
                gainsDict[pol] = (
                    gains[pol]["gains_real"][:] + 1j * gains[pol]["gains_imag"][:]
                )

            if verbose:
                logger.info(f"Dataset keys: {list(hdf_file.keys())}")
                logger.info(f"File attributes: {list(hdf_file.attrs.keys())}")

    return gainsDict


def read_hdf5_time_freq(
    filePath,
    verbose=False,
    returnLST=False,
    location=NARRIBRI,
    time_range=None,
    index_range=None,
):
    """read_hdf5_time_freq retrieves the time and frequency information for a
    given LAMBDA observation, assuming Narribri is the location.

    Parameters
    ----------
    filePath : _type_
        _description_
    verbose : bool, optional
        _description_, by default False
    returnLST : bool, optional
        _description_, by default False
    location : _type_, optional
        _description_, by default NARRIBRI
    time_range : tuple of float, optional
        ``(t_start, t_end)`` in seconds relative to the first sample.
    index_range : tuple of int, optional
        ``(start_idx, stop_idx)`` sample indices (python-slice convention,
        ``stop_idx`` exclusive). Takes precedence over ``time_range`` if
        both are given. Use the same ``index_range`` passed to
        ``read_hdf5_data_capture`` to keep ``timeVec`` aligned with the
        loaded visibilities.

    Returns
    -------
    _type_
        _description_
    """

    if isinstance(filePath, str):
        filePaths = [filePath]
    elif isinstance(filePath, list):
        filePaths = filePath

    # Pre-compute time step from first file metadata when time slicing is needed.
    dt = None
    if time_range is not None:
        with h5.File(filePaths[0], "r") as _hf:
            _Nchans = _hf["visibilities"].shape[1]
            _Npkts = _hf["vis_missing_nums"][0][1]
            _Nant = len(np.unique(_hf["baseline_ids"][:] // 256))
            _Ntime_pkts = _Npkts / _Nchans / (_Nant // 10)
            dt = _Ntime_pkts * 64 * 1.08 / 1000000

    timeVecList = []
    tLSTList = []
    for file in filePaths:
        with h5.File(file, "r") as hf:
            if verbose:
                logger.info(hf.keys())
                logger.info(hf.attrs.keys())

            Ntimes = hf["visibilities"].shape[0]
            Nchans = hf["visibilities"].shape[1]

            #
            chanMin = hf.attrs["min_channel"]
            chanMax = hf.attrs["max_channel"]
            chans = np.arange(chanMin, chanMax + 1, 1)
            freqs = chans * CHANWIDTH * 1e6

            # calculate the spacing between visibilty measurements in seconds
            Npkts_per_vis = hf["vis_missing_nums"][0][1]
            Nant = len(np.unique(hf["baseline_ids"][:] // 256))
            # There will be correlation block size * NChan * NFPGA packets, so divide
            # out the NFPGA and NChan terms.
            Ntime_pkts = Npkts_per_vis / Nchans / math.ceil(Nant / 10)
            # 64 time samples in a packet, multiply by sampling time of 1.08 us
            # to get total time.
            time_offset_between_visibilities = Ntime_pkts * 64 * 1.08 / 1000000

            if verbose:
                logger.info(f"Number of times: {Ntimes}")
                logger.info(f"Number of channels: {Nchans}")
                logger.info(f"Min channel: {chanMin}")
                logger.info(f"Max channel: {chanMax}")
                logger.info(f"Npkts_per_vis: {Npkts_per_vis}")
                logger.info(f"Num Time Packets: {Ntime_pkts}")
                logger.info(
                    f"Time offset between visibilities: {time_offset_between_visibilities} s / {time_offset_between_visibilities * 1000} ms"
                )

            t0 = hf.attrs["mjd_start"]
            timeVec = (
                Time(t0, format="mjd", scale="utc", location=location)
                + time_offset_between_visibilities * np.arange(Ntimes) * u.s
            )
            tLST = timeVec.sidereal_time("apparent", longitude=location.lon)

            timeVecList.append(timeVec)
            tLSTList.append(tLST)

    timeVec = np.concatenate(timeVecList)
    tLST = np.concatenate(tLSTList)

    if index_range is not None:
        Ntimes_total = len(timeVec)
        start_idx, stop_idx = index_range
        start_idx = 0 if start_idx is None else max(0, start_idx)
        stop_idx = Ntimes_total if stop_idx is None else min(Ntimes_total, stop_idx)
        timeVec = timeVec[start_idx:stop_idx]
        tLST = tLST[start_idx:stop_idx]
        if verbose:
            logger.info(
                f"Applying index range: [{start_idx}, {stop_idx}) "
                f"({stop_idx - start_idx} out of {Ntimes_total} samples selected)."
            )
    elif time_range is not None and dt is not None:
        # Uniform grid, matching how read_hdf5_data_capture turns a time_range
        # into sample indices -- keeps the two functions in step.
        Ntimes_total = len(timeVec)
        t_seconds = np.arange(Ntimes_total) * dt
        t_start_s, t_end_s = time_range
        t_max = t_seconds[-1]
        if t_end_s is None:
            t_end_s = t_max
        elif t_end_s > t_max:
            logger.warning(
                f"Requested end time {t_end_s:.2f}s exceeds available data "
                f"({t_max:.2f}s). Using maximum available time."
            )
            t_end_s = t_max
        
        if t_start_s is None:
            t_start_s = 0.0
        elif t_start_s < 0:
            logger.warning(
                f"Requested start time {t_start_s:.2f}s is negative. Using 0s instead."
            )
            t_start_s = 0.0
    
        time_mask = (t_seconds >= t_start_s) & (t_seconds <= t_end_s)
        if verbose:
            logger.info(
                f"Applying time range mask: {t_start_s:.2f}s to {t_end_s:.2f}s "
                f"({time_mask.sum()} out of {Ntimes_total} samples selected)."
            )
        
        timeVec = timeVec[time_mask]
        tLST = tLST[time_mask]

    if returnLST:
        return timeVec, freqs, tLST
    else:
        return timeVec, freqs

    if returnLST:
        return timeVec, freqs, tLST
    else:
        return timeVec, freqs


def split_baseline(baselineIDs):
    """
    Function for determining the antenna IDs from the baseline ID. Baseline
    ID is determined by ant1*256 + ant2.

    Parameters
    ----------
    baselineIDs : ndarray
        Numpy array containing the baseline IDs (ant1*256+ant2).

    Returns
    -------
    ant1 : ndarray
        Numpy array containing the the antenna1 ID.
    ant2 : ndarray
        Numpy array containing the the antenna2 ID.
    """
    baselineIDs = np.asarray(baselineIDs)
    if baselineIDs.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if np.max(baselineIDs) >= 65536:
        ant1 = ((baselineIDs - 65536) // 2048).astype(int)
        ant2 = ((baselineIDs - 65536) % 2048).astype(int)
    else:
        ant1 = (baselineIDs // 256).astype(int)
        ant2 = (baselineIDs % 256).astype(int)

    return ant1, ant2


def make_telescope_model(antIDs, telescope="LAMBDA36", verbose=False):
    """make_telescope_model _summary_

    Parameters
    ----------
    antIDs : array-like
        Array of antenna IDs to include in the model.
    telescope : str, optional
        Name of the telescope configuration to use (default is 'LAMBDA36').
    verbose : bool, optional
        If True, print additional information during model creation (default is False).

    Returns
    -------
    InterferometerModel
        The created interferometer model.
    """
    from lambda_rfi_mitigation.constants import NARRIBRI
    from lambda_rfi_mitigation.interferometers import make_radio_array

    # mappingFile = "LAMBDA36-antenna-mappings.json"
    mappingFile = "LAMBDA36-antenna-mappings-reordered.json"
    with antennaConfigPath.joinpath(mappingFile).open("r") as f:
        antennaDict = json.load(f)

    eastVec = np.zeros(antIDs.size)
    northVec = np.zeros(antIDs.size)
    for i, antID in enumerate(antIDs):
        eastVec[i] = antennaDict[f"{antID}"]["east"]
        northVec[i] = antennaDict[f"{antID}"]["north"]

    #
    height = NARRIBRI.height.value
    eastNorthHeight = np.vstack((eastVec, northVec, np.ones(eastVec.size) * height))

    if verbose:
        logger.info(eastNorthHeight.T.shape)

    InterferometerModel = make_radio_array(
        eastNorthHeight=eastNorthHeight.T,
        lat=NARRIBRI.lat.value,
        lon=NARRIBRI.lon.value,
        telescope=telescope,
    )

    return InterferometerModel


N_POL = 2
N_ANTENNAS_PER_ALVEO = 10
N_TIME_SAMPLES_PER_PACKET = 64

# Binary layout of a single LAMBDA correlator UDP payload (2622 bytes):
#   seq_no    <u8        (8 bytes)
#   fpga_id   <u4        (4 bytes, unused)
#   freq_chan <u2        (2 bytes)
#   padding              (8 bytes, unused)
#   scales    <u2[10][2] (40 bytes)  -- per (antenna, pol) gain scale factor
#   samples   i1[64][10][2][I,Q] (2560 bytes) -- per (time, antenna, pol) I/Q
_LAMBDA_PACKET_DTYPE = np.dtype(
    [
        ("seq_no", "<u8"),
        ("fpga_id", "<u4"),
        ("freq_chan", "<u2"),
        ("_padding", "V8"),
        ("scales", "<u2", (N_ANTENNAS_PER_ALVEO, N_POL)),
        (
            "samples",
            "i1",
            (N_TIME_SAMPLES_PER_PACKET, N_ANTENNAS_PER_ALVEO, N_POL, 2),
        ),
    ]
)


