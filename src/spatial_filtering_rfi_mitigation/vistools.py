import numpy as np
from spatial_filtering_rfi_mitigation.constants import c
from astropy.coordinates import SkyCoord


def beamformer_vectors(Naxis, antLoc, freq):
    """
    Compute the per-pixel beamformer steering vectors used by `DFT_image`.

    These depend only on the map size, antenna locations and frequency, so
    for a movie/sequence of images they should be computed once and reused
    across frames rather than being recomputed for every covariance matrix
    (see `DFT_image_batch`).

    Parameters
    ----------
    Naxis : int
       required size of the map, assumes a square map
    antLoc : numpy array, float
        antenna array locations
    freq : float
        system frequency in Hz

    Returns
    -------
    beamVech : numpy array, complex
        Steering vectors with shape (Nant, Naxis, Naxis).
    """

    lVec = np.linspace(-1.0, 1.0, Naxis)
    mVec = np.linspace(-1.0, 1.0, Naxis)
    lam = c / freq

    L, M = np.meshgrid(lVec, mVec, indexing="ij")
    disk = (L**2 + M**2) < 1.0
    lm = np.stack([L, M])

    beamVech = np.exp(
        -1j * 2 * np.pi * np.einsum("ij,jkl->ikl", antLoc[:, 0:2], lm) / lam
    )
    beamVech[:, ~disk] = np.nan  # steering vectors are undefined outside the unit disk
    return beamVech


def DFT_image(covMatrix, Naxis, antLoc, freq):
    """
    Function to generate post-correlation beamformed images. i.e. DFT images.

    Parameters
    ----------
    covMatrix : numpy array, complex
        The Hermitian correlation matrix
    Naxis : int
       required size of the map, assumes a square map
    antLoc : numpy array, float
        antenna array locations
    freq : float
        system frequency in Hz

    Returns
    -------
    skyImg : numpy array, float
        The generated map
    """

    beamVech = beamformer_vectors(Naxis, antLoc, freq)
    covMatrix[np.isnan(covMatrix)] = 0
    skyImg = np.einsum(
        "ijk,il,ljk->jk", beamVech, covMatrix, np.conj(beamVech), optimize="optimal"
    )
    return skyImg


# ---------------------------------------------------------------------------
# Sky-overlay utilities: A-team catalog, Solar System bodies, Galactic plane
# ---------------------------------------------------------------------------

# Fixed sky positions for bright "A-team" sources. Resolved once at import
# time via astropy/Sesame -- no network call required at plot time.
# Coordinates from the standard A-team reference catalog (e.g. Willson 1996).
BRIGHT_SOURCE_CATALOG = {
    "Cen A": SkyCoord("13h25m28s", "-43d01m09s"),
    "Tau A": SkyCoord("05h34m32s", "+22d00m52s"),
    "Her A": SkyCoord("16h51m08s", "+04d59m33s"),
    "Hyd A": SkyCoord("09h18m06s", "-12d05m44s"),
    "Pic A": SkyCoord("05h19m50s", "-45d46m44s"),
    "Vir A": SkyCoord("12h30m49s", "+12d23m28s"),
    "Orn A": SkyCoord("05h35m17s", "-05d23m23s"),
    "Cas A": SkyCoord("23h23m24s", "+58d48m54s"),
    "Cyg A": SkyCoord("19h59m28s", "+40d44m02s"),
    "For A": SkyCoord("03h22m42s", "-37d12m30s"),
    "Sgr A*": SkyCoord("17h45m40s", "-29d00m28s"),
}

# Order-of-magnitude flux densities (Jy) near 150 MHz, used only to rank
# sources for the `max_sources` cut in `overlay_sky_sources` -- not a
# calibrated flux scale. Solar System bodies are highly variable (the Sun
# especially, by orders of magnitude with activity); values here are typical
# quiescent levels, adequate for ranking but not for photometry. Extend with
# a proper low-frequency catalog (e.g. Perley & Butler 2017, GLEAM) if exact
# values start to matter.
_APPROX_FLUX_150MHZ_JY = {
    "Cas A": 30000, "Cyg A": 10500, "Sgr A*": 2500, "Vir A": 1700,
    "Tau A": 1500, "Cen A": 1400, "Her A": 650, "Hyd A": 450,
    "Orn A": 300, "For A": 260, "Pic A": 380,
    "Sun": 50000, "Jupiter": 50, "Moon": 50,
}

_GAL_PLANE_LON = np.linspace(0, 360, 721)
_GAL_PLANE_COORD = SkyCoord(
    l=_GAL_PLANE_LON, b=np.zeros(721), unit="deg", frame="galactic"
)


