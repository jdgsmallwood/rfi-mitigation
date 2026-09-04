import numpy as np

from spatial_filtering_rfi_mitigation.vistools import DFT_image


def _DFT_image_reference(covMatrix, Naxis, antLoc, freq):
    """Original, unvectorized implementation kept here as a reference for
    `DFT_image`/`DFT_image_batch`."""
    c = 299792458  # m s^-1

    lVec = np.linspace(-1.0, 1.0, Naxis)
    mVec = np.linspace(-1.0, 1.0, Naxis)
    lm = np.full([2, Naxis, Naxis], np.nan, dtype=np.float64)
    lam = c / freq

    for i in range(Naxis):
        for j in range(Naxis):
            l = lVec[i]
            m = mVec[j]
            if (l**2 + m**2 < 1.0):
                lm[:, i, j] = [l, m]

    beamVech = np.exp(-1j * 2 * np.pi * np.einsum('ij,jkl->ikl', antLoc[:, 0:2], lm) / lam)

    skyImg = np.einsum("ijk,il,ljk->jk", beamVech, covMatrix,
                        np.conj(beamVech), optimize='optimal')
    return skyImg


def _make_inputs(seed=0, Nant=6, Naxis=16):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(Nant, Nant)) + 1j * rng.normal(size=(Nant, Nant))
    covMatrix = A @ A.conj().T  # Hermitian
    antLoc = rng.uniform(-50, 50, size=(Nant, 2))
    freq = 200e6
    return covMatrix, Naxis, antLoc, freq


def test_DFT_image_matches_reference():
    covMatrix, Naxis, antLoc, freq = _make_inputs()

    expected = _DFT_image_reference(covMatrix, Naxis, antLoc, freq)
    actual = DFT_image(covMatrix, Naxis, antLoc, freq)

    np.testing.assert_allclose(actual, expected)
