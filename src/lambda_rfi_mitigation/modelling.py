import numpy as np

def calc_lmn(altVec,azVec,degrees=True):
    """
    Function for calculating the direction cosines (l,m,n) for a given altitude
    and azimuth vector. The inputs are assumed to be in degrees.

    Parameters
    ----------
    altVec : numpy np.ndarray, float
        Vector of altitude values.
    azVec : numpy np.ndarray, float
        Vector of azimuth values.
    degrees : bool, default=True
        If True, inputs are assumed to be in degrees.

    Returns
    -------
    lVec : numpy np.ndarray, float
        Vector of direction cosine l values.
    mVec : numpy np.ndarray, float
        Vector of direction cosine m values.
    nVec : numpy np.ndarray, float
        Vector of direction cosine n values.
    """
    if degrees:
        altVec = np.radians(altVec)
        azVec = np.radians(azVec)

    lVec = np.cos(altVec)*np.sin(azVec)
    mVec = np.cos(altVec)*np.cos(azVec)
    nVec = np.sqrt(1 - lVec**2 - mVec**2)

    return lVec,mVec,nVec


