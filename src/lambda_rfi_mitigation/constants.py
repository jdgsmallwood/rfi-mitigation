from astropy.coordinates import EarthLocation

LAT = -30.3128
LON = 149.5644
CHANWIDTH = 0.78125#[MHz]
height = 211#[m]
NARRIBRI = EarthLocation.from_geodetic(LON,LAT,height=height)
c = 299792458 # m s^-1
kb = 1380.64 # K^-1 Jy