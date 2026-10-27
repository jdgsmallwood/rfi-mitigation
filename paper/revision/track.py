"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

from astropy.time import TimeDelta
import numpy as np
import pandas as pd
from common import OUT, load_namespace

ns = load_namespace()
sat = ns["Satrec"].twoline2rv(*ns["config"].tle)
obs = ns["observation"]
stats = []
series = []
elevation = []


def brightness(cube, lm, freq):
    a = ns["steering_vectors"](lm, obs.antenna_locations, freq)
    # Match paper dirty images: omit autocorrelations, retain unnormalised sum.
    return (
        np.einsum("ti,tij,tj->t", a.conj(), cube, a, optimize=True).real
        - np.trace(cube, axis1=1, axis2=2, dtype=np.complex128).real
    )


for w in range(4):
    sel = slice(int(ns["starts"][w]), int(ns["stops"][w]))
    ts = obs.time[sel]
    t = np.arange(len(ts))
    step = 56
    nodes = np.unique(np.r_[t[::step], t[-1]])
    track = np.array(
        [
            ns["satellite_lm"](ts[int(j)], sat, ns["config"].ephemeris_time_offset_s)
            for j in nodes
        ]
    )
    interp = np.column_stack([np.interp(t, nodes, track[:, j]) for j in range(3)])
    checks = nodes[:-1] + np.diff(nodes) // 2
    exact = np.array(
        [
            ns["satellite_lm"](ts[int(j)], sat, ns["config"].ephemeris_time_offset_s)
            for j in checks
        ]
    )
    err = float(np.max(np.abs(interp[checks, :2] - exact[:, :2])))
    assert err < 1e-5, err
    for second in range(121):
        when = ts[0] + TimeDelta(second, format="sec")
        elevation.append(
            dict(
                window=w + 1,
                seconds_from_window_start=second,
                elevation_deg=ns["satellite_lm"](
                    when, sat, ns["config"].ephemeris_time_offset_s
                )[2],
            )
        )
    target, tv = ns["load_window"](sel, ns["config"].target_channel)
    clean, cv = ns["load_window"](sel, ns["clean_channel"])
    # Verify sign and scaling against one actual DFT image pixel.
    axis = np.linspace(-1, 1, ns["snapshot_pixels"])
    lm = np.array([[axis[130], axis[125]]])
    np.testing.assert_allclose(
        brightness(target[:1], lm, ns["target_frequency"])[0],
        ns["image_of"](target[0])[130, 125],
        rtol=1e-6,
        atol=1e-5,
    )
    b = brightness(target, interp[:, :2], ns["target_frequency"])
    c = brightness(clean, interp[:, :2], ns["clean_frequency"])
    valid = tv & cv & (interp[:, 2] > 0)
    row = dict(
        window=w + 1,
        mean_target=float(b[valid].mean()),
        median_target=float(np.median(b[valid])),
        mean_clean=float(c[valid].mean()),
        above_horizon_fraction=float(valid.sum() / (tv & cv).sum()),
        start_altitude_deg=interp[0, 2],
        end_altitude_deg=interp[-1, 2],
        track_interpolation_max_error=err,
    )
    stats.append(row)
    # One-second means for a readable time series; retain only above-horizon samples.
    for start in range(0, len(t), step):
        sl = slice(start, min(start + step, len(t)))
        v = valid[sl]
        if v.any():
            series.append(
                dict(
                    window=w + 1,
                    seconds_from_window_start=float(
                        t[sl][v].mean() * ns["cadence_seconds"]
                    ),
                    target=float(b[sl][v].mean()),
                    clean=float(c[sl][v].mean()),
                    altitude_deg=float(interp[sl, 2][v].mean()),
                )
            )
    print(row, flush=True)
s = pd.DataFrame(stats)
s["relative_to_window1_db"] = 10 * np.log10(s.mean_target / s.mean_target.iloc[0])
s.to_csv(OUT / "satellite_brightness.csv", index=False)
d = pd.DataFrame(series)
d.to_csv(OUT / "satellite_brightness_timeseries.csv", index=False)
pd.DataFrame(elevation).to_csv(OUT / "elevation.csv", index=False)
