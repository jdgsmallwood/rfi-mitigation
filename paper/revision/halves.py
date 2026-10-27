"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

import json
import numpy as np
import pandas as pd
from common import OUT, load_namespace

ns = load_namespace()
obs = ns["observation"]
start = int(ns["starts"][2])
stop = int(ns["stops"][2])
mid = (start + stop) // 2
sel = slice(start, stop)
ts = obs.time[sel]
full, fullvalid = ns["load_window"](sel, ns["config"].target_channel)
clean, cleanvalid = ns["load_window"](sel, ns["clean_channel"])
fullmodel = ns["physical_haslam_model"](ts)
fulldirs = ns["sun_steering"](ts, ns["target_frequency"])


def reference(cube, valid, times):
    cm = cube[valid].mean(axis=0)
    cd = ns["sun_steering"](times[valid], ns["clean_frequency"])
    shrunk, _ = ns["shrink_directional_eigenmodes_to_noise"](
        cm, cd, obs.snapshots_per_integration * int(valid.sum())
    )
    return ns["image_of"](shrunk, ns["clean_frequency"])


fullref = reference(clean, cleanvalid, ts)
rows = []
sums = {}
weights = {}
images = {}
for half, sl in [(1, slice(0, mid - start)), (2, slice(mid - start, stop - start))]:
    target = full[sl]
    valid = fullvalid[sl]
    dirs = fulldirs[sl]
    model = fullmodel[sl]
    ref = reference(clean[sl], cleanvalid[sl], ts[sl])
    images[f"half{half}_clean"] = ref
    for rank in range(11):
        row = dict(half=half, rank=rank)
        for method, mod in [("protected", model), ("unprotected", None)]:
            result = ns["_null_and_restore_uncached"](
                target,
                dirs,
                valid,
                rank,
                model=mod,
                protect_sun=True,
                shrink_sun=True,
                reduce="running",
            )
            cov = result["covariance"]
            n = int(result["valid"].sum())
            key = (method, rank)
            sums[key] = sums.get(key, 0) + n * cov
            weights[key] = weights.get(key, 0) + n
            im = ns["image_of"](cov)
            row[method + "_clean"] = ns["image_corr"](im, ref)
        assert np.isfinite(list(row.values())).all()
        rows.append(row)
        print(json.dumps(row), flush=True)
old = pd.read_csv(OUT / "rank_sweeps.csv").query("window==3").set_index("fixed_rank")
for (method, rank), total in sums.items():
    corr = ns["image_corr"](ns["image_of"](total / weights[(method, rank)]), fullref)
    np.testing.assert_allclose(
        corr, old.loc[rank, method + "_clean"], atol=2e-5, rtol=0
    )
print("PASS: recombined halves reproduce all 22 full-window correlations", flush=True)
np.savez_compressed(OUT / "images.npz", **images)

pd.DataFrame(rows).to_csv(OUT / "correlations.csv", index=False)
