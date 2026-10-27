"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

import json
import numpy as np
import pandas as pd
from common import OUT, load_namespace

ns = load_namespace()
obs = ns["observation"]
sel = slice(int(ns["starts"][2]), int(ns["stops"][2]))
ts = obs.time[sel]
target, valid = ns["load_window"](sel, ns["config"].target_channel)
clean, cv = ns["load_window"](sel, ns["clean_channel"])
cm, _ = ns["shrink_directional_eigenmodes_to_noise"](
    clean[cv].mean(axis=0),
    ns["sun_steering"](ts[cv], ns["clean_frequency"]),
    obs.snapshots_per_integration * int(cv.sum()),
)
ref = ns["image_of"](cm, ns["clean_frequency"])
model = ns["physical_haslam_model"](ts)
dirs = ns["sun_steering"](ts, ns["target_frequency"])
mid = len(ts) // 2
sums = []
counts = []
checks = []
oldimgs = np.load(OUT / "images.npz")
halves = pd.read_csv(OUT / "correlations.csv")
for half, sl, rank in [(1, slice(0, mid), 3), (2, slice(mid, len(ts)), 1)]:
    result = ns["_null_and_restore_uncached"](
        target[sl],
        dirs[sl],
        valid[sl],
        rank,
        model=model[sl],
        protect_sun=True,
        shrink_sun=True,
        reduce="running",
    )
    n = int(result["valid"].sum())
    sums.append(n * result["covariance"])
    counts.append(n)
    im = ns["image_of"](result["covariance"])
    corr = ns["image_corr"](im, oldimgs[f"half{half}_clean"])
    expected = halves.query("half==@half and rank==@rank").protected_clean.iloc[0]
    np.testing.assert_allclose(corr, expected, atol=2e-5, rtol=0)
    checks.append(
        dict(half=half, rank=rank, integrations=n, correlation_with_half_reference=corr)
    )
combined = ns["image_of"](sum(sums) / sum(counts))
adaptive = ns["image_corr"](combined, ref)
record = dict(
    first_half_rank=3, second_half_rank=1, combined_correlation=adaptive, checks=checks
)
(OUT / "window3_split_rank.json").write_text(json.dumps(record, indent=2))

print(json.dumps(record), flush=True)
