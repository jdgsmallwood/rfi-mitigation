"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

import json
import numpy as np
import pandas as pd
from common import OUT, load_namespace

ns = load_namespace()
obs = ns["observation"]
sel = slice(int(ns["starts"][3]), int(ns["stops"][3]))
ts = obs.time[sel]
sat = ns["Satrec"].twoline2rv(*ns["config"].tle)


def altitude(i):
    return ns["satellite_lm"](ts[i], sat, ns["config"].ephemeris_time_offset_s)[2]


assert altitude(0) > 0 and altitude(len(ts) - 1) < 0
lo, hi = 0, len(ts) - 1
while hi - lo > 1:
    mid = (lo + hi) // 2
    if altitude(mid) > 0:
        lo = mid
    else:
        hi = mid
split = hi
assert altitude(split - 1) > 0 and altitude(split) <= 0
cube, valid = ns["load_window"](sel, ns["config"].target_channel)
clean, cv = ns["load_window"](sel, ns["clean_channel"])


def ref(c, v, t):
    m, _ = ns["shrink_directional_eigenmodes_to_noise"](
        c[v].mean(axis=0),
        ns["sun_steering"](t[v], ns["clean_frequency"]),
        obs.snapshots_per_integration * int(v.sum()),
    )
    return ns["image_of"](m, ns["clean_frequency"])


reference = ref(clean, cv, ts)
model = ns["physical_haslam_model"](ts)
dirs = ns["sun_steering"](ts, ns["target_frequency"])
parts = []
results = {}
rows = []
for part, sl in [("above", slice(0, split)), ("below", slice(split, len(ts)))]:
    rr = ref(clean[sl], cv[sl], ts[sl])
    for rank in [0, 1]:
        r = ns["_null_and_restore_uncached"](
            cube[sl],
            dirs[sl],
            valid[sl],
            rank,
            model=model[sl],
            protect_sun=True,
            shrink_sun=True,
            reduce="running",
        )
        n = int(r["valid"].sum())
        results[(part, rank)] = (r["covariance"], n)
        rows.append(
            dict(
                part=part,
                rank=rank,
                integrations=n,
                correlation=ns["image_corr"](ns["image_of"](r["covariance"]), rr),
            )
        )


def combined(r1, r2):
    c1, n1 = results[("above", r1)]
    c2, n2 = results[("below", r2)]
    return ns["image_of"]((c1 * n1 + c2 * n2) / (n1 + n2))


old = pd.read_csv(OUT / "rank_sweeps.csv").query("window==4").set_index("fixed_rank")
for rank in [0, 1]:
    np.testing.assert_allclose(
        ns["image_corr"](combined(rank, rank), reference),
        old.loc[rank, "protected_clean"],
        atol=2e-5,
        rtol=0,
    )
im = combined(1, 0)
record = dict(
    split_index=split,
    seconds_from_start=float((ts[split] - ts[0]).sec),
    setting_utc=ts[split].utc.isot,
    altitude_before=altitude(split - 1),
    altitude_after=altitude(split),
    combined_correlation=ns["image_corr"](im, reference),
    fixed_rank0=float(old.loc[0, "protected_clean"]),
    fixed_rank1=float(old.loc[1, "protected_clean"]),
    parts=rows,
)
(OUT / "window4_setting_split.json").write_text(json.dumps(record, indent=2))

print(json.dumps(record, indent=2), flush=True)
print("PASS: recombined fixed-rank controls match original Window 4", flush=True)
