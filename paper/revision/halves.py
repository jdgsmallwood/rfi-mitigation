"""Select the best protected rank in each 60 s half of every window.

Ranks 0--10 are scored against the corresponding half-window clean image.
This is retrospective selection, not an automatic RFI rank estimator.
"""

import json
import numpy as np
import pandas as pd
from common import OUT, load_namespace

ns = load_namespace()
obs = ns["observation"]
fixed = pd.read_csv(OUT / "rank_sweeps.csv")
rows = []
adaptive = []


def reference(cube, valid, times):
    cm = cube[valid].mean(axis=0)
    cd = ns["sun_steering"](times[valid], ns["clean_frequency"])
    shrunk, _ = ns["shrink_directional_eigenmodes_to_noise"](
        cm, cd, obs.snapshots_per_integration * int(valid.sum())
    )
    return ns["image_of"](shrunk, ns["clean_frequency"])


for w in range(4):
    start, stop = int(ns["starts"][w]), int(ns["stops"][w])
    sel = slice(start, stop)
    ts = obs.time[sel]
    mid = len(ts) // 2
    assert mid == len(ts) - mid
    full, fullvalid = ns["load_window"](sel, ns["config"].target_channel)
    clean, cleanvalid = ns["load_window"](sel, ns["clean_channel"])
    fullmodel = ns["physical_haslam_model"](ts)
    fulldirs = ns["sun_steering"](ts, ns["target_frequency"])
    fullref = reference(clean, cleanvalid, ts)
    results = {}
    best = []
    for half, sl in [(1, slice(0, mid)), (2, slice(mid, len(ts)))]:
        ref = reference(clean[sl], cleanvalid[sl], ts[sl])
        scores = []
        for rank in range(11):
            result = ns["_null_and_restore_uncached"](
                full[sl], fulldirs[sl], fullvalid[sl], rank,
                model=fullmodel[sl], protect_sun=True, shrink_sun=True,
                reduce="running",
            )
            cov = result["covariance"]
            count = int(result["valid"].sum())
            assert count > 0
            corr = ns["image_corr"](ns["image_of"](cov), ref)
            assert np.isfinite(corr)
            results[half, rank] = (cov, count)
            scores.append(corr)
            row = dict(window=w + 1, half=half, rank=rank,
                       protected_clean=corr, integrations=count)
            rows.append(row)
            print(json.dumps(row), flush=True)
        # Ascending ranks make an exact tie choose the smaller rank.
        best.append(int(np.argmax(scores)))

    def combined(r1, r2):
        c1, n1 = results[1, r1]
        c2, n2 = results[2, r2]
        return ns["image_of"]((n1 * c1 + n2 * c2) / (n1 + n2))

    old = fixed.query("window == @w + 1").set_index("fixed_rank")
    for rank in range(11):
        np.testing.assert_allclose(
            ns["image_corr"](combined(rank, rank), fullref),
            old.loc[rank, "protected_clean"], atol=2e-5, rtol=0,
        )
    record = dict(
        window=w + 1, first_half_rank=best[0], second_half_rank=best[1],
        first_half_integrations=results[1, best[0]][1],
        second_half_integrations=results[2, best[1]][1],
        split_seconds=float((ts[mid] - ts[0]).sec),
        combined_correlation=ns["image_corr"](combined(*best), fullref),
        best_fixed_rank=int(old.protected_clean.idxmax()),
        best_fixed_correlation=float(old.protected_clean.max()),
    )
    assert np.isfinite(record["combined_correlation"])
    adaptive.append(record)
    print("PASS: all 11 recombined fixed-rank controls match Window", w + 1, flush=True)
    print(json.dumps(record), flush=True)

pd.DataFrame(rows).to_csv(OUT / "correlations.csv", index=False)
pd.DataFrame(adaptive).to_csv(OUT / "adaptive_ranks.csv", index=False)
