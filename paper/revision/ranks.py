"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

import json
import numpy as np
import pandas as pd
from common import OUT, REPO, load_namespace

ns = load_namespace()
rows = []
for w in [1, 0, 2, 3]:
    sel = slice(int(ns["starts"][w]), int(ns["stops"][w]))
    ts = ns["observation"].time[sel]
    target, valid = ns["load_window"](sel, ns["config"].target_channel)
    clean, cv = ns["load_window"](sel, ns["clean_channel"])
    dirs = ns["sun_steering"](ts, ns["target_frequency"])
    cd = ns["sun_steering"](ts[cv], ns["clean_frequency"])
    cm = clean[cv].mean(axis=0)
    shrunk, _ = ns["shrink_directional_eigenmodes_to_noise"](
        cm, cd, ns["observation"].snapshots_per_integration * int(cv.sum())
    )
    ref = ns["image_of"](shrunk, ns["clean_frequency"])
    model = ns["physical_haslam_model"](ts)
    rotated = ns["rotated_haslam_model"](ts, 50.0)

    def run(rank, mod, shrink=True):
        return ns["_null_and_restore_uncached"](
            target,
            dirs,
            valid,
            rank,
            model=mod,
            protect_sun=True,
            shrink_sun=shrink,
            reduce="running",
        )

    template = ns["image_of"](run(0, model, False)["removed_mean"])
    rtemplate = ns["image_of"](run(0, rotated, False)["removed_mean"])
    for rank in range(11):
        p = ns["image_of"](run(rank, model)["covariance"])
        u = ns["image_of"](run(rank, None)["covariance"])
        r = ns["image_of"](run(rank, rotated)["covariance"])
        row = dict(
            window=w + 1,
            fixed_rank=rank,
            protected_clean=ns["image_corr"](p, ref),
            protected_template=ns["image_corr"](p, template),
            unprotected_clean=ns["image_corr"](u, ref),
            rotated_50_protected_template=ns["image_corr"](r, rtemplate),
        )
        assert np.isfinite(list(row.values())).all()
        rows.append(row)
        print(json.dumps(row), flush=True)
    if w == 1:
        old = pd.read_csv(REPO / "paper/haslam_fixed7_window2_rank_sweep_120s.csv")
        new = pd.DataFrame(rows)
        for col in [
            "protected_clean",
            "protected_template",
            "unprotected_clean",
            "rotated_50_protected_template",
        ]:
            np.testing.assert_allclose(new[col], old[col], atol=2e-5, rtol=0)
        print("PASS: all four Window 2 curves reproduce published CSV", flush=True)

pd.DataFrame(rows).to_csv(OUT / "rank_sweeps.csv", index=False)
