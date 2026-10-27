"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

import json
from dataclasses import replace
import numpy as np
import pandas as pd
from common import OUT, REPO, load_namespace

ns = load_namespace()
obs = ns["observation"]
rows = []
checks = []


def load(sel, ch, pol):
    o = replace(obs, config=replace(obs.config, polarisations=(pol,)))
    return o.load_channel(sel, ch)


def valid(c):
    return np.isfinite(c).all(axis=(1, 2)) & (
        np.diagonal(c, axis1=-2, axis2=-1).real > 0
    ).all(axis=1)


for w in [1, 0, 2, 3]:
    sel = slice(int(ns["starts"][w]), int(ns["stops"][w]))
    ts = obs.time[sel]
    target = {p: load(sel, 4, p) for p in ["XX", "YY"]}
    clean = {p: load(sel, 0, p) for p in ["XX", "YY"]}
    for d in [target, clean]:
        d["I"] = (d["XX"] + d["YY"]) * 0.5
    original = obs.load_channel(sel, 4)
    np.testing.assert_allclose(original, target["I"], rtol=1e-6, atol=1e-5)
    checks.append(
        {
            "window": w + 1,
            "recombined_input_max_error": float(np.max(np.abs(original - target["I"]))),
        }
    )
    mask = np.logical_and.reduce(
        [valid(c) for c in [*target.values(), *clean.values()]]
    )
    dirs = ns["sun_steering"](ts, ns["target_frequency"])
    cd = ns["sun_steering"](ts[mask], ns["clean_frequency"])
    model = ns["physical_haslam_model"](ts)
    for pol in ["I", "XX", "YY"]:
        cm = clean[pol][mask].mean(axis=0)
        rawref = ns["image_of"](cm, ns["clean_frequency"])
        shrunk, _ = ns["shrink_directional_eigenmodes_to_noise"](cm, cd)
        ref = ns["image_of"](shrunk, ns["clean_frequency"])
        for mode in ["combined", "sun_only"]:
            for rank in [0, 1, 2, 3] if mode == "combined" else [1, 2]:
                result = ns["_null_and_restore_uncached"](
                    target[pol],
                    dirs,
                    mask,
                    rank,
                    model=model if mode == "combined" else None,
                    protect_sun=True,
                    shrink_sun=mode == "combined",
                    reduce="running",
                )
                im = ns["image_of"](result["covariance"])
                row = {
                    "window": w + 1,
                    "pol": pol,
                    "mode": mode,
                    "rank": rank,
                    "correlation_clean": ns["image_corr"](
                        im, ref if mode == "combined" else rawref
                    ),
                    "integrations": int(result["valid"].sum()),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
        if w == 1 and pol == "I":
            stored = pd.read_csv(
                REPO / "paper/haslam_fixed7_window2_rank_sweep_120s.csv"
            )
            vals = [
                r["correlation_clean"]
                for r in rows
                if r["window"] == 2 and r["pol"] == "I" and r["mode"] == "combined"
            ]
            np.testing.assert_allclose(
                vals, stored.protected_clean.iloc[:4], atol=2e-5, rtol=0
            )
            print("PASS: averaged-hand control reproduces paper rank sweep", flush=True)
    del target, clean, model, original
pd.DataFrame(checks).to_csv(OUT / "single_pol_checks.csv", index=False)

pd.DataFrame(rows).to_csv(OUT / "single_pol.csv", index=False)
