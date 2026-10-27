"""Reproduce the revision diagnostic; see README.md for inputs and ordering."""

import json
import numpy as np
import pandas as pd
from common import OUT, REPO, load_namespace

ns = load_namespace()
obs = ns["observation"]
rows = []
summary = []
ranks = pd.read_csv(OUT / "rank_sweeps.csv")
for w in range(4):
    sel = slice(int(ns["starts"][w]), int(ns["stops"][w]))
    ts = obs.time[sel]
    images = []
    for ch, freq in enumerate(obs.frequencies):
        cube, valid = ns["load_window"](sel, ch)
        peeled, _ = ns["shrink_directional_eigenmodes_to_noise"](
            cube[valid].mean(axis=0),
            ns["sun_steering"](ts[valid], float(freq)),
            obs.snapshots_per_integration * int(valid.sum()),
        )
        images.append(ns["image_of"](peeled, float(freq)))
        row = dict(
            window=w + 1,
            channel=ch,
            frequency_MHz=float(freq) / 1e6,
            correlation_with_clean=ns["image_corr"](
                images[-1], images[ns["clean_channel"]]
            ),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    d = pd.DataFrame(rows).query("window==@w+1")
    if w == 1:
        old = pd.read_csv(REPO / "paper/channel_vs_clean_correlation_window2_120s.csv")
        np.testing.assert_allclose(
            d.correlation_with_clean, old.correlation_with_clean, atol=2e-5, rtol=0
        )
        print("PASS: Window 2 reproduces existing channel scan", flush=True)
    other = d.channel != ns["config"].target_channel
    fit = np.polyfit(
        d.loc[other, "frequency_MHz"], d.loc[other, "correlation_with_clean"], 1
    )
    targetfreq = float(ns["target_frequency"]) / 1e6
    r = ranks.query("window==@w+1").set_index("fixed_rank")
    summary.append(
        dict(
            window=w + 1,
            expected=float(np.polyval(fit, targetfreq)),
            unmitigated=float(d.loc[~other, "correlation_with_clean"].iloc[0]),
            protected_rank2=float(r.loc[2, "protected_clean"]),
            unprotected_rank2=float(r.loc[2, "unprotected_clean"]),
            best_rank=int(r.protected_clean.idxmax()),
            best_protected=float(r.protected_clean.max()),
            slope=fit[0],
            intercept=fit[1],
        )
    )

pd.DataFrame(rows).to_csv(OUT / "channels.csv", index=False)
pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
