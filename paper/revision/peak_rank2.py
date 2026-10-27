"""Peak dirty-image brightness reduction, not isolated satellite suppression."""

import json
from common import OUT, load_namespace

ns = load_namespace()
import numpy as np

rows = []
for w in range(4):
    sel = slice(int(ns["starts"][w]), int(ns["stops"][w]))
    ts = ns["observation"].time[sel]
    cube, valid = ns["load_window"](sel, ns["config"].target_channel)
    result = ns["_null_and_restore_uncached"](
        cube,
        ns["sun_steering"](ts, ns["target_frequency"]),
        valid,
        2,
        model=ns["physical_haslam_model"](ts),
        protect_sun=True,
        shrink_sun=True,
        reduce="running",
    )
    raw = ns["peak_in_disk"](ns["image_of"](cube[valid].mean(axis=0)))
    after = ns["peak_in_disk"](ns["image_of"](result["covariance"]))
    row = dict(
        window=w + 1,
        rank=2,
        raw_peak=float(raw),
        mitigated_peak=float(after),
        after_minus_before_db=float(10 * np.log10(after / raw)),
        reduction_db=float(10 * np.log10(raw / after)),
        valid_integrations=int(result["valid"].sum()),
    )
    rows.append(row)
    print(json.dumps(row), flush=True)
(OUT / "rank2_peak.json").write_text(json.dumps(rows, indent=2))
