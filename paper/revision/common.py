"""Reuse the notebook's tagged setup/toolkit cells without running its plots.

Run from any directory. RFI_DATA_PATH retains the notebook's data-location
convention. Revision results are always recomputed, never read from its cache.
"""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lambda-paper-mpl")
)
from astropy.utils import iers

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "results"


def load_namespace():
    iers.conf.auto_download = False
    OUT.mkdir(exist_ok=True)
    notebook = json.loads((REPO / "paper/paper-figures.ipynb").read_text())
    ns = {"__name__": "revision_analysis"}
    previous = Path.cwd()
    try:
        os.chdir(REPO / "paper")
        for tag in ("revision-setup", "revision-toolkit"):
            cells = [
                c
                for c in notebook["cells"]
                if tag in c.get("metadata", {}).get("tags", [])
            ]
            if len(cells) != 1:
                raise ValueError(f"Expected one notebook cell tagged {tag}")
            if tag == "revision-toolkit":
                ns["output_root"] = OUT
            exec(
                compile(
                    "".join(cells[0]["source"]), f"paper-figures.ipynb:{tag}", "exec"
                ),
                ns,
            )
    finally:
        os.chdir(previous)
    ns["USE_DISK_CACHE"] = False
    return ns
