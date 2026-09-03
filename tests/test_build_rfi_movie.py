import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "paper"))
from build_rfi_movie import load_valid


def test_load_valid_flags_nan_and_nonpositive_autos():
    cube = np.ones((3, 2, 2), dtype=np.complex64)
    cube[1, 0, 0] = np.nan          # bad: non-finite
    cube[2, 1, 1] = -1.0            # bad: non-positive autocorrelation
    observation = SimpleNamespace(load_channel=lambda selection, channel: cube)

    _, valid = load_valid(observation, slice(0, 3), channel=0)

    assert list(valid) == [True, False, False]
