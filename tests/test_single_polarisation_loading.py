"""The HDF5 receptor axes must not depend on the requested hand order."""

import h5py
import numpy as np
import pytest
from spatial_filtering_rfi_mitigation.protected_pipeline import (
    Observation,
    ProtectedRunConfig,
    prepare_observation,
)


@pytest.mark.parametrize(
    "hands,expected",
    [(("XX",), 2), (("YY",), 7), (("XX", "YY"), 4.5), (("YY", "XX"), 4.5)],
)
def test_load_channel_uses_fixed_receptor_axis(tmp_path, hands, expected):
    path = tmp_path / "capture.hdf5"
    raw = np.zeros((2, 1, 1, 2, 2, 2), dtype=np.float32)
    raw[:, 0, 0, 0, 0, 0] = 8
    raw[:, 0, 0, 1, 1, 0] = 63
    with h5py.File(path, "w") as f:
        f["visibilities"] = raw
    obs = Observation.__new__(Observation)
    obs.config = ProtectedRunConfig(str(path), "", polarisations=hands)
    obs.time = np.arange(2)
    obs.selected_start = 0
    obs.antenna_ids = np.array([1])
    obs.good_antennas = np.array([True])
    obs.antenna1 = obs.antenna2 = np.array([1])
    obs.baseline_valid = np.array([True])
    obs.baseline_gain_i = obs.baseline_gain_j = np.array([0])
    obs.gains = {"XX": np.array([[2.0]]), "YY": np.array([[3.0]])}
    np.testing.assert_allclose(obs.load_channel(slice(None), 0), expected)


@pytest.mark.parametrize("hands", [(), ("XY",), ("XX", "XX")])
def test_prepare_rejects_unsupported_hands_before_io(hands):
    with pytest.raises(ValueError, match="polarisations"):
        prepare_observation(
            ProtectedRunConfig("missing", "missing", polarisations=hands)
        )
