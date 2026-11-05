"""Paper and movie geometry must retain the capture's reordered antenna map."""

import json
from importlib.resources import files

import numpy as np

from spatial_filtering_rfi_mitigation.utils import make_telescope_model


def test_geometry_preserves_requested_antenna_order():
    mapping = json.loads(
        files('spatial_filtering_rfi_mitigation.data')
        .joinpath('LAMBDA36-antenna-mappings-reordered.json').read_text()
    )
    antenna_ids = np.array([36, 1, 17, 5])
    model = make_telescope_model(antenna_ids)
    np.testing.assert_array_equal(
        model.east, [mapping[str(i)]['east'] for i in antenna_ids]
    )
    np.testing.assert_array_equal(
        model.north, [mapping[str(i)]['north'] for i in antenna_ids]
    )
    np.testing.assert_allclose(model.height, 211.0, rtol=0, atol=1e-8)
