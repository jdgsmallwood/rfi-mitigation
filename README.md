# spatial-filtering-rfi-mitigation

Code and figures for

> Smallwood et al., *Protecting astronomical sources during adaptive spatial
> filtering for radio frequency interference mitigation with the LAMBDA dipole
> array*

using data from **LAMBDA**, a 36-element dipole array at Narrabri.

<!-- TODO(before deposit): replace with the minted software DOI badge. -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://spdx.org/licenses/MIT.html)

## What this does

Radio-frequency interference from an Orbcomm downlink near 137.5 MHz is removed
from LAMBDA visibilities *without* removing the sky along with it. Per 17.7 ms
integration, on the short-term covariance $R_k$:

1. **Peel the known sky** — subtract a scaled Haslam-408 model covariance, the
   scale capped per integration so the residual stays positive semi-definite
   (`psd_safe_subtract`).
2. **Whiten**, then **null the interference subspace** — project out the leading
   `rank` eigenmodes, while **protecting** chosen directions (the Sun) so the
   projector cannot eat them.
3. **Restore** the peeled sky model and image by direct DFT of the visibilities.

Channel 0, RFI-free at the same time on the same sky, is the reference:
successful mitigation makes the mitigated channel 4 resemble channel 0.

## Reproducing the paper figures

### 1. Install

Dependencies are managed with [Hatch](https://hatch.pypa.io/) (install it with
`pipx install hatch`). From the repository root:

```
hatch shell
```

This creates the environment and installs the package in editable mode. Every
dependency is pinned in `pyproject.toml` to the version that produced the
published figures (Python 3.13.1); `requirements-frozen.txt` records the full
transitive set.

Sanity check:

```
python -c "import spatial_filtering_rfi_mitigation; print(spatial_filtering_rfi_mitigation.__file__)"
pytest
```

### 2. Get the data

The notebook consumes two visibility captures of ~10 GiB each, which are **not
in this repository** — they are archived as a separate Zenodo data record:

> **Data:** *LAMBDA36 Orbcomm visibility captures, 2026-06-11* —
> <https://doi.org/10.5281/zenodo.22255363>

| File | Role |
|---|---|
| `visibilities_20260611-1511_172_179_ALVEO0_ALVEO1_ALVEO2_ALVEO3.hdf5` | target capture |
| `visibilities_20260611-1349_172_179_ALVEO0_ALVEO1_ALVEO2_ALVEO3.hdf5` | earlier Sun capture, used to solve the gains |

Point `$RFI_DATA_PATH` at the directory holding them (copy
`rfi_env.sh.example` to `rfi_env.sh` and edit, then `source rfi_env.sh`),
or place the two files — or symlinks to them — at the repository root. The
notebook searches `$RFI_DATA_PATH`, then the repository root, then the
working directory, and names the DOI if it finds neither.

### 3. Run the notebook

```
cd paper
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --inplace paper-figures.ipynb
```

Every figure and table is written next to the notebook in `paper/`. Expensive
per-window results are memoised under `paper/cache/` keyed by an md5 of their
inputs; set `USE_DISK_CACHE = False` in cell 1 to force full recomputation.

### The trained Haslam amplitude

The notebook takes the sky-model amplitude `trained_haslam_scale = 42.84` as a
fixed constant, so no free parameter is tuned against the results it presents.
That number is fitted separately, by `paper/fit_haslam_scale.py`:

```
cd paper && python fit_haslam_scale.py
```

The fit runs on the satellite-free start of the capture (integrations
`[2000, 26000)`, 24 blocks of 300 integrations) and against the **clean**
reference channels 0 and 1 — the target channel carries the downlink and is
never RFI-free in time. Per block and channel it fits non-negative sky and Sun
amplitudes to the upper-triangular cross-visibilities (autocorrelations
excluded), real and imaginary parts stacked, with a `soft_l1` loss scaled by the
MAD of an initial bounded least-squares residual
(`spatial_filtering_rfi_mitigation.low_rank_sky_experiment._fit_sky_and_sun`). Blocks are
combined by median with a MAD-derived spread:

| | |
|---|---|
| Haslam scale | 42.84 ± 4.8 (MAD over 24 blocks) |
| Sun power | 2.084e+04 ± 5.9e+03 |
| Median two-template fit residual fraction | 0.697 |

Re-running the script reproduces 42.840431127 against the published
42.840432534 — 1.4e-06 apart, or 3e-8 relative, from the convergence tolerance
of the least-squares refinement under a newer scipy. That is eight orders of
magnitude below the fit's own ±4.8 spread. The notebook keeps the published
value so its figures remain exactly those in the paper.

Only the unrotated template is ever fitted. The rotated sky used in the
falsification control is forced to this same amplitude and never tuned to the
data.

### Figures produced

| Output (`paper/`) | Content |
|---|---|
| `lambda36-array-layout.pdf` | LAMBDA36 antenna positions |
| `haslam_fixed7_window2_rank_sweep_120s.{pdf,csv}` | correlation against the clean channel vs nulled rank, with the 50°-rotated-sky falsification control |
| `haslam_fixed7_window2_rank_sweep_dirty_images_120s.png` | dirty images across the rank sweep |
| `haslam_fixed7_window2_rank2_sun_protection_120s.png` | raw / post-null / restored, showing the Sun surviving the projector |
| `haslam_fixed7_window2_sun_protection_isolation_120s.{png,csv}` | protection isolated: identical pipeline, protection the only difference |
| `haslam_fixed2_dirty_protection_120s_sun_shrunk.pdf` | the four 120 s windows at rank 2, all five image products |
| `haslam_fixed7_peak_suppression_120s.{png,csv}` | peak suppression per window, in dB relative to raw |
| `haslam_fixed7_satellite_free_sun_only_correlation_120s.csv` | satellite-free control stretch, Sun only |
| `channel_scan_correlation_flux_window2_120s.pdf` | per-channel correlation with the clean channel, and in-horizon flux |
| `channel_{vs_clean_correlation,flux_stats}_window2_120s.csv` | the tables behind that figure |

## Repository layout

| Path | Contents |
|---|---|
| `src/spatial_filtering_rfi_mitigation/` | the package (see below) |
| `paper/` | the paper notebook, `build_rfi_movie.py`, and the figures/tables it produces |
| `tests/` | pytest suite, mirroring the module names |

Package modules, by concern:

| Module | Role |
|---|---|
| `haslam_protection.py` | the mitigation core: `psd_safe_subtract`, `fixed_rank_subspace_null_batch`, `shrink_directional_eigenmodes_to_noise`, `interpolate_covariances` |
| `protected_pipeline.py` | run configuration and imaging: `ProtectedRunConfig`, `prepare_observation`, `steering_vectors`, `dirty_image`, and the Sun / satellite / Galactic-plane ephemerides |
| `low_rank_sky_experiment.py` | beam-weighted Haslam sky-model covariances, and `_fit_sky_and_sun`, the fit behind the trained amplitude |
| `utils.py` | HDF5 capture and gain reading, and the LAMBDA36 telescope model |
| `interferometers.py` | `RadioArray`: array geometry and baselines |
| `vistools.py`, `modelling.py`, `constants.py` | DFT imaging, direction cosines, site constants |

This package contains only what the paper's figures need. The commissioning and
calibration tooling this code grew out of — the `run-diagnostics` / `run-convert`
CLIs, Sun holography calibration, inter-ALVEO delay calibration, raw packet
readers, UVFITS conversion, MIRIAD interop, SEFD estimation — lives in the
predecessor repository and is deliberately not archived here.

## Citing

Cite both the software and the data record. Metadata for the software is in
`CITATION.cff` (GitHub renders a "Cite this repository" widget from it).

## License

MIT — see `LICENSE.txt`.
