# Data and model contracts

## AIF CSV

The first column contains positive, increasing time points in **minutes**. Each remaining column is one measured input function, with one activity value per time point. The supplied file contains **35 rows and four AIFs**: Sub001, Sub011, Sub080 and Sub087. The HDF5 file labels activity as Bq/mL; confirm the same units and correction convention for all AIFs before public release.

The same input is used for plasma and blood terms in this implementation. The generator uses midpoint observations, a forward Euler step nominally 0.01 min, linear interpolation of the input, and `searchsorted` observation indices. It does not calculate exact frame averages. The noise term uses `[t[0], diff(t)]` as delta time, preserving the original script; this is not an explicit frame-duration vector.

## Simulation NPZ

| Key | Shape | Meaning |
| :--- | :--- | :--- |
| `x_train`, `x_val`, `x_test` | N × 6 | K₁, k₂, k₃, k₄, Vᵦ, noise amplitude |
| `y_train`, `y_val`, `y_test` | N × 70 | TAC values followed by AIF values |
| `t_meas` | 35 | Time points in minutes |
| `aif_index_train`, `aif_index_val`, `aif_index_test` | N | Zero-based AIF column index, excluding time |

Training uses only the first five columns of x. The sixth column controls simulation noise and is not predicted. Parameter normalization uses five independent training means/stds. Conditioning uses one scalar training mean/std across all TAC and AIF values, preserving the supplied script. No log transform is used.

## Real-data HDF5

A pandas HDF table with a single readable object:

- Rows: 35 time frames.
- Column 0: frame duration in minutes.
- Column 1: time in minutes.
- Column 2: AIF in Bq/mL.
- Columns 3 onward: voxel TACs, in their original order.

The supplied file contains **24,237 voxel TACs**. Integer column labels are preserved as identifiers; they are not assumed to be flat indices in the displayed coronal slice.

## Predictions NPZ

`params_mean`, `params_median`, `params_std`: N × 5, ordered K₁, k₂, k₃, k₄, Vᵦ. `Ki_mean`, `Ki_median`, `Ki_std`: N. Standard deviations use ddof=0. `Ki_valid_fraction`: fraction of samples with |k₂+k₃| > 1e-12. `params_negative_fraction`: N × 5. `voxel_columns`: string identifiers in prediction order. Time vectors and a JSON metadata string record the training schedule, observed schedule and inference settings.

Kᵢ = K₁k₃/(k₂+k₃), evaluated samplewise without clipping. For the reversible 2TCM this derived quantity should be interpreted consistently with the manuscript's use; the demo does not assert irreversibility or a Patlak equivalence.

## Geometry

The geometry NPZ contains `mask` (2D), `voxel_indices` (C-order flat positions in that mask), and `voxel_columns` (corresponding HDF5 identifiers). Its JSON records display origin, pixel aspect ratio and verification flags. The candidate mask is `Sub001_mask[:, 75, :]`, shape **486 × 150**, with **24,237 foreground voxels**. Mapping by row order reproduces the assumption in the original notebook but still requires author verification. Spacing/orientation must be confirmed separately.
