# Parametric imaging and interactive voxel inspection

[User manual](../README.md) · [UC Davis steps 1–6](UCDAVIS.md) · [Prediction](PREDICTION.md)

**6 · Parametric imaging** converts voxel predictions into an image. The default is **Kᵢ posterior mean** for the supplied UC Davis Sub001 coronal slice. The released example is one slice through the body; it is not a complete 3D patient volume.

![Interactive UC Davis map with selected-voxel TAC, input function and posterior KDE](../../Assets/LoRACM_Imaging_Workbench.jpg)

*Interface preview using the small installation-check model.*

## Choose the imaging model

Select **Model for parametric imaging**: LoRACM or a trained scratch CM. A map always belongs to the selected model’s own prediction file. **Generate a map prediction with the selected model** creates it directly from step 6, with checkpoint, sample count, sampling steps and memory controls. Existing outputs are protected; enter a new prediction filename when regenerating. The clicked-voxel inspector checks the input and checkpoint checksums against the map.

## GUI workflow

1. In step 4, predict **UC Davis patient slice** using the model trained in steps 1–3. Include all 24,237 foreground voxels. Simulated test pairs can assess accuracy but have no anatomical coordinates.
2. Open **6 · Parametric imaging**. The prediction output and model folder follow your preceding run. The supplied geometry is selected automatically.
3. Choose Kᵢ or any predicted kinetic parameter. Choose **Posterior mean**, **Posterior median**, or **95% credible-interval width**.
4. Click a foreground voxel in the image on the **left**. On the **right**, the app displays that voxel's measured TAC and input function in one graph, followed by a KDE of the chosen parameter's posterior. Entering a voxel ID also supports keyboard access.
5. Export the map figure, numeric map or selected voxel's joint posterior samples.

The app generates and caches joint samples only for the selected voxel, using the model/checkpoint and measured data that generated the image. It verifies voxel identities, schedule and file checksums. These fresh Monte Carlo draws can differ slightly from the bulk map summaries. A KDE is a smoothed display of samples; interval limits are empirical percentiles.

The initial colour scale uses the 2nd and 98th percentiles of finite image values. You can change its bounds. Colour scaling does not alter, clip, smooth or filter exported map values. A tiny two-epoch walkthrough model and eight draws per voxel demonstrate the interface; they do not establish posterior calibration or useful research estimates.

## What uncertainty means here

For each voxel and parameter, the equal-tailed 95% credible interval is:

```text
lower = percentile(posterior draws, 2.5)
upper = percentile(posterior draws, 97.5)
width = upper - lower
```

Width is in the parameter's units. It is **not** its posterior standard deviation or the uncertainty of its posterior mean. Wider intervals describe more spread in the model's samples; assess calibration separately.

For built-in 2TC, Kᵢ is calculated for every joint draw as `K1*k3/(k2+k3)`, then its percentiles are calculated. The code does not substitute mean parameters into that equation. Draws with `abs(k2+k3) <= 1e-12` are excluded and `Ki_valid_fraction` reports the retained fraction. Negative draws are not silently clipped. Custom models expose their predicted parameters; this 2TC-specific Kᵢ transform is not imposed on them.

New prediction files contain `params_ci95_low`, `params_ci95_high`, `params_ci95_width`, and corresponding `Ki_*` arrays where applicable. Older files can display available mean/median maps, but need updated inference to obtain interval-width maps. Keeping all whole-image draws is unnecessary: exact percentiles are calculated within each inference batch.

## Geometry for your own project

Supply a geometry NPZ with:

| Array | Required content |
| :--- | :--- |
| `mask` | Binary 2D slice or 3D volume |
| `voxel_indices` | One unique C-order flat foreground index per TAC |
| `voxel_columns` | Matching unique string identifiers, also used in the prediction/input file |

Supply a JSON sidecar with `mapping_verified: true`, `orientation_and_spacing_verified: true`, `origin: "lower"` or `"upper"`, and a positive `pixel_aspect_ratio`. Optional `shape` must match the mask. Verify these facts against the original image/data export before setting the flags. For 3D arrays, select the display axis and slice index; axes are array axes unless you supply an external anatomical interpretation.

Complete imaging requires all foreground voxel identities. Unknown, duplicate, missing or background indices are rejected. Predictions are joined to geometry by **voxel ID**, so their row order can differ without changing spatial correspondence. The UC Davis author confirmed the supplied C-order mapping and original notebook display convention; its geometry JSON records that confirmation. The display uses pixel coordinates because physical millimetre spacing is not supplied.

## Terminal equivalents

Run from `PET-TRACER/LoRACM`. For the small walkthrough:

```bash
python parametric_imaging.py --predictions workspaces/ucdavis_walkthrough/runs/scenario1_small/predictions.npz --parameter Ki --statistic mean --output workspaces/ucdavis_walkthrough/Ki_mean.png
python parametric_imaging.py --predictions workspaces/ucdavis_walkthrough/runs/scenario1_small/predictions.npz --parameter Ki --statistic ci95_width --output workspaces/ucdavis_walkthrough/Ki_ci95_width.png
python parametric_imaging.py --predictions workspaces/ucdavis_walkthrough/runs/scenario1_small/predictions.npz --parameter K1 --statistic ci95_width --output workspaces/ucdavis_walkthrough/K1_ci95_width.png
```

Each command exports a PNG, a numeric map NPZ and a provenance JSON. Pass `--geometry` and `--geometry-metadata` for your own data, and `--axis` / `--slice` for a 3D array. Existing outputs are protected.

Inspect one voxel and save its TAC/AIF and joint samples:

```bash
python inspect_voxel.py --run workspaces/ucdavis_walkthrough/runs/scenario1_small --input data/real/Sub001_slice75_adjusted.h5 --voxel-id 33449 --output workspaces/ucdavis_walkthrough/voxel_33449.npz --samples 1000 --device cpu
python plot_posteriors.py --posterior workspaces/ucdavis_walkthrough/voxel_33449.npz --output workspaces/ucdavis_walkthrough/voxel_33449_posteriors.png
```

The extraction file `voxel_33449.input.npz` can also be used with the [single-TAC NLS commands](PREDICTION.md#terminal-single-tac-analysis).
