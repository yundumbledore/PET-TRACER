# Bayesian kinetic model selection

This module releases **pretrained inference and the interactive viewer**. Model-selection dataset generation, priors and training workflows are not included. They are planned for a future release.

## Included demonstration

- FDG subject **P001, coronal slice 116**.
- 35 measured frames and **83,608 foreground voxel TACs**.
- The V8TD **epoch-1169 EMA checkpoint**, its original architecture and scaling parameters.
- Model indices: **0 = 0TC**, **1 = 1TC**, **2 = 1TC-Dual**, **3 = 2TC**, **4 = 2TC-Dual**, **5 = 3TC**, **6 = 3TC-Dual**.
- A slice-only mask extracted from `P001_mask.npy[:, 116, :]`, preserving C-order masked-voxel identity. The full volume is unnecessary for this demo.

Only this subject/slice is enabled in the current GUI. Other subjects and custom protocols will be added later. The model's learned posterior depends on its simulation distribution; anatomical plausibility does not establish physiological ground truth.

## Browser workflow

Launch PET-TRACER with the usual launcher. The welcome page has three independent modules. Choose **Bayesian Model Selection**.

1. **P001 demonstration**: review the supplied subject, model library and reference map.
2. **Inference settings**: set posterior draws per voxel, reverse sampling steps, voxel curve batch size, posterior sample chunk size, device and random seed. Settings save automatically for this project.
3. **Run inference**: start sampling and follow progress. Each completed run gets a new output folder. Stop requests discard incomplete results instead of publishing a partial map.
4. **Explore results**: click a voxel on the map, then click a model-probability bar. The viewer shows that model's conditional parameter posteriors and forward-simulates every selected-model joint draw using **its own kinetics and input-function delay**. The plot shows the posterior mean curve and a transparent pointwise 95% credible band (2.5th–97.5th percentiles). Coordinates match the notebook's original x/y convention, with row zero at the bottom of the display.

The map can use opacity to show the winning model's posterior probability. A parameter summary reports medians and 95% equal-tailed credible intervals. Curve summaries preserve dependencies within each joint parameter/delay draw and condition on the selected model. They describe uncertainty in the noiseless model response, without adding new measurement noise; the band is pointwise rather than a simultaneous 95% envelope. Non-finite curves are omitted with their count reported; finite curves with nonphysical parameters are retained and flagged without clipping samples. The tissue y-axis follows the observed voxel TAC, and the aorta input uses a separate right-hand scale.

Download the model-selection CSV, map PNG, parameter-posterior PNG or selected-voxel joint samples from the viewer.

## Sampling and memory

Defaults are **200 draws per voxel and 3 reverse CM sampling steps**, following the research inference script. Voxel batch size **16** and sample chunk **64** are conservative starting values instead of expanding 1,000 × 200 draws simultaneously. They may be increased on suitable hardware. A seed of 42 makes repeated runs with the same device and chunk settings reproducible subject to backend determinism; changing batch/chunk size changes random-number assignment and does not imply identical Monte Carlo samples.

Conditioning uses the original concatenation **[35 TAC values, 35 AIF values]**, its saved mean/std, and the original inverse transform for the 11-element joint output:

```text
K1, k2, k3, k4, k5, k6, Vb, ka, fA, model_index, delay_min
```

Model indices are rounded and clipped to 0–6; inactive kinetic parameters are zeroed per draw. **Delay is preserved for every model, in minutes.** The forward models, dense-grid Euler convention and delay shift are ported from `InteractiveViewer.py`.

The full 200-draw array occupies approximately **0.69 GiB on disk** before any optional export. Sampling writes to a memory-mapped array and the viewer loads only the selected voxel's samples. Enabling notebook-compatible `Posteriors.npz` requires additional storage and compression time.

CPU, supported CUDA GPUs and Apple MPS are selectable. Auto uses CUDA, then MPS, then CPU. MPS expanded batches are capped at 4,096 draws and the actual model evaluation is checked against CPU before publishing results. Device availability alone does not establish numerical equivalence on every hardware/software combination.

## Terminal workflow

From `LoRACM/`, using the same Python environment as the GUI:

```bash
python model_selection/infer.py --output /path/to/project/results/P001_run1 --samples 200 --steps 3 --batch-size 16 --sample-chunk 64 --device auto --seed 42
```

The output folder must not already exist. A small installation check is available without running the entire slice:

```bash
python model_selection/infer.py --output /path/to/project/results/installation_check --samples 16 --steps 3 --batch-size 2 --sample-chunk 8 --device cpu --limit 4
```

Omit `--limit` to infer the full slice. Add `--export-npz` for the old notebook-compatible posterior archive.

Each result contains:

```text
posteriors.npy       Memory-mapped joint samples [voxel, draw, 11]
selection.npz        Seven model probabilities, MAP index, confidence and voxel identities
Model_Selection.csv  Per-voxel MAP index and posterior probability
metadata.json        Exact checkpoint/data hashes, settings, device and elapsed time
Posteriors.npz       Optional compatibility archive
```

The included asset manifest records checkpoint, config, TAC and mask hashes. No precomputed 406-MB posterior archive is included; users run inference with the supplied model.

## Scope of the release

The seven model definitions and decoder are included because inference and forward checks need them. The original notebook's ipympl desktop dashboard is reimplemented as browser components; it does not require a Jupyter server. Neither the model-selection trainer nor simulator/prior code is released by this module.

The original CM posterior-estimation and LoRACM training modules continue to support the UC Davis example, user-supplied labelled data and custom kinetic models. This fixed P001 demonstration does not expand those models' transfer-validation claims.
