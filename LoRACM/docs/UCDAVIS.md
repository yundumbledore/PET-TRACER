# UC Davis demonstration · Scenario 1

[User manual](../README.md) · [Installation](INSTALLATION.md) · [Prediction guide](PREDICTION.md)

## Default six-step walkthrough (all three platforms)

Launch the GUI and choose a fresh project name. **UC Davis Scenario 1** is the default example setup. Choose **Quick installation check** to learn quickly, or retain **Full Scenario 1** for the research-sized settings. Settings are editable at every numbered step; unfinished edits and saved settings are retained when you revisit any step. The sidebar’s **Project defaults** describes both scales; these initialise the project and never reset it during navigation.

| Step | Action | UC Davis default |
| :--- | :--- | :--- |
| **1 · Model & protocol** | Review and save the model, AIFs, priors and schedule | FDG 2TC, four AIFs, 35 Sub001 frames over 60 minutes, half-life 109.8 min |
| **2 · Dataset** | Generate, then validate/select the labelled NPZ | Full: 50,000 / 100,000 / 100 pairs; small: 32 / 16 / 8 |
| **3 · Fine-tune** | Review settings and train | Rank 4, alpha 8; full: 5,000 epochs; small: two CPU epochs |
| **4 · Predict** | Predict the supplied patient slice | All 24,237 foreground TACs; full: 1,000 draws; small: eight draws per voxel |
| **5 · Single TAC & NLS** | Choose one patient TAC, inspect its posterior and optional NLS fit | Same input data and saved forward-model protocol |
| **6 · Parametric imaging** | Display Kᵢ or a parameter/interval-width map; click voxels for TAC and posterior KDE | Supplied author-confirmed coronal geometry, Kᵢ mean selected initially |

The small model and low draw count check installation; they do not produce scientifically useful estimates. It is reasonable to inspect a few TACs before running whole-slice prediction. The [imaging guide](PARAMETRIC_IMAGING.md) explains voxel interaction and uncertainty.

The original AIF CSV has slightly different midpoints from the patient HDF5. The new walkthrough explicitly uses the **patient's exact frame midpoints and durations**. `data/aif/UCDavis_Sub001_schedule.csv` linearly interpolates all four original AIFs to those midpoints, adds zero activity at the origin, and holds the last value to 60 minutes. `configs/protocol_ucdavis.json` records this alignment. This preserves Scenario 1 kinetics, priors and training settings while making prediction schedule-compatible. It uses the portable RK4 simulator, explicit durations and midpoint observations; it is not an exact numerical reproduction of the historical Euler simulation. The source CSV and historical configs remain unchanged.

Training/validation share the four-AIF pool; test kinetic priors use the central 50% of each range. These are not subject-held-out splits. New custom projects retain the separate 100,000-pair / rank-8 / alpha-16 defaults.

### Numbered terminal walkthrough

Run from `PET-TRACER/LoRACM` after [installation](INSTALLATION.md). Choose a fresh workspace:

```bash
# 1. Export the default UC Davis model and training settings.
python ucdavis.py --workspace workspaces/ucdavis_walkthrough --size small
# 2. Simulate labelled pairs on the measured patient's schedule.
python simulate_protocol.py --config workspaces/ucdavis_walkthrough/protocol.json
python protocol.py workspaces/ucdavis_walkthrough/dataset.npz
# 3. Fine-tune and follow the reported training/validation losses.
python train.py --config workspaces/ucdavis_walkthrough/train.json
# 4. Predict every foreground voxel, including 95% intervals.
python infer.py --run workspaces/ucdavis_walkthrough/runs/scenario1_small --input data/real/Sub001_slice75_adjusted.h5 --output workspaces/ucdavis_walkthrough/runs/scenario1_small/predictions.npz --samples 8 --batch-size 64 --sample-chunk 8 --device cpu
# 5. Save and inspect one voxel's joint posterior samples.
python inspect_voxel.py --run workspaces/ucdavis_walkthrough/runs/scenario1_small --input data/real/Sub001_slice75_adjusted.h5 --voxel-id 33449 --output workspaces/ucdavis_walkthrough/voxel_33449.npz --samples 1000 --device cpu
python analyse_tac.py --protocol workspaces/ucdavis_walkthrough/runs/scenario1_small/protocol.json --input workspaces/ucdavis_walkthrough/voxel_33449.input.npz --output workspaces/ucdavis_walkthrough/voxel_33449_nls.json
python plot_posteriors.py --posterior workspaces/ucdavis_walkthrough/voxel_33449.npz --nls workspaces/ucdavis_walkthrough/voxel_33449_nls.json --output workspaces/ucdavis_walkthrough/voxel_33449_posteriors.png
# 6. Export parameter and uncertainty images.
python parametric_imaging.py --predictions workspaces/ucdavis_walkthrough/runs/scenario1_small/predictions.npz --parameter Ki --output workspaces/ucdavis_walkthrough/Ki_mean.png
python parametric_imaging.py --predictions workspaces/ucdavis_walkthrough/runs/scenario1_small/predictions.npz --parameter Ki --statistic ci95_width --output workspaces/ucdavis_walkthrough/Ki_ci95_width.png
```

For a full run, omit `--size small`; the generated run name is `scenario1`. Use a larger posterior sample count and an appropriate compute device. The **UC Davis demonstration** shortcut page retains the earlier supplied-data/AIF-grid example; the numbered pages are the end-to-end patient-imaging route.

## Historical Scenario 1 workflow

The terminal instructions below retain the historical JAX generation conventions. Install the optional legacy requirements to regenerate pairs, or use the distributed `data/simulated/scenario1.npz` and skip generation.

Adapt a pretrained PET-TRACER consistency model to **FDG–2TCM on the UC Davis uEXPLORER protocol**, then estimate voxel posteriors and display a coronal parametric image.

## 0. Set up

Run all commands below from **`PET-TRACER/LoRACM`**. Paths in configuration files are relative to that working directory.

```bash
cd PET-TRACER/LoRACM
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the supporting packages:

```bash
python -m pip install -r requirements-legacy.txt
```

Install a PyTorch build suitable for your hardware using the [official installation selector](https://pytorch.org/get-started/locally/). The generator uses JAX with 64-bit precision; CPU generation is sufficient for the included dataset. Training supports CPU, CUDA and an optional MPS path. CUDA is recommended for full training; multi-GPU inference uses PyTorch DataParallel on a **single node**, not distributed multi-node execution. Hardware memory requirements depend on batch size and sample count.

Files needed for the complete demo:

| Asset | Location |
| :--- | :--- |
| Four measured AIFs | `data/aif/UCDavis_1H35.csv` |
| Foundation checkpoint | `checkpoints/foundation.pth` |
| Pre-generated adaptation pairs | `data/simulated/scenario1.npz` |
| Real coronal-slice TACs | `data/real/Sub001_slice75_adjusted.h5` |
| Mask, column mapping and display metadata | `data/real/Sub001_slice75_geometry.npz` and `.json` |

This repository checkout includes all five asset groups above, including the generated Scenario 1 dataset and foundation weights. File checksums are recorded in the root `MANIFEST.json`. For a separately archived research release, retain the same assets and record their permanent identifiers.

## 1. Generate adaptation pairs

Either use the supplied pre-generated NPZ or generate a new one:

```bash
python create_dataset.py --config configs/dataset_scenario1.json
```

If the NPZ already exists, move it aside or change `output` in the configuration; the generator refuses to overwrite it. Each sample combines a simulated tissue curve with one measured AIF. Training and validation use broad parameter priors; the supplied research script narrows test priors to the central 50% of each range. Train/validation/test pairs share the same four-AIF pool and are **not subject-held-out splits**.

| Setting | Scenario 1 |
| :--- | :--- |
| Training / validation / test pairs | 50,000 / 100,000 / 100 |
| Kinetic parameters | K₁, k₂, k₃, k₄, Vᵦ |
| Conditioning | 35 TAC values followed by 35 AIF values |
| Noise amplitude | Uniform [0.1, 7] |
| Half-life | 109.8 minutes |
| Seed | 42 |
| Simulation chunk size | 256 |

The generator writes `scenario1.npz` and a provenance JSON. It preserves the supplied forward-model and noise equations. Chunked generation uses new, explicitly recorded random streams, so its samples are **not byte-identical to the original research run**.

## 2. Fine-tune

```bash
python train.py --config configs/train_scenario1.json
```

The configuration retains rank **4**, alpha **8**, dropout **0.05**, batch size **512**, initial learning rate **0.0001**, and **5,000 epochs**. Normalization parameters and LoRA weights are trainable. The model builder transfers compatible foundation weights and trains missing or resized parameters.

Outputs appear under `runs/scenario1/`:

- `model_params.json`: resolved model and training configuration.
- `scaling_params.json`: training-set normalization and time grid.
- `best.pth`: EMA checkpoint with the lowest validation loss across every completed epoch. The historical research script used epoch 151 eligibility; the current workflow deliberately saves from epoch 1 so an early-stopped run remains usable.
- `last.pth`: final EMA checkpoint, including for short runs.
- `tb/`: TensorBoard logs.

Keep the checkpoint, model configuration and scaling file together. Checkpoints contain the **full adapted model**, not just LoRA matrices. They are inference checkpoints, not complete optimizer/RNG snapshots for exact training resumption. Change `exp_name` for independent runs; training refuses to overwrite an existing run directory.

## 3. Predict voxel posteriors

The distributed real slice has a different time grid from the adaptation AIF CSV. Standard inference now rejects that mismatch. Adapt to the measured schedule or supply a validated, appropriately resampled input before running the command below. The override is only for installation checks.

After training and aligning your input protocol, run

```bash
python infer.py --run runs/scenario1 --samples 1000 --batch-size 16 --sample-chunk 64
```

Posterior draws are generated in small GPU chunks. A voxel batch's samples are retained on CPU to calculate exact medians. Reduce `--batch-size` or `--sample-chunk` if memory is limited. To enable specific GPUs, for example Mac mps, use the command

```bash
python infer.py --run runs/scenario1 --device mps --samples 100 --batch-size 16 --sample-chunk 64
```

For multiple GPUs on one node, expose the allocated GPUs with your cluster's scheduler and add:

```bash
python infer.py --run runs/scenario1 --multi-gpu --samples 1000 --batch-size 32 --sample-chunk 128
```

The output `runs/scenario1/predictions.npz` contains correctly named parameter means, medians and standard deviations, and the corresponding Kᵢ statistics. Kᵢ is calculated **for every joint posterior draw**, then summarised. No silent positivity clipping is applied. Nearly zero denominators are excluded and `Ki_valid_fraction` records their effect. Check the negative-draw fractions and Kᵢ tails before interpreting maps.

## 4. Make parametric images

```bash
jupyter lab notebooks/01_parametric_imaging.ipynb
```

The original notebook remains available. For identity-checked mapping, selectable parameters, 95% interval widths and interactive voxel inspection, use the new [GUI/terminal imaging module](PARAMETRIC_IMAGING.md). The author has confirmed the supplied mask mapping and display convention; a matching voxel count alone still does not establish correspondence for other datasets.

## Quick installation smoke test

This tiny run checks the plumbing. It does not produce scientifically useful adapted weights.

```bash
python create_dataset.py --config configs/dataset_smoke.json
python train.py --config configs/train_smoke.json
python infer.py --run runs/smoke --checkpoint last.pth --output runs/smoke/predictions.npz --samples 8 --batch-size 2 --sample-chunk 4 --max-voxels 4 --allow-time-mismatch --device cpu
python -m unittest discover -s tests -v
```

If a generated smoke dataset, run directory or prediction file already exists, move it aside before repeating the command. The viewer's partial-voxel support is for debugging only; a four-voxel smoke output is not a total-body map.

## Further reading

[Custom input functions](CUSTOM_DATA.md) · [Data formats](DATA_FORMATS.md) · [Scientific/release review](RELEASE_REVIEW.md) · [Provenance](DATA_PROVENANCE.md) · [Validation report](VALIDATION.md)
