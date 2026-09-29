# LoRACM · Scenario 1

Adapt a pretrained PET-TRACER consistency model to **FDG–2TCM on the UC Davis uEXPLORER protocol**, then estimate voxel posteriors and display a coronal parametric image.

## 0. Set up

Run all commands below from **`PET-TRACER/LoRACM`**. Paths in configuration files are relative to that working directory.

```bash
cd PET-TRACER/LoRACM
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux, or `.venv\Scripts\Activate.ps1` in Windows PowerShell. Then install the supporting packages:

```bash
python -m pip install -r requirements.txt
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

This repository archive includes all five asset groups above, including the generated Scenario 1 dataset and foundation weights. File checksums are recorded in the root `MANIFEST.json`. For a separately archived research release, retain the same assets and record their permanent identifiers.

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
- `best.pth`: best EMA checkpoint from epoch 151 onward, matching the supplied script's eligibility rule.
- `last.pth`: final EMA checkpoint, including for short runs.
- `tb/`: TensorBoard logs.

Keep the checkpoint, model configuration and scaling file together. Checkpoints contain the **full adapted model**, not just LoRA matrices. They are inference checkpoints, not complete optimizer/RNG snapshots for exact training resumption. Change `exp_name` for independent runs; training refuses to overwrite an existing run directory.

## 3. Predict voxel posteriors

After the time-grid issue is resolved:

```bash
python infer.py --run runs/scenario1 --samples 1000 --batch-size 16 --sample-chunk 64
```

Posterior draws are generated in small GPU chunks. A voxel batch's samples are retained on CPU to calculate exact medians. Reduce `--batch-size` or `--sample-chunk` if memory is limited.

For multiple GPUs on one node, expose the allocated GPUs with your cluster's scheduler and add:

```bash
python infer.py --run runs/scenario1 --multi-gpu --samples 1000 --batch-size 32 --sample-chunk 128
```

The output `runs/scenario1/predictions.npz` contains correctly named parameter means, medians and standard deviations, and the corresponding Kᵢ statistics. Kᵢ is calculated **for every joint posterior draw**, then summarised. No silent positivity clipping is applied. Nearly zero denominators are excluded and `Ki_valid_fraction` records their effect. Check the negative-draw fractions and Kᵢ tails before interpreting maps.

For an explicitly exploratory run on the supplied files, `--allow-time-mismatch` records the exception in the output metadata. It does not correct or resample either dataset and must not be described as a validated protocol reproduction.

## 4. Make parametric images

```bash
jupyter lab notebooks/01_parametric_imaging.ipynb
```

Set the prediction and geometry paths in the first configuration cell, then run the notebook. It maps predictions through explicit voxel indices, verifies voxel column identities, displays parameter/Kᵢ summaries and saves PNG images in the run's `figures/` folder. It stops until `mapping_verified` and `orientation_and_spacing_verified` in the geometry JSON have been confirmed by the data author. A matching voxel count alone does not establish spatial correspondence.

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

[Custom input functions](docs/CUSTOM_DATA.md) · [Data formats](docs/DATA_FORMATS.md) · [Scientific/release review](docs/RELEASE_REVIEW.md) · [Provenance](docs/DATA_PROVENANCE.md) · [Validation report](docs/VALIDATION.md)
