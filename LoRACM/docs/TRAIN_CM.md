# Train a consistency model from scratch

[User manual](../README.md) · [LoRACM fine-tuning](FINE_TUNING.md) · [Prediction and NLS](PREDICTION.md)

Train a new CM on your own labelled parameter–curve pairs, then compare it with LoRACM on the same held-out data. The scratch module initializes a plain consistency-model U-Net randomly, loads **no foundation checkpoint**, and trains all learnable model weights. The time encoder's Gaussian Fourier features remain fixed, as in the underlying architecture.

## Choose the comparison you want

| LoRACM | CM from scratch |
| :--- | :--- |
| Starts with the pretrained foundation and adds LoRA adaptation | Starts with random model weights |
| Trains adapters, optional normalization, and new/resized layers | Trains all learnable layers |
| Needs a foundation checkpoint | Needs no pretrained checkpoint |
| `train.py` | `train_cm.py` |

Both use the same dataset contract, training-only normalization, consistency objective, EMA inference model, and validation/checkpoint rules. This is a controlled scratch baseline using the repository's current training objective; it does not claim to recreate every detail or dataset of the original foundation pretraining experiment.

## GUI workflow

1. Prepare and select your labelled NPZ on **2 · Dataset**. You can use the same file for both training routes.
2. Enable **CM from scratch comparison** in the sidebar, then open **CM from scratch**. Parameter dimensions and names appear automatically from the data.
3. Choose a run name, epochs, learning rate, batch size, and device. Default values are ready to use.
4. Open **Advanced settings** only when you need to change embedding width, U-Net channels, optimizer, scheduler, consistency intervals, loss weights, seed, or epoch-checkpoint interval.
5. Choose **Review configuration** to inspect/download the exact settings, then **Start CM training**.
6. Use **4 · Predict** or **5 · Single TAC & NLS** with the scratch run folder. The inference module selects the correct CM architecture automatically.
7. Enable **Overlay CM from scratch posterior** in step 5 for the same voxel, or open **Compare models** for a held-out file. In step 6, select the model and generate its own prediction file before mapping.

Both routes support graceful early stopping and the [same best/last/epoch checkpoint selection](FINE_TUNING.md#interrupted-jobs-and-memory-limits).

The scratch settings are saved independently in `workspaces/<project>/train_cm.json`; LoRACM uses `train.json`. Different run names preserve both models and their metrics.

## Defaults and settings

This table lists general training defaults. The UC Davis GUI preset follows Scenario 1 training settings for matched comparisons; its small walkthrough uses two epochs and batch size 8. Dimensions always follow the selected dataset. An imported labelled dataset can be used without a simulation protocol; a matched protocol is optional for later NLS.

| JSON setting | Default | What it controls |
| :--- | :--- | :--- |
| `epochs` | **5,000** | Overall passes through the training loader |
| `lr` | **0.0001** | Initial AdamW learning rate |
| `batch_size` | **512** | Pairs per training batch; GUI caps the starting value to the dataset size |
| `x_dim` | **`"auto"`** | Number of predicted parameters P, excluding the final noise column |
| `y_dim` | **`"auto"`** | Conditioning width 2T: TAC values, then input-function values |
| `parameter_names`, `loss_weights` | **`"auto"`** | Names from the data, equal parameter loss weights |
| `embed_dim` | **1,024** | Width of time and conditioning embeddings; must be even |
| `channels` | **`[32,64,128,256]`** | Four U-Net widths; first divisible by 4, others by 32 |
| `embedy` | **`"Linear"`** | Conditioning encoder; also supports MLP, Conv1D, GRU and LSTM |
| `sigma_data` | **1.0** | Consistency-model boundary scale in normalized parameter space |
| `noise_levels` | **15** | Number of uniform consistency-noise intervals between 0 and 1 |
| `ema_decay` | **0.99** | Exponential moving average used for the target/inference model |
| `weight_decay` | **0.01** | AdamW regularization |
| `lr_scheduler_patience`, `lr_factor` | **25 / 0.75** | Reduce learning rate after a validation-loss plateau |
| `min_lr` | **0.00000001** | Scheduler floor; must not exceed the initial rate |
| `seed` | **0** | Random initialization and training streams |
| `checkpoint_interval` | **25** | Save selectable epoch snapshots every 25 epochs, plus epoch 1; best and last update every completed epoch |
| `drop_last` | **`true`** | Drop an incomplete final training batch; set false to include it |
| `device` | **`"auto"`** | CUDA, then MPS, then CPU for training |

Use these as starting settings, then assess convergence and held-out posterior quality. Training from scratch can need more data or compute than adaptation. Keep architecture, splits and evaluation settings aligned when studying the effect of initialization. A tiny installation run is not a useful trained research model.

Explicit numeric `x_dim` or `y_dim` values are accepted in JSON/CLI, but must match the dataset; conflicts are rejected before a run folder is created. The GUI displays these dimensions rather than inviting a change that would contradict the input file.

## Terminal training

Run from `PET-TRACER/LoRACM`. Edit `configs/train_cm.json` to point at your dataset, optional matched forward-model protocol, output folder and run name:

```bash
python train_cm.py --config configs/train_cm.json
```

You can override critical settings without editing JSON:

```bash
python train_cm.py --config configs/train_cm.json --epochs 5000 --lr 0.0001 --batch-size 512 --embed-dim 1024 --seed 0 --device cpu
```

The same overrides work with `train.py`. `--x-dim` and `--y-dim` are optional exact schema checks; leave the JSON values as `"auto"` for normal use. A scratch config must not contain a nonempty `pretrained` path; the module rejects it to keep initialization unambiguous.

## Outputs and inference

A run saves `best.pth`, `last.pth`, `model_params.json`, `scaling_params.json`, `metrics.csv`, TensorBoard events, and optional copied kinetic-model source/protocol. `training_summary.json` records the training mode, target-dataset checksum, trainable/total parameter counts, seed, training time and lowest completed validation loss. The transfer report states random initialization and no foundation loading for a scratch CM.

Training and validation graphs are consistency/self-consistency losses. They are not direct parameter RMSE or proof of posterior calibration. Checkpoints contain complete EMA **inference models**, not optimizer/RNG snapshots for exact resumption.

```bash
python infer.py --run workspaces/my_project/runs/cm_scratch --input workspaces/my_project/testing.npz --output workspaces/my_project/cm_predictions.npz --samples 1000 --device cpu
```

The same single-TAC posterior/NLS commands in [the prediction guide](PREDICTION.md) accept a scratch run. Custom kinetic parameter sets and 2TC-specific Kᵢ rules work identically for both model types.

## Compare the two models

Prepare one held-out prediction NPZ/HDF5 and use it for both models. An optional `truth [N,P]` array supplies known parameter labels. Use independent subjects/acquisitions where appropriate, and keep test pairs out of training and model selection.

```bash
python compare_models.py --lora-run workspaces/my_project/runs/adaptation --cm-run workspaces/my_project/runs/cm_scratch --input workspaces/my_project/testing.npz --output workspaces/my_project/comparisons/first --samples 1000 --device cpu
```

The comparison saves `loracm.npz`, `cm.npz`, `metrics.csv` and `comparison.json`. With truth, metrics include parameter **RMSE and bias of posterior means**. Without truth, it reports uncertainty summaries and negative-draw fractions, which do not establish accuracy. The report records dataset-file agreement, training-setting differences and timings. Inference timing includes model loading and file operations; use a dedicated benchmark for throughput claims.

For a small curve set, add `--save-samples`. The GUI then displays overlaid LoRACM and scratch-CM histograms for a selected TAC. To export a single-TAC overlay in a terminal:

```bash
python plot_posteriors.py --posterior workspaces/my_project/comparisons/first/loracm.npz --comparison workspaces/my_project/comparisons/first/cm.npz --output workspaces/my_project/comparisons/first/posteriors.png
```

The plot command requires exactly one TAC. Saved joint draws are capped at two million parameter values per model. Match sampling steps, sample count and inference seed; repeat training/evaluation across seeds for robust conclusions. Inspect fit quality, tail behaviour and calibration separately. LoRACM's foundation pretraining is an additional resource, even when both methods use the same target dataset and fine-tuning/training epochs.
