# Module 1: Train a model

New simulated training data use [log-normal measurement noise with actual frame durations](MEASUREMENT_NOISE.md). The supplied checkpoints retain their original Gaussian training provenance. Generate a fresh dataset/run to evaluate the new noise model.

[User manual](../README.md) · [LoRACM fine-tuning](FINE_TUNING.md) · [Prediction and NLS](PREDICTION.md)

Module 1 learns to sample kinetic parameters given a tissue time–activity curve (TAC) and arterial input function (AIF). It uses labelled parameter–curve pairs, starts with random weights and loads **no foundation checkpoint**. Use the same dataset to compare the three methods.

## Choose a posterior method

| Method | What it learns | Default sampling steps |
| :--- | :--- | :--- |
| **CM** (`cm`, default) | The original consistency model predicts clean parameters consistently at adjacent levels on a discrete noise grid. | 3 |
| **Improved MeanFlow** (`imf`) | Predicts the average velocity needed to move between two noise levels. Training matches instantaneous velocity using a directional derivative of the average-velocity network. | 1 |
| **Continuous-Time Consistency Model** (`scm`) | Predicts a clean endpoint that stays consistent along a continuous noise path. Training uses TrigFlow and a normalized directional derivative. | 2 |

The iMF option adapts the no-guidance training objective from [Geng et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Geng_Improved_Mean_Flows_On_the_Challenges_of_Fastforward_Generative_Models_CVPR_2026_paper.html). The sCM option uses the teacher-free **sCT** objective from [Lu and Song, ICLR 2025](https://arxiv.org/abs/2410.11081), with random initialization in this implementation. Both are conditioned on the observed TAC and AIF. These are PET implementations of the training objectives, not ports of the published image networks or weights.

Module 1 runs share the dataset contract, training-only normalization and checkpoint workflow. **LoRACM** in Module 2 still fine-tunes the original pretrained CM via `train.py`; the new methods do not change the foundation, LoRA or model-selection workflows.

## GUI workflow

1. Open **CM Posterior Estimation** from the welcome page and save the kinetic model, AIFs and timing in **1 · Model & protocol**.
2. Choose **2 · Dataset**. Generate a dataset, or import and validate a labelled NPZ. Starting sizes are **1,000,000 training / 100,000 validation / 1,000 test pairs**; the quick check uses **32 / 16 / 8**. These are editable starting values, not evidence of sufficient training data.
3. Open **3 · Train a model**. Choose **CM**, **iMF** or **sCM** under **Posterior method**. The selected dataset and its dimensions appear automatically.
4. Choose the run name, epochs, learning rate, batch size and device. **Check compute setup** checks the device selected for that method.
5. Open **Advanced settings** for architecture, optimizer, scheduler, loss weights, seed or checkpoint controls. The discrete noise grid is specific to CM; sCM instead offers a tangent warmup.
6. Choose **Review configuration**, then **Start training**.
7. Open **4 · Predict**. Choose the trained run and checkpoint. The saved method selects the sampler and sets the starting step count. Use a testing file or **My own patient scan** with its corresponding mask.
8. Use **5 · Single TAC & NLS** in the CM module to inspect its posteriors against NLS, then step 6 for parametric images. For a paired LoRACM/scratch comparison on shared held-out curves, use the terminal `compare_models.py` workflow below.

Both routes support graceful early stopping and the [same best/last/epoch checkpoint selection](FINE_TUNING.md#interrupted-jobs-and-memory-limits).

Module 1 uses `cm_dataset.npz`, `cm_protocol.json`, `cm_dataset_selection.json` and `train_cm.json` independently of LoRACM. Use distinct run names such as `cm_scratch`, `imf_scratch`, and `scm_scratch` to preserve each model and its metrics. For a comparison at equal training-data size, use the same labelled NPZ and keep a common evaluation file out of training and model selection.

## Defaults and settings

This table lists general training defaults. The UC Davis GUI preset follows Scenario 1 training settings for matched comparisons; its small walkthrough uses two epochs and batch size 8. Dimensions always follow the selected dataset. An imported labelled dataset can be used without a simulation protocol; a matched protocol is optional for later NLS.

| JSON setting | Default | What it controls |
| :--- | :--- | :--- |
| `posterior_method` | **`"cm"`** | Choose `"cm"`, `"imf"` or `"scm"`; older configurations without this setting use CM |
| `epochs` | **5,000** | Overall passes through the training loader |
| `lr` | **0.0001** | Initial AdamW learning rate |
| `batch_size` | **512** | Pairs per training batch; GUI caps the starting value to the dataset size |
| `x_dim` | **`"auto"`** | Number of predicted parameters P, excluding the final noise column |
| `y_dim` | **`"auto"`** | Conditioning width 2T: TAC values, then input-function values |
| `parameter_names`, `loss_weights` | **`"auto"`** | Names from the data, equal parameter loss weights |
| `embed_dim` | **1,024** | Width of time and conditioning embeddings; must be even |
| `channels` | **`[32,64,128,256]`** | Four U-Net widths, each divisible by 32 |
| `embedy` | **`"Linear"`** | Conditioning encoder; MLP and Conv1D also work for all methods, GRU and LSTM for CM only |
| `sigma_data` | **1.0** | CM boundary scale or sCM TrigFlow data scale in normalized parameter space |
| `noise_levels` | **15** | CM only: number of uniform consistency-noise intervals between 0 and 1 |
| `scm_warmup_steps` | **10,000** | sCM only: training batches over which the tangent term ramps up; set 0 to disable |
| `ema_decay` | **0.99** | Exponential moving average used for the target/inference model |
| `weight_decay` | **0.01** | AdamW regularization |
| `lr_scheduler_patience`, `lr_factor` | **25 / 0.75** | Reduce learning rate after a validation-loss plateau |
| `min_lr` | **0.00000001** | Scheduler floor; must not exceed the initial rate |
| `seed` | **0** | Random initialization and training streams |
| `checkpoint_interval` | **25** | Save selectable epoch snapshots every 25 epochs, plus epoch 1; best and last update every completed epoch |
| `drop_last` | **`true`** | Drop an incomplete final training batch; set false to include it |
| `device` | **`"auto"`** | CM training: CUDA, then MPS, then CPU. iMF and sCM training: CUDA, then CPU. |

Use these as starting settings, then assess convergence and held-out posterior quality. Training from scratch can need more data or compute than adaptation. Keep architecture, splits and evaluation settings aligned when studying the effect of initialization. A tiny installation run is not a useful trained research model.

iMF and sCM training need PyTorch directional derivatives and use CPU or CUDA. On Apple Silicon without CUDA, choose **Auto** or **CPU** for training. Inference does not compute these derivatives and can use Apple MPS if the usual compute checks pass.

Explicit numeric `x_dim` or `y_dim` values are accepted in JSON/CLI, but must match the dataset; conflicts are rejected before a run folder is created. The GUI displays these dimensions rather than inviting a change that would contradict the input file.

## Terminal training

Run from `PET-TRACER/LoRACM`. Edit `configs/train_cm.json` to point at your dataset, optional matched forward-model protocol, output folder and run name. Set `posterior_method` in JSON or override it on the command line:

```bash
python train_cm.py --config configs/train_cm.json --posterior-method imf
```

Prepare the UC Davis example with independent datasets and both training routes:

```bash
python ucdavis.py --workspace workspaces/cm_comparison --size small --with-cm
python simulate_protocol.py --config workspaces/cm_comparison/protocol.json
python simulate_protocol.py --config workspaces/cm_comparison/cm_protocol.json
python train.py --config workspaces/cm_comparison/train.json
python train_cm.py --config workspaces/cm_comparison/train_cm.json
```

Omit `--size small` for full starting sizes. GUI exports use the same config files and commands for a custom project.

You can override critical settings without editing JSON:

```bash
python train_cm.py --config configs/train_cm.json --epochs 5000 --lr 0.0001 --batch-size 512 --embed-dim 1024 --seed 0 --device cpu
```

The shared training overrides work with `train.py`, but method selection is for `train_cm.py` only. `--x-dim` and `--y-dim` are optional exact schema checks; leave the JSON values as `"auto"` for normal use. A scratch config must not contain a nonempty `pretrained` path.

## Outputs and inference

A run saves `best.pth`, `last.pth`, `model_params.json`, `scaling_params.json`, `metrics.csv`, TensorBoard events, and optional copied kinetic-model source/protocol. `training_summary.json` records the posterior method, target-dataset checksum, trainable/total parameter counts, seed, training time and lowest completed validation loss. The transfer report records random initialization.

Training and validation losses are specific to the method. Use them to monitor a run; compare methods on held-out parameter accuracy and posterior calibration. Checkpoints contain complete EMA **inference models**, not optimizer/RNG snapshots for exact resumption.

```bash
python infer.py --run workspaces/my_project/runs/cm_scratch --input workspaces/my_project/testing.npz --output workspaces/my_project/cm_predictions.npz --samples 1000 --device cpu
```

The same single-TAC posterior/NLS commands in [the prediction guide](PREDICTION.md) accept any Module 1 run. Custom kinetic parameter sets and 2TC-specific Kᵢ rules work across all methods. The sampler reads the saved run's method. Omit `--steps` for the method default or supply it to test a different step count.

## Compare trained models

Prepare one held-out prediction NPZ/HDF5 and use it for both models. An optional `truth [N,P]` array supplies known parameter labels. Use independent subjects/acquisitions where appropriate, and keep test pairs out of training and model selection.

```bash
python compare_models.py --lora-run workspaces/my_project/runs/adaptation --cm-run workspaces/my_project/runs/cm_scratch --input workspaces/my_project/testing.npz --output workspaces/my_project/comparisons/first --samples 1000 --device cpu
```

The comparison saves `loracm.npz`, `cm.npz`, `metrics.csv` and `comparison.json`. With truth, metrics include parameter **RMSE and bias of posterior means**. Without truth, it reports uncertainty summaries and negative-draw fractions, which do not establish accuracy. The report records dataset-file agreement, training-setting differences and timings. Inference timing includes model loading and file operations; use a dedicated benchmark for throughput claims.

The `--cm-run` argument accepts a CM, iMF or sCM run from Module 1. For a small curve set, add `--save-samples`. To export a single-TAC overlay in a terminal:

```bash
python plot_posteriors.py --posterior workspaces/my_project/comparisons/first/loracm.npz --comparison workspaces/my_project/comparisons/first/cm.npz --output workspaces/my_project/comparisons/first/posteriors.png
```

The plot command requires exactly one TAC. Saved joint draws are capped at two million parameter values per model. Match sampling steps, sample count and inference seed; repeat training/evaluation across seeds for robust conclusions. Inspect fit quality, tail behaviour and calibration separately. LoRACM's foundation pretraining is an additional resource, even when both methods use the same target dataset and fine-tuning/training epochs.
