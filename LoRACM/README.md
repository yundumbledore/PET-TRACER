# LoRACM user manual

**Adapt PET-TRACER to your own tracer and protocol, then estimate kinetic parameter posteriors.**

[Project overview](../README.md) · [Install](docs/INSTALLATION.md) · [UC Davis steps 1–6](docs/UCDAVIS.md) · [Parametric imaging](docs/PARAMETRIC_IMAGING.md) · [Legacy framework](../docs/legacy-framework.md)

## Choose a route

| Your starting point | What to do |
| :--- | :--- |
| You need labelled fine-tuning data | Define your kinetic model, input functions and acquisition protocol; simulate parameter–curve pairs. |
| You already have labelled fine-tuning data | Convert it to the NPZ format, validate it, and skip simulation. |
| You have only measured TACs and input functions | These are prediction inputs. Obtain or simulate labelled pairs to adapt the model first. |
| You want a scratch-CM baseline | Train the same labelled data with [CM from scratch](docs/TRAIN_CM.md), then evaluate both models on shared held-out curves. |
| You want to learn with supplied assets | Follow the [UC Davis demonstration](docs/UCDAVIS.md), starting with the small CPU check. |

## 1. Install and start

Follow the [Windows/Apple Silicon macOS/Linux installation guide](docs/INSTALLATION.md). Start the GUI from the repository root:

```bash
python launch_gui.py
```

The browser interface runs on your computer. **Project name** selects a folder under `LoRACM/workspaces/`; configs, uploaded files, logs, runs and analyses stay there. Use a new project/run name for an independent experiment. A single active background job is allowed per browser session. Its status and log remain available while you switch pages; keep the application running while jobs execute.

**UC Davis Scenario 1 is the default example setup.** Follow numbered steps 1–6 with its 2TC model, four AIFs and patient-aligned 35-frame schedule. Choose **Quick installation check** before saving step 1 for a quick check, or retain full research-sized settings. **My own project** supplies the general defaults below. Settings and unfinished edits are retained across pages and browser sessions. Use a fresh project name for independent settings. [Read the GUI guide](docs/WORKBENCH.md) for project defaults, early stopping and inference controls.

Terminal users run the following workflow commands from **`PET-TRACER/LoRACM`**. Relative config paths are interpreted from that directory. Absolute paths also work; quote paths containing spaces.

## 2. Define the model and protocol

GUI: **1 · Model & protocol**. Select built-in 2TC or a custom ODE, edit priors, load/edit the frame schedule, choose half-life and observation mode, and supply measured or fitted Feng input functions. Save the validated protocol.

Read [models, AIFs and protocols](docs/MODELS_AND_PROTOCOLS.md) before changing tracers. `configs/dataset_custom.json` is an editable starting point with a **five-minute illustrative schedule**, not a recommended clinical protocol. Replace its timings, durations, units assumptions, priors and input functions for your study.

## 3. Simulate or import a dataset

GUI: **2 · Dataset**. Choose **Simulate my protocol** or **Use my own NPZ dataset**.

My own project defaults: **100,000 training pairs**, plus **10,000 validation** and **1,000 test** pairs. UC Davis defaults to **50,000 / 100,000 / 100**, or **32 / 16 / 8** in the small walkthrough. Simulation uses rank-independent settings; rank and alpha are selected before training.

Always choose **Validate and select dataset** before moving on. Validation checks shapes, time grids, parameter names/order, finite values, nonnegative noise scales and nonzero training normalization scales. It cannot establish whether units, labels, priors, cohort splits or the forward model are scientifically correct. Follow [dataset preparation](docs/PREPARE_DATA.md).

## 4. Fine-tune

GUI: **3 · Fine-tune**. Select the dataset and foundation checkpoint, confirm the kinetic model, choose your run name and start training. My own project defaults are **rank 8**, **alpha 16**, dropout 0.05, batch size 512, learning rate 0.0001 and 5,000 epochs. UC Davis uses **rank 4 / alpha 8**; its small walkthrough uses two CPU epochs and batch size 8. Reduce batch size for limited memory.

Dimensions and parameter names are derived from the dataset. **Advanced settings** includes embedding width, channels, optimizer, consistency intervals, EMA, scheduler, seed and loss weights; defaults are documented in [the shared settings table](docs/TRAIN_CM.md#defaults-and-settings). Training loss and validation self-consistency loss refresh in the job monitor. See [fine-tuning settings and transfer limits](docs/FINE_TUNING.md). Preserve the checkpoint, model configuration and scaling together.

### Train a scratch CM for comparison

Choose **CM from scratch** to train a new randomly initialized model without foundation weights. It shares the configuration controls and output format with LoRACM. **Compare models** evaluates both runs on the same testing file and can overlay their posterior samples. Follow the [CM training and comparison manual](docs/TRAIN_CM.md).

## 5. Predict on your project

GUI: **4 · Predict** for batch NPZ/HDF5 data, or **5 · Single TAC & NLS** for a CSV or one selected curve. Prediction accepts either trained model type; inputs must match the model's schedule, units and observation convention. Models fitted to another schedule need adaptation or justified resampling.

The single-TAC page displays parameter posterior histograms, exports joint samples and figures, and optionally overlays dashed nonlinear least-squares estimates. Read the [prediction manual](docs/PREDICTION.md) for an exact file example, terminal commands and interpretation.

## 6. Make parametric images and inspect voxels

GUI: **6 · Parametric imaging**. Kᵢ posterior mean is selected by default for the supplied UC Davis coronal slice. Choose another predicted parameter or its **95% credible-interval width**. The image appears on the left; clicking a foreground voxel displays its TAC plus input function and posterior KDE on the right. Export figures, map values and selected joint samples. Samples for inspected voxels are generated on demand and cached.

Run step 4 on the complete patient-slice file to provide anatomical voxel predictions. Simulated test pairs have no image geometry. The module checks explicit voxel identities and verified geometry, and supports custom 2D slices or 3D masks. Read [parametric imaging](docs/PARAMETRIC_IMAGING.md) for exact formats and terminal equivalents.

## Terminal quick start

After installing and activating the environment, change to `LoRACM/`. Edit the custom configs first. The GUI exports compatible configs under `workspaces/<project>/`, so you can move a GUI project to a terminal/server.

```bash
# Optional: fit a cohort of measured AIFs, then configure aif_mode=feng_cohort.
python fit_feng.py --input data/aif/UCDavis_1H35.csv --output workspaces/my_project/feng.csv

# Simulate your explicitly configured protocol, or skip this for your own NPZ.
python simulate_protocol.py --config configs/dataset_custom.json

# Check the labelled dataset before spending time training.
python protocol.py workspaces/my_project/dataset.npz

# Set data paths; dimensions, parameter names and equal weights resolve automatically.
python train.py --config configs/train_custom.json

# Optional baseline on the same labelled data, initialized randomly.
python train_cm.py --config configs/train_cm.json

# Predict your own TACs. Supply a testing file in the documented format.
python infer.py --run workspaces/my_project/runs/adaptation --input workspaces/my_project/testing.npz --output workspaces/my_project/predictions.npz --samples 1000
```

If using your own fine-tuning file, change `dataset_dir` in the training config to its path; no simulation is required. See [preparation](docs/PREPARE_DATA.md) for conversion examples and [single-TAC terminal analysis](docs/PREDICTION.md#terminal-single-tac-analysis) for posterior samples plus NLS figures. Outputs are protected against accidental overwriting; choose a new name or deliberately move earlier results aside.

## UC Davis example and original material

The [UC Davis guide](docs/UCDAVIS.md) covers the complete numbered GUI/terminal route, supplied assets, small/full runs, posterior inference and interactive imaging. The portable walkthrough aligns AIFs to the measured patient schedule and uses explicit frame durations with RK4. The separate shortcut page and optional JAX generator preserve the earlier AIF-grid simulation conventions.

[Original framework](../docs/legacy-framework.md) · [Detailed formats](docs/DATA_FORMATS.md) · [Provenance](docs/DATA_PROVENANCE.md) · [Scientific review](docs/RELEASE_REVIEW.md) · [Validation](docs/VALIDATION.md)
