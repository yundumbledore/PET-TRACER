# LoRACM user manual

**Adapt PET-TRACER to your own tracer and protocol, then estimate kinetic parameter posteriors.**

[Project overview](../README.md) · [Install](docs/INSTALLATION.md) · [UC Davis steps 1–6](docs/UCDAVIS.md) · [Parametric imaging](docs/PARAMETRIC_IMAGING.md) · [Legacy framework](../docs/legacy-framework.md)

## Choose a route

| Your starting point | What to do |
| :--- | :--- |
| You need labelled fine-tuning data | Define your kinetic model, input functions and acquisition protocol; simulate parameter–curve pairs. |
| You already have labelled fine-tuning data | Convert it to the NPZ format, validate it, and skip simulation. |
| You have only measured TACs and input functions | These are prediction inputs. Obtain or simulate labelled pairs to adapt the model first. |
| You want a scratch-CM baseline | Prepare a separately sized dataset and train [CM from scratch](docs/TRAIN_CM.md), then evaluate both models on shared held-out curves. |
| You have a new patient scan | Reuse a compatible trained model with the coronal TAC/AIF slice and its corresponding mask; [prepare and image your patient](docs/PARAMETRIC_IMAGING.md#your-own-patient-scan-and-mask). |
| You want to learn with supplied assets | Follow the [UC Davis demonstration](docs/UCDAVIS.md), starting with the small CPU check. |

**Supplied paper model:** [checkpoint](paper/ucdavis/best.pth), [normalization](paper/ucdavis/scaling_params.json), [configuration and provenance](paper/ucdavis/README.md). Keep these files together; GUI model selectors prepare the complete run automatically.

## 1. Install and start

Follow the [Windows/Apple Silicon macOS/Linux installation guide](docs/INSTALLATION.md). Install once with root `setup_environment.py`; the shared environment lives outside the checkout and is reused by new downloads. Start the GUI from the repository root:

```bash
python launch_gui.py
```

The browser interface runs on your computer. **Project name** selects a folder under `LoRACM/workspaces/`; configs, uploaded files, logs, runs and analyses stay there. Use a new project/run name for an independent experiment. A single active background job is allowed per browser session. Its status and log remain available while you switch pages; keep the application running while jobs execute.

**UC Davis Scenario 1 is the default example setup.** For paper-model imaging, open **UC Davis demonstration → Reproduce paper Ki map**, or choose **Use supplied paper checkpoint** in step 3. Retraining is a separate experiment with matching equations/settings and device-dependent results. Follow numbered steps 1–6 with its 2TC model, four original AIFs from `UCDavis_1H35.csv` and Sub001's 35-frame schedule. Choose **Quick installation check** before saving step 1 for a quick check, or retain full research-sized settings. **My own project** supplies the general defaults below. Settings and unfinished edits are retained across pages and browser sessions. Use a fresh project name for independent settings. [Read the GUI guide](docs/WORKBENCH.md) for project defaults, early stopping and inference controls.

Terminal users run the following workflow commands from **`PET-TRACER/LoRACM`**. Relative config paths are interpreted from that directory. Absolute paths also work; quote paths containing spaces.

## 2. Define the model and protocol

GUI: **1 · Model & protocol**. Select built-in 2TC or a custom ODE, edit priors, load/edit the frame schedule, choose half-life and observation mode, and supply measured or fitted Feng input functions. Save the validated protocol.

Read [models, AIFs and protocols](docs/MODELS_AND_PROTOCOLS.md) before changing tracers. `configs/dataset_custom.json` is an editable starting point with a **five-minute illustrative schedule**, not a recommended clinical protocol. Replace its timings, durations, units assumptions, priors and input functions for your study.

## 3. Simulate or import a dataset

GUI: **2 · Dataset**. Choose **Simulate my protocol** or **Use my own NPZ dataset**.

My own project defaults: **100,000 training pairs**, plus **10,000 validation** and **1,000 test** pairs. UC Davis defaults to **100,000 / 100,000 / 100**, or **32 / 16 / 8** in the small walkthrough. Simulation uses rank-independent settings; rank and alpha are selected before training.

Always choose **Validate and select dataset** before moving on. Validation checks shapes, time grids, parameter names/order, finite values, nonnegative noise scales and nonzero training normalization scales. It cannot establish whether units, labels, priors, cohort splits or the forward model are scientifically correct. Follow [dataset preparation](docs/PREPARE_DATA.md).

## 4. Fine-tune

GUI: **3 · Fine-tune**. Select the dataset and foundation checkpoint, confirm the kinetic model, choose your run name and start training. My own project defaults are **rank 8**, **alpha 16**, dropout 0.05, batch size 512, learning rate 0.0001 and 5,000 epochs. UC Davis uses **rank 4 / alpha 8**; its small walkthrough uses two CPU epochs and batch size 8. Reduce batch size for limited memory.

Dimensions and parameter names are derived from the dataset. **Advanced settings** includes embedding width, channels, optimizer, consistency intervals, EMA, scheduler, seed and loss weights; defaults are documented in [the shared settings table](docs/TRAIN_CM.md#defaults-and-settings). Training loss and validation self-consistency loss refresh in the job monitor. See [fine-tuning settings and transfer limits](docs/FINE_TUNING.md). Preserve the checkpoint, model configuration and scaling together.

### Train a scratch CM for comparison

Enable the optional CM route, then use **CM · Dataset → CM from scratch → CM · Predict**. It prepares independently sized data and trains a randomly initialized model without foundation weights. The full starting dataset is 1,000,000 / 100,000 / 1,000 pairs; edit these values for your study. Training controls and output formats are shared with LoRACM. **Compare models** evaluates both runs on one held-out testing file. Follow the [CM training and comparison manual](docs/TRAIN_CM.md).

## 5. Predict on your project

GUI: **4 · Predict** for LoRACM, **CM · Predict** for scratch CM, or **5 · Single TAC & NLS** for a CSV/selected curve from either model. Both batch pages support another testing file. **My own patient scan** requires the coronal TAC/AIF slice and matching mask. Inputs must match the model's schedule, units and observation convention. Repeated predictions preserve previous files with numbered names and an explicit history selector.

The single-TAC page displays parameter posterior histograms, exports joint samples and figures, and optionally overlays dashed nonlinear least-squares estimates. Read the [prediction manual](docs/PREDICTION.md) for an exact file example, terminal commands and interpretation.

## 6. Make parametric images and inspect voxels

GUI: **6 · Parametric imaging**. Kᵢ posterior mean is selected by default for the supplied UC Davis coronal slice. Choose another predicted parameter or its **95% credible-interval width**. The image appears on the left; clicking a foreground voxel displays its TAC plus input function and posterior KDE on the right. Export figures, map values and selected joint samples. Samples for inspected voxels are generated on demand and cached.

Predict the complete slice from either model's page. Your own patient route prepares and carries its mask/geometry into step 6; the same Ki, parameter, uncertainty and voxel-inspection controls work on your data. Simulated test pairs have no anatomical geometry. Read [parametric imaging](docs/PARAMETRIC_IMAGING.md) for formats and terminal equivalents.

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

# Optional baseline: configure its separate dataset (or explicitly reuse data).
python train_cm.py --config configs/train_cm.json

# Predict your own TACs. Supply a testing file in the documented format.
python infer.py --run workspaces/my_project/runs/adaptation --input workspaces/my_project/testing.npz --output workspaces/my_project/predictions.npz --samples 1000
```

If using your own fine-tuning file, change `dataset_dir` in the training config to its path; no simulation is required. See [preparation](docs/PREPARE_DATA.md) for conversion examples and [single-TAC terminal analysis](docs/PREDICTION.md#terminal-single-tac-analysis) for posterior samples plus NLS figures. Outputs are protected against accidental overwriting; choose a new name or deliberately move earlier results aside.

## UC Davis example and original material

The [UC Davis guide](docs/UCDAVIS.md) covers the complete numbered GUI/terminal route, supplied assets, small/full runs, posterior inference and interactive imaging. The default UC Davis GUI uses the original HPC Euler solver, noise timing and unchanged AIF conditioning, narrower paper priors, scheduler patience 50, and the original 2×2-filtered Ki display. Use **Use supplied paper checkpoint** in step 3 or the **UC Davis demonstration** page to reproduce the paper model without retraining. [Exact settings and provenance](docs/UCDAVIS.md).

[Original framework](../docs/legacy-framework.md) · [Detailed formats](docs/DATA_FORMATS.md) · [Provenance](docs/DATA_PROVENANCE.md) · [Scientific review](docs/RELEASE_REVIEW.md) · [Validation](docs/VALIDATION.md)
