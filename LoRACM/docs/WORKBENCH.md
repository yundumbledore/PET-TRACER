# PET-TRACER local workbench

[Installation and upgrades](INSTALLATION.md) · [Project overview](../../README.md)

## Welcome page

The GUI opens with toolbox graphics, team information and three independent modules:

1. **CM Posterior Estimation** trains a consistency model from random initialisation.
2. **LoRACM Adaptation** fine-tunes a pretrained CM using LoRA.
3. **Bayesian Model Selection** runs the pretrained P001 FDG demonstration and its linked model/parameter posterior viewer. Model-selection training is reserved for a future release.

Choose a module to see its workflow steps in the left menu. **All modules** returns to the welcome page. The old scratch-comparison checkbox is replaced by a separate CM module; terminal comparison commands remain available.

## CM Posterior Estimation

1. Define a kinetic model, input functions and acquisition timing.
2. Simulate labelled pairs or upload your dataset.
3. Configure the U-Net and training, then train from scratch.
4. Predict on your own data.
5. Inspect a single TAC, parameter posteriors and an NLS reference.
6. Generate parametric images and click voxels to inspect their TACs and posteriors.

[Scratch training guide](TRAIN_CM.md) · [Models and protocols](MODELS_AND_PROTOCOLS.md)

## LoRACM Adaptation

The separate left menu follows the same preparation and prediction stages, with **Fine-tune** in step 3. It exposes LoRA settings and the foundation checkpoint. The UC Davis paper demonstration remains available without retraining.

[Fine-tuning](FINE_TUNING.md) · [UC Davis demonstration](UCDAVIS.md)

## Defaults and custom studies

Both training modules keep **UC Davis Scenario 1** as their default example, with the existing input curves, kinetics, frame schedule and training settings. **Quick installation check** reduces the dataset/training workload for a short interface check. **My own project** starts with the existing generic defaults. These are starting settings, not fixed limits.

Use measured AIFs, a fitted Feng cohort, a custom kinetic-model plugin or your own labelled NPZ. Each route keeps its existing validation requirements. [Prepare data](PREPARE_DATA.md) · [Prediction](PREDICTION.md) · [Parametric imaging](PARAMETRIC_IMAGING.md)

## Bayesian Model Selection

Its left menu contains **P001 demonstration**, **Inference settings**, **Run inference** and **Explore results**. Configure posterior draws per voxel, reverse sampling steps, voxel batch size, posterior sample chunk, compute device and seed before sampling. Then click a voxel and a model bar to inspect conditional kinetic posteriors and a forward simulation using that model's median delay. [Full model-selection guide](MODEL_SELECTION.md).

Only the supplied P001 coronal slice 116 is enabled in this release. The training code and general dataset/model-selection workflow are not included.

## Projects and job monitoring

Project names use letters, numbers, underscores and hyphens. CM, LoRACM and model-selection project files are separate. In-progress edits survive page changes and refresh. Use a new project name to initialise fresh defaults.

New installations use `~/.pet-tracer/workspaces`, or the folder selected by `PET_TRACER_WORKSPACES`. Existing checkout-local projects continue to work. Follow the installation guide before removing an older download; changed custom models are never silently replaced when reconnecting bundled assets.

Computations run locally as background jobs with saved logs. Return to the originating module/project to monitor progress. A running job blocks starting another job in the same browser session. Stop requests are cooperative. Incomplete model-selection runs are not published as completed results; training retains its completed checkpoints. Restarting the app can interrupt jobs, so retain project configurations and outputs.

The browser does not require Jupyter. CPU, supported CUDA installations and Apple MPS are available through the existing PyTorch environment. [Hardware and platform guide](INSTALLATION.md#hardware-and-platform-support).
