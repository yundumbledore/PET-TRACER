# UC Davis · Reproduce the paper example

[User manual](../README.md) · [Installation](INSTALLATION.md) · [Parametric imaging](PARAMETRIC_IMAGING.md)

The supplied paper-checkpoint route follows the original **HPC inference conventions**. New training datasets use [log-normal measurement noise and actual frame durations](MEASUREMENT_NOISE.md); they are a new experiment rather than an exact regeneration of the historical Gaussian training data. It includes the epoch-2755 paper LoRACM checkpoint, its original normalization, the four measured AIFs, the Sub001 slice and its verified mask/mapping. No retraining is needed to use the paper model.

## Reproduce the Ki image in the GUI

1. Start `python launch_gui.py` from the repository root and choose a **new project name**.
2. Retain **UC Davis Scenario 1**. Open **UC Davis demonstration**, then **Reproduce paper Ki map**. Auto selects CUDA, Apple MPS, then CPU.
3. Follow the dynamic job progress. The command samples 1,000 joint draws per voxel with three reverse steps and writes a Ki figure and numeric maps in your project folder.
4. Open **6 · Parametric imaging** for the completed result. Choose Ki, another kinetic parameter, or a 95% credible-interval width. Click a voxel to inspect its TAC, input function and posterior.

Alternatively, choose **Use supplied paper checkpoint** in **3 · Fine-tune**, then run **4 · Predict → UC Davis patient slice** and open step 6. This uses the original model without creating a new training dataset. Predictions and selected-voxel samples have Monte Carlo variation across seeds, batch/chunk settings, PyTorch versions and devices; they are not promised to be pixel-identical to the archived image.

The expandable **Archived HPC reference map** shows the original saved Ki summaries rendered with the original display convention. Its arrays are in `paper/ucdavis/reference_ki.npz`. The checkpoint/normalization and source-asset hashes are recorded in `paper/ucdavis/manifest.json`. The reference archive has no posterior draws or exact Ki-transform provenance; the release explicitly uses the flooring convention present in the original HPC selected-voxel inference code.

## Learn simulation and fine-tuning in steps 1–6

| Step | Default GUI action | Settings |
| :--- | :--- | :--- |
| **1 · Model & protocol** | Review the supplied FDG 2TC recipe | Original `UCDavis_1H35.csv`, five parameters, half-life 109.8 min |
| **2 · Dataset** | Generate and validate labelled pairs | 100,000 train / 100,000 validation / 100 test; seed 42; log-normal noise |
| **3 · Fine-tune** | Select **Use supplied paper checkpoint**, or train a new adaptation | Rank 4, alpha 8, dropout 0.05; batch 512; 5,000 epochs; LR 0.0001; scheduler patience **50**, factor 0.75 |
| **4 · Predict** | Use the supplied Sub001 patient slice | 1,000 draws, three reverse steps at 1, 2/3, 1/3; all 24,237 foreground TACs |
| **5 · Single TAC & NLS** | Select a voxel, sample its posterior, optionally fit NLS | Same Euler forward model; posterior-median forward TAC and NLS comparison |
| **6 · Parametric imaging** | Choose parameter/statistic and inspect voxels | Ki mean initially; **2×2 median filter**, fixed **0–0.06** Ki estimate scale |

Full-run prior ranges:

| Parameter | Uniform range |
| :--- | :--- |
| K1 | 0.0001–2.0 |
| k2 | 0.0001–2.0 |
| k3 | 0.0001–0.5 |
| k4 | 0.0001–0.2 |
| Vb | 0.001–1 |
| Noise amplitude | 3–7 |

The simulation ranges follow `REGION_PRIORS['universal']` in the supplied HPC `Create_dataset.py`. The author confirmed that fine-tuning used 100,000 examples. The supplied epoch-2755 checkpoint retains its original normalization; its historical saved scales do not set the defaults for new simulations. The older bundled `data/simulated/scenario1.npz` remains available for earlier workflows and is not selected by the paper-checkpoint route.

New projects use these default ranges. Existing projects retain their saved settings.

**Quick installation check** uses 32 / 16 / 8 pairs, two CPU epochs and a small posterior sample count. It checks software operation and cannot reproduce a useful research map. New datasets use the original JAX split keys, parameter/noise array shapes and NumPy AIF-selection sequence before numerical chunking. Reproducing the draws requires the same split sizes, seed, JAX version and PRNG settings. CUDA/MPS training can still follow different numerical trajectories. Use the supplied checkpoint for paper-model prediction.

## Timing, forward model and noise

The original generator reads the CSV's first column as `t_meas`, linearly interpolates inputs **inside the Euler solver**, and saves the original AIF activity values unchanged in each conditioning vector. The first Euler update has `dt=t_meas[0]`; subsequent updates use the dense-grid differences with nominal step 0.01 min. Observations use the original `searchsorted` grid indexing. There is no added zero-at-origin sample in this recipe.

The CSV and HDF5 carry matching activity samples in frame order but slightly different recorded time labels: their second labels are 0.110216667 and 0.125 minutes. The protocol explicitly stores **CSV simulation times** in `t_meas` and **patient frame midpoints** in `prediction_t_meas`, with the measured frame durations separately. The trained model receives TAC values followed by AIF values, without timestamp inputs. The explicit HPC recipe accepts those two recorded label conventions without interpolating any patient activity. Other schedules and durations are rejected for the supplied model. Custom projects use their configured midpoints consistently for simulation and testing.

New simulations use a mean-preserving log-normal transform, with variance scaled by **actual acquisition frame durations**. Noise amplitude defaults to 3–7, half-life to 109.8 min and the mean floor to 1e-8. [Equations, assumptions and configuration](MEASUREMENT_NOISE.md). The supplied historical checkpoint and datasets used Gaussian noise with `delta_t=[t_meas[0], diff(t_meas)]` and remain unchanged. Training and validation share the four-AIF pool; test kinetic priors use the central 50% of each range. These are not subject-held-out splits.

For Ki, the paper recipe floors K1, k2 and k3 at **1e-4 for each joint posterior draw**, computes `Ki=K1*k3/(k2+k3)`, then calculates means, medians and empirical intervals. The transform is recorded in prediction metadata as `ki_policy=hpc_floor_1e-4`. General custom projects retain the separately documented raw transform. Parameter posterior samples themselves are preserved unchanged.

The 2×2 spatial filter affects the **display image**. Numeric exports preserve the raw map under `image` and include the displayed plane under `display_image`. Voxel selection always identifies the original measured TAC and posterior, not a spatially filtered posterior. Select **Raw values · no spatial filtering** to inspect unsmoothed estimates. Other parameter/uncertainty maps use percentile colour limits; the fixed 0–0.06 scale applies to Ki mean/median maps.

## Terminal equivalents

Run from `PET-TRACER/LoRACM` after installation:

```bash
# Original checkpoint → posterior prediction → paper-style Ki map.
python reproduce_ucdavis.py --workspace workspaces/ucdavis_paper --device auto

# Or prepare the original model, then use the individual modules.
python reproduce_ucdavis.py --workspace workspaces/ucdavis_model --prepare-only
python infer.py --run workspaces/ucdavis_model/runs/paper_ucdavis --input data/real/Sub001_slice75_adjusted.h5 --output workspaces/ucdavis_model/predictions.npz --samples 1000 --steps 3 --device auto
python parametric_imaging.py --predictions workspaces/ucdavis_model/predictions.npz --parameter Ki --display ucdavis_hpc --output workspaces/ucdavis_model/Ki.png
```

For the editable six-step retraining route:

```bash
python ucdavis.py --workspace workspaces/ucdavis_retrain
python simulate_protocol.py --config workspaces/ucdavis_retrain/protocol.json
python protocol.py workspaces/ucdavis_retrain/dataset.npz
python train.py --config workspaces/ucdavis_retrain/train.json
python infer.py --run workspaces/ucdavis_retrain/runs/scenario1 --input data/real/Sub001_slice75_adjusted.h5 --output workspaces/ucdavis_retrain/predictions.npz --samples 1000 --steps 3
python inspect_voxel.py --run workspaces/ucdavis_retrain/runs/scenario1 --input data/real/Sub001_slice75_adjusted.h5 --voxel-id 33449 --output workspaces/ucdavis_retrain/voxel.npz --samples 1000
python parametric_imaging.py --predictions workspaces/ucdavis_retrain/predictions.npz --parameter Ki --output workspaces/ucdavis_retrain/Ki.png
```

Use `--size small` with `ucdavis.py` for an installation check; its run is named `scenario1_small`. Use `--with-cm` to prepare separate scratch-CM dataset/training configs. Existing GUI projects and datasets are not reset or migrated automatically: choose a new project for the updated defaults. The [single-TAC guide](PREDICTION.md) explains NLS and posterior-median TAC overlays; the [imaging guide](PARAMETRIC_IMAGING.md) covers your own patients and masks.
