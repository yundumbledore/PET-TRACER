# Workflow validation

## Log-normal measurement noise — 27 October 2026

**100 checks passed, with one Windows-only check skipped** (101 total) in Python 3.11 / PyTorch 2.6.0 CPU with JAX/jaxlib 0.10.2. New checks compare the log-normal transform against independently parameterized SciPy quantiles, verify conditional mean/variance with stratified normal draws, exercise zero/near-zero activity and numerical limits, and reject invalid timing/configuration. The legacy Gaussian implementation still agrees with the preserved JAX reference.

Small log-normal UC Davis datasets agree across the portable and research generators, retain unchanged AIF conditioning and remain invariant to numerical chunk size. The terminal wrapper honors explicit log-normal settings. Both GUI training modules retain noise selections across navigation, while unmarked existing projects keep their Gaussian settings. Dataset provenance survives validation and test-file export; conflicting training-protocol provenance is rejected.

Separate one-epoch CPU workflows completed dataset generation, LoRACM 2TC adaptation or custom-1TC scratch-CM training, posterior inference, NLS and posterior-figure export. Both runs retained distribution/timing provenance in the model configuration, scaling and training summary. These small workflows check interfaces, not posterior calibration or research-quality convergence. Source syntax, JSON and changed documentation links were checked; the root README, paper assets, model-selection module/assets, checkpoints and bundled datasets remain unchanged. Windows/Linux and MPS/CUDA execution were not tested in this update.

New simulations deliberately use log-normal measurement noise with actual frame durations; the Gaussian parity results recorded below describe the historical path. See [measurement-noise assumptions and references](MEASUREMENT_NOISE.md).

## Windows double-click installer — 16 October 2026

`Setup PET-TRACER.cmd` invokes a Windows PowerShell 5.1 helper that detects or prepares 64-bit Python 3.11, offers CPU/NVIDIA installation, calls the existing reusable-environment setup, imports the installed packages and runs the selected device's calculation/gradient check. Setup writes logs outside the checkout and leaves project datasets and model files alone. The launcher keeps the window open and preserves its exit status.

The existing **70 automated checks passed** locally on Apple Silicon CPU. A new Windows-only check is included in the existing Windows CI matrix; it was **skipped locally** because Windows PowerShell is unavailable. It parses the installer, exercises real native argument passing with spaces/Unicode, rejects failed/missing interpreters and child errors, and checks both setup choices, verification, bootstrap dispatch and cancellation with installation commands mocked. It does not download/install Python or packages. Documentation links and launcher line endings were checked locally. **Actual Windows execution, Python bootstrap downloads, fresh package installation and NVIDIA verification have not been run in this session.** Hosted CI has not yet run on these changes.

## Dataset and trainer parity — 15 October 2026

**70/70 automated checks passed** in the Apple Silicon Python 3.11 / PyTorch 2.6.0 CPU environment, with JAX/jaxlib 0.10.2, x64 enabled and `jax_threefry_partitionable=true`. Step 1 no longer exposes simulation-convention, observation-mode or endpoint controls. GUI tests check midpoint defaults, unchanged UC Davis AIF conditioning, edited-schedule handling and persistence. Custom 1TC and Feng-cohort Euler datasets also generate and validate.

A direct comparison with the author's `Create_dataset.py` used the same broad source priors, seed 42, four measured AIFs and 32 / 16 / 8 split sizes. Parameter/noise-amplitude arrays and AIF indices matched **exactly**; maximum absolute TAC/AIF difference was **2.91e-11**. The reference sampling, solver, model, noise and central-prior functions are syntactically identical to the preserved `simulation.py` functions. A separate regression compares against those preserved functions and verifies that changing numerical chunk size leaves every dataset array unchanged. Matching historical draws also requires matching the original JAX version/PRNG settings and all split sizes; those historical runtime settings are unavailable.

With the same dataset, seed, batch size 8, full 1024-dimensional embedding and [32, 64, 128, 256] channels, the current and original trainers completed two CPU epochs. All **82 saved model tensors matched exactly** (maximum absolute difference 0). The reference device and scheduler were explicitly set to CPU and the saved experiment's patience 50; automatic dimensions and equal loss weights were resolved before invoking the original trainer. This checks implementation parity under controlled conditions, not full-training convergence or equality across MPS/CUDA/PyTorch versions. Training summaries now record package versions and the batch-shuffle RNG policy.

No full 5,000-epoch retraining or new MPS/CUDA/Windows/Linux execution was performed. Existing user datasets, runs, checkpoint tensors and paper normalization were unchanged. The current source priors differ from the distribution indicated by the archived dataset and paper normalization; agreement with today's generator does not establish regeneration of that historical dataset.

## UC Davis source-prior correction — 15 October 2026

UC Davis simulation and forward-protocol bounds follow `REGION_PRIORS['universal']` in the supplied `Create_dataset.py`: K1/k2 0.0001–2.0, k3 0.0001–0.5, k4 0.0001–0.2 and Vb 0.001–1. The earlier narrower defaults inferred from the archive were corrected. Original checkpoint weights and normalization remain unchanged. A GUI regression check verifies the default ranges and persistence of user edits across pages. The current suite has **67 passing checks**; older dated results below are retained as history.

## HPC reproduction and reusable environment — 15 October 2026

**66/66 automated checks passed** on Apple Silicon macOS, Python 3.11, PyTorch 2.6.0 CPU. Tests include direct paper-checkpoint selection without training, patient/CSV time-label contracts, unchanged AIF conditioning, HPC solver/noise behavior, explicit Ki flooring, raw/display-map separation, and shared environment discovery for POSIX/Windows paths. Previous project-persistence, custom-model, CM comparison, voxel identity, logging and memory checks continue to pass.

The portable Euler solver was compared directly with the original `Create_dataset.py` JAX `solve_ode` for four parameter vectors and all four measured AIFs. Maximum absolute clean-TAC difference was **7.28e-12**; applying the same JAX-generated noise gave a maximum difference of **1.82e-12**. This verifies the equations for tested inputs, not historical random draws or full retraining equivalence.

The original checkpoint generated **1,000 samples for 16 real patient voxels**, with stored joint draws reproducing exported Ki summaries exactly under the recorded paper policy. A complete **24,237-voxel slice with eight draws per voxel** verified coverage, Ki/parameter/interval-width mapping and the 2×2/fixed-scale display. This low-draw whole-slice run checks operation, not the paper's 1,000-draw numerical result. A separate 32/16/8 dataset completed two full-width training epochs, predicted simulated and real-patient TACs with both explicit label conventions, and passed the posterior-median Euler forward overlay.

A real external virtual environment was created and reused in a writable test folder with package installation skipped; the second setup retained it. Launcher discovery and shell syntax were checked. Dependencies were already available in the verification environment; a fresh network installation and Windows/CUDA/MPS execution were not run here. The environment is outside the source checkout and launch does not run pip.

## Separate CM data, repeated prediction and patient masks — 14 October 2026

**57/57 automated checks passed** in the Python 3.11 / PyTorch 2.6.0 Apple Silicon CPU environment. The GUI now separates CM dataset preparation and prediction from LoRACM, keeps repeated prediction files and settings, and prepares verified geometry for a user-provided patient TAC slice plus mask. New checks cover independent dataset sizes/protocols/selections, model-type filtering, cleared pending prediction displays, original-file preservation, patient-specific Ki/parameter/interval-width maps, explicit voxel mappings, and rejection of another patient's geometry even with matching voxel IDs. GUI checks were repeated after the final model/geometry handoff changes. Source parsing, whitespace checks and 124 local documentation links passed.

An end-to-end run through the GUI's actual subprocess launcher exported both small UC Davis configurations, generated separate 32 / 16 / 8 datasets, trained each full-width model for one CPU epoch, and passed the compute diagnostic. A synthetic new patient with three 35-frame TACs and its own binary mask was prepared and predicted twice by each model. Original prediction files remained unchanged. Ki mean, K1 mean and Ki 95%-interval-width maps exported successfully for both models with verified voxel identities. These tiny runs check software interfaces, not trained-model quality or posterior calibration.

Windows-style transient and permanent file locks were injected in automated tests: short locks recover, permanent locks preserve the prior settings file, and a locked progress display does not abort computation. Child startup exceptions and native exits without a Python traceback retain interpreters, arguments and readable logs. A simulated incompatible CUDA kernel reports the environment and installation guidance. **The original Windows WinError 5 and RTX 5060 training failure could not be reproduced without the original traceback/device environment. Actual Windows/CUDA and MPS execution remain unverified locally.** Run **Check compute setup** on the target computer and retain its report and job log if a failure recurs.

## Original UC Davis measured AIFs — 14 October 2026

**42/42 automated checks passed** on the same Python 3.11 / PyTorch 2.6.0 Apple Silicon CPU environment. UC Davis now defaults to the unchanged original `data/aif/UCDavis_1H35.csv`; the intermediate resampled CSV has been removed. The simulator interpolates directly from original samples, with explicitly saved zero-origin and final-value-hold policies.

New regression checks verify original AIF samples remain unchanged; midpoint coverage ends at 57.5 minutes while frame-average coverage ends at 60 minutes; strict coverage and invalid policies are rejected; small 32 / 16 / 8 datasets generate and validate in both observation modes; source checksums and boundary policies are saved; and GUI coverage checks prevent an invalid protocol or dataset job from being launched. The GUI's endpoint choice survives page navigation. Existing projects keep their saved settings and need to select the endpoint policies explicitly in step 1. This change does not alter the acceleration code or establish scientific posterior accuracy.

## Project persistence, inference and early stopping — 14 October 2026

**36/36 automated checks passed** on the existing Python 3.11 / PyTorch 2.6.0 Apple Silicon CPU environment. Added checks cover arbitrary page navigation, project switching and browser-session restoration of unfinished edits; pending voxel analyses clearing old results; lowest-validation checkpoint selection after an early stop before the historical epoch-151 threshold; per-epoch snapshot selection; LoRA weight folding for linear/convolution/transposed-convolution layers; automatic device routing; MPS expanded-batch limits; cooperative cancellation and structured progress; and forward simulation at the posterior medians of both LoRACM and scratch CM.

A two-epoch UC Davis pair of models produced posterior overlays and median forward curves for a synthetic noiseless 2TC TAC on the supplied 35-frame schedule. Multistart NLS converged with TAC RMSE approximately `4.6e-11` in input units. This checks the forward-fitting implementation, not the predictive quality of the tiny models. CSV and terminal TAC-figure exports passed.

Browser review confirmed an unsaved schedule-cell edit survived leaving/reopening step 1, a 4,120-pair simulation completed with a native progress bar, checkpoint epochs/losses and inference controls appeared, completed comparisons were explicitly selectable, and an image click populated the correct voxel ID with TAC/AIF and posterior panels. Old unsafe MPS maps show a warning. Source parsing and local documentation-link checks also passed.

The author's Mac prediction file used PyTorch 2.14.1, MPS, 512 curves and a 200-draw chunk: **102,400 network rows per call**. Bright-band errors repeat in the 184-voxel tail above the reported 65,536-row MPS linear-kernel boundary. A 64-voxel, 1,000-draw CPU control using the same checkpoint removed those tail jumps. The updated code caps expanded MPS calls at 4,096 and runs a fixed-input CPU/MPS comparison before prediction. **MPS execution of the updated guard and CUDA remain unverified locally**, because GPU access was unavailable in this verification environment. Hosted Windows/Linux/macOS CI has not been run on these local changes.

## UC Davis numbered workflow and interactive imaging — 14 October 2026

The updated default route was checked on the same Apple Silicon / PyTorch 2.6.0 CPU environment recorded below. **29/29 automated checks passed**, including default-preset agreement with the measured Sub001 schedule, empirical 95% interval widths, joint-draw Kᵢ intervals, voxel-ID reordering, geometry rejection, 3D slice coordinates and imported custom-data configuration without a simulation protocol.

The small UC Davis protocol generated 32 / 16 / 8 labelled pairs on the patient's exact 35-frame schedule, then trained for two CPU epochs. Whole-slice inference completed for **all 24,237 foreground voxels**, with no timing override. Parameter interval-width arrays were finite, nonnegative and correctly shaped. Kᵢ and K₁ interval-width maps exported through the terminal module. A measured voxel was extracted by ID, sampled with 1,000 joint draws, fitted by bounded NLS and plotted with dashed NLS estimates. NLS convergence does not establish goodness of fit or kinetic-model adequacy.

Browser review confirmed a foreground image click selects the matching voxel ID and displays the TAC plus input function and posterior KDE beside the map. Switching from Kᵢ to K₁ and selecting 95% interval-width maps updates the image and corresponding posterior. Voxel samples are cached; whole-image draws are not stored. The author confirmed the supplied geometry's mapping and display convention.

That earlier walkthrough validation used an intermediate resampled input file, actual frame durations and RK4. Those intermediate defaults were superseded by the 15 October HPC reproduction recipe above; the earlier results are retained here as validation history. The two-epoch weights and eight-draw maps are installation artifacts, not scientific results. Windows/Linux hosted CI, acceleration and full research training remain unverified on these local changes.

## Local browser and terminal workflows — 13 October 2026

Validated on **Apple Silicon macOS, Python 3.11, CPU**, with **PyTorch 2.6.0**. Exact runtime versions are recorded in [workflow-test-environment.json](workflow-test-environment.json). The test environment is an isolated virtual environment that also uses existing scientific packages; it is not a fresh complete dependency installation.

| Check | Result |
| :--- | :--- |
| Automated checks | **22/22 passed**, including all GUI pages, data/mapping contracts, schema inference, training configuration validation and transfer behaviour |
| Numerical simulation | Custom 1TC RK4 agrees with its analytical solution; linear frame-average integration and duration-dependent noise checked |
| Nonlinear least squares | Recovers noise-free custom 1TC parameters; convergence and bound diagnostics saved |
| LoRACM / built-in 2TC | 32 train / 16 validation / 8 test pairs → one CPU epoch → 32 posterior draws → NLS and exported figure; passed |
| LoRACM / custom 1TC | Same complete workflow with three predicted parameters and reset parameter heads; passed |
| CM from scratch | Same complete 2TC workflow, with no foundation load or adapters; learned weights change and Fourier features remain fixed |
| Paired comparison | LoRACM and scratch CM predict on the same held-out file; truth-based RMSE/bias, posterior samples and provenance reports checked |
| Input guards | Rejects schedule/duration conflicts, nonfinite values, schema conflicts and existing output folders |
| Text portability | Unicode configuration paths load under a non-UTF-8 locale; manual/configuration/log text uses explicit UTF-8 |
| Browser review | Scratch training launched and completed; progress/losses, comparison metrics and posterior overlays displayed; single-TAC posterior/NLS workflow reviewed |
| Feng cohort | Supplied four UC Davis AIFs fitted; peak-relative fit RMSE approximately 1.2–1.6% |

The CI workflow in [portable-workflow.yml](../../.github/workflows/portable-workflow.yml) defines CPU checks and all three installation workflows for **Windows, Linux and Apple Silicon macOS**. Those hosted checks have not run on these local changes. Windows/Linux hardware and CUDA/MPS acceleration have not been verified locally; MPS was unavailable in the test session. Intel Macs are outside the support scope.

These small runs verify software interfaces and numerical contracts. They do not establish scientific posterior accuracy, calibration, clinical suitability, or convergence after the default 5,000 epochs. The preserved notebooks and historical generator were not revalidated in this update.

## Historical LoRACM release checks — 29 September 2026

Validated locally on 29 September 2026 using the author's available Python environment. Exact versions are recorded in `test-environment.json`; this records the test environment, not a portable dependency lock.

| Check | Result |
| :--- | :--- |
| Python source parsing | All included Python files parse |
| Array, statistic and mapping tests | 6/6 passed |
| Tiny dataset generation | 32 train / 16 validation / 8 test pairs generated |
| Foundation transfer | Loaded the exact checkpoint named by the supplied research training script; no missing weights for Scenario 1 |
| Two-epoch CPU training | Passed; 59,564 trainable parameters of 2,193,350 total (2.72%); best and final EMA checkpoints saved |
| Real-slice inference smoke test | Passed on four voxels, eight posterior samples each; explicit timing-mismatch override recorded |
| Timing guard | Correctly rejected the supplied real slice without the override |
| Dimension-changing transfer | A 60-input model trained its resized conditioning weight while keeping transferred convolution weights frozen |
| Full release dataset | Generated 50,000 train / 100,000 validation / 100 test pairs; checked shapes, finite values and valid AIF indices |
| Notebook | Valid notebook schema; unverified geometry correctly rejected; all cells executed with a synthetic plotting fixture |
| Figure export | Synthetic fixture rendered and visually inspected; labels and colour bar were legible |

The smoke-trained model and synthetic plot are not distributed as research results. No full 5,000-epoch adaptation, CUDA/MPS test, multi-GPU test, scientific posterior validation against MCMC, verified patient parametric map, or fresh-environment installation has been performed here. The smoke test establishes working interfaces, not reproduction of manuscript accuracy or runtime. The full generated dataset uses the documented chunked random streams, not the original research run's exact samples.
