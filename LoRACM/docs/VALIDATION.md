# Workflow validation

## UC Davis numbered workflow and interactive imaging — 14 October 2026

The updated default route was checked on the same Apple Silicon / PyTorch 2.6.0 CPU environment recorded below. **29/29 automated checks passed**, including default-preset agreement with the measured Sub001 schedule, empirical 95% interval widths, joint-draw Kᵢ intervals, voxel-ID reordering, geometry rejection, 3D slice coordinates and imported custom-data configuration without a simulation protocol.

The small UC Davis protocol generated 32 / 16 / 8 labelled pairs on the patient's exact 35-frame schedule, then trained for two CPU epochs. Whole-slice inference completed for **all 24,237 foreground voxels**, with no timing override. Parameter interval-width arrays were finite, nonnegative and correctly shaped. Kᵢ and K₁ interval-width maps exported through the terminal module. A measured voxel was extracted by ID, sampled with 1,000 joint draws, fitted by bounded NLS and plotted with dashed NLS estimates. NLS convergence does not establish goodness of fit or kinetic-model adequacy.

Browser review confirmed a foreground image click selects the matching voxel ID and displays the TAC plus input function and posterior KDE beside the map. Switching from Kᵢ to K₁ and selecting 95% interval-width maps updates the image and corresponding posterior. Voxel samples are cached; whole-image draws are not stored. The author confirmed the supplied geometry's mapping and display convention.

The walkthrough uses an explicitly aligned AIF CSV, actual frame durations and RK4; historical Euler-generated Scenario 1 files remain separate. The two-epoch weights and eight-draw maps are installation artifacts, not scientific results. Windows/Linux hosted CI, acceleration and full research training remain unverified on these local changes.

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
