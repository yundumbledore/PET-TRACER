# Validation performed on this release candidate

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
