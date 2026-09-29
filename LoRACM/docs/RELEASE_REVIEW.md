# Author review before public release

This review is grounded in the supplied manuscript, scripts, CSV, and locally available supporting files. It separates scientific choices from packaging repairs.

## Resolve these before claiming an exact Scenario 1 reproduction

| Finding | Evidence | Action |
| :--- | :--- | :--- |
| Time-grid discrepancy | After the first frame, HDF5 times exceed CSV times by about 0.0147833 min (0.887 s); the Sub001 AIF values agree numerically | Confirm the meaning of the adjustment and which grid the trained model uses. Correct metadata or resample only when scientifically justified. The inference guard currently detects this. |
| Spatial mapping unverified | The notebook assigns estimates in mask order. Voxel counts match (24,237), but HDF5 identifiers are not slice-flat indices | Verify the exact column-to-mask correspondence against the exporting code, then update geometry metadata. Confirm orientation and spacing too. |
| Checkpoint selection differs in wording | Manuscript says lowest validation loss; script considers best checkpoints only after epoch 150 | Document that eligibility rule in the paper or deliberately change and revalidate it. Default remains epoch 151 onward. |
| Noise delta-time convention | Code uses first midpoint then midpoint differences, while the real file includes explicit frame durations | State the actual convention in the method. Do not silently substitute durations. |
| Shared measured input pool | Four supplied AIFs are sampled in all three simulation splits | Describe this as simulated-pair splitting, not unseen-subject validation. |
| Restricted test priors | Original generator contracts test parameter ranges to their central 50% | Report the test sampling bounds or revise the evaluation deliberately. |
| New dataset random streams | Portable generation batches before dense interpolation and folds a key per chunk | Release the generated NPZ and checksum; do not claim it is the exact original training dataset. |

## Repairs made in this candidate

- All workflow paths are configurable and relative to the LoRACM working directory.
- The original training objective, scalar conditioning normalization, EMA update, prior bounds and noise model are retained.
- Scheduler settings use the passed configuration instead of a global dictionary.
- Training rejects empty/incompatible data and batch sizes that would produce zero batches.
- The builder refuses a foundation checkpoint with no matching tensors and unfreezes missing/new parameters using exact name matching.
- Best and final checkpoints have predictable names. Short smoke runs use epoch 1 eligibility; the manuscript configuration preserves epoch 151.
- The original `params_mean` output actually contained medians. Means and medians now have separate, accurate names.
- Inference now calculates the Kᵢ summaries requested by the viewer. It uses joint draws, not a plug-in transformation of parameter medians.
- Memory-heavy posterior generation is split into sample chunks; only one voxel batch of draws is retained on CPU.
- The viewer uses relative paths, explicit geometry and exported figures. It removes unrelated t-tests, unused imports and the incorrect unused Kᵢ histogram expression from the research notebook.
- No smoothing or positivity clipping is silently applied to displayed estimates.

## Release packaging

Preserve existing `Source/`, original notebooks, `Assets/`, `Pretrained/` and `Sample_data/`. Add LoRACM as an independent module and replace the root README with the new entry point. Keep the old README in `docs/legacy-framework.md` so the original paper remains usable. Confirm the published citation and add the extension's public link when available.

Before submission, make a versioned software release and archive the exact code, foundation weights, adaptation dataset, real demo, geometry and a validated adapted model with checksums. This archive includes the intended demo assets, and `.gitignore` explicitly permits those files while excluding new experiment outputs. They can also be archived in release storage or a research data archive. Record the code commit, environment, dataset checksum and model checksum alongside the paper. Add a downloadable adapted model so readers can choose inference-only use; this candidate does not label its smoke-trained weights as such a model.

Data redistribution terms and provenance need author confirmation. The manuscript is not copied into this bundle. No GitHub changes, uploads, commits or releases have been made.
