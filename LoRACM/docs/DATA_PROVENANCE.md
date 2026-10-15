# Data and model provenance

The four-AIF CSV, Sub001 coronal TAC file, foundation checkpoint and verified slice mask/mapping retain their original contents. The author confirmed the C-order foreground mapping, origin and pixel aspect ratio. Physical millimetre spacing is not supplied.

The original **epoch-2755** adapted model and its x/y normalization are supplied in `paper/ucdavis/`. The checkpoint tensor bytes are unchanged. JSON metadata adds portable paths, parameter names, the explicit time-label convention and the Ki/display policy. `manifest.json` records hashes, original checkpoint filename and provenance limits. Its recorded 100,000 fine-tuning examples follow the author's confirmation, not the current archive row count.

`reference_ki.npz` contains original saved HPC Ki mean/std arrays, with the supplied voxel IDs. `reference_ki.png` renders those arrays with the original 2×2 median filter and fixed 0–0.06 scale. The summary archive has no original draw-level Ki transform metadata. This release explicitly uses the 1e-4 parameter flooring convention found in the original HPC selected-voxel inference code; the checkpoint association follows the author's recollection and current inference source.

New UC Davis training pairs use the original Euler/observation and noise equations with unchanged CSV AIF conditioning, the parameter priors specified in the supplied HPC `Create_dataset.py` and 100,000 training examples. NumPy streams and bounded-memory generation do not recreate the original JAX sample arrays. The older bundled `data/simulated/scenario1.npz` has 50,000 training rows, and is not selected by the paper reproduction route. Use the supplied checkpoint for paper-model inference; retraining across backends is not bitwise reproducible.

The existing MIT software licence is retained. Dataset attribution, activity units/correction conventions, clinical-data reuse terms and model-weight distribution terms should accompany the associated research release. The historical root `MANIFEST.json` describes an earlier inventory; the paper asset manifest describes the new supplied paper model.
