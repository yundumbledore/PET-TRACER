# Data and weights provenance

This local review package copies the four-AIF CSV and Sub001 coronal HDF5 from the supplied research paths. A candidate geometry file is derived from the mask referenced by the supplied notebook. The foundation checkpoint is the exact file referenced by the supplied training script, renamed `foundation.pth`; it is not the separate `LoRA-CM/pretrained.pth` file. Its `model_state_dict` is the transfer source, preserving the supplied loader's preference over `ema_state_dict`.

The adaptation dataset is newly generated from the supplied AIFs and configured Scenario 1 priors. Its JSON sidecar records generation settings and software versions. The full array set and hashes are listed in the package manifest. Smoke data and smoke weights are validation artifacts, not the paper's trained adaptation model.

Before publication, fill in the original dataset source/attribution, applicable reuse terms, permission to distribute the clinical slice and mask, activity units and correction conventions, and the intended licence for model weights and derived simulation data. The repository's existing MIT licence is preserved for code; it is not being assigned automatically to clinical data or weights by this packaging step.
