# Prepare your own fine-tuning dataset

[Manual](../README.md) · [Prediction inputs](PREDICTION.md)

Fine-tuning needs **labelled parameter–curve pairs**. Clinical TACs without kinetic labels are prediction data, not a direct replacement for this training dataset. You can obtain labels from a validated process or generate synthetic pairs from your forward model.

## Required file layout

Use a compressed NumPy NPZ containing:

| Key | Shape | Contents |
| :--- | :--- | :--- |
| `x_train`, `x_val`, `x_test` | `[N_split, P+1]` | Ordered kinetic parameters, then nonnegative noise amplitude |
| `y_train`, `y_val`, `y_test` | `[N_split, 2T]` | **All T TAC values**, then **all T input-function values** |
| `t_meas` | `[T]` | Common increasing positive frame midpoints, minutes |
| `parameter_names` | `[P]` strings | Exact ordered labels; required for custom parameter sets |
| `frame_durations_min` | `[T]` | Actual positive durations; recommended for imported data |
| `observation` | scalar string | `midpoint` or `frame_average`; recommended |

The old five-parameter format without names/durations is still readable. If names are absent, it is interpreted as `K1,k2,k3,k4,Vb`; its final x column is the noise amplitude. For imported labels without a meaningful recorded simulation-noise amplitude, a documented zero placeholder is accepted and is not predicted.

## Conversion example

```python
import numpy as np

# theta_split: [N,P]; tac_split and aif_split: [N,T].
# Expand a shared AIF to [N,T] if necessary. Split before normalization.
arrays = {}
for split, theta, tac, aif in (
    ("train", theta_train, tac_train, aif_train),
    ("val", theta_val, tac_val, aif_val),
    ("test", theta_test, tac_test, aif_test),
):
    aif = np.broadcast_to(aif, tac.shape)
    arrays["x_" + split] = np.column_stack([theta, np.zeros(len(theta))])
    arrays["y_" + split] = np.column_stack([tac, aif])

np.savez_compressed(
    "my_finetune.npz", **arrays,
    t_meas=np.asarray(times_min),
    frame_durations_min=np.asarray(durations_min),
    parameter_names=np.asarray(["K1", "k2", "k3", "k4", "Vb"]),
    observation=np.asarray("midpoint"),
)
```

Supply your own arrays in this template. Save ordinary numeric/string arrays; object/pickled arrays are rejected. Use consistent activity units, parameter units, observation conventions and correction settings across all splits. Avoid patient leakage: split by subject/acquisition when evaluating generalization to new subjects, rather than dividing repeated curves from the same patient at random.

## Validate and select

GUI: **2 · Dataset → Use my own NPZ dataset → Validate and select dataset**. Larger files can be selected by local path rather than uploaded through the browser.

Terminal:

```bash
python protocol.py /path/to/my_finetune.npz
```

Validation checks all required arrays, widths, finite values, nonempty splits, positive increasing times, optional duration vectors, nonnegative noise amplitudes and nonzero training scales. It reports `x_dim=P`, `y_dim=2T`, parameter names and split counts. Use these in the training config; set `kinetic_model="custom"` for another parameter set.

Keep data in physical units; training derives normalization from the **training split only**, with independent parameter means/stds and a shared scalar TAC/AIF mean/std. If a parameter is fixed in every training row, its zero standard deviation is rejected: remove that parameter from the predicted set and represent the fixed quantity inside a custom model.

A passing format check does not verify truth labels, units, priors or population coverage. Retain your own provenance and split definitions beside the file. You can then skip simulation and proceed directly to [fine-tuning](FINE_TUNING.md).
