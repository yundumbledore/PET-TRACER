# Kinetic models, AIFs and acquisition protocols

[Manual](../README.md) · [Dataset preparation](PREPARE_DATA.md)

## Kinetic models

Built-in **2TC** predicts `K1, k2, k3, k4, Vb`, in that order. Rate constants use minutes; `Vb` is a blood-volume fraction. The same input function is used for plasma forcing and the blood-volume term. Add a custom model if your application needs different input handling or kinetics.

The new simulator accepts a local Python plugin implementing:

```python
PARAMETER_NAMES = ["K1", "k2", "Vb"]
STATE_SIZE = 1

def rhs(t, state, parameters, input_value):
    # state [N,S], parameters [N,P], input_value [N]; return [N,S].
    ...

def observe(state, parameters, input_value):
    # Return the predicted tissue activity [N].
    ...
```

A complete working 1TC example is in [`examples/custom_1tc.py`](../examples/custom_1tc.py). Copy it, edit the equations, then set `model="custom"`, `model_file`, `parameter_names`, and the ordered priors in your protocol. The GUI permits different parameter counts. Plugins execute local Python; inspect their source before use.

Validate the configuration before generation with `python protocol.py --config configs/dataset_custom.json`.

GUI simulation uses zero initial states, one input channel, midpoint observations and fixed-step Euler integration. Test a custom model against analytical solutions or a validated independent implementation, and check convergence at smaller step sizes. Delay equations, multiple independent inputs and state-dependent discontinuities need an extension of this contract; the old dual-input/SRTM research classes are not automatically exposed as GUI plugins. No AATH scenario is claimed as validated by this update. Existing terminal RK4 protocols remain readable. See [the UC Davis guide](UCDAVIS.md#timing-forward-model-and-noise) for the original observation equations and [measurement noise](MEASUREMENT_NOISE.md) for the new log-normal defaults.

**Transfer limits.** Compatible foundation backbone weights transfer. Changed parameter names/order or any custom model resets and trains the parameter encoder/decoder, even if the number of parameters remains five. Changed frame counts train resized conditioning weights. LoRA and optional normalization weights train alongside these layers. The original pretrained model represents FDG–2TC; using a new tracer or model does not establish posterior accuracy or calibration. Review `transfer_report.json` and validate on held-out labels or an independent estimator.

## Input functions

### Use measured inputs

Use a CSV whose first column is time in minutes, with one AIF per subsequent column:

```csv
time_min,subject_01,subject_02
0,0,0
0.25,12000,10000
0.5,9000,8500
1,4000,4200
```

Provide finite nonnegative activity. The AIF grid can differ from the output frame schedule. The simulator reads the original CSV samples, linearly interpolates on its integration grid and randomly selects one curve for each pair. It never rewrites or creates a resampled AIF CSV.

The GUI interpolates measured AIFs to the selected midpoints, holding the first/last measured values outside their sampled range. Supply an AIF covering your measurement times whenever possible. The default UC Davis CSV and unchanged Sub001 schedule retain their original activity samples, with the recorded CSV simulation times and patient labels stored separately. The dataset provenance records the source checksum and interpolation policy.

For older terminal `simulation_mode="rk4"` protocols, `aif_before_first` selects `require_zero` or `zero_at_origin`, and `aif_after_last` selects `require_coverage` or `hold_last`. Those saved protocols retain their original endpoint and frame-average behavior.

A small AIF pool is acceptable, but repeated use does not create new independent subjects. Input units, tracer metabolite correction, plasma/blood treatment and delay/dispersion conventions must agree between simulation and testing.

### Fit and sample a Feng cohort

With multiple input functions, fit each to the repository's Feng equation:

```bash
python fit_feng.py --input data/aif/UCDavis_1H35.csv --output workspaces/my_project/feng.csv
```

The fit needs at least two curves and eight samples. It exports the seven columns `beta1,beta2,beta3,kappa1,kappa2,kappa3,t0`, AIF identifiers and RMSE diagnostics, plus a `.fits.csv` containing measured and fitted curves. Review the errors and curves. A successful numerical fit alone does not establish a suitable input model for your tracer.

Set `aif_mode="feng_cohort"` and `feng_parameters_csv` in the protocol. Whole parameter rows are resampled together; `feng_log_jitter=0.05` adds small multiplicative variation. This preserves within-row dependence before perturbation, and is not an estimated multivariate population model. Start with `0` for pure row resampling; the simulator rejects negative generated activity and asks you to review/reduce perturbations. Physiological plausibility and cohort coverage remain study-specific.

## Acquisition protocol

| Setting | Meaning |
| :--- | :--- |
| `t_meas` | Positive increasing **frame midpoints**, minutes |
| `frame_durations_min` | One positive actual acquisition duration per frame, minutes |
| `observation` | GUI: `midpoint`; saved terminal RK4 protocols also support `frame_average` |
| `half_life_min` | Isotope half-life; noise decay constant is `ln(2)/half_life_min` |
| `integration_step_min` | Integration step (default 0.01 min); check convergence |
| `kinetic_priors` | Ordered uniform ranges, one per predicted parameter |
| `noise_model` | New templates: `lognormal`; `gaussian` for historical comparisons |
| `noise_timing` | New templates: `frame_duration` (actual acquisition durations) |
| `noise_epsilon` | Positive clean-activity floor for noise calculations, default `1e-8` |
| `noise_prior` | Uniform nonnegative noise-amplitude range; new default 3–7 |
| `seed`, `chunk_size` | Reproducibility and bounded simulation working memory |

Midpoints/durations define nonoverlapping frames starting at or after zero; gaps are allowed. New GUI/terminal simulations use midpoint observations and [log-normal noise](MEASUREMENT_NOISE.md) scaled by actual frame durations; these durations are not inferred from neighbouring midpoints. Activity is treated as decay-corrected by this noise equation; use consistent correction conventions throughout your project. The UC Davis recipe stores its original CSV observation labels separately from patient frame midpoints. Legacy saved protocols retain Gaussian noise and their original timing unless explicitly changed.

Dataset generation uses CPU JAX for the original split-key/random-shape sequence, then integrates in bounded chunks. The seed, all three split sizes, priors, JAX version and PRNG settings determine the draws; changing chunk size does not change them. The dataset JSON records these settings. The original unmarked research-config path in `create_dataset.py` retains its older behavior; use `simulate_protocol.py` for the current protocol workflow. Use the supplied epoch-2755 checkpoint and original scaling for paper-model prediction.

Use [the schedule CSV](../examples/schedule.csv) as a format example. Its five-minute schedule and the default FDG half-life of 109.8 min are illustrative; replace them for your tracer and scan. Config units are minutes, including Feng rates/delay and kinetic rates.
