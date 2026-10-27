# Measurement noise for simulated TACs

[Manual](../README.md) · [Dataset preparation](PREPARE_DATA.md) · [UC Davis example](UCDAVIS.md)

New CM and LoRACM simulation projects use **mean-preserving log-normal noise**, with the **actual acquisition duration of each frame**. In **2 · Dataset** (or **CM · Dataset**), retain **Measurement noise → Log-normal** and choose the amplitude range for your study. The starting range is **3–7**, with FDG half-life **109.8 min**. Existing projects retain their saved settings; **Gaussian (legacy)** remains available for comparisons.

## Equations

For the clean, decay-corrected tissue activity at frame midpoint $t_i$, define $\bar C_i=\max(C_T(t_i),\varepsilon)$, with $\varepsilon=10^{-8}$. Draw one amplitude $\ell\sim\mathcal U(3,7)$ per TAC and independent $\epsilon_i\sim\mathcal N(0,1)$ per frame, conditional on that amplitude. Set

$$
\lambda=\frac{\ln 2}{T_{1/2}},\qquad
\sigma_i=\ell\sqrt{\frac{\bar C_i e^{-\lambda t_i}}{\Delta t_i}}\,e^{\lambda t_i}
=\ell\sqrt{\frac{\bar C_i e^{\lambda t_i}}{\Delta t_i}},
$$

$$
\eta_i^2=\ln\!\left(1+\frac{\sigma_i^2}{\bar C_i^2}\right),\qquad
\widetilde C_i=\bar C_i\exp\!\left(\eta_i\epsilon_i-\frac12\eta_i^2\right).
$$

Here **$\Delta t_i$ is the acquisition duration**, not the distance between neighbouring midpoints. All times, durations and half-life use minutes. The conditioning AIF is kept unchanged. The UC Davis recipe retains its recorded CSV simulation times and separate patient labels; changing noise does not resample either curve.

The log-normal moment identities give

$$
\mathbb E[\widetilde C_i\mid C_T,\ell]=\bar C_i,\qquad
\operatorname{Var}[\widetilde C_i\mid C_T,\ell]
=\bar C_i^2(\exp(\eta_i^2)-1)=\sigma_i^2.
$$

The subtraction of $\eta_i^2/2$ preserves the mean. The median is $\bar C_i\exp(-\eta_i^2/2)$ and is therefore below the mean when noise is nonzero. These statements are conditional on the chosen TAC and amplitude; mixing amplitudes produces a mixture of log-normal distributions. [NIST: log-normal distribution](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3669.htm).

## Interpretation and limits

Barrett, Wilson and Tsui derived an approximate log-normal description of EM-reconstructed **voxel** values and expressions for image covariance. Wilson, Tsui and Barrett tested that theory with Monte Carlo simulations. Their results motivate the log-normal marginal approximation; they do not establish the frame-wise variance formula above or the amplitude range 3–7 for every scanner and reconstruction. [Part I: Theory](https://pubmed.ncbi.nlm.nih.gov/15552088/) · [Part II: Monte Carlo simulations](https://pubmed.ncbi.nlm.nih.gov/15552089/).

This simulator uses a **moment-matched, frame-wise approximation**, without reconstructed-image spatial covariance. Validate its variance and skewness against your own reconstruction, count levels, corrections and voxel/ROI data. The amplitude depends on activity units and reconstruction conditions; 3–7 is a starting study setting, not a universal dimensionless calibration. The positive decay exponent assumes decay-corrected activity; this recipe does not model other correction conventions automatically. ROI averaging and modern reconstruction options can change the appropriate noise distribution.

Near zero activity, the specified mean floor and a large coefficient of variation create highly skewed distributions with rare large values. The implementation calculates the variance ratio in log space using float64, never clips noisy samples back to $\varepsilon$, floors only floating-point underflow to the smallest positive float64, and rejects overflow. Mathematical moment identities apply before that numerical limit. Check units and amplitude if tails become extreme.

## Terminal configuration and provenance

The four supplied new-dataset templates include:

```json
{
  "noise_model": "lognormal",
  "noise_timing": "frame_duration",
  "noise_epsilon": 1e-8,
  "noise_prior": {"type": "uniform", "min": 3.0, "max": 7.0},
  "half_life_min": 109.8
}
```

Keep `frame_durations_min` in the protocol, with one positive duration per frame. Run `python simulate_protocol.py --config <protocol.json>`, then validate and select the resulting dataset as usual. The same settings apply to LoRACM adaptation and scratch-CM generation. The final `x_*` column remains the sampled amplitude; it is not a predicted kinetic parameter. No logarithmic transform of training inputs is introduced.

Dataset NPZ files record `measurement_noise_json`; their JSON sidecars, dataset-validation report and trained run's configuration, scaling and summary retain the distribution, timing, floor, amplitude prior and half-life. Training rejects a supplied simulation protocol whose noise provenance conflicts with the dataset. Imported datasets without that field remain readable, with noise provenance reported as unknown.

For historical Gaussian experiments, use `noise_model="gaussian"`; `noise_timing="sample_spacing"` reproduces the original Euler generator's `[t[0], diff(t)]` scaling, whereas `"frame_duration"` uses acquisition durations. Unmarked saved protocols keep their previous Gaussian behavior. The preserved `simulation.generate_dataset` research API also retains its old default; explicitly pass `noise_model="lognormal"` and `frame_durations=<durations>` for the new model.

The supplied **epoch-2755 LoRACM checkpoint**, foundation weights, archived Gaussian datasets and **pretrained Bayesian model-selection demonstration** are unchanged. Their original simulations have not been relabelled or retrained. To study the new noise model, generate a new dataset and train a new CM or LoRACM run. Paper-checkpoint imaging continues to use the original model and normalization. CM's Gaussian latent diffusion/consistency perturbations are part of the learning algorithm and remain unchanged. Posterior forward-curve credible bands remain noiseless model responses; the NLS comparison retains its existing least-squares objective.
