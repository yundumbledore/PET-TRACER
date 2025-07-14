# PET-TRACER
PET-TRACER (PET **T**otal-body Paramet**R**ic **A**nalysis via **C**onsistency **E**stimation for **R**adiotracers)
#### Yun Zhao, Steven Meikle (The University of Sydney, Australia), Email: yun.zhao@sydney.edu.au, steven.meikle@sydney.edu.au

PET-TRACER is an open-source Python framework designed to bring state-of-the-art Bayesian kinetic parameter estimation to dynamic total-body positron emission tomography (PET) imaging. At its core, PET-TRACER implements a novel generative consistency model (CM) pipeline that accelerates posterior inference of two-tissue compartment parameters—namely $K_1, k_2, k_3, k_4$, and blood volume fraction $V_b$—from time–activity curves (TACs) and arterial input functions (AIFs). By collapsing what traditionally requires hundreds of denoising steps into just three highly optimized U-Net passes, PET-TRACER enables rapid, high-fidelity sampling of per-voxel kinetic posteriors, paving the way for truly quantitative, uncertainty-aware parametric imaging at whole-body scale.

## Methods
The conditional consistency model in PET-TRACER reframes diffusion-based posterior estimation as a single-shot denoising task. At its core is a lightweight 1D U-Net $f_{\theta}(x_t, t, y)$ that, given a noisy parameter vector $x_t$ at noise level $t$ and the measured TAC + AIF $y$, predicts the clean kinetic parameters $x_0$. During training, a “student” network $f_{\theta}$ learns to match the outputs of an exponential-moving-average “teacher” network $f_{\theta^-}$ across adjacent noise scales. Paired noisy inputs at levels $t_{n+1}$ and $t_n$ are fed to the student and teacher respectively, and the student is optimized to minimize the consistency loss:
$$
\mathcal{L}_{\mathrm{consistency}}(\theta) \;=\; \mathbb{E}_{t_n,\,x_0,\,\epsilon}
\Bigl\|\,f_{\theta}\bigl(x_{t_{n+1}},\,t_{n+1},\,y\bigr)
- f_{\theta^-}\bigl(x_{t_n},\,t_n,\,y\bigr)\Bigr\|^2,
\end{equation}
where
\begin{equation}
x_{t} \;=\; \sqrt{1 - t^2}\,x_0 \;+\; t\,\epsilon,
\quad \epsilon \sim \mathcal{N}(0, I).
$$
The teacher weights $\theta^-$ are updated via exponential moving average of the student weights $\theta$, enabling the network to collapse a full diffusion trajectory into a single forward pass while preserving conditional fidelity on $y$.


## Getting Started
1. Clone the repo and create a conda environment via
2. Visualize posteriors predicted by CM with three example TACs.

## Adaptation to your data
The consistency model in PET-TRACER was trained and validated on dynamic PET curves discretized into 35 frames, as shown below. Because the posterior inference network expects input TACs and AIFs sampled at these exact time points, you should resample your real dynamic PET data to this same 35-frame schedule before running inference. Likewise, if you’re generating synthetic data for training or testing, be sure to simulate both the tissue time–activity curve and arterial input function at these 35 time points. This alignment ensures that the model’s learned temporal features correctly match your input, enabling accurate, uncertainty-aware kinetic parameter estimation.

tspan = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 75.0, 90.0, 105.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 600.0, 720.0, 840.0, 960.0, 1080.0, 1260.0, 1440.0, 1620.0, 1800.0, 2100.0, 2400.0, 2700.0, 3000.0])/60

## Support and Help
Please raise your queries via the "Issues" tab or contact me (yun.zhao@sydney.edu.au). I will respond as soon as possible.
