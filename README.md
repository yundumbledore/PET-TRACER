# PET-TRACER
PET-TRACER (PET **T**otal-body Paramet**R**ic **A**nalysis via **C**onsistency **E**stimation for **R**adiotracers)
#### Yun Zhao, Steven Meikle (The University of Sydney, Australia), Email: yun.zhao@sydney.edu.au, steven.meikle@sydney.edu.au
| Total body dynamic PET | Parametric imaging of net influx rate Ki |
|:----------:|:-----------:|
| ![demo1](Assets/Dynamic_pet_data.gif) | ![demo2](Assets/Dynamic_pet_data.gif) |

PET-TRACER is an open-source Python framework designed to bring state-of-the-art Bayesian kinetic parameter estimation to dynamic total-body positron emission tomography (PET) imaging. At its core, PET-TRACER implements a novel generative consistency model (CM) pipeline that accelerates posterior inference of two-tissue compartment parameters—namely $K_1, k_2, k_3, k_4$, and blood volume fraction $V_b$—from time–activity curves (TACs) and arterial input functions (AIFs). By collapsing what traditionally requires hundreds of denoising steps into just three highly optimized U-Net passes, PET-TRACER enables rapid, high-fidelity sampling of per-voxel kinetic posteriors, paving the way for truly quantitative, uncertainty-aware parametric imaging at whole-body scale.

## Highlights
1. Processes total body PET containing tens of millions of voxels in **90 minutes**.
2. Outperforms traditional diffusion models (DDPM, score-based diffusion) by **at least 100×** and is **3×** faster than GPU-based parallel ABC (Approximate Bayesian Computation).
3. Matches full MCMC-based inference quality, while reducing uncertainty estimation error by **at least 10%** compared to ABC when MCMC posteriors are treated as ground truth.

## Methods
The **consistency model** at the heart of PET-TRACER is a conditional generative framework that learns to map noisy, partially diffused kinetic parameter estimates back to their true posterior distributions in just a handful of passes. Built on a lightweight 1D U-Net architecture, the model is trained to denoise and “roll back” samples through a learned consistency function, rather than simulating every diffusion timestep. During training, the network sees paired noisy and clean two-tissue compartment parameter curves $K_1, k_2, k_3, k_4, V_b$ alongside their corresponding TAC + AIF inputs and learns to enforce consistency between successive denoising steps. The result is a model that captures the underlying posterior geometry with high fidelity, learning to produce accurate, uncertainty-aware parameter samples from arbitrary starting noise levels.

The **multistep consistency sampling algorithm** then leverages this trained model for fast posterior draws. First, a random Gaussian vector $x_T$ is sampled at the highest noise scale $T$ and is fed to the U-Net with TAC + AIF pair and noise level $T$. The U-net produces an initial coarse estimate. Next, one steps through a strictly decreasing sequence of intermediate noise levels $t_1>t_2>\dots> t_{N-1}. At each step $n$, fresh Gaussian noise is injected to corrupt the denoised sample back to noise level $n$ and the U-Net refines it back to noise free sample $x_0$. After processing all $N-1$ levels, the final $x_0$ is returned as a sample from the posterior. By repeating the sampling many times, one can get a bunch of posterior samples. The multistep sampling algorithm is illustrated in the figure below.

![](Assets/Multistep_consistency_sampling.png)

This figure shows an example of posterior estimation with one TAC-AIF pair predicted by CM, MCMC, and other baselines.
![](Assets/Posterior_estimation_example.png)

## Getting Started
1. Clone the repo and create a conda environment via environment.yml.
2. Visualize posteriors predicted by CM with three example TACs.

## Adaptation to your data
The consistency model in PET-TRACER was trained and validated on dynamic PET curves discretized into 35 frames, as shown below. Because the posterior inference network expects input TACs and AIFs sampled at these exact time points, you should resample your real dynamic PET data to this same 35-frame schedule before running inference. Likewise, if you’re generating synthetic data for training or testing, be sure to simulate both the tissue time–activity curve and arterial input function at these 35 time points. This alignment ensures that the model’s learned temporal features correctly match your input, enabling accurate, uncertainty-aware kinetic parameter estimation.

tspan = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 75.0, 90.0, 105.0, 120.0, 180.0, 240.0, 300.0, 360.0, 420.0, 480.0, 600.0, 720.0, 840.0, 960.0, 1080.0, 1260.0, 1440.0, 1620.0, 1800.0, 2100.0, 2400.0, 2700.0, 3000.0])/60

## Support and Help
Please raise your queries via the "Issues" tab or contact me (yun.zhao@sydney.edu.au). I will respond as soon as possible.
