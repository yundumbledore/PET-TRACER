# PET-TRACER

**Bayesian kinetic modelling for dynamic total-body PET**

Latest Release: Adapt a pretrained consistency model to your tracer, time sampling schedule, and acquisition protocol.

[User manual](LoRACM/README.md) · [GUI guide](LoRACM/docs/WORKBENCH.md) · [Installation](LoRACM/docs/INSTALLATION.md) · [UC Davis steps 1–6](LoRACM/docs/UCDAVIS.md) · [Parametric imaging](LoRACM/docs/PARAMETRIC_IMAGING.md) · [Original framework](docs/legacy-framework.md)

PET-TRACER (**PET** **T**otal-body Paramet**R**ic **A**nalysis via **C**onsistency **E**stimation for **R**adiotracers) estimates kinetic parameter posteriors from tissue time–activity curves (TACs) and input functions. **LoRACM** adds low-rank adaptation: fine-tune the pretrained consistency model with labelled examples from your application, then predict on your own curves. A separate **CM from scratch** module trains a random-initialization baseline for comparison on shared held-out data.

Team: **Yun Zhao and Steven Meikle** (The University of Sydney, Australia), **Andrew Reader** (King’s College London, UK), and **Yanan Fan** (CSIRO’s Data61, Australia). The [original framework page](docs/legacy-framework.md) contains the method introduction, figures, team information, and first-paper demonstrations.

## Two ways to work

| Local graphical interface | Terminal workflow |
| :--- | :--- |
| Guided pages for protocol setup, data validation, training, and prediction. Runs locally in your browser. | The same workflow through Python commands and editable configuration files. Suitable for servers and batch jobs. |
| [Launch the GUI](LoRACM/docs/INSTALLATION.md#launch-the-gui) | [Terminal quick start](LoRACM/README.md#terminal-quick-start) |

The new workflow is designed for **Windows, Apple Silicon macOS (M-series), and Linux** with Python 3.11. CPU is the common baseline; PyTorch can use a supported CUDA GPU or Apple MPS device. The preserved research generator has separate [JAX installation requirements](LoRACM/docs/INSTALLATION.md#optional-legacy-research-generator).

![PET-TRACER local workbench with LoRACM, scratch CM and shared prediction workflows](Assets/LoRACM_Workbench.png)

## Start here

1. **[Install the environment](LoRACM/docs/INSTALLATION.md)** and launch the GUI or open a terminal.
2. **[Follow UC Davis steps 1–6](LoRACM/docs/UCDAVIS.md)** with the default template, supplied data and foundation weights. Choose **Quick installation check** for an installation check.
3. Choose your data route:
   - **[Simulate labelled pairs](LoRACM/docs/MODELS_AND_PROTOCOLS.md)** using 2TC or your own kinetic model, measured AIFs or a fitted Feng cohort, and your frame schedule.
   - **[Prepare your own dataset](LoRACM/docs/PREPARE_DATA.md)** and validate its format before training.
4. **[Fine-tune](LoRACM/docs/FINE_TUNING.md)** with defaults of **100,000 training pairs, LoRA rank 8, alpha 16**; track training loss, validation loss and learning rate. For a baseline, **[train a CM from scratch](LoRACM/docs/TRAIN_CM.md)** on the same pairs.
5. **[Predict on your own project](LoRACM/docs/PREDICTION.md)** in batches, or analyse one TAC with parameter posterior histograms, an NLS comparison and forward TACs from posterior medians.
6. **[Make parametric images](LoRACM/docs/PARAMETRIC_IMAGING.md)** of Kᵢ, other kinetic parameters or 95% credible-interval widths. Click a voxel to view its TAC, input function and posterior KDE beside the map.

UC Davis is the GUI's default template. It retains Scenario 1 priors and rank **4 / alpha 8** and uses the exact supplied patient's frame schedule for the complete imaging route. **My own project** retains the separate **100,000-pair / rank-8 / alpha-16** starting settings. Project settings, including unfinished edits, survive page navigation and browser refresh. A new project name creates an independent workspace. Scratch CM training is optional; enable it in the sidebar when you want a comparison.

```mermaid
flowchart LR
    A[Your kinetic model + input functions + protocol] --> B[Simulated labelled pairs]
    C[Your own labelled dataset] --> D[Validate data]
    B --> D
    D --> E[LoRACM fine-tuning]
    D --> M[CM training from scratch]
    F[Pretrained consistency model] --> E
    E --> G[Adapted model]
    H[Your testing TACs + input functions] --> I[Posterior prediction]
    G --> I
    M --> I
    I --> J[Parameter samples + summaries]
    H --> K[Nonlinear least squares]
    J --> L[Single-TAC comparison]
    K --> L
    J --> N[Parameter and 95% interval-width images]
    N --> O[Click voxel: TAC + AIF + posterior]
```

## What you can do

| Feature | Guide |
| :--- | :--- |
| Choose 2TC or define a custom single-input ODE with a different parameter set | [Kinetic models and transfer limits](LoRACM/docs/MODELS_AND_PROTOCOLS.md#kinetic-models) |
| Use measured AIFs, fit a cohort with Feng, and simulate new inputs | [Input functions](LoRACM/docs/MODELS_AND_PROTOCOLS.md#input-functions) |
| Set isotope half-life/decay, frame midpoints, durations, noise, and observation convention | [Acquisition protocol](LoRACM/docs/MODELS_AND_PROTOCOLS.md#acquisition-protocol) |
| Check your labelled training/validation/test file | [Dataset preparation](LoRACM/docs/PREPARE_DATA.md) |
| Select dataset size, LoRA rank/alpha, and training settings | [Fine-tuning](LoRACM/docs/FINE_TUNING.md) |
| Train a new CM without foundation weights and compare it with LoRACM | [Scratch training and paired evaluation](LoRACM/docs/TRAIN_CM.md) |
| Configure epochs, learning rate, embedding width, architecture and automatic data dimensions | [Shared training settings](LoRACM/docs/TRAIN_CM.md#defaults-and-settings) |
| Follow losses and export an adapted model | [Training progress and outputs](LoRACM/docs/FINE_TUNING.md#progress-and-outputs) |
| Predict batches or inspect one TAC, its posterior samples, and dashed NLS estimates | [Prediction manual](LoRACM/docs/PREDICTION.md) |
| Display Kᵢ or any predicted parameter, with 95% credible-interval-width maps and clickable voxel inspection | [Parametric imaging](LoRACM/docs/PARAMETRIC_IMAGING.md) |
| Reproduce the original first-paper examples | [Legacy page](docs/legacy-framework.md) · [Single TAC](Single_TAC_Demo.ipynb) · [Total-body imaging](Total_Body_Parametric_Imaging_Demo.ipynb) |

## Original framework and LoRACM

The first-paper framework remains accessible at **[docs/legacy-framework.md](docs/legacy-framework.md)**. Original source, notebooks, sample data, and pretrained assets retain their paths, so a paper link to this repository still leads to the original material. Existing published GitHub release archives are unchanged by this update.

LoRACM's included research example adapts **FDG–2TCM from the Siemens Biograph Vision Quadra protocol to the UC Davis uEXPLORER protocol**. Its preserved settings (rank 4, alpha 8) are distinct from new-project defaults. Custom models are supported through a documented plugin contract; transferring compatible weights to a new model or tracer still requires application-specific validation.

## Repository map

```text
PET-TRACER/
├── README.md                       Project overview and manual index
├── launch_gui.py                   Cross-platform local GUI launcher
├── Launch PET-TRACER.command        macOS launcher
├── Launch PET-TRACER.cmd            Windows launcher
├── launch_gui.sh                    Linux/macOS launcher
├── docs/legacy-framework.md        Original framework page
├── Source/, Pretrained/, Sample_data/, Assets/
├── Single_TAC_Demo.ipynb, Total_Body_Parametric_Imaging_Demo.ipynb
└── LoRACM/
    ├── README.md                   Step-by-step user manual
    ├── gui/                        Guided local interface
    ├── configs/, examples/         Editable configs, schedule and custom model
    ├── protocol.py                 Dataset/protocol validation
    ├── fit_feng.py, simulate_protocol.py
    ├── train.py, train_cm.py        LoRACM adaptation and scratch CM training
    ├── infer.py, compare_models.py  Shared prediction and paired comparison
    ├── analyse_tac.py, plot_posteriors.py
    ├── ucdavis.py                   Default numbered demonstration settings
    ├── parametric_imaging.py, inspect_voxel.py
    ├── create_dataset.py            Preserved UC Davis research generator
    ├── data/, checkpoints/, notebooks/
    ├── docs/, tests/
    └── workspaces/                 Your projects and outputs (Git-ignored)
```

## Research, support, and licence

- **Original paper:** *Generative Consistency Models for Estimation of Kinetic Parametric Image Posteriors in Total-Body PET*. [IEEE Transactions on Medical Imaging](https://ieeexplore.ieee.org/abstract/document/11517565).
- **LoRACM:** *A General-Purpose Adaptable Foundation Model for Total-Body PET Kinetic Modelling*. The repository describes the manuscript as in preparation; public citation metadata will be added when available.

Please cite the original framework and the adaptation paper when its public reference is available. For help, [open an issue](https://github.com/yundumbledore/PET-TRACER/issues) or contact **yun.zhao@sydney.edu.au** / **steven.meikle@sydney.edu.au**.

Software: [MIT licence](LICENSE). Assets: [data and model provenance](LoRACM/docs/DATA_PROVENANCE.md). Checks performed and their limits: [validation report](LoRACM/docs/VALIDATION.md). Historical file inventory: [release manifest](MANIFEST.json), whose original hashes describe the earlier archive rather than this working-tree update.
