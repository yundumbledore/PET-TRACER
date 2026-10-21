# PET-TRACER

**Bayesian kinetic modelling for dynamic total-body PET**

PET-TRACER uses consistency models (CMs) to estimate kinetic parameter posteriors, create parametric images, and perform Bayesian kinetic model selection from dynamic PET time–activity curves (TACs) and input functions.

The toolbox provides a local browser GUI and terminal workflows for Windows, Linux, and Apple Silicon macOS. Computation supports NVIDIA CUDA and Apple MPS, with CPU as a fallback.

## Three functions

1. **Train a new CM for kinetic posterior estimation.** Define your kinetic model, simulate or upload training data, and train a CM to estimate parameter posteriors and generate parametric images.
2. **Adapt a pre-trained CM to a new scenario with LoRA.** Use LoRACM to fine-tune a pre-trained CM for a new study or acquisition protocol.
3. **Use CM for Bayesian kinetic model selection.** Explore model probabilities, model-conditional parameter posteriors, and forward-simulated TACs with credible bands. The current release includes a pre-trained seven-model FDG demonstration for P001 coronal slice 116; model-selection training is planned for a future release.

## Get started

Follow the [installation guide](LoRACM/docs/INSTALLATION.md), then launch the GUI from the repository root:

```bash
python launch_gui.py
```

[User manual and terminal workflows](LoRACM/README.md) · [GUI guide](LoRACM/docs/WORKBENCH.md) · [Model-selection guide](LoRACM/docs/MODEL_SELECTION.md) · [Original framework](docs/legacy-framework.md)

## Research, support, and licence

- **CM:** *Generative Consistency Models for Estimation of Kinetic Parametric Image Posteriors in Total-Body PET*. [IEEE Transactions on Medical Imaging](https://ieeexplore.ieee.org/abstract/document/11517565).
- **LoRACM:** *A General-Purpose Adaptable Foundation Model for Total-Body PET Kinetic Modelling*. Manuscript in preparation; public citation details will be added when available.

Please cite the original framework and the adaptation paper when its public reference is available.

**Team:** Yun Zhao and Steven Meikle (The University of Sydney), Andrew Reader (King’s College London), and Yanan Fan (CSIRO’s Data61).

For support, [open an issue](https://github.com/yundumbledore/PET-TRACER/issues) or contact [yun.zhao@sydney.edu.au](mailto:yun.zhao@sydney.edu.au) / [steven.meikle@sydney.edu.au](mailto:steven.meikle@sydney.edu.au).

Software: [MIT licence](LICENSE). See [data and model provenance](LoRACM/docs/DATA_PROVENANCE.md) for the bundled assets.
