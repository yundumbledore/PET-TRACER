# Install PET-TRACER

[Manual](../README.md) · [Project overview](../../README.md)

Use **64-bit Python 3.11** for the documented workflow. Clone/download the repository with its data and checkpoints, open a terminal in `PET-TRACER/`, then create an isolated environment:

```bash
python -m venv .venv
```

On systems where Python is named `python3`, use `python3 -m venv .venv`. On Windows, `py -3.11 -m venv .venv` also selects an installed Python 3.11 interpreter.

| Operating system | Activate |
| :--- | :--- |
| macOS / Linux | `source .venv/bin/activate` |
| Windows Command Prompt | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |

If PowerShell blocks activation, use Command Prompt or call `.venv\Scripts\python.exe` directly; changing a machine-wide execution policy is unnecessary.

Install the PyTorch build appropriate for your hardware using the [official selector](https://pytorch.org/get-started/locally/), then install the workflow dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r LoRACM/requirements-gui.txt
```

Terminal-only users can install `LoRACM/requirements.txt` instead. The GUI adds Streamlit; both interfaces use the same simulation, fitting, training and prediction code. This is a dependency specification, not a frozen research environment. The original `environment_no_builds.yml` remains available for the legacy notebooks, with its original platform-specific package history.

## Launch the GUI

From the root:

```bash
python launch_gui.py
```

The launcher selects the root `.venv` when present and serves the app on **127.0.0.1**. Open the local URL printed in the terminal if the browser does not open automatically. Files and computation stay on your computer. Source-file auto-reloading is disabled for a stable desktop workflow; restart the launcher after code updates. See [Streamlit's run guide](https://docs.streamlit.io/develop/concepts/architecture/run-your-app).

After the initial environment installation:

- **macOS:** double-click `Launch PET-TRACER.command`, or run `sh launch_gui.sh`.
- **Windows:** double-click `Launch PET-TRACER.cmd`.
- **Linux:** run `sh launch_gui.sh` from the repository.

These are launch scripts, not self-contained `.app`, `.exe` or package installers. Native installers would require per-platform packaging, distribution and signing. The local browser interface provides the common GUI now.

## Hardware and platform support

macOS support targets **Apple M-series chips**. Use PyTorch 2.6 or later for the active workflow.

| Platform | Common path | Optional acceleration |
| :--- | :--- | :--- |
| Windows x86-64 | CPU PyTorch + NumPy/SciPy workflow | Supported CUDA PyTorch build |
| macOS Apple Silicon | CPU workflow | `--device mps` for PyTorch |
| Linux | CPU workflow | Supported CUDA PyTorch build |

Simulation in `simulate_protocol.py` and NLS use NumPy/SciPy on CPU. Training and prediction `auto` select CUDA, then MPS, then CPU. Prediction caps the expanded MPS batch and checks identical model evaluations against CPU before generating results. Available operations and memory vary by hardware; use CPU if your device cannot run the model. Multi-GPU prediction uses PyTorch DataParallel on one node. See [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

### Windows NVIDIA setup and diagnostics

For an RTX 50-series GPU, choose a current CUDA PyTorch build supporting Blackwell with the [official installation selector](https://pytorch.org/get-started/locally/). Older CUDA builds can lack kernels for these cards; current packaging guidance is described in the [PyTorch release notes](https://pytorch.org/blog/pytorch-2-12-release-blog/). The CUDA version embedded in the PyTorch wheel, the GPU architecture and the NVIDIA driver must be compatible; merely selecting `cuda:0` does not install GPU support.

On either training page, select `cuda:0` and choose **Check compute setup**. Its report includes the actual Python interpreter, PyTorch version, CUDA build, GPU and a kernel/gradient test. The terminal equivalent, from `LoRACM/`, is:

```bash
python check_environment.py --device cuda:0 --output environment-check.json
```

If a job fails, use **Download job log** and **Download compute report**. Startup details are written before scientific imports, so an early process failure still leaves diagnostics. Out-of-memory errors call for a smaller training batch or inference batch/chunk; try a training batch of 64. Brief file-lock errors are retried. A permanent permission error includes the exact path: keep the checkout/project in a writable personal folder and check whether another application has locked the file.

## Optional legacy research generator

The preserved `create_dataset.py` uses JAX; install it separately:

```bash
python -m pip install -r LoRACM/requirements-legacy.txt
```

Current upstream JAX CPU wheels cover Linux, Apple Silicon macOS, and experimental Windows x86-64; Consult the [JAX installation page](https://docs.jax.dev/en/latest/installation.html) for supported versions/builds. The portable simulator, supplied-data GUI demo, fine-tuning and prediction do not require JAX.

## Check installation

```bash
cd LoRACM
python -m unittest discover -s tests -v
python smoke_workflow.py --output workspaces/installation_check
# Optional: check the scratch training route.
python smoke_workflow.py --output workspaces/cm_installation_check --from-scratch
```

The portable smoke workflow generates a tiny synthetic dataset, transfers the supplied foundation weights, trains one CPU epoch, predicts a held-out TAC, fits NLS and saves a posterior figure. It checks interfaces, not scientific performance. Choose a fresh output folder each time.

For the supplied UC Davis data, follow [the demonstration](UCDAVIS.md). Local validation details and the Windows/Apple Silicon macOS/Linux CI matrix are described in [VALIDATION.md](VALIDATION.md).
