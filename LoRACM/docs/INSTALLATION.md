# Install PET-TRACER once, reuse it across downloads

[Manual](../README.md) · [Project overview](../../README.md) · [Windows double-click setup](WINDOWS.md)

Use **64-bit Python 3.11**. The recommended environment lives in your home folder, outside every repository checkout:

- **macOS / Linux:** `~/.venvs/pet-tracer`
- **Windows:** `%USERPROFILE%\.venvs\pet-tracer`

Downloading a new PET-TRACER version does not remove or duplicate this environment. The launchers find it and run the code from the download you launched. Keep its base Python 3.11 installation available. This environment is reused on the same computer; it is not copied between operating systems. See the [Python virtual-environment documentation](https://docs.python.org/3/library/venv.html).

## First installation

**Windows users:** extract the ZIP and double-click **`Setup PET-TRACER.cmd`**. Choose NVIDIA GPU or CPU, wait for **SETUP COMPLETE**, then double-click **`Launch PET-TRACER.cmd`**. The installer prepares Python if needed, creates or updates the reusable environment and checks the selected device. [Windows setup and troubleshooting](WINDOWS.md).

On Apple Silicon macOS/Linux, open a terminal in the downloaded repository root:

```bash
python3.11 setup_environment.py
python3 launch_gui.py
```

Optional Windows PowerShell alternative, after installing Python 3.11 and opening PowerShell in the extracted repo root:

Choose **NVIDIA GPU** (including RTX 5060):

```powershell
py -3.11 .\setup_environment.py --pytorch-index https://download.pytorch.org/whl/cu128
```

Or choose **CPU**:

```powershell
py -3.11 .\setup_environment.py
```

Then launch using the environment that setup created:

```powershell
& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe" .\launch_gui.py
```

The shared Windows environment is `C:\Users\YOUR_WINDOWS_USERNAME\.venvs\pet-tracer`. Setup creates it automatically. No manual activation is needed; later launches can use **Launch PET-TRACER.cmd**. [Verify the CPU/GPU installation](WINDOWS.md#4-verify-the-selected-device) before training.

Setup creates or reuses the external environment, installs the GUI requirements and checks package dependencies. It records installed versions inside the environment. It never puts project source code inside the environment. For a terminal-only installation add `--terminal-only`.

**NVIDIA users:** choose a CUDA-enabled PyTorch build using the [official selector](https://pytorch.org/get-started/locally/). To install that build before other dependencies, run `setup_environment.py --no-install`, invoke the selector's PyTorch installation command using the environment's Python below, then rerun `setup_environment.py`. Alternatively pass its official wheel index with `--pytorch-index`. Selecting `cuda:0` inside the GUI does not install CUDA-enabled PyTorch. The normal package-index installation remains a CPU fallback where that is the available build.

## Next repo version: launch without reinstalling

Download/unzip the new version into its own folder and use its launcher:

- **macOS:** double-click `Launch PET-TRACER.command`, or run `sh launch_gui.sh`.
- **Windows:** double-click `Launch PET-TRACER.cmd`.
- **Linux:** run `sh launch_gui.sh`.

Or run `python launch_gui.py` from the new download. No activation or reinstall is needed. The launcher checks the selected environment and serves the app on **127.0.0.1**. Restart it after code changes. These are launch scripts for the local browser GUI, not standalone desktop installers.

If a release changes dependency requirements, rerun its setup command. It reuses the existing environment and installs dependencies needed by that release; it does not recreate the environment. Keep separate environments with `--prefix` if two releases require incompatible dependencies.

## Terminal commands in the reusable environment

Activation is optional. Use its Python directly from any downloaded `LoRACM` folder:

```bash
# macOS / Linux
~/.venvs/pet-tracer/bin/python reproduce_ucdavis.py --device auto
```

```powershell
# Windows PowerShell
& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe" reproduce_ucdavis.py --device auto
```

To activate, use `source ~/.venvs/pet-tracer/bin/activate` on macOS/Linux, `%USERPROFILE%\.venvs\pet-tracer\Scripts\activate.bat` in Command Prompt, or `& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\Activate.ps1"` in PowerShell. If PowerShell blocks activation, calling the environment's Python directly avoids changing execution policy.

## Existing environments and custom locations

You can continue using an activated Conda/venv environment that already has the requirements. The launcher checks, in order: explicit **PET_TRACER_PYTHON**, activated virtual/Conda environments, the shared home-folder environment, the checkout's older `.venv`, and the invoking Python. Missing/incomplete automatic candidates are skipped; an explicit Python selection is respected and reports missing dependencies.

Create another reusable environment with `python3.11 setup_environment.py --prefix /your/environment/path`. Activate it before launching, or set `PET_TRACER_PYTHON` to its full Python executable path. No global packages or administrator installation are needed.

The environment is separate from your **projects**. Retain/copy the previous `LoRACM/workspaces` when migrating experiments, or set `PET_TRACER_WORKSPACES` to a stable external project folder before launching. New downloads do not automatically relocate existing project files.

## Hardware and platform support

macOS support targets **Apple M-series chips**. Use PyTorch 2.6 or later for the active workflow.

| Platform | Common path | Optional acceleration |
| :--- | :--- | :--- |
| Windows x86-64 | CPU PyTorch + NumPy/SciPy workflow | Supported CUDA PyTorch build |
| macOS Apple Silicon | CPU workflow | `--device mps` for PyTorch |
| Linux | CPU workflow | Supported CUDA PyTorch build |

Simulation and NLS run on CPU. Dataset generation uses CPU JAX for the original random draws and NumPy for bounded-memory integration; JAX is included in the standard requirements and needs no GPU add-on. Training and prediction `auto` select CUDA, then MPS, then CPU. Prediction caps the expanded MPS batch and checks identical model evaluations against CPU before generating results. Available operations and memory vary by hardware; use CPU if your device cannot run the model. Multi-GPU prediction uses PyTorch DataParallel on one node. See [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

### Windows NVIDIA setup and diagnostics

The [Windows installer](WINDOWS.md#double-click-setup) includes the RTX 5060 setup path and runs a device calculation/gradient check. For another build, use the [official PyTorch selector](https://pytorch.org/get-started/locally/). The PyTorch CUDA build, GPU architecture and NVIDIA driver must be compatible; selecting `cuda:0` inside the app does not install GPU support.

On either training page, select `cuda:0` and choose **Check compute setup**. Its report includes the actual Python interpreter, PyTorch version, CUDA build, GPU and a kernel/gradient test. The terminal equivalent, from `LoRACM/`, is:

```bash
python check_environment.py --device cuda:0 --output environment-check.json
```

If a job fails, use **Download job log** and **Download compute report**. Startup details are written before scientific imports, so an early process failure still leaves diagnostics. Out-of-memory errors call for a smaller training batch or inference batch/chunk; try a training batch of 64. Brief file-lock errors are retried. A permanent permission error includes the exact path: keep the checkout/project in a writable personal folder and check whether another application has locked the file.

## Update an existing environment

CPU JAX is included to reproduce the original dataset generator’s random sampling. On Windows, double-click **Setup PET-TRACER.cmd** to update dependencies. On macOS/Linux, rerun `setup_environment.py` from the repository root using the command above. It installs missing dependencies in the same external environment; your projects and model files stay in place.

The preserved research generator also uses JAX. CPU wheels are available for Apple Silicon macOS, Linux and Windows x86-64; Windows wheel support is currently experimental. See the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for platform requirements. JAX is imported only for dataset generation; using an existing checkpoint for prediction does not require it.

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
