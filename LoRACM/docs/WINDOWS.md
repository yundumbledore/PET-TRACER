# Windows installation

[Project overview](../../README.md) · [User manual](../README.md) · [All platforms](INSTALLATION.md)

Use a **64-bit Intel/AMD Windows PC**. The recommended installer uses a simple numbered menu; you do not need to type PowerShell commands or activate an environment.

## Double-click setup

1. Download PET-TRACER and **extract the complete ZIP**.
2. Close any running PET-TRACER app. Open the extracted folder and double-click **[`Setup PET-TRACER.cmd`](../../Setup%20PET-TRACER.cmd)**. Keep it beside `setup_environment.py` and `setup_windows.ps1`.
3. Choose **1 · NVIDIA GPU** (including RTX 5060) or **2 · CPU**. Choose **Q** to cancel.
4. Keep the window open while setup prepares Python 3.11 if needed, installs the packages and checks the selected device. An internet connection is required.
5. When **SETUP COMPLETE** appears, close the setup window and double-click **`Launch PET-TRACER.cmd`**.

The installer creates or updates this reusable environment:

```text
C:\Users\YOUR_WINDOWS_USERNAME\.venvs\pet-tracer
```

It uses the official Python Install Manager when Python is missing, installing it through WinGet or Python's official App Installer fallback. [Python's Windows documentation](https://docs.python.org/3/using/windows.html) describes these installation methods. Python 3.11 must stay installed because the environment depends on it. Dataset generation uses CPU JAX at either setup choice; NVIDIA setup installs PyTorch from its CUDA 12.8 channel for training and prediction.

Setup keeps your projects, datasets and model files. It saves a timestamped log and, after the device check, a compute report under:

```text
%LOCALAPPDATA%\PET-TRACER\setup-logs
```

If **SETUP STOPPED** appears, read the error in the open window and keep that log. For a GPU check failure, check the NVIDIA driver/build or rerun setup and choose CPU. A managed PC that blocks Python installation may require the manual route below or help from its administrator.

## Future downloads

Extract the new repo and double-click **Launch PET-TRACER.cmd**. It reuses the same environment. Rerun **Setup PET-TRACER.cmd** only when installing dependencies for an update or changing CPU/GPU packages. Keep your previous `LoRACM\workspaces` when moving projects.

## Manual installation and troubleshooting

<details>
<summary>Show the PowerShell alternative, terminal usage and troubleshooting</summary>

The following commands are an alternative for users who prefer PowerShell or need a custom installation. The double-click installer above performs these steps automatically.

## 1. Check Python 3.11

Open **PowerShell** from the Start menu and run:

```powershell
py -3.11 --version
```

If this prints `Python 3.11.x`, keep that installation and check its bitness below. If the command fails or Python 3.11 is missing, install the **Python Install Manager** from [python.org](https://www.python.org/downloads/windows/), close and reopen PowerShell, then run:

```powershell
pymanager install 3.11
py -3.11 --version
```

Check that the selected Python is 64-bit:

```powershell
py -3.11 -c "import struct; print(struct.calcsize('P') * 8)"
```

This must print **`64`**. If it prints `32`, install the 64-bit Python 3.11 runtime before continuing.

`pymanager` avoids a name conflict with the older `py` launcher. See [Python's Windows instructions](https://docs.python.org/3/using/windows.html). Keep Python 3.11 installed: the reusable environment depends on it.

## 2. Open the downloaded repo in PowerShell

1. Download PET-TRACER and **extract the ZIP**.
2. Open the extracted folder containing **`setup_environment.py`** and **`Launch PET-TRACER.cmd`**. Do not work inside the ZIP or inside `LoRACM` yet.
3. Click File Explorer's address bar, type `powershell`, and press Enter. This opens PowerShell in that folder.
4. Confirm the setup file is present:

```powershell
Get-Item .\setup_environment.py
```

If it is not found, open the folder containing that file and repeat this step. Downloaded folders may be called `PET-TRACER-main` rather than `PET-TRACER`.

## 3. Install into the reusable environment

Choose **one** of the following commands. Run it from the repo folder opened in step 2.

### NVIDIA GPU, including RTX 5060

For your RTX 5060 PC, use:

```powershell
py -3.11 .\setup_environment.py --pytorch-index https://download.pytorch.org/whl/cu128
```

This selects PyTorch's CUDA 12.8 package channel before installing the remaining requirements. PyTorch introduced Blackwell support with its CUDA 12.8 builds; see the [official release information](https://pytorch.org/blog/pytorch-2-7/). The installed build and NVIDIA driver must support your card. The [PyTorch selector](https://pytorch.org/get-started/locally/) provides other supported CUDA choices; pass its `--index-url` address to `--pytorch-index`.

### CPU

For a PC without an NVIDIA GPU, or to use CPU computation:

```powershell
py -3.11 .\setup_environment.py
```

Wait for setup to finish and print **`Ready.`** and **`Python:`**. The default environment is:

```text
C:\Users\YOUR_WINDOWS_USERNAME\.venvs\pet-tracer
```

Setup creates this folder and installs the dependencies there, including CPU JAX for dataset generation. `YOUR_WINDOWS_USERNAME` is illustrative; the commands use your actual home folder automatically. The repo contains the code, while this external folder contains Python and its packages.

## 4. Verify the selected device

Still in the repo root, check the NVIDIA installation with:

```powershell
& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe" .\LoRACM\check_environment.py --device cuda:0 --output "$env:TEMP\pet-tracer-compute.json"
```

For CPU setup, use:

```powershell
& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe" .\LoRACM\check_environment.py --device cpu --output "$env:TEMP\pet-tracer-compute.json"
```

Look for **`"state": "passed"`**. The NVIDIA report should also show **`"cuda_available": true`** and your GPU. The check runs a calculation and gradient test. The leading `&` tells PowerShell to run the quoted Python path.

## 5. Open the GUI

Run from the repo root:

```powershell
& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe" .\launch_gui.py
```

Alternatively, double-click **`Launch PET-TRACER.cmd`** in File Explorer. Keep its terminal window open while using the app. Open the local URL printed there if your browser does not open automatically.

On training and prediction pages, **`auto`** selects an available CUDA GPU. You can select **`cuda:0`** explicitly for the NVIDIA card, or **`cpu`** for CPU setup. Choosing `cuda:0` requires the GPU installation above; it does not install CUDA support.

## Terminal use

From the repo root, enter `LoRACM` and call the same environment's Python:

```powershell
Set-Location .\LoRACM
& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe" .\reproduce_ucdavis.py --workspace workspaces\ucdavis_paper --device auto
```

For other commands in the manual, replace the initial `python` with `& "$env:USERPROFILE\.venvs\pet-tracer\Scripts\python.exe"`. This runs them in the correct environment without PowerShell activation scripts.

## Next download or dependency update

Extract the new repo and double-click its **`Launch PET-TRACER.cmd`**. It finds the same environment. If a release adds dependencies, rerun **Setup PET-TRACER.cmd** or repeat the manual setup command; setup reuses the existing environment. Keep your previous `LoRACM\workspaces` when moving projects to a new download.

## Common problems

| What you see | What to do |
| :--- | :--- |
| `py` or `pymanager` is not recognized | Run **Setup PET-TRACER.cmd** to prepare Python automatically, or install the Python Install Manager and reopen PowerShell. |
| `setup_environment.py` cannot be found | Repeat step 2 in the extracted repo root. |
| `Activate.ps1` is blocked | Use the direct Python commands above; activation is unnecessary. |
| `No module named ...` | Repeat step 3, then use the external environment's Python rather than a different system Python. |
| CUDA unavailable or an unsupported GPU kernel | Repeat the NVIDIA setup, check the driver/build against the official PyTorch selector, then rerun step 4. Retain the compute report and job log if it still fails. |
| `WinError 5: Access is denied` | Check the exact reported path. Use an extracted folder you can write to and close applications locking the file. |


</details>
