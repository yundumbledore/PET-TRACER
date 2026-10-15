# Using the local workbench

[User manual](../README.md) · [UC Davis walkthrough](UCDAVIS.md) · [Prediction](PREDICTION.md)

The local browser GUI and terminal tools share the same datasets, configurations, checkpoints and prediction files. The GUI runs on Windows, Linux and Apple M-series macOS.

## Your project owns your settings

Enter a **Project name** to create or reopen an independent workspace. Model, protocol, schedule, dataset, training and inference controls are retained when you move between any pages, including skipping steps or returning to an earlier step. Unfinished edits are saved in `workspaces/<project>/gui_state.json`, so reopening the browser restores them. Uploaded files are copied into the project’s `inputs/` folder.

**Save and validate protocol**, **Validate and select dataset**, and training start still perform the scientific/schema checks and publish the configuration used by computation. An autosaved draft is not yet a validated protocol. Changing a protocol after creating a dataset does not regenerate that dataset or alter a trained model; create a new dataset/run when needed.

Use a new project name for independent settings. Returning to an existing name reopens its settings and outputs. A different run name creates another training experiment inside the same project. Jobs remain associated with the project that launched them; navigation does not restart them.

## What are project defaults?

**Example setup** replaces the former “Starting template” label. It provides initial values so you can learn with the supplied **UC Davis Scenario 1**, or begin **My own project** with generic settings. It does not choose a different neural network architecture or overwrite your edits.

**Walkthrough scale** replaces “UC Davis run size”. It chooses initial dataset and training sizes:

| Scale | Training / validation / test pairs | Training epochs | Purpose |
| :--- | :--- | :--- | :--- |
| Full Scenario 1 | 100,000 / 100,000 / 100 | 5,000 | Research-sized starting settings |
| Quick installation check | 32 / 16 / 8 | 2 | Learn the interface with a short practice run |

UC Davis uses rank 4 and alpha 8 at either scale. My own project starts with 100,000 / 10,000 / 1,000 pairs, rank 8 and alpha 16. Once you begin configuring the workflow, defaults are fixed for that project; all individual settings remain editable on their relevant pages. The tiny practice model is for checking software, not interpreting patient parameters.

## Measured input functions

Step 1 uses midpoint observations. Review or change the model, schedule, input functions, priors and isotope half-life, then save. The default `data/aif/UCDavis_1H35.csv` and matching Sub001 schedule preserve the original activity samples in frame order. Custom schedules interpolate measured inputs to the selected midpoints; Feng inputs and custom kinetic models remain available. [Technical details](MODELS_AND_PROTOCOLS.md).

The model selectors in steps 4–6 include **Paper LoRACM · UC Davis · epoch 2755** without requiring a dataset or training. Step 3 also has **Use supplied paper checkpoint**. The original weights and normalization are copied into your project's `runs/paper_ucdavis`; fine-tuning creates a separate run. The **UC Davis demonstration** page predicts and exports a paper-style Ki map in one job.

## Training and early stopping

The monitor shows dynamic batch progress, train/validation loss, current learning rate and separate loss/rate graphs. **Stop active job** requests a graceful stop. Training publishes `best.pth` and `last.pth` after every completed epoch. The first contains the lowest validation loss across completed epochs; the second contains the latest completed epoch. An unfinished training epoch is discarded when stopping. At least one completed epoch is required.

In **Advanced settings**, **Save an epoch checkpoint every** defaults to 25. Epoch 1 is always retained. Set this to 1 if you need every epoch, allowing enough disk space (about 44 GB for 5,000 snapshots of the full-width model). The checkpoint selector displays the epoch and validation loss for best, last and retained snapshots. These are EMA inference checkpoints; restarting training from an exact optimizer/RNG state is not implemented.

## Optional scratch CM comparison

Enable **CM from scratch comparison** to show **CM · Dataset**, **CM from scratch**, **CM · Predict**, and **Compare models**. Scratch dataset counts, source selection and simulation protocol are independent of LoRACM. Its starting dataset size is 1,000,000 / 100,000 / 1,000 pairs, or 32 / 16 / 8 for the quick check. Adjust these values, or explicitly import the same NPZ for a comparison at equal target-data size.

**4 · Predict** uses LoRACM; **CM · Predict** uses scratch CM. Each accepts another patient or testing file. In step 5, enable **Overlay CM from scratch posterior** for a paired voxel comparison. Step 6 can image either model's own predictions.

## Prediction and plots

Repeated predictions keep earlier files: `predictions.npz`, `predictions_002.npz`, `predictions_003.npz`, etc. **Open an earlier prediction** restores a completed output and its recorded patient/geometry. The active display clears while a new request is running.

Choose **My own patient scan** to load a coronal TAC slice plus its corresponding binary NumPy mask. Verify C-order foreground mapping, or supply an explicit voxel mapping CSV; choose the display origin and pixel aspect ratio. The app checks one TAC per foreground pixel and carries patient-specific geometry into step 6. A source checksum prevents a map from using another patient's mask even when voxel IDs match. Generic curve files without anatomy use **My own file**.

Batch prediction and single-TAC analysis expose posterior sample count, reverse sampling steps, sample chunk, seed and device. `auto` chooses CUDA, then Apple MPS, then CPU. The inference metadata records the actual device and settings. The current default is three consistency-sampling evaluations; changing the step count is an experiment, not a guarantee of improved accuracy.

Step 5 shows parameter posteriors with dashed NLS estimates and a TAC graph containing the measured TAC, input function, NLS fitted TAC and forward simulation from the posterior median. The optional scratch model adds its own posterior and forward curve. Starting a new voxel analysis clears the active old result; **Open an earlier completed analysis** lets you deliberately retrieve saved results.

Step 6 places the map on the left and the clicked voxel’s TAC/AIF and posterior density on the right. Choose a kinetic parameter, Ki (built-in 2TC), or a 95% credible-interval width. Older predictions made with oversized MPS batches are flagged; regenerate them using the [current MPS safeguards](PREDICTION.md#apple-mps-batch-safety).

Dataset creation, fitting, training and prediction use dynamic progress bars outside the collapsible log. Logs hold diagnostics and file locations, so you do not need to scroll through printed progress updates.

## Compute checks and failed jobs

Training pages provide **Check compute setup** and a downloadable report with the Python interpreter, PyTorch version, CUDA build and actual device-kernel test. Every job log contains startup details before scientific imports, captures Python tracebacks and has **Download job log**. A process that exits without a traceback still leaves its command, interpreter and exit code for diagnosis. Project/progress writes retry short file-lock errors; permanent permission errors identify the path and retain the previous settings file. A locked progress-display file is reported in the log and does not abort computation.
