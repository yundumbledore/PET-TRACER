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
| Full Scenario 1 | 50,000 / 100,000 / 100 | 5,000 | Research-sized starting settings |
| Quick installation check | 32 / 16 / 8 | 2 | Learn the interface with a short practice run |

UC Davis uses rank 4 and alpha 8 at either scale. My own project starts with 100,000 / 10,000 / 1,000 pairs, rank 8 and alpha 16. Once you begin configuring the workflow, defaults are fixed for that project; all individual settings remain editable on their relevant pages. The tiny practice model is for checking software, not interpreting patient parameters.

## Measured input functions

The UC Davis setup uses the original `data/aif/UCDavis_1H35.csv`. Step 1 previews its original samples and exposes both endpoint assumptions: zero activity at the origin, and holding the last measured value after the final sample. UC Davis enables both by default. Other projects require measured coverage after the last sample unless you explicitly choose **Hold the last measured AIF value**.

The final AIF sample is at 57.48521667 minutes; the Sub001 schedule's last midpoint is at 57.5 minutes. Frame averaging instead requires coverage through 60 minutes. A coverage warning explains the missing interval before you save the protocol or launch simulation. See [input functions and endpoint policies](MODELS_AND_PROTOCOLS.md#use-measured-inputs) for terminal settings. Existing projects retain their earlier options: check the zero-origin option, choose the final-value hold and save step 1 before rerunning step 2.

## Training and early stopping

The monitor shows dynamic batch progress, train/validation loss, current learning rate and separate loss/rate graphs. **Stop active job** requests a graceful stop. Training publishes `best.pth` and `last.pth` after every completed epoch. The first contains the lowest validation loss across completed epochs; the second contains the latest completed epoch. An unfinished training epoch is discarded when stopping. At least one completed epoch is required.

In **Advanced settings**, **Save an epoch checkpoint every** defaults to 25. Epoch 1 is always retained. Set this to 1 if you need every epoch, allowing enough disk space (about 44 GB for 5,000 snapshots of the full-width model). The checkpoint selector displays the epoch and validation loss for best, last and retained snapshots. These are EMA inference checkpoints; restarting training from an exact optimizer/RNG state is not implemented.

## Optional scratch CM comparison

Enable **CM from scratch comparison** in the sidebar to show its training page and batch comparison page. Train it on the same labelled dataset for a controlled comparison. In step 5, enable **Overlay CM from scratch posterior** while using the LoRACM run to analyse the same voxel with both estimators. In steps 4 and 6, select either trained model. Each anatomical map needs predictions from that selected model.

## Prediction and plots

Batch prediction and single-TAC analysis expose posterior sample count, reverse sampling steps, sample chunk, seed and device. `auto` chooses CUDA, then Apple MPS, then CPU. The inference metadata records the actual device and settings. The current default is three consistency-sampling evaluations; changing the step count is an experiment, not a guarantee of improved accuracy.

Step 5 shows parameter posteriors with dashed NLS estimates and a TAC graph containing the measured TAC, input function, NLS fitted TAC and forward simulation from the posterior median. The optional scratch model adds its own posterior and forward curve. Starting a new voxel analysis clears the active old result; **Open an earlier completed analysis** lets you deliberately retrieve saved results.

Step 6 places the map on the left and the clicked voxel’s TAC/AIF and posterior density on the right. Choose a kinetic parameter, Ki (built-in 2TC), or a 95% credible-interval width. Older predictions made with oversized MPS batches are flagged; regenerate them using the [current MPS safeguards](PREDICTION.md#apple-mps-batch-safety).

Dataset creation, fitting, training and prediction use dynamic progress bars outside the collapsible log. Logs hold diagnostics and file locations, so you do not need to scroll through printed progress updates.
