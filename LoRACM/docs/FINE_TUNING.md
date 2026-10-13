# Fine-tune the pretrained consistency model

[Manual](../README.md) · [Prediction](PREDICTION.md) · [CM from scratch and comparison](TRAIN_CM.md)

## Choose settings

| Setting | New-project default | Practical meaning |
| :--- | :--- | :--- |
| Training pairs | **100,000** | Number of labelled training examples; validation/test are additional |
| Validation / test pairs | 10,000 / 1,000 | Evaluate consistency loss and independent predictive performance |
| LoRA rank / alpha | **8 / 16** | Adapter capacity and scaling; scaling is alpha/rank |
| Dropout | 0.05 | LoRA dropout during training |
| Batch size | 512 | Reduce to fit memory; never exceed training-pair count |
| Epochs | 5,000 | Configurable research starting point; begin with a tiny smoke run |
| Learning rate | 0.0001 | Initial AdamW rate; validation-loss plateau scheduler reduces it |
| Embedding dimension | **1,024** | Keep the foundation width initially; changing it resets incompatible embedding/projection weights |
| Parameter / input dimensions | **Automatic** | `x_dim=P`, `y_dim=2T`, inferred and checked against the dataset |
| Checkpoint start | 1 | Best validation checkpoint can be saved from the first epoch |

These are starting values, not optimized recommendations for every tracer/protocol. Track convergence and posterior performance on held-out data. The preserved UC Davis research configs use **rank 4, alpha 8**, a different split size and best-checkpoint eligibility from epoch 151; do not confuse those with new-project defaults.

## GUI

Open **3 · Fine-tune**, choose a validated NPZ and foundation checkpoint, set the kinetic model and run name, and start. For simulation projects, retain the protocol JSON so the same model and bounds are available for single-TAC NLS. The main panel displays epochs, batch size, learning rate, device and LoRA rank/alpha. **Advanced settings** exposes embedding dimension, channels, conditioning encoder, noise intervals, EMA, weight decay, scheduler, seed, parameter loss weights and best-checkpoint eligibility. Use **Review configuration** before starting. Training/validation curves, epoch progress and a collapsible job log show progress.

## Terminal

Edit `configs/train_custom.json`, then:

```bash
python train.py --config configs/train_custom.json
# Optional CLI overrides (also supported by train_cm.py):
python train.py --config configs/train_custom.json --epochs 5000 --lr 0.0001 --batch-size 512 --embed-dim 1024
```

Set `dataset_dir`, `pretrained` and your output location. New-project configs default `x_dim`, `y_dim`, `parameter_names` and `loss_weights` to `"auto"`; the dataset resolves them. Explicit dimensions/names must match. The [shared settings table](TRAIN_CM.md#defaults-and-settings) explains every critical training control. For custom models, set `kinetic_model="custom"`. Optional `simulation_protocol` saves a matched copy of the forward-model protocol; for custom plugins, the model source is copied into the run folder.

Examples can also use GUI-generated configs:

```bash
python train.py --config workspaces/my_project/train.json
python -m tensorboard.main --logdir workspaces/my_project/runs
```

## Progress and outputs

Each run contains:

- `model_params.json`: resolved model/training config and parameter order.
- `scaling_params.json`: training-only scales, schedule, optional durations and observation convention.
- `transfer_report.json`: newly initialized/resized parameters and parameter-head reset decision.
- `training_summary.json`: dataset checksum, model mode, trainable parameter count, seed and elapsed training time.
- `metrics.csv`: epoch, training loss, validation loss, learning rate and elapsed seconds, updated each epoch.
- `tb/`: TensorBoard events.
- `best.pth`: best eligible EMA model; `last.pth`: final EMA model.
- Optional `protocol.json` and `kinetic_model.py`: matched forward model for NLS.

These are **full adapted inference checkpoints**, not LoRA-only files or complete optimizer/RNG states for exact resumption. Keep run configuration, scaling and checkpoint together. Changed parameter names/order or any custom model resets parameter heads; resized conditioning layers train when frame counts change. Review the transfer report.

Consistency training and validation self-consistency losses are not errors against independent parameter truth. Assess parameter RMSE/bias, posterior calibration and model fit on suitable held-out data. A test NPZ with optional `truth [N,P]` lets inference report RMSE and bias of posterior means.

## Interrupted jobs and memory limits

Use a new run name after a failed/stopped run; the program protects existing output folders. Stopping during training can leave incomplete outputs, and a final checkpoint may not yet exist. Lower batch size for training; lower curve batch size and sample chunk for prediction. `cpu` is the portable baseline; `auto` training prefers CUDA, then MPS, then CPU.

For a controlled scratch baseline, train the same target dataset with [train_cm.py](TRAIN_CM.md), then use the paired comparison module. Retain the foundation architecture for the initial comparison; changing embedding width/channel sizes alters both transferred weights and model capacity.
