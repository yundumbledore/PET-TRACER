"""Fine-tune the supplied consistency model using scenario configuration."""
import argparse
from pathlib import Path
import os, json, time, copy
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset import myDataset
from model_lora import build_lora_unet
from common import load_config, validate_dataset

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def main(cfg):
    if (Path(cfg["save_dir"]) / cfg["exp_name"]).exists():
        raise FileExistsError("Run directory already exists; choose a new exp_name or move the old run")
    # Repro
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = get_device() if cfg["device"] == "auto" else torch.device(cfg["device"])
    print("[train_lora_from_config] Device:", device)

    data = np.load(cfg['dataset_dir'], allow_pickle=False)
    validate_dataset(data, cfg['x_dim'], cfg['y_dim'])
    if cfg['batch_size'] > len(data['x_train']):
        raise ValueError('batch_size exceeds training set; reduce it to avoid an empty loader')
    if cfg['epochs'] < 1:
        raise ValueError('epochs must be positive')
    x_data = data['x_train'][:,:cfg['x_dim']]
    y_data = data['y_train']

    x_mean = np.mean(x_data, axis=0)
    x_std  = np.std(x_data, axis=0)
    y_mean = np.mean(y_data, axis=(0,1))
    y_std  = np.std(y_data, axis=(0,1))
    if np.any(x_std <= 0) or not y_std > 0:
        raise ValueError("Training data must have nonzero standard deviations")

    train_set = myDataset(data['x_train'][:,:cfg['x_dim']], 
                          data['y_train'],
                          x_mean=x_mean, 
                          x_std=x_std, 
                          y_mean=y_mean, 
                          y_std=y_std,
                          log_transform=False)
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    val_set = myDataset(data['x_val'][:,:cfg['x_dim']], 
                        data['y_val'],
                        x_mean=x_mean, 
                        x_std=x_std, 
                        y_mean=y_mean, 
                        y_std=y_std,
                        log_transform=False)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)

    save_dir = os.path.join(cfg["save_dir"], cfg["exp_name"])
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(save_dir, "tb"))

    with open(os.path.join(save_dir, "scaling_params.json"), "w") as f:
        json.dump({"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                   "y_mean": y_mean.tolist(), "y_std": y_std.tolist(), "t_meas": data["t_meas"].tolist()}, f)
        
    with open(os.path.join(save_dir, "model_params.json"), "w") as f:
        json.dump(cfg, f)

    model = build_lora_unet(
        x_dim=cfg['x_dim'],
        y_dim=cfg['y_dim'],
        embed_dim=cfg["embed_dim"],
        channels=cfg["channels"],
        embedy=cfg["embedy"],
        sigma_data=cfg["sigma_data"],
        r=cfg["lora_r"],
        alpha=cfg["lora_alpha"],
        dropout=cfg["lora_dropout"],
        include_names=None,
        exclude_names=("decodex",),
        train_norms=cfg["train_norms"],
        pretrained_ckpt=cfg["pretrained"],
        device=device,
    )
    # Create EMA target model
    target_model = copy.deepcopy(model)
    target_model.eval()
    for param in target_model.parameters():
        param.requires_grad = False
    
    # Define the EMA update function
    def update_ema(student_model, teacher_model, mu=0.99):
        with torch.no_grad():
            for (s_name, s_param), (t_name, t_param) in zip(student_model.named_parameters(), teacher_model.named_parameters()):
                # Only update the parameters that the student is actively training
                if s_param.requires_grad:
                    t_param.data.mul_(mu).add_(s_param.data, alpha=1 - mu)

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # We want the metric (validation loss) to minimize
        factor=cfg['lr_factor'],      # Multiply the learning rate by 
        patience=cfg['lr_scheduler_patience'],     # Wait for X epochs of no improvement before dropping LR
        min_lr=1e-8      # Don't let the learning rate drop below this
    )

    loss_weights = torch.tensor(cfg["loss_weights"], dtype=torch.float32, device=device)
    loss_weights = loss_weights.view(1, 1, -1)

    Nk = 15
    best = float("inf")
    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        run_loss = 0.0
        for i, (y, x) in enumerate(train_loader):
            y = y.to(device)
            x = x.unsqueeze(1).float().to(device)

            z = torch.randn_like(x)
            n = torch.randint(0, Nk, (x.size(0),), device=device)
            t_n = n / Nk
            t_n_plus_1 = (n + 1) / Nk
            x_noisy_n        = x + t_n.unsqueeze(1).unsqueeze(2) * z
            x_noisy_n_plus_1 = x + t_n_plus_1.unsqueeze(1).unsqueeze(2) * z

            pred_n_plus_1 = model(x_noisy_n_plus_1, y, t_n_plus_1)
            with torch.no_grad():
                # USE THE TARGET MODEL HERE
                target_n = target_model(x_noisy_n, y, t_n)
            
            # Apply the weighted penalty
            loss = torch.mean(loss_weights * (pred_n_plus_1 - target_n) ** 2)
            # loss = torch.mean((pred_n_plus_1 - target_n) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the EMA model weights
            update_ema(model, target_model, mu=0.99)
            run_loss += loss.item()

        avg_loss = run_loss / (i + 1)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if val_loader is not None:
            target_model.eval() 
            v_run = 0.0
            with torch.no_grad():
                for j, (vy, vx) in enumerate(val_loader):
                    vy = vy.to(device)
                    vx = vx.unsqueeze(1).float().to(device)
                    vz = torch.randn_like(vx)
                    vn = torch.randint(0, Nk, (vx.size(0),), device=device)
                    vt_n = vn / Nk
                    vt_n_plus_1 = (vn + 1) / Nk
                    vx_noisy_n        = vx + vt_n.unsqueeze(1).unsqueeze(2) * vz
                    vx_noisy_n_plus_1 = vx + vt_n_plus_1.unsqueeze(1).unsqueeze(2) * vz
                    
                    # Both predictions come from the EMA model to test its self-consistency
                    vpred_n_plus_1 = target_model(vx_noisy_n_plus_1, vy, vt_n_plus_1)
                    vtarget_n      = target_model(vx_noisy_n, vy, vt_n)
                    
                    # Use weighted loss
                    vloss = torch.mean(loss_weights * (vpred_n_plus_1 - vtarget_n) ** 2)
                    # vloss = torch.mean((vpred_n_plus_1 - vtarget_n) ** 2)
                    v_run += vloss.item()

            avg_v = v_run / (j + 1)
            writer.add_scalar("Loss/val", avg_v, epoch)

            scheduler.step(avg_v)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)
            print(f"Epoch {epoch:04d} | LR: {current_lr:.6f} | train {avg_loss:.6f} | val {avg_v:.6f}")
            # print(f"Epoch {epoch:04d} | train {avg_loss:.6f} | val {avg_v:.6f}")
        else:
            print(f"Epoch {epoch:04d} | train {avg_loss:.6f}")

        if avg_v < best and epoch >= cfg["checkpoint_start_epoch"]:
            best = avg_v
            torch.save(
                {
                    "model_state_dict": target_model.state_dict(),
                    "lr": cfg["lr"],
                    "lora_r": cfg["lora_r"],
                    "lora_alpha": cfg["lora_alpha"],
                    "lora_dropout": cfg["lora_dropout"],
                    "channels": cfg["channels"],
                    "embed_dim": cfg["embed_dim"],
                    "embedy": cfg["embedy"],
                    "x_dim": cfg["x_dim"],
                    "y_dim": cfg["y_dim"]
                },
                os.path.join(save_dir, "best.pth"),
            )
        

    torch.save(
                    {
                        "model_state_dict": target_model.state_dict(),
                        "lr": cfg["lr"],
                        "lora_r": cfg["lora_r"],
                        "lora_alpha": cfg["lora_alpha"],
                        "lora_dropout": cfg["lora_dropout"],
                        "channels": cfg["channels"],
                        "embed_dim": cfg["embed_dim"],
                        "embedy": cfg["embedy"],
                        "x_dim": cfg["x_dim"],
                        "y_dim": cfg["y_dim"]
                    },
                    os.path.join(save_dir, "last.pth"),
                )
    
    writer.close()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Execution time: {elapsed:.3f} seconds...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_scenario1.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for key in ("pretrained", "dataset_dir", "save_dir"):
        cfg[key] = str(Path(cfg[key]).resolve())
    main(cfg)
