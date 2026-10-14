"""Fine-tune the supplied consistency model using scenario configuration."""
import argparse
from pathlib import Path
import os, json, time, copy, csv, shutil, hashlib
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from prepare_dataset import myDataset
from model_factory import build_model
from training_config import resolve_training_config
from common import load_config, validate_dataset, PARAMETERS
from protocol import validate_file, validate_names
from runtime import select_device, Progress, stop_requested, write_json_atomic
import signal

def get_device():
    return select_device('auto')

def main(cfg, training_mode="lora"):
    if not cfg.get('dataset_dir'):
        raise ValueError('Specify dataset_dir for your labelled NPZ file')
    report = validate_file(cfg['dataset_dir'])
    cfg = resolve_training_config(cfg, report, training_mode)
    if (Path(cfg["save_dir"]) / cfg["exp_name"]).exists():
        raise FileExistsError("Run directory already exists; choose a new exp_name or move the old run")
    names = cfg['parameter_names']
    protocol = None
    if cfg.get('simulation_protocol'):
        from protocol import validate_protocol, load_protocol
        protocol = load_protocol(cfg['simulation_protocol'])
        validate_protocol(protocol)
        if protocol['parameter_names'] != names or not np.allclose(protocol['t_meas'],report['t_meas'],atol=1e-6,rtol=0):
            raise ValueError('Simulation protocol names/times must match the training dataset')
        if report['frame_durations_min'] is not None and not np.allclose(protocol['frame_durations_min'],report['frame_durations_min'],atol=1e-6,rtol=0):
            raise ValueError('Simulation protocol durations must match the dataset')
        if protocol['model'] != cfg.get('kinetic_model', '2TC'):
            raise ValueError('Training kinetic_model must match the simulation protocol model')
        if 'observation' in report and report['observation'] != protocol.get('observation', 'midpoint'):
            raise ValueError('Simulation observation mode must match the dataset')
    # Repro
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = select_device(cfg["device"])
    print(f"[{training_mode}] Device: {device}; epochs: {cfg['epochs']}; embedding: {cfg['embed_dim']}; dimensions: {cfg['x_dim']} / {cfg['y_dim']}", flush=True)

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
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, drop_last=cfg["drop_last"],
                              generator=torch.Generator().manual_seed(cfg["seed"]))

    val_set = myDataset(data['x_val'][:,:cfg['x_dim']], 
                        data['y_val'],
                        x_mean=x_mean, 
                        x_std=x_std, 
                        y_mean=y_mean, 
                        y_std=y_std,
                        log_transform=False)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False)

    save_dir = os.path.join(cfg["save_dir"], cfg["exp_name"])
    os.makedirs(save_dir, exist_ok=False)
    if protocol is not None:
        protocol = dict(protocol)
        if protocol['model'] == 'custom':
            shutil.copyfile(protocol['model_file'], Path(save_dir, 'kinetic_model.py'))
            protocol['model_file'] = 'kinetic_model.py'
        Path(save_dir, 'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n', encoding='utf-8')
    writer = SummaryWriter(os.path.join(save_dir, "tb"))

    with open(os.path.join(save_dir, "scaling_params.json"), "w", encoding='utf-8') as f:
        json.dump({"x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                   "y_mean": y_mean.tolist(), "y_std": y_std.tolist(), "t_meas": data["t_meas"].tolist(),
                   "frame_durations_min": report['frame_durations_min'],
                   "observation": str(data['observation']) if 'observation' in data else 'midpoint',
                   "parameter_names": names}, f)
        
    with open(os.path.join(save_dir, "model_params.json"), "w", encoding='utf-8') as f:
        json.dump(cfg, f)

    model = build_model(cfg, device, training=True)
    total_parameters=sum(p.numel() for p in model.parameters())
    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_parameters:,} / {total_parameters:,}', flush=True)
    Path(save_dir, 'transfer_report.json').write_text(json.dumps(model.transfer_report, indent=2)+'\n', encoding='utf-8')
    metrics_path = Path(save_dir, 'metrics.csv')
    metrics_path.write_text('epoch,train_loss,val_loss,learning_rate,elapsed_seconds\n', encoding='utf-8')
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

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # We want the metric (validation loss) to minimize
        factor=cfg['lr_factor'],      # Multiply the learning rate by 
        patience=cfg['lr_scheduler_patience'],     # Wait for X epochs of no improvement before dropping LR
        min_lr=cfg["min_lr"]      # Don't let the learning rate drop below this
    )

    loss_weights = torch.tensor(cfg["loss_weights"], dtype=torch.float32, device=device)
    loss_weights = loss_weights.view(1, 1, -1)

    Nk = cfg["noise_levels"]
    best = float("inf")
    start_time = time.time()

    completed_epoch = 0
    checkpoint_index = {}
    interrupted = False
    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True
    old_handlers = {}
    # Main-thread CLI runs support Ctrl+C; GUI jobs use a cooperative stop file.
    import threading
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, request_stop)
    def cancelled(): return interrupted or stop_requested()
    progress = Progress('Training', cfg['epochs']*(len(train_loader)+len(val_loader)), 'batches')
    completed_batches = 0
    for epoch in range(1, cfg["epochs"] + 1):
        if cancelled(): break
        model.train()
        run_loss = 0.0
        for i, (y, x) in enumerate(train_loader):
            if cancelled(): break
            y = y.to(device)
            x = x.unsqueeze(1).float().to(device)

            noise_device = 'cpu' if device.type == 'mps' else device
            z = torch.randn(x.shape, dtype=x.dtype, device=noise_device).to(device)
            n = torch.randint(0, Nk, (x.size(0),), device=noise_device).to(device)
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

            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; check data/scaling and learning rate')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the EMA model weights
            update_ema(model, target_model, mu=cfg["ema_decay"])
            run_loss += loss.item()
            completed_batches += 1
            progress.update(completed_batches, epoch=epoch, epochs=cfg['epochs'], phase='Training',
                            train_loss=run_loss/(i+1), learning_rate=optimizer.param_groups[0]['lr'], device=str(device))

        if cancelled(): break
        avg_loss = run_loss / (i + 1)
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if val_loader is not None:
            target_model.eval() 
            v_run = 0.0
            with torch.no_grad():
                for j, (vy, vx) in enumerate(val_loader):
                    vy = vy.to(device)
                    vx = vx.unsqueeze(1).float().to(device)
                    vz = torch.randn(vx.shape, dtype=vx.dtype, device=noise_device).to(device)
                    vn = torch.randint(0, Nk, (vx.size(0),), device=noise_device).to(device)
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
                    completed_batches += 1
                    progress.update(completed_batches, epoch=epoch, epochs=cfg['epochs'], phase='Validation',
                                    train_loss=avg_loss, val_loss=v_run/(j+1),
                                    learning_rate=optimizer.param_groups[0]['lr'], device=str(device))

            avg_v = v_run / (j + 1)
            writer.add_scalar("Loss/val", avg_v, epoch)

            if not np.isfinite(avg_v):
                raise ValueError('Non-finite validation loss')
            scheduler.step(avg_v)
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)
            if not os.environ.get('PET_TRACER_PROGRESS_FILE'):
                print(f"Epoch {epoch:04d} | LR: {current_lr:.8g} | train {avg_loss:.6f} | val {avg_v:.6f}")
            # print(f"Epoch {epoch:04d} | train {avg_loss:.6f} | val {avg_v:.6f}")
        else:
            print(f"Epoch {epoch:04d} | train {avg_loss:.6f}")

        with metrics_path.open('a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([epoch, avg_loss, avg_v, optimizer.param_groups[0]['lr'], time.time()-start_time])
        writer.flush()
        completed_epoch = epoch
        item = {'epoch':epoch, 'val_loss':avg_v, 'train_loss':avg_loss, 'learning_rate':current_lr}
        save_checkpoint(target_model, cfg, Path(save_dir, 'last.pth'), epoch, avg_v)
        checkpoint_index['last.pth'] = item
        if avg_v < best:
            best = avg_v
            save_checkpoint(target_model, cfg, Path(save_dir, 'best.pth'), epoch, avg_v)
            checkpoint_index['best.pth'] = item
        if epoch == 1 or epoch % cfg['checkpoint_interval'] == 0:
            name = f'epochs/epoch_{epoch:06d}.pth'
            save_checkpoint(target_model, cfg, Path(save_dir, name), epoch, avg_v)
            checkpoint_index[name] = item
        write_json_atomic(Path(save_dir, 'checkpoints.json'), checkpoint_index)
    stopped = cancelled()
    for sig, handler in old_handlers.items(): signal.signal(sig, handler)
    progress.update(completed_batches, state='stopped' if stopped else 'complete',
                    epoch=completed_epoch, epochs=cfg['epochs'], best_validation_loss=best if np.isfinite(best) else None,
                    best_epoch=checkpoint_index.get('best.pth',{}).get('epoch'), device=str(device))
    writer.close()
    end_time = time.time()
    elapsed = end_time - start_time
    digest=hashlib.sha256()
    with Path(cfg['dataset_dir']).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    summary={'training_mode':training_mode,'dataset_sha256':digest.hexdigest(),
             'parameter_names':names,'split_counts':report['counts'],'seed':cfg['seed'],
             'trainable_parameters':trainable_parameters,'total_parameters':total_parameters,
             'training_seconds':elapsed,'epochs':completed_epoch,'planned_epochs':cfg['epochs'],'stopped_early':stopped,'device':str(device),
             'best_validation_loss':best if np.isfinite(best) else None,
             'best_epoch':checkpoint_index.get('best.pth',{}).get('epoch'),
             'checkpoint_policy':'Lowest validation loss across all completed epochs; last completed epoch; periodic EMA snapshots',
             'fixed_features':'Gaussian Fourier projection is held fixed; transferred when compatible, random for scratch or resized embeddings'}
    Path(save_dir,'training_summary.json').write_text(json.dumps(summary,indent=2)+'\n', encoding='utf-8')
    print(f"Training {'stopped' if stopped else 'completed'} after {completed_epoch} completed epochs in {elapsed:.3f} seconds. Run: {save_dir}",flush=True)
    return Path(save_dir)


def save_checkpoint(model, cfg, path, epoch, val_loss=None):
    checkpoint={'model_state_dict':model.state_dict(),'training_mode':cfg['training_mode'],'epoch':epoch,'val_loss':val_loss}
    for key in ('lr','lora_r','lora_alpha','lora_dropout','channels','embed_dim','embedy','x_dim','y_dim'):
        if key in cfg: checkpoint[key]=cfg[key]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def run_cli(mode='lora'):
    parser=argparse.ArgumentParser(description=__doc__ if mode=='lora' else 'Train a consistency model from scratch')
    parser.add_argument('--config',default='configs/train_scenario1.json' if mode=='lora' else 'configs/train_cm.json')
    for flag,kind in [('checkpoint-interval',int),('epochs',int),('lr',float),('batch-size',int),('embed-dim',int),('x-dim',int),('y-dim',int),('seed',int)]:
        parser.add_argument('--'+flag,type=kind,help='Override the corresponding JSON setting')
    parser.add_argument('--device',choices=['auto','cpu','cuda:0','mps'])
    args=parser.parse_args();cfg=load_config(args.config)
    for key,value in vars(args).items():
        if key!='config' and value is not None: cfg[key]=value
    for key in ('pretrained','dataset_dir','save_dir'):
        if cfg.get(key): cfg[key]=str(Path(cfg[key]).resolve())
    main(cfg,training_mode=mode)


if __name__ == '__main__':
    run_cli('lora')
