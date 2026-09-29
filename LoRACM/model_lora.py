import torch
from model import OneDUnet
from lora import inject_lora_adapters, mark_only_lora_as_trainable, lora_parameter_count

def build_lora_unet(
    x_dim: int,
    y_dim: int,
    embed_dim: int,
    channels=[32, 64, 128, 256],
    embedy: str = "Linear",
    sigma_data: float = 1.0,
    r: int = 8,
    alpha: int = 8,
    dropout: float = 0.05,
    include_names=None,
    exclude_names=("decodex",),
    train_norms: bool = True,
    pretrained_ckpt: str = None,
    device: torch.device = torch.device("cpu"),
):
    # 1. Initialize model with the NEW dimensions
    model = OneDUnet(x_dim, y_dim, embed_dim, channels, embedy, sigma_data).to(device)

    # Track keys that are skipped due to dimension mismatch
    mismatched_keys = set()

    if pretrained_ckpt is not None:
        # Load checkpoint (suppressing the warning if possible, but reading purely)
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        state = ckpt.get("model_state_dict", None)
        if state is None:
            state = ckpt.get("ema_state_dict", ckpt)

        # 2. Smart Loading: Filter out weights with shape mismatches
        model_state = model.state_dict()
        filtered_state = {}
        
        print(f"[LoRA] Checking parameter compatibility...")
        for k, v in state.items():
            if k in model_state:
                # Check if the shape in checkpoint matches the new model's shape
                # e.g. Checkpoint [1024, 70] vs New Model [1024, 60] -> Mismatch!
                if v.shape != model_state[k].shape:
                    print(f"   - Skipping {k}: ckpt {v.shape} vs new {model_state[k].shape}")
                    mismatched_keys.add(k)
                    continue # Skip loading this key; keep random init
                
                # If shapes match, we keep it
                filtered_state[k] = v
        
        # Load only the matching weights
        if not filtered_state:
            raise ValueError("No compatible foundation weights were found")
        missing, unexpected = model.load_state_dict(filtered_state, strict=False)
        mismatched_keys.update(missing)
        print(f"[LoRA] Loaded pretrained weights. Missing keys (new heads): {len(missing)}")

    # 3. Inject LoRA adapters (Backbone adaptation)
    model = inject_lora_adapters(
        model,
        r=r,
        alpha=alpha,
        dropout=dropout,
        include_names=include_names,
        exclude_names=exclude_names,
    )

    # Ensure newly injected modules are on the same device/dtype
    model = model.to(device)

    # 4. Freeze all parameters except LoRA and Norms
    mark_only_lora_as_trainable(model, train_norms=train_norms)

    # 5. Unfreeze the Mismatched Layers (New Heads/Tails)
    # The layers we skipped loading (because they changed size) are currently random 
    # and frozen by step 4. We must unfreeze them so they can learn.
    if mismatched_keys:
        unfrozen_count = 0
        for n, p in model.named_parameters():
            # Check for direct match or wrapped match
            clean_name = n.replace(".wrapped", "")
            
            # If this parameter belongs to a layer we skipped loading...
            if clean_name in mismatched_keys:
                p.requires_grad = True
                unfrozen_count += 1
        
        print(f"[LoRA] Re-enabled training for {unfrozen_count} parameters in resized heads/tails.")

    tr, tot = lora_parameter_count(model)
    print(f"[LoRA] Trainable params: {tr:,} / {tot:,} ({100.0 * tr / tot:.2f}%)")
    return model