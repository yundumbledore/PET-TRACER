
import math
from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------- Core LoRA base ----------------------

class LoRAModule(nn.Module):
    def __init__(self, r: int, alpha: int, dropout: float = 0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 0.0
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _init(A: nn.Parameter, B: nn.Parameter):
        if A is not None:
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        if B is not None:
            nn.init.zeros_(B)

# ---------------------- Linear (unchanged: weight-space) ----------------------

class LoRALinear(LoRAModule):
    def __init__(self, wrapped: nn.Linear, r: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__(r, alpha, dropout)
        self.wrapped = wrapped
        if r > 0:
            dev, dt = wrapped.weight.device, wrapped.weight.dtype
            self.A = nn.Parameter(torch.zeros((r, wrapped.in_features), device=dev, dtype=dt))
            self.B = nn.Parameter(torch.zeros((wrapped.out_features, r), device=dev, dtype=dt))
            self._init(self.A, self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        base = self.wrapped(x)
        if self.r == 0:
            return base
        x_d = self.dropout(x)
        update = F.linear(x_d, self.A)
        update = F.linear(update, self.B)
        return base + self.scaling * update

# ---------------------- Exact weight-space LoRA for Conv1d ----------------------

class LoRAConv1d(LoRAModule):
    def __init__(self, wrapped: nn.Conv1d, r: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__(r, alpha, dropout)
        self.wrapped = wrapped
        if r > 0:
            Cout, Cin, K = wrapped.out_channels, wrapped.in_channels, wrapped.kernel_size[0]
            dev, dt = wrapped.weight.device, wrapped.weight.dtype
            # A: (r, Cin*K), B: (Cout, r)
            self.A = nn.Parameter(torch.zeros((r, Cin * K), device=dev, dtype=dt))
            self.B = nn.Parameter(torch.zeros((Cout, r), device=dev, dtype=dt))
            self._init(self.A, self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        base = self.wrapped(x)
        if self.r == 0:
            return base
        Cout, Cin, K = self.wrapped.out_channels, self.wrapped.in_channels, self.wrapped.kernel_size[0]
        delta_flat = self.B @ self.A                      # (Cout, Cin*K)
        delta_w = delta_flat.view(Cout, Cin, K)           # (Cout, Cin, K)
        x_d = self.dropout(x)
        update = F.conv1d(
            x_d, delta_w, bias=None,
            stride=self.wrapped.stride,
            padding=self.wrapped.padding,
            dilation=self.wrapped.dilation,
            groups=self.wrapped.groups,
        )
        return base + self.scaling * update

# ---------------------- Exact weight-space LoRA for ConvTranspose1d ----------------------

class LoRAConvTranspose1d(LoRAModule):
    def __init__(self, wrapped: nn.ConvTranspose1d, r: int = 8, alpha: int = 8, dropout: float = 0.0):
        super().__init__(r, alpha, dropout)
        self.wrapped = wrapped
        if r > 0:
            CinT, CoutT, K = wrapped.in_channels, wrapped.out_channels, wrapped.kernel_size[0]
            dev, dt = wrapped.weight.device, wrapped.weight.dtype
            # Build delta for transposed conv:
            # Create (CoutT, CinT*K) then reshape to (CoutT, CinT, K) and permute to (CinT, CoutT, K)
            self.A = nn.Parameter(torch.zeros((r, CinT * K), device=dev, dtype=dt))
            self.B = nn.Parameter(torch.zeros((CoutT, r), device=dev, dtype=dt))
            self._init(self.A, self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def forward(self, x):
        base = self.wrapped(x)
        if self.r == 0:
            return base
        CinT, CoutT, K = self.wrapped.in_channels, self.wrapped.out_channels, self.wrapped.kernel_size[0]
        delta_flat_out = self.B @ self.A                         # (CoutT, CinT*K)
        delta_w = delta_flat_out.view(CoutT, CinT, K).permute(1, 0, 2)  # (CinT, CoutT, K)
        x_d = self.dropout(x)
        update = F.conv_transpose1d(
            x_d, delta_w, bias=None,
            stride=self.wrapped.stride,
            padding=self.wrapped.padding,
            output_padding=self.wrapped.output_padding,
            dilation=self.wrapped.dilation,
            groups=self.wrapped.groups,
        )
        return base + self.scaling * update

# ---------------------- Utilities ----------------------

def _replace_module(parent: nn.Module, child_name: str, new_module: nn.Module):
    setattr(parent, child_name, new_module)

def inject_lora_adapters(
    model: nn.Module,
    r: int = 8,
    alpha: int = 8,
    dropout: float = 0.0,
    include_names: Optional[Iterable[str]] = None,
    exclude_names: Optional[Iterable[str]] = None,
) -> nn.Module:
    include = set(include_names) if include_names is not None else None
    exclude = set(exclude_names) if exclude_names is not None else set()

    def should_wrap(qual_name: str, module: nn.Module) -> bool:
        if include is not None and qual_name not in include:
            return False
        if qual_name in exclude:
            return False
        return isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d))

    for qual_name, module in list(model.named_modules()):
        if qual_name == "":
            continue
        if should_wrap(qual_name, module):
            parent_name = ".".join(qual_name.split(".")[:-1])
            child_name = qual_name.split(".")[-1]

            parent = model
            if parent_name:
                for attr in parent_name.split("."):
                    parent = getattr(parent, attr)

            if isinstance(module, nn.Linear):
                wrapped = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
            elif isinstance(module, nn.Conv1d):
                wrapped = LoRAConv1d(module, r=r, alpha=alpha, dropout=dropout)
            elif isinstance(module, nn.ConvTranspose1d):
                wrapped = LoRAConvTranspose1d(module, r=r, alpha=alpha, dropout=dropout)
            else:
                continue
            _replace_module(parent, child_name, wrapped)

    return model

def mark_only_lora_as_trainable(model: nn.Module, train_norms: bool = True):
    for n, p in model.named_parameters():
        p.requires_grad = False

    for n, p in model.named_parameters():
        if any(s in n for s in ["A", "B"]):
            p.requires_grad = True
        elif train_norms and ("norm" in n or "gnorm" in n or "bn" in n or "layernorm" in n):
            p.requires_grad = True

def lora_parameter_count(model: nn.Module) -> Tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
