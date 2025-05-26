import torch
import torch.nn as nn
from dataclasses import dataclass, asdict
from typing import List

from .foura import FouRA

@dataclass
class FouRAConfig:
    rank: int = 16
    foura_alpha: float = 32.0
    transform_type: str = "dct"            # "none", "dct", or "fft"
    target_modules: List[str] = None       # substrings to match 
    use_gate: bool = False
    def __post_init__(self):
        if self.target_modules is None:
            # sane ViT default
            self.target_modules = ["query", "value"]
        self.transform_type = self.transform_type.lower().strip()

class FouRAInjectedLinear(nn.Module):
    """
    y = W0(x) + adapter_projection(FouRA(x)) * scaling
    """
    def __init__(self, base_linear: nn.Linear, cfg: FouRAConfig):
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("FouRAInjectedLinear expects an nn.Linear to wrap.")

        self.in_features  = base_linear.in_features
        self.out_features = base_linear.out_features
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.foura = FouRA(
            N=self.in_features,
            rank=cfg.rank,
            use_gate=cfg.use_gate,
            transform_type=cfg.transform_type
        )
        self.scaling = cfg.foura_alpha / cfg.rank

        if self.in_features != self.out_features:
            self.adapter_projection = nn.Linear(self.in_features, self.out_features, bias=False)
            nn.init.zeros_(self.adapter_projection.weight)
        else:
            self.adapter_projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        foura_output = self.foura(x)
        adapted_output = self.adapter_projection(foura_output)
        return self.base(x) + adapted_output * self.scaling

def _get_parent(model: nn.Module, module_name: str):
    comps = module_name.split(".")
    parent = model
    for comp in comps[:-1]:
        parent = getattr(parent, comp)
    return parent, comps[-1]

def _matches(name: str, patterns: List[str]) -> bool:
    return any(pat in name for pat in patterns)

def get_foura_model(model: nn.Module, cfg: FouRAConfig) -> nn.Module:
    model.requires_grad_(False)
    # Make classifier trainable only if it exists
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        model.classifier.requires_grad_(True)
    
    replacement_count = 0
    for full_name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches(full_name, cfg.target_modules):
            parent, attr_name = _get_parent(model, full_name)
            wrapped = FouRAInjectedLinear(module, cfg)
            setattr(parent, attr_name, wrapped)
            replacement_count += 1

    if replacement_count == 0:
        raise ValueError("get_foura_model: found no matching Linear layers "
                         f"for patterns {cfg.target_modules}")
    model.foura_config = asdict(cfg)
    print(f"[FouRA] Injected adapters into {replacement_count} Linear layers.")
    return model 