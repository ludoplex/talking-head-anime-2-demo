from torch.nn import Module
from torch.nn.utils import spectral_norm


def apply_spectral_norm(module: Module, use_spectrial_norm: bool = False) -> Module:
    return spectral_norm(module) if use_spectrial_norm else module
