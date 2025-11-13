import torch
import torch.nn as nn

from transformer.models.encoder import Encoder
from transformer.models.decoder import Decoder
from transformer.utils.position import sinusoidal_encoding


class Transformer(nn.Module):
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(self):
        super().__init__()
        pass

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        pass


