import torch
import torch.nn as nn

from transformer.models.encoder import EncoderLayer
from transformer.models.decoder import DecoderLayer
from transformer.utils.position import sinusoidal_encoding


class Transformer(nn.Module):
    """
    
    
    Args:
        d_model (int): Embedding dimensionality of the model
        num_heads (int): Number of attention heads
        num_encoder_layers (int): Number of Encoder layers
        num_decoder_layers (int): Number of Decoder layers
        dropout (float): Dropout probability
    
    Returns:
    
    """
    def __init__(
            self, 
            d_model: int, 
            num_heads: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dropout: float=0.1
    ):
        super().__init__()
        # ----------
        # Encoder
        # ----------

        # ----------
        # Decoder
        # ----------


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # ----------
        # Encoder
        # ----------


        # ----------
        # Decoder
        # ----------


