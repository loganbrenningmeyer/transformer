import torch
import torch.nn as nn

from transformer.models.encoder import Encoder
from transformer.models.decoder import Decoder
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
            vocab_size: int,
            d_model: int, 
            num_heads: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dropout: float=0.1
    ):
        super().__init__()
        # ----------
        # Encoder / Decoder
        # ----------
        self.encoder = Encoder(d_model, num_heads, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, num_decoder_layers, dropout)
        # ----------
        # Output Projection
        # ----------
        self.out_proj = nn.Linear(d_model, vocab_size)


    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        # ----------
        # Encoder
        # ----------
        memory = self.encoder(src)        # (B, T_enc, d_model)
        # ----------
        # Decoder
        # ----------
        H = self.decoder(tgt, memory)     # (B, T_dec, d_model)
        # ----------
        # Output Projection
        # => W_\text{out} \in \mathcal{R}^{d_\text{model} \times V}
        # => \text{logits} = HW_\text{out},\quad \text{logits} \in \mathcal{R}^{B \times T_\text{dec} \times V}
        # ----------
        logits = self.out_proj(H)         # (B, T_dec, V)

        return logits
