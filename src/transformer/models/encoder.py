import torch
import torch.nn as nn

from transformer.models.attention import SelfAttention
from transformer.models.ffn import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    Transformer Encoder layer, consisting of:
        1. Multi-head self-attention
        2. Residual connection + LayerNorm
        3. Position-wise feed-forward network
        4. Residual connection + LayerNorm
    
    Args:
        d_model (int): Embedding dimensionality of the model
        h (int): Number of attention heads
    
    Returns:
        src (Tensor): The updated encoder hidden states of shape (B, T_enc, d_model)
    """
    def __init__(self, d_model: int, h: int):
        super().__init__()
        # ----------
        # Self-Attention
        # ----------
        self.self_attn = SelfAttention(d_model, h)
        # ----------
        # LayerNorms
        # ----------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # ----------
        # Feed-Forward Network
        # ----------
        self.ffn = FeedForwardNetwork(d_model)

    def forward(self, src: torch.Tensor):
        # ----------
        # Self-Attention
        # ----------
        residual = src   # store residual
        src = self.self_attn(src)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        src += residual
        src = self.norm1(src)
        residual = src   # store residual
        # ----------
        # Feed-Forward Network
        # ----------
        src = self.ffn(src)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        src += residual
        src = self.norm2(src)

        return src