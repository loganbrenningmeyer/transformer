import torch
import torch.nn as nn

from transformer.models.attention import SelfAttention, CrossAttention
from transformer.models.ffn import FeedForwardNetwork


class DecoderLayer(nn.Module):
    """
    Transformer Decoder layer, consisting of:
        1. Masked multi-head self-attention
        2. Residual connection + LayerNorm
        3. Encoder memory multi-head cross-attention
        4. Residual connection + LayerNorm
        5. Position-wise feed-forward network
        6. Residual connection + LayerNorm
    
    Args:
        d_model (int): Embedding dimensionality of the model
        h (int): Number of attention heads
    
    Returns:
        tgt (Tensor): The updated decoder hidden states of shape (B, T_dec, d_model)
    """
    def __init__(self, d_model: int, h: int):
        super().__init__()
        # ----------
        # Masked Self-Attention
        # ----------
        self.self_attn = SelfAttention(d_model, h, is_causal=True)
        # ----------
        # LayerNorms
        # ----------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        # ----------
        # Encoder Memory Cross-Attention
        # ----------
        self.cross_attn = CrossAttention(d_model, h)
        # ----------
        # Position-Wise Feed-Forward Network
        # ----------
        self.ffn = FeedForwardNetwork(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor):
        # ----------
        # Masked Self-Attention
        # ----------
        residual = tgt   # store residual
        tgt = self.self_attn(tgt)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        tgt += residual
        tgt = self.norm1(tgt)
        residual = tgt   # store residual
        # ----------
        # Encoder Memory Cross-Attention
        # ----------
        tgt = self.cross_attn(tgt, memory)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        tgt += residual
        tgt = self.norm2(tgt)
        residual = tgt
        # ----------
        # Feed-Forward Network
        # ----------
        tgt = self.ffn(tgt)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        tgt += residual
        tgt = self.norm3(tgt)

        return tgt
