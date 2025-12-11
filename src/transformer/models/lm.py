import torch
import torch.nn as nn

from transformer.models.attention import SelfAttention
from transformer.models.ffn import FeedForwardNetwork


class DecoderLayerLM(nn.Module):
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        # ----------
        # Masked Self-Attention
        # ----------
        self.self_attn = SelfAttention(d_model, num_heads, dropout, is_causal=True)

        # ----------
        # LayerNorms
        # ----------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # ----------
        # Position-Wise Feed-Forward Network
        # ----------
        self.ffn = FeedForwardNetwork(d_model, dropout)

        # ----------
        # Dropout
        # ----------
        self.self_attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # ----------
        # Masked Self-Attention + Dropout
        # ----------
        residual = x
        x = self.self_attn(x)
        x = self.self_attn_drop(x)

        # ----------
        # Add Residual + LayerNorm
        # ----------
        x += residual
        x = self.norm1(x)
        residual = x

        # ----------
        # Feed-Forward Network + Dropout
        # ----------
        x = self.ffn(x)
        x = self.ffn_drop(x)

        # ----------
        # Add Residual + LayerNorm
        # ----------
        x += residual
        x = self.norm2(x)

        return x
    

class DecoderLM(nn.Module):
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            num_layers: int,
            dropout: float
    ):
        super().__init__()
        # ----------
        # DecoderLayerLMs / LayerNorm
        # ----------
        self.layers = nn.ModuleList([
            DecoderLayerLM(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------
        # Compute Hidden States
        # ----------
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return x