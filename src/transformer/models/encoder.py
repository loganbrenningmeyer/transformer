import torch
import torch.nn as nn

from transformer.models.attention import SelfAttention
from transformer.models.ffn import FeedForwardNetwork


class EncoderLayer(nn.Module):
    """
    Transformer Encoder layer, consisting of:
        1. Multi-head self-attention
        2. Residual connection + LayerNorm
        3. Position-wise feed-forward network + Dropout
        4. Residual connection + LayerNorm
    
    Args:
        d_model (int): Embedding dimensionality of the model
        h (int): Number of attention heads
        dropout (float): Dropout probability
    
    Returns:
        src (Tensor): The updated encoder hidden states of shape (B, T_enc, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        # ----------
        # Self-Attention
        # ----------
        self.self_attn = SelfAttention(d_model, num_heads, dropout)
        # ----------
        # LayerNorms
        # ----------
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # ----------
        # Feed-Forward Network
        # ----------
        self.ffn = FeedForwardNetwork(d_model, dropout)
        # ----------
        # Dropout
        # ----------
        self.self_attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # ----------
        # Self-Attention + Dropout
        # ----------
        residual = src   # store residual
        src = self.self_attn(src)
        src = self.self_attn_drop(src)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        src += residual
        src = self.norm1(src)
        residual = src   # store residual
        # ----------
        # Feed-Forward Network + Dropout
        # ----------
        src = self.ffn(src)
        src = self.ffn_drop(src)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        src += residual
        src = self.norm2(src)

        return src
    

class Encoder(nn.Module):
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            num_layers: int,
            dropout: float=0.1
    ):
        super().__init__()
        # ----------
        # EncoderLayers / LayerNorm
        # ----------
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            src = layer(src)
        src = self.norm(src)

        return src