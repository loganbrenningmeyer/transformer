import torch
import torch.nn as nn

from transformer.models.attention import SelfAttention
from transformer.models.ffn import FeedForwardNetwork


class EncoderBlock(nn.Module):
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
        source (Tensor): The updated encoder hidden states of shape (B, T_enc, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float):
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

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # ----------
        # Self-Attention + Dropout
        # ----------
        residual = source   # store residual
        source = self.self_attn(source)
        source = self.self_attn_drop(source)

        # ----------
        # Add Residual + LayerNorm
        # ----------
        source += residual
        source = self.norm1(source)
        residual = source   # store residual

        # ----------
        # Feed-Forward Network + Dropout
        # ----------
        source = self.ffn(source)
        source = self.ffn_drop(source)
        
        # ----------
        # Add Residual + LayerNorm
        # ----------
        source += residual
        source = self.norm2(source)

        return source
    

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
            dropout: float
    ):
        super().__init__()
        # ----------
        # EncoderLayers / LayerNorm
        # ----------
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, source: torch.Tensor) -> torch.Tensor:
        # ----------
        # Compute Encoder Memory
        # ----------
        for layer in self.layers:
            source = layer(source)
        source = self.norm(source)

        return source