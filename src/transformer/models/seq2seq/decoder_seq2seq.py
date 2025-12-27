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
        5. Position-wise feed-forward network + Dropout
        6. Residual connection + LayerNorm
    
    Args:
        d_model (int): Embedding dimensionality of the model
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    
    Returns:
        target (Tensor): The updated decoder hidden states of shape (B, T_dec, d_model)
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
        self.norm3 = nn.LayerNorm(d_model)

        # ----------
        # Encoder Memory Cross-Attention
        # ----------
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)

        # ----------
        # Position-Wise Feed-Forward Network
        # ----------
        self.ffn = FeedForwardNetwork(d_model, dropout)

        # ----------
        # Dropout
        # ----------
        self.self_attn_drop = nn.Dropout(dropout)
        self.cross_attn_drop = nn.Dropout(dropout)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # ----------
        # Masked Self-Attention + Dropout
        # ----------
        residual = target   # store residual
        target = self.self_attn(target)
        target = self.self_attn_drop(target)

        # ----------
        # Add Residual + LayerNorm
        # ----------
        target += residual
        target = self.norm1(target)
        residual = target   # store residual

        # ----------
        # Encoder Memory Cross-Attention + Dropout
        # ----------
        target = self.cross_attn(target, memory)
        target = self.cross_attn_drop(target)

        # ----------
        # Add Residual + LayerNorm
        # ----------
        target += residual
        target = self.norm2(target)
        residual = target

        # ----------
        # Feed-Forward Network + Dropout
        # ----------
        target = self.ffn(target)
        target = self.ffn_drop(target)
        
        # ----------
        # Add Residual + LayerNorm
        # ----------
        target += residual
        target = self.norm3(target)

        return target


class Decoder(nn.Module):
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
        # DecoderLayers / LayerNorm
        # ----------
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, target: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # ----------
        # Compute Hidden States
        # ----------
        for layer in self.layers:
            target = layer(target, memory)
        target = self.norm(target)

        return target