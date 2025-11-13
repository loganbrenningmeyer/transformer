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
        tgt (Tensor): The updated decoder hidden states of shape (B, T_dec, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float=0.1):
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

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # ----------
        # Masked Self-Attention + Dropout
        # ----------
        residual = tgt   # store residual
        tgt = self.self_attn(tgt)
        tgt = self.self_attn_drop(tgt)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        tgt += residual
        tgt = self.norm1(tgt)
        residual = tgt   # store residual
        # ----------
        # Encoder Memory Cross-Attention + Dropout
        # ----------
        tgt = self.cross_attn(tgt, memory)
        tgt = self.cross_attn_drop(tgt)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        tgt += residual
        tgt = self.norm2(tgt)
        residual = tgt
        # ----------
        # Feed-Forward Network + Dropout
        # ----------
        tgt = self.ffn(tgt)
        tgt = self.ffn_drop(tgt)
        # ----------
        # Add Residual + LayerNorm
        # ----------
        tgt += residual
        tgt = self.norm3(tgt)

        return tgt


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
            dropout: float=0.1
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
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            tgt = layer(tgt, memory)
        tgt = self.norm(tgt)

        return tgt