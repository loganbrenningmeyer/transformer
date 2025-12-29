import torch
import torch.nn as nn

from transformer.models.attention import SelfAttention, CrossAttention
from transformer.models.ffn import FeedForwardNetwork


class DecoderBlock(nn.Module):
    """
    Transformer Decoder block, consisting of:
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
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            dropout: float,
            use_cross_attn: bool
    ):
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
        self.cross_attn = CrossAttention(d_model, num_heads, dropout) if use_cross_attn else None

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

    def forward(
            self, 
            target: torch.Tensor, 
            memory: torch.Tensor | None = None,
            enc_pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # ----------
        # Masked Self-Attention / Residual + Norm
        # ----------
        residual = target   # store residual
        target = self.self_attn(target)
        target = self.self_attn_drop(target)
        target = self.norm1(target + residual)

        # ----------
        # TransformerSeq2Seq: Encoder Memory Cross-Attention
        # ----------
        if self.cross_attn is not None:
            if memory is None:
                raise ValueError("memory must be provided when use_cross_attn=True")
            
            residual = target   # store residual
            target = self.cross_attn(target, memory, enc_pad_mask)
            target = self.cross_attn_drop(target)
            target = self.norm2(target + residual)

        # ----------
        # Feed-Forward Network / Residual + Norm
        # ----------
        residual = target
        target = self.ffn(target)
        target = self.ffn_drop(target)
        target = self.norm3(target + residual)

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
            dropout: float,
            use_cross_attn: bool
    ):
        super().__init__()
        # ----------
        # DecoderLayers / LayerNorm
        # ----------
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(d_model, num_heads, dropout, use_cross_attn)
            for _ in range(num_layers)
        ])
        
    def forward(
            self, 
            target: torch.Tensor, 
            memory: torch.Tensor | None = None,
            enc_pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # ----------
        # Compute Hidden States
        # ----------
        for dec_block in self.dec_blocks:
            target = dec_block(target, memory, enc_pad_mask)

        return target