import torch
import torch.nn as nn

from transformer.utils.tokenizer import BPETokenizer
from transformer.models.encoder import Encoder
from transformer.models.decoder import Decoder
from transformer.utils.position import sinusoidal_encoding


class TransformerSeq2Seq(nn.Module):
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
            dropout: float,
            vocab_size: int,
    ):
        super().__init__()

        self.d_model = d_model

        # ----------
        # Embedding Table
        # ----------
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

        # ----------
        # Encoder / Decoder
        # ----------
        self.encoder = Encoder(d_model, num_heads, num_encoder_layers, dropout)
        self.decoder = Decoder(d_model, num_heads, num_decoder_layers, dropout, use_cross_attn=True)

        # ----------
        # Output Projection
        # ----------
        self.out_proj = nn.Linear(d_model, vocab_size)


    def forward(self, source: torch.Tensor, target: torch.Tensor):
        # ----------
        # Get token embeddings
        # ----------
        source_emb = self.token_embeddings(source)
        target_emb = self.token_embeddings(target)

        # ----------
        # Add positional encodings
        # ----------
        source_idx = torch.arange(source.shape[1], device=source.device)
        target_idx = torch.arange(target.shape[1], device=target.device)

        source_pos_emb = sinusoidal_encoding(source_idx, self.d_model)
        target_pos_emb = sinusoidal_encoding(target_idx, self.d_model)

        source_emb = source_emb + source_pos_emb
        target_emb = target_emb + target_pos_emb

        # ----------
        # Encoder
        # ----------
        memory = self.encoder(source_emb)        # (B, T_enc, d_model)

        # ----------
        # Decoder
        # ----------
        H = self.decoder(target_emb, memory)     # (B, T_dec, d_model)
        
        # ----------
        # Output Projection
        # => W_\text{out} \in \mathcal{R}^{d_\text{model} \times V}
        # => \text{logits} = HW_\text{out},\quad \text{logits} \in \mathcal{R}^{B \times T_\text{dec} \times V}
        # ----------
        logits = self.out_proj(H)         # (B, T_dec, V)

        return logits