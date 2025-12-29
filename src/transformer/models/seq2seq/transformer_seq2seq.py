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
        memory = self.encoder(source_emb)        # (B, T_src, d_model)

        # ----------
        # Decoder
        # ----------
        H = self.decoder(target_emb, memory)     # (B, T_tgt, d_model)
        
        # ----------
        # Output Projection
        # => W_\text{out} \in \mathcal{R}^{d_\text{model} \times V}
        # => \text{logits} = HW_\text{out},\quad \text{logits} \in \mathcal{R}^{B \times T_\text{tgt} \times V}
        # ----------
        logits = self.out_proj(H)         # (B, T_tgt, V)

        return logits
    
    @torch.no_grad()
    def generate(
        self, 
        source: torch.Tensor,
        special_ids: dict,
        block_size: int,
        max_tokens: int
    ) -> torch.Tensor:
        """
        
        
        Args:
            source (torch.Tensor): Batch of source samples of shape (B, T_src)
            device (torch.device): 
            block_size (int): 
            max_tokens (int): 
        
        Returns:
        
        """
        self.eval()
        device = source.device

        # -- Define special token ids
        bos_id = special_ids["bos"]
        eos_id = special_ids["eos"]
        pad_id = special_ids["pad"]

        # ----------
        # Get source token / positional embeddings
        # ----------
        source_emb = self.token_embeddings(source)  

        source_idx = torch.arange(source.shape[1], device=device)
        source_emb += sinusoidal_encoding(source_idx, self.d_model)     # (B, T_src, d_model)

        # ----------
        # Create padding mask / compute Encoder memory 
        # ----------
        enc_pad_mask = (source == pad_id)
        memory = self.encoder(source_emb, enc_pad_mask)                 # (B, T_src, d_model)

        # ----------
        # Initialize Decoder output batch to [BOS] / boolean finished output mask
        # ----------
        output_ids = torch.full((source.shape[0], 1), bos_id, dtype=torch.long, device=device)     # (B, 1)
        finished = torch.zeros((source.shape[0],), dtype=torch.bool, device=device)                 # (B, )

        # ----------
        # Iteratively generate tokens until [EOS]
        # ----------
        i = 0
        while i < max_tokens:
            # ----------
            # Get token / absolute positional embeddings
            # ----------
            context_ids = output_ids[:, -block_size:]
            context_emb = self.token_embeddings(context_ids)

            start_pos = output_ids.shape[1] - context_ids.shape[1]
            context_idx = start_pos + torch.arange(context_ids.shape[1], device=device)

            context_emb += sinusoidal_encoding(context_idx, self.d_model)

            # ----------
            # Pass through Decoder w/ padding mask
            # ----------
            H = self.decoder(context_emb, memory, enc_pad_mask)         # (B, T_tgt, d_model)

            # ----------
            # Sample token from final token's logits
            # ----------
            logits = self.out_proj(H)                                   # (B, T_tgt, V)
            logits = logits[:, -1, :]                                   # (B, V): last token
            next_token_id = torch.argmax(logits, dim=1)                 # (B, )

            # ----------
            # Add next token to output ([PAD] if [EOS] already reached)
            # ----------
            next_token_id[finished] = pad_id
            output_ids = torch.cat([output_ids, next_token_id.unsqueeze(1)], dim=1)

            # ----------
            # Set output finished = True if [EOS] / Complete if all reach [EOS]
            # ----------
            finished = finished | (next_token_id == eos_id)     # True if already [EOS] or now [EOS]

            if torch.all(finished):
                break

            i += 1

        return output_ids
        