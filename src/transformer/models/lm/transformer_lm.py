import torch
import torch.nn as nn

from transformer.models.decoder import Decoder
from transformer.utils.tokenizer import BPETokenizer
from transformer.utils.position import sinusoidal_encoding


class TransformerLM(nn.Module):
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            d_model: int, 
            num_heads: int,
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
        # Decoder / Output Projection
        # ----------
        self.decoder = Decoder(d_model, num_heads, num_decoder_layers, dropout, use_cross_attn=False)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor):
        # ----------
        # Get token embeddings / add positional embeddings
        # ----------
        x_emb = self.token_embeddings(x)

        x_idx = torch.arange(x.shape[1], device=x.device)
        x_pos_emb = sinusoidal_encoding(x_idx, self.d_model)
        x_emb = x_emb + x_pos_emb

        # ----------
        # Decoder / Output Projection
        # ----------
        H = self.decoder(x_emb)
        logits = self.out_proj(H)

        return logits

    @torch.no_grad()
    def generate(
        self, 
        bpe: BPETokenizer, 
        prompt: str, 
        block_size: int, 
        max_tokens: int,
        device: torch.device
    ):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        self.eval()

        # ----------
        # Tokenize prompt
        # ----------
        prompt_ids = list(prompt.encode("utf-8"))
        prompt_ids = bpe.tokenize(prompt_ids)

        ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        ids = ids.unsqueeze(0)    # (1, T_prompt)

        # ----------
        # Autoregressive generation
        # ----------
        for _ in range(max_tokens):
            # ----------
            # Crop to block_size (T = min(T_prompt, block_size))
            # ----------
            context_ids = ids[:, -block_size:]      # (1, T)

            # ----------
            # Predict next token logits
            # ----------
            logits = self(context_ids)              # (1, T, vocab_size)
            # -- Get final token logits
            logits = logits[:, -1, :]               # (1, vocab_size)

            # ----------
            # Sample next generated token / append to inputs
            # ----------
            next_token_id = torch.argmax(logits, dim=1).unsqueeze(0)
            ids = torch.cat([ids, next_token_id], dim=1)

        # ----------
        # Convert generated ids to text
        # ----------
        ids = ids[0].detach().cpu().tolist()
        output = bpe.ids_to_string(ids)

        return output