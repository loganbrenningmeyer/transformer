import torch
import torch.nn as nn

from transformer.models.decoder import Decoder
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
        x_emb += sinusoidal_encoding(x_idx, self.d_model)

        # ----------
        # Decoder / Output Projection
        # ----------
        H = self.decoder(x_emb)
        logits = self.out_proj(H)

        return logits

    @torch.no_grad()
    def generate(
        self, 
        prompt_ids: list[int], 
        device: torch.device,
        block_size: int, 
        max_tokens: int=256,
        multinomial: bool=True,
        temperature: float=1.0
    ) -> list[int]:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        self.eval()

        output_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        output_ids = output_ids.unsqueeze(0)    # (1, T_prompt)

        # ----------
        # Autoregressive generation
        # ----------
        for _ in range(max_tokens):
            # ----------
            # Crop to block_size (T = min(T_prompt, block_size))
            # ----------
            context_ids = output_ids[:, -block_size:]      # (1, T)

            # ----------
            # Predict next token logits
            # ----------
            logits = self(context_ids)              # (1, T, V)
            # -- Get final token logits
            logits = logits[:, -1, :]               # (1, V)

            # ----------
            # Sample next generated token / append to inputs
            # ----------
            if multinomial:
                scaled_logits = logits / temperature
                probs = torch.softmax(scaled_logits, dim=1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=1).unsqueeze(0)
                
            output_ids = torch.cat([output_ids, next_token_id], dim=1)

        return output_ids[0].detach().cpu().tolist()