import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Performs multi-head self-attention on the batch of input
    sequences x of shape (B, T, d_model). Allows masked self-attention
    if is_causal = True.
    
    Args:
        d_model (int): Embedding dimensionality of the model
        h (int): Number of attention heads
        is_causal (bool): Use masked self-attention if True
    
    Returns:
        O (Tensor): Self-attention output of shape (B, T, d_model)
    """
    def __init__(self, d_model: int, h: int, is_causal: bool=False):
        super().__init__()

        self.d_model = d_model
        self.d_h = d_model // h
        self.h = h
        self.is_causal = is_causal

        # ----------
        # Query, Key, Value, and Output Projections
        # => W_Q,W_K,W_V,W_O \in \mathcal{R}^{d_\text{model} \times d_\text{model}}
        # ----------
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------
        # Compute Queries, Keys, and Values
        # => x \in \mathcal{R}^{B \times T \times d_\text{model}}
        # => Q = xW_Q,\ K = xW_K,\ V = xW_V,\quad Q,K,V \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        Q: torch.Tensor = self.W_q(x)    # (B, T, d_model)
        K: torch.Tensor = self.W_k(x)    # (B, T, d_model)
        V: torch.Tensor = self.W_v(x)    # (B, T, d_model)

        # ----------
        # Split into heads
        # => Q^{(i)}, K^{(i)}, V^{(i)} \in \mathcal{R}^{B \times T \times d_h},\quad i=1,2,\ldots,h,\quad d_h = \frac{d_\text{model}}{h}
        # ----------
        B, T, _ = x.shape

        Q = Q.view(B, T, self.h, self.d_h).transpose(1, 2)    # (B, h, T, d_h)
        K = K.view(B, T, self.h, self.d_h).transpose(1, 2)    # (B, h, T, d_h)
        V = V.view(B, T, self.h, self.d_h).transpose(1, 2)    # (B, h, T, d_h)

        # ----------
        # Compute scaled dot-product attention scores
        # => S^{(i)} = \frac{Q^{(i)}K^{(i)\intercal}}{\sqrt{d_\text{h}}} \in \mathcal{R}^{B \times T \times T}
        # ----------
        K_T = K.transpose(2, 3)                 # (B, h, d_h, T)
        S = Q @ K_T / math.sqrt(self.d_h)       # (B, h, T, T)

        # ----------
        # Masked Self-Attention
        # ----------
        if self.is_causal:
            mask = torch.triu(torch.ones((1, 1, T, T), dtype=torch.bool, device=S.device), diagonal=1)    # True: Mask out, False: Keep
            S = S.masked_fill(mask, float('-inf'))

        # ----------
        # Perform row-wise softmax to compute attention weights
        # => A^{(i)} = \text{softmax}_\text{row}(S^{(i)}) \in \mathcal{R}^{B \times T \times T}
        # ----------
        A = torch.softmax(S, dim=-1)    # (B, h, T, T)

        # ----------
        # Apply attention weights to values to get outputs
        # => Y^{(i)} = A^{(i)}V^{(i)} \in \mathcal{R}^{B \times T \times d_h}
        # ----------
        Y = A @ V   # (B, h, T, d_h)

        # ----------
        # Concatenate head outputs
        # => Y_\text{concat} = [Y^{(1)};Y^{(2)};\ldots;Y^{(h)}] \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        Y = Y.transpose(1, 2).contiguous()  # (B, T, h, d_h)
        Y = Y.view(B, T, self.d_model)      # (B, T, d_model)

        # ----------
        # Apply output projection
        # => O = Y_\text{concat}W_O \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        O = self.W_o(Y)     # (B, T, d_model)

        return O


class CrossAttention(nn.Module):
    """
    Performs multi-head cross-attention between the batch of query 
    inputs tgt of shape (B, T_dec, d_model) from the decoder and 
    context inputs memory of shape (B, T_enc, d_model) from the encoder output.
    
    Args:
        d_model (int): Embedding dimensionality of the model
        h (int): Number of attention heads
    
    Returns:
        O (Tensor): Cross-attention output of shape (B, T_dec, d_model)
    """
    def __init__(self, d_model: int, h: int):
        super().__init__()

        self.d_model = d_model
        self.d_h = d_model // h
        self.h = h

        # ----------
        # Query, Key, Value, and Output Projections
        # => W_Q,W_K,W_V,W_O \in \mathcal{R}^{d_\text{model} \times d_\text{model}}
        # ----------
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # ----------
        # Compute Queries, Keys, and Values
        # => x_\text{dec} \in \mathcal{R}^{B \times T_\text{dec} \times d_\text{model}},\quad x_\text{enc} \in \mathcal{R}^{B \times T_\text{enc} \times d_\text{model}}
        # => Q = x_\text{dec}W_Q \in \mathcal{R}^{B \times T_\text{dec} \times d_\text{model}}
        # => K = x_\text{enc}W_K, V = x_\text{enc}W_V \in \mathcal{R}^{B \times T_\text{enc} \times d_\text{model}}
        # ----------
        Q: torch.Tensor = self.W_q(tgt)     # (B, T_dec, d_model)
        K: torch.Tensor = self.W_k(memory)  # (B, T_enc, d_model)
        V: torch.Tensor = self.W_v(memory)  # (B, T_enc, d_model)

        # ----------
        # Split into heads
        # => Q^{(i)} \in \mathcal{R}^{B \times T_\text{dec} \times d_h},\quad K^{(i)},V^{(i)} \in \mathcal{R}^{B \times T_\text{enc} \times d_h}
        # ----------
        B, T_dec, _ = tgt.shape
        T_enc = memory.shape[1]

        Q = Q.view(B, T_dec, self.h, self.d_h).transpose(1, 2)    # (B, h, T_dec, d_h)
        K = K.view(B, T_enc, self.h, self.d_h).transpose(1, 2)    # (B, h, T_enc, d_h)
        V = V.view(B, T_enc, self.h, self.d_h).transpose(1, 2)    # (B, h, T_enc, d_h)

        # ----------
        # Compute scaled dot-product attention scores
        # => S^{(i)} = \frac{Q^{(i)}K^{(i)\intercal}}{\sqrt{d_\text{h}}} \in \mathcal{R}^{B \times T_\text{dec} \times T_\text{enc}}
        # ----------
        K_T = K.transpose(2, 3)                 # (B, h, d_h, T_enc)
        S = Q @ K_T / math.sqrt(self.d_h)       # (B, h, T_dec, T_enc)

        # ----------
        # Perform row-wise softmax to compute attention weights
        # => A^{(i)} = \text{softmax}_\text{row}(S^{(i)}) \in \mathcal{R}^{B \times T_\text{dec} \times T_\text{enc}}
        # ----------
        A = torch.softmax(S, dim=-1)    # (B, h, T_dec, T_enc)

        # ----------
        # Apply attention weights to values to get outputs
        # => Y^{(i)} = A^{(i)}V^{(i)} \in \mathcal{R}^{B \times T_\text{dec} \times d_h}
        # ----------
        Y = A @ V   # (B, h, T_dec, d_h)

        # ----------
        # Concatenate head outputs
        # => Y_\text{concat} = [Y^{(1)};Y^{(2)};\ldots;Y^{(h)}] \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        Y = Y.transpose(1, 2).contiguous()      # (B, T_dec, h, d_h)
        Y = Y.view(B, T_dec, self.d_model)      # (B, T_dec, d_model)

        # ----------
        # Apply output projection
        # => O = Y_\text{concat}W_O \in \mathcal{R}^{B \times T_\text{dec} \times d_\text{model}}
        # ----------
        O = self.W_o(Y)     # (B, T_dec, d_model)

        return O
