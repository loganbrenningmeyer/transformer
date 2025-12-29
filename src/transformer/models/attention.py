import math
import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Applies multi-head self-attention on the batch of input
    sequences x of shape (B, T, d_model). Performs masked self-attention
    if is_causal = True.
    
    Args:
        d_model (int): Embedding dimensionality of the model
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        is_causal (bool): Use masked self-attention if True
    
    Returns:
        O (Tensor): Self-attention output of shape (B, T, d_model)
    """
    def __init__(
            self, 
            d_model: int, 
            num_heads: int, 
            dropout: float,
            is_causal: bool=False
    ):
        super().__init__()

        self.d_model = d_model
        self.d_h = d_model // num_heads
        self.h = num_heads
        self.is_causal = is_causal

        # ----------
        # Query, Key, Value, and Output Projections
        # => W_Q,W_K,W_V,W_O \in \mathcal{R}^{d_\text{model} \times d_\text{model}}
        # ----------
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # ----------
        # Dropout
        # ----------
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor=None) -> torch.Tensor:
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
            causal_mask = torch.triu(torch.ones((1, 1, T, T), dtype=torch.bool, device=S.device), diagonal=1)    # True: Mask out, False: Keep
            S = S.masked_fill(causal_mask, float('-inf'))

        # ----------
        # Padding mask where token == pad_id
        # ----------
        if pad_mask is not None:
            pad_mask = pad_mask[:, None, None, :]   # (B, 1, 1, T)
            S = S.masked_fill(pad_mask, float('-inf'))

        # ----------
        # Perform row-wise softmax to compute attention weights + Dropout
        # => A^{(i)} = \text{softmax}_\text{row}(S^{(i)}) \in \mathcal{R}^{B \times T \times T}
        # ----------
        A = torch.softmax(S, dim=-1)    # (B, h, T, T)
        A = self.drop(A)

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
    Applies multi-head cross-attention between the batch of query 
    inputs target of shape (B, T_tgt, d_model) from the decoder and 
    context inputs memory of shape (B, T_src, d_model) from the encoder output.
    
    Args:
        d_model (int): Embedding dimensionality of the model
        num_heads (int): Number of attention heads
    
    Returns:
        O (Tensor): Cross-attention output of shape (B, T_tgt, d_model)
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_h = d_model // num_heads
        self.h = num_heads

        # ----------
        # Query, Key, Value, and Output Projections
        # => W_Q,W_K,W_V,W_O \in \mathcal{R}^{d_\text{model} \times d_\text{model}}
        # ----------
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # ----------
        # Dropout
        # ----------
        self.drop = nn.Dropout(dropout)

    def forward(
            self, 
            target: torch.Tensor, 
            memory: torch.Tensor, 
            enc_pad_mask: torch.Tensor=None
    ) -> torch.Tensor:
        # ----------
        # Compute Queries, Keys, and Values
        # => x_\text{tgt} \in \mathcal{R}^{B \times T_\text{tgt} \times d_\text{model}},\quad x_\text{src} \in \mathcal{R}^{B \times T_\text{src} \times d_\text{model}}
        # => Q = x_\text{tgt}W_Q \in \mathcal{R}^{B \times T_\text{tgt} \times d_\text{model}}
        # => K = x_\text{src}W_K, V = x_\text{src}W_V \in \mathcal{R}^{B \times T_\text{src} \times d_\text{model}}
        # ----------
        Q: torch.Tensor = self.W_q(target)      # (B, T_tgt, d_model)
        K: torch.Tensor = self.W_k(memory)      # (B, T_src, d_model)
        V: torch.Tensor = self.W_v(memory)      # (B, T_src, d_model)

        # ----------
        # Split into heads
        # => Q^{(i)} \in \mathcal{R}^{B \times T_\text{tgt} \times d_h},\quad K^{(i)},V^{(i)} \in \mathcal{R}^{B \times T_\text{src} \times d_h}
        # ----------
        B, T_tgt, _ = target.shape
        T_src = memory.shape[1]

        Q = Q.view(B, T_tgt, self.h, self.d_h).transpose(1, 2)    # (B, h, T_tgt, d_h)
        K = K.view(B, T_src, self.h, self.d_h).transpose(1, 2)    # (B, h, T_src, d_h)
        V = V.view(B, T_src, self.h, self.d_h).transpose(1, 2)    # (B, h, T_src, d_h)

        # ----------
        # Compute scaled dot-product attention scores
        # => S^{(i)} = \frac{Q^{(i)}K^{(i)\intercal}}{\sqrt{d_\text{h}}} \in \mathcal{R}^{B \times T_\text{tgt} \times T_\text{src}}
        # ----------
        K_T = K.transpose(2, 3)                 # (B, h, d_h, T_src)
        S = Q @ K_T / math.sqrt(self.d_h)       # (B, h, T_tgt, T_src)

        # ----------
        # Padding mask where source token == pad_id
        # ----------
        if enc_pad_mask is not None:
            enc_pad_mask = enc_pad_mask[:, None, None, :]   # (B, 1, 1, T_src)
            S = S.masked_fill(enc_pad_mask, float('-inf'))

        # ----------
        # Perform row-wise softmax to compute attention weights
        # => A^{(i)} = \text{softmax}_\text{row}(S^{(i)}) \in \mathcal{R}^{B \times T_\text{tgt} \times T_\text{src}}
        # ----------
        A = torch.softmax(S, dim=-1)    # (B, h, T_tgt, T_src)
        A = self.drop(A)

        # ----------
        # Apply attention weights to values to get outputs
        # => Y^{(i)} = A^{(i)}V^{(i)} \in \mathcal{R}^{B \times T_\text{tgt} \times d_h}
        # ----------
        Y = A @ V   # (B, h, T_tgt, d_h)

        # ----------
        # Concatenate head outputs
        # => Y_\text{concat} = [Y^{(1)};Y^{(2)};\ldots;Y^{(h)}] \in \mathcal{R}^{B \times T \times d_\text{model}}
        # ----------
        Y = Y.transpose(1, 2).contiguous()      # (B, T_tgt, h, d_h)
        Y = Y.view(B, T_tgt, self.d_model)      # (B, T_tgt, d_model)

        # ----------
        # Apply output projection
        # => O = Y_\text{concat}W_O \in \mathcal{R}^{B \times T_\text{tgt} \times d_\text{model}}
        # ----------
        O = self.W_o(Y)     # (B, T_tgt, d_model)

        return O
