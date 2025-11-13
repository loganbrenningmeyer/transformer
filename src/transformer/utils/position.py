import math
import torch

def sinusoidal_encoding(p: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    Computes sinusoidal positional encodings for sequence positions p with d_model encoding dimensionality.
    
    Args:
        p (Tensor): Tensor of shape (B,) containing integer positions
        d_model (int): Dimensionality of the positional encoding
    
    Returns:
        pos_enc (Tensor): Tensor of shape (B, dim) containing sinusoidal positional encodings
    """
    half_dim = d_model // 2
    # ----------
    # => k = 0,1,2,\ldots,\frac{d_\text{model}}{2}-1
    # ----------
    k_vals = torch.arange(half_dim, device=p.device)
    # ----------
    # => \omega_k p = \frac{p}{10000^{2k/d_\text{model}}} = \exp(-2k/d_\text{model} \cdot \ln(10000))
    # ----------
    freqs = torch.exp(-2*k_vals/d_model * math.log(10000))              # (d_model/2,)
    freqs = freqs[None, :] * p[:, None].float()                         # (B, d_model/2)
    # ----------
    # => \text{PE}(p) = \{\sin(\omega_0 p),\sin(\omega_1 p),\ldots,\sin(\omega_{d_\text{model}/2-1}p),\cos(\omega_0 p),\cos(\omega_1 p),\ldots,\cos(\omega_{d_\text{model}/2-1}p)\}
    # ----------
    pos_enc = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=1)  # (B, d_model)
    return pos_enc