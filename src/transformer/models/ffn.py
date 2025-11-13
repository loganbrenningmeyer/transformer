import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with one hidden layer
    
    Args:
        d_model (int): Embedding dimensionality of the model
    
    Returns:
        x (Tensor): Output of position-wise feed-forward network of shape (B, T, d_model)
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x