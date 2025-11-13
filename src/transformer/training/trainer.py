import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
import wandb
from tqdm import tqdm

from transformer.models.transformer import Transformer


class Trainer:
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            model: Transformer,
            optimizer: Optimizer,
            dataloader: DataLoader,
            device: torch.device,
            train_config: DictConfig
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.device = device

    def train(self, epochs: int):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        pass

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        pass