import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
import wandb
from tqdm import tqdm

from transformer.models.transformer import TransformerLM
from transformer.data.datasets import LMDataset


class Trainer:
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            model: TransformerLM,
            optimizer: Optimizer,
            dataset: LMDataset,
            device: torch.device
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.device = device

    def train(self, steps: int):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        step = 1

        with tqdm(total=steps, initial=step, desc="Training Transformer") as pbar:

            while step <= steps:
                self.model.train()

                # ----------
                # Forward pass
                # ----------
                x = self.dataset.get_batch()
                x = x.to(self.device)

                logits = self.model(x)

                # ----------
                # Compute loss / update
                # ----------
                loss = self.compute_loss(logits, x)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if step % 100 == 0:
                    print(f"({step}) loss = {loss.item():.4f}")

                step += 1
                pbar.update(1)

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        """
        
        """
        B, T, C = logits.shape
        logits  = logits.view(B*T, C)
        target = target.view(B*T)

        return F.cross_entropy(logits, target)
