import os
import json
import wandb
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from omegaconf import DictConfig
from tqdm import tqdm

from transformer.utils.tokenizer import BPETokenizer
from transformer.models.lm.transformer_lm import TransformerLM
from transformer.data.datasets import LMDataset


class TrainerLM:
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            model: TransformerLM,
            bpe: BPETokenizer,
            optimizer: Optimizer,
            dataset: LMDataset,
            device: torch.device,
            train_dir: str,
            logging_config: DictConfig
    ):
        self.model = model
        self.bpe = bpe
        self.optimizer = optimizer
        self.dataset = dataset
        self.device = device
        self.train_dir = train_dir

        # -- Logging parameters
        self.wandb_enabled = logging_config.wandb.enable
        self.wandb_save_ckpt = logging_config.wandb.save_ckpt
        self.loss_interval = logging_config.loss_interval
        self.ckpt_interval = logging_config.ckpt_interval
        self.sample_interval = logging_config.sample_interval

        # -- Sampling parameters
        self.prompt = logging_config.sampling.prompt
        self.max_tokens = logging_config.sampling.max_tokens
        self.samples = {}

    def train(self, steps: int):
        """
        Trains the TransformerLM model for the specified number of steps.
        
        Parameters:
            steps (int): Total number of training steps
        """
        step = 1

        with tqdm(total=steps, initial=step, desc="Training TransformerLM") as pbar:

            while step <= steps:
                self.model.train()

                # ----------
                # Perform train step
                # ----------
                input_ids, target_ids = self.dataset.get_batch()
                input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)

                loss = self.train_step(input_ids, target_ids)

                # ----------
                # Log / save loss and checkpoint
                # ----------
                self.log_and_save(loss.item(), step)

                step += 1
                pbar.update(1)

        self.save_samples()

    def train_step(self, input_ids: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        self.optimizer.zero_grad()

        # ----------
        # Forward pass
        # ----------
        logits = self.model(input_ids)

        # ----------
        # Compute loss / update
        # ----------
        loss = self.compute_loss(logits, target_ids)
        loss.backward()
        self.optimizer.step()

        return loss

    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor):
        """
        
        """
        # ----------
        # Flatten vocab logits / target ids for each token
        # ----------
        B, T, V = logits.shape
        logits  = logits.view(B*T, V)
        target_ids = target_ids.view(B*T)

        return F.cross_entropy(logits, target_ids)
    
    def log_and_save(self, loss: float, step: int):
        """
        Logs batch loss, logs/saves samples, and saves checkpoint 
        if at the specified step count
        """
        # ----------
        # Log step loss
        # ----------
        if step > 0 and step % self.loss_interval == 0:
            self.log_loss(loss, step)

        # ----------
        # Generate / log sample
        # ----------
        if step > 0 and step % self.sample_interval == 0:
            self.log_sample(step)

        # ----------
        # Save checkpoint
        # ----------
        if step > 0 and step % self.ckpt_interval == 0:
            self.save_and_log_checkpoint(step)

    def log_loss(self, loss: float, step: int):
        """
        Logs loss to wandb dashboard
        """
        if self.wandb_enabled:
            wandb.log({"loss": loss}, step=step)

    def log_sample(self, step: int):
        """
        Generates sample text from input prompt and logs to wandb
        """
        # ----------
        # Generate / log text from input prompt
        # ----------
        output = self.model.generate(
            bpe=self.bpe,
            prompt=self.prompt,
            block_size=self.dataset.block_size,
            max_tokens=self.max_tokens,
            device=self.device
        )
        self.samples[step] = output

        print(
            f"\n\n==== (Step {step}) Output ====",
            f"\n\n{output}",
            f"\n\n==============================\n"
        )

    def save_samples(self):
        """
        
        """
        save_path = os.path.join(self.train_dir, "samples.json")
        with open(save_path, 'w') as f:
            json.dump(self.samples, f, indent=4)

    def save_and_log_checkpoint(self, step: int):
        """
        Saves model checkpoint at ckpt_path and logs artifact to wandb.
        """
        ckpt_path = os.path.join(self.train_dir, "checkpoints", f"model-step{step}.ckpt")

        torch.save({
            "model": self.model.state_dict(), 
            "optimizer": self.optimizer.state_dict(),
            "step": step
        }, ckpt_path)

        if self.wandb_enabled and self.wandb_save_ckpt:
            artifact = wandb.Artifact(
                name=f"model-step{step}",
                type="model"
            )
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)