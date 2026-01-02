import os
import json
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
from tqdm import tqdm

from transformer.tokenization.bpe.model import BPEModel
from transformer.models.architectures.transformer_lm import TransformerLM
from transformer.data.datasets import LMDataset


class TrainerLM:
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            model: TransformerLM,
            bpe: BPEModel,
            optimizer: Optimizer,
            device: torch.device,
            train_loader: DataLoader,
            valid_loader: DataLoader,
            train_dir: str,
            logging_config: DictConfig,
            sample_config: DictConfig
    ):
        self.model = model
        self.bpe = bpe
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_dir = train_dir

        # -- Logging parameters
        self.wandb_enabled = logging_config.wandb.enable
        self.wandb_save_ckpt = logging_config.wandb.save_ckpt
        self.loss_steps = logging_config.loss_steps
        self.ckpt_steps = logging_config.ckpt_steps
        self.sample_steps = logging_config.sample_steps

        # -- Sampling parameters
        self.prompts = sample_config.prompts
        self.max_tokens = sample_config.max_tokens
        self.multinomial = sample_config.multinomial
        self.temperature = sample_config.temperature
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
                for input_ids, target_ids in self.train_loader:
                    input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)

                    loss = self.train_step(input_ids, target_ids)

                    # ----------
                    # Log loss / samples
                    # ----------
                    self.log_train_step(loss.item(), step, "train")

                    if step > 0 and step % self.ckpt_steps == 0:
                        self.save_checkpoint(step)

                    # ----------
                    # Test validation
                    # ----------
                    valid_loss = self.validate()
                    self.log_loss(valid_loss, step, "valid")

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
    
    @torch.no_grad()
    def validate(self):
        """
        
        """
        self.model.eval()

        valid_loss = 0.0
        num_batches = 0

        for input_ids, target_ids in tqdm(self.valid_loader, desc="Validation"):
            input_ids, target_ids = input_ids.to(self.device), target_ids.to(self.device)

            logits = self.model(input_ids)
            loss = self.compute_loss(logits, target_ids)

            valid_loss += loss.item()
            num_batches += 1

        valid_loss /= num_batches

        return valid_loss

    def compute_loss(self, logits: torch.Tensor, target_ids: torch.Tensor):
        """
        
        """
        # ----------
        # Flatten vocab logits / target ids for each token
        # ----------
        B, T, V = logits.shape
        logits = logits.view(B*T, V)
        target_ids = target_ids.view(B*T)

        return F.cross_entropy(logits, target_ids)
    
    def log_train_step(self, loss: float, step: int, label: str):
        """
        Logs batch loss, logs/saves samples, and saves checkpoint 
        if at the specified step count
        """
        # ----------
        # Log step loss
        # ----------
        if step > 0 and step % self.loss_steps == 0:
            self.log_loss(loss, step, label)

        # ----------
        # Generate / log sample
        # ----------
        if step > 0 and step % self.sample_steps == 0:
            self.log_sample(step)

    def log_loss(self, loss: float, step: int, label: str):
        """
        Logs loss to wandb dashboard
        """
        if self.wandb_enabled:
            wandb.log({f"{label} loss": loss}, step=step)

    def log_sample(self, step: int):
        """
        Generates sample text from input prompt and logs to wandb
        """
        self.samples[step] = {}
        print(f"\n\n========= Step {step} =========\n")

        for prompt in self.prompts:
            # ----------
            # Tokenize prompt
            # ----------
            prompt_ids = self.bpe.encode(prompt)

            # ----------
            # Generate output ids / convert to text
            # ----------
            output_ids = self.model.generate(
                prompt_ids=prompt_ids,
                device=self.device,
                block_size=self.dataset.block_size,
                max_tokens=self.max_tokens,
                multinomial=self.multinomial,
                temperature=self.temperature
            )

            output_text = self.bpe.decode(output_ids)
            self.samples[step][prompt] = output_text

            print(
                f"\n\n==============================\n",
                f"\n\n(Prompt): {prompt}"
                f"\n\n(Output): {output_text}",
                f"\n\n==============================\n"
            )

    def save_samples(self):
        """
        
        """
        save_path = os.path.join(self.train_dir, "samples.json")
        with open(save_path, 'w') as f:
            json.dump(self.samples, f, indent=4)

    def save_checkpoint(self, step: int):
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