import os
import json
import wandb
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from omegaconf import DictConfig
from tqdm import tqdm

from transformer.utils.tokenizer import BPEModel
from transformer.models.seq2seq.transformer_seq2seq import TransformerSeq2Seq


class TrainerSeq2Seq:
    """
    
    
    Args:
    
    
    Returns:
    
    """
    def __init__(
            self,
            model: TransformerSeq2Seq,
            bpe: BPEModel,
            optimizer: Optimizer,
            train_loader: DataLoader,
            device: torch.device,
            train_dir: str,
            logging_config: DictConfig,
            sample_config: DictConfig
    ):
        self.model = model
        self.bpe = bpe
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device
        self.train_dir = train_dir

        # -- Logging parameters
        self.wandb_enabled = logging_config.wandb.enable
        self.wandb_save_ckpt = logging_config.wandb.save_ckpt
        self.loss_interval = logging_config.loss_interval
        self.ckpt_interval = logging_config.ckpt_interval
        self.sample_interval = logging_config.sample_interval

        # -- Sampling parameters
        self.num_samples = sample_config.num_samples
        self.max_tokens = sample_config.max_tokens
        self.samples = {}

    def train(self, steps: int):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        step = 1
        epoch = 1

        while step <= steps:
            self.model.train()

            # ----------
            # Run training epoch
            # ----------
            for source, target in tqdm(self.train_loader, desc=f"Epoch {epoch}, Step {step}", unit="Step"):
                # ----------
                # Perform train step
                # ----------
                source, target = source.to(self.device), target.to(self.device)
                loss = self.train_step(source, target)

                # ----------
                # Log loss / save checkpoint 
                # ----------
                self.log_and_save(loss.item(), step)

                step += 1
            
            epoch += 1

        self.save_samples()

    def train_step(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        
        
        Args:
        
        
        Returns:
        
        """
        self.optimizer.zero_grad()

        # ----------
        # Separate target_in / target_out
        # ----------
        target_in = target[:, :-1]    # (B, T_tgt_in) : [<bos>, y_1, y_2, ..., y_T]
        target_out = target[:, 1:]    # (B, T_tgt_out): [y_1, y_2, ..., y_T, <eos>]

        # ----------
        # Define Encoder/Decoder padding masks 
        # ----------
        enc_pad_mask = (source == self.bpe.pad_id)
        dec_pad_mask = (target_in == self.bpe.pad_id)

        # ----------
        # Forward pass
        # ----------
        logits = self.model(source, target_in, enc_pad_mask, dec_pad_mask)      # (B, T_tgt_in, V)

        # ----------
        # Compute loss / update
        # ----------
        loss = self.compute_loss(logits, target_out)
        loss.backward()
        self.optimizer.step()

        return loss

    def compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        """
        
        """
        # ----------
        # Flatten vocab logits / target ids for each token
        # ----------
        B, T, V = logits.shape
        logits = logits.reshape(B*T, V)
        target = target.reshape(B*T)

        return F.cross_entropy(logits, target, ignore_index=self.bpe.pad_id)

    def log_and_save(self, loss: float, step: int):
        """
        
        """
        # ----------
        # Log step loss
        # ----------
        if step > 0 and step % self.loss_interval == 0:
            self.log_loss(loss, step)

        # ----------
        # Generate / log samples
        # ----------
        if step > 0 and step % self.sample_interval == 0:
            self.log_samples(step)

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

    def log_samples(self, step: int):
        """
        
        """
        block_size = self.train_loader.dataset.block_size
        max_tokens = self.max_tokens
        
        # ----------
        # Randomly sample source inputs / target outputs
        # ----------
        dataset = self.train_loader.dataset
        idxs = torch.randint(len(dataset), (self.num_samples,))
        batch = [dataset[i] for i in idxs]

        # -- Pad samples
        source, target = dataset.collate_fn(batch)
        source = source.to(self.device)

        # ----------
        # Generate batch of samples
        # ----------
        self.samples[step] = {"source": [], "output": [], "target": []}

        output_ids = self.model.generate(source, self.bpe.special_ids, block_size, max_tokens)

        for i in range(len(output_ids)):
            source_text = self.bpe.ids_to_string(source[i].detach().cpu().tolist())
            output_text = self.bpe.ids_to_string(output_ids[i].detach().cpu().tolist())
            target_text = self.bpe.ids_to_string(target[i].detach().cpu().tolist())
            self.samples[step]["source"].append(source_text)
            self.samples[step]["output"].append(output_text)
            self.samples[step]["target"].append(target_text)

        print(f"\n\n======== (Step {step}) Outputs ========")
        for i in range(len(output_ids)):
            print(f"\n\n---- Sample {i+1} ----")
            print(f"\n(Source): {self.samples[step]['source'][i]}")
            print(f"\n(Output): {self.samples[step]['output'][i]}")
            print(f"\n(Target): {self.samples[step]['target'][i]}")
        print(f"\n\n======================================\n")

    def save_samples(self):
        """
        
        """
        save_path = os.path.join(self.train_dir, "samples.json")
        with open(save_path, 'w') as f:
            json.dump(self.samples, f, indent=4)

    def save_and_log_checkpoint(self, step: int):
        """
        Saves model checkpoint at ckpt_path and logs artifact to wandb
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

