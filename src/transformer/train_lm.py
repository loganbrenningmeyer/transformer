import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import wandb

from transformer.utils.tokenizer import BPETokenizer
from transformer.data.datasets import LMDataset
from transformer.models.lm.transformer_lm import TransformerLM
from transformer.training.trainer_lm import TrainerLM


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)

def main():
    # ----------
    # Parse Arguments / Load Config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    # ----------
    # Create Training Dirs / Save Config
    # ----------
    train_dir = os.path.join(config.run.runs_dir, config.run.name, "training")
    os.makedirs(os.path.join(train_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'figs'), exist_ok=True)

    save_config(config, os.path.join(train_dir, 'config.yml'))

    # ----------
    # Create BPETokenizer / LMDataset
    # ----------
    vocab_size = config.data.vocab_size
    bpe = BPETokenizer(vocab_size)

    dataset = LMDataset(bpe, config.data)

    # ----------
    # Initialize TransformerLM model
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerLM(
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        num_decoder_layers=config.model.num_decoder_layers,
        dropout=config.model.dropout,
        vocab_size=vocab_size
    )
    model.to(device)

    # ----------
    # Define Optimizer
    # ----------
    optimizer = torch.optim.AdamW(model.parameters(), config.train.lr)
    
    # ----------
    # Create TrainerLM / train model
    # ----------
    trainer = TrainerLM(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        device=device
    )
    trainer.train(config.train.steps)

    # ---------
    # Generate sample
    # ----------
    prompt = "ROMEO:"
    output = model.generate(
        bpe=bpe,
        prompt=prompt,
        block_size=config.data.block_size,
        max_tokens=256,
        device=device
    )
    print(f"output: {output}")


if __name__ == "__main__":
    main()