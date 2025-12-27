import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import wandb

from transformer.models.seq2seq.transformer_seq2seq import TransformerSeq2Seq
from transformer.utils.tokenizer import BPETokenizer
from transformer.data.datasets import Seq2SeqDataset
from transformer.training.trainer_seq2seq import TrainerSeq2Seq


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
    # Create BPETokenizer / Seq2SeqDataset
    # ----------
    vocab_size = config.data.vocab_size
    bpe = BPETokenizer(vocab_size)

    dataset = Seq2SeqDataset(bpe, config.data)

    # ----------
    # Initialize TransformerSeq2Seq model
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(
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
    # Create Trainer
    # ----------
    trainer = TrainerSeq2Seq(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        device=device
    )
    trainer.train(config.train.steps)


if __name__ == "__main__":
    main()