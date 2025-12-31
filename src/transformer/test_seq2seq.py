import os
import json
import argparse
import torch
from omegaconf import OmegaConf, DictConfig

from transformer.utils.tokenizer import BPETokenizer
from transformer.models.lm.transformer_lm import TransformerLM


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)


def main():
    # ---------
    # Parse arguments / load config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    test_config = load_config(args.config)

    train_dir = os.path.join(test_config.run.run_dir, "training")
    train_config = load_config(os.path.join(train_dir, "config.yml"))

    # ---------
    # Create testing dirs / save config
    # ----------
    test_dir = os.path.join(test_config.run.run_dir, "testing", test_config.run.name)
    os.makedirs(test_dir, exist_ok=True)

    save_config(test_config, os.path.join(test_dir, 'config.yml'))

    # ---------
    # Set device
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------
    # Load vocab
    # ----------
    vocab_size = train_config.data.vocab_size
    vocab_path = os.path.join(train_dir, "vocab.json")

    bpe = BPETokenizer(vocab_size)
    bpe.load_vocab(vocab_path)

    