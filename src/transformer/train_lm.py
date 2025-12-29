import os
import argparse
import torch
import wandb
from omegaconf import OmegaConf, DictConfig

from transformer.utils.tokenizer import BPETokenizer
from transformer.data.datasets import LMDataset
from transformer.models.lm.transformer_lm import TransformerLM
from transformer.training.trainer_lm import TrainerLM


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)

def init_wandb(run_name: str):
    """
    Initializes wandb for logging, runs in offline mode on failure  
    """
    try:
        wandb.init(
            name=run_name,
            project=os.environ.get("WANDB_PROJECT", "transformer"), 
            entity=os.environ.get("WANDB_ENTITY", None)
        )
    except Exception as e:
        # -- Use offline if init fails
        print(f"---- wandb.init() failed, running offline: {e}")
        wandb.init(
            name=run_name,
            mode='offline'
        )

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

    save_config(config, os.path.join(train_dir, 'config.yml'))

    # ----------
    # Initialize wandb logging
    # ----------
    if config.logging.wandb.enable:
        init_wandb(config.run.name)

    # ----------
    # Initialize BPETokenizer
    # ----------
    vocab_size = config.data.vocab_size
    bpe = BPETokenizer(vocab_size)

    # ----------
    # Create LMDataset / save vocab
    # ----------
    dataset = LMDataset(bpe, config.data)

    vocab_path = os.path.join(train_dir, "vocab.json")
    bpe.save_vocab(vocab_path)

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
        bpe=bpe,
        optimizer=optimizer,
        dataset=dataset,
        device=device,
        train_dir=train_dir,
        logging_config=config.logging,
        sample_config=config.sampling
    )
    trainer.train(config.train.steps)


if __name__ == "__main__":
    main()