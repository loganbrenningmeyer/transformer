import os
import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf, DictConfig

from transformer.data.datasets import LMDataset
from transformer.data.registry import LM_BUILDERS
from transformer.tokenization.bpe.model import BPEModel
from transformer.models.architectures.transformer_lm import TransformerLM
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
            project=os.environ.get("WANDB_PROJECT", "TransformerLM"), 
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
    # Load dataset training splits
    # ----------
    splits = LM_BUILDERS[config.data.dataset]()
    train_text, valid_text = splits["train"], splits["valid"]

    # ----------
    # Initialize BPEModel / build vocab on train 
    # ----------
    vocab_size = config.data.vocab_size
    vocab_path = os.path.join(train_dir, "vocab.json")

    bpe = BPEModel(vocab_size)

    bpe.build_vocab([train_text])
    bpe.save(vocab_path)

    # ----------
    # Create LMDatasets / DataLoaders
    # ----------
    train_dataset = LMDataset(train_text, bpe, config.data.block_size)
    valid_dataset = LMDataset(valid_text, bpe, config.data.block_size)

    train_loader = DataLoader(train_dataset, config.data.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, config.data.batch_size, shuffle=False)

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
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        train_dir=train_dir,
        logging_config=config.logging,
        sample_config=config.sampling
    )
    trainer.train(config.train.steps)


if __name__ == "__main__":
    main()