import os
import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf, DictConfig

from transformer.utils.tokenizer import BPETokenizer
from transformer.data.datasets import Seq2SeqDataset
from transformer.models.seq2seq.transformer_seq2seq import TransformerSeq2Seq
from transformer.training.trainer_seq2seq import TrainerSeq2Seq


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
            project=os.environ.get("WANDB_PROJECT", "TransformerSeq2Seq"), 
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
    # Create DataLoader / save vocab
    # ----------
    train_dataset = Seq2SeqDataset(bpe, config.data, "2014")
    valid_dataset = Seq2SeqDataset(bpe, config.data, "2015")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.sampling.num_samples,
        shuffle=True,
        collate_fn=valid_dataset.collate_fn
    )

    vocab_path = os.path.join(train_dir, "vocab.json")
    bpe.save_vocab(vocab_path)

    # ----------
    # Initialize TransformerSeq2Seq model
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        num_encoder_layers=config.model.num_encoder_layers,
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
    # Create TrainerSeq2Seq / train model
    # ----------
    trainer = TrainerSeq2Seq(
        model=model,
        bpe=bpe,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        train_dir=train_dir,
        logging_config=config.logging,
        sample_config=config.sampling
    )
    trainer.train(config.train.steps)


if __name__ == "__main__":
    main()