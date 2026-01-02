import os
import argparse
import torch
from torch.utils.data import DataLoader
import wandb
from omegaconf import OmegaConf, DictConfig

from transformer.data.datasets import Seq2SeqDataset
from transformer.data.registry import SEQ2SEQ_BUILDERS
from transformer.tokenization.bpe.model import BPEModel
from transformer.models.architectures.transformer_seq2seq import TransformerSeq2Seq
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
    # Load dataset training splits
    # ----------
    splits = SEQ2SEQ_BUILDERS[config.data.dataset](config.data)

    # ----------
    # Initialize BPEModel
    # ----------
    vocab_size = config.tokenizer.vocab_size
    vocab_path = config.tokenizer.vocab_path

    bpe = BPEModel(vocab_size)

    if vocab_path is not None:
        bpe.load(vocab_path)
    else:
        train_texts = [src for src, _ in splits["train"]] + [tgt for _, tgt in splits["train"]]
        bpe.build_vocab(train_texts)
        
    bpe.save(os.path.join(train_dir, "vocab.json"))

    # ----------
    # Create DataLoader / save vocab
    # ----------
    context_length = config.data.context_length
    batch_size = config.data.batch_size

    train_dataset = Seq2SeqDataset(splits["train"], bpe, context_length)
    valid_dataset = Seq2SeqDataset(splits["valid"], bpe, context_length)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False, collate_fn=valid_dataset.collate_fn)

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
        vocab_size=vocab_size,
        context_length=config.data.context_length
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