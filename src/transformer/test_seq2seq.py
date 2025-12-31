import os
import json
import random
import argparse
import torch
from omegaconf import OmegaConf, DictConfig

from transformer.data.datasets import Seq2SeqDataset
from transformer.utils.tokenizer import BPETokenizer
from transformer.models.seq2seq.transformer_seq2seq import TransformerSeq2Seq


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
    # Load Seq2SeqDataset w/ training vocab
    # ----------
    vocab_size = train_config.data.vocab_size
    vocab_path = os.path.join(train_dir, "vocab.json")
    
    bpe = BPETokenizer(vocab_size)

    dataset = Seq2SeqDataset(
        bpe=bpe,
        data_config=train_config.data,
        vocab_path=vocab_path
    )

    # ---------
    # Load TransformerSeq2Seq Model
    # ----------
    model = TransformerSeq2Seq(
        d_model=train_config.model.d_model,
        num_heads=train_config.model.num_heads,
        num_encoder_layers=train_config.model.num_encoder_layers,
        num_decoder_layers=train_config.model.num_decoder_layers,
        dropout=train_config.model.dropout,
        vocab_size=vocab_size
    )

    ckpt_path = os.path.join(train_dir, "checkpoints", test_config.run.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval() 

    # ---------
    # Randomly sample source inputs / target outputs
    # ----------
    idxs = random.sample(range(len(dataset)), test_config.sampling.num_samples)
    batch = [dataset[i] for i in idxs]

    # -- Pad samples
    source, target = dataset.collate_fn(batch)
    source = source.to(device)

    # ---------
    # Generate batch of samples
    # ----------
    samples = {"source": [], "output": [], "target": []}

    output_ids = model.generate(
        source=source, 
        special_ids=bpe.special_ids, 
        block_size=train_config.data.block_size,
        max_tokens=test_config.sampling.max_tokens
    )

    for i in range(len(output_ids)):
        source_text = bpe.ids_to_string(source[i].detach().cpu().tolist())
        output_text = bpe.ids_to_string(output_ids[i].detach().cpu().tolist())
        target_text = bpe.ids_to_string(target[i].detach().cpu().tolist())
        samples["source"].append(source_text)
        samples["output"].append(output_text)
        samples["target"].append(target_text)

    print(f"\n\n======================================")
    for i in range(len(output_ids)):
        print(f"\n\n---- Sample {i+1} ----")
        print(f"\n(Source): {samples['source'][i]}")
        print(f"\n(Output): {samples['output'][i]}")
        print(f"\n(Target): {samples['target'][i]}")
    print(f"\n\n======================================\n")

    sample_path = os.path.join(test_dir, "samples.json")
    with open(sample_path, 'w') as f:
        json.dump(samples, f, indent=4)


if __name__ == "__main__":
    main()
