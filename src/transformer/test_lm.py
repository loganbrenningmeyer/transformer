import os
import json
import argparse
import torch
from omegaconf import OmegaConf, DictConfig

from transformer.utils.tokenizer import BPEModel
from transformer.models.lm.transformer_lm import TransformerLM


def load_config(config_path: str) -> DictConfig:
    config = OmegaConf.load(config_path)
    return config

def save_config(config: DictConfig, save_path: str):
    OmegaConf.save(config, save_path)
    

def main():
    # ----------
    # Parse arguments / load config
    # ----------
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    test_config = load_config(args.config)

    train_dir = os.path.join(test_config.run.run_dir, "training")
    train_config = load_config(os.path.join(train_dir, "config.yml"))

    # ----------
    # Create testing dirs / save config
    # ----------
    test_dir = os.path.join(test_config.run.run_dir, "testing", test_config.run.name)
    os.makedirs(test_dir, exist_ok=True)

    save_config(test_config, os.path.join(test_dir, 'config.yml'))

    # ----------
    # Set device
    # ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------
    # Load vocab
    # ----------
    vocab_size = train_config.data.vocab_size
    vocab_path = os.path.join(train_dir, "vocab.json")

    bpe = BPEModel(vocab_size)
    bpe.load_vocab(vocab_path)

    # ----------
    # Load TransformerLM model
    # ----------
    model = TransformerLM(
        d_model=train_config.model.d_model,
        num_heads=train_config.model.num_heads,
        num_decoder_layers=train_config.model.num_decoder_layers,
        dropout=train_config.model.dropout,
        vocab_size=vocab_size
    )

    ckpt_path = os.path.join(train_dir, "checkpoints", test_config.run.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    # ----------
    # Generate / save samples
    # ----------
    prompts = test_config.sampling.prompts

    samples = {}

    for prompt in prompts:
        prompt_ids = bpe.encode_text(prompt)
        prompt_ids = bpe.tokenize(prompt_ids)

        output_ids = model.generate(
            prompt_ids=prompt_ids,
            device=device,
            block_size=train_config.data.block_size,
            max_tokens=test_config.sampling.max_tokens,
            multinomial=test_config.sampling.multinomial,
            temperature=test_config.sampling.temperature
        )

        output_text = bpe.ids_to_string(output_ids)
        samples[prompt] = output_text

        print(
            f"\n\n==============================\n",
            f"\n\n(Prompt): {prompt}"
            f"\n\n(Output): {output_text}",
            f"\n\n==============================\n"
        )

    sample_path = os.path.join(test_dir, "samples.json")
    with open(sample_path, 'w') as f:
        json.dump(samples, f, indent=4)


if __name__ == "__main__":
    main()