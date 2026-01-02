import datasets
from omegaconf import DictConfig


def load_tiny_shakespeare() -> dict:
    dataset = datasets.load_dataset("karpathy/tiny_shakespeare")

    splits = {
        "train": dataset["train"]["text"][0],
        "valid": dataset["validation"]["text"][0]
    }

    return splits