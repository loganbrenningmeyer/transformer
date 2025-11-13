import os
import argparse
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import wandb


def load_config(config_path):
    config = OmegaConf.load(config_path)
    return config

def main():
    return

if __name__ == "__main__":
    main()