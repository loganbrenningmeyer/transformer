import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import datasets
from omegaconf import DictConfig

from transformer.utils.tokenizer import BPETokenizer


class Seq2SeqDataset(Dataset):
    def __init__(self, bpe: BPETokenizer, data_config: DictConfig):
        self.block_size = data_config.block_size
        self.batch_size = data_config.batch_size
        self.source_key = data_config.source_key
        self.target_key = data_config.target_key

        # ----------
        # Load dataset
        # ----------
        if data_config.dataset == "ted_talks":
            dataset = datasets.load_dataset(
                "IWSLT/ted_talks_iwslt", 
                language_pair=("en","fr"),
                year="2014"
            )

            self.train_data = dataset["train"]
            self.source_texts = []
            self.target_texts = []

            for sample in self.train_data["translation"]:
                self.source_texts.append(sample[self.source_key])
                self.target_texts.append(sample[self.target_key])

        # ----------
        # Build vocabulary
        # ----------
        all_texts = self.source_texts + self.target_texts
        bpe.build_vocab(all_texts)

        # ----------
        # Tokenize text
        # ----------
        base_source_ids = [bpe.encode_text(text) for text in self.source_texts]
        base_target_ids = [bpe.encode_text(text) for text in self.target_texts]

        self.source_ids = [bpe.tokenize(ids) for ids in base_source_ids]
        self.target_ids = [bpe.tokenize(ids) for ids in base_target_ids]

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx: int):
        """

        """
        source = torch.tensor(self.source_ids[idx], dtype=torch.long)
        target = torch.tensor(self.target_ids[idx], dtype=torch.long)

        return source, target


class LMDataset:
    def __init__(self, bpe: BPETokenizer, data_config: DictConfig):
        self.block_size = data_config.block_size
        self.batch_size = data_config.batch_size

        # ----------
        # Load dataset
        # ----------
        if data_config.dataset == "tiny_shakespeare":
            dataset = datasets.load_dataset("karpathy/tiny_shakespeare")
            self.train_text = dataset["train"]["text"][0]
            self.valid_text = dataset["validation"]["text"][0]
            self.test_text = dataset["test"]["text"][0]

        # ----------
        # Build vocabulary
        # ----------
        bpe.build_vocab([self.train_text])

        # ----------
        # Tokenize text
        # ----------
        base_ids = bpe.encode_text(self.train_text)
        self.ids = torch.tensor(bpe.tokenize(base_ids), dtype=torch.long)

    def get_batch(self):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # ----------
        # Select random starting token index
        # ----------
        indices = torch.randint(len(self.ids) - self.block_size - 1, (self.batch_size,))

        # ----------
        # Extract batch of input/target token ids
        # ----------
        inputs = torch.stack([self.ids[i : i+self.block_size] for i in indices])
        targets = torch.stack([self.ids[i+1 : i+self.block_size+1] for i in indices])

        return inputs, targets