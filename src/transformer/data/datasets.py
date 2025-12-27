import os
import torch
import pandas as pd

from transformer.utils.vocab import BPETokenizer


# class TokenDataset:
#     def __init__(self, bpe: BPETokenizer, data_path: str, block_size: int=256, batch_size: int=8):
#         self.block_size = block_size
#         self.batch_size = batch_size

#         # ----------
#         # Build vocab
#         # ----------
#         base_ids = bpe.load_data(data_path)
#         bpe.build_vocab(base_ids)

#         # ----------
#         # Tokenize Text
#         # ----------
#         self.ids = torch.tensor(bpe.tokenize(base_ids), dtype=torch.long)

#     def get_batch(self):
#         """
        
        
#         Args:
        
        
#         Returns:
        
#         """
#         indices = torch.randint(len(self.ids) - self.block_size, (self.batch_size,))

#         source  = torch.stack([self.ids[i : i+self.block_size] for i in indices])
#         target = torch.stack([self.ids[i+1 : i+self.block_size+1] for i in indices])

#         return source, target


class LMDataset:
    def __init__(self, bpe: BPETokenizer, data_path: str, block_size: int=256, batch_size: int=8):
        self.block_size = block_size
        self.batch_size = batch_size

        # ----------
        # Build vocab
        # ----------
        base_ids = bpe.load_data(data_path)
        bpe.build_vocab(base_ids)

        # ----------
        # Tokenize Text
        # ----------
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