import datasets
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from tqdm import tqdm

from transformer.tokenization.bpe.model import BPEModel


class Seq2SeqDataset(Dataset):
    def __init__(
            self, 
            texts: list[tuple[str, str]],
            bpe: BPEModel, 
            context_length: int
    ):
        self.bpe = bpe
        self.context_length = context_length

        # ----------
        # Tokenize text
        # ----------
        self.source_ids = [self.bpe.encode(src) for src, _ in texts]
        self.target_ids = [self.bpe.encode(tgt) for _, tgt in texts]

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """

        """
        # ----------
        # source: ids + [EOS]
        # ----------
        source = self.source_ids[idx][:self.context_length - 1]
        source = source + [self.bpe.eos_id]

        # ----------
        # target: [BOS] + ids + [EOS]
        # ----------
        target = self.target_ids[idx][:self.context_length - 2]
        target = [self.bpe.bos_id] + target + [self.bpe.eos_id]

        return (
            torch.tensor(source, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )
    
    def collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Pads source/target samples in batch to be of the same length
        """
        base_sources = [src for (src, _) in batch]
        base_targets = [tgt for (_, tgt) in batch]

        # ----------
        # Find max source/target length in batch
        # ----------
        src_max_len = max([len(src) for src in base_sources])
        tgt_max_len = max([len(tgt) for tgt in base_targets])

        # ----------
        # Pad samples to max length
        # ----------
        source = torch.full((len(batch), src_max_len), self.bpe.pad_id, dtype=torch.long)
        target = torch.full((len(batch), tgt_max_len), self.bpe.pad_id, dtype=torch.long)

        # ----------
        # Copy samples over
        # ----------
        for i, base_source in enumerate(base_sources):
            source[i, :len(base_source)] = base_source

        for i, base_target in enumerate(base_targets):
            target[i, :len(base_target)] = base_target

        return (source, target)
    

class LMDataset(Dataset):
    def __init__(self, text: str, bpe: BPEModel, context_length: int):
        self.bpe = bpe
        self.context_length = context_length

        # ----------
        # Tokenize text once
        # ----------
        ids = bpe.encode(text)
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.ids) - self.context_length
    
    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.context_length]
        y = self.ids[idx + 1 : idx + self.context_length + 1]

        return x, y