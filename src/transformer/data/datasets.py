import datasets
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from tqdm import tqdm

from transformer.tokenization.bpe.model import BPEModel


class Seq2SeqDataset(Dataset):
    def __init__(
            self, 
            bpe: BPEModel, 
            data_config: DictConfig,
            vocab_path: str | None = None
    ):
        self.bpe = bpe
        self.block_size = data_config.block_size
        self.batch_size = data_config.batch_size
        self.source_key = data_config.source_key
        self.target_key = data_config.target_key

        # ----------
        # Load dataset
        # ----------
        if data_config.dataset == "ted_talks":
            self.data = []

            years = ["2014", "2015", "2016"]
            for year in years:
                dataset = datasets.load_dataset(
                    "IWSLT/ted_talks_iwslt", 
                    language_pair=(self.source_key, self.target_key),
                    year=year
                )
                self.data.extend(dataset["train"]["translation"])

            self.source_texts = []
            self.target_texts = []

            for sample in self.data:
                self.source_texts.append(sample[self.source_key])
                self.target_texts.append(sample[self.target_key])

        # ----------
        # Build vocabulary
        # ----------
        if vocab_path is not None:
            self.bpe.load(vocab_path)

        else:
            all_texts = self.source_texts + self.target_texts
            self.bpe.build_vocab(all_texts)

        # ----------
        # Tokenize text
        # ----------
        self.source_ids = [self.bpe.encode(text) for text in self.source_texts]
        self.target_ids = [self.bpe.encode(text) for text in self.target_texts]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """

        """
        # ----------
        # source: ids + [EOS]
        # ----------
        source = self.source_ids[idx][:self.block_size - 1]
        source = source + [self.bpe.eos_id]

        # ----------
        # target: [BOS] + ids + [EOS]
        # ----------
        target = self.target_ids[idx][:self.block_size - 2]
        target = [self.bpe.bos_id] + target + [self.bpe.eos_id]

        return (
            torch.tensor(source, dtype=torch.long),
            torch.tensor(target, dtype=torch.long)
        )
    
    def collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]):
        """
        Pads source/target samples in batch to be of the same length
        """
        base_sources = [src for (src, tgt) in batch]
        base_targets = [tgt for (src, tgt) in batch]

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


class LMDataset:
    def __init__(self, bpe: BPEModel, data_config: DictConfig):
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
        self.ids = torch.tensor(bpe.encode(self.train_text), dtype=torch.long)

    def get_batch(self):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        # ----------
        # Select random starting token index
        # ----------
        idxs = torch.randint(len(self.ids) - self.block_size - 1, (self.batch_size,))

        # ----------
        # Extract batch of input/target token ids
        # ----------
        inputs = torch.stack([self.ids[i : i+self.block_size] for i in idxs])
        targets = torch.stack([self.ids[i+1 : i+self.block_size+1] for i in idxs])

        return inputs, targets