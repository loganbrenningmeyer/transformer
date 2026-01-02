import json

from transformer.tokenization.bpe.trainer import BPETrainer
from transformer.tokenization.bpe.tokenizer import BPETokenizer


class BPEModel:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

        # ----------
        # Define special token ids (final 3 of vocab_size)
        # ----------
        self.pad_id = vocab_size - 3
        self.bos_id = vocab_size - 2
        self.eos_id = vocab_size - 1
        self.special_ids = {
            "pad": self.pad_id,
            "bos": self.bos_id,
            "eos": self.eos_id
        }

    def build_vocab(self, texts: list[str]):
        # ----------
        # Create BPETrainer / train
        # ----------
        trainer = BPETrainer(self.vocab_size)
        self.vocab, self.merges = trainer.train(texts)

        self.vocab[self.pad_id] = b"<pad>"
        self.vocab[self.bos_id] = b"<bos>"
        self.vocab[self.eos_id] = b"<eos>"

    def encode(self, text: str):
        """
        
        """
        ids = self._text_to_bytes(text)

        tokenizer = BPETokenizer(ids, self.merges)
        tokenized_ids = tokenizer.tokenize()
        
        return tokenized_ids

    def decode(self, ids: list[int]):
        """
    
        """
        text = []

        for id in ids:
            # -- Don't include special tokens
            if id not in [self.bos_id, self.eos_id, self.pad_id]:
                text.append(self.vocab[id])

        text_str = b"".join(text).decode("utf-8", errors="replace")
        
        return text_str
    
    def load(self, vocab_path: str):
        """
        
        """
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        # ----------
        # Initialize vocab with bytes 0-255
        # ----------
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # ----------
        # Read merges from json
        # ----------
        merges = vocab_data["merges"]

        for merge in merges:
            self.merges[tuple(merge["pair"])] = {"id": merge["id"], "rank": merge["rank"]}
            self.vocab[merge["id"]] = self.encode(merge["text"])

    def save(self, save_path: str):
        """
        
        """
        merges_data = [
            {
                "pair": pair,
                "id": self.merges[pair]["id"],
                "rank": self.merges[pair]["rank"],
                "text": self.decode(pair)
            }
            for pair in self.merges
        ]
        vocab_data = {"vocab_size": self.vocab_size, "merges": merges_data}

        with open(save_path, 'w') as f:
            json.dump(vocab_data, f, indent=4)

    def _text_to_bytes(self, text: str):
        return text.encode("utf-8")
