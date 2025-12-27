import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class BPETokenizer:
    def __init__(self, vocab_size: int):
        # ----------
        # Initialize byte vocab
        # ----------
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}

    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, token):
        return self.vocab[token]
    
    def encode_text(self, text: str) -> list[int]:
        """
        Converts a text string into a list of UTF-8 bytes
        """
        return list(text.encode("utf-8"))

    def build_vocab(self, texts: list[str]):
        """
        
        """
        # ----------
        # Encode texts
        # ----------
        ids = []

        for i in range(len(texts)):
            ids.extend(self.encode_text(texts[i]))
            # -- Add newline between samples
            if i < len(texts) - 1:
                ids.extend(self.encode_text("\n"))

        # ----------
        # Initialize vocab with bytes 0-255
        # ----------
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}
        curr_size = 256

        merged_ids = ids

        with tqdm(total=self.vocab_size, initial=curr_size, desc="Building Vocab") as pbar:
         
            while curr_size < self.vocab_size:
                # ----------
                # Get most common new token
                # ----------
                pair = self.get_new_pair(merged_ids)
                id = curr_size

                self.merges[pair] = id
                self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]

                # ----------
                # Merge pair in text
                # ----------
                merged_ids = self.merge_pair(merged_ids, pair, id)
                curr_size += 1

                pbar.update(1)

    def get_new_pair(self, ids: list[int]):
        """
        
        """
        # ----------
        # Count token pair occurrences
        # ----------
        pair_counts = defaultdict(int)

        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i+1])
            pair_counts[pair] += 1

        # ----------
        # Add most common token
        # ----------
        new_pair = max(pair_counts, key=pair_counts.get)

        return new_pair
    
    def merge_pair(self, ids: list[int], pair: tuple[int], id: int):
        """
        
        """
        merged_ids = []

        i = 0
        while i < len(ids):
            if i == len(ids) - 1:
                merged_ids.append(ids[i])
                break

            if (ids[i], ids[i+1]) == pair:
                merged_ids.append(id)
                i += 2
            else:
                merged_ids.append(ids[i])
                i += 1

        return merged_ids

    def tokenize(self, ids: list[int]):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])

        tokenized_ids = ids

        for (pair, id) in sorted_merges:
            tokenized_ids = self.merge_pair(tokenized_ids, pair, id)

        return tokenized_ids
    
    def ids_to_string(self, ids: list[int]):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        text = []

        for id in ids:
            text.append(self.vocab[id])

        text_str = b"".join(text).decode("utf-8", errors="replace")
        
        return text_str