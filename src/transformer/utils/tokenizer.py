import json
import heapq
from tqdm import tqdm
from collections import Counter


class TokenHeap:
    def __init__(self, ids: bytes, merges: dict):
        self.heap = []
        self.merges = merges

        self.token = list(ids)
        self.prev = [-1] + list(range(len(ids) - 1))    # prev[i] = i - 1 with prev[0] = -1
        self.next = list(range(1, len(ids))) + [-1]     # next[i] = i + 1 with next[N-1] = -1
        self.alive = [True] * len(ids)

        self.init_heap()

    def init_heap(self):
        """
        
        """
        for i in range(len(self.token) - 1):
            j = self.next[i]
            pair = (self.token[i], self.token[j])

            if pair in self.merges:
                rank = self.merges[pair]["rank"]
                # -- Push (rank, left_index) to heap
                heapq.heappush(self.heap, (rank, i))

    def get_tokenized_ids(self):
        """
        
        """
        token_ids = [self.token[0]]

        j = self.next[0]

        while j != -1:
            token_ids.append(self.token[j])
            j = self.next[j]

        return token_ids
    
    def merge_heap(self):
        """
        
        """
        while len(self.heap) != 0:
            self.merge_pair()

    def merge_pair(self):
        """
        
        """
        # ---------
        # Pop min-rank pair / validate
        # ----------
        while True:
            # -- If heap is empty, tokenization is finished
            if len(self.heap) == 0:
                return
            
            # -- Pop until valid merge before applying
            rank, i = heapq.heappop(self.heap)

            if self.is_valid_merge(rank, i):
                # ---------
                # Apply merge to heap / recompute adjacent pairs
                # ----------
                self.apply_merge(i)
                self.update_adj_pairs(i)
                return

    def apply_merge(self, i: int):
        """
        
        """
        pair = (self.token[i], self.token[self.next[i]])
        merge_id = self.merges[pair]["id"]

        # -- Update left index to merged id
        self.token[i] = merge_id

        # ---------
        # Get next two links' indices
        # ----------
        j = self.next[i]
        k = self.next[j]

        # ---------
        # Mark next[i] as dead / remove links
        # ----------
        self.alive[j] = False
        self.next[j] = -1
        self.prev[j] = -1

        # ---------
        # Rewire next/prev links
        # ----------
        self.next[i] = k     # token[i] --> token[k]

        # -- Do not set prev link if k is past last token
        if k != -1:
            self.prev[k] = i     # token[k] --> token[i]

    def update_adj_pairs(self, i: int):
        """
        
        """
        # -- Left Pair: (token[i - 1], token[i])
        if self.prev[i] != -1:
            left_pair = (self.token[self.prev[i]], self.token[i])

            if left_pair in self.merges:
                left_rank = self.merges[left_pair]["rank"]
                heapq.heappush(self.heap, (left_rank, self.prev[i]))

        # -- Right Pair: (token[i], token[i + 1])
        if self.next[i] != -1:
            right_pair = (self.token[i], self.token[self.next[i]])

            if right_pair in self.merges:
                right_rank = self.merges[right_pair]["rank"]
                heapq.heappush(self.heap, (right_rank, i))

    def is_valid_merge(self, rank: int, i: int):
        """
        
        """
        # ---------
        # token[i] is still alive
        # ----------
        if not self.alive[i]:
            return False
        
        # ---------
        # next[i] exists and is alive
        # ----------
        j = self.next[i]

        if j == -1 or not self.alive[j]:
            return False
        
        # ---------
        # Pair is mergeable
        # ----------
        pair = (self.token[i], self.token[j])

        if pair not in self.merges:
            return False
        
        # ---------
        # Pair rank matches the popped rank
        # ----------
        if self.merges[pair]["rank"] != rank:
            return False
        
        return True


class BPETokenizer:
    def __init__(self, vocab_size: int):
        # ----------
        # Initialize byte vocab
        # ----------
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

    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, token):
        return self.vocab[token]
    
    def load_vocab(self, vocab_path: str):
        """
        
        """
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)

        # ---------
        # Initialize vocab with bytes 0-255
        # ----------
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        # ---------
        # Read merges from json
        # ----------
        merges = vocab_data["merges"]

        for merge in merges:
            self.merges[tuple(merge["pair"])] = {"id": merge["id"], "rank": merge["rank"]}
            self.vocab[merge["id"]] = self.encode_text(merge["text"])

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
        curr_size = 256

        merged_ids = ids
        curr_rank = 0

        with tqdm(total=self.vocab_size, initial=curr_size, desc="Building Vocab") as pbar:
         
            while curr_size < self.vocab_size - 3:
                # ----------
                # Get most common new token
                # ----------
                pair = self.get_new_pair(merged_ids)
                id = curr_size

                self.merges[pair] = {"id": id, "rank": curr_rank}
                self.vocab[id] = self.vocab[pair[0]] + self.vocab[pair[1]]

                # ----------
                # Merge pair in text
                # ----------
                merged_ids = self.merge_pair(merged_ids, pair, id)

                curr_size += 1
                curr_rank += 1

                pbar.update(1)

        # ----------
        # Add special tokens
        # ----------
        self.vocab[self.pad_id] = b"<pad>"
        self.vocab[self.bos_id] = b"<bos>"
        self.vocab[self.eos_id] = b"<eos>"

    def get_new_pair(self, ids: list[int]):
        """
        
        """
        # ----------
        # Count token pair occurrences / add most common
        # ----------
        pair_counts = Counter(zip(ids, ids[1:]))

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
        # ---------
        # Initialize heap
        # ----------
        token_heap = TokenHeap(ids, self.merges)

        # ---------
        # Merge tokens / get tokenized ids
        # ----------
        token_heap.merge_heap()
        tokenized_ids = token_heap.get_tokenized_ids()

        return tokenized_ids
    
    def encode_text(self, text: str) -> list[int]:
        """
        Converts a text string into a list of UTF-8 bytes
        """
        return text.encode("utf-8")
    
    def ids_to_string(self, ids: list[int]):
        """
        
        
        Args:
        
        
        Returns:
        
        """
        text = []

        for id in ids:
            # -- Don't include special tokens
            if id not in [self.bos_id, self.eos_id, self.pad_id]:
                text.append(self.vocab[id])

        text_str = b"".join(text).decode("utf-8", errors="replace")
        
        return text_str
    
    def save_vocab(self, save_path: str):
        """
        
        """
        merges_data = [
            {
                "pair": pair,
                "id": self.merges[pair]["id"],
                "rank": self.merges[pair]["rank"],
                "text": self.ids_to_string(pair)
            }
            for pair in self.merges
        ]
        vocab_data = {"vocab_size": self.vocab_size, "merges": merges_data}

        with open(save_path, 'w') as f:
            json.dump(vocab_data, f, indent=4)
