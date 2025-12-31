import json
import heapq
from tqdm import tqdm
from collections import defaultdict


class BPETrainer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        # ----------
        # vocab: Dictionary mapping id -> bytes
        # merges: Dictionary mapping (id_a, id_b) -> rank, new_id
        # ----------
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

        self.curr_size = 256
        self.curr_rank = 0

    def train(self, texts: list[str]):
        """
        
        """
        # ----------
        # Encode texts to bytes
        # ----------
        token_ids = []
        for text in texts:
            token_ids.extend(self.encode_text(text))

        # ----------
        # Initialize token information
        # ----------
        self.tokens = [id for id in token_ids]
        self.prev = [-1] + list(range(len(self.tokens) - 1))
        self.next = list(range(1, len(self.tokens))) + [-1]
        self.alive = [True] * len(self.tokens)

        # ----------
        # Get initial token pair counts / occurrences
        # ----------
        self.counts, self.occs = self.count_pairs(self.tokens)

        # ----------
        # Initialize count heap (-count for min)
        # ----------
        self.heap = []
        for pair, count in self.counts.items():
            heapq.heappush(self.heap, (-count, pair))

        # ----------
        # Build vocabulary
        # ----------
        with tqdm(desc="Building vocabulary", total=self.vocab_size - 3) as pbar:

            while self.curr_size != self.vocab_size - 3:
                pair, new_id = self.create_merge()
                # -- If heap is empty, end early
                if new_id is None:
                    break
                # -- Apply merge
                self.apply_merge(pair, new_id)

                pbar.update(1)

        return self.vocab, self.merges

    def create_merge(self):
        """
        
        """
        # ----------
        # Pop highest count
        # ----------
        while True:
            if len(self.heap) == 0:
                return None, None

            neg_count, pair = heapq.heappop(self.heap)
            count = -neg_count

            if self.counts[pair] == count:
                # -- Add new merged pair
                a, b = pair
                new_id, rank = self.curr_size, self.curr_rank

                self.vocab[new_id] = self.vocab[a] + self.vocab[b]
                self.merges[pair] = {"id": new_id, "rank": rank}
                self.curr_size += 1
                self.curr_rank += 1

                break

        return pair, new_id

    def apply_merge(self, pair: tuple[int, int], new_id: int):
        """
        
        """
        # ----------
        # Get indices of all pair occurrences
        # ----------
        idxs = self.occs[pair]

        for i in idxs:
            # ----------
            # Verify occurrence
            # ----------
            if self.is_valid_merge(pair, i):
                l = self.prev[i]
                j = self.next[i]
                k = self.next[j]

                # ----------
                # Substitute merged pair / kill next token
                # ----------
                self.tokens[i] = new_id

                self.alive[j] = False
                self.prev[j] = -1
                self.next[j] = -1

                # ----------
                # Rewire adjacent tokens
                # ----------
                self.next[i] = k

                if k != -1:
                    self.prev[k] = i

                # ----------
                # Decrement removed pairs / increment added pairs
                # ----------
                self.counts[pair] -= 1
                heapq.heappush(self.heap, (-self.counts[pair], pair))
        
                a, b = pair

                if l != -1:
                    l_pair_old = (self.tokens[l], a)
                    l_pair_new = (self.tokens[l], new_id)

                    self.counts[l_pair_old] -= 1
                    self.counts[l_pair_new] += 1
                    # -- Update occurrences index
                    self.occs[l_pair_new].append(l)
                    # -- Push new counts to heap
                    if self.counts[l_pair_old] > 0:
                        heapq.heappush(self.heap, (-self.counts[l_pair_old], l_pair_old))
                    heapq.heappush(self.heap, (-self.counts[l_pair_new], l_pair_new))
                
                if k != -1:
                    r_pair_old = (b, self.tokens[k])
                    r_pair_new = (new_id, self.tokens[k])

                    self.counts[r_pair_old] -= 1
                    self.counts[r_pair_new] += 1
                    # -- Update occurrences index
                    self.occs[r_pair_new].append(i)
                    # -- Push new count to heap
                    if self.counts[r_pair_old] > 0:
                        heapq.heappush(self.heap, (-self.counts[r_pair_old], r_pair_old))
                    heapq.heappush(self.heap, (-self.counts[r_pair_new], r_pair_new))

        # -- Remove pair occurrences
        self.occs[pair] = []
                
    def is_valid_merge(self, pair: tuple[int, int], i: int):
        """
        
        """
        # ----------
        # tokens[i] is still alive
        # ----------
        if not self.alive[i]:
            return False
        
        # ----------
        # next[i] exists and is alive
        # ----------
        j = self.next[i]

        if j == -1 or not self.alive[j]:
            return False
        
        # ----------
        # Pair matches current tokens
        # ----------
        if not (self.tokens[i] == pair[0] and self.tokens[j] == pair[1]):
            return False
        
        return True
        
    def count_pairs(self, tokens: list[int]):
        """
        
        """
        counts = defaultdict(int)
        occs = defaultdict(list)

        for i in range(len(tokens)):
            j = self.next[i]

            if j != -1:
                pair = (tokens[i], tokens[j])
                counts[pair] += 1
                occs[pair].append(i)

        return counts, occs

    def encode_text(self, text: str) -> list[int]:
        """
        Converts a text string into a list of UTF-8 bytes
        """
        return text.encode("utf-8")


class BPETokenizer:
    def __init__(self, ids: bytes, merges: dict):
        self.heap = []
        self.merges = merges

        self.tokens = list(ids)
        self.prev = [-1] + list(range(len(ids) - 1))    # prev[i] = i - 1 with prev[0] = -1
        self.next = list(range(1, len(ids))) + [-1]     # next[i] = i + 1 with next[N-1] = -1
        self.alive = [True] * len(ids)

        self.init_heap()

    def init_heap(self):
        """
        
        """
        for i in range(len(self.tokens) - 1):
            j = self.next[i]
            pair = (self.tokens[i], self.tokens[j])

            if pair in self.merges:
                rank = self.merges[pair]["rank"]
                # -- Push (rank, left_index) to heap
                heapq.heappush(self.heap, (rank, i))

    def get_tokenized_ids(self):
        """
        
        """
        tokenized_ids = [self.tokens[0]]

        j = self.next[0]

        while j != -1:
            tokenized_ids.append(self.tokens[j])
            j = self.next[j]

        return tokenized_ids
    
    def tokenize(self):
        """
        
        """
        while len(self.heap) != 0:
            self.merge_pair()

        return self.get_tokenized_ids()

    def merge_pair(self):
        """
        
        """
        # ----------
        # Pop min-rank pair / validate
        # ----------
        while True:
            # -- If heap is empty, tokenization is finished
            if len(self.heap) == 0:
                return
            
            # -- Pop until valid merge before applying
            rank, i = heapq.heappop(self.heap)

            if self.is_valid_merge(rank, i):
                # ----------
                # Apply merge to heap / recompute adjacent pairs
                # ----------
                self.apply_merge(i)
                self.update_adj_pairs(i)
                return

    def apply_merge(self, i: int):
        """
        
        """
        pair = (self.tokens[i], self.tokens[self.next[i]])
        merge_id = self.merges[pair]["id"]

        # -- Update left index to merged id
        self.tokens[i] = merge_id

        # ----------
        # Get next two links' indices
        # ----------
        j = self.next[i]
        k = self.next[j]

        # ----------
        # Mark next[i] as dead / remove links
        # ----------
        self.alive[j] = False
        self.next[j] = -1
        self.prev[j] = -1

        # ----------
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
            left_pair = (self.tokens[self.prev[i]], self.tokens[i])

            if left_pair in self.merges:
                left_rank = self.merges[left_pair]["rank"]
                heapq.heappush(self.heap, (left_rank, self.prev[i]))

        # -- Right Pair: (token[i], token[i + 1])
        if self.next[i] != -1:
            right_pair = (self.tokens[i], self.tokens[self.next[i]])

            if right_pair in self.merges:
                right_rank = self.merges[right_pair]["rank"]
                heapq.heappush(self.heap, (right_rank, i))

    def is_valid_merge(self, rank: int, i: int):
        """
        
        """
        # ----------
        # tokens[i] is still alive
        # ----------
        if not self.alive[i]:
            return False
        
        # ----------
        # next[i] exists and is alive
        # ----------
        j = self.next[i]

        if j == -1 or not self.alive[j]:
            return False
        
        # ----------
        # Pair is mergeable
        # ----------
        pair = (self.tokens[i], self.tokens[j])

        if pair not in self.merges:
            return False
        
        # ----------
        # Pair rank matches the popped rank
        # ----------
        if self.merges[pair]["rank"] != rank:
            return False
        
        return True
    

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
