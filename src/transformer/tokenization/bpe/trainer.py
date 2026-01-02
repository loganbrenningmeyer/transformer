import heapq
from collections import defaultdict
from tqdm import tqdm


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