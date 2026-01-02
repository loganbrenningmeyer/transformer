import heapq


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