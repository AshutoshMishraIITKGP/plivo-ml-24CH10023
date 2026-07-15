import json
import os
import re
from collections import Counter
from typing import Dict, List, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int = 256, merges: Dict[Tuple[int, int], int] = None, vocab: Dict[int, bytes] = None):
        """
        Initializes the BPE Tokenizer.
        """
        self.vocab_size = vocab_size
        self.merges = merges if merges is not None else {}
        
        # Base vocabulary is always bytes 0-255
        self.vocab = vocab if vocab is not None else {i: bytes([i]) for i in range(256)}
        
        # We compile a strict non-backtracking regex to isolate words, punctuation, and spaces.
        self.pat = re.compile(r"\w+|[^\w\s]+|\s+")

    def _get_stats(self, vocab_dict: Counter) -> Counter:
        """Computes frequencies of adjacent pairs in the current word vocabulary."""
        counts = Counter()
        for word, freq in vocab_dict.items():
            for pair in zip(word, word[1:]):
                counts[pair] += freq
        return counts

    def _merge_vocab(self, vocab_dict: Counter, pair: Tuple[int, int], new_idx: int) -> Counter:
        """Applies a merge operation across all words in the vocabulary."""
        new_vocab = Counter()
        p0, p1 = pair
        for word, freq in vocab_dict.items():
            if len(word) < 2:
                new_vocab[word] += freq
                continue
                
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == p0 and word[i+1] == p1:
                    new_word.append(new_idx)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] += freq
        return new_vocab

    def train(self, text: str, target_vocab_size: int):
        """Trains the BPE tokenizer on the given text to the target vocabulary size."""
        assert target_vocab_size >= 256
        num_merges = target_vocab_size - 256
        
        # 1. Chunk text to preserve word boundaries
        text_chunks = self.pat.findall(text)
        
        # 2. Convert chunks to byte tuples to serve as our base vocabulary
        vocab_dict = Counter(tuple(chunk.encode("utf-8")) for chunk in text_chunks)
        
        # 3. Iteratively find the most frequent pair and merge it
        for i in range(num_merges):
            stats = self._get_stats(vocab_dict)
            if not stats:
                break
                
            # Find the pair with the highest frequency
            best_pair = max(stats, key=stats.get)
            idx = 256 + i
            
            # Apply merge to vocab_dict
            vocab_dict = self._merge_vocab(vocab_dict, best_pair, idx)
            
            # Record merge
            self.merges[best_pair] = idx
            self.vocab[idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
        self.vocab_size = 256 + len(self.merges)

    def encode(self, text: str) -> List[int]:
        """Encodes a UTF-8 string into a list of token IDs."""
        # if the input is empty, return an empty list
        if not text:
            return []
            
        # 1. Split text identically to training
        text_chunks = self.pat.findall(text)
        
        result = []
        for chunk in text_chunks:
            # 2. Convert chunk to bytes
            ids = list(chunk.encode("utf-8"))
            
            # 3. Iteratively apply known merges in priority order (lowest merge index first)
            while len(ids) >= 2:
                best_pair = None
                best_idx = float('inf')
                
                # Find the mergeable pair with the lowest merge index (meaning it was merged earliest during training)
                for i in range(len(ids) - 1):
                    pair = (ids[i], ids[i+1])
                    idx = self.merges.get(pair)
                    if idx is not None and idx < best_idx:
                        best_pair = pair
                        best_idx = idx
                
                if best_pair is None:
                    break # No more applicable merges
                    
                # Execute the merge
                new_ids = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                        new_ids.append(best_idx)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            result.extend(ids)
        return result

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a UTF-8 string. Guaranteed lossless."""
        b = bytearray()
        for i in ids:
            b.extend(self.vocab[i])
        # Use errors="replace" for safety, though trained tokens are guaranteed valid UTF-8 parts
        return bytes(b).decode("utf-8", errors="replace")

    def save(self, path: str):
        """Serializes the tokenizer metadata and merges to JSON."""
        # Keys in JSON must be strings, so we format tuples as "p0,p1"
        merges_str = {f"{p0},{p1}": idx for (p0, p1), idx in self.merges.items()}
        
        data = {
            "version": 1,
            "algorithm": "BPE_Regex_WordBoundary",
            "vocab_size": self.vocab_size,
            "base_vocab": 256,
            "merges": merges_str
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str) -> "BPETokenizer":
        """Loads a BPETokenizer from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        
        for k, idx in data.get("merges", {}).items():
            p0, p1 = map(int, k.split(","))
            merges[(p0, p1)] = idx
            vocab[idx] = vocab[p0] + vocab[p1]
            
        return cls(vocab_size=data["vocab_size"], merges=merges, vocab=vocab)

def load(path=None):
    """
    Module-level load function required by evaluate.py.
    """
    if path is None:
        # Default path relative to this script for grading environment
        path = os.path.join(os.path.dirname(__file__), "tokenizer.json")
    
    # If the file does not exist (e.g. before training), return an empty tokenizer
    if not os.path.exists(path):
        return BPETokenizer()
        
    return BPETokenizer.from_file(path)
