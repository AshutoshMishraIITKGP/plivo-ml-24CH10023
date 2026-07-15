import time
import os
import tracemalloc
from collections import Counter
import tokenizer as tokenizer_mod

def benchmark():
    print("=== Tokenizer Benchmark Utility ===")
    
    # Measure memory
    tracemalloc.start()
    
    t0 = time.time()
    tok = tokenizer_mod.load()
    load_time = time.time() - t0
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print(f"Tokenizer loaded in {load_time:.3f}s")
    print(f"Memory Usage: Peak = {peak_mem / 1024**2:.2f} MB")
    print(f"Reported Vocab Size: {tok.vocab_size}\n")
    
    english_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the future of technology.",
        "Byte pair encoding significantly compresses sequence length."
    ]
    
    hindi_sentences = [
        "नमस्ते दुनिया, यह एक परीक्षण है।",
        "कृत्रिम बुद्धिमत्ता भविष्य की तकनीक को बदल रही है।",
        "यह प्रतियोगिता बहुत ही दिलचस्प होने वाली है।"
    ]
    
    mixed_sentences = [
        "The translation of 'Hello' is 'नमस्ते'.",
        "We can mix English and हिंदी in the same sequence without issues."
    ]
    
    def test_sentences(name, sents):
        total_tokens = 0
        total_bytes = 0
        for s in sents:
            ids = tok.encode(s)
            assert tok.decode(ids) == s, f"Round-trip failed on: {s}"
            total_tokens += len(ids)
            total_bytes += len(s.encode("utf-8"))
        avg_tokens = total_tokens / len(sents)
        print(f"[{name}]")
        print(f"  Avg Tokens/Sentence: {avg_tokens:.2f}")
        print(f"  Avg Bytes/Token:     {total_bytes / total_tokens if total_tokens else 0:.2f}")
        print(f"  Compression Ratio:   {total_bytes / total_tokens if total_tokens else 0:.2f}x\n")

    test_sentences("English text", english_sentences)
    test_sentences("Hindi text", hindi_sentences)
    test_sentences("Mixed text", mixed_sentences)
    
    # Speed test on training corpus
    corpus_path = "../data/train_corpus.txt"
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        print(f"Evaluating full corpus ({len(text.encode('utf-8')) / 1024**2:.2f} MB)...")
        t0 = time.time()
        encoded = tok.encode(text)
        encode_time = time.time() - t0
        
        t0 = time.time()
        decoded = tok.decode(encoded)
        decode_time = time.time() - t0
        
        assert decoded == text, "Full corpus round-trip validation failed!"
        print("✓ Round-trip validation successful on full corpus.")
        
        print(f"Encoding Speed: {len(encoded) / encode_time:,.0f} tokens/s ({encode_time:.2f}s total)")
        print(f"Decoding Speed: {len(encoded) / decode_time:,.0f} tokens/s ({decode_time:.2f}s total)")
        
        total_bytes = len(text.encode("utf-8"))
        total_tokens = len(encoded)
        print(f"Full Corpus Compression Ratio: {total_bytes / total_tokens:.2f}x")
        
        # Utilization
        id_counts = Counter(encoded)
        used_vocab = len(id_counts)
        fallback_tokens = sum(count for i, count in id_counts.items() if i < 256)
        
        print(f"Fallback Frequency (raw byte tokens): {fallback_tokens}/{total_tokens} ({(fallback_tokens/total_tokens)*100:.1f}%)")

    # Parameter Budget Calculation
    print("\n=== Parameter Budget Analysis (Tied Embeddings) ===")
    budget = 2_000_000
    embd_dim = 160
    # Assuming tie_weights = True, so embedding params = vocab_size * embd_dim
    # Baseline non-embedding params from model.py ~1,257,920 for 4 layers.
    # But wait, we just want to compute the embedding parameters themselves!
    print("Assuming n_embd = 160 and tie_weights = True (Standard practice to save parameters).")
    print(f"{'Vocab Size':<12} | {'Embedding Params':<18} | {'% of 2M Budget':<16} | {'Remaining Budget':<18}")
    print("-" * 72)
    for vs in [512, 768, 1024, 1536, 2048]:
        emb_params = vs * embd_dim
        pct = (emb_params / budget) * 100
        rem = budget - emb_params
        print(f"{vs:<12} | {emb_params:<18,} | {pct:<15.1f}% | {rem:<18,}")

if __name__ == "__main__":
    benchmark()

