import argparse
import time
import tokenizer as tokenizer_mod

def main():
    parser = argparse.ArgumentParser(description="Train the BPE Tokenizer")
    parser.add_argument("--data", default="../data/train_corpus.txt", help="Path to training corpus")
    parser.add_argument("--vocab_size", type=int, default=1024, help="Target vocabulary size")
    parser.add_argument("--out", default="tokenizer.json", help="Path to save tokenizer metadata")
    args = parser.parse_args()
    
    print(f"Initializing BPETokenizer...")
    tok = tokenizer_mod.BPETokenizer()
    
    print(f"Reading corpus from {args.data}...")
    t0 = time.time()
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Read {len(text.encode('utf-8')):,} bytes in {time.time()-t0:.2f}s")
    
    print(f"Training tokenizer to vocab size {args.vocab_size}...")
    t1 = time.time()
    tok.train(text, args.vocab_size)
    print(f"Training completed in {time.time()-t1:.2f}s")
    
    print(f"Saving tokenizer to {args.out}...")
    tok.save(args.out)
    print("Done!")

if __name__ == "__main__":
    main()
