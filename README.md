# 2,000 Step LLM Speedrun Submission

This repository contains the final deliverables and source code for the 2,000 Step LLM Speedrun. Our final configuration achieves a Bits-Per-Byte (BPB) of **1.9101** on the dev set within a strict 500-step constraint and 2M parameter cap.

## Reproducing the Results

### 1. Training the Model
To train the model from scratch using the provided 7MB corpus, run:
```bash
python train.py --data ../data/train_corpus.txt --steps 500 --out ckpt.pt
```
This will train the 1.96M parameter model using our custom SwiGLU + RMSNorm + RoPE architecture. The script relies on our custom BPE tokenizer which is loaded directly from `tokenizer.json`.

### 2. Evaluating the Model
To evaluate the final checkpoint and reproduce our BPB score, run:
```bash
python evaluate.py --checkpoint ckpt.pt --text_file ../data/dev_eval.txt
```

### 3. Tokenizer Re-Training (Optional)
Our BPE tokenizer configuration is fully contained in `tokenizer.json` and requires no external libraries. However, if you wish to completely recreate the tokenizer from scratch, you can run:
```bash
python train_tokenizer.py
```
This script parses `../data/train_corpus.txt`, learns the BPE merges up to a vocabulary size of 1024 (with a strict byte fallback), and outputs a fresh `tokenizer.json` file.

## Repository Structure
- `model.py`: The LLaMA-style transformer architecture (RMSNorm, SwiGLU, RoPE, purely in PyTorch).
- `train.py`: The training loop utilizing gradient accumulation, AdamW, and a cosine decay schedule.
- `evaluate.py`: The official evaluation script for computing BPB.
- `tokenizer.py`: Custom BPE Tokenizer class implementation.
- `tokenizer.json`: The trained BPE vocabulary and merge rules.
- `train_tokenizer.py`: The script used to train the BPE tokenizer on the training corpus.
- `ckpt.pt`: The final model checkpoint at step 500.
- `RUNLOG.md`: Detailed ablation journey and experiment logs.
- `NOTES.md`: Brief summary of the final configuration.
- `SUMMARY.html`: Comprehensive technical overview of the architecture and tokenization strategy.
- `benchmark_tokenizer.py` / `test_model.py`: Internal scripts used during development for profiling and validation.
