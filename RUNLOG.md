# RUNLOG: Architecture & Training Ablation Studies

## Experiment 1: True Baseline
**Hypothesis:** Determine the starting Bits-Per-Byte (BPB) for the deliberately mediocre starter code over a 500-step limit.
**What changed:** Nothing. Ran `model.py` and `train.py` exactly as provided in the starter code, capped at 500 steps.
**Dev BPB before:** N/A
**Dev BPB after:** 2.9464
**Conclusion:** The baseline is extremely inefficient. It uses standard SGD-like constant Adam, a tiny vocabulary (256 bytes), and leaves ~45% of the parameter budget unused.

## Experiment 2: Tokenizer Upgrade (BPE)
**Hypothesis:** A trained BPE tokenizer will compress the Hindi/English corpus much better than raw bytes, reducing sequence lengths and packing more information into the model's context window.
**What changed:** Implemented a custom `BPETokenizer` (vocab 1024) with strict regex pre-tokenization to prevent cross-character merges. Ensured a byte-fallback to losslessly encode unseen UTF-8 text per the rules.
**Dev BPB before:** 2.9464
**Dev BPB after:** 2.7102
**Conclusion:** Massive win. The tokenizer achieved a 2.1x compression ratio over raw bytes (2.9x on Hindi text). This effectively doubles our context window and makes the metric calculation much more favorable.

## Experiment 3: Modernizing Layers (RMSNorm + SwiGLU + RoPE)
**Hypothesis:** Replacing legacy GPT-2 era components with LLaMA-era equivalents will improve parameter efficiency and representational capacity.
**What changed:** 
- Replaced `nn.LayerNorm` with `RMSNorm` (removes bias/mean, faster on CPU).
- Replaced `GELU` FFN with `SwiGLU`. Adjusted the intermediate dimension to `512` so the 3-matrix SwiGLU matches the baseline's 2-matrix parameter count.
- Replaced learned absolute position embeddings with Rotary Embeddings (RoPE).
**Dev BPB before:** 2.7102
**Dev BPB after:** 2.5020
**Conclusion:** SwiGLU's gating mechanism and RoPE's relative positional awareness clearly outperform the baseline equivalents, providing a clean boost in evaluation BPB while freeing up parameter space.

## Experiment 4: Deep & Narrow Architecture Swap
**Hypothesis:** Literature often suggests "deep and narrow" beats "shallow and wide" for small models. We attempted an aggressive architectural swap to 9 layers with an embedding dimension of 96 and 12 heads (head_dim=8) to see if extreme depth would yield better linguistic abstraction.
**What changed:** Reconfigured the model to `n_layer=9`, `n_embd=96`, `n_head=12`.
**Dev BPB before:** 2.5020
**Dev BPB after:** 2.8155
**Conclusion:** **FAILURE.** The loss plateaued quickly and the BPB was significantly worse. We realized that a `head_dim` of 8 severely choked the attention mechanism's ability to model complex mixed-language representations. Furthermore, the 96-dim residual stream was simply too narrow to propagate information effectively across 9 layers without degradation.

## Experiment 5: The Fix & Parameter Maximization (Final Run)
**Hypothesis:** We need to abandon the deep/narrow approach and scale back to a wider network that gives the attention heads enough representational bandwidth. We will also maximize the model size to perfectly hit the 2M cap.
**What changed:** 
- **The Fix:** Reverted to a wider, shallower network (`n_embd = 192`, `n_layer = 4`, `n_head = 6`), which restored the `head_dim` to a healthy 32. 
- **The Scaling:** Set `tie_weights = True` to share input/output embedding matrices (saving ~196k params). Added residual init scaling (`1/sqrt(2 * n_layer)`) to stabilize the wider residual stream.
**Dev BPB before:** 2.8155
**Dev BPB after:** 1.9101
**Conclusion:** The fix worked perfectly. The wider embedding dimension instantly removed the informational bottleneck, and tying the weights allowed us to allocate maximum capacity to the deep SwiGLU layers. This configuration maximizes the 2M cap and achieves our best possible score within 500 steps.
