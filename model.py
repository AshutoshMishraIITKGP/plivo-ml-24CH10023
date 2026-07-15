"""LLM Speedrun GPT — Modern architecture with RMSNorm, SwiGLU, and RoPE.

Changes from baseline:
  Phase 1: Weight tying + GPT-2 init (std=0.02) + residual scaling
  Phase 3: RMSNorm replaces LayerNorm (faster on CPU, LLaMA convention)
  Phase 4: SwiGLU FFN replaces GELU FFN (LLaMA/Gemma convention)
  Phase 5: RoPE replaces learned position embeddings (eliminates pos_emb params)
  Phase 6: Architecture scaled to 4L/192d/6h for ~1.97M params

evaluate.py compatibility: fully preserved. Config serialization works
via the existing dict comprehension in train.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Config:
    vocab_size = 1024     # BPE tokenizer
    block_size = 256      # context length (tokens)
    n_layer = 4
    n_head = 6
    n_embd = 192
    ffn_dim = 512         # SwiGLU intermediate dimension = (2/3)*4*n_embd
    dropout = 0.0
    tie_weights = True
    rope_theta = 10000.0
    bias = False          # no bias in Linear layers (modern convention)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich 2019).
    Used by LLaMA, Gemma, Mistral. Faster than LayerNorm on CPU."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (Su et al. 2021, RoFormer).
    Applied to Q and K tensors. Used by LLaMA, GPT-NeoX, Mistral."""
    def __init__(self, head_dim, max_seq_len=1024, theta=10000.0):
        super().__init__()
        # Precompute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        # Precompute cos/sin cache
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, head_dim]
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def forward(self, x, seq_len):
        # x shape: [B, n_head, T, head_dim]
        return self.cos_cache[:seq_len], self.sin_cache[:seq_len]


def apply_rope(q, k, cos, sin):
    """Apply rotary embeddings to Q and K tensors."""
    # q, k: [B, n_head, T, head_dim]
    # cos, sin: [T, head_dim]
    cos = cos[None, None, :, :]  # [1, 1, T, head_dim]
    sin = sin[None, None, :, :]
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def _rotate_half(x):
    """Split x into two halves and rotate."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)
        self.rope = rope

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        cos, sin = self.rope(q, T)
        q, k = apply_rope(q, k, cos, sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class SwiGLU(nn.Module):
    """Gated FFN with SiLU activation (Shazeer 2020, used by LLaMA).
    gate(x) = down_proj(silu(gate_proj(x)) * up_proj(x))
    Uses 3 matrices instead of 2, so ffn_dim is set to (2/3)*4*d for param parity."""
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, cfg.ffn_dim, bias=cfg.bias)
        self.down_proj = nn.Linear(cfg.ffn_dim, cfg.n_embd, bias=cfg.bias)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg, rope):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = SelfAttention(cfg, rope)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Shared RoPE module for all attention layers
        head_dim = cfg.n_embd // cfg.n_head
        rope = RotaryEmbedding(head_dim, max_seq_len=cfg.block_size,
                               theta=getattr(cfg, 'rope_theta', 10000.0))

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # No pos_emb — RoPE handles positions
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg, rope) for _ in range(cfg.n_layer))
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight

        self.apply(self._init)
        self._apply_residual_scaling()

    def _init(self, m):
        """GPT-2 style init: normal with std=0.02."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _apply_residual_scaling(self):
        """Scale output projections by 1/sqrt(2*n_layer) per GPT-2."""
        factor = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * factor)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=0.02 * factor)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.reshape(-1))
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
