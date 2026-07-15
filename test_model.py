from model import GPT, Config
import torch

cfg = Config()
print(f"Config: n_embd={cfg.n_embd}, n_layer={cfg.n_layer}, n_head={cfg.n_head}, "
      f"ffn_dim={cfg.ffn_dim}, block_size={cfg.block_size}, vocab_size={cfg.vocab_size}")

m = GPT(cfg)
n = m.n_params()
print(f"Total params: {n:,}")
print(f"Under 2M cap: {n <= 2_000_000} (margin: {2_000_000 - n:,})")
print(f"Weight tying: {cfg.tie_weights}")

# Forward pass
x = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
logits, _ = m(x)
print(f"Forward pass OK. Logits shape: {logits.shape}")

# Loss computation
t = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
_, loss = m(x, t)
print(f"Initial loss: {loss.item():.4f} (expected ~{torch.log(torch.tensor(float(cfg.vocab_size))).item():.4f})")

# Verify checkpoint save/load compatibility (mimics evaluate.py)
config_dict = {k: getattr(cfg, k) for k in dir(cfg)
               if not k.startswith("_") and not callable(getattr(cfg, k))}
print(f"Config dict keys: {sorted(config_dict.keys())}")

# Simulate evaluate.py's load_model
cfg2 = Config()
for k, v in config_dict.items():
    setattr(cfg2, k, v)
m2 = GPT(cfg2)
m2.load_state_dict(m.state_dict())
_, loss2 = m2(x, t)
print(f"Reload loss matches: {abs(loss.item() - loss2.item()) < 1e-5}")

print("\nAll phases PASS!")
