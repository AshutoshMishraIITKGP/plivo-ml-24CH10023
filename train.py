"""Production trainer for the LLM Speedrun competition.

Changes from baseline:
  * AdamW with betas=(0.9, 0.95) and weight_decay=0.1
  * Proper parameter groups (no weight decay on bias/norm/embedding)
  * Cosine LR schedule with linear warmup
  * Gradient clipping (max_norm=1.0)
  * Gradient accumulation for effective batch size 32

    python train.py --data ../data/train_corpus.txt --steps 2000 --out ckpt.pt
"""
import argparse
import math
import time

import torch

from model import GPT, Config
import tokenizer as tokenizer_mod

MAX_STEPS = 2000
MAX_PARAMS = 2_000_000


def get_batch(ids, block, batch, device):
    ix = torch.randint(len(ids) - block - 1, (batch,))
    x = torch.stack([ids[i:i + block] for i in ix])
    y = torch.stack([ids[i + 1:i + 1 + block] for i in ix])
    return x.to(device), y.to(device)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    # Cosine decay from max_lr to min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min_lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", default="ckpt.pt")
    ap.add_argument("--log_every", type=int, default=50)
    args = ap.parse_args()
    assert args.steps <= MAX_STEPS, f"cap: max {MAX_STEPS} steps"
    torch.manual_seed(args.seed)
    device = "cpu"

    text = open(args.data, encoding="utf-8").read()
    tok = tokenizer_mod.load()
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(text.encode('utf-8')):,} bytes -> {len(ids):,} tokens "
          f"(vocab {tok.vocab_size})")

    cfg = Config()
    cfg.vocab_size = tok.vocab_size
    model = GPT(cfg).to(device)
    n = model.n_params()
    print(f"model: {n:,} params")
    assert n <= MAX_PARAMS, f"cap: max {MAX_PARAMS:,} params"

    # Separate parameter groups: weight decay only on 2D weight tensors
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    print(f"optimizer: {len(decay_params)} decay params, {len(no_decay_params)} no-decay params")

    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=False)

    model.train()
    t0 = time.time()
    losses = []
    raw_loss_accum = 0.0

    for step in range(1, args.steps + 1):
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        opt.zero_grad(set_to_none=True)
        for micro_step in range(args.accum_steps):
            x, y = get_batch(ids, cfg.block_size, args.micro_batch, device)
            _, loss = model(x, y)
            # Scale loss by accumulation steps so gradients are averaged
            scaled_loss = loss / args.accum_steps
            scaled_loss.backward()
            raw_loss_accum += loss.item()

        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()

        avg_loss = raw_loss_accum / args.accum_steps
        losses.append(avg_loss)
        raw_loss_accum = 0.0

        if step % args.log_every == 0 or step == 1:
            recent = losses[-args.log_every:]
            avg = sum(recent) / len(recent)
            elapsed = time.time() - t0
            ms_per_step = elapsed / step * 1000
            tokens_per_sec = (args.micro_batch * args.accum_steps * cfg.block_size) / (elapsed / step)
            print(f"step {step:5d}  loss {avg:.4f}  lr {lr:.2e}  "
                  f"({ms_per_step:.0f} ms/step, {tokens_per_sec:.0f} tok/s)")

    # every public config attribute is saved — if you add fields to Config,
    # they ride along automatically and evaluate.py rebuilds the same model
    torch.save({"model": model.state_dict(),
                "config": {k: getattr(cfg, k) for k in dir(cfg)
                           if not k.startswith("_")
                           and not callable(getattr(cfg, k))},
                "steps": args.steps,
                "train_loss_curve": losses}, args.out)
    print(f"saved {args.out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
