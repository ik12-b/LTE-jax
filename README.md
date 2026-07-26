# lte-jax

A JAX/Flax reimplementation of **LoRA-the-Explorer (LTE)**
([Huh et al. 2024, arXiv:2402.16828](https://arxiv.org/abs/2402.16828)),
ported from the reference PyTorch implementation at
[minyoungg/LTE](https://github.com/minyoungg/LTE).

> This is an independent port (built by reading the original PyTorch source
> for `ddp_lte.py`, `linear.py`, and `merge.py` and reproducing the same math
> functionally in JAX), not an official fork. Anthropic tooling doesn't have
> GitHub write access to fork the repo on your behalf — clone this folder,
> `git init`, and push it to your own GitHub fork/repo to keep it under
> version control.

## Why a port was needed

The original repo hardcodes `.cuda()` and PyTorch DDP/DMP primitives — it
has no TPU/XLA path, and porting it 1:1 via `torch_xla` would fight the
codebase's dynamic `torch.vmap`/`torch.func.functional_call` tricks that
aren't reliably XLA-traceable. Since JAX/Flax is TPU-native, this
reimplements the same **algorithm** (not the PyTorch code) directly in JAX:

- Multiple LoRA heads (`A`, `B`) per layer, stacked along a leading axis.
- Training-time forward: chunk the batch across heads (or, for the
  multi-core sketch, one head per physical device), apply each head's LoRA
  to its slice.
- Periodic **reset-less merge**: fold the *average* delta-weight across
  heads into the frozen base weight, subtracting whatever was already
  merged in previously (so LoRA heads never need to be reset to zero).

## Package layout

```
lte_jax/
  core.py     # pure JAX math: LoRA init, parallel forward, delta/merge math
  layers.py   # MultiHeadLoRADense — drop-in nn.Dense replacement
  optim.py    # MergeScheduler, merge_params, optimizer partitioning
examples/
  toy_mlp_train.py            # single-device/vmap version, runs anywhere
  pmap_multi_device_sketch.py # one LoRA head per TPU core via jax.pmap
```

## Quick start

```bash
pip install -e .
python examples/toy_mlp_train.py
```

```python
import flax.linen as nn
from lte_jax import MultiHeadLoRADense, MergeScheduler, merge_params_with_scaling, partition_lora_optimizer

class MyModel(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool):
        return MultiHeadLoRADense(
            features=4096, num_heads=8, lora_r=16, lora_alpha=32,
        )(x, train=train)
```

Key difference from Flax's usual pattern: variables are split across
`params` (base kernel/bias + trainable lora_A/lora_B) and `lte_state`
(mutable `prev_lora_A/B`, used only for the reset-less merge math). Use
`partition_lora_optimizer` so gradient descent only touches `lora_A`/`lora_B`
— the base weight only changes via `merge_params_with_scaling`.

## Running on Kaggle TPU v3-8

Two options, matching the two example scripts:

1. **`toy_mlp_train.py` style** — `num_heads` is just a `vmap` dimension on
   a single core. Simplest to adapt to an existing Flax model (e.g. your
   T5/CAT pipeline): swap `nn.Dense` for `MultiHeadLoRADense` in the model
   definition. You won't get extra parallelism from the other 7 cores this
   way unless you also wrap the whole thing in your existing `pmap`
   data-parallel training loop (heads then just add compute per-core, not
   across-core).

2. **`pmap_multi_device_sketch.py` style** — one LoRA head *per TPU core*
   (`num_heads == jax.local_device_count()`, i.e. 8 on a v3-8), each core
   training on its own data shard, deltas averaged via `jax.lax.pmean` at
   merge time. This is the setup closest to the original paper's "parallel
   compute nodes" framing, and is the one that actually uses all 8 cores
   for LoRA-head diversity rather than just batch data-parallelism.

   To integrate into your Qwen fine-tuning notebook (`02-train-embedding-2xt4-galore.ipynb`
   is on 2×T4, but if you move this piece to a TPU notebook): replace the
   toy `BaseDense`/single linear layer with the actual attention/MLP
   projection matrices you're adapting, keep `pmap(axis_name="heads")` for
   both the train step and merge step, and drive your existing data
   pipeline to hand each core a distinct shard.

## Differences from the original PyTorch repo

| | `minyoungg/LTE` (PyTorch) | `lte-jax` |
|---|---|---|
| Backend | CUDA / `torch.vmap` | JAX (CPU/GPU/TPU) |
| Head parallelism | `vmap` (DDP) or real multi-GPU (DMP) | `vmap` (single-core) or `pmap` (multi-core sketch) |
| Merge | in-place `self.weight.data +=` | pure function returning new params |
| Optimizer masking | `requires_grad_(False)` | `optax.multi_transform` label function |
| Config | `yacs` CfgNode | plain Python kwargs on the Flax module |

## Known limitations / TODO

- `pmap_multi_device_sketch.py` demonstrates the mechanism with a single toy
  `Dense` layer, not a full model — wiring it into a real transformer block
  (attention q/k/v/o, MLP up/down) is left for you to adapt, since it
  depends on your model's exact module structure.
- No `MergedLinear`-equivalent (the original repo's helper for fused QKV
  projections) yet.
- Only "step" merge scheduling is implemented (matches the original's
  default); other merge conditions from `lte.misc.merge` weren't ported.
- Not benchmarked against the PyTorch version's throughput/memory numbers —
  this prioritizes a faithful, testable re-port of the *algorithm* over
  matching kernel-level performance.
