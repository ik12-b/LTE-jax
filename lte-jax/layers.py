"""
Flax layers for LTE.

Flax params are immutable pytrees, unlike PyTorch's in-place nn.Module, so the
"merge into the base weight" step can't happen silently inside `__call__`.
Instead:

    1. `MultiHeadLoRADense` is a drop-in replacement for `nn.Dense`. During
       `train=True` it adds the parallel multi-head LoRA delta on top of the
       frozen base kernel/bias. During `train=False` (eval/inference) it
       just uses the base kernel/bias as-is (equivalent to "already merged").
    2. Base kernel/bias live in the `params` collection like any other Flax
       param, but you mask them out of the optimizer (see `partition_lora`
       in optim.py) so only lora_A / lora_B get gradients.
    3. `prev_lora_A` / `prev_lora_B` (needed for the reset-less merge trick)
       and a `merged` flag live in a separate `lte_state` mutable collection,
       since they change outside of gradient descent.
    4. Call `merge_params(params, lte_state, config)` (see optim.py) at your
       chosen cadence (e.g. via `MergeScheduler`) to fold the LoRA delta into
       the base kernel and update `lte_state`. This is a pure function you
       call in your training loop / train_state, not something Flax's
       `__call__` does for you.

Usage (see examples/qwen_style_mlp.py for a fuller example):

    class MyBlock(nn.Module):
        @nn.compact
        def __call__(self, x, train: bool):
            x = MultiHeadLoRADense(
                features=4096, num_heads=8, lora_r=16, lora_alpha=32,
            )(x, train=train)
            return x
"""
from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn
from typing import Optional

from .core import LoRAParams, init_lora_params, multihead_lora_train_forward


class MultiHeadLoRADense(nn.Module):
    """Drop-in replacement for `nn.Dense` with parallel multi-head LoRA.

    Attributes:
        features: output feature size (like nn.Dense).
        num_heads: number of parallel LoRA heads.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling numerator (scaling = lora_alpha / lora_r).
        use_bias: whether the base layer has a bias.
        dtype: compute dtype.
    """
    features: int
    num_heads: int = 8
    lora_r: int = 16
    lora_alpha: int = 32
    use_bias: bool = True
    dtype: Optional[jnp.dtype] = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        in_features = x.shape[-1]
        scaling = self.lora_alpha / self.lora_r

        # --- base (frozen, main) weight: a normal Flax param ---
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (in_features, self.features),
            self.param_dtype,
        )
        bias = None
        if self.use_bias:
            bias = self.param(
                "bias", nn.initializers.zeros, (self.features,), self.param_dtype
            )

        # --- LoRA heads: trainable params ---
        lora_A = self.param(
            "lora_A",
            lambda key, shape: init_lora_params(
                key, in_features, self.features, self.lora_r, self.num_heads,
                self.param_dtype,
            ).A,
            (self.num_heads, in_features, self.lora_r),
        )
        lora_B = self.param(
            "lora_B",
            nn.initializers.zeros,
            (self.num_heads, self.lora_r, self.features),
            self.param_dtype,
        )

        # --- reset-less merge bookkeeping: mutable, not trained by gradients ---
        prev_lora_A = self.variable(
            "lte_state", "prev_lora_A", lambda: jnp.zeros_like(lora_A)
        )
        prev_lora_B = self.variable(
            "lte_state", "prev_lora_B", lambda: jnp.zeros_like(lora_B)
        )

        out = jnp.dot(x, kernel)
        if bias is not None:
            out = out + bias

        if train:
            lead_shape = x.shape[:-1]
            x2d = x.reshape(-1, in_features)
            delta = multihead_lora_train_forward(
                x2d,
                LoRAParams(lora_A, lora_B),
                LoRAParams(prev_lora_A.value, prev_lora_B.value),
                scaling=scaling,
                num_heads=self.num_heads,
            )
            delta = delta.reshape(*lead_shape, self.features)
            out = out + delta

        return out
