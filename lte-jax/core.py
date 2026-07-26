"""
Core JAX functions for LTE (LoRA-the-Explorer), reimplemented from
https://github.com/minyoungg/LTE (PyTorch) into JAX/Flax.

Algorithm (faithful to the original "reset-less" DDP mode):
    - Each Dense/Linear layer gets `num_heads` independent LoRA (A, B) pairs.
    - During training, the batch is split into `num_heads` chunks; chunk h is
      routed through LoRA head h (parallel_lora_forward).
    - Every `merge_steps` optimizer steps, the *average* delta-weight across
      heads is folded into the frozen base weight (merge_parameters). Because
      LoRA heads are never reset, we subtract the delta they contributed at
      the previous merge so we don't double count it (this is the
      "reset-less" trick from the paper's appendix).

All functions here are pure and pytree-friendly so they compose cleanly with
jax.jit / jax.vmap / jax.pmap / shard_map.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import NamedTuple, Optional


class LoRAParams(NamedTuple):
    """Stacked multi-head LoRA parameters for a single Dense layer.

    Shapes follow Flax's [in, out] kernel convention (not PyTorch's [out, in]).
        A: [num_heads, in_features, r]
        B: [num_heads, r, out_features]
    """
    A: jnp.ndarray
    B: jnp.ndarray


def init_lora_params(
    key: jax.Array,
    in_features: int,
    out_features: int,
    r: int,
    num_heads: int,
    dtype=jnp.float32,
) -> LoRAParams:
    """Orthogonal init for A, zero init for B (same scheme as the original repo,
    so each head starts as a no-op and heads are decorrelated from init)."""
    key_a, _ = jax.random.split(key)

    def _orthogonal_stack(k, shape):
        # shape: [num_heads, fan_in, fan_out]
        keys = jax.random.split(k, shape[0])
        init_fn = jax.nn.initializers.orthogonal()
        mats = jax.vmap(lambda kk: init_fn(kk, shape[1:], dtype))(keys)
        # match the original repo's rescaling: p *= sqrt(fan_out / fan_in)
        scale = jnp.sqrt(shape[2] / shape[1])
        return mats * scale

    A = _orthogonal_stack(key_a, (num_heads, in_features, r))
    B = jnp.zeros((num_heads, r, out_features), dtype=dtype)
    return LoRAParams(A=A, B=B)


def parallel_lora_forward(x_chunked: jnp.ndarray, lora: LoRAParams) -> jnp.ndarray:
    """Applies each LoRA head to its own batch chunk.

    Args:
        x_chunked: [num_heads, chunk_batch, in_features]
        lora: LoRAParams with A: [H, in, r], B: [H, r, out]
    Returns:
        [num_heads, chunk_batch, out_features]
    """
    h = jnp.einsum("hbi,hir->hbr", x_chunked, lora.A)
    y = jnp.einsum("hbr,hro->hbo", h, lora.B)
    return y


def multihead_lora_train_forward(
    x: jnp.ndarray,
    lora: LoRAParams,
    prev_lora: LoRAParams,
    scaling: float,
    num_heads: int,
) -> jnp.ndarray:
    """Training-time LoRA contribution: chunk the batch across heads, run each
    head's *current minus previously-merged* delta, and reassemble.

    Args:
        x: [batch, in_features] with batch % num_heads == 0
        lora: current LoRA params
        prev_lora: LoRA params as they were at the last merge (for reset-less
            delta subtraction). Pass zeros_like(lora) if never merged.
        scaling: lora_alpha / lora_r
        num_heads: number of LoRA heads
    Returns:
        [batch, out_features] delta to add to the frozen base layer's output.
    """
    batch, in_features = x.shape
    if batch % num_heads != 0:
        raise ValueError(
            f"batch size ({batch}) must be divisible by num_heads ({num_heads}); "
            "pad or drop-remainder your batch, or replicate it num_heads times "
            "if you want every head to see the same data (MHLoRA-style eval)."
        )
    x_chunked = x.reshape(num_heads, batch // num_heads, in_features)

    y = parallel_lora_forward(x_chunked, lora)
    y_prev = parallel_lora_forward(x_chunked, prev_lora)
    delta = (y - y_prev) * scaling

    out_features = delta.shape[-1]
    return delta.reshape(batch, out_features)


def compute_average_delta_weight(lora: LoRAParams, scaling: float) -> jnp.ndarray:
    """Mean delta-weight across heads: scaling * mean_h(A_h @ B_h) -> [in, out]."""
    per_head = jnp.einsum("hir,hro->hio", lora.A, lora.B) * scaling
    return per_head.mean(axis=0)


def merge_delta(
    lora: LoRAParams,
    prev_lora: Optional[LoRAParams],
    scaling: float,
) -> jnp.ndarray:
    """Delta-weight to fold into the base kernel at merge time (reset-less:
    subtract what was already merged in previously)."""
    delta = compute_average_delta_weight(lora, scaling)
    if prev_lora is not None:
        delta = delta - compute_average_delta_weight(prev_lora, scaling)
    return delta
