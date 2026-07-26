"""
Training-loop utilities: merging LoRA heads into the base weight, and
freezing the base weight from the optimizer.

Because Flax params are immutable pytrees (unlike the PyTorch original which
mutates `self.weight.data` in place), merging is a pure function you call
explicitly in your training loop:

    state, lte_state = train_step(state, lte_state, batch)   # normal grad step
    if merge_scheduler.step():
        state, lte_state = merge_params(state, lte_state)

This mirrors `lte.misc.merge.MergeCondition` from the original PyTorch repo.
"""
from __future__ import annotations

import jax.numpy as jnp
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
from typing import Any, Dict, Tuple

from .core import LoRAParams, merge_delta

PyTree = Any


class MergeScheduler:
    """Step-count based merge scheduler (mirrors lte.misc.merge.MergeCondition).

    Example::
        merge_scheduler = MergeScheduler(merge_steps=10)

        for batch in dataloader:
            params, opt_state, lte_state = train_step(params, opt_state, lte_state, batch)
            if merge_scheduler.step():
                params, lte_state = merge_params(params, lte_state, lora_config)
    """

    def __init__(self, merge_steps: int = 1):
        if merge_steps < 1:
            raise ValueError("merge_steps must be >= 1")
        self.merge_steps = int(merge_steps)
        self.clock = 0

    def peek(self) -> bool:
        """Whether the *next* call to `step()` will trigger a merge."""
        return (self.clock + 1) % self.merge_steps == 0

    def step(self) -> bool:
        """Increments the internal clock. Returns True if a merge should happen now."""
        self.clock += 1
        if self.clock % self.merge_steps == 0:
            self.clock = 0
            return True
        return False


def _find_lora_layer_paths(flat_params: Dict[Tuple[str, ...], jnp.ndarray]):
    """Finds module paths that contain a `lora_A` leaf, i.e. every
    MultiHeadLoRADense instance in the tree."""
    paths = set()
    for path in flat_params:
        if path[-1] == "lora_A":
            paths.add(path[:-1])
    return sorted(paths)


def merge_params(
    variables: Dict[str, PyTree],
    lora_alpha_by_path: Dict[Tuple[str, ...], float] = None,
    default_scaling: float = None,
) -> Dict[str, PyTree]:
    """Folds the average multi-head LoRA delta into each layer's base kernel.

    Args:
        variables: the full Flax variable dict, i.e.
            {"params": {...}, "lte_state": {...}} as returned by `model.init`
            / carried through your train state.
        lora_alpha_by_path: optional override of `scaling = lora_alpha / lora_r`
            per layer path, if layers use different configs. If None, scaling
            is recovered from lora_A/lora_B shapes assuming `default_scaling`
            was passed, or defaults to 1.0 (rank canceled by explicit scaling
            you already baked into training) -- **in practice, prefer calling
            `merge_params_with_scaling` below**, which is safer.
        default_scaling: fallback scaling to use for every layer.

    Returns:
        A new variables dict with base kernels updated and `lte_state`
        (prev_lora_A/B, merged flag) advanced.
    """
    scaling = default_scaling if default_scaling is not None else 1.0
    return merge_params_with_scaling(variables, lambda path: (
        lora_alpha_by_path.get(path, scaling) if lora_alpha_by_path else scaling
    ))


def merge_params_with_scaling(variables: Dict[str, PyTree], scaling_fn) -> Dict[str, PyTree]:
    """Same as `merge_params` but `scaling_fn(path) -> float` lets you supply
    the correct `lora_alpha / lora_r` per layer explicitly (recommended)."""
    params_flat = flatten_dict(variables["params"])
    lte_flat = flatten_dict(variables["lte_state"]) if "lte_state" in variables else {}

    for layer_path in _find_lora_layer_paths(params_flat):
        kernel_path = layer_path + ("kernel",)
        A_path = layer_path + ("lora_A",)
        B_path = layer_path + ("lora_B",)
        prev_A_path = layer_path + ("prev_lora_A",)
        prev_B_path = layer_path + ("prev_lora_B",)

        A = params_flat[A_path]
        B = params_flat[B_path]
        prev_lora = None
        if prev_A_path in lte_flat:
            prev_lora = LoRAParams(lte_flat[prev_A_path], lte_flat[prev_B_path])

        scaling = scaling_fn(layer_path)
        delta = merge_delta(LoRAParams(A, B), prev_lora, scaling)

        params_flat[kernel_path] = params_flat[kernel_path] + delta.astype(
            params_flat[kernel_path].dtype
        )
        lte_flat[prev_A_path] = A
        lte_flat[prev_B_path] = B

    new_variables = dict(variables)
    new_variables["params"] = unflatten_dict(params_flat)
    if lte_flat:
        new_variables["lte_state"] = unflatten_dict(lte_flat)
    return new_variables


def partition_lora_optimizer(
    variables: Dict[str, PyTree],
    lora_optimizer: optax.GradientTransformation,
) -> optax.GradientTransformation:
    """Wraps an optax optimizer so only `lora_A` / `lora_B` params receive
    gradient updates; `kernel` / `bias` (the base weights) are frozen and
    only ever change via `merge_params`.

    Usage::
        tx = partition_lora_optimizer(variables, optax.adamw(1e-3))
        opt_state = tx.init(variables["params"])
    """

    def label_fn(params):
        flat = flatten_dict(params)
        labels = {
            path: ("lora" if path[-1] in ("lora_A", "lora_B") else "frozen")
            for path in flat
        }
        return unflatten_dict(labels)

    return optax.multi_transform(
        {"lora": lora_optimizer, "frozen": optax.set_to_zero()},
        label_fn,
    )
