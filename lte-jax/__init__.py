"""
lte-jax: a JAX/Flax reimplementation of LoRA-the-Explorer (LTE)
https://github.com/minyoungg/LTE (original PyTorch implementation).

    Huh et al., "Training Neural Networks from Scratch with Parallel
    Low-Rank Adapters", https://arxiv.org/abs/2402.16828

This is an independent re-port, not an official fork -- ported by inspecting
the reference PyTorch source (lte/ddp/ddp_lte.py, lte/ddp/linear.py,
lte/misc/merge.py) and reproducing the same math functionally in JAX.
"""

from .core import LoRAParams, init_lora_params, parallel_lora_forward, merge_delta
from .layers import MultiHeadLoRADense
from .optim import MergeScheduler, merge_params, merge_params_with_scaling, partition_lora_optimizer

__all__ = [
    "LoRAParams",
    "init_lora_params",
    "parallel_lora_forward",
    "merge_delta",
    "MultiHeadLoRADense",
    "MergeScheduler",
    "merge_params",
    "merge_params_with_scaling",
    "partition_lora_optimizer",
]
