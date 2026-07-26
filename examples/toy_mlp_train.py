"""
Minimal end-to-end example: train a small MLP with LTE-style parallel
multi-head LoRA, merging periodically into the base weights.

This runs on CPU, GPU, or a single TPU core as-is (num_heads is just a
vmap dimension here, not tied to physical devices). For a version that
maps each head to its own TPU core (closer to the original paper's
"parallel compute nodes" setup), see `pmap_multi_device_sketch.py`.

Run:
    python examples/toy_mlp_train.py
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lte_jax import MultiHeadLoRADense, MergeScheduler, merge_params_with_scaling, partition_lora_optimizer


NUM_HEADS = 8       # e.g. one per TPU v3-8 core, but works as a plain vmap dim here too
LORA_R = 16
LORA_ALPHA = 32
MERGE_EVERY = 20


class MLP(nn.Module):
    hidden: int = 512
    out_dim: int = 10

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = MultiHeadLoRADense(
            self.hidden, num_heads=NUM_HEADS, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        )(x, train=train)
        x = nn.relu(x)
        x = MultiHeadLoRADense(
            self.out_dim, num_heads=NUM_HEADS, lora_r=LORA_R, lora_alpha=LORA_ALPHA,
        )(x, train=train)
        return x


def main():
    key = jax.random.PRNGKey(0)
    model = MLP()

    batch_size = NUM_HEADS * 4  # must be divisible by NUM_HEADS
    dummy_x = jnp.ones((batch_size, 256))

    variables = model.init(key, dummy_x, train=True)
    tx = partition_lora_optimizer(variables, optax.adamw(1e-3))
    opt_state = tx.init(variables["params"])

    def loss_fn(params, lte_state, x, y):
        logits = model.apply({"params": params, "lte_state": lte_state}, x, train=True)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    @jax.jit
    def train_step(params, opt_state, lte_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, lte_state, x, y)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    merge_scheduler = MergeScheduler(merge_steps=MERGE_EVERY)
    scaling = LORA_ALPHA / LORA_R

    params, lte_state = variables["params"], variables.get("lte_state", {})
    rng = jax.random.PRNGKey(1)

    for step in range(200):
        rng, data_rng, label_rng = jax.random.split(rng, 3)
        x = jax.random.normal(data_rng, (batch_size, 256))
        y = jax.random.randint(label_rng, (batch_size,), 0, 10)

        params, opt_state, loss = train_step(params, opt_state, lte_state, x, y)

        if merge_scheduler.step():
            merged = merge_params_with_scaling(
                {"params": params, "lte_state": lte_state}, lambda path: scaling
            )
            params, lte_state = merged["params"], merged["lte_state"]
            print(f"step {step:4d} | loss {loss:.4f} | merged LoRA heads into base weights")
        elif step % 10 == 0:
            print(f"step {step:4d} | loss {loss:.4f}")


if __name__ == "__main__":
    main()
