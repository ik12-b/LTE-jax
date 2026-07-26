"""
TPU-native LTE: one LoRA head per physical TPU core, using jax.pmap.

`toy_mlp_train.py` puts all `num_heads` LoRA heads on a single device and
uses vmap to run them "in parallel" logically -- fine for correctness, but
it doesn't get you any real speedup from a TPU v3-8's 8 cores, since only
one core is doing the vmap.

This script instead maps num_heads == jax.local_device_count() (8 on a
v3-8), replicates the frozen base weight across cores, gives every core its
OWN single LoRA (A, B) pair, and trains each core on its own data shard --
exactly the "each device trains a unique LoRA head on a different data
partition, periodically averaged into the main weights" setup from the LTE
paper (Huh et al. 2024, Sec 3).

Run on a TPU v3-8 VM / Kaggle TPU notebook:
    python examples/pmap_multi_device_sketch.py
"""
import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


LORA_R = 16
LORA_ALPHA = 32
MERGE_EVERY = 20
SCALING = LORA_ALPHA / LORA_R


class BaseDense(nn.Module):
    """Just the frozen base layer -- one copy, replicated across cores."""
    features: int

    @nn.compact
    def __call__(self, x):
        kernel = self.param("kernel", nn.initializers.lecun_normal(),
                             (x.shape[-1], self.features))
        bias = self.param("bias", nn.initializers.zeros, (self.features,))
        return jnp.dot(x, kernel) + bias, kernel


def init_single_head_lora(key, in_features, out_features, r):
    key_a, _ = jax.random.split(key)
    A = nn.initializers.orthogonal()(key_a, (in_features, r)) * jnp.sqrt(r / in_features)
    B = jnp.zeros((r, out_features))
    return A, B


def lora_delta(x, A, B, scaling):
    return scaling * (x @ A) @ B


def main():
    num_devices = jax.local_device_count()
    print(f"Found {num_devices} local devices (expect 8 on a TPU v3-8).")

    in_features, out_features = 256, 128
    base = BaseDense(out_features)

    key = jax.random.PRNGKey(0)
    base_vars = base.init(key, jnp.ones((1, in_features)))
    base_params = base_vars["params"]

    # one LoRA head per device
    lora_keys = jax.random.split(jax.random.PRNGKey(1), num_devices)
    lora_A, lora_B = jax.vmap(
        lambda k: init_single_head_lora(k, in_features, out_features, LORA_R)
    )(lora_keys)
    prev_A, prev_B = jnp.zeros_like(lora_A), jnp.zeros_like(lora_B)

    lora_opt = optax.adamw(1e-3)
    lora_opt_state = jax.vmap(lora_opt.init)((lora_A, lora_B))

    # replicate the (shared, frozen-except-at-merge) base params across devices
    # (pmap shards the leading axis across local devices automatically)
    replicated_base = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape), base_params
    )

    def loss_fn(lora_params, base_params, x, y):
        A, B = lora_params
        base_out, kernel = base.apply({"params": base_params}, x)
        out = base_out + lora_delta(x, A, B, SCALING)
        loss = optax.softmax_cross_entropy_with_integer_labels(out, y).mean()
        return loss

    @functools.partial(jax.pmap, axis_name="heads")
    def train_step(lora_params, base_params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(lora_params, base_params, x, y)
        updates, opt_state = lora_opt.update(grads, opt_state, lora_params)
        lora_params = optax.apply_updates(lora_params, updates)
        return lora_params, opt_state, loss

    @functools.partial(jax.pmap, axis_name="heads")
    def merge_step(lora_params, prev_lora_params, base_params):
        """Average each device's delta across all devices (pmean over the
        'heads' axis) and fold into the (replicated) base kernel. Every
        device ends up with the identical, updated base kernel."""
        A, B = lora_params
        prev_A, prev_B = prev_lora_params
        delta = SCALING * (jnp.einsum("ir,ro->io", A, B) - jnp.einsum("ir,ro->io", prev_A, prev_B))
        delta = jax.lax.pmean(delta, axis_name="heads")
        base_params = dict(base_params)
        base_params["kernel"] = base_params["kernel"] + delta.astype(base_params["kernel"].dtype)
        return base_params, (A, B)  # new prev = current

    rng = jax.random.PRNGKey(2)
    lora_params = (lora_A, lora_B)
    merge_clock = 0

    for step in range(100):
        rng, *shard_rngs = jax.random.split(rng, num_devices + 1)
        xs = jnp.stack([jax.random.normal(r, (32, in_features)) for r in shard_rngs])
        ys = jnp.stack([jax.random.randint(r, (32,), 0, out_features) for r in shard_rngs])

        lora_params, lora_opt_state, loss = train_step(
            lora_params, replicated_base, lora_opt_state, xs, ys
        )

        merge_clock += 1
        if merge_clock % MERGE_EVERY == 0:
            replicated_base, (prev_A, prev_B) = merge_step(
                lora_params, (prev_A, prev_B), replicated_base
            )
            print(f"step {step:4d} | loss {loss.mean():.4f} | merged across {num_devices} cores")
        elif step % 10 == 0:
            print(f"step {step:4d} | loss {loss.mean():.4f}")


if __name__ == "__main__":
    main()
