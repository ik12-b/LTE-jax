from setuptools import setup, find_packages

setup(
    name="lte-jax",
    version="0.1.0",
    description="JAX/Flax reimplementation of LoRA-the-Explorer (LTE)",
    packages=find_packages(include=["lte_jax", "lte_jax.*"]),
    python_requires=">=3.9",
    install_requires=[
        "jax",
        "jaxlib",
        "flax>=0.8",
        "optax",
    ],
)
