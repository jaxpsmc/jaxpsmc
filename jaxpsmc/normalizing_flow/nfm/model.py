"""Vector adapter for Ruscio et al., Normalizing Flow Maps, arXiv:2610.22711.

Requires the authors' JAX package (top-level import 'nfm') at revision
af456b1242ae29e88348cde789136db407340ec3 of francescoshox/NormalizingFlowMaps.
Import this adapter via jaxpsmc.normalizing_flow.nfm.model. Neural conditioners
use upstream FP32 computation. Affine particle arithmetic and Jacobian sums
use the particle dtype. Both directions return the full event log determinant.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from nfm.models.nfmd import NFMDMetaBlock, TimestepEmbedding
from nfm.models.permutation import PermutationFlip

from .config import NFMConfig


def _block(cfg: NFMConfig, index: int) -> NFMDMetaBlock:
    return NFMDMetaBlock(
        in_channels=1, channels=cfg.channels, num_patches=cfg.n_dim,
        num_layers=cfg.n_layers, num_classes=1, permutation=PermutationFlip(bool(index % 2)),
        clip_range=cfg.clip_range, dtype=jnp.float32, attn_impl="einsum",
    )


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class NFMFlow:
    """Trainable JAX PyTree. Only parameter arrays are dynamic leaves."""

    config: NFMConfig
    params: Any

    def tree_flatten(self):
        return (self.params,), self.config

    @classmethod
    def tree_unflatten(cls, config, children):
        return cls(config, children[0])

    @property
    def bijection(self):
        return self

    @classmethod
    def initialize(cls, config: NFMConfig, key: jax.Array):
        keys = jax.random.split(key, config.n_blocks + 2)
        embedding = TimestepEmbedding(config.channels)
        time = jnp.zeros((1,), jnp.float32)
        s_params = embedding.init(keys[0], time)["params"]
        t_params = embedding.init(keys[1], time + 1)["params"]
        tokens = jnp.zeros((1, config.n_dim, 1), jnp.float32)
        context = jnp.zeros((1, config.channels), jnp.float32)
        blocks = tuple(
            _block(config, i).init(keys[i + 2], tokens, None, context, context)["params"]
            for i in range(config.n_blocks)
        )
        return cls(config, (s_params, t_params, blocks))

    def _batch(self, x):
        x = jnp.asarray(x)
        if x.ndim < 1 or x.shape[-1] != self.config.n_dim:
            raise ValueError("expected particle array with final axis n_dim")
        if x.dtype not in (jnp.float32, jnp.float64):
            raise TypeError("expected float32 or float64 particle coordinates")
        return x.reshape((-1, self.config.n_dim)), x.shape

    def _contexts(self, s, batch_size):
        emb = TimestepEmbedding(self.config.channels)
        s_params, t_params, _ = self.params
        # Constant endpoint embeddings need only a single network evaluation.
        c_s = emb.apply({"params": s_params}, s.reshape(-1))
        c_t = emb.apply({"params": t_params}, jnp.ones((1,), jnp.float32))
        shape = (batch_size, self.config.channels)
        return jnp.broadcast_to(c_s, shape), jnp.broadcast_to(c_t, shape)

    def encode(self, x, s):
        """Apply the single-time map from source time s to terminal time 1.

        x is already at time s. The trainer constructs the stochastic
        interpolant and divides the returned logdet by n_dim for its loss.
        """
        values, shape = self._batch(x)
        s = jnp.asarray(s, dtype=jnp.float32)
        if s.ndim and s.shape != shape[:-1]:
            raise ValueError("s must be scalar or match the leading particle axes")
        contexts = self._contexts(s, values.shape[0])
        logdet = jnp.zeros(values.shape[:1], values.dtype)
        for i, params in enumerate(self.params[2]):
            block = _block(self.config, i)
            _, _, alpha, shift = block.apply(
                {"params": params}, values[..., None].astype(jnp.float32), None, *contexts
            )
            alpha = alpha[..., 0].astype(values.dtype)
            shift = shift[..., 0].astype(values.dtype)
            ordered = block.permutation(values)
            values = block.permutation((ordered - shift) * jnp.exp(-alpha))
            logdet = logdet - alpha.sum(axis=-1)
        return values.reshape(shape), logdet.reshape(shape[:-1])

    def transform_and_log_det(self, u, condition=None):
        if condition is not None:
            raise ValueError("single-time endpoint NFM-D is unconditional")
        return self.encode(u, 0.0)

    def inverse_and_log_det(self, theta, condition=None):
        if condition is not None:
            raise ValueError("single-time endpoint NFM-D is unconditional")
        values, shape = self._batch(theta)
        batch, dimension = values.shape
        determinant = jnp.zeros((batch,), values.dtype)
        # The upstream shifted conditioner leaves its first coordinate fixed.
        if dimension == 1:
            return values.reshape(shape), determinant.reshape(shape[:-1])
        contexts = self._contexts(jnp.zeros((1,), jnp.float32), batch)
        cfg = self.config
        cache_shape = (batch, dimension, cfg.n_heads, cfg.channels // cfg.n_heads)
        for i in reversed(range(cfg.n_blocks)):
            block = _block(cfg, i)
            ordered = block.permutation(values)
            variables = {"params": self.params[2][i]}
            empty = tuple(jnp.zeros(cache_shape, jnp.float32) for _ in range(cfg.n_layers))

            def reconstruct(index, carry):
                coords, logdet, keys, vals = carry
                # Upstream reverse_step writes into local Python containers.
                # Keep the loop carry functional and allocate only the active cache.
                k_cache = [{"cond": k, "idx": index} for k in keys]
                v_cache = [{"cond": v, "idx": index} for v in vals]
                alpha, shift, k_cache, v_cache = block.apply(
                    variables, coords[..., None], index, *contexts, y=None,
                    k_cache=k_cache, v_cache=v_cache, method=block.reverse_step,
                )
                alpha = alpha[:, 0, 0].astype(coords.dtype)
                shift = shift[:, 0, 0].astype(coords.dtype)
                next_index = index + 1
                decoded = coords[:, next_index] * jnp.exp(alpha) + shift
                return (coords.at[:, next_index].set(decoded), logdet + alpha,
                        tuple(c["cond"] for c in k_cache), tuple(c["cond"] for c in v_cache))

            ordered, determinant, _, _ = jax.lax.fori_loop(
                0, dimension - 1, reconstruct, (ordered, determinant, empty, empty)
            )
            values = block.permutation(ordered)
        return values.reshape(shape), determinant.reshape(shape[:-1])
