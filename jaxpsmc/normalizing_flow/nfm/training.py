"""Stage-wise fitting of the vector single-time NFM-D model.

The objective follows nfm/objectives/nfmd.py at the revision in model.py.
Particle weights enter once, in the reduction over each shuffled minibatch.
"""

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from .model import NFMFlow


class NFMFitState(NamedTuple):
    """A failed fit preserves the previous map and sets fit_ok to false."""

    flow: NFMFlow
    fitted: jax.Array
    fit_ok: jax.Array

    @classmethod
    def initial(cls, flow: NFMFlow):
        return cls(flow, jnp.asarray(False), jnp.asarray(True))


def _batch_loss(flow, particles, weights, key):
    time_key, noise_key = jax.random.split(key)
    times = jax.random.uniform(
        time_key, (particles.shape[0],), dtype=jnp.float32, maxval=flow.config.s_max
    )
    noise = jax.random.normal(noise_key, particles.shape, dtype=particles.dtype)
    source = (1 - times[:, None]) * particles + times[:, None] * noise
    z, logdet = flow.encode(source, times)
    per_particle = 0.5 * jnp.mean(jnp.square(z), axis=-1) - logdet / flow.config.n_dim
    # Each executed batch contains positive mass. Scaling first avoids a
    # large reciprocal for a batch whose weights are all small.
    scaled = weights / weights.max()
    return jnp.sum(per_particle * (scaled / scaled.sum()))


@partial(jax.jit, static_argnames=("interval", "optimizer_factory"))
def refresh_nfm(
    state: NFMFitState,
    particles: jax.Array,
    weights: jax.Array,
    key: jax.Array,
    *,
    iteration: jax.Array,
    beta: jax.Array,
    interval: int,
    optimizer_factory: Callable[[float], optax.GradientTransformation],
) -> NFMFitState:
    """Refresh after reweighting and before geometry fitting/resampling.

    Supply the selected mode's retained u-space rows and weights, and a
    dedicated fit key. iteration is PocoMC's post-reweight counter, including
    completed warmup batches. Compute interval once with config.training_interval.
    The optimizer factory takes config.learning_rate and must be JIT-static.
    A caller must inspect fit_ok after each refresh before continuing sampling.
    A skipped stage returns the input state. The sampler owns its random stream.
    """
    cfg = state.flow.config
    particles, weights = jnp.asarray(particles), jnp.asarray(weights)
    if particles.ndim != 2 or particles.shape[1] != cfg.n_dim or not particles.shape[0]:
        raise ValueError("particles must have shape (capacity, n_dim) with positive capacity")
    if weights.shape != (particles.shape[0],):
        raise ValueError("weights must have one entry per particle")
    if particles.dtype not in (jnp.float32, jnp.float64) or not jnp.issubdtype(weights.dtype, jnp.floating):
        raise TypeError("expected float32/float64 particles and floating-point weights")
    if interval < 1:
        raise ValueError("interval must be positive")

    def fit(old):
        positive = weights > 0
        count = jnp.sum(positive, dtype=jnp.int32)
        valid = (
            (count > 0) & jnp.all(jnp.isfinite(weights) & (weights >= 0))
            & jnp.all(~positive[:, None] | jnp.isfinite(particles))
        )
        width = min(cfg.batch_size, particles.shape[0])
        offsets = jnp.arange(width, dtype=jnp.int32)

        def train(_):
            optimizer = optimizer_factory(cfg.learning_rate)
            grad_loss = jax.grad(_batch_loss)

            def epoch(epoch_id, carry):
                epoch_key = jax.random.fold_in(key, epoch_id)
                order_key, batch_key = jax.random.split(epoch_key)
                order = jax.random.permutation(order_key, particles.shape[0])
                # Compact shuffled indices, not the particle matrix. The inner
                # loop runs only over positive-mass rows even for padded history.
                slots = jnp.nonzero(positive[order], size=particles.shape[0], fill_value=0)[0]
                indices = order[slots]

                def step(batch_id, carry):
                    flow, moments = carry
                    positions = batch_id * width + offsets
                    rows = indices[jnp.minimum(positions, count - 1)]
                    batch_weights = jnp.where(positions < count, weights[rows], 0)
                    grads = grad_loss(flow, particles[rows], batch_weights,
                                      jax.random.fold_in(batch_key, batch_id))
                    updates, moments = optimizer.update(grads, moments, flow)
                    return optax.apply_updates(flow, updates), moments

                return jax.lax.fori_loop(0, (count + width - 1) // width, step, carry)

            flow, _ = jax.lax.fori_loop(
                0, cfg.epochs, epoch, (old.flow, optimizer.init(old.flow))
            )
            finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(p)) for p in jax.tree.leaves(flow)]))
            return jax.lax.cond(
                finite, lambda: NFMFitState(flow, jnp.asarray(True), finite),
                lambda: old._replace(fit_ok=finite),
            )

        return jax.lax.cond(valid, train, lambda _: old._replace(fit_ok=valid), None)

    due = (~state.fitted) | (jnp.asarray(iteration) % interval == 0) | (jnp.asarray(beta) == 1)
    return jax.lax.cond(due, fit, lambda old: old, state)
