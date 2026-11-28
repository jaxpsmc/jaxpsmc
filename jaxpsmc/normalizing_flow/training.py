"""Shared stage-wise fitting policy for adaptive particle flows."""

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from .diagnostics import fit_stats


class FlowFitState(NamedTuple):
    """A failed fit preserves the previous map and sets fit_ok to false."""

    flow: Any
    fitted: jax.Array
    fit_ok: jax.Array
    initial_n_effective: jax.Array
    last_fit_epochs: jax.Array

    diagnostics: Any = None

    @classmethod
    def initial(cls, flow: Any, initial_n_effective):
        return cls(flow, jnp.asarray(False), jnp.asarray(True),
                   jnp.asarray(initial_n_effective, jnp.int32), jnp.int32(0))


def _select_epochs(flow, moments, epoch_fn, patience, dtype, limit=5000):
    """Keep best validation checkpoint; at the epoch limit use the final iterate."""
    def active(carry):
        epoch, _, _, _, _, stale, finite = carry
        return (epoch < limit) & (stale < patience) & finite

    def step(carry):
        epoch, current, moments, best, best_loss, stale, _ = carry
        current, moments, loss = epoch_fn(epoch, current, moments)
        improved = loss < best_loss
        best = jax.lax.cond(improved, lambda: current, lambda: best)
        return (epoch + 1, current, moments, best, jnp.minimum(loss, best_loss),
                jnp.where(improved, 0, stale + 1), jnp.isfinite(loss))

    start = (jnp.int32(0), flow, moments, flow, jnp.asarray(jnp.inf, dtype),
             jnp.int32(0), jnp.asarray(True))
    epochs, final, _, best, _, stale, finite = jax.lax.while_loop(active, step, start)
    selected = jax.lax.cond(stale >= patience, lambda: best, lambda: final)
    return selected, epochs, finite, epochs - stale


def _run_epochs(flow, moments, epoch_fn, patience, dtype, limit=5000):
    return _select_epochs(flow, moments, epoch_fn, patience, dtype, limit)[:3]


def _logical_batch_size(initial_n_effective, count):
    # A singleton effective-size target still needs one row per update.
    return jnp.maximum(1, jnp.minimum(512, jnp.minimum(initial_n_effective // 2, count // 2)))


def _identity(flow):
    return flow


def _keep_initial_flow(state, particles, weights, rows, n_train):
    return state.flow


def _fit_partition(state, particles, weights, key, count, batch_loss, prepare, project, patience):
    capacity = len(weights)
    split_key, epoch_key = jax.random.split(key)
    permutation = jax.random.permutation(split_key, capacity)
    slots = jnp.nonzero(weights[permutation] > 0, size=capacity, fill_value=0)[0]
    rows = permutation[slots]
    n_train = count // 2
    n_valid = count - n_train
    width = max(1, min(512, capacity // 2))
    batch_size = _logical_batch_size(state.initial_n_effective, count)
    offsets = jnp.arange(width, dtype=jnp.int32)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=0.0))
    gradient = jax.grad(batch_loss)
    dtype = jnp.result_type(particles, weights)

    def shuffled_indices(key, start, size, max_size):
        # Randomize the order within this fixed holdout partition only.
        order = jax.random.permutation(key, max_size)
        valid = jnp.nonzero(order < size, size=max_size, fill_value=0)[0]
        return rows[start + order[valid]]

    def batch(indices, size, i):
        pos = i * batch_size + offsets
        idx = indices[jnp.minimum(pos, size - 1)]
        mass = jnp.where((offsets < batch_size) & (pos < size), weights[idx], 0)
        return particles[idx], mass

    def epoch(epoch_id, flow, moments):
        kt, kv, noise = jax.random.split(jax.random.fold_in(epoch_key, epoch_id), 3)
        train = shuffled_indices(kt, 0, n_train, capacity // 2)
        valid = shuffled_indices(kv, n_train, n_valid, (capacity + 1) // 2)

        def update(i, carry):
            current, opt_state = carry
            x, w = batch(train, n_train, i)
            grad = gradient(current, x, w, jax.random.fold_in(noise, i))
            delta, opt_state = optimizer.update(grad, opt_state, current)
            return project(optax.apply_updates(current, delta)), opt_state

        flow, moments = jax.lax.fori_loop(
            0, (n_train + batch_size - 1) // batch_size, update, (flow, moments)
        )

        def score(i, total):
            x, w = batch(valid, n_valid, i)
            z, logdet = flow.transform_and_log_det(x)
            dimension = flow.config.n_dim
            nll = 0.5 * (jnp.sum(z * z, axis=-1) + dimension * jnp.log(2 * jnp.pi)) - logdet
            # PocoMC first normalizes weights within each validation batch.
            scaled = w / w.max()
            return total + 1000 * jnp.sum((scaled / scaled.sum()) * nll)

        loss = jax.lax.fori_loop(0, (n_valid + batch_size - 1) // batch_size,
                                 score, jnp.zeros((), dtype)) / n_valid
        return flow, moments, loss

    flow = prepare(state, particles, weights, rows, n_train)
    selected, epochs, ok, best_epoch = _select_epochs(flow, optimizer.init(flow), epoch, patience, dtype)
    if state.diagnostics is not None:
        state = state._replace(diagnostics=fit_stats(
            state, selected, particles, weights, rows, count, best_epoch, epochs))
    finite = jax.tree.map(lambda p: jnp.all(jnp.isfinite(p)), selected)
    ok &= jnp.all(jnp.stack(jax.tree.leaves(finite)))
    return jax.lax.cond(
        ok, lambda: state._replace(flow=selected, fitted=ok, fit_ok=ok, last_fit_epochs=epochs),
        lambda: state._replace(fit_ok=ok, last_fit_epochs=epochs),
    )


@partial(jax.jit, static_argnames=("batch_loss", "prepare", "project"))
def refresh_flow(state, particles, weights, key, *, iteration, beta, n_active, patience,
                 batch_loss, prepare=_keep_initial_flow, project=_identity):
    """Fit after reweighting. All model-specific callbacks are static under JIT.

    batch_loss takes (flow, batch, weights, key). prepare receives the old
    state, the full particle and weight arrays, the shuffled row indices, and
    n_train. Only the first n_train indices are training observations.
    project is applied after each optimizer update.
    """
    particles, weights = jnp.asarray(particles), jnp.asarray(weights)
    if particles.ndim != 2 or particles.shape[1] != state.flow.config.n_dim:
        raise ValueError("particles must be a matrix with n_dim columns")
    if weights.shape != (particles.shape[0],) or not len(weights):
        raise ValueError("weights must have one entry per row and positive capacity")
    if particles.dtype not in (jnp.float32, jnp.float64) or not jnp.issubdtype(weights.dtype, jnp.floating):
        raise TypeError("expected float32/float64 coordinates and floating-point weights")

    def refresh(old):
        positive = weights > 0
        count = jnp.sum(positive, dtype=jnp.int32)
        valid = (jnp.all(jnp.isfinite(weights) & (weights >= 0))
                 & jnp.all(~positive[:, None] | jnp.isfinite(particles))
                 & (old.initial_n_effective > 0) & (n_active > 0))
        if len(weights) < 2:
            return old._replace(fit_ok=valid)
        return jax.lax.cond(valid & (count >= 2),
                            lambda: _fit_partition(old, particles, weights, key, count,
                                                   batch_loss, prepare, project, patience),
                            lambda: old._replace(fit_ok=valid))

    interval = jnp.maximum(1, state.initial_n_effective // (2 * n_active))
    due = (~state.fitted) | (iteration % interval == 0) | (beta == 1)
    return jax.lax.cond(due, refresh, lambda old: old, state)
