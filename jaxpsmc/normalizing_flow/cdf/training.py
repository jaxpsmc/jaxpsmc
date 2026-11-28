"""Stage-wise maximum-likelihood fitting of particle-vector CDFlow."""

import jax
import jax.numpy as jnp

from ..training import FlowFitState as CDFlowFitState
from ..training import refresh_flow
from .model import CDFlow, _forward


def _batch_loss(flow, particles, weights, key):
    """Gaussian-base negative log likelihood, omitting its constant."""
    z, logdet = flow.transform_and_log_det(particles)
    losses = 0.5 * jnp.sum(jnp.square(z), axis=-1) - logdet
    scaled = weights / jnp.max(weights)
    return 1000.0 * jnp.sum((scaled / jnp.sum(scaled)) * losses)


def _prepare(state, particles, weights, rows, n_train):
    def initialize():
        # Repeat the first training row in padding slots; never read holdout data.
        slots = jnp.arange(rows.size // 2)
        indices = rows[jnp.where(slots < n_train, slots, 0)]
        x = particles[indices]
        mass = jnp.where(slots < n_train, weights[indices], 0)
        scaled = mass / jnp.max(mass)
        p = scaled / jnp.sum(scaled)
        # 2 sum_{j<i} p_i p_j equals 1 - sum(p**2) without subtracting near-equals.
        prefix = jnp.concatenate((jnp.zeros_like(p[:1]), jnp.cumsum(p[:-1])))
        correction = 2 * jnp.sum(p * prefix)
        anchor = jnp.argmax(p)

        def initialize_step(values, step):
            offsets = values - values[anchor]
            center = jnp.sum(p[:, None] * offsets, axis=0)
            variance = jnp.sum(p[:, None] * jnp.square(offsets - center), axis=0) / correction
            std = jnp.sqrt(variance) + 1e-6
            step = step._replace(log_scale=(-jnp.log(std)).astype(step.log_scale.dtype),
                                 shift=(-(values[anchor] + center) / std).astype(step.shift.dtype))
            working_step = jax.tree.map(lambda a: a.astype(values.dtype), step)
            (output, _), _ = _forward((values, jnp.zeros(values.shape[0], values.dtype)), working_step)
            return output, step

        _, params = jax.lax.scan(initialize_step, x, state.flow.params)
        return CDFlow(state.flow.config, params)

    return jax.lax.cond(state.fitted, lambda: state.flow, initialize)


@jax.jit
def refresh_cdf(state, particles, weights, key, *, iteration, beta, n_active):
    """Initialize ActNorm on first fit, then warm-start scheduled refits."""
    particles, weights = jnp.asarray(particles), jnp.asarray(weights)
    positive = weights > 0
    defer = ~state.fitted & (jnp.sum(positive) < 4)

    def pending():
        valid = (jnp.all(jnp.isfinite(weights) & (weights >= 0))
                 & jnp.all(~positive[:, None] | jnp.isfinite(particles))
                 & (state.initial_n_effective > 0) & (n_active > 0))
        return state._replace(fit_ok=valid)

    def fit():
        return refresh_flow(state, particles, weights, key, iteration=iteration,
                            beta=beta, n_active=n_active, patience=3 * state.flow.config.n_dim // 2,
                            batch_loss=_batch_loss, prepare=_prepare, project=CDFlow.normalize_mixer)

    return jax.lax.cond(defer, pending, fit)
