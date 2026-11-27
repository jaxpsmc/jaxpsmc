"""Stage-wise fitting of the vector single-time NFM-D model.

The objective follows nfm/objectives/nfmd.py at the revision in model.py.
Particle weights enter once, in the reduction over each shuffled minibatch.
"""

import jax
import jax.numpy as jnp

from ..training import FlowFitState as NFMFitState
from ..training import _logical_batch_size, _run_epochs, refresh_flow


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


def refresh_nfm(state, particles, weights, key, *, iteration, beta, n_active):
    """Fit the single-time NFM-D objective on retained particles."""
    return refresh_flow(state, particles, weights, key, iteration=iteration, beta=beta,
                        n_active=n_active, patience=state.flow.config.stale_epochs,
                        batch_loss=_batch_loss)
