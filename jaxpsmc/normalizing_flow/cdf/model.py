"""Vector CDFlow. The direct map takes particle coordinates to base coordinates."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import CDFlowConfig


class _Dense(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class _Step(NamedTuple):
    log_scale: jax.Array
    shift: jax.Array
    diagonal: jax.Array       # Left and right factors of D_left C D_right.
    real: jax.Array           # Real coefficients of the one-sided circulant spectrum.
    imag: jax.Array           # Imaginary coefficients of conjugate-paired bins only.
    first: _Dense
    hidden: _Dense
    output: _Dense


def _spectrum(step):
    imaginary = jnp.zeros_like(step.real).at[1:1 + step.imag.size].set(step.imag)
    return jax.lax.complex(step.real, imaginary)


def _normalize(step):
    """Source max-magnitude normalization, independently within each flow step."""
    magnitude = jnp.max(jnp.abs(_spectrum(step)))
    return step._replace(diagonal=step.diagonal / jnp.max(jnp.abs(step.diagonal)),
                         real=step.real / magnitude, imag=step.imag / magnitude)


def _init_step(key, config, dtype):
    d = config.n_dim
    split = max(1, d // 2)
    # Empty networks for scalar events give an affine-only map with zero MLP parameters.
    width = config.hidden_width * min(1, d - split)
    kd, kc, k1, k2, k3 = jax.random.split(key, 5)
    diagonal = 1 + jax.random.normal(kd, (2, d), dtype) / d
    spectrum = 1 + jnp.fft.rfft(jax.random.normal(kc, (d,), dtype)) / d

    def dense(layer_key, inputs, outputs):
        # Two independent networks, for log-scale and translation respectively.
        kw, kb = jax.random.split(layer_key)
        bound = max(inputs, 1) ** -0.5
        return _Dense(jax.random.uniform(kw, (2, inputs, outputs), dtype, -bound, bound),
                      jax.random.uniform(kb, (2, outputs), dtype, -bound, bound))

    zeros = jnp.zeros((d,), dtype)
    return _normalize(_Step(zeros, zeros, diagonal, spectrum.real,
                            spectrum.imag[1:1 + (d - 1) // 2], dense(k1, split, width),
                            dense(k2, width, width), dense(k3, width, d - split)))


def _condition(step, x):
    def affine(h, layer):
        return jnp.einsum("...ni,nio->...no", h, layer.weight) + layer.bias

    first = jnp.einsum("...i,nio->...no", x, step.first.weight) + step.first.bias
    hidden = jnp.tanh(affine(jnp.tanh(first), step.hidden))
    output = affine(hidden, step.output)
    return output[..., 0, :], output[..., 1, :]


def _mix(x, right, spectrum, left):
    return left * jnp.fft.irfft(jnp.fft.rfft(x * right) * spectrum, n=x.shape[-1])


def _linear_logdet(step, spectrum):
    # Each interior bin represents a conjugate pair. For odd d the last bin is interior.
    d = step.shift.size
    k = jnp.arange(spectrum.size)
    multiplicity = 1 + ((k > 0) & (2 * k < d)).astype(step.shift.dtype)
    return (jnp.sum(step.log_scale) + jnp.sum(jnp.log(jnp.abs(step.diagonal)))
            + jnp.sum(multiplicity * jnp.log(jnp.abs(spectrum))))


def _forward(carry, step):
    x, ld = carry
    spectrum = _spectrum(step)
    x = _mix(x * jnp.exp(step.log_scale) + step.shift,
             step.diagonal[1], spectrum, step.diagonal[0])
    a, b = jnp.split(x, (step.first.weight.shape[1],), axis=-1)
    scale, shift = _condition(step, a)
    y = jnp.concatenate((a, b * jnp.exp(scale) + shift), axis=-1)
    return (y, ld + _linear_logdet(step, spectrum) + scale.sum(axis=-1)), None


def _inverse(carry, step):
    y, ld = carry
    spectrum = _spectrum(step)
    a, b = jnp.split(y, (step.first.weight.shape[1],), axis=-1)
    scale, shift = _condition(step, a)
    y = jnp.concatenate((a, (b - shift) * jnp.exp(-scale)), axis=-1)
    x = _mix(y, 1 / step.diagonal[0], 1 / spectrum, 1 / step.diagonal[1])
    x = (x - step.shift) * jnp.exp(-step.log_scale)
    return (x, ld - _linear_logdet(step, spectrum) - scale.sum(axis=-1)), None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class CDFlow:
    config: CDFlowConfig
    params: _Step             # Every leaf carries a leading flow-step axis.

    def tree_flatten(self):
        return (self.params,), self.config

    @classmethod
    def tree_unflatten(cls, config, children):
        return cls(config, children[0])

    @property
    def bijection(self):
        return self

    @classmethod
    def initialize(cls, config, key, dtype=jnp.float32):
        keys = jax.random.split(key, config.n_steps)
        return cls(config, jax.vmap(lambda k: _init_step(k, config, dtype))(keys))

    def normalize_mixer(self):
        """Project mixer parameters after an optimizer update, before freezing the map."""
        return CDFlow(self.config, jax.vmap(_normalize)(self.params))

    def _apply(self, x, condition, step_fn, reverse):
        x = jnp.asarray(x)
        # These checks inspect static metadata only, before numerical computation.
        if condition is not None or x.ndim == 0 or x.shape[-1] != self.config.n_dim:
            raise ValueError("expected unconditional particles with final axis n_dim")
        if x.dtype not in (jnp.float32, jnp.float64):
            raise TypeError("expected float32 or float64 coordinates")
        params = jax.tree_util.tree_map(lambda p: p.astype(x.dtype), self.params)
        carry = (x, jnp.zeros(x.shape[:-1], x.dtype))
        result, _ = jax.lax.scan(step_fn, carry, params, reverse=reverse)
        return result

    def transform_and_log_det(self, u, condition=None):
        return self._apply(u, condition, _forward, False)

    def inverse_and_log_det(self, theta, condition=None):
        return self._apply(theta, condition, _inverse, True)
