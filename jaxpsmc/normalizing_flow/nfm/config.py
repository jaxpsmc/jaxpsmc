"""Configuration for scalar-token, single-time NFM-D in jaxpsmc."""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class NFMConfig:
    """Architecture sizes and fitting budget are explicit experiment inputs.

    n_layers counts transformer layers per autoregressive block. Dimensions
    follow the particle vector, with exactly one scalar per token.
    """

    n_dim: int
    channels: int
    n_blocks: int
    n_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    s_max: float = 0.1
    clip_range: float = 1.0
    train_frequency: int | None = None

    def __post_init__(self) -> None:
        for name in ("n_dim", "channels", "n_blocks", "n_layers", "epochs", "batch_size"):
            _positive_int(name, getattr(self, name))
        if self.channels % 2 or self.channels % self.n_heads:
            raise ValueError("channels must be even and divisible by the attention head count")
        for name in ("learning_rate", "s_max", "clip_range"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.clip_range <= 0:
            raise ValueError("learning_rate and clip_range must be positive")
        if not 0 <= self.s_max <= 1:
            raise ValueError("s_max must be in [0, 1]")
        if self.train_frequency is not None:
            _positive_int("train_frequency", self.train_frequency)

    @property
    def n_heads(self) -> int:
        return max(1, self.channels // 64)

    def training_interval(self, initial_n_effective: int, n_active: int) -> int:
        """Evaluate once, before adaptive changes to n_effective."""
        _positive_int("initial_n_effective", initial_n_effective)
        _positive_int("n_active", n_active)
        if self.train_frequency is not None:
            return self.train_frequency
        return max(initial_n_effective // (2 * n_active), 1)
