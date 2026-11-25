"""Configuration for scalar-token, single-time NFM-D in jaxpsmc."""

from dataclasses import KW_ONLY, dataclass
from math import isfinite
from numbers import Integral, Real


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class NFMConfig:
    """Architecture of the vector endpoint map.

    n_layers counts transformer layers per autoregressive block. Dimensions
    follow the particle vector, with exactly one scalar per token.
    """

    n_dim: int
    channels: int
    n_blocks: int
    n_layers: int
    _: KW_ONLY
    s_max: float = 0.1
    clip_range: float = 1.0

    def __post_init__(self) -> None:
        for name in ("n_dim", "channels", "n_blocks", "n_layers"):
            _positive_int(name, getattr(self, name))
        if self.channels % 2 or self.channels % self.n_heads:
            raise ValueError("channels must be even and divisible by the attention head count")
        for name in ("s_max", "clip_range"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.clip_range <= 0:
            raise ValueError("clip_range must be positive")
        if not 0 <= self.s_max <= 1:
            raise ValueError("s_max must be in [0, 1]")

    @property
    def n_heads(self) -> int:
        return max(1, self.channels // 64)

    @property
    def mutation_steps(self) -> int:
        return max(1, self.n_dim // 2)

    @property
    def mutation_cap(self) -> int:
        return 10 * self.mutation_steps

    @property
    def stale_epochs(self) -> int:
        return 3 * self.n_dim // 2
