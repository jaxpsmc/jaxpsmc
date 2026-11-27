"""Static architecture for the particle-vector adaptation of CDFlow."""

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True)
class CDFlowConfig:
    n_dim: int
    n_steps: int = 2
    hidden_width: int = 32

    def __post_init__(self):
        # Host-side construction only. Architecture is static under JIT.
        def positive_integer(value):
            return isinstance(value, Integral) and not isinstance(value, bool) and value > 0

        if not all(map(positive_integer, (self.n_dim, self.n_steps, self.hidden_width))):
            raise ValueError("n_dim, n_steps and hidden_width must be positive integers")
