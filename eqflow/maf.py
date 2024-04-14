from typing import Optional, Tuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from eqflow.bijectors.ar import MADE, MAF
from eqflow.bijectors.bijectors import (
    ChainConditional,
    InverseConditional,
    NormalizingFlow,
    Permute,
)
from eqflow.custom_types import Array, Key
from eqflow.distributions import EqDistribution


class MaskedAutoregressiveFlow(eqx.Module):

    n_dim: int
    n_context: int = 0
    n_transforms: int = 4
    hidden_dims: Tuple[int, ...] = eqx.static_field()
    activation: str = "leaky_relu"
    unroll_loop: bool = True
    use_random_permutations: bool = True
    inverse: bool = False

    made: list[MADE]
    flow: NormalizingFlow

    def __init__(
        self,
        rng: Key,
        n_dim: int,
        n_context: int = 0,
        n_transforms: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 128),
        activation: str = "leaky_relu",
        unroll_loop: bool = True,
        use_random_permutations: bool = True,
        inverse: bool = False,
    ):
        super().__init__()
        self.n_dim = n_dim
        self.n_context = n_context
        self.n_transforms = n_transforms
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.unroll_loop = unroll_loop
        self.use_random_permutations = use_random_permutations
        self.inverse = inverse

        keys = jax.random.split(rng, self.n_transforms)

        self.made = [
            MADE(
                keys[i],
                n_params=self.n_dim,
                n_context=self.n_context,
                hidden_dims=self.hidden_dims,
                activation=self.activation,
            )
            for i in range(self.n_transforms)
        ]

        bijectors = []
        for i in range(self.n_transforms):
            # Permutation
            if self.use_random_permutations:
                permutation = jax.random.choice(
                    rng, jnp.arange(self.n_dim), shape=(self.n_dim,), replace=False
                )
                rng, _ = jax.random.split(rng)
            else:
                permutation = list(reversed(range(self.n_dim)))
            bijectors.append(Permute(permutation))

            bijector_af = MAF(bijector_fn=self.made[i], unroll_loop=self.unroll_loop)
            if self.inverse:
                bijector_af = InverseConditional(
                    bijector_af
                )  # Flip forward and reverse directions for IAF
            bijectors.append(bijector_af)

        bijector = InverseConditional(
            ChainConditional(bijectors)
        )  # Forward direction goes from target to base distribution
        base_dist = EqDistribution(
            distrax.MultivariateNormalDiag,
            jnp.zeros(self.n_dim),
            jnp.ones(self.n_dim),
        )
        self.flow = NormalizingFlow(base_dist, bijector)

    def log_prob(self, x: Array, context: Optional[Array] = None) -> Array:
        return self.flow.log_prob(x, context=context)

    def sample(
        self, key: Key, n_samples: int = 1, context: Optional[Array] = None
    ) -> Array:
        return self.flow.sample(key, n_samples=n_samples, context=context)

    def sample_and_log_prob(
        self, key: Key, n_samples: int = 1, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        return self.flow.sample_and_log_prob(key, n_samples=n_samples, context=context)
