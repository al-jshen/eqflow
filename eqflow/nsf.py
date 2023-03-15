from typing import List, Optional, Tuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .distributions import EqDistribution
from .bijectors.bijectors import (
    ChainConditional,
    InverseConditional,
    MaskedCouplingConditional,
    NormalizingFlow,
)
from .bijectors.rqs import RationalQuadraticSpline
from .custom_types import Array, Key


class MLP(eqx.Module):

    layers: list

    def __init__(
        self,
        key: Key,
        n_in: int,
        n_out: int,
        n_conditional: int = 0,
        n_hidden=(16, 16, 16),
        act=None,
    ):
        if act is None:
            act = [jax.nn.leaky_relu for _ in range(len(n_hidden) + 1)]
        assert len(act) == len(n_hidden) + 1

        self.layers = []
        n_ = [n_in + n_conditional, *n_hidden, n_out]

        keys = jax.random.split(key, len(n_) - 1)

        for i in range(len(n_) - 1):
            self.layers.append(eqx.nn.Linear(n_[i], n_[i + 1], key=keys[i]))
            self.layers.append(act[i])
        self.layers.pop()

    def __call__(self, x: Array, context: Optional[Array] = None):
        x = x.flatten()
        if context is not None:
            context = context.flatten()
            x = jnp.concatenate([x, context], axis=0)
        for layer in self.layers:
            x = layer(x)
        return x


class NeuralSplineFlow(eqx.Module):
    n_dim: int = eqx.static_field()
    n_context: int = eqx.static_field()
    n_transforms: int = eqx.static_field()
    n_bins: int = eqx.static_field()
    hidden_dims: List[int] = eqx.static_field()
    flow: NormalizingFlow

    def __init__(
        self,
        key: Key,
        n_dim: int,
        n_context: int,
        n_transforms: int,
        n_bins: int,
        hidden_dims: List[int] = [128, 128],
    ):
        self.n_dim = n_dim
        self.n_context = n_context
        self.n_transforms = n_transforms
        self.n_bins = n_bins
        self.hidden_dims = hidden_dims

        def bijector_fn(params: Array):
            return RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)

        event_shape = (self.n_dim,)

        mask = jnp.arange(0, self.n_dim) % 2
        mask = jnp.reshape(mask, event_shape)
        mask = mask.astype(bool)

        num_bijector_params = 3 * self.n_bins + 1

        keys = jax.random.split(key, self.n_transforms)

        conditioners = [
            MLP(
                key=keys[i],
                n_in=self.n_dim,
                n_out=num_bijector_params,
                n_conditional=self.n_context,
                n_hidden=self.hidden_dims,
            )
            for i in range(self.n_transforms)
        ]

        bijectors = []
        for i in range(self.n_transforms):
            bijectors.append(
                MaskedCouplingConditional(
                    mask=mask, bijector_fn=bijector_fn, conditioner=conditioners[i]
                )
            )
            mask = jnp.logical_not(mask)

        bijector = InverseConditional(ChainConditional(bijectors))
        base_dist = EqDistribution(
            distrax.MultivariateNormalDiag,
            jnp.zeros(event_shape),
            jnp.ones(event_shape),
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
