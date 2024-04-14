from typing import List, Optional, Tuple

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
from distrax._src.distributions.transformed import Transformed
from distrax._src.utils import math

from ..custom_types import Array, Key
from ..distributions import EqDistribution


class NormalizingFlow(eqx.Module):
    distribution: EqDistribution = eqx.static_field()
    bijector: eqx.Module

    def __init__(self, distribution: EqDistribution, flow: eqx.Module):
        super().__init__()
        self.distribution = distribution
        self.bijector = flow

    def sample(
        self, key: Key, n_samples: int = 1, context: Optional[Array] = None
    ) -> Array:
        x = self.distribution.sample(key, (n_samples,))
        y, _ = jax.vmap(self.bijector.forward_and_log_det, in_axes=(0, None))(
            x, context
        )
        return y

    def log_prob(self, x: Array, context: Optional[Array] = None) -> Array:
        x, ildj_y = self.bijector.inverse_and_log_det(x, context)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(
        self, key: Key, n_samples: int = 1, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(key, (n_samples,))
        y, fldj = jax.vmap(self.bijector.forward_and_log_det, in_axes=(0, None))(
            x, context
        )
        lp_y = lp_x - fldj
        return y, lp_y


class InverseConditional(eqx.Module):
    bijector: eqx.Module

    def __init__(self, bijector):
        super().__init__()
        self.bijector = bijector

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        return self.bijector.inverse(x, context)

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        return self.bijector.forward(y, context)

    def forward_and_log_det(
        self, x: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        return self.bijector.inverse_and_log_det(x, context)

    def inverse_and_log_det(
        self, y: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        return self.bijector.forward_and_log_det(y, context)


class ChainConditional(eqx.Module):
    bijectors: list

    def __init__(self, bijectors):
        super().__init__()
        self.bijectors = bijectors

    def forward(self, x: Array, context: Optional[Array] = None) -> Array:
        for bijector in reversed(self.bijectors):
            x = bijector.forward(x, context)
        return x

    def inverse(self, y: Array, context: Optional[Array] = None) -> Array:
        for bijector in self.bijectors:
            y = bijector.inverse(y, context)
        return y

    def forward_and_log_det(
        self, x: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        x, log_det = self.bijectors[-1].forward_and_log_det(x, context)
        for bijector in reversed(self.bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x, context)
            log_det += ld
        return x, log_det

    def inverse_and_log_det(
        self, y: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        y, log_det = self.bijectors[0].inverse_and_log_det(y, context)
        for bijector in self.bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y, context)
            log_det += ld
        return y, log_det


class MaskedCouplingConditional(eqx.Module):
    mask: Array = eqx.static_field()
    bijector_fn: eqx.Module
    conditioner: eqx.Module

    def __init__(self, mask, bijector_fn, conditioner):
        super().__init__()
        self.mask = mask
        self.bijector_fn = bijector_fn
        self.conditioner = conditioner

    def forward_and_log_det(
        self, x: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        masked_x = jnp.where(self.mask, x, 0.0)
        params = self.conditioner(masked_x, context)
        y0, log_d = self.bijector_fn(params).forward_and_log_det(x)
        y = jnp.where(self.mask, x, y0)
        logdet = math.sum_last(
            jnp.where(self.mask, 0.0, log_d),
            self.mask.ndim,
        )
        return y, logdet

    def inverse_and_log_det(
        self, y: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        masked_y = jnp.where(self.mask, y, 0.0)
        params = self.conditioner(masked_y, context)
        x0, log_d = self.bijector_fn(params).inverse_and_log_det(y)
        x = jnp.where(self.mask, y, x0)
        logdet = math.sum_last(
            jnp.where(self.mask, 0.0, log_d),
            self.mask.ndim,
        )
        return x, logdet


class Permute(eqx.Module):

    permutation: Array = eqx.static_field()
    axis: int = eqx.static_field()
    event_ndims_in: int = 1

    def __init__(self, permutation: Array, axis: int = -1):

        self.permutation = jnp.array(permutation)
        self.axis = axis

    def permute_along_axis(self, x: Array, permutation: Array, axis: int = -1) -> Array:
        x = jnp.moveaxis(x, axis, 0)
        x = x[permutation, ...]
        x = jnp.moveaxis(x, 0, axis)
        return x

    def forward_and_log_det(
        self, x: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        y = self.permute_along_axis(x, self.permutation, axis=self.axis)
        return y, jnp.zeros(x.shape[: -self.event_ndims_in])

    def inverse_and_log_det(
        self, y: Array, context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        inv_permutation = jnp.zeros_like(self.permutation)
        inv_permutation = inv_permutation.at[self.permutation].set(
            jnp.arange(len(self.permutation))
        )
        x = self.permute_along_axis(y, inv_permutation)
        return x, jnp.zeros(y.shape[: -self.event_ndims_in])


class TransformedConditional(Transformed):
    def __init__(self, distribution, flow):
        super().__init__(distribution, flow)

    def sample(
        self, seed: Key, sample_shape: List[int], context: Optional[Array] = None
    ) -> Array:
        x = self.distribution.sample(seed=seed, sample_shape=sample_shape)
        y, _ = self.bijector.forward_and_log_det(x, context)
        return y

    def log_prob(self, x: Array, context: Optional[Array] = None) -> Array:
        x, ildj_y = self.bijector.inverse_and_log_det(x, context)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(
        self, seed: Key, sample_shape: List[int], context: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        x, lp_x = self.distribution.sample_and_log_prob(
            seed=seed, sample_shape=sample_shape
        )
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x, context)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y
