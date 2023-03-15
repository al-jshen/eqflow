from eqflow.bijectors.rqs import RationalQuadraticSpline
from eqflow.distributions import EqDistribution
from eqflow.bijectors.bijectors import (
    ChainConditional,
    InverseConditional,
    MaskedCouplingConditional,
    NormalizingFlow,
)
from eqflow.nsf import MLP, NeuralSplineFlow
from eqflow.custom_types import Array

import distrax
import jax
import jax.numpy as jnp
import equinox as eqx

rng = jax.random.PRNGKey(0)


## Make our own normalizing flow
## =============================

ndim = 2  # input/output dims
ncontext = 5  # conditioning dims
event_shape = (ndim,)
context_shape = (ncontext,)
num_bijector_params = 64
ntransforms = 3

# use rqs as the bijector
def bijector_fn(params: Array):
    return RationalQuadraticSpline(params, range_min=0.0, range_max=1.0)


# make a mask
mask = jnp.arange(0, ndim) % 2
mask = jnp.reshape(mask, event_shape)
mask = mask.astype(bool)

# make MLP conditioners
conditioners = [
    MLP(rng, ndim, num_bijector_params, ncontext, (16, 16, 16))
    for _ in range(ntransforms)
]

# make the bijectors, given the mask and the conditioner
bijectors = [
    MaskedCouplingConditional(
        mask=mask, bijector_fn=bijector_fn, conditioner=conditioners[i]
    )
    for i in range(ntransforms)
]

# chain and invert the bijectors
bijector = InverseConditional(ChainConditional(bijectors))

# test inputs
x1 = jax.random.normal(rng, (1, ndim))
x2 = jax.random.normal(rng, (1, ncontext))

# define base distribution using wrapped distrax distribution
mvn = EqDistribution(distrax.MultivariateNormalDiag, jnp.zeros(ndim), jnp.ones(ndim))

# make the flow and test it
# it's all compatible with equinox!
flow = NormalizingFlow(mvn, bijector)
print(eqx.filter_jit(flow.log_prob)(x1, x2))
print(eqx.filter_jit(flow.sample)(rng, 5, x2))


## Use the NeuralSplineFlow class instead
## ======================================
nsf = NeuralSplineFlow(rng, ndim, ncontext, ntransforms, 8)
print(eqx.filter_jit(nsf.log_prob)(x1, x2))
print(eqx.filter_jit(nsf.sample)(rng, 5, x2))
