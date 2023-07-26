from typing import Any

import equinox as eqx
from jaxtyping import PyTree


class EqDistribution(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def log_prob(self, x):
        return self.cls(*self.args, **self.kwargs).log_prob(x)

    def sample(self, key, sample_shape):
        return self.cls(*self.args, **self.kwargs).sample(
            seed=key, sample_shape=sample_shape
        )

    def sample_and_log_prob(self, key, sample_shape):
        return self.cls(*self.args, **self.kwargs).sample_and_log_prob(
            seed=key, sample_shape=sample_shape
        )
