from typing import Any, Callable, List, Optional, Tuple

from jaxtyping import Array, Float, PRNGKeyArray
import numpy as np

import distrax
import equinox as eqx
import jax
import jax.numpy as jnp


def _make_dense_autoregressive_masks(
    params,
    event_size,
    hidden_units,
    input_order="left-to-right",
    hidden_degrees="equal",
    seed=None,
):
    """Creates masks for use in dense MADE [Germain et al. (2015)][1] networks.

    See the documentation for `AutoregressiveNetwork` for the theory and
    application of MADE networks. This function lets you construct your own dense
    MADE networks by applying the returned masks to each dense layer. E.g. a
    consider an autoregressive network that takes `event_size`-dimensional vectors
    and produces `params`-parameters per input, with `num_hidden` hidden layers,
    with `hidden_size` hidden units each.

    ```python
    def random_made(x):
      masks = tfb._make_dense_autoregressive_masks(
          params=params,
          event_size=event_size,
          hidden_units=[hidden_size] * num_hidden)
      output_sizes = [hidden_size] * num_hidden
      input_size = event_size
      for (mask, output_size) in zip(masks, output_sizes):
        mask = tf.cast(mask, tf.float32)
        x = tf.matmul(x, tf.random.normal([input_size, output_size]) * mask)
        x = tf.nn.relu(x)
        input_size = output_size
      x = tf.matmul(
          x,
          tf.random.normal([input_size, params * event_size]) * masks[-1])
      x = tf.reshape(x, [-1, event_size, params])
      return x

    y = random_made(tf.zeros([1, event_size]))
    assert [1, event_size, params] == y.shape
    ```

    Each mask is a Numpy boolean array. All masks have the shape `[input_size,
    output_size]`. For example, if we `hidden_units` is a list of two integers,
    the mask shapes will be: `[event_size, hidden_units[0]], [hidden_units[0],
    hidden_units[1]], [hidden_units[1], params * event_size]`.

    You can extend this example with trainable parameters and constraints if
    necessary.

    Args:
      params: Python integer specifying the number of parameters to output
        per input.
      event_size: Python integer specifying the shape of the input to this layer.
      hidden_units: Python `list`-like of non-negative integers, specifying
        the number of units in each hidden layer.
      input_order: Order of degrees to the input units: 'random', 'left-to-right',
        'right-to-left', or an array of an explicit order. For example,
        'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'. If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree. Default: 'equal'.
      seed: If not `None`, seed to use for 'random' `input_order` and
        `hidden_degrees`.

    Returns:
      masks: A list of masks that should be applied the dense matrices of
        individual densely connected layers in the MADE network. Each mask is a
        Numpy boolean array.

    #### References

    [1]: Mathieu Germain, Karol Gregor, Iain Murray, and Hugo Larochelle. MADE:
         Masked Autoencoder for Distribution Estimation. In _International
         Conference on Machine Learning_, 2015. https://arxiv.org/abs/1502.03509
    """
    if seed is None:
        input_order_seed = None
        degrees_seed = None
    else:
        input_order_seed, degrees_seed = np.random.RandomState(seed).randint(
            2**31, size=2
        )
    input_order = _create_input_order(event_size, input_order, seed=input_order_seed)
    masks = _create_masks(
        _create_degrees(
            input_size=event_size,
            hidden_units=hidden_units,
            input_order=input_order,
            hidden_degrees=hidden_degrees,
            seed=degrees_seed,
        )
    )
    # In the final layer, we will produce `params` outputs for each of the
    # `event_size` inputs.  But `masks[-1]` has shape `[hidden_units[-1],
    # event_size]`.  Thus, we need to expand the mask to `[hidden_units[-1],
    # event_size * params]` such that all units for the same input are masked
    # identically.  In particular, we tile the mask so the j-th element of
    # `tf.unstack(output, axis=-1)` is a tensor of the j-th parameter/unit for
    # each input.
    #
    # NOTE: Other orderings of the output could be faster -- should benchmark.
    masks[-1] = np.reshape(
        np.tile(masks[-1][..., np.newaxis], [1, 1, params]),
        [masks[-1].shape[0], event_size * params],
    )
    return masks


def _create_input_order(input_size, input_order="left-to-right", seed=None):
    """Returns a degree vectors for the input."""
    if isinstance(input_order, str):
        if input_order == "left-to-right":
            return np.arange(start=1, stop=input_size + 1)
        elif input_order == "right-to-left":
            return np.arange(start=input_size, stop=0, step=-1)
        elif input_order == "random":
            ret = np.arange(start=1, stop=input_size + 1)
            if seed is None:
                rng = np.random
            else:
                rng = np.random.RandomState(seed)
            rng.shuffle(ret)
            return ret
    elif np.all(np.sort(np.array(input_order)) == np.arange(1, input_size + 1)):
        return np.array(input_order)

    raise ValueError('Invalid input order: "{}".'.format(input_order))


def _create_degrees(
    input_size,
    hidden_units=None,
    input_order="left-to-right",
    hidden_degrees="equal",
    seed=None,
):
    """Returns a list of degree vectors, one for each input and hidden layer.

    A unit with degree d can only receive input from units with degree < d. Output
    units always have the same degree as their associated input unit.

    Args:
      input_size: Number of inputs.
      hidden_units: list with the number of hidden units per layer. It does not
        include the output layer. Each hidden unit size must be at least the size
        of length (otherwise autoregressivity is not possible).
      input_order: Order of degrees to the input units: 'random', 'left-to-right',
        'right-to-left', or an array of an explicit order. For example,
        'left-to-right' builds an autoregressive model
        p(x) = p(x1) p(x2 | x1) ... p(xD | x<D).
      hidden_degrees: Method for assigning degrees to the hidden units:
        'equal', 'random'.  If 'equal', hidden units in each layer are allocated
        equally (up to a remainder term) to each degree.  Default: 'equal'.
      seed: If not `None`, use as a seed for the 'random' hidden_degrees.

    Raises:
      ValueError: invalid input order.
      ValueError: invalid hidden degrees.
    """
    input_order = _create_input_order(input_size, input_order)
    degrees = [input_order]

    if hidden_units is None:
        hidden_units = []

    for units in hidden_units:
        if isinstance(hidden_degrees, str):
            if hidden_degrees == "random":
                if seed is None:
                    rng = np.random
                else:
                    rng = np.random.RandomState(seed)
                # samples from: [low, high)
                degrees.append(
                    rng.randint(
                        low=min(np.min(degrees[-1]), input_size - 1),
                        high=input_size,
                        size=units,
                    )
                )
            elif hidden_degrees == "equal":
                min_degree = min(np.min(degrees[-1]), input_size - 1)
                degrees.append(
                    np.maximum(
                        min_degree,
                        # Evenly divide the range `[1, input_size - 1]` in to `units + 1`
                        # segments, and pick the boundaries between the segments as degrees.
                        np.ceil(
                            np.arange(1, units + 1)
                            * (input_size - 1)
                            / float(units + 1)
                        ).astype(np.int32),
                    )
                )
        else:
            raise ValueError('Invalid hidden order: "{}".'.format(hidden_degrees))

    return degrees


def _create_masks(degrees):
    """Returns a list of binary mask matrices enforcing autoregressivity."""
    return [
        # Create input->hidden and hidden->hidden masks.
        inp[:, np.newaxis] <= out
        for inp, out in zip(degrees[:-1], degrees[1:])
    ] + [
        # Create hidden->output mask.
        degrees[-1][:, np.newaxis] < degrees[0]
    ]


class MaskedDense(eqx.Module):
    in_dims: int = eqx.static_field()
    out_dims: int = eqx.static_field()
    mask: Array = eqx.static_field()
    weights: Float[Array, "..."]
    bias: Float[Array, "..."]

    def __init__(
        self, rng: PRNGKeyArray, in_dims: int, out_dims: int, mask: Float[Array, "..."]
    ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.mask = mask
        self.weights = jax.random.normal(rng, (in_dims, out_dims)) / (
            in_dims * out_dims
        )
        self.bias = jnp.zeros(out_dims)

    def __call__(
        self, x: Float[Array, "..."], key: Optional[PRNGKeyArray] = None
    ) -> Float[Array, "..."]:
        y = jax.lax.dot_general(
            x,
            self.weights,
            (((x.ndim - 1,), (0,)), ((), ())),
        )
        y = y + jnp.reshape(self.bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class MADE(eqx.Module):
    layers: eqx.nn.Sequential
    activation: str = eqx.static_field()
    n_params: int = eqx.static_field()
    n_context: int = eqx.static_field()
    hidden_dims: Tuple[int, ...] = eqx.static_field()

    def __init__(
        self,
        rng: PRNGKeyArray,
        n_params: int,
        n_context: int = 0,
        hidden_dims: Tuple[int, ...] = (32, 32),
        activation: str = "leaky_relu",
    ):
        self.n_params = n_params
        self.n_context = n_context
        self.hidden_dims = hidden_dims
        self.activation = activation

        masks = _make_dense_autoregressive_masks(
            params=2,
            event_size=self.n_params + self.n_context,
            hidden_units=self.hidden_dims,
            input_order="left-to-right",
        )  # 2 parameters for scale and shift factors

        keys = jax.random.split(rng, len(masks))

        in_shapes = [self.n_params + self.n_context] + [m.shape[-1] for m in masks[:-1]]

        layers = []
        for i, mask in enumerate(masks[:-1]):
            layers.append(
                MaskedDense(
                    rng=keys[i],
                    in_dims=in_shapes[i],
                    out_dims=mask.shape[-1],
                    mask=mask,
                ),
            )
            layers.append(
                eqx.nn.Lambda(getattr(jax.nn, activation)),
            )
        layers.append(
            MaskedDense(
                rng=keys[-1],
                in_dims=in_shapes[-1],
                out_dims=masks[-1].shape[-1],
                mask=masks[-1],
            )
        )
        self.layers = eqx.nn.Sequential(layers)

    def __call__(self, y: Float[Array, "..."], context=None):
        if context is not None:
            # Stack with context on the left so that the parameters are autoregressively conditioned on it with left-to-right ordering
            y = jnp.hstack([context, y])

        broadcast_dims = y.shape[:-1]

        y = self.layers(y)

        # Unravel the inputs and parameters
        params = y.reshape(broadcast_dims + (self.n_params + self.n_context, 2))

        # Only take the values corresponding to the parameters of interest for scale and shift; ignore context outputs
        params = params[..., self.n_context :, :]

        return params


class MAF(eqx.Module):
    autoregressive_fn: Callable = eqx.static_field()
    unroll_loop: bool = eqx.static_field()

    def __init__(self, bijector_fn, unroll_loop=False):
        self.autoregressive_fn = bijector_fn
        self.unroll_loop = unroll_loop

    def forward_and_log_det(self, x, context):
        event_ndims = x.shape[-1]

        if self.unroll_loop:
            y = jnp.zeros_like(x)
            log_det = None

            for _ in range(event_ndims):
                params = self.autoregressive_fn(y, context)
                shift, log_scale = params[..., 0], params[..., 1]
                y, log_det = distrax.ScalarAffine(
                    shift=shift, log_scale=log_scale
                ).forward_and_log_det(x)

        else:

            def update_fn(_, y_and_log_det):
                y, log_det = y_and_log_det
                params = self.autoregressive_fn(y)
                shift, log_scale = params[..., 0], params[..., 1]
                y, log_det = distrax.ScalarAffine(
                    shift=shift, log_scale=log_scale
                ).forward_and_log_det(x)
                return y, log_det

            y, log_det = jax.lax.fori_loop(
                0, event_ndims, update_fn, (jnp.zeros_like(x), jnp.zeros_like(x))
            )

        return y, log_det.sum(-1)

    def inverse_and_log_det(self, y, context):
        params = self.autoregressive_fn(y, context)
        shift, log_scale = params[..., 0], params[..., 1]
        x, log_det = distrax.ScalarAffine(
            shift=shift, log_scale=log_scale
        ).inverse_and_log_det(y)

        return x, log_det.sum(-1)
