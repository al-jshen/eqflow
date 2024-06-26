from typing import Tuple

from jaxtyping import Array, Float

import equinox as eqx
import jax
import jax.numpy as jnp


def _normalize_bin_sizes(
    unnormalized_bin_sizes: Float[Array, "..."], total_size: float, min_bin_size: float
) -> Float[Array, "..."]:
    """Make bin sizes sum to `total_size` and be no less than `min_bin_size`."""
    num_bins = unnormalized_bin_sizes.shape[-1]
    if num_bins * min_bin_size > total_size:
        raise ValueError(
            f"The number of bins ({num_bins}) times the minimum bin size"
            f" ({min_bin_size}) cannot be greater than the total bin size"
            f" ({total_size})."
        )
    bin_sizes = jax.nn.softmax(unnormalized_bin_sizes, axis=-1)
    return bin_sizes * (total_size - num_bins * min_bin_size) + min_bin_size


def _normalize_knot_slopes(
    unnormalized_knot_slopes: Float[Array, "..."], min_knot_slope: float
) -> Float[Array, "..."]:
    """Make knot slopes be no less than `min_knot_slope`."""
    # The offset is such that the normalized knot slope will be equal to 1
    # whenever the unnormalized knot slope is equal to 0.
    if min_knot_slope >= 1.0:
        raise ValueError(
            f"The minimum knot slope must be less than 1; got" f" {min_knot_slope}."
        )
    min_knot_slope = jnp.array(min_knot_slope, dtype=unnormalized_knot_slopes.dtype)
    offset = jnp.log(jnp.exp(1.0 - min_knot_slope) - 1.0)
    return jax.nn.softplus(unnormalized_knot_slopes + offset) + min_knot_slope


def _rational_quadratic_spline_fwd(
    x: Float[Array, "..."],
    x_pos: Float[Array, "..."],
    y_pos: Float[Array, "..."],
    knot_slopes: Float[Array, "..."],
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Applies a rational-quadratic spline to a scalar.
    Args:
      x: a scalar (0-dimensional array). The scalar `x` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[x_pos[0], x_pos[-1]]`, and it will be transformed linearly if it's
        outside that interval.
      x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
      y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
      knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
    Returns:
      A tuple of two scalars: the output of the transformation and the log of the
      absolute first derivative at `x`.
    """
    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = x <= x_pos[0]
    above_range = x >= x_pos[-1]
    correct_bin = jnp.logical_and(x >= x_pos[:-1], x < x_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)
    # If y does not fall into any bin, we use the first spline in the following
    # computations to avoid numerical issues.
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    # Dot product of each parameter with the correct bin mask.
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width

    z = (x - x_pos_bin[0]) / bin_width
    # `z` should be in range [0, 1] to avoid NaNs later. This can happen because
    # of small floating point issues or when x is outside of the range of bins.
    # To avoid all problems, we restrict z in [0, 1].
    z = jnp.clip(z, 0.0, 1.0)
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    numerator = bin_height * (bin_slope * sq_z + knot_slopes_bin[0] * z1mz)
    denominator = bin_slope + slopes_term * z1mz
    y = y_pos_bin[0] + numerator / denominator

    # Compute log det Jacobian.
    # The logdet is a sum of 3 logs. It is easy to see that the inputs of the
    # first two logs are guaranteed to be positive because we ensured that z is in
    # [0, 1]. This is also true of the log(denominator) because:
    # denominator
    # == bin_slope + (knot_slopes_bin[1] + knot_slopes_bin[0] - 2 * bin_slope) *
    # z*(1-z)
    # >= bin_slope - 2 * bin_slope * z * (1-z)
    # >= bin_slope - 2 * bin_slope * (1/4)
    # == bin_slope / 2
    logdet = (
        2.0 * jnp.log(bin_slope)
        + jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        - 2.0 * jnp.log(denominator)
    )

    # If x is outside the spline range, we default to a linear transformation.
    y = jnp.where(below_range, (x - x_pos[0]) * knot_slopes[0] + y_pos[0], y)
    y = jnp.where(above_range, (x - x_pos[-1]) * knot_slopes[-1] + y_pos[-1], y)
    logdet = jnp.where(below_range, jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, jnp.log(knot_slopes[-1]), logdet)
    return y, logdet


def _safe_quadratic_root(
    a: Float[Array, "..."], b: Float[Array, "..."], c: Float[Array, "..."]
) -> Float[Array, "..."]:
    """Implement a numerically stable version of the quadratic formula."""
    # This is not a general solution to the quadratic equation, as it assumes
    # b ** 2 - 4. * a * c is known a priori to be positive (and which of the two
    # roots is to be used, see https://arxiv.org/abs/1906.04032).
    # There are two sources of instability:
    # (a) When b ** 2 - 4. * a * c -> 0, sqrt gives NaNs in gradient.
    # We clip sqrt_diff to have the smallest float number.
    sqrt_diff = b**2 - 4.0 * a * c
    safe_sqrt = jnp.sqrt(jnp.clip(sqrt_diff, jnp.finfo(sqrt_diff.dtype).tiny))
    # If sqrt_diff is non-positive, we set sqrt to 0. as it should be positive.
    safe_sqrt = jnp.where(sqrt_diff > 0.0, safe_sqrt, 0.0)
    # (b) When 4. * a * c -> 0. We use the more stable quadratic solution
    # depending on the sign of b.
    # See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf (eq 7 and 8).
    # Solution when b >= 0
    numerator_1 = 2.0 * c
    denominator_1 = -b - safe_sqrt
    # Solution when b < 0
    numerator_2 = -b + safe_sqrt
    denominator_2 = 2 * a
    # Choose the numerically stable solution.
    numerator = jnp.where(b >= 0, numerator_1, numerator_2)
    denominator = jnp.where(b >= 0, denominator_1, denominator_2)
    return numerator / denominator


def _rational_quadratic_spline_inv(
    y: Float[Array, "..."],
    x_pos: Float[Array, "..."],
    y_pos: Float[Array, "..."],
    knot_slopes: Float[Array, "..."],
) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
    """Applies the inverse of a rational-quadratic spline to a scalar.
    Args:
      y: a scalar (0-dimensional array). The scalar `y` can be any real number; it
        will be transformed by the spline if it's in the closed interval
        `[y_pos[0], y_pos[-1]]`, and it will be transformed linearly if it's
        outside that interval.
      x_pos: array of shape [num_bins + 1], the bin boundaries on the x axis.
      y_pos: array of shape [num_bins + 1], the bin boundaries on the y axis.
      knot_slopes: array of shape [num_bins + 1], the slopes at the knot points.
    Returns:
      A tuple of two scalars: the output of the inverse transformation and the log
      of the absolute first derivative of the inverse at `y`.
    """
    # Search to find the right bin. NOTE: The bins are sorted, so we could use
    # binary search, but this is more GPU/TPU friendly.
    # The following implementation avoids indexing for faster TPU computation.
    below_range = y <= y_pos[0]
    above_range = y >= y_pos[-1]
    correct_bin = jnp.logical_and(y >= y_pos[:-1], y < y_pos[1:])
    any_bin_in_range = jnp.any(correct_bin)
    first_bin = jnp.concatenate(
        [jnp.array([1]), jnp.zeros(len(correct_bin) - 1)]
    ).astype(bool)
    # If y does not fall into any bin, we use the first spline in the following
    # computations to avoid numerical issues.
    correct_bin = jnp.where(any_bin_in_range, correct_bin, first_bin)
    # Dot product of each parameter with the correct bin mask.
    params = jnp.stack([x_pos, y_pos, knot_slopes], axis=1)
    params_bin_left = jnp.sum(correct_bin[:, None] * params[:-1], axis=0)
    params_bin_right = jnp.sum(correct_bin[:, None] * params[1:], axis=0)

    # These are the parameters for the corresponding bin.
    x_pos_bin = (params_bin_left[0], params_bin_right[0])
    y_pos_bin = (params_bin_left[1], params_bin_right[1])
    knot_slopes_bin = (params_bin_left[2], params_bin_right[2])

    bin_width = x_pos_bin[1] - x_pos_bin[0]
    bin_height = y_pos_bin[1] - y_pos_bin[0]
    bin_slope = bin_height / bin_width
    w = (y - y_pos_bin[0]) / bin_height
    w = jnp.clip(w, 0.0, 1.0)  # Ensure w is in [0, 1].
    # Compute quadratic coefficients: az^2 + bz + c = 0
    slopes_term = knot_slopes_bin[1] + knot_slopes_bin[0] - 2.0 * bin_slope
    c = -bin_slope * w
    b = knot_slopes_bin[0] - slopes_term * w
    a = bin_slope - b

    # Solve quadratic to obtain z and then x.
    z = _safe_quadratic_root(a, b, c)
    z = jnp.clip(z, 0.0, 1.0)  # Ensure z is in [0, 1].
    x = bin_width * z + x_pos_bin[0]

    # Compute log det Jacobian.
    sq_z = z * z
    z1mz = z - sq_z  # z(1-z)
    sq_1mz = (1.0 - z) ** 2
    denominator = bin_slope + slopes_term * z1mz
    logdet = (
        -2.0 * jnp.log(bin_slope)
        - jnp.log(
            knot_slopes_bin[1] * sq_z
            + 2.0 * bin_slope * z1mz
            + knot_slopes_bin[0] * sq_1mz
        )
        + 2.0 * jnp.log(denominator)
    )

    # If y is outside the spline range, we default to a linear transformation.
    x = jnp.where(below_range, (y - y_pos[0]) / knot_slopes[0] + x_pos[0], x)
    x = jnp.where(above_range, (y - y_pos[-1]) / knot_slopes[-1] + x_pos[-1], x)
    logdet = jnp.where(below_range, -jnp.log(knot_slopes[0]), logdet)
    logdet = jnp.where(above_range, -jnp.log(knot_slopes[-1]), logdet)
    return x, logdet


class RationalQuadraticSpline(eqx.Module):
    dtype: type = eqx.static_field()
    params: Float[Array, "..."]
    num_bins: int = eqx.static_field()
    knot_slopes: Float[Array, "..."]
    x_pos: Float[Array, "..."]
    y_pos: Float[Array, "..."]

    def __init__(
        self,
        params: Float[Array, "..."],
        range_min: float,
        range_max: float,
        boundary_slopes: str = "unconstrained",
        min_bin_size: float = 1e-4,
        min_knot_slope: float = 1e-4,
    ):
        """Initializes a RationalQuadraticSpline bijector.
        Args:
          params: array of shape `[..., 3 * num_bins + 1]`, the unconstrained
            parameters of the bijector. The number of bins is implicitly defined by
            the last dimension of `params`. The parameters can take arbitrary
            unconstrained values; the bijector will reparameterize them internally
            and make sure they obey appropriate constraints. If `params` is the
            all-zeros array, the bijector becomes the identity function everywhere
            on the real line.
          range_min: the lower bound of the spline's range. Below `range_min`, the
            bijector defaults to a linear transformation.
          range_max: the upper bound of the spline's range. Above `range_max`, the
            bijector defaults to a linear transformation.
          boundary_slopes: controls the behaviour of the slope of the spline at the
            range boundaries (`range_min` and `range_max`). It is used to enforce
            certain boundary conditions on the spline. Available options are:
            - 'unconstrained': no boundary conditions are imposed; the slopes at the
              boundaries can vary freely.
            - 'lower_identity': the slope of the spline is set equal to 1 at the
              lower boundary (`range_min`). This makes the bijector equal to the
              identity function for values less than `range_min`.
            - 'upper_identity': similar to `lower_identity`, but now the slope of
              the spline is set equal to 1 at the upper boundary (`range_max`). This
              makes the bijector equal to the identity function for values greater
              than `range_max`.
            - 'identity': combines the effects of 'lower_identity' and
              'upper_identity' together. The slope of the spline is set equal to 1
              at both boundaries (`range_min` and `range_max`). This makes the
              bijector equal to the identity function outside the interval
              `[range_min, range_max]`.
            - 'circular': makes the slope at `range_min` and `range_max` be the
              same. This implements the "circular spline" introduced by:
              > Rezende et al., Normalizing Flows on Tori and Spheres,
              > https://arxiv.org/abs/2002.02428, 2020.
              This option should be used when the spline operates on a circle
              parameterized by an angle in the interval `[range_min, range_max]`,
              where `range_min` and `range_max` correspond to the same point on the
              circle.
          min_bin_size: The minimum bin size, in either the x or the y axis. Should
            be a small positive number, chosen for numerical stability. Guarantees
            that no bin in either the x or the y axis will be less than this value.
          min_knot_slope: The minimum slope at each knot point. Should be a small
            positive number, chosen for numerical stability. Guarantess that no knot
            will have a slope less than this value.
        """
        super().__init__()
        if params.shape[-1] % 3 != 1 or params.shape[-1] < 4:
            raise ValueError(
                f"The last dimension of `params` must have size"
                f" `3 * num_bins + 1` and `num_bins` must be at least 1."
                f" Got size {params.shape[-1]}."
            )
        if range_min >= range_max:
            raise ValueError(
                f"`range_min` must be less than `range_max`. Got"
                f" `range_min={range_min}` and `range_max={range_max}`."
            )
        if min_bin_size <= 0.0:
            raise ValueError(
                f"The minimum bin size must be positive; got" f" {min_bin_size}."
            )
        if min_knot_slope <= 0.0:
            raise ValueError(
                f"The minimum knot slope must be positive; got" f" {min_knot_slope}."
            )
        self.dtype = params.dtype
        self.num_bins = (params.shape[-1] - 1) // 3
        # Extract unnormalized parameters.
        unnormalized_bin_widths = params[..., : self.num_bins]
        unnormalized_bin_heights = params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_knot_slopes = params[..., 2 * self.num_bins :]
        # Normalize bin sizes and compute bin positions on the x and y axis.
        range_size = range_max - range_min
        bin_widths = _normalize_bin_sizes(
            unnormalized_bin_widths, range_size, min_bin_size
        )
        bin_heights = _normalize_bin_sizes(
            unnormalized_bin_heights, range_size, min_bin_size
        )
        x_pos = range_min + jnp.cumsum(bin_widths[..., :-1], axis=-1)
        y_pos = range_min + jnp.cumsum(bin_heights[..., :-1], axis=-1)
        pad_shape = params.shape[:-1] + (1,)
        pad_below = jnp.full(pad_shape, range_min, dtype=self.dtype)
        pad_above = jnp.full(pad_shape, range_max, dtype=self.dtype)
        self.x_pos = jnp.concatenate([pad_below, x_pos, pad_above], axis=-1)
        self.y_pos = jnp.concatenate([pad_below, y_pos, pad_above], axis=-1)
        # Normalize knot slopes and enforce requested boundary conditions.
        knot_slopes = _normalize_knot_slopes(unnormalized_knot_slopes, min_knot_slope)
        if boundary_slopes == "unconstrained":
            self.knot_slopes = knot_slopes
        elif boundary_slopes == "lower_identity":
            ones = jnp.ones(pad_shape, self.dtype)
            self.knot_slopes = jnp.concatenate([ones, knot_slopes[..., 1:]], axis=-1)
        elif boundary_slopes == "upper_identity":
            ones = jnp.ones(pad_shape, self.dtype)
            self.knot_slopes = jnp.concatenate([knot_slopes[..., :-1], ones], axis=-1)
        elif boundary_slopes == "identity":
            ones = jnp.ones(pad_shape, self.dtype)
            self.knot_slopes = jnp.concatenate(
                [ones, knot_slopes[..., 1:-1], ones], axis=-1
            )
        elif boundary_slopes == "circular":
            self.knot_slopes = jnp.concatenate(
                [knot_slopes[..., :-1], knot_slopes[..., :1]], axis=-1
            )
        else:
            raise ValueError(
                f"Unknown option for boundary slopes:" f" `{boundary_slopes}`."
            )

        self.params = params

    def forward_and_log_det(
        self, x: Float[Array, "..."]
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        fn = jnp.vectorize(
            _rational_quadratic_spline_fwd, signature="(),(n),(n),(n)->(),()"
        )
        y, logdet = fn(x, self.x_pos, self.y_pos, self.knot_slopes)
        return y, logdet

    def inverse_and_log_det(
        self, y: Float[Array, "..."]
    ) -> Tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        fn = jnp.vectorize(
            _rational_quadratic_spline_inv, signature="(),(n),(n),(n)->(),()"
        )
        x, logdet = fn(y, self.x_pos, self.y_pos, self.knot_slopes)
        return x, logdet
