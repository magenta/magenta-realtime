# File under the MIT license, see https://github.com/adefossez/julius/blob/main/LICENSE for details.
# Author: adefossez, 2020
"""
Differentiable, Pytorch based resampling.
Implementation of Julius O. Smith algorithm for resampling.
See https://ccrma.stanford.edu/~jos/resample/ for details.
This implementation is specially optimized for when new_sr / old_sr is a fraction
with a small numerator and denominator when removing the gcd (e.g. new_sr = 700, old_sr = 500).

Very similar to [bmcfee/resampy](https://github.com/bmcfee/resampy) except this implementation
is optimized for the case mentioned before, while resampy is slower but more general.

"""

import math
from typing import Optional

from einops import rearrange
from jax import lax
from jax import vmap
from jax import numpy as jnp


def sinc(x):
    """Sinc function implemented in JAX."""
    return jnp.sinc(x / jnp.pi)


def resample(
    x: jnp.ndarray,
    old_sr: int,
    new_sr: int,
    zeros: int = 24,
    rolloff: float = 0.945,
    output_length: Optional[int] = None,
    full: bool = False,
) -> jnp.ndarray:
    """Resampling algorithm adapted from the pytorch library Julius:
    https://github.com/adefossez/julius/blob/main/julius/resample.py

    Args:
        x (jnp.ndarray): Input array shaped
        old_sr (int): sample rate of the input signal x.
        new_sr (int): sample rate of the output.
        zeros (int): number of zero crossing to keep in the sinc filter.
        rolloff (float): use a lowpass filter that is `rolloff * new_sr / 2`,
            to ensure sufficient margin due to the imperfection of the FIR filter used.
            Lowering this value will reduce anti-aliasing, but will reduce some of the
            highest frequencies.

    Shape:

        - Input: `[B, C, T]`
        - Output: `[B, C, T']` with `T' = int(new_sr * T / old_sr)`

    .. caution::
        After dividing `old_sr` and `new_sr` by their GCD, both should be small
        for this implementation to be fast.
    """

    if not isinstance(old_sr, int) or not isinstance(new_sr, int):
        raise ValueError("old_sr and new_sr should be integers")

    assert x.ndim == 3

    if new_sr == old_sr:
        return x

    batch_size, c, length = x.shape

    gcd = math.gcd(old_sr, new_sr)
    old_sr = old_sr // gcd
    new_sr = new_sr // gcd
    zeros = zeros
    rolloff = rolloff

    if old_sr == new_sr:
        return x

    sr = min(new_sr, old_sr) * rolloff
    _width = math.ceil(zeros * old_sr / sr)
    idx = jnp.arange(-_width, _width + old_sr)

    @vmap
    def get_kernels(i):
        t = (-i / new_sr + idx / old_sr) * sr
        t = jnp.clip(t, -zeros, zeros) * jnp.pi
        window = jnp.cos(t / zeros / 2) ** 2
        kernel = sinc(t) * window
        kernel = kernel / kernel.sum()
        return kernel

    kernel = get_kernels(jnp.arange(new_sr))
    kernel = kernel.reshape((new_sr, 1, -1))

    x = rearrange(x, "b (c one) t -> (b c) one t", b=batch_size, c=c, one=1)

    y = lax.conv_general_dilated(
        x,
        kernel,
        window_strides=(old_sr,),
        padding=((int(_width), int(_width + old_sr)),),
    )
    y = rearrange(y, "(b c) w t -> b c (t w)", b=batch_size, c=c)

    float_output_length = new_sr * length / old_sr
    max_output_length = math.ceil(float_output_length)
    default_output_length = math.floor(float_output_length)

    if output_length is None:
        applied_output_length = max_output_length if full else default_output_length
    elif output_length < 0 or output_length > max_output_length:
        raise ValueError(f"output_length must be between 0 and {max_output_length}")
    else:
        applied_output_length = output_length
        if full:
            raise ValueError("You cannot pass both full=True and output_length")

    return y[..., :applied_output_length]
