# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numpy SpectroStream decoder: the executable spec for the ggml codec port.

Tensors are NHWC with H = time (frames) and W = frequency (bins). Padding
follows sequence_layers:

  time  'semicausal': pad_left = max(ek - stride, 0), pad_right = ek - 1 - left
                      (so a stride-1 3-tap conv is fully causal: (2, 0))
  freq  explicit    : pad = max((k - 1) * d + 1 - stride, 0), split evenly
"""

import numpy as np


def elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))


def semicausal_pad(kernel: int, stride: int, dilation: int = 1):
    ek = (kernel - 1) * dilation + 1
    left = max(ek - stride, 0)
    return left, ek - 1 - left


def freq_pad(kernel: int, stride: int, dilation: int = 1):
    pad = max((kernel - 1) * dilation + 1 - stride, 0)
    return pad // 2, pad - pad // 2


def conv2d(x, w, b, strides=(1, 1), dilation=(1, 1), time_pad=None, freq_pad_=None):
    """x: [T, F, IC]; w: [KH, KW, IC, OC] -> [T', F', OC]."""
    kh, kw, ic, oc = w.shape
    sh, sw = strides
    dh, dw = dilation
    tp = time_pad if time_pad is not None else semicausal_pad(kh, sh, dh)
    fp = freq_pad_ if freq_pad_ is not None else freq_pad(kw, sw, dw)
    xp = np.pad(x, ((tp[0], tp[1]), (fp[0], fp[1]), (0, 0)))

    ekh, ekw = (kh - 1) * dh + 1, (kw - 1) * dw + 1
    out_t = (xp.shape[0] - ekh) // sh + 1
    out_f = (xp.shape[1] - ekw) // sw + 1
    # im2col: [out_t, out_f, kh*kw*ic] @ [kh*kw*ic, oc]
    cols = np.empty((out_t, out_f, kh * kw * ic), dtype=np.float32)
    idx = 0
    for i in range(kh):
        for j in range(kw):
            patch = xp[i * dh: i * dh + out_t * sh: sh, j * dw: j * dw + out_f * sw: sw, :]
            cols[:, :, idx * ic:(idx + 1) * ic] = patch
            idx += 1
    y = cols.reshape(-1, kh * kw * ic) @ w.reshape(kh * kw * ic, oc)
    y = y.reshape(out_t, out_f, oc)
    return y + b if b is not None else y


def conv2d_transpose(x, w, b, strides, time_padding="causal"):
    """Transposed conv: time padding 'causal', frequency padding 'same'.

    w is the Flax ConvTranspose kernel, [KH, KW, IC, OC].

    JAX's conv_transpose performs a true mathematical transposition, which
    flips the kernel along both spatial axes; MLX's conv_transpose2d does not
    (see magenta_rt/mlx/spectrostream/load_weights.py). We match JAX here.
    """
    kh, kw, ic, oc = w.shape
    w = w[::-1, ::-1]
    sh, sw = strides
    T, F, _ = x.shape
    # full transposed-conv output, before cropping
    full_t = (T - 1) * sh + kh
    full_f = (F - 1) * sw + kw
    acc = np.zeros((full_t, full_f, oc), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            # x [T,F,ic] @ w[i,j] [oc,ic]^T -> [T,F,oc]
            contrib = x.reshape(-1, ic) @ w[i, j]
            contrib = contrib.reshape(T, F, oc)
            acc[i:i + T * sh:sh, j:j + F * sw:sw, :] += contrib
    # time: causal -> keep the first T*sh outputs (drop the kh-1 tail)
    out_t = T * sh
    acc = acc[:out_t]
    # frequency: 'same' -> center crop to F*sw
    out_f = F * sw
    start = (full_f - out_f) // 2
    acc = acc[:, start:start + out_f, :]
    return acc + b if b is not None else acc


def upsample2d(x, rate):
    """Nearest-neighbour repeat along time and frequency."""
    return np.repeat(np.repeat(x, rate[0], axis=0), rate[1], axis=1)


def hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def inverse_stft_window(frame_length: int, frame_step: int) -> np.ndarray:
    """Griffin & Lim eq. 7: forward window / summed squared overlaps."""
    fw = hann_periodic(frame_length)
    denom = fw ** 2
    overlaps = -(-frame_length // frame_step)
    denom = np.pad(denom, (0, overlaps * frame_step - frame_length))
    denom = denom.reshape(overlaps, frame_step).sum(axis=0)
    denom = np.tile(denom, overlaps)[:frame_length]
    return fw / np.where(denom > 1e-10, denom, 1.0)


def inverse_stft(bins: np.ndarray, frame_length=960, frame_step=480,
                 fft_length=960, keep_dc=True) -> np.ndarray:
    """bins: [T, num_bins, 2*C] interleaved real/imag -> [samples, C]."""
    T, nb, ch = bins.shape
    pad = (0, 1) if keep_dc else (1, 0)
    padded = np.pad(bins, ((0, 0), pad, (0, 0)))
    spec = padded[..., 0::2] + 1j * padded[..., 1::2]      # [T, fft//2+1, C]
    frames = np.fft.irfft(spec, n=fft_length, axis=1).real  # [T, fft_length, C]
    frames = frames * inverse_stft_window(frame_length, frame_step)[None, :, None]
    n = (T - 1) * frame_step + frame_length
    out = np.zeros((n, frames.shape[-1]), np.float32)
    for t in range(T):
        out[t * frame_step: t * frame_step + frame_length] += frames[t]
    return out


class SpectroStreamDecoderNumpy:
    """SpectroStream decoder. mrt2_small geometry is derived from the config."""

    # (name, strides) per decoder block, innermost first.
    BLOCKS = [
        ("decoder_0", (2, 1)), ("decoder_1", (2, 2)), ("decoder_2", (1, 2)),
        ("decoder_3", (1, 2)), ("decoder_4", (1, 3)), ("decoder_5", (1, 2)),
        ("decoder_6", (1, 2)),
    ]
    SPLIT_AFTER = "decoder_0"   # channel_recombo_block: groups split after this
    CHANNEL_SPLITS = 2
    INPUT_BINS = 5
    BASE_CHANNELS = 512

    def __init__(self, weights: dict):
        self.W = weights

    def w(self, p: str):
        return self.W["params/soundstream/" + p]

    def has(self, p: str) -> bool:
        return "params/soundstream/" + p in self.W

    def _conv(self, prefix, x, strides=(1, 1), dilation=(1, 1)):
        return conv2d(x, self.w(prefix + "/kernel"), self.w(prefix + "/bias"),
                      strides=strides, dilation=dilation)

    def _residual_block(self, name, x, strides):
        """conv2d_residual_unit with transposed=True."""
        p = f"decoder/{name}/"
        main = elu(x)
        if strides == (1, 1):
            main = self._conv(p + "conv2d_3x3_a/conv", main)
        else:
            kh_kw = (max(3, 2 * strides[0]), max(3, 2 * strides[1]))
            tname = f"conv2dtranspose_{kh_kw[0]}x{kh_kw[1]}"
            main = conv2d_transpose(main, self.w(p + tname + "/conv/kernel"),
                                    self.w(p + tname + "/conv/bias"), strides)
        main = self._conv(p + "conv2d_3x3/conv", elu(main))

        sc = x
        if self.has(p + "shortcut_layer/conv1x1/conv/kernel"):
            sc = self._conv(p + "shortcut_layer/conv1x1/conv", sc)
        if strides != (1, 1):
            sc = upsample2d(sc, strides)
        return main + sc

    def decode(self, codes: np.ndarray, return_bins: bool = False) -> np.ndarray:
        """codes: [T, num_codebooks] non-unique RVQ indices -> audio [samples, 2].

        With return_bins=True, returns the pre-ISTFT [T*4, 480, 4] tensor
        instead (used to validate the ggml port stage by stage).
        """
        q = self.w("quantizer/embedding")                      # [64, 1024, 256]
        feat = np.stack([q[i, codes[:, i]] for i in range(codes.shape[1])]).sum(0)

        ip = "decoder/input_layer/"
        main = feat @ self.w(ip + "conv1x1_first/conv/kernel")[0, 0] + \
            self.w(ip + "conv1x1_first/conv/bias")
        s = ip + "shortcut_layer/"
        h = elu(feat @ self.w(s + "conv1x1_b1/conv/kernel")[0, 0] + self.w(s + "conv1x1_b1/conv/bias"))
        sc = h @ self.w(s + "conv1x1_b2/conv/kernel")[0, 0] + self.w(s + "conv1x1_b2/conv/bias")
        x = (main + sc).reshape(-1, self.INPUT_BINS, self.BASE_CHANNELS)

        x = self._residual_block("input_layers_residual_unit", x, (1, 1))
        x = self._residual_block("decoder_0", x, (2, 1))

        # Channel groups share weights (share_scope=True in the config).
        groups = np.split(x, self.CHANNEL_SPLITS, axis=-1)
        outs = []
        for g in groups:
            for name, strides in self.BLOCKS[1:]:
                g = self._residual_block(name, g, strides)
            g = self._conv("decoder/output_layer/base_conv_last/conv", elu(g))
            outs.append(g)
        y = np.concatenate(outs, axis=-1)                      # [T, 480, 4]
        return y if return_bins else inverse_stft(y)
