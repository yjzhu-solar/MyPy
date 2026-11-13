import numpy as np
from astropy.visualization import BaseStretch


def _prepare(values, clip, out):
    values = np.asarray(values, dtype=float)
    if clip:
        return np.clip(values, 0.0, 1.0, out=out)
    if out is None:
        return np.array(values, copy=True)
    out[...] = values
    return out


class SymmetricAsinhStretch(BaseStretch):
    r"""Asinh stretch that is odd-symmetric about x = 0.5.

    .. math::
        y = \frac{1}{2} + \frac{1}{2}\,
            \frac{{\rm asinh}\left((2x - 1)/a\right)}{{\rm asinh}(1/a)}

    Used with ``ImageNormalize(vmin=-M, vmax=M)``, data = 0 maps to 0.5 and
    positive / negative values are compressed identically. Since
    2x - 1 = data / M, the curve is linear for |data| <~ a * M and
    logarithmic beyond that.
    """

    def __init__(self, a=0.1):
        super().__init__()
        if a <= 0:
            raise ValueError("a must be > 0")
        self.a = a

    def __call__(self, values, clip=True, out=None):
        v = _prepare(values, clip, out)
        v *= 2.0
        v -= 1.0                      # [0, 1] -> [-1, 1]
        v /= self.a
        np.arcsinh(v, out=v)
        v /= np.arcsinh(1.0 / self.a)
        v += 1.0
        v *= 0.5                      # [-1, 1] -> [0, 1]
        return v

    @property
    def inverse(self):
        return SymmetricSinhStretch(a=self.a)


class SymmetricSinhStretch(BaseStretch):
    """Inverse of SymmetricAsinhStretch (needed for colorbars)."""

    def __init__(self, a=0.1):
        super().__init__()
        self.a = a

    def __call__(self, values, clip=True, out=None):
        v = _prepare(values, clip, out)
        v *= 2.0
        v -= 1.0
        v *= np.arcsinh(1.0 / self.a)
        np.sinh(v, out=v)
        v *= self.a
        v += 1.0
        v *= 0.5
        return v

    @property
    def inverse(self):
        return SymmetricAsinhStretch(a=self.a)
