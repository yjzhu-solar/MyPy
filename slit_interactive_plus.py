"""
slit_interactive_plus
=====================

Space-time (time-distance) diagrams from (y, x, t) image sequences. This is a rebuild of
``slit_interactive.py`` for speed and correctness; the old module stays as the reference
for its behaviour and for reading the files it saved.

Changes from slit_interactive.py
--------------------------------
- Slits through 3 or more points use an interpolating spline, so the curve passes through
  every clicked point. Two points give a straight slit, and ``method='polyline'`` joins
  the points with straight segments.
- Every slit is sampled at uniform pixel arc length (``spacing``, default 1 px), and
  ``line_width`` always means that many parallel sample lines.
- SunPy data are resampled onto uniform world distance (Mm). The distance comes from one
  vectorised separation call instead of a Python loop over SkyCoord elements.
- Bilinear weights are computed once per slit and reused for every frame. ``cache=True``
  keeps a float32 copy of the cube for faster repeated extraction.
- Background removal works along the time axis, ignores NaNs (e.g. coalignment edges) and
  can be extended with :func:`register_background`.
- A spacetime fit keeps its pixel and world results in one object. Relocating points to
  the ridge cannot jump away on flat or monotonic profiles.
- One HDF5 file per slit (:meth:`SlitResult.save` / :meth:`SlitResult.load`).
- The GUI (:class:`SlitPickPlus`) works with ipympl (Jupyter/JupyterHub) and Qt. It redraws
  once per action, blits on Qt, throttles redraws on ipympl and never patches matplotlib
  globally. PyQt5 is optional: it only provides the save-directory dialog under Qt.

Quick start
-----------
>>> import astropy.units as u
>>> from slit_interactive_plus import SlitPickPlus, make_slit, remove_background
>>> pick = SlitPickPlus(map_sequence)          # or a (ny, nx, nt) ndarray
>>> pick(bottom_left=[100, 200]*u.pix, top_right=[500, 600]*u.pix)
>>> res = make_slit(map_sequence, [100, 180, 260], [50, 90, 70], line_width=5)
>>> clean, bg = remove_background(res.intensity, 'running_mean', window=20)
"""

import json
import os
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import FuncFormatter, MaxNLocator
from matplotlib.transforms import Bbox
from matplotlib.widgets import Button, CheckButtons, RangeSlider, Slider, TextBox
import astropy.units as u
from astropy.time import Time
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
import h5py
from scipy import ndimage
from scipy.interpolate import make_interp_spline
import sunpy.map
from sunpy.map import GenericMap, MapSequence

__all__ = ['ImageStack', 'SlitGeometry', 'SlitSampler', 'SlitResult', 'SpacetimeFit',
           'make_slit', 'make_straight_slit', 'extract_slits', 'remove_background',
           'register_background', 'BACKGROUND_METHODS', 'fit_curve_on_image',
           'plot_slit_position', 'plot_spacetime', 'preview_grid', 'SlitPickPlus']

FORMAT_VERSION = 1


# ----------------------------------------------------------------------------- data access

def _as_map_corner(corner):
    """Corner for GenericMap.submap: SkyCoord / Quantity pass through, plain numbers are pixels."""
    if isinstance(corner, u.Quantity) or hasattr(corner, 'frame'):
        return corner
    return u.Quantity(corner, u.pix)


def _as_pixel_corner(corner):
    if hasattr(corner, 'frame'):
        raise TypeError('SkyCoord corners need sunpy maps; give pixel corners for arrays')
    if isinstance(corner, u.Quantity):
        corner = corner.to_value(u.pix)
    x, y = np.round(np.asarray(corner, dtype=float)).astype(int)
    return x, y


class ImageStack:
    """
    Read access to an image sequence of shape (ny, nx, nt), whatever its container.

    Parameters
    ----------
    data : np.ndarray, MapSequence, GenericMap, list of GenericMap or 3-D array-like
        Arrays must be ordered (ny, nx, nt). Array-likes that are not ndarrays (dask,
        h5py, zarr) are read one frame at a time.
    cache : bool, optional
        Keep a float32 (ny, nx, nt) copy in memory, built on first use. This makes repeated
        slit extraction 5-12x faster and costs ny*nx*nt*4 bytes.

    Notes
    -----
    An ImageStack never modifies the data it wraps. ``crop``, ``apply`` and ``wow`` return
    new stacks.
    """

    def __init__(self, data, cache=False):
        self.maps = None
        self._array = None
        if isinstance(data, ImageStack):
            self.maps, self._array = data.maps, data._array
        elif isinstance(data, GenericMap):
            self.maps = [data]
        elif isinstance(data, MapSequence):
            self.maps = list(data.maps)
        elif (isinstance(data, (list, tuple)) and len(data) > 0
              and all(isinstance(m, GenericMap) for m in data)):
            self.maps = list(data)
        elif hasattr(data, 'shape') and hasattr(data, '__getitem__'):
            if len(data.shape) != 3:
                raise ValueError(f'array data must be 3-D (ny, nx, nt), got shape {data.shape}')
            self._array = data
        else:
            raise TypeError('data must be a (ny, nx, nt) array, a sunpy GenericMap or MapSequence, '
                            f'or a list of GenericMaps; got {type(data).__name__}')
        if self.maps is not None:
            shapes = {m.data.shape for m in self.maps}
            if len(shapes) != 1:
                raise ValueError(f'all maps must have the same shape, got {sorted(shapes)}')
            self.ny, self.nx = self.maps[0].data.shape
            self.nt = len(self.maps)
        else:
            self.ny, self.nx, self.nt = (int(n) for n in self._array.shape)
        if self.ny < 2 or self.nx < 2:
            raise ValueError(f'frames must be at least 2x2 pixels, got {self.ny}x{self.nx}')
        self.cache = bool(cache)
        self._cube = None
        self._times = None

    def __repr__(self):
        kind = 'maps' if self.is_map else type(self._array).__name__
        return f'<ImageStack {kind} ny={self.ny} nx={self.nx} nt={self.nt} cache={self.cache}>'

    @property
    def is_map(self):
        return self.maps is not None

    @property
    def shape(self):
        return (self.ny, self.nx, self.nt)

    @property
    def times(self):
        """Observation times (astropy Time) for map data, None for arrays."""
        if self.is_map and self._times is None:
            self._times = Time([m.date for m in self.maps])
        return self._times

    @property
    def unit(self):
        return self.maps[0].unit if self.is_map else None

    def frame(self, t):
        """2-D frame t (a view where possible; may be big-endian for FITS data)."""
        if self._cube is not None:
            return self._cube[:, :, t]
        if self.is_map:
            return self.maps[t].data
        return np.asarray(self._array[:, :, t])

    def cube(self):
        """The cached float32 (ny, nx, nt) cube, built on first call."""
        if self._cube is None:
            cube = np.empty((self.ny, self.nx, self.nt), dtype=np.float32)
            for t in range(self.nt):
                cube[:, :, t] = self.frame(t)
            self._cube = cube
        return self._cube

    def reference_map(self, index=0, shift=None):
        """Map `index`, with its reference coordinate shifted by `shift` = (dx, dy) if given."""
        if not self.is_map:
            return None
        m = self.maps[index]
        return m.shift_reference_coord(*shift) if shift is not None else m

    def wcs(self, index=0, shift=None):
        m = self.reference_map(index, shift)
        return None if m is None else m.wcs

    def dsun(self, index=0):
        return self.maps[index].dsun

    def crop(self, bottom_left, top_right):
        """
        New stack cut to the inclusive box [bottom_left, top_right].

        Corners are (x, y) pixels as numbers or ``u.pix`` Quantities. Map data also accept
        SkyCoord corners (passed to ``GenericMap.submap``).
        """
        if self.is_map:
            bl, tr = _as_map_corner(bottom_left), _as_map_corner(top_right)
            return ImageStack([m.submap(bl, top_right=tr) for m in self.maps], cache=self.cache)
        (x0, y0), (x1, y1) = _as_pixel_corner(bottom_left), _as_pixel_corner(top_right)
        x0, y0 = max(x0, 0), max(y0, 0)
        return ImageStack(self._array[y0:y1 + 1, x0:x1 + 1, :], cache=self.cache)

    def apply(self, func):
        """New stack with ``func(frame)`` applied to every frame; the input is not modified."""
        if self.is_map:
            return ImageStack([sunpy.map.Map(func(np.asarray(m.data)), m.meta) for m in self.maps],
                              cache=self.cache)
        return ImageStack(np.stack([func(np.asarray(self.frame(t))) for t in range(self.nt)], axis=-1),
                          cache=self.cache)

    def wow(self, **kwargs):
        """Wavelet-optimised whitening (watroo.wow) of every frame, as a new stack."""
        from watroo import wow
        return self.apply(lambda d: wow(np.asarray(d, dtype=float), **kwargs)[0])

    def sigma_over_mean(self, stride=None):
        """
        Temporal standard deviation over mean for every pixel.

        Accumulated frame by frame, so peak memory stays at a few frames. NaNs are ignored.
        ``stride`` uses every stride-th frame; by default about 100 frames are used.
        """
        if stride is None:
            stride = max(1, int(np.ceil(self.nt / 100)))
        s = np.zeros((self.ny, self.nx))
        s2 = np.zeros_like(s)
        n = np.zeros_like(s)
        for t in range(0, self.nt, stride):
            f = np.array(self.frame(t), dtype=np.float64)
            ok = np.isfinite(f)
            f[~ok] = 0.0
            s += f
            s2 += f * f
            n += ok
        with np.errstate(invalid='ignore', divide='ignore'):
            mean = s / n
            std = np.sqrt(np.maximum(s2 / n - mean ** 2, 0.0))
            return std / mean


# ----------------------------------------------------------------------------- geometry

class SlitGeometry:
    """
    Sample positions of a straight or curved slit, in pixel coordinates.

    The centre line is sampled at uniform arc length. ``line_width`` parallel lines are offset
    along the local normal by ``(k - (line_width - 1) / 2) * width_spacing``, k = 0 .. line_width-1.
    Column 0 lies on the right-hand side of the direction of travel, as in
    ``skimage.measure.profile_line``; a straight slit therefore samples exactly the same
    points as ``slit_interactive.py``.

    Parameters
    ----------
    x, y : array-like
        Control points in pixels (x = column, y = row), at least two distinct points.
    line_width : int
        Number of parallel sample lines across the slit.
    method : {'auto', 'polyline', 'spline'}
        'auto' gives 'polyline' for two points and 'spline' for more. 'spline' is an
        interpolating spline through every point (quadratic for 3 points, cubic otherwise).
    spacing : float
        Requested sample spacing along the slit in pixels. The actual spacing is
        ``length / (n_dist - 1) <= spacing``, so that both ends are sampled.
    width_spacing : float
        Spacing between the parallel lines in pixels.
    param : {'centripetal', 'chord', 'uniform'}
        Spline parameterisation. Centripetal avoids loops and cusps between unevenly spaced points.
    """

    METHODS = ('auto', 'polyline', 'spline')

    def __init__(self, x, y, line_width=5, method='auto', spacing=1.0, width_spacing=1.0,
                 param='centripetal'):
        x = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        y = np.atleast_1d(np.asarray(y, dtype=float)).ravel()
        if x.shape != y.shape:
            raise ValueError(f'x and y must have the same length, got {x.size} and {y.size}')
        if not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError('control points must be finite')
        keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 1e-9]
        x, y = x[keep], y[keep]
        if x.size < 2:
            raise ValueError('a slit needs at least two distinct points')
        if int(line_width) != line_width or line_width < 1:
            raise ValueError(f'line_width must be a positive integer, got {line_width}')
        if spacing <= 0 or width_spacing <= 0:
            raise ValueError('spacing and width_spacing must be positive')
        if method not in self.METHODS:
            raise ValueError(f'method must be one of {self.METHODS}, got {method!r}')
        if param not in ('centripetal', 'chord', 'uniform'):
            raise ValueError(f'unknown spline parameterisation {param!r}')
        if method == 'auto':
            method = 'polyline' if x.size == 2 else 'spline'

        self.control_x, self.control_y = x, y
        self.line_width = int(line_width)
        self.method = method
        self.param = param
        self.requested_spacing = float(spacing)
        self.width_spacing = float(width_spacing)

        cx, cy, tx, ty, s = self._centre_line()
        self.centre_x, self.centre_y, self.arc_length = cx, cy, s
        self.tangent_x, self.tangent_y = tx, ty
        offsets = (np.arange(self.line_width) - (self.line_width - 1) / 2) * self.width_spacing
        self.x = cx[:, None] - ty[:, None] * offsets[None, :]
        self.y = cy[:, None] + tx[:, None] * offsets[None, :]

    @classmethod
    def from_center(cls, x0, y0, length, angle, line_width=5, spacing=1.0, width_spacing=1.0):
        """Straight slit of `length` pixels centred on (x0, y0); `angle` in degrees CCW from +x."""
        a = np.deg2rad(angle)
        dx, dy = 0.5 * length * np.cos(a), 0.5 * length * np.sin(a)
        return cls([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], line_width, 'polyline', spacing,
                   width_spacing)

    def _parameter(self):
        d = np.hypot(np.diff(self.control_x), np.diff(self.control_y))
        dt = {'centripetal': np.sqrt(d), 'chord': d, 'uniform': np.ones_like(d)}[self.param]
        t = np.r_[0.0, np.cumsum(dt)]
        return t / t[-1]

    def _centre_line(self):
        x, y = self.control_x, self.control_y
        if self.method == 'polyline':
            knots = np.r_[0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
            length = knots[-1]
            n = int(np.ceil(length / self.requested_spacing)) + 1
            s = np.linspace(0.0, length, n)
            cx, cy = np.interp(s, knots, x), np.interp(s, knots, y)
            dx, dy = np.gradient(cx), np.gradient(cy)
        else:
            spline = make_interp_spline(self._parameter(), np.c_[x, y], k=min(3, x.size - 1))
            polygon = np.hypot(np.diff(x), np.diff(y)).sum()
            t_dense = np.linspace(0.0, 1.0, max(1000, int(20 * polygon) + 1))
            xy = spline(t_dense)
            s_dense = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            length = s_dense[-1]
            n = int(np.ceil(length / self.requested_spacing)) + 1
            s = np.linspace(0.0, length, n)
            t = np.interp(s, s_dense, t_dense)
            cx, cy = spline(t).T
            dx, dy = spline.derivative()(t).T
            bad = np.hypot(dx, dy) == 0
            if bad.any():
                gx, gy = np.gradient(cx), np.gradient(cy)
                dx, dy = np.where(bad, gx, dx), np.where(bad, gy, dy)
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1.0
        return cx, cy, dx / norm, dy / norm, s

    @property
    def n_dist(self):
        return self.centre_x.size

    @property
    def n_width(self):
        return self.line_width

    @property
    def length(self):
        """Arc length of the centre line in pixels."""
        return float(self.arc_length[-1])

    @property
    def spacing(self):
        """Actual sample spacing along the slit in pixels."""
        return self.length / (self.n_dist - 1)

    def boundary(self):
        """Closed outline (x, y) around all sample points."""
        bx = np.r_[self.x[:, 0], self.x[-1, 1:], self.x[::-1, -1], self.x[0, ::-1]]
        by = np.r_[self.y[:, 0], self.y[-1, 1:], self.y[::-1, -1], self.y[0, ::-1]]
        return bx, by

    def position_at(self, s):
        """Pixel position (x, y) of the centre line at arc length `s` (pixels)."""
        return np.interp(s, self.arc_length, self.centre_x), np.interp(s, self.arc_length, self.centre_y)

    def to_dict(self):
        return dict(control_x=self.control_x, control_y=self.control_y, line_width=self.line_width,
                    method=self.method, spacing=self.requested_spacing,
                    width_spacing=self.width_spacing, param=self.param)

    def __repr__(self):
        return (f'<SlitGeometry {self.method} n_points={self.control_x.size} length={self.length:.1f}px '
                f'n_dist={self.n_dist} line_width={self.line_width}>')


# ----------------------------------------------------------------------------- sampling

class SlitSampler:
    """
    Bilinear interpolation at fixed pixel positions, set up once and applied to many frames.

    Positions outside the frame (beyond the centres of the edge pixels) give NaN. A NaN pixel
    affects a sample only when its interpolation weight is non-zero.
    """

    def __init__(self, x, y, ny, nx):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.shape = x.shape
        x, y = x.ravel(), y.ravel()
        inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
        x0 = np.clip(np.floor(x), 0, nx - 2).astype(np.intp)
        y0 = np.clip(np.floor(y), 0, ny - 2).astype(np.intp)
        dx = np.clip(x - x0, 0.0, 1.0)
        dy = np.clip(y - y0, 0.0, 1.0)
        iy = np.stack([y0, y0, y0 + 1, y0 + 1])
        ix = np.stack([x0, x0 + 1, x0, x0 + 1])
        w = np.stack([(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx])
        # Point zero-weight corners at the dominant corner, so a NaN there cannot leak in.
        main = np.argmax(w, axis=0)
        cols = np.arange(w.shape[1])
        zero = w == 0
        iy = np.where(zero, iy[main, cols], iy)
        ix = np.where(zero, ix[main, cols], ix)
        iy[:, ~inside] = 0
        ix[:, ~inside] = 0
        self.iy, self.ix, self.w, self.inside = iy, ix, w, inside

    @property
    def n_samples(self):
        return self.w.shape[1]

    def sample(self, frame):
        """Values of one 2-D frame at the sample positions (shape of the input positions)."""
        v = (frame[self.iy, self.ix] * self.w).sum(axis=0)
        v[~self.inside] = np.nan
        return v.reshape(self.shape)

    def sample_stack(self, stack, frames=None):
        """(n_samples, n_frames) array for the given frame indices (default: all frames)."""
        frames = np.arange(stack.nt) if frames is None else np.asarray(frames, dtype=int)
        if stack.cache:
            cube = stack.cube()
            sel = slice(None) if np.array_equal(frames, np.arange(stack.nt)) else frames
            out = np.zeros((self.n_samples, len(frames)))
            for k in range(4):
                out += cube[self.iy[k], self.ix[k]][:, sel] * self.w[k][:, None]
        else:
            out = np.empty((self.n_samples, len(frames)))
            for j, t in enumerate(frames):
                out[:, j] = (stack.frame(t)[self.iy, self.ix] * self.w).sum(axis=0)
        out[~self.inside] = np.nan
        return out


REDUCERS = {'mean': np.nanmean, 'median': np.nanmedian, 'sum': np.nansum, 'max': np.nanmax}


def _resolve_reducer(reducer):
    if callable(reducer):
        return reducer, getattr(reducer, '__name__', 'custom')
    if reducer not in REDUCERS:
        raise ValueError(f'reducer must be callable or one of {sorted(REDUCERS)}, got {reducer!r}')
    return REDUCERS[reducer], reducer


def _resample_rows(values, x_old, x_new):
    """Linear interpolation along axis 0 (like np.interp per column, NaN only where weighted)."""
    j = np.clip(np.searchsorted(x_old, x_new, side='right') - 1, 0, len(x_old) - 2)
    w = (x_new - x_old[j]) / (x_old[j + 1] - x_old[j])
    w = np.clip(w, 0.0, 1.0)[:, None]
    lo, hi = values[j], values[j + 1]
    with np.errstate(invalid='ignore'):
        out = lo * (1 - w) + hi * w
    out = np.where(w == 0, lo, out)
    return np.where(w == 1, hi, out)


def _interp_extrap(x, xp, fp):
    """np.interp with linear extrapolation beyond both ends."""
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    if len(xp) > 1:
        lo, hi = x < xp[0], x > xp[-1]
        y = np.where(lo, fp[0] + (x - xp[0]) * (fp[1] - fp[0]) / (xp[1] - xp[0]), y)
        y = np.where(hi, fp[-1] + (x - xp[-1]) * (fp[-1] - fp[-2]) / (xp[-1] - xp[-2]), y)
    return y


def _json_safe(value):
    if isinstance(value, u.Quantity):
        return {'value': np.asarray(value.value).tolist(), 'unit': str(value.unit)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def extract_slits(data, geometries, reducer='mean', wcs_index=0, wcs_shift=None, frames=None,
                  cache=False):
    """
    Extract space-time diagrams for several slits in one pass over the frames.

    Parameters
    ----------
    data : ImageStack or anything ImageStack accepts
    geometries : sequence of SlitGeometry
    reducer : {'mean', 'median', 'sum', 'max'} or callable(array, axis=1)
        Combines the parallel lines across the slit (NaN-aware by default).
    wcs_index : int
        Frame whose WCS and observer distance define world coordinates (map data).
    wcs_shift : Quantity (dx, dy), optional
        Shift of the reference coordinate, as in ``GenericMap.shift_reference_coord``. It
        changes only the world coordinates of the slit, never the data or headers.
    frames : array of int, optional
        Frame indices to use (default: all).
    cache : bool
        Only used when `data` is not already an ImageStack.

    Returns
    -------
    list of SlitResult
    """
    stack = data if isinstance(data, ImageStack) else ImageStack(data, cache=cache)
    geometries = list(geometries)
    reduce_fn, reducer_name = _resolve_reducer(reducer)
    frames = np.arange(stack.nt) if frames is None else np.asarray(frames, dtype=int)

    sizes = [g.x.size for g in geometries]
    sampler = SlitSampler(np.concatenate([g.x.ravel() for g in geometries]),
                          np.concatenate([g.y.ravel() for g in geometries]), stack.ny, stack.nx)
    samples = sampler.sample_stack(stack, frames)
    bounds = np.r_[0, np.cumsum(sizes)]

    times = stack.times[frames] if stack.is_map else None
    if stack.is_map:
        wcs = stack.wcs(wcs_index, wcs_shift)
        dsun_mm = stack.dsun(wcs_index).to_value(u.Mm)
        cbounds = np.r_[0, np.cumsum([g.n_dist for g in geometries])]
        coords = wcs.pixel_to_world(np.concatenate([g.centre_x for g in geometries]),
                                    np.concatenate([g.centre_y for g in geometries]))
        sep = coords[1:].separation(coords[:-1]).to_value(u.rad)

    meta = dict(source='map' if stack.is_map else 'array', reducer=reducer_name,
                wcs_index=int(wcs_index),
                wcs_shift=None if wcs_shift is None else u.Quantity(wcs_shift).to_value(u.arcsec).tolist())
    results = []
    for i, g in enumerate(geometries):
        block = samples[bounds[i]:bounds[i + 1]].reshape(g.n_dist, g.n_width, len(frames))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            profile = reduce_fn(block, axis=1)
        if stack.is_map:
            c0, c1 = cbounds[i], cbounds[i + 1]
            dist = np.r_[0.0, np.cumsum(sep[c0:c1 - 1])] * dsun_mm
            uniform = np.linspace(0.0, dist[-1], g.n_dist)
            intensity = _resample_rows(profile, dist, uniform)
            distance = uniform * u.Mm
            distance_pix = np.interp(uniform, dist, g.arc_length)
            centre = coords[c0:c1]
        else:
            intensity = profile
            distance = g.arc_length * u.pix
            distance_pix = g.arc_length.copy()
            centre = None
        results.append(SlitResult(intensity=np.asarray(intensity, dtype=float), distance=distance,
                                  geometry=g, frames=frames.copy(), times=times, unit=stack.unit,
                                  distance_pix=distance_pix, centre_coord=centre, meta=dict(meta)))
    return results


def make_slit(data, x, y, line_width=5, method='auto', spacing=1.0, reducer='mean', wcs_index=0,
              wcs_shift=None, frames=None, cache=False):
    """Space-time diagram along the slit through control points (x, y). Returns a SlitResult."""
    geometry = SlitGeometry(x, y, line_width=line_width, method=method, spacing=spacing)
    return extract_slits(data, [geometry], reducer, wcs_index, wcs_shift, frames, cache)[0]


def make_straight_slit(data, x0, y0, length, angle, line_width=5, spacing=1.0, reducer='mean',
                       wcs_index=0, wcs_shift=None, frames=None, cache=False):
    """Straight slit centred on (x0, y0), `length` pixels long, `angle` degrees CCW from +x."""
    geometry = SlitGeometry.from_center(x0, y0, length, angle, line_width, spacing)
    return extract_slits(data, [geometry], reducer, wcs_index, wcs_shift, frames, cache)[0]


# ----------------------------------------------------------------------------- results

@dataclass
class SlitResult:
    """
    A space-time diagram and everything needed to interpret it.

    Attributes
    ----------
    intensity : ndarray (n_dist, n_time)
        Rows are positions along the slit, columns are frames.
    distance : Quantity (n_dist,)
        Uniformly spaced distance along the slit: Mm for map data (uniform world distance),
        pixels for arrays.
    geometry : SlitGeometry
    frames : ndarray of int
        Frame index of every column.
    times : Time or None
        Observation time of every column (map data).
    unit : astropy unit or None
        Unit of the intensities (the map unit).
    distance_pix : ndarray (n_dist,)
        Arc length in pixels of every row along the slit centre line.
    centre_coord : SkyCoord or None
        World coordinates of the geometry's centre samples (map data).
    meta : dict
        Extraction settings (reducer, wcs_index, wcs_shift, source, ...).
    """
    intensity: np.ndarray
    distance: u.Quantity
    geometry: SlitGeometry
    frames: np.ndarray
    times: Time = None
    unit: object = None
    distance_pix: np.ndarray = None
    centre_coord: object = None
    meta: dict = field(default_factory=dict)

    def __repr__(self):
        return (f'<SlitResult {self.n_dist}x{self.n_time} distance 0-{self.distance[-1]:.3g} '
                f'{self.geometry.method} lw={self.geometry.line_width}>')

    @property
    def n_dist(self):
        return self.intensity.shape[0]

    @property
    def n_time(self):
        return self.intensity.shape[1]

    @property
    def time_seconds(self):
        """Seconds since the first column (map data) or the frame index (arrays)."""
        if self.times is None:
            return self.frames.astype(float)
        return (self.times - self.times[0]).to_value(u.s)

    @property
    def distance_step(self):
        return (self.distance[-1] - self.distance[0]) / max(self.n_dist - 1, 1)

    def time_at(self, column):
        """Seconds since the first column (or frame number) at fractional column index."""
        return _interp_extrap(column, np.arange(self.n_time), self.time_seconds)

    def distance_at(self, row):
        """Distance at fractional row index (Quantity)."""
        return self.distance[0] + np.asarray(row, dtype=float) * self.distance_step

    def position_at(self, row):
        """Pixel position (x, y) on the image of a (fractional) row of the diagram."""
        s = np.interp(row, np.arange(self.n_dist), self.distance_pix)
        return self.geometry.position_at(s)

    def remove_background(self, method='median', **params):
        """Shortcut for ``remove_background(self.intensity, method, **params)``."""
        return remove_background(self.intensity, method, **params)

    def to_ndcube(self):
        """NDCube with a time lookup table (axis 1) and a distance lookup table (axis 0)."""
        from ndcube import NDCube
        from ndcube.extra_coords import QuantityTableCoordinate, TimeTableCoordinate
        if self.times is None:
            raise ValueError('to_ndcube needs observation times (map data)')
        wcs = (TimeTableCoordinate(self.times, physical_types='time', names='time')
               & QuantityTableCoordinate(self.distance, physical_types='length', names='distance')).wcs
        return NDCube(self.intensity, wcs, unit=self.unit)

    # -- I/O
    def save(self, path, fits=(), overwrite=True):
        """
        Write the result and its spacetime fits to one HDF5 file. Read it back with SlitResult.load.
        """
        with h5py.File(path, 'w' if overwrite else 'w-') as f:
            f.attrs['format'] = 'slit_interactive_plus'
            f.attrs['format_version'] = FORMAT_VERSION
            f.attrs['created'] = datetime.now(timezone.utc).isoformat()
            g = f.create_group('geometry')
            for key, value in self.geometry.to_dict().items():
                if isinstance(value, np.ndarray):
                    g[key] = value
                else:
                    g.attrs[key] = value
            for key in ('centre_x', 'centre_y', 'arc_length', 'x', 'y'):
                g[key] = getattr(self.geometry, key)
            s = f.create_group('slit')
            s['intensity'] = self.intensity
            s['intensity'].attrs['unit'] = '' if self.unit is None else str(self.unit)
            s['distance'] = self.distance.value
            s['distance'].attrs['unit'] = str(self.distance.unit)
            s['distance_pix'] = self.distance_pix
            s['frames'] = self.frames
            if self.times is not None:
                s['time_isot'] = np.array(self.times.isot, dtype='S')
                s['time_isot'].attrs['scale'] = self.times.scale
            s.attrs['meta'] = json.dumps(_json_safe(self.meta))
            fg = f.create_group('fits')
            for i, fit in enumerate(fits):
                fit._write(fg.create_group(f'fit_{i:03d}'))
        return path

    @classmethod
    def load(cls, path):
        """Read a file written by save. Returns (SlitResult, list of SpacetimeFit)."""
        with h5py.File(path, 'r') as f:
            if f.attrs.get('format') != 'slit_interactive_plus':
                raise ValueError(f'{path} was not written by slit_interactive_plus')
            g = f['geometry']
            geometry = SlitGeometry(g['control_x'][()], g['control_y'][()],
                                    line_width=int(g.attrs['line_width']), method=str(g.attrs['method']),
                                    spacing=float(g.attrs['spacing']),
                                    width_spacing=float(g.attrs['width_spacing']),
                                    param=str(g.attrs['param']))
            s = f['slit']
            unit = s['intensity'].attrs['unit']
            times = None
            if 'time_isot' in s:
                times = Time([t.decode() for t in s['time_isot'][()]], format='isot',
                             scale=s['time_isot'].attrs.get('scale', 'utc'))
            result = cls(intensity=s['intensity'][()],
                         distance=s['distance'][()] * u.Unit(s['distance'].attrs['unit']),
                         geometry=geometry, frames=s['frames'][()], times=times,
                         unit=u.Unit(unit) if unit else None, distance_pix=s['distance_pix'][()],
                         meta=json.loads(s.attrs['meta']))
            fits = [SpacetimeFit._read(f['fits'][k]) for k in sorted(f['fits'])]
        return result, fits

    # -- plotting
    def plot_spacetime(self, ax, data=None, **kwargs):
        """imshow of the diagram (or of `data` with the same shape) on `ax`; see plot_spacetime."""
        return plot_spacetime(ax, self, data=data, **kwargs)

    def plot_position(self, ax, **kwargs):
        """Outline of the slit on an image axis; see plot_slit_position."""
        return plot_slit_position(ax, self.geometry, **kwargs)


# ----------------------------------------------------------------------------- background

BACKGROUND_METHODS = {}
BACKGROUND_DEFAULTS = {}


def register_background(name, **defaults):
    """
    Decorator that adds a background method usable by remove_background and the GUI.

    The function receives the (n_dist, n_time) intensity array plus keyword parameters, and
    returns a background of the same shape (or broadcastable to it).

    >>> @register_background('row_min')
    ... def row_min(intensity):
    ...     return np.nanmin(intensity, axis=1, keepdims=True)
    """
    def decorator(func):
        BACKGROUND_METHODS[name] = func
        BACKGROUND_DEFAULTS[name] = dict(defaults)
        return func
    return decorator


def _filled(st):
    """Copy of st with NaNs replaced by the row median, and the NaN mask (None if no NaNs)."""
    nan = ~np.isfinite(st)
    if not nan.any():
        return st, None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        fill = np.nanmedian(st, axis=1, keepdims=True)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    return np.where(nan, fill, st), nan


def _nan_safe_filter(func):
    """Run a sliding filter on NaN-filled data and put the NaNs back afterwards."""
    def wrapper(st, **kw):
        filled, nan = _filled(st)
        bg = func(filled, **kw)
        if nan is not None:
            bg = np.where(nan, np.nan, bg)
        return bg
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


@register_background('none')
def _bg_none(st):
    """No background (returns zeros)."""
    return np.zeros_like(st)


@register_background('median')
def _bg_median(st):
    """Median over time for every row."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if np.isfinite(st).all():
            return np.median(st, axis=1, keepdims=True)
        return np.nanmedian(st, axis=1, keepdims=True)


@register_background('percentile', percentile=10)
def _bg_percentile(st, percentile=10):
    """Given percentile over time for every row."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if np.isfinite(st).all():
            return np.percentile(st, percentile, axis=1, keepdims=True)
        return np.nanpercentile(st, percentile, axis=1, keepdims=True)


@register_background('morphological', window=5)
@_nan_safe_filter
def _bg_morphological(st, window=5):
    """Grey opening along time with a `window`-frame line element (keeps sharp peaks out)."""
    return ndimage.grey_opening(st, size=(1, int(window)))


@register_background('running_min', window=15)
@_nan_safe_filter
def _bg_running_min(st, window=15):
    """Running minimum along time."""
    return ndimage.minimum_filter1d(st, int(window), axis=1)


@register_background('running_percentile', window=15, percentile=10)
@_nan_safe_filter
def _bg_running_percentile(st, window=15, percentile=10):
    """Running percentile along time."""
    out = np.empty_like(st, dtype=float)
    for i in range(st.shape[0]):  # faster than the 2-D filter for long rows
        out[i] = ndimage.percentile_filter(st[i], percentile, size=int(window))
    return out


@register_background('running_mean', window=15)
@_nan_safe_filter
def _bg_running_mean(st, window=15):
    """Running mean along time."""
    return ndimage.uniform_filter1d(st, int(window), axis=1)


@register_background('gaussian', sigma=5.0)
@_nan_safe_filter
def _bg_gaussian(st, sigma=5.0):
    """Gaussian smoothing along time (sigma in frames)."""
    return ndimage.gaussian_filter1d(st, float(sigma), axis=1)


def remove_background(intensity, method='median', **params):
    """
    Subtract a background estimated along the time axis.

    Parameters
    ----------
    intensity : ndarray (n_dist, n_time)
    method : str
        A name in BACKGROUND_METHODS: 'median' (default), 'percentile' (percentile=10),
        'morphological' (window=5), 'running_min' (window=15), 'running_percentile'
        (window=15, percentile=10), 'running_mean' (window=15), 'gaussian' (sigma=5),
        'none', or one added with register_background. Windows are in frames.
    **params
        Overrides of the method's defaults.

    Returns
    -------
    (intensity - background, background)
        A NaN input pixel stays NaN; it does not spread through its row.
    """
    if method not in BACKGROUND_METHODS:
        raise ValueError(f'unknown background method {method!r}; known: {sorted(BACKGROUND_METHODS)}')
    st = np.asarray(intensity, dtype=float)
    if st.ndim != 2:
        raise ValueError(f'intensity must be 2-D (n_dist, n_time), got shape {st.shape}')
    background = BACKGROUND_METHODS[method](st, **{**BACKGROUND_DEFAULTS[method], **params})
    background = np.broadcast_to(background, st.shape)
    return st - background, np.array(background)


# ----------------------------------------------------------------------------- fitting

def _relocate_to_ridge(x, y, image, half_window):
    """
    Move each y to the local maximum within +-half_window rows in its column, refined to sub-pixel.

    A point moves only if the brightest pixel of its window is a true local maximum of the
    column (not a window edge on a slope). Otherwise it keeps its position and is flagged False.
    """
    ny, nx = image.shape
    y_new = np.array(y, dtype=float)
    ok = np.zeros(len(y_new), dtype=bool)
    for i, (xv, yv) in enumerate(zip(x, y)):
        xi, yi = int(round(xv)), int(round(yv))
        if not (0 <= xi < nx and 0 <= yi < ny):
            continue
        lo, hi = max(0, yi - half_window), min(ny, yi + half_window + 1)
        column = image[lo:hi, xi]
        if not np.isfinite(column).any():
            continue
        m = lo + int(np.nanargmax(column))
        b = image[m, xi]
        a = image[m - 1, xi] if m > 0 else -np.inf
        c = image[m + 1, xi] if m < ny - 1 else -np.inf
        if (np.isfinite(a) and a > b) or (np.isfinite(c) and c > b):
            continue  # the window only covers a slope
        best = float(m)
        curvature = a - 2 * b + c
        if np.isfinite(curvature) and curvature < 0:
            best = m + float(np.clip(0.5 * (a - c) / curvature, -0.5, 0.5))
        y_new[i] = best
        ok[i] = True
    return y_new, ok


def fit_curve_on_image(xdata, ydata, image, poly_order=2, relocate=False, window_half_size=1,
                       fit_weights=None):
    """
    Fit y(x) = polynomial to points on a 2-D image, optionally snapping them to a bright ridge.

    With ``relocate=True`` every point moves to the brightest pixel within
    ``window_half_size`` rows in its column, provided that pixel is a local maximum of the
    column. The position is refined with a 3-point parabola (at most +-0.5 px). Points that
    cannot be relocated keep their position and are flagged False in ``relocation_ok``:
    points outside the image, all-NaN columns, and windows that only cover a slope.

    Returns
    -------
    dict with fit_params (highest order first), xdata_original, ydata_original, xdata_fitted,
    ydata_fitted, fit_curve, relocated, relocation_ok, poly_order
    """
    x = np.asarray(xdata, dtype=float).ravel()
    y = np.asarray(ydata, dtype=float).ravel()
    if x.size != y.size:
        raise ValueError(f'xdata and ydata must have the same length, got {x.size} and {y.size}')
    if poly_order < 0:
        raise ValueError(f'poly_order must be >= 0, got {poly_order}')
    if x.size < poly_order + 1:
        raise ValueError(f'need at least {poly_order + 1} points for order {poly_order}, got {x.size}')
    if relocate:
        y_fit, ok = _relocate_to_ridge(x, y, np.asarray(image, dtype=float), int(window_half_size))
    else:
        y_fit, ok = y.copy(), np.ones(x.size, dtype=bool)
    params = np.polyfit(x, y_fit, poly_order, w=fit_weights)
    return dict(fit_params=params, xdata_original=x, ydata_original=y, xdata_fitted=x.copy(),
                ydata_fitted=y_fit, fit_curve=np.polyval(params, x), relocated=bool(relocate),
                relocation_ok=ok, poly_order=int(poly_order))


@dataclass
class SpacetimeFit:
    """
    Polynomial fit to a feature in a space-time diagram, in index and world coordinates.

    ``x``/``y`` are column/row indices of the diagram. ``t``/``d`` are the same points in world
    units: seconds since the first column and km (map data), or frame and pixels (arrays).
    ``params_world`` is the polynomial d(t), highest order first.
    """
    x: np.ndarray
    y: np.ndarray
    params: np.ndarray
    poly_order: int
    t: np.ndarray
    d: np.ndarray
    params_world: np.ndarray
    world_units: tuple = ('s', 'km')
    relocated: bool = False
    relocation_ok: np.ndarray = None
    artist: object = field(default=None, repr=False, compare=False)

    @classmethod
    def from_points(cls, result, x, y, poly_order=2, relocate=False, image=None, window_half_size=1):
        """Fit points (column, row) picked on `result`'s diagram; `image` is used for relocation."""
        image = result.intensity if image is None else image
        fit = fit_curve_on_image(x, y, image, poly_order, relocate, window_half_size)
        x, y = fit['xdata_fitted'], fit['ydata_fitted']
        t = result.time_at(x)
        if result.times is not None:
            d = result.distance_at(y).to_value(u.km)
            units = ('s', 'km')
        else:
            d = result.distance_at(y).to_value(u.pix)
            units = ('frame', 'pix')
        return cls(x=x, y=y, params=fit['fit_params'], poly_order=int(poly_order), t=t, d=d,
                   params_world=np.polyfit(t, d, poly_order), world_units=units,
                   relocated=bool(relocate), relocation_ok=fit['relocation_ok'])

    def curve(self, n=200):
        """Fitted curve in index coordinates over the x range of the points."""
        xc = np.linspace(self.x.min(), self.x.max(), n)
        return xc, np.polyval(self.params, xc)

    def velocity(self, t=None):
        """d'(t) in world units (km/s for map data) at times `t` (default: the fitted points)."""
        t = self.t if t is None else np.asarray(t, dtype=float)
        return np.polyval(np.polyder(self.params_world), t)

    def _write(self, group):
        for key in ('x', 'y', 'params', 't', 'd', 'params_world', 'relocation_ok'):
            group[key] = getattr(self, key)
        group.attrs['poly_order'] = self.poly_order
        group.attrs['relocated'] = self.relocated
        group.attrs['world_units'] = list(self.world_units)

    @classmethod
    def _read(cls, group):
        return cls(x=group['x'][()], y=group['y'][()], params=group['params'][()],
                   poly_order=int(group.attrs['poly_order']), t=group['t'][()], d=group['d'][()],
                   params_world=group['params_world'][()],
                   world_units=tuple(str(s) for s in group.attrs['world_units']),
                   relocated=bool(group.attrs['relocated']), relocation_ok=group['relocation_ok'][()])


# ----------------------------------------------------------------------------- plotting

def plot_slit_position(ax, slit, color='#58B2DC', curve_color='#FF6B6B', show_boundary=True,
                       show_curve=True, show_points=True, show_direction=True, direction_size=20,
                       direction_ratio=0.6, label=None, alpha=0.9):
    """
    Draw a slit outline on an image axis (pixel coordinates; WCSAxes are fine).

    Parameters
    ----------
    slit : SlitGeometry or SlitResult
    color, curve_color : colours of the outline and of the centre line and control points
    show_direction : bool
        Draw a triangle on the outline pointing in the direction of increasing distance.
    label : str, optional
        Text placed next to the direction triangle.

    Returns
    -------
    dict of the created artists ('boundary', 'curve', 'points', 'triangle', 'text')
    """
    g = slit.geometry if isinstance(slit, SlitResult) else slit
    artists = dict(boundary=None, curve=None, points=None, triangle=None, text=None)
    if show_boundary:
        bx, by = g.boundary()
        artists['boundary'] = ax.plot(bx, by, color=color, lw=1, alpha=0.8)[0]
    if show_curve:
        artists['curve'] = ax.plot(g.centre_x, g.centre_y, color=curve_color, lw=1.5, alpha=alpha)[0]
    if show_points:
        artists['points'] = ax.plot(g.control_x, g.control_y, 'o', ms=6, mfc=curve_color, mec='white',
                                    mew=1.5, ls='none', alpha=alpha)[0]
    if show_direction and g.n_dist > 2:
        i0 = min(g.n_dist // 4, g.n_dist - 2)
        i1 = int(np.argmin(np.abs(g.arc_length - (g.arc_length[i0] + direction_size))))
        i1 = max(i1, i0 + 1)
        p0 = np.array([g.x[i0, 0], g.y[i0, 0]])
        p1 = np.array([g.x[i1, 0], g.y[i1, 0]])
        base = p1 - p0
        apex = p0 + np.array([base[1], -base[0]]) * direction_ratio
        artists['triangle'] = ax.add_patch(Polygon(np.vstack([p0, p1, apex]), closed=True, color=color,
                                                   alpha=0.8))
        if label is not None:
            normal = np.array([base[1], -base[0]]) / max(np.hypot(*base), 1e-9)
            pos = (p0 + p1) / 2 + normal * (direction_ratio * np.hypot(*base) + 5)
            artists['text'] = ax.text(*pos, label, color=color, ha='center', va='center', fontsize=9)
    return artists


def _time_formatter(result, fmt='%H:%M:%S'):
    if result.times is None:
        return FuncFormatter(lambda v, pos: f'{v:.0f}')
    t0 = result.times[0].to_datetime()
    columns = np.arange(result.n_time)
    seconds = result.time_seconds

    def fmt_tick(v, pos=None):
        return (t0 + timedelta(seconds=float(_interp_extrap(v, columns, seconds)))).strftime(fmt)
    return FuncFormatter(fmt_tick)


def _distance_formatter(result):
    d0, step = result.distance[0].value, result.distance_step.value
    return FuncFormatter(lambda v, pos: f'{d0 + v * step:.3g}')


def plot_spacetime(ax, result, data=None, norm=None, cmap='magma', time_format='%H:%M:%S', **kwargs):
    """
    Show a space-time diagram in column/row index coordinates with time and distance tick labels.

    Mouse positions on the axis are column (time) and row (distance) indices, the coordinates
    that SpacetimeFit expects.
    """
    data = result.intensity if data is None else data
    im = ax.imshow(data, origin='lower', aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest', **kwargs)
    _decorate_spacetime_axis(ax, result, time_format)
    return im


def _decorate_spacetime_axis(ax, result, time_format='%H:%M:%S'):
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.xaxis.set_major_formatter(_time_formatter(result, time_format))
    ax.yaxis.set_major_formatter(_distance_formatter(result))
    if result.times is not None:
        ax.set_xlabel(f'Time [UTC] from {result.times[0].iso[:19]}')
    else:
        ax.set_xlabel('Frame')
    ax.set_ylabel(f'Distance [{result.distance.unit.to_string()}]')

    def format_coord(x, y):
        text = f'column={x:.1f} row={y:.1f}'
        if result.times is not None:
            text += f'  t={float(result.time_at(x)):.1f} s'
        return text + f'  d={result.distance_at(y):.4g}'
    ax.format_coord = format_coord


# ----------------------------------------------------------------------------- batch previews

_PREVIEW = {}


def _preview_init(image, sigma, cmap, dpi):
    _PREVIEW.update(image=image, sigma=sigma, cmap=cmap, dpi=dpi,
                    norm_image=ImageNormalize(image, vmin=np.nanpercentile(image, 1),
                                              vmax=np.nanpercentile(image, 99.5),
                                              stretch=AsinhStretch(0.5)),
                    norm_sigma=ImageNormalize(sigma, vmin=np.nanpercentile(sigma, 1),
                                              vmax=np.nanpercentile(sigma, 99),
                                              stretch=AsinhStretch(0.5)))


def _preview_init_from_file(path, cmap, dpi):
    # Arrays come through a file: large initargs would block the parent while each spawned
    # worker imports this module, so the workers would start one after another.
    with np.load(path) as data:
        _preview_init(data['image'], data['sigma'], cmap, dpi)


def _render_preview(task):
    from matplotlib.figure import Figure
    boundary_x, boundary_y, diagram, filename, title = task
    fig = Figure(figsize=(7, 6), layout='constrained')
    gs = fig.add_gridspec(2, 2)
    ax1, ax2, ax3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    ax1.imshow(_PREVIEW['image'], origin='lower', cmap=_PREVIEW['cmap'], norm=_PREVIEW['norm_image'])
    ax2.imshow(_PREVIEW['sigma'], origin='lower', cmap=_PREVIEW['cmap'], norm=_PREVIEW['norm_sigma'])
    for ax in (ax1, ax2):
        ax.plot(boundary_x, boundary_y, color='#58B2DC', lw=1)
    finite = diagram[np.isfinite(diagram)]
    norm = (ImageNormalize(finite, interval=ZScaleInterval(), stretch=AsinhStretch(0.5))
            if finite.size else None)
    ax3.imshow(diagram, origin='lower', aspect='auto', cmap=_PREVIEW['cmap'], norm=norm)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Row')
    fig.suptitle(title, fontsize=10)
    fig.savefig(filename, dpi=_PREVIEW['dpi'])
    return filename


def preview_grid(data, n_x=9, n_y=9, n_angles=4, length=15, line_width=5, save_dir=None,
                 background='median', reference_frame=0, wcs_index=0, wcs_shift=None, reducer='mean',
                 ncpu=1, dpi=150, cmap='magma', sigma_stride=None, cache=False):
    """
    Short straight slits on a regular grid, to scan a field for oscillations or flows.

    Slit centres lie on an interior n_x by n_y grid (the image edges are excluded). Each centre
    gets n_angles orientations, spaced by 180/n_angles degrees CCW from +x. All slits are
    extracted in one pass over the frames. With `save_dir`, one PNG per slit is written
    (image, sigma/mean map, and the background-removed diagram), about 0.4 s each at 150 dpi.
    ncpu > 1 renders in worker processes. Starting them costs about 6 s, so this pays off
    from a few dozen figures on. Scripts that use it need an ``if __name__ == '__main__':``
    guard on macOS and Windows.

    Returns
    -------
    list of SlitResult, in the order x, y, angle
    """
    stack = data if isinstance(data, ImageStack) else ImageStack(data, cache=cache)
    xs = np.linspace(0, stack.nx - 1, n_x + 2)[1:-1]
    ys = np.linspace(0, stack.ny - 1, n_y + 2)[1:-1]
    angles = np.arange(n_angles) * 180.0 / n_angles
    geometries, names = [], []
    for x0 in xs:
        for y0 in ys:
            for angle in angles:
                geometries.append(SlitGeometry.from_center(x0, y0, length, angle, line_width))
                names.append((x0, y0, angle))
    results = extract_slits(stack, geometries, reducer, wcs_index, wcs_shift)
    if save_dir is None:
        return results

    os.makedirs(save_dir, exist_ok=True)
    tasks = []
    for (x0, y0, angle), res in zip(names, results):
        diagram = remove_background(res.intensity, background)[0]
        bx, by = res.geometry.boundary()
        tasks.append((bx, by, diagram,
                      os.path.join(save_dir, f'slit_x{x0:04.0f}_y{y0:04.0f}_a{angle:03.0f}.png'),
                      f'centre ({x0:.0f}, {y0:.0f}) px, angle {angle:.0f} deg, length {length} px'))
    image = np.array(stack.frame(reference_frame), dtype=np.float32)
    sigma = stack.sigma_over_mean(sigma_stride).astype(np.float32)
    if ncpu is None:
        ncpu = os.cpu_count()
    if ncpu <= 1:
        _preview_init(image, sigma, cmap, dpi)
        for task in tasks:
            _render_preview(task)
        return results
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'preview_images.npz')
        np.savez(path, image=image, sigma=sigma)
        with ProcessPoolExecutor(max_workers=ncpu, initializer=_preview_init_from_file,
                                 initargs=(path, cmap, dpi)) as ex:
            list(ex.map(_render_preview, tasks, chunksize=max(1, len(tasks) // (4 * ncpu))))
    return results


# ----------------------------------------------------------------------------- GUI helpers

_NON_INTERACTIVE_BACKENDS = {'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'}


def _in_jupyter():
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and hasattr(shell, 'kernel')


def _select_backend(backend):
    """'auto': ipympl in Jupyter, QtAgg elsewhere, unless an interactive backend is already active."""
    if backend in (None, 'keep'):
        return
    current = matplotlib.get_backend()
    lower = current.lower()
    if backend == 'auto':
        if _in_jupyter():
            if lower not in _NON_INTERACTIVE_BACKENDS and 'inline' not in lower:
                return
            backend = 'ipympl'
        else:
            if 'qt' in lower:
                return
            backend = 'QtAgg'
    if backend.lower() == lower:
        return
    try:
        plt.switch_backend(backend)
    except Exception as err:  # keep working with whatever is available
        warnings.warn(f'could not switch matplotlib backend to {backend} ({err}); using {current}')


def _redraw_axes_only(canvas, ax):
    """Redraw one axes (blit where possible, else a coalesced full redraw)."""
    if canvas.supports_blit:
        try:
            ax.figure.draw_artist(ax)
            canvas.blit(ax.bbox)
            return
        except Exception:
            pass
    canvas.draw_idle()


class _QuietTextBox(TextBox):
    """
    TextBox without matplotlib's full-figure redraws.

    matplotlib's TextBox redraws the whole figure synchronously when the mouse enters or leaves
    it, on every keystroke, and on every click anywhere in the figure (via stop_typing). With
    several boxes that made each click cost several full redraws. Here only the box is redrawn,
    through ``redraw_hook(ax)`` if it is set.
    """

    redraw_hook = None

    def _redraw(self):
        if self.redraw_hook is not None:
            self.redraw_hook(self.ax)
        else:
            _redraw_axes_only(self.ax.figure.canvas, self.ax)

    def _rendercursor(self):
        canvas = self.ax.figure.canvas
        try:
            has_renderer = self.ax.figure._get_renderer() is not None
        except AttributeError:
            has_renderer = False
        if not has_renderer:
            return super()._rendercursor()
        canvas.draw = lambda *args, **kwargs: None
        try:
            super()._rendercursor()
        finally:
            del canvas.draw
        self._redraw()

    def stop_typing(self):
        if not self.capturekeystrokes:
            if self.cursor.get_visible():
                self.cursor.set_visible(False)
                self._redraw()
            return
        if getattr(self, '_on_stop_typing', None) is not None:
            self._on_stop_typing()
            self._on_stop_typing = None
        self.capturekeystrokes = False
        self.cursor.set_visible(False)
        self._redraw()
        if self.eventson:
            self._observers.process('submit', self.text)

    def _motion(self, event):
        pass  # no hover colour (it would redraw the whole figure)

    def set_text_silently(self, text):
        """Show `text` without firing callbacks or redrawing (the caller redraws)."""
        self.text_disp.set_text(str(text))


class _QuietButton(Button):
    """Button without hover handling (one hit test per widget on every mouse move, for nothing)."""

    def _motion(self, event):
        pass


def _single_shot(canvas, seconds, func):
    """Call func after `seconds` on the GUI event loop (immediately if the canvas has no timers)."""
    try:
        timer = canvas.new_timer(interval=max(1, int(seconds * 1000)))
        timer.single_shot = True
        timer.add_callback(func)
        timer.start()
        return timer
    except Exception:
        func()
        return None


class _Redrawer:
    """
    Coalesced full redraws: at most one in flight; requests made meanwhile give one more
    redraw after it, so redraws never pile up even when a single draw is slow.
    """

    def __init__(self, canvas, stale_after=1.0):
        self.canvas = canvas
        self.stale_after = stale_after  # give up waiting for a draw that never happened
        self._requested_at = None
        self._trailing = False
        self._timer = None
        canvas.mpl_connect('draw_event', self._on_draw)

    def _on_draw(self, event):
        if self.canvas.is_saving():
            return
        self._requested_at = None
        if self._trailing:
            self._trailing = False
            self._timer = _single_shot(self.canvas, 0.01, self.now)

    def now(self):
        self._requested_at = time.perf_counter()
        self.canvas.draw_idle()

    def request(self):
        if self._requested_at is not None and time.perf_counter() - self._requested_at < self.stale_after:
            self._trailing = True
            return
        self.now()


class _Throttle:
    """Rate-limit `func`, leaving at least as much idle time as the last call took; the last call always runs."""

    def __init__(self, canvas, func, min_interval=0.04):
        self.canvas, self.func, self.min_interval = canvas, func, min_interval
        self._next = 0.0
        self._pending = False
        self._timer = None

    def __call__(self):
        if self._pending:
            return
        wait = self._next - time.perf_counter()
        if wait <= 0:
            self._run()
        else:
            self._pending = True
            self._timer = _single_shot(self.canvas, wait, self._run)

    def _run(self):
        self._pending = False
        t0 = time.perf_counter()
        self.func()
        elapsed = time.perf_counter() - t0
        self._next = time.perf_counter() + max(self.min_interval, elapsed)


class _Layers:
    """
    Blitting in three layers, so that most interactions avoid a full figure redraw.

    static   what matplotlib draws normally (sigma/mu panel, control panel)
    dynamic  artists that change often (image and space-time panels, slider rows, status line)
    overlay  artists that follow the mouse (slit points, fit stroke, time line)

    Dynamic and overlay artists are 'animated': matplotlib skips them in normal draws, and they
    are drawn here after every full draw, on top of cached backgrounds. When the figure is
    saved they are drawn into the saved image. Canvases that cannot blit (ipympl) keep normal
    artists and get coalesced full redraws instead.
    """

    def __init__(self, canvas, redrawer):
        self.canvas = canvas
        self.redrawer = redrawer
        self.enabled = bool(canvas.supports_blit)
        self.dynamic = []
        self.overlay = []
        self._bg_static = None
        self._bg_dynamic = None
        if self.enabled:
            canvas.mpl_connect('draw_event', self._on_draw)

    def add_dynamic(self, artist):
        if self.enabled:
            artist.set_animated(True)
        self.dynamic.append(artist)
        return artist

    def add_overlay(self, artist):
        if self.enabled:
            artist.set_animated(True)
        self.overlay.append(artist)
        return artist

    def discard(self, artist):
        for group in (self.dynamic, self.overlay):
            if artist in group:
                group.remove(artist)

    def _draw(self, artists, renderer=None):
        renderer = renderer or self.canvas.get_renderer()
        for artist in artists:
            if artist.figure is not None and artist.get_visible():
                artist.draw(renderer)

    def _on_draw(self, event):
        if self.canvas.is_saving():
            # Axes.draw includes animated children while saving, so overlays come along.
            self._draw(self.dynamic, event.renderer)
            return
        bbox = self.canvas.figure.bbox
        self._bg_static = self.canvas.copy_from_bbox(bbox)
        self._draw(self.dynamic)
        self._bg_dynamic = self.canvas.copy_from_bbox(bbox)
        self._draw(self.overlay)

    def _ready(self):
        return self.enabled and self._bg_static is not None and self._bg_dynamic is not None

    def overlay_changed(self, axes=None):
        """Show moved overlay artists (blit only the given axes, or the whole figure)."""
        if not self._ready():
            self.redrawer.request()
            return
        self.canvas.restore_region(self._bg_dynamic)
        self._draw(self.overlay)
        for bbox in ([ax.bbox for ax in axes] if axes else [self.canvas.figure.bbox]):
            self.canvas.blit(bbox)

    def dynamic_changed(self):
        """Redraw the dynamic layer (and the overlay on top of it)."""
        if not self._ready():
            self.redrawer.now()
            return
        bbox = self.canvas.figure.bbox
        self.canvas.restore_region(self._bg_static)
        self._draw(self.dynamic)
        self._bg_dynamic = self.canvas.copy_from_bbox(bbox)
        self._draw(self.overlay)
        self.canvas.blit(bbox)

    def static_changed(self, ax):
        """Redraw one static axes that does not overlap the dynamic layer (e.g. a widget)."""
        if not self._ready() or ax.get_animated():
            if self._ready():
                self.dynamic_changed()
            else:
                self.redrawer.now()
            return
        bbox = self.canvas.figure.bbox
        self.canvas.restore_region(self._bg_static)
        self._draw([ax])
        self._bg_static = self.canvas.copy_from_bbox(bbox)
        self.canvas.restore_region(self._bg_dynamic)
        self._draw([ax])
        self._bg_dynamic = self.canvas.copy_from_bbox(bbox)
        self._draw(self.overlay)
        self.canvas.blit(ax.bbox)


def _bin_image(image, b):
    """Block-mean binning by an integer factor b (NaN-aware); the edges that do not fit are dropped."""
    if b <= 1:
        return image
    ny, nx = (image.shape[0] // b) * b, (image.shape[1] // b) * b
    blocks = np.asarray(image[:ny, :nx], dtype=float).reshape(ny // b, b, nx // b, b)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmean(blocks, axis=(1, 3))


# ----------------------------------------------------------------------------- GUI

class SlitPickPlus:
    """
    Interactive slit picker for (y, x, t) data, for ipympl (Jupyter) and Qt.

    Usage
    -----
    >>> pick = SlitPickPlus(map_sequence)                # or ndarray (ny, nx, nt)
    >>> pick(bottom_left=[100, 200]*u.pix, top_right=[500, 600]*u.pix)
    >>> pick.slit, pick.fits                             # results, also after closing

    Mouse
    -----
    - Slit "Start", then on either image: left click adds a point, dragging a point moves it
      (left or middle button), right click deletes the nearest point. "End" makes the slit.
    - Fit "Start", then on the space-time diagram: click or drag along a feature. "End" fits
      a polynomial (order box); "Relocate" snaps the points to the ridge first.
    - Clicks are ignored while the pan or zoom tool is active.

    Extensions
    ----------
    ``add_action(label, func)`` adds a panel button that calls ``func(picker)``. Callbacks in
    ``on_slit`` receive each new SlitResult. ``make_slit`` makes a slit without the mouse.
    """

    POINT_PICK_RADIUS = 8      # display pixels
    MIN_CANVAS_SIZE = (640, 520)  # Qt: logical pixels below which the panel would not fit

    def __init__(self, data, cache=False):
        self.source = data if isinstance(data, ImageStack) else ImageStack(data, cache=cache)
        self.actions = []
        self.on_slit = []
        self.stack = None
        self.slit = None
        self.fits = []
        self.points = []
        self.fig = None

    # ------------------------------------------------------------------ setup
    def __call__(self, bottom_left=None, top_right=None, wcs_index=0, wcs_shift=None, norm=None,
                 line_width=5, img_wow=False, method='auto', spacing=1.0, reducer='mean',
                 init_gui=True, backend='auto', save_dir=None, sigma_stride=None, display_bin=1,
                 cmap='magma', relocate_window=1):
        """
        Prepare the data and (by default) open the GUI.

        Parameters
        ----------
        bottom_left, top_right : corners of a crop (pixels, ``u.pix`` Quantities, or SkyCoord for maps)
        wcs_index : int
            Reference frame for world coordinates; also the frame shown first.
        wcs_shift : Quantity (dx, dy), optional
            Shift of the reference coordinate (plot axes and slit world coordinates only).
        norm : ImageNormalize, optional
            Initial norm of the image panel (default: asinh with the frame's min/max).
        line_width, method, spacing, reducer : slit settings (see SlitGeometry, extract_slits)
        img_wow : bool
            Apply watroo.wow to a copy of every frame.
        backend : 'auto', None or a matplotlib backend name
            'auto' uses ipympl in Jupyter and QtAgg otherwise, unless an interactive backend
            is already active. None leaves the backend alone.
        save_dir : str, optional
            Directory for "Save" (default: current directory).
        sigma_stride : int, optional
            Use every n-th frame for the sigma/mean panel (default: about 100 frames).
        display_bin : int
            Show the images binned by this factor (faster redraws for large frames). Picking
            and extraction always use full resolution.
        relocate_window : int
            Half window (rows) for snapping fit points to the ridge.
        """
        if (bottom_left is None) != (top_right is None):
            raise ValueError('give both bottom_left and top_right, or neither')
        stack = self.source
        if bottom_left is not None:
            stack = stack.crop(bottom_left, top_right)
        if img_wow:
            stack = stack.wow()
        self.stack = stack
        self.backend = backend
        self.bottom_left, self.top_right = bottom_left, top_right
        self.wcs_index = int(np.clip(wcs_index, 0, stack.nt - 1))
        self.wcs_shift = wcs_shift
        self.wcs = stack.wcs(self.wcs_index, wcs_shift)
        self.line_width = int(line_width)
        self.method, self.spacing, self.reducer = method, spacing, reducer
        self.save_dir = save_dir or os.getcwd()
        self.sigma_stride = sigma_stride
        self.display_bin = max(1, int(display_bin))
        self.cmap = cmap
        self.relocate_window = int(relocate_window)
        self.frame_index = self.wcs_index
        self.norm = norm
        self.mode = 'idle'
        self.points, self.slit, self.fits = [], None, []
        self.bg_on = False
        self.bg_methods = [m for m in BACKGROUND_METHODS if m != 'none']
        self.bg_method = 'median'
        self.fit_order = 2
        if init_gui:
            self.close()
            self._init_gui()

    def add_action(self, label, func):
        """Add a panel button calling ``func(picker)``. Takes effect when the GUI is (re)opened."""
        self.actions.append((label, func))
        if self.fig is not None:
            warnings.warn('the new button appears the next time the GUI is opened')

    def make_slit(self, x, y, line_width=None, method=None):
        """Make a slit through control points (x, y) in pixels, with or without the GUI."""
        if self.stack is None:
            self(init_gui=False)
        geometry = SlitGeometry(x, y, line_width=line_width or self.line_width,
                                method=method or self.method, spacing=self.spacing)
        result = extract_slits(self.stack, [geometry], self.reducer, self.wcs_index, self.wcs_shift)[0]
        result.meta.update(bottom_left=_json_safe(self.bottom_left), top_right=_json_safe(self.top_right))
        self.points = [(float(a), float(b)) for a, b in zip(geometry.control_x, geometry.control_y)]
        self.slit = result
        self._clear_fits()
        if self.fig is not None:
            self._show_points()
            self._show_slit()
        for callback in self.on_slit:
            callback(result)
        return result

    def save(self, save_dir=None, figure=True, dpi=300):
        """Write slit.h5 (slit + fits) and, with the GUI open, slit_plot.png into save_dir."""
        if self.slit is None:
            raise RuntimeError('no slit to save yet')
        save_dir = save_dir or self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        path = self.slit.save(os.path.join(save_dir, 'slit.h5'), self.fits)
        if figure and self.fig is not None:
            right = self.ax2.get_position().x1 + 0.02
            bbox = Bbox([[0, 0], [right, 1.0]]).transformed(self.fig.transFigure).transformed(
                self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(os.path.join(save_dir, 'slit_plot.png'), dpi=dpi, bbox_inches=bbox)
        print(f'saved {path}')
        return path

    # ------------------------------------------------------------------ figure
    def _figure_size(self):
        aspect = self.stack.nx / self.stack.ny
        width_scale = min(1.4, aspect / 1.5) if aspect > 1.5 else 1.0
        height_scale = min(1.4, 1.5 / aspect) if aspect < 0.67 else 1.0
        size_scale = min(1.2, max(1.0, max(self.stack.nx, self.stack.ny) / 512.0))
        width = float(np.clip(8.0 * width_scale * size_scale, 7.0, 12.0))
        height = float(np.clip(7.0 * height_scale * size_scale, 6.5, 11.0))
        return width, height

    def _display(self, image):
        return _bin_image(image, self.display_bin)

    def _image_extent(self):
        b = self.display_bin
        nx, ny = (self.stack.nx // b) * b, (self.stack.ny // b) * b
        return (-0.5, nx - 0.5, -0.5, ny - 0.5)

    def _init_gui(self):
        _select_backend(self.backend)
        stack = self.stack
        self.fig = plt.figure(figsize=self._figure_size())
        # Redraws are managed here. pyplot's interactive mode (on in IPython and Jupyter) would
        # otherwise redraw the whole figure after every artist change, undoing the blitting.
        self.fig.stale_callback = None
        canvas = self.fig.canvas
        try:
            canvas.manager.set_window_title('Slit Pick Plus')
        except AttributeError:
            pass
        if hasattr(canvas, 'setMinimumSize'):  # Qt: below this the control panel cannot fit
            canvas.setMinimumSize(*self.MIN_CANVAS_SIZE)
        self.redrawer = _Redrawer(canvas)
        self.layers = _Layers(canvas, self.redrawer)

        self.ax1 = self.fig.add_axes([0.1, 0.6, 0.3, 0.3], projection=self.wcs)
        self.ax2 = self.fig.add_axes([0.45, 0.6, 0.3, 0.3], projection=self.wcs)
        self.ax3 = self.fig.add_axes([0.1, 0.1, 0.6, 0.3])
        self.ax2.sharex(self.ax1)
        self.ax2.sharey(self.ax1)

        frame = np.asarray(stack.frame(self.frame_index))
        if self.norm is None:
            finite = frame[np.isfinite(frame)]
            self.norm = ImageNormalize(vmin=finite.min(), vmax=finite.max(), stretch=AsinhStretch(0.5))
        extent = self._image_extent()
        self.im1 = self.ax1.imshow(self._display(frame), origin='lower', cmap=self.cmap, norm=self.norm,
                                   extent=extent)
        sigma = stack.sigma_over_mean(self.sigma_stride)
        finite = sigma[np.isfinite(sigma)]
        self.im2 = self.ax2.imshow(self._display(sigma), origin='lower', cmap=self.cmap, extent=extent,
                                   norm=ImageNormalize(vmin=np.percentile(finite, 2),
                                                       vmax=np.percentile(finite, 98),
                                                       stretch=AsinhStretch(0.5)))
        self._setup_image_axes()
        self.ax2.set_title(r'$\sigma/\mu$')
        self._placeholder = None
        self._show_placeholder()

        self._point_artists = [self.layers.add_overlay(ax.plot([], [], 'x', color='white', ms=7, mew=2,
                                                               ls='none')[0])
                               for ax in (self.ax1, self.ax2)]
        self._slit_artists = []
        self.im3 = None
        self._timeline = None
        self._stroke = self.layers.add_overlay(self.ax3.plot([], [], color='white', lw=1, zorder=3)[0])
        self._stroke_x, self._stroke_y = [], []
        self._stroke_active = False
        self._drag_index = None
        self._status = self.layers.add_dynamic(
            self.fig.text(0.01, 0.005, '', fontsize=8, ha='left', va='bottom', color='0.3'))
        self.layers.add_dynamic(self.ax1)
        self.layers.add_dynamic(self.ax3)

        self._build_panel()
        self._layout()
        canvas.mpl_connect('resize_event', self._on_resize)
        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_move)
        canvas.mpl_connect('button_release_event', self._on_release)
        self._set_frame_title()
        self._set_status('Slit: Start, then click points on an image; End makes the slit.')
        if self.points:
            self._show_points()
        if self.slit is not None:
            self._show_slit()
        if matplotlib.get_backend().lower() not in _NON_INTERACTIVE_BACKENDS:
            plt.show()

    def _setup_image_axes(self):
        """Axis labels and ticks of the two image panels (the sigma/mean panel shares y)."""
        if self.stack.is_map:
            for ax in (self.ax1, self.ax2):
                lon, lat = ax.coords[0], ax.coords[1]
                lon.set_ticks(number=4)
                lat.set_ticks(number=4)
                lon.set_ticklabel_position('b')
                lon.set_axislabel_position('b')
                lat.set_ticklabel_position('l')
                lat.set_axislabel_position('l')
                lon.set_ticklabel(exclude_overlapping=True)
                lat.set_ticklabel(exclude_overlapping=True)
            self._axis_labels = {self.ax1: ('Solar-X [arcsec]', 'Solar-Y [arcsec]'),
                                 self.ax2: ('Solar-X [arcsec]', '')}
            self.ax2.coords[1].set_ticklabel_visible(False)
        else:
            for ax in (self.ax1, self.ax2):
                ax.set_aspect('equal')
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_major_locator(MaxNLocator(5))
            self.ax2.tick_params(labelleft=False)
            self._axis_labels = {self.ax1: ('Pixel-X', 'Pixel-Y'), self.ax2: ('Pixel-X', '')}

    def _button(self, ax, label, callback):
        button = _QuietButton(ax, label, color='0.88', hovercolor='0.88')
        button.on_clicked(lambda event: self._guard(callback))
        return button

    def _text_box(self, ax, initial, callback, align='center'):
        box = _QuietTextBox(ax, None, initial=str(initial), textalignment=align, hovercolor='.95')
        box.redraw_hook = self.layers.static_changed
        box.on_submit(lambda text: self._guard(callback, text))
        return box

    def _check(self, ax, label, callback):
        ax.axis('off')
        check = CheckButtons(ax, [label], [False], useblit=False, frame_props=dict(sizes=(50,)))
        check.drawon = False
        check.on_clicked(lambda lab: self._guard(callback))
        return check

    def _guard(self, callback, *args):
        """Run a GUI callback; report errors instead of breaking the event loop."""
        try:
            callback(*args)
        except Exception as err:
            self._set_status(f'Error: {err}')
            self.layers.dynamic_changed()
            warnings.warn(f'{type(err).__name__}: {err}')

    def _build_panel(self):
        """Create the control panel; positions and font sizes are set by _layout."""
        fig = self.fig
        self._rows = []

        def row(fractions, title=None, dynamic_title=False):
            cells, left = [], 0.0
            for frac in fractions:
                cells.append((fig.add_axes([0, 0, 0.01, 0.01]), left, frac))
                left += frac
            text = None
            if title is not None:
                text = fig.text(0, 0, title, ha='center', va='top', fontsize=9)
                if dynamic_title:
                    self.layers.add_dynamic(text)
            self._rows.append(dict(title=text, cells=cells))
            return [c[0] for c in cells] + ([text] if dynamic_title else [])

        a_prev, a_frame, a_next = row([0.2, 0.6, 0.2], 'Frame index')
        self.button_prev = self._button(a_prev, '<', lambda: self.set_frame(self.frame_index - 1))
        self.button_next = self._button(a_next, '>', lambda: self.set_frame(self.frame_index + 1))
        self.text_frame = self._text_box(a_frame, self.frame_index, self._on_frame_text)
        if self.stack.is_map:
            (a_time,) = row([1], 'Time (UTC)')
            self.text_time = self._text_box(a_time, self._time_string(self.frame_index), self._on_time_text)
        else:
            self.text_time = None
        (a_lw,) = row([1], 'Line width [pix]')
        self.text_lw = self._text_box(a_lw, self.line_width, self._on_line_width_text)

        a_start, a_end = row([0.5, 0.5], 'Slit')
        self.button_start = self._button(a_start, 'Start', self._start_selection)
        self.button_end = self._button(a_end, 'End', self._end_selection)
        (a_clean,) = row([1])
        self.button_clean = self._button(a_clean, 'Clean', self.clean)

        a_asinh, self._title_asinh = row([1], '', dynamic_title=True)
        self.slider_asinh = Slider(a_asinh, None, 0.05, 1.0, valinit=0.5, valstep=0.05)
        a_range, self._title_range = row([1], '', dynamic_title=True)
        vmin, vmax = (float(v) for v in self.im1.get_clim())
        lo = min(0.0, vmin) if vmin >= 0 else vmin - (vmax - vmin)
        hi = vmax + (vmax - lo)
        self.slider_range = RangeSlider(a_range, None, lo, hi, valinit=(vmin, vmax))
        self._norm_throttle = _Throttle(fig.canvas, self._apply_norm)
        for slider in (self.slider_asinh, self.slider_range):
            slider.drawon = False
            slider.valtext.set_visible(False)  # the value is shown in the row title
            self.layers.add_dynamic(slider.ax)
            slider.on_changed(lambda val: self._guard(self._norm_throttle))
        self._update_slider_titles()

        a_bg, a_bg_method = row([0.5, 0.5], 'Background')
        self.check_bg = self._check(a_bg, 'Rm BG', self._toggle_background)
        self.button_bg_method = self._button(a_bg_method, self.bg_method, self._cycle_background)

        a_order, a_reloc = row([0.35, 0.65], 'Spacetime fit: order, relocate')
        self.text_order = self._text_box(a_order, self.fit_order, self._on_order_text)
        self.check_reloc = self._check(a_reloc, 'Relocate',
                                       lambda: self.layers.static_changed(self.check_reloc.ax))
        a_fit_start, a_fit_end = row([0.5, 0.5])
        self.button_fit_start = self._button(a_fit_start, 'Start', self._start_fit)
        self.button_fit_end = self._button(a_fit_end, 'End', self._end_fit)
        a_fit_delete, a_fit_clean = row([0.5, 0.5])
        self.button_fit_delete = self._button(a_fit_delete, 'Delete', self._delete_fit)
        self.button_fit_clean = self._button(a_fit_clean, 'Clean', self._clean_fits_gui)

        self.action_buttons = []
        for i in range(0, len(self.actions), 2):
            pair = self.actions[i:i + 2]
            axes = row([1.0 / len(pair)] * len(pair), 'Actions' if i == 0 else None)
            for ax, (label, func) in zip(axes, pair):
                self.action_buttons.append(self._button(ax, label, lambda f=func: self._run_action(f)))

        (a_dir,) = row([1], 'Save directory')
        self.text_dir = self._text_box(a_dir, self.save_dir, lambda text: None, align='left')
        a_save, a_close = row([0.5, 0.5])
        self.button_save = self._button(a_save, 'Save', self._save_gui)
        self.button_close = self._button(a_close, 'Close', self.close)

    def _on_resize(self, event):
        self._layout()

    def _layout(self):
        """Place everything in physical units and scale fonts with the window size."""
        fig = self.fig
        W, H = fig.get_size_inches()
        s = float(np.clip(min(W / 8.0, H / 7.0), 0.55, 1.0))
        self._font_scale = s

        # control panel on the right
        pw = float(np.clip(0.25 * W, 1.9, 2.8))
        px0 = W - pw - 0.1
        top, bottom = H - 0.08, 0.06
        step = (top - bottom) / len(self._rows)
        title_h = (9 * s + 4) / 72
        widget_h = max(0.08, min(0.3 * s + 0.04, step - title_h - 0.04))
        gap = 0.06
        for i, row in enumerate(self._rows):
            row_top = top - i * step
            y0 = row_top - title_h - widget_h
            usable = pw - gap * (len(row['cells']) - 1)
            for k, (ax, left, frac) in enumerate(row['cells']):
                ax.set_position([(px0 + left * usable + k * gap) / W, y0 / H, frac * usable / W, widget_h / H])
            if row['title'] is not None:
                row['title'].set_position(((px0 + pw / 2) / W, row_top / H))
                row['title'].set_fontsize(9 * s)

        # plots on the left: two image panels above the space-time diagram
        left, right = 0.95 * s + 0.15, px0 - 0.3 * s
        bottom_plots, top_plots = 0.85 * s + 0.05, H - 0.35 * s
        mid_gap, col_gap = 0.8 * s, 0.25 * s
        h = (top_plots - bottom_plots - mid_gap) / 2
        w = (right - left - col_gap) / 2
        y_top = bottom_plots + h + mid_gap
        self.ax1.set_position([left / W, y_top / H, w / W, h / H])
        self.ax2.set_position([(left + w + col_gap) / W, y_top / H, w / W, h / H])
        self.ax3.set_position([left / W, bottom_plots / H, (right - left) / W, h / H])
        self._status.set_position((0.1 / W, 0.04 / H))
        self._apply_fonts()

    def _apply_fonts(self):
        s = self._font_scale
        fs_label, fs_tick, fs_panel = 9 * s, 8 * s, 9 * s
        for ax in (self.ax1, self.ax2):
            ax.title.set_fontsize(10 * s)
            xlabel, ylabel = self._axis_labels[ax]
            if self.stack.is_map:
                ax.coords[0].set_axislabel(xlabel, fontsize=fs_label)
                ax.coords[1].set_axislabel(ylabel, fontsize=fs_label)
                for coord in ax.coords:
                    coord.set_ticklabel(size=fs_tick)
            else:
                ax.set_xlabel(xlabel, fontsize=fs_label)
                ax.set_ylabel(ylabel, fontsize=fs_label)
                ax.tick_params(labelsize=fs_tick)
        self.ax3.tick_params(labelsize=fs_tick)
        self.ax3.xaxis.label.set_fontsize(fs_label)
        self.ax3.yaxis.label.set_fontsize(fs_label)
        if self._placeholder is not None:
            self._placeholder.set_fontsize(fs_label)
        self._status.set_fontsize(8 * s)
        for button in [self.button_prev, self.button_next, self.button_start, self.button_end,
                       self.button_clean, self.button_fit_start, self.button_fit_end,
                       self.button_fit_delete, self.button_fit_clean, self.button_save,
                       self.button_close] + self.action_buttons:
            button.label.set_fontsize(fs_panel)
        self.button_bg_method.label.set_fontsize(7.5 * s)
        for box in [self.text_frame, self.text_time, self.text_lw, self.text_order]:
            if box is not None:
                box.text_disp.set_fontsize(fs_panel)
        self.text_dir.text_disp.set_fontsize(7 * s)
        for check in (self.check_bg, self.check_reloc):
            for label in check.labels:
                label.set_fontsize(fs_panel)
            for name in ('_frames', '_checks'):
                collection = getattr(check, name, None)
                if collection is not None:
                    collection.set_sizes([50 * s * s])

    # ------------------------------------------------------------------ helpers
    def _set_status(self, text):
        if self.fig is not None:
            self._status.set_text(text)

    def _set_frame_title(self):
        self.ax1.set_title(f'Image, frame {self.frame_index}', fontsize=10 * self._font_scale)

    def _update_slider_titles(self):
        def fmt(v):
            return f'{v:.0f}' if abs(v) >= 100 else f'{v:.3g}'
        vmin, vmax = self.slider_range.val
        self._title_asinh.set_text(f'Asinh a = {self.slider_asinh.val:.2f}')
        self._title_range.set_text(f'Vmin / Vmax = {fmt(vmin)} / {fmt(vmax)}')

    def _time_string(self, index):
        return self.stack.times[index].iso[:-4] if self.stack.is_map else str(index)

    def _toolbar_busy(self):
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        return toolbar is not None and getattr(toolbar, 'mode', '') not in ('', None)

    def _nearest_point(self, event):
        if not self.points:
            return None
        xy = event.inaxes.transData.transform(np.asarray(self.points))
        d = np.hypot(xy[:, 0] - event.x, xy[:, 1] - event.y)
        i = int(np.argmin(d))
        return i if d[i] <= self.POINT_PICK_RADIUS else None

    def _show_points(self):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        for artist in self._point_artists:
            artist.set_data(xs, ys)

    def _remove_slit_artists(self):
        for artist in self._slit_artists:
            artist.remove()
        self._slit_artists = []

    def _spacetime_data(self):
        if self.slit is None:
            return None, None
        if self.bg_on:
            data = remove_background(self.slit.intensity, self.bg_method)[0]
            finite = data[np.isfinite(data)]
            norm = (ImageNormalize(finite, interval=ZScaleInterval(), stretch=AsinhStretch(0.5))
                    if finite.size else None)
            return data, norm
        return self.slit.intensity, self.norm

    def _show_placeholder(self):
        self._placeholder = self.ax3.text(0.5, 0.5, 'Slit: Start, click points on an image, End',
                                          transform=self.ax3.transAxes, ha='center', va='center',
                                          color='0.4')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.ax3.set_xlabel('')
        self.ax3.set_ylabel('')

    def _remove_spacetime(self):
        if self.im3 is not None:
            self.im3.remove()
        if self._timeline is not None:
            self.layers.discard(self._timeline)
            self._timeline.remove()
        self.im3, self._timeline = None, None

    def _show_slit(self):
        self._remove_slit_artists()
        for ax in (self.ax1, self.ax2):
            artists = plot_slit_position(ax, self.slit.geometry, show_direction=True)
            self._slit_artists += [a for a in artists.values() if a is not None]
        if self._placeholder is not None:
            self._placeholder.remove()
            self._placeholder = None
        self._remove_spacetime()
        data, norm = self._spacetime_data()
        self.im3 = plot_spacetime(self.ax3, self.slit, data=data, norm=norm, cmap=self.cmap)
        self.im3.format_cursor_data = lambda value: ''
        self._timeline = self.layers.add_overlay(
            self.ax3.axvline(self.frame_index, color='white', lw=1, alpha=0.5, ls=':'))
        self.ax3.set_xlim(-0.5, self.slit.n_time - 0.5)
        self.ax3.set_ylim(-0.5, self.slit.n_dist - 0.5)
        self._apply_fonts()
        self._reset_home_view(self.ax3)
        g = self.slit.geometry
        self._set_status(f'Slit: {g.method}, {len(g.control_x)} points, {g.length:.1f} px, '
                         f'{self.slit.n_dist} x {self.slit.n_time}. Fit: Start, draw on the diagram, End.')
        self.redrawer.now()  # the outline on the sigma/mean panel is in the static layer

    def _reset_home_view(self, ax):
        """Make the toolbar's Home button restore `ax` to its current (full) view."""
        toolbar = getattr(self.fig.canvas, 'toolbar', None)
        stack = getattr(toolbar, '_nav_stack', None)
        try:
            if stack is None or stack() is None:
                return  # the first pan/zoom will record the current views as home
            home = stack._elements[0]
            home[ax] = (ax._get_view(), (ax.get_position(True).frozen(), ax.get_position().frozen()))
        except Exception:
            pass

    def _run_action(self, func):
        func(self)
        self.redrawer.now()

    # ------------------------------------------------------------------ frame and display
    def set_frame(self, index):
        """Show frame `index` (clipped to the valid range)."""
        index = int(np.clip(index, 0, self.stack.nt - 1))
        self.frame_index = index
        if self.fig is None:
            return
        self.im1.set_data(self._display(np.asarray(self.stack.frame(index))))
        self._set_frame_title()
        self.text_frame.set_text_silently(index)
        if self._timeline is not None:
            self._timeline.set_xdata([index, index])
        self.layers.dynamic_changed()
        self.layers.static_changed(self.text_frame.ax)
        if self.text_time is not None:
            self.text_time.set_text_silently(self._time_string(index))
            self.layers.static_changed(self.text_time.ax)

    def _on_frame_text(self, text):
        try:
            index = int(float(text))
        except ValueError:
            self._set_status(f'Frame index must be an integer, got {text!r}')
            self.text_frame.set_text_silently(self.frame_index)
            self.layers.static_changed(self.text_frame.ax)
            self.layers.dynamic_changed()
            return
        self.set_frame(index)

    def _on_time_text(self, text):
        try:
            target = Time(text.strip())
        except ValueError:
            self._set_status(f'Could not read time {text!r}')
            self.text_time.set_text_silently(self._time_string(self.frame_index))
            self.layers.static_changed(self.text_time.ax)
            self.layers.dynamic_changed()
            return
        self.set_frame(int(np.argmin(np.abs((self.stack.times - target).to_value(u.s)))))

    def _on_line_width_text(self, text):
        try:
            value = int(text)
            if value < 1:
                raise ValueError
        except ValueError:
            self._set_status(f'Line width must be a positive integer, got {text!r}')
            self.text_lw.set_text_silently(self.line_width)
            self.layers.static_changed(self.text_lw.ax)
            self.layers.dynamic_changed()
            return
        self.line_width = value

    def _apply_norm(self):
        vmin, vmax = self.slider_range.val
        self.norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch(self.slider_asinh.val))
        self.im1.set_norm(self.norm)
        if self.im3 is not None and not self.bg_on:
            self.im3.set_norm(self.norm)
        self._update_slider_titles()
        self.layers.dynamic_changed()

    def _toggle_background(self):
        self.bg_on = self.check_bg.get_status()[0]
        self.layers.static_changed(self.check_bg.ax)
        self._refresh_spacetime()

    def _cycle_background(self):
        i = (self.bg_methods.index(self.bg_method) + 1) % len(self.bg_methods)
        self.bg_method = self.bg_methods[i]
        self.button_bg_method.label.set_text(self.bg_method)
        self.layers.static_changed(self.button_bg_method.ax)
        if self.bg_on:
            self._refresh_spacetime()

    def _refresh_spacetime(self):
        if self.im3 is not None:
            data, norm = self._spacetime_data()
            self.im3.set_data(data)
            self.im3.set_norm(norm)
        self.layers.dynamic_changed()

    # ------------------------------------------------------------------ slit picking
    def _start_selection(self):
        self.mode = 'select'
        self._set_status('Select: left click adds a point, drag moves it, right click deletes; End makes the slit.')
        self.layers.dynamic_changed()

    def _end_selection(self):
        if len(self.points) < 2:
            self._set_status('Need at least two points for a slit.')
            self.layers.dynamic_changed()
            return
        self.mode = 'idle'
        xs, ys = zip(*self.points)
        self.make_slit(xs, ys)

    def clean(self):
        """Remove the points, the slit and its fits."""
        self.points, self.slit = [], None
        self.mode = 'idle'
        if self.fig is None:
            self.fits = []
            return
        self._clear_fits()
        self._show_points()
        self._remove_slit_artists()
        self._remove_spacetime()
        if self._placeholder is None:
            self._show_placeholder()
            self._apply_fonts()
        self._stroke_x, self._stroke_y = [], []
        self._stroke.set_data([], [])
        self._set_status('Cleaned. Slit: Start to pick a new slit.')
        self.redrawer.now()

    # ------------------------------------------------------------------ spacetime fits
    def _start_fit(self):
        if self.slit is None:
            self._set_status('Make a slit first.')
            self.layers.dynamic_changed()
            return
        self.mode = 'fit'
        self._stroke_x, self._stroke_y = [], []
        self._stroke.set_data([], [])
        self._set_status('Fit: click or drag along a feature on the diagram; End fits it.')
        self.layers.dynamic_changed()

    def _end_fit(self):
        if self.mode != 'fit':
            self._set_status('Fit: press Start first.')
            self.layers.dynamic_changed()
            return
        self.mode = 'idle'
        if len(self._stroke_x) < self.fit_order + 1:
            self._set_status(f'Need at least {self.fit_order + 1} points for order {self.fit_order}.')
            self.layers.dynamic_changed()
            return
        data = self._spacetime_data()[0]
        fit = SpacetimeFit.from_points(self.slit, self._stroke_x, self._stroke_y, self.fit_order,
                                       relocate=self.check_reloc.get_status()[0], image=data,
                                       window_half_size=self.relocate_window)
        xc, yc = fit.curve()
        fit.artist = self.ax3.plot(xc, yc, color='#81C7D4', lw=1.2, zorder=3)[0]
        self.fits.append(fit)
        self._stroke_x, self._stroke_y = [], []
        self._stroke.set_data([], [])
        t_unit, d_unit = fit.world_units
        speed = fit.velocity()
        print(f'fit {len(self.fits)}: d(t) coefficients [{d_unit}, {t_unit}], highest order first: '
              f'{np.array2string(fit.params_world, precision=5)}')
        self._set_status(f'Fit {len(self.fits)}: mean speed {np.mean(speed):.3g} {d_unit}/{t_unit}')
        self.layers.dynamic_changed()

    def _delete_fit(self):
        if self._stroke_x:
            self._stroke_x, self._stroke_y = [], []
            self._stroke.set_data([], [])
        elif self.fits:
            fit = self.fits.pop()
            if fit.artist is not None:
                fit.artist.remove()
        self.layers.dynamic_changed()

    def _clear_fits(self):
        for fit in self.fits:
            if fit.artist is not None and fit.artist.axes is not None:
                fit.artist.remove()
            fit.artist = None
        self.fits = []

    def _clean_fits_gui(self):
        self._clear_fits()
        self._stroke_x, self._stroke_y = [], []
        self._stroke.set_data([], [])
        self.layers.dynamic_changed()

    def _on_order_text(self, text):
        try:
            order = int(text)
            if order < 0:
                raise ValueError
        except ValueError:
            self._set_status(f'Order must be a non-negative integer, got {text!r}')
            self.text_order.set_text_silently(self.fit_order)
            self.layers.static_changed(self.text_order.ax)
            self.layers.dynamic_changed()
            return
        self.fit_order = order

    # ------------------------------------------------------------------ saving
    def _save_gui(self):
        save_dir = self.text_dir.text.strip() or self.save_dir
        if 'qt' in matplotlib.get_backend().lower():
            try:
                from PyQt5.QtWidgets import QFileDialog
                chosen = QFileDialog.getExistingDirectory(None, 'Select directory', save_dir)
                if not chosen:
                    self._set_status('Save cancelled.')
                    self.layers.dynamic_changed()
                    return
                save_dir = chosen
                self.text_dir.set_text_silently(save_dir)
                self.layers.static_changed(self.text_dir.ax)
            except ImportError:
                pass
        self.save_dir = save_dir
        path = self.save(save_dir)
        self._set_status(f'Saved {path}')
        self.layers.dynamic_changed()

    def redraw(self):
        """Redraw the GUI, e.g. after adding your own artists to pick.ax1/ax2/ax3."""
        if self.fig is not None:
            self.redrawer.now()

    def close(self):
        """Close the GUI window; results stay in .slit and .fits."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    # ------------------------------------------------------------------ mouse
    def _on_press(self, event):
        if event.inaxes is None or self._toolbar_busy():
            return
        if self.mode == 'select' and event.inaxes in (self.ax1, self.ax2) and event.xdata is not None:
            index = self._nearest_point(event)
            if event.button == 3:
                if index is not None:
                    self.points.pop(index)
                    self._show_points()
                    self.layers.overlay_changed([self.ax1, self.ax2])
            elif event.button in (1, 2) and index is not None:
                self._drag_index = index
            elif event.button == 1:
                self.points.append((float(event.xdata), float(event.ydata)))
                self._show_points()
                self.layers.overlay_changed([self.ax1, self.ax2])
        elif self.mode == 'fit' and event.inaxes is self.ax3 and event.button == 1 \
                and event.xdata is not None:
            self._stroke_active = True
            self._stroke_x.append(float(event.xdata))
            self._stroke_y.append(float(event.ydata))
            self._stroke.set_data(self._stroke_x, self._stroke_y)
            self.layers.overlay_changed([self.ax3])

    def _on_move(self, event):
        if self._drag_index is not None:
            if event.inaxes in (self.ax1, self.ax2) and event.xdata is not None:
                self.points[self._drag_index] = (float(event.xdata), float(event.ydata))
                self._show_points()
                self.layers.overlay_changed([self.ax1, self.ax2])
        elif self._stroke_active:
            if event.inaxes is self.ax3 and event.xdata is not None:
                self._stroke_x.append(float(event.xdata))
                self._stroke_y.append(float(event.ydata))
                self._stroke.set_data(self._stroke_x, self._stroke_y)
                self.layers.overlay_changed([self.ax3])

    def _on_release(self, event):
        if self._drag_index is not None:
            self._drag_index = None
            self.layers.overlay_changed([self.ax1, self.ax2])
        elif self._stroke_active:
            self._stroke_active = False
            self.layers.overlay_changed([self.ax3])


def _main():
    import argparse
    from glob import glob
    parser = argparse.ArgumentParser(description='Interactive space-time slits from FITS images.')
    parser.add_argument('files', nargs='+', help='FITS files or glob patterns')
    parser.add_argument('--bottom-left', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--top-right', nargs=2, type=float, metavar=('X', 'Y'))
    parser.add_argument('--line-width', type=int, default=5)
    parser.add_argument('--display-bin', type=int, default=1)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()
    files = sorted(f for pattern in args.files for f in glob(pattern))
    pick = SlitPickPlus(sunpy.map.Map(files, sequence=True), cache=args.cache)
    crop = {}
    if args.bottom_left and args.top_right:
        crop = dict(bottom_left=args.bottom_left * u.pix, top_right=args.top_right * u.pix)
    pick(line_width=args.line_width, display_bin=args.display_bin, **crop)


if __name__ == '__main__':
    _main()
