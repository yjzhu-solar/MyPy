"""
Coalign a sequence of solar images (e.g. Solar Orbiter EUI/HRIEUV) that suffer
from pointing jitter, following the recipe of Chitta et al. 2022, A&A 667, A166,
Appendix A.1:

* The sequence is divided into *overlapping chunks* (segments) of ``nframes``
  images such that the last image of one chunk is the first image of the next.
* Every image in a chunk is cross-correlated with the first image of that chunk.
  Because the chunks overlap, chaining the chunk-wise shifts aligns the whole
  sequence to a single reference image.
* The procedure is iterated (``iter`` times). This is needed because the
  sub-pixel estimate from a single cross-correlation pass is biased (the
  parabolic peak interpolation is good to only ~0.1 pix); re-measuring on the
  already shifted images drives the residual to ~1e-3 pix.

Sign convention used throughout this module
-------------------------------------------
``xshift``/``yshift`` (in pixels) is the displacement of the solar structures in a
frame *relative to the reference frame*, i.e. a positive ``xshift`` means the
structures appear further to the right (larger column index) than in the
reference. The data are therefore moved by ``(-yshift, -xshift)`` to coalign them.
The per-frame cumulative shift actually applied is stored in the FITS keywords
``XSHIFT`` and ``YSHIFT``. The WCS keywords are deliberately left untouched.

The arrays ``xshifts_pixel``/``yshifts_pixel`` with shape ``(nt, iter)`` hold the
*increment* measured in each iteration; ``xshift_total``/``yshift_total`` hold
their sum over iterations, which is the shift that was applied to the data.

Shifting is done only once, at the end, on the original data with the total
shift (``scipy.ndimage.shift``, cubic spline). Pixels that fall outside the
original frame after the shift are set to NaN. During the iterations only the
cross-correlation window of each frame is shifted, so the full frames are
interpolated a single time and memory stays at one copy of the sequence.

Command line usage::

    python map_coalign.py '/path/to/maps/*.fits' /path/to/output_dir -ref 0 -i 3

The low-level peak-finding helpers at the end of this file were copied from
``sunpy.image.coalignment`` (sunpy 0.5.1, since removed) with the off-by-one
bug in ``_find_best_match_location`` fixed.
"""

import sunpy
import sunpy.map
from sunpy.map import Map, MapSequence, GenericMap
from sunpy.util.exceptions import SunpyUserWarning
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
import matplotlib.patches as patches
import matplotlib.animation
from skimage.feature import match_template
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import astropy.units as u
from astropy.visualization import (ImageNormalize, AsinhStretch)
from astropy.io.fits import CompImageHDU
from tqdm import tqdm
from glob import glob
from copy import deepcopy
import os
import h5py
import argparse


class MapSequenceCoalign(MapSequence):
    """
    A `sunpy.map.MapSequence` with a Chitta et al. (2022) style jitter
    coalignment, a pixel-coordinate animation and an interactive region
    selector.

    All frames must have the same shape and (for the WCS sanity check) the
    same pixel scale and rotation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nt = len(self)

    def __setitem__(self, key, value):
        if not isinstance(value, GenericMap):
            raise TypeError("MapSequenceCoalign items must be sunpy GenericMap objects, "
                            f"got {type(value)}")
        self.maps[key] = value

    # ------------------------------------------------------------------ plotting
    def plot(self, axes=None, resample=None, annotate=True,
             interval=200, plot_function=None, no_wcs=False, **kwargs):
        """
        Animate the sequence.

        With ``no_wcs=True`` the frames are drawn with `matplotlib.axes.Axes.imshow`
        in pixel coordinates (fast, no WCSAxes), optionally ``resample``-d by a
        factor ``[fx, fy]``. Otherwise this falls through to
        `sunpy.map.MapSequence.plot` (``resample`` is not supported there).

        ``norm`` and ``cmap`` keyword arguments override the maps' own plot
        settings for every frame.

        Returns
        -------
        `matplotlib.animation.FuncAnimation`
        """
        if not no_wcs:
            if resample is not None:
                warnings.warn("'resample' is only supported with no_wcs=True and is ignored.",
                              SunpyUserWarning, stacklevel=2)
            # sunpy's signature is (axes, annotate, interval, plot_function, clip_interval),
            # so pass everything by keyword.
            return super().plot(axes=axes, annotate=annotate, interval=interval,
                                plot_function=plot_function, **kwargs)

        if axes is None:
            fig, axes = plt.subplots(layout='constrained')
        else:
            fig = axes.get_figure()

        if not plot_function:
            def plot_function(fig, ax, smap):
                return []
        removes = []

        def annotate_frame(i):
            axes.set_title(f"{self[i].name}")
            axes.set_xlabel('X [pixel]')
            axes.set_ylabel('Y [pixel]')

        if resample:
            if not self.all_same_shape:
                raise ValueError('Maps in mapsequence do not all have the same shape.')
            resample = u.Quantity(self.maps[0].dimensions) * np.array(resample)
            ani_data = [amap.resample(resample) for amap in self.maps]
        else:
            ani_data = self.maps

        # Resolve norm/cmap once; a user supplied value wins over the map's plot_settings
        # for every frame (previously the user cmap was lost after the first frame).
        user_norm = kwargs.pop('norm', None)
        user_cmap = kwargs.pop('cmap', None)

        def frame_norm(i):
            return user_norm if user_norm is not None else ani_data[i].plot_settings.get('norm')

        def frame_cmap(i):
            return user_cmap if user_cmap is not None else ani_data[i].plot_settings.get('cmap')

        im = axes.imshow(ani_data[0].data, origin='lower', norm=frame_norm(0),
                         cmap=frame_cmap(0), **kwargs)

        def updatefig(i, im, annotate, ani_data, removes, update_norm=False):
            while removes:
                removes.pop(0).remove()

            im.set_array(ani_data[i].data)
            im.set_cmap(frame_cmap(i))

            if update_norm:
                norm = deepcopy(frame_norm(i))
                # Explicit autoscale for bugged versions of astropy's ImageNormalize
                norm.autoscale_None(ani_data[i].data)
                im.set_norm(norm)

            if annotate:
                annotate_frame(i)
            removes += list(plot_function(fig, axes, ani_data[i]))

        ani = matplotlib.animation.FuncAnimation(fig, updatefig,
                                                 frames=list(range(0, len(ani_data))),
                                                 fargs=[im, annotate, ani_data, removes],
                                                 interval=interval,
                                                 blit=False)
        return ani

    # --------------------------------------------------------------- coalignment
    def coalign(self, reference_index=0, bottom_left=None, top_right=None, check_header=True,
                nframes=10, iter=3, method='match_template', search_margin=50,
                upsample_factor=100, interp_order=3):
        """
        Measure and remove frame-to-frame jitter (Chitta et al. 2022, App. A.1).

        Parameters
        ----------
        reference_index : int
            Index of the frame that is *not* moved; all other frames are aligned to it.
        bottom_left, top_right : 2-element sequences of int, optional
            ``[x, y]`` pixel corners (in the first frame) of the region used as
            cross-correlation template. Give both or neither. If neither is given
            an interactive window opens to select the region (default: central
            half of the frame).
        check_header : bool
            Check that all frames share shape, CDELT and rotation.
        nframes : int
            Number of frames per chunk. Chunks overlap by one frame.
        iter : int
            Number of measure-and-shift iterations. 3 is usually enough to reach
            ~1e-3 pix.
        method : {'match_template', 'phase'}
            ``'match_template'``: normalised cross-correlation
            (`skimage.feature.match_template`) of the template with the frame,
            restricted to the template box grown by ``search_margin`` pixels,
            with parabolic sub-pixel peak interpolation.
            ``'phase'``: `skimage.registration.phase_cross_correlation` of the
            (mean-subtracted, Hann-windowed) template box in the reference and in
            the target frame, with Fourier upsampling (``upsample_factor``) for
            the sub-pixel estimate. About 10x faster than ``'match_template'``
            for a 1024x1024 box; precision is limited to ``1/upsample_factor``.
        search_margin : int
            Half-width (pixels) of the search area around the template box for
            ``method='match_template'``. Must exceed the largest expected shift.
        upsample_factor : int
            Sub-pixel resolution ``1/upsample_factor`` for ``method='phase'``.
        interp_order : int
            Spline order used by `scipy.ndimage.shift` when the frames are shifted.

        Notes
        -----
        The frames are interpolated once, after the last iteration, with the
        total shift. Pixels moved in from outside the frame are NaN. The
        applied shift is written to the ``XSHIFT``/``YSHIFT`` metadata keys; the
        WCS is not modified.
        """
        if (bottom_left is None) != (top_right is None):
            raise ValueError("Give both 'bottom_left' and 'top_right', or neither.")
        if method not in ('match_template', 'phase'):
            raise ValueError("method must be 'match_template' or 'phase'")
        if search_margin < 2:
            raise ValueError("search_margin must be >= 2 pixels")
        if not 0 <= reference_index < self.nt:
            raise IndexError(f"reference_index {reference_index} out of range for {self.nt} frames")

        if check_header:
            self._check_header()

        self.ny, self.nx = self[0].data.shape

        if bottom_left is None:
            self.bottom_left = [self.nx//4, self.ny//4]
            self.top_right = [3*self.nx//4, 3*self.ny//4]
            self._get_common_map_extent()
        else:
            self.bottom_left = [int(bottom_left[0]), int(bottom_left[1])]
            self.top_right = [int(top_right[0]), int(top_right[1])]

        if not (0 <= self.bottom_left[0] < self.top_right[0] <= self.nx
                and 0 <= self.bottom_left[1] < self.top_right[1] <= self.ny):
            raise ValueError(f"Invalid template region bottom_left={self.bottom_left}, "
                             f"top_right={self.top_right} for a {self.nx}x{self.ny} frame")

        self.reference_index = reference_index
        self.nframes = nframes
        self.iter = iter
        self.method = method
        self.search_margin = int(search_margin)
        self.upsample_factor = int(upsample_factor)
        self.interp_order = int(interp_order)
        self.n_segment = int(np.ceil((self.nt - 1)/nframes))  # overlapping chunks
        self.xshifts_pixel = np.zeros((self.nt, iter))
        self.yshifts_pixel = np.zeros((self.nt, iter))
        self.xshift_total = np.zeros(self.nt)
        self.yshift_total = np.zeros(self.nt)
        self._nan_warned = False

        for ii in range(iter):
            print(f'--------Starting iteration {ii+1}/{iter}--------')
            self._calculate_shifts(ii)
            self.xshift_total += self.xshifts_pixel[:, ii]
            self.yshift_total += self.yshifts_pixel[:, ii]
            print(f'--------Iteration {ii+1}/{iter} finished--------')

        self._apply_shifts()

    def _segment_bounds(self, ii_seg):
        """
        First index and (exclusive) end index of chunk ``ii_seg``.

        Chunk k covers frames ``k*nframes .. (k+1)*nframes`` inclusive, i.e. it
        includes the first frame of the next chunk (the one-frame overlap of
        Chitta et al. 2022) so that the chunks can be chained. The last chunk is
        simply cut at ``nt``.
        """
        start = ii_seg*self.nframes
        end = min(start + self.nframes + 1, self.nt)
        return start, end

    def _window(self, jj, y0, y1, x0, x1):
        """
        Return ``data[y0:y1, x0:x1]`` of frame ``jj`` *after* shifting the frame by
        its current total shift ``(-yshift_total, -xshift_total)``.

        Only a padded cut-out around the window is interpolated, so this is cheap
        and the full frame is never touched before the final `_apply_shifts`.
        Out-of-frame pixels are extrapolated with the nearest edge value so that
        the correlation never sees NaN.
        """
        data = self[jj].data
        dy, dx = self.yshift_total[jj], self.xshift_total[jj]
        pad = int(np.ceil(max(abs(dy), abs(dx)))) + 4   # spline support
        Y0, Y1 = max(y0 - pad, 0), min(y1 + pad, data.shape[0])
        X0, X1 = max(x0 - pad, 0), min(x1 + pad, data.shape[1])
        cut = np.asarray(data[Y0:Y1, X0:X1], dtype=np.float64)
        if not np.all(np.isfinite(cut)):
            if not self._nan_warned:
                warnings.warn("Non-finite pixels inside the correlation window are replaced "
                              "by the window mean for the shift measurement.",
                              SunpyUserWarning, stacklevel=3)
                self._nan_warned = True
            cut = np.where(np.isfinite(cut), cut, np.nanmean(cut))
        if dy != 0 or dx != 0:
            cut = shift(cut, (-dy, -dx), order=self.interp_order, mode='nearest')
        # float32 is enough for the correlation (differences < 1e-4 pix on HRIEUV
        # data) and halves the match_template run time.
        return cut[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0].astype(np.float32)

    def _template(self, jj):
        """Template box ``[bottom_left, top_right)`` of (shifted) frame ``jj``."""
        return self._window(jj, self.bottom_left[1], self.top_right[1],
                            self.bottom_left[0], self.top_right[0])

    def _measure(self, jj, template):
        """
        Position ``(y, x)`` in pixels of ``template`` in the (shifted) frame ``jj``.

        For ``'match_template'`` the frame is searched only within the template
        box grown by ``search_margin``. For ``'phase'`` the template box of the
        frame itself is compared with the template and the returned position is
        ``bottom_left`` plus the measured displacement.
        """
        bl, tr = self.bottom_left, self.top_right
        if self.method == 'match_template':
            m = self.search_margin
            y0, y1 = max(bl[1] - m, 0), min(tr[1] + m, self.ny)
            x0, x1 = max(bl[0] - m, 0), min(tr[0] + m, self.nx)
            layer = self._window(jj, y0, y1, x0, x1)
            yshift, xshift = _calculate_shift(layer, template)
            # float64 before adding the (large) window offset, otherwise float32
            # rounding limits the result to ~1e-4 pix
            return float(yshift.value) + y0, float(xshift.value) + x0
        else:
            moving = self._window(jj, bl[1], tr[1], bl[0], tr[0])
            # The FFT assumes periodic images; the discontinuity at the box edges
            # biases the correlation peak towards zero lag and makes the iteration
            # converge slowly. A mean-subtracted Hann window removes this (tests:
            # 0.04 -> 0.01 pix after 3 iterations). 'phase' normalisation performs
            # poorly on smooth EUV images, so plain cross-correlation is used.
            if getattr(self, '_hann', None) is None or self._hann.shape != moving.shape:
                self._hann = np.outer(np.hanning(moving.shape[0]),
                                      np.hanning(moving.shape[1])).astype(np.float32)
            ref_w = (template - template.mean()) * self._hann
            mov_w = (moving - moving.mean()) * self._hann
            # phase_cross_correlation returns the shift that registers `moving`
            # onto `template`; the structure displacement is minus that.
            reg_shift, _, _ = phase_cross_correlation(ref_w, mov_w,
                                                      upsample_factor=self.upsample_factor,
                                                      normalization=None)
            reg_shift = np.asarray(reg_shift, dtype=np.float64)  # see note above
            return bl[1] - reg_shift[0], bl[0] - reg_shift[1]

    def _calculate_shifts(self, iter_index):
        """
        Measure the shift increment of every frame for iteration ``iter_index``.

        Chunk by chunk, the template is the box of the chunk's first frame and
        every other frame of the chunk (including the first frame of the next
        chunk, because of the overlap) is matched against it. The measured
        displacement is referred to the first frame's own increment (already
        known from the previous chunk), which chains the chunks together.
        Finally all increments are referred to ``reference_index`` so that
        frame stays fixed.
        """
        for ii_seg in tqdm(range(self.n_segment)):
            start_index_, end_index_ = self._segment_bounds(ii_seg)
            template_ = self._template(start_index_)

            # Baseline: position found when matching the template against its own
            # frame. For match_template this is bottom_left plus the (small) bias of
            # the parabolic peak fit on an autocorrelation peak; subtracting it makes
            # an identical frame measure exactly zero shift.
            if self.method == 'match_template':
                y_keep, x_keep = self._measure(start_index_, template_)
            else:
                y_keep, x_keep = self.bottom_left[1], self.bottom_left[0]

            for jj in range(start_index_ + 1, end_index_):
                y, x = self._measure(jj, template_)
                self.yshifts_pixel[jj, iter_index] = (y - y_keep) + self.yshifts_pixel[start_index_, iter_index]
                self.xshifts_pixel[jj, iter_index] = (x - x_keep) + self.xshifts_pixel[start_index_, iter_index]

        # Make the reference frame the one with zero shift
        self.yshifts_pixel[:, iter_index] -= self.yshifts_pixel[self.reference_index, iter_index]
        self.xshifts_pixel[:, iter_index] -= self.xshifts_pixel[self.reference_index, iter_index]

    def _apply_shifts(self):
        """
        Shift every frame once by its total shift and replace the maps.

        Pixels whose source lies outside the original frame are set to NaN; the
        applied shift is recorded in the ``XSHIFT``/``YSHIFT`` metadata keys.
        """
        for ii in range(self.nt):
            data = _shift_with_nan(self[ii].data, -self.yshift_total[ii], -self.xshift_total[ii],
                                   order=self.interp_order)
            meta = deepcopy(self[ii].meta)
            meta['xshift'] = float(self.xshift_total[ii])
            meta['yshift'] = float(self.yshift_total[ii])
            self[ii] = sunpy.map.Map(data, meta)

    # --------------------------------------------------------------- utilities
    def submap(self, *args, **kwargs) -> 'MapSequenceCoalign':
        """Apply `sunpy.map.GenericMap.submap` to every frame. Shift bookkeeping is not carried over."""
        return MapSequenceCoalign([self[ii].submap(*args, **kwargs) for ii in range(self.nt)])

    def save(self, filepath, filetype='auto', compress=False, quantize_level=0, **kwargs):
        """
        Save every frame to FITS and the coalignment bookkeeping to
        ``coalign_info.h5`` in the same directory.

        Parameters
        ----------
        filepath : str
            Template containing ``{index}``, e.g. ``'dir/map_{index:03}.fits'``.
        compress : bool
            Write tile-compressed FITS (`astropy.io.fits.CompImageHDU`).
        quantize_level : float
            ``0`` (default) uses lossless GZIP_1 compression. A positive value
            uses RICE_1 with that quantisation level, which is *lossy* for float
            data (astropy's default of 16 gave errors of a few DN/s on HRIEUV).
        kwargs :
            Passed to `sunpy.map.GenericMap.save` (e.g. ``overwrite=True``).
        """
        filedir = os.path.dirname(filepath)
        if filedir and not os.path.exists(filedir):
            os.makedirs(filedir)

        if compress:
            if filepath.format(index=0) == filepath:
                raise ValueError("'{index}' must appear in the file path")
            if quantize_level == 0:
                hdu_kwargs = dict(compression_type='GZIP_1', quantize_level=0)
            else:
                warnings.warn(f"RICE_1 compression with quantize_level={quantize_level} is lossy "
                              "for floating point data.", SunpyUserWarning, stacklevel=2)
                hdu_kwargs = dict(compression_type='RICE_1', quantize_level=quantize_level)
            for index, amap in enumerate(self.maps):
                # sunpy only forwards kwargs to writeto(); compression settings must
                # be given through a pre-built HDU instance, one per frame.
                amap.save(filepath.format(index=index), filetype,
                          hdu_type=CompImageHDU(**hdu_kwargs), **kwargs)
        else:
            super().save(filepath, filetype, **kwargs)

        if hasattr(self, 'xshifts_pixel'):
            h5path = os.path.join(filedir, "coalign_info.h5")
            if os.path.exists(h5path):
                warnings.warn(f"Overwriting existing {h5path}", SunpyUserWarning, stacklevel=2)
            with h5py.File(h5path, 'w') as hf:
                hf.create_dataset('xshifts_pixel', data=self.xshifts_pixel)
                hf.create_dataset('yshifts_pixel', data=self.yshifts_pixel)
                hf.create_dataset('xshift_total', data=self.xshift_total)
                hf.create_dataset('yshift_total', data=self.yshift_total)
                hf.create_dataset('bottom_left', data=self.bottom_left)
                hf.create_dataset('top_right', data=self.top_right)
                hf.create_dataset('reference_index', data=self.reference_index)
                hf.create_dataset('nframes', data=self.nframes)
                hf.create_dataset('iter', data=self.iter)
                hf.create_dataset('n_segment', data=self.n_segment)
                hf.create_dataset('method', data=self.method)
                hf.create_dataset('search_margin', data=self.search_margin)
                hf.create_dataset('upsample_factor', data=self.upsample_factor)
                hf.attrs['description'] = ("xshifts_pixel/yshifts_pixel: per-iteration increments "
                                           "(nt, iter); x/yshift_total: their sum = structure "
                                           "displacement relative to the reference; data were "
                                           "moved by -shift_total.")

    def _check_header(self, rot_atol=1e-2):
        """
        Check that all frames have the same shape, pixel scale and rotation.

        ``rot_atol`` is the tolerance in degrees for CROTA/CROTA2 and (as a
        dimensionless tolerance) for the PCi_j matrix elements. HRIEUV L2
        headers vary by a few 1e-3 deg within a sequence, which is <0.1 pix at
        the frame corners.
        """
        if not self.all_same_shape:
            raise ValueError("All maps in the sequence must have the same shape")

        metas = [m.meta for m in self.maps]

        def same(key, atol):
            vals = [meta[key] for meta in metas]
            return np.allclose(vals, vals[0], atol=atol)

        for key in ('cdelt1', 'cdelt2'):
            if key in metas[0]:
                if not same(key, atol=0):
                    raise ValueError(f"All maps in the sequence must have the same {key.upper()} value")
            else:
                warnings.warn(f"{key.upper()} not found in header. Assuming equal pixel scales",
                              SunpyUserWarning, stacklevel=3)

        if 'crota' in metas[0]:
            if not same('crota', rot_atol):
                raise ValueError("All maps in the sequence must have the same CROTA value")
        elif 'crota2' in metas[0]:
            if not same('crota2', rot_atol):
                raise ValueError("All maps in the sequence must have the same CROTA2 value")
        elif 'pc1_1' in metas[0]:
            if not all(same(k, rot_atol) for k in ('pc1_1', 'pc1_2', 'pc2_1', 'pc2_2')):
                raise ValueError("All maps in the sequence must have the same PCi_j values")
        else:
            warnings.warn("No rotation information found in header. Assuming no rotation",
                          SunpyUserWarning, stacklevel=3)

    # ------------------------------------------------- interactive region picker
    def _get_common_map_extent(self) -> None:
        """
        Open a window showing the first frame with the template region drawn
        in red. The corners can be typed into the text boxes or, with 'Select'
        ticked, dragged with the mouse. Blocks until the window is closed.
        """
        fig = plt.figure(figsize=(7, 5))
        self.fig = fig
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.9])
        self.ax = ax

        ax.imshow(self[0].data, origin='lower', cmap='gray', norm=ImageNormalize(stretch=AsinhStretch()))

        ax_bottom_left_x = fig.add_axes([0.75, 0.7, 0.1, 0.08])
        ax_bottom_left_y = fig.add_axes([0.87, 0.7, 0.1, 0.08])
        ax_top_right_x = fig.add_axes([0.75, 0.5, 0.1, 0.08])
        ax_top_right_y = fig.add_axes([0.87, 0.5, 0.1, 0.08])

        ax_bottom_left_x.text(1.25, 1.5, "Bottom Left", fontsize=12, ha='center',
                              va='center', transform=ax_bottom_left_x.transAxes)
        ax_top_right_x.text(1.25, 1.5, "Top Right", fontsize=12, ha='center',
                            va='center', transform=ax_top_right_x.transAxes)

        self.textbox_bottom_left_x = TextBox(ax_bottom_left_x, None, textalignment='center',
                                             initial=str(self.bottom_left[0]))
        self.textbox_bottom_left_y = TextBox(ax_bottom_left_y, None, textalignment='center',
                                             initial=str(self.bottom_left[1]))
        self.textbox_top_right_x = TextBox(ax_top_right_x, None, textalignment='center',
                                           initial=str(self.top_right[0]))
        self.textbox_top_right_y = TextBox(ax_top_right_y, None, textalignment='center',
                                           initial=str(self.top_right[1]))

        self.rectangle = patches.Rectangle((self.bottom_left[0], self.bottom_left[1]),
                                           self.top_right[0]-self.bottom_left[0],
                                           self.top_right[1]-self.bottom_left[1],
                                           edgecolor='red', facecolor='none')
        ax.add_patch(self.rectangle)

        self.textbox_bottom_left_x.on_submit(self._update_bottom_left_x)
        self.textbox_bottom_left_y.on_submit(self._update_bottom_left_y)
        self.textbox_top_right_x.on_submit(self._update_top_right_x)
        self.textbox_top_right_y.on_submit(self._update_top_right_y)

        ax_close_button = fig.add_axes([0.87, 0.3, 0.1, 0.08])
        self.close_button = Button(ax_close_button, 'Close')
        self.close_button.on_clicked(self._close_window)

        ax_select_button = fig.add_axes([0.75, 0.3, 0.1, 0.08])
        self.select_button = CheckButtons(ax_select_button, ['Select'],
                                          frame_props={'sizes': [50]})

        self.select_rectangle = patches.Rectangle((0, 0), 1, 1, edgecolor='blue', facecolor='none',
                                                  ls='--')
        ax.add_patch(self.select_rectangle)
        self.select_rectangle.set_visible(False)

        self.is_selecting = False
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_select)

        plt.show()

    @staticmethod
    def _parse_corner(expression):
        try:
            return int(expression)
        except (TypeError, ValueError):
            print(f"Cannot interpret '{expression}' as an integer pixel index; value not changed.")
            return None

    def _update_bottom_left_x(self, expression):
        val = self._parse_corner(expression)
        if val is not None:
            self.bottom_left[0] = val
            self._update_common_extent()

    def _update_bottom_left_y(self, expression):
        val = self._parse_corner(expression)
        if val is not None:
            self.bottom_left[1] = val
            self._update_common_extent()

    def _update_top_right_x(self, expression):
        val = self._parse_corner(expression)
        if val is not None:
            self.top_right[0] = val
            self._update_common_extent()

    def _update_top_right_y(self, expression):
        val = self._parse_corner(expression)
        if val is not None:
            self.top_right[1] = val
            self._update_common_extent()

    def _update_text_from_rectangle(self):
        self.textbox_bottom_left_x.set_val(str(self.bottom_left[0]))
        self.textbox_bottom_left_y.set_val(str(self.bottom_left[1]))
        self.textbox_top_right_x.set_val(str(self.top_right[0]))
        self.textbox_top_right_y.set_val(str(self.top_right[1]))

    def _update_common_extent(self):
        self.rectangle.set_bounds(self.bottom_left[0], self.bottom_left[1],
                                  self.top_right[0]-self.bottom_left[0],
                                  self.top_right[1]-self.bottom_left[1])
        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if self.select_button.get_status()[0] and event.inaxes == self.ax:
            self.bottom_left_selecting = [event.xdata, event.ydata]
            self.is_selecting = True

    def _on_select(self, event):
        if self.is_selecting and self.select_button.get_status()[0] and event.inaxes == self.ax:
            self.top_right_selecting = [event.xdata, event.ydata]
            self._draw_select_rectangle()
            self.fig.canvas.draw_idle()

    def _draw_select_rectangle(self):
        self.select_rectangle.set_bounds(self.bottom_left_selecting[0], self.bottom_left_selecting[1],
                                         self.top_right_selecting[0]-self.bottom_left_selecting[0],
                                         self.top_right_selecting[1]-self.bottom_left_selecting[1])
        self.select_rectangle.set_visible(True)

    def _on_release(self, event):
        if self.is_selecting and self.select_button.get_status()[0] and event.inaxes == self.ax:
            self.is_selecting = False
            self.bottom_left = [int(np.min([self.bottom_left_selecting[0], self.top_right_selecting[0]])),
                                int(np.min([self.bottom_left_selecting[1], self.top_right_selecting[1]]))]
            self.top_right = [int(np.max([self.bottom_left_selecting[0], self.top_right_selecting[0]])),
                              int(np.max([self.bottom_left_selecting[1], self.top_right_selecting[1]]))]
            self.select_rectangle.set_visible(False)
            self._update_text_from_rectangle()
            self._update_common_extent()

    def _close_window(self, event):
        print('The selected region is: ', 'Bottom Left', self.bottom_left, 'Top Right', self.top_right)
        plt.close(self.fig)


def _shift_with_nan(data, dy, dx, order=3):
    """
    `scipy.ndimage.shift` of ``data`` by ``(dy, dx)`` (output[y, x] = input[y-dy, x-dx])
    with pixels whose source lies outside the input set to NaN.

    The interpolation itself uses ``mode='nearest'`` so that NaN does not leak
    into valid pixels through the spline support; the invalid border is then
    masked explicitly.
    """
    data = np.asarray(data, dtype=np.float32 if np.asarray(data).dtype.itemsize <= 4 else np.float64)
    out = shift(data, (dy, dx), order=order, mode='nearest')
    ny, nx = data.shape
    # valid source rows: 0 <= y - dy <= ny-1
    y_lo, y_hi = int(np.ceil(dy)), int(np.floor(ny - 1 + dy))
    x_lo, x_hi = int(np.ceil(dx)), int(np.floor(nx - 1 + dx))
    out[:max(y_lo, 0), :] = np.nan
    out[y_hi + 1:, :] = np.nan
    out[:, :max(x_lo, 0)] = np.nan
    out[:, x_hi + 1:] = np.nan
    return out


# ---------------------------------------------------------------------------
# Peak finding helpers copied from sunpy.image.coalignment version 0.5.1
# (removed from sunpy; now in sunkit_image.coalignment). The slice bounds in
# _find_best_match_location originally read ``corr.shape[i] - 1``, which cut the
# 3x3 window to 3x2 when the peak sits on the second-to-last row/column and
# then returned a position off by one pixel. Fixed here.
# ---------------------------------------------------------------------------

def _calculate_shift(this_layer, template):
    """
    Calculates the pixel shift required to put the template in the "best"
    position on a layer.

    Parameters
    ----------
    this_layer : `numpy.ndarray`
        A numpy array of size ``(ny, nx)``, where the first two dimensions are
        spatial dimensions.
    template : `numpy.ndarray`
        A numpy array of size ``(N, M)`` where ``N < ny`` and ``M < nx``.

    Returns
    -------
    `tuple`
        Pixel shifts ``(yshift, xshift)`` relative to the offset of the template
        to the input array.
    """
    # Warn user if any NANs, Infs, etc are present in the layer or the template
    _check_for_nonfinite_entries(this_layer, template)
    # Calculate the correlation array matching the template to this layer
    corr = match_template(this_layer, template)
    if corr.shape[0] < 2 or corr.shape[1] < 2:
        raise ValueError("The search area is too small for the template; increase search_margin.")
    # Calculate the y and x shifts in pixels
    return _find_best_match_location(corr)


def _find_best_match_location(corr):
    """
    Calculate an estimate of the location of the peak of the correlation result
    in image pixels.

    Parameters
    ----------
    corr : `numpy.ndarray`
        A 2D correlation array.

    Returns
    -------
    `~astropy.units.Quantity`
        The shift amounts ``(y, x)`` in image pixels. Subpixel values are
        possible.
    """
    # Get the index of the maximum in the correlation function
    ij = np.unravel_index(np.argmax(corr), corr.shape)
    cor_max_x, cor_max_y = ij[::-1]

    # Get the correlation function around the maximum (up to 3x3; the upper
    # bound is exclusive, hence no "-1")
    array_maximum = corr[
        np.max([0, cor_max_y - 1]) : np.min([cor_max_y + 2, corr.shape[0]]),
        np.max([0, cor_max_x - 1]) : np.min([cor_max_x + 2, corr.shape[1]]),
    ]
    y_shift_maximum, x_shift_maximum = _get_correlation_shifts(array_maximum)

    # Get shift relative to correlation array
    y_shift_correlation_array = y_shift_maximum + cor_max_y * u.pix
    x_shift_correlation_array = x_shift_maximum + cor_max_x * u.pix

    return y_shift_correlation_array, x_shift_correlation_array


def _get_correlation_shifts(array):
    """
    Estimate the location of the maximum of a fit to the input array. The
    estimation in the "x" and "y" directions are done separately. The location
    estimates can be used to implement subpixel shifts between two different
    images.

    Parameters
    ----------
    array : `numpy.ndarray`
        An array with at least one dimension that has three elements. The
        input array is at most a 3x3 array of correlation values calculated
        by matching a template to an image.

    Returns
    -------
    `~astropy.units.Quantity`
        The ``(y, x)`` location of the peak of a parabolic fit, in image pixels,
        relative to the maximum pixel of ``array``.
    """
    # Check input shape; work in float64 so that a float32 correlation array does
    # not limit the sub-pixel precision
    array = np.asarray(array, dtype=np.float64)
    ny = array.shape[0]
    nx = array.shape[1]
    if nx > 3 or ny > 3:
        msg = "Input array dimension should not be greater than 3 in any dimension."
        raise ValueError(msg)

    # Find where the maximum of the input array is
    ij = np.unravel_index(np.argmax(array), array.shape)
    x_max_location, y_max_location = ij[::-1]

    # Estimate the location of the parabolic peak if there is enough data
    # (3 samples centred on the maximum). Otherwise the maximum is at the edge
    # of the correlation array and no sub-pixel refinement is possible: return 0.
    y_location = _parabolic_turning_point(array[:, x_max_location]) if ny == 3 else 0.0
    x_location = _parabolic_turning_point(array[y_max_location, :]) if nx == 3 else 0.0

    return y_location * u.pix, x_location * u.pix


def _parabolic_turning_point(y):
    """
    Find the location of the turning point for a parabola ``y(x) = ax^2 + bx +
    c``, given input values ``y(-1), y(0), y(1)``. The maximum is located at
    ``x0 = -b / 2a``. Assumes that the input array represents an equally spaced
    sampling at the locations ``y(-1), y(0) and y(1)``.

    Parameters
    ----------
    y : `numpy.ndarray`
        A one dimensional numpy array of shape "3" with entries that sample the
        parabola at "-1", "0", and "1".

    Returns
    -------
    `float`
        A float, the location of the parabola maximum.
    """
    numerator = -0.5 * y.dot([-1, 0, 1])
    denominator = y.dot([1, -2, 1])
    return numerator / denominator


def _check_for_nonfinite_entries(layer_image, template_image):
    """
    Issue a warning if there is any nonfinite entry in the layer or template
    images.

    Parameters
    ----------
    layer_image : `numpy.ndarray`
        A two-dimensional `numpy.ndarray`.
    template_image : `numpy.ndarray`
        A two-dimensional `numpy.ndarray`.
    """
    if not np.all(np.isfinite(layer_image)):
        warnings.warn(
            "The layer image has nonfinite entries. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure there are no infinity or "
            "Not a Number values. For instance, replacing them with a "
            "local mean.",
            SunpyUserWarning,
            stacklevel=3,
        )

    if not np.all(np.isfinite(template_image)):
        warnings.warn(
            "The template image has nonfinite entries. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure there are no infinity or "
            "Not a Number values. For instance, replacing them with a "
            "local mean.",
            SunpyUserWarning,
            stacklevel=3,
        )


if __name__ == "__main__":
    '''
    Example usage:
    python map_coalign.py '/path/to/map_sequence/*.fits' /path/to/output_dir -ref 0 -i 3 -nh
    '''
    parser = argparse.ArgumentParser(description='Coalign a sequence of maps using a cross-correlation method '
                                                 '(Chitta et al. 2022, A&A 667, A166, App. A.1)')
    parser.add_argument('filename', type=str, help='input map sequence filename pattern (quote the wildcard)')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('-ref', '--reference_index', type=int, default=0, help='Index of the reference map')
    parser.add_argument('-bl', '--bottom_left', type=int, nargs=2, default=None, help='Bottom left corner [x y] of the region to align')
    parser.add_argument('-tr', '--top_right', type=int, nargs=2, default=None, help='Top right corner [x y] of the region to align')
    parser.add_argument('-n', '--nframes', type=int, default=10, help='Number of frames in each segment')
    parser.add_argument('-i', '--iter', type=int, default=3, help='Number of iterations')
    parser.add_argument('-m', '--method', type=str, default='match_template', choices=['match_template', 'phase'],
                        help='Shift measurement method')
    parser.add_argument('-sm', '--search_margin', type=int, default=50,
                        help='Search margin in pixels around the template box (match_template)')
    parser.add_argument('-up', '--upsample_factor', type=int, default=100,
                        help='Fourier upsampling factor for sub-pixel shifts (phase)')
    parser.add_argument('-nh', '--no_header_check', action='store_true', help='Do not check the header of the maps')
    parser.add_argument('-np', '--no_preview', action='store_true', help='Do not show the video preview of the coaligned map sequence')
    args = parser.parse_args()

    map_files = sorted(glob(args.filename))
    if len(map_files) == 0:
        raise SystemExit(f"No files match {args.filename}")
    ms = MapSequenceCoalign(sunpy.map.Map(map_files))
    ms.coalign(reference_index=args.reference_index, bottom_left=args.bottom_left, top_right=args.top_right,
               check_header=not args.no_header_check, nframes=args.nframes, iter=args.iter,
               method=args.method, search_margin=args.search_margin, upsample_factor=args.upsample_factor)

    if not args.no_preview:
        anim = ms.plot(no_wcs=True)
        plt.show()

    do_save = input('Do you want to save the coaligned map sequence? (y/n): ')
    if do_save.lower().startswith('y'):
        choice = input('Compression? [n]one / [g]zip (lossless) / [r]ice (lossy, quantize_level=16): ').lower()
        outpath = os.path.join(args.output_dir, "map_seq_coalign_{index:03}.fits")
        if choice.startswith('g'):
            ms.save(outpath, overwrite=True, compress=True, quantize_level=0)
        elif choice.startswith('r'):
            ms.save(outpath, overwrite=True, compress=True, quantize_level=16)
        else:
            ms.save(outpath, overwrite=True)
