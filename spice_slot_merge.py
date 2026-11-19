r"""
Merge Solar Orbiter/SPICE 30 arcsec slot (wide-slit) rasters into 2-D images.

With the 30 arcsec slot SPICE records, at each raster step, an *image* of the
slot instead of a spectrum. Along the detector rows the slot is sampled at
the slit plate scale (``CDELT2`` = 1.098 arcsec/pixel). Along the dispersion
axis an isolated emission line acts as a monochromator, so each dispersion
pixel samples a different position across the slot, 0.98 arcsec apart
(``|CDELT1 * PC1_3|``). A slot image is therefore the line radiance times the
slot transmission, smeared along the dispersion axis by the line profile.

This module turns each raster -- one L2 file, one window -- into a single
map on a regular helioprojective grid,

    S(x, y) = sum_i I_i(x, y) / sum_i R_i(x, y),

where ``I_i`` is the background-subtracted slot image of raster step ``i``
resampled onto the output grid and ``R_i`` the *slot response* (the slot
image of a uniform source, normalised to 1 on its plateau) resampled the same
way. This is the estimator of the 2022 crazy-mosaic pipeline
(github.com/gpelouze/spice_crazy_mosaic_reconstruction), and is the
inverse-variance optimum for photon-noise-limited data.

What differs from the 2022 pipeline
-----------------------------------
* **Parametric slot response.** ``R`` is an error-function top hat across the
  slot times one along it, with edge positions and widths fitted to the
  level-weighted mean of row-normalised slot images (`build_slot_response`).
  The 2022 code used the empirical median image itself, which needs hundreds
  of different scenes to average out solar structure; a movie study repeats
  the same few pointings. (A median of ratios is also biased at the slot
  edges for a structured scene; see `_mean_profile`.)
* **Out-of-slot pedestal is subtracted.** The slot images sit on a pedestal
  -- continuum integrated over the slot width, plus overlapping slot images
  of neighbouring lines -- of ~25% of the plateau for Ne VIII 770 and
  7-15% for the LW lines. Its levels either side of the slot correlate at
  r = 0.91-0.97 with each other, better than with the slot edges, i.e. it is
  additive rather than line wings. It is fitted per detector row from bands
  on either side of the slot and removed (`fit_background`); left in, it is
  divided by the falling response at every seam between raster steps.
* **Despiked pixels** are filled with the pipeline's ``ESTIMATED`` values from
  the ``SPIKPIXLIST`` tables, and the first pixel inside the boundary of the
  distortion-corrected window, which is biased by the resampling, is dropped.
* **Pointing.** The per-step ``WCSDVARR`` jitter correction is applied, the
  slot centre is taken from the fitted response rather than the window centre
  (``CRPIX3``; ~3 pixels apart here), and the dispersion axis is rotated
  with the spacecraft roll (`dispersion_vector`). ``step_scale`` corrects the
  raster step, which in the 2023-04-10 data is 1.036 x ``CDELT1`` according
  to FSI (`euispice_coalign.fit_slot_step_scale`).
* **Output** is one FITS file per L2 file and window, readable by
  `sunpy.map.Map`, with the coverage, an uncertainty map from the `sospice`
  noise model, the observation time of every pixel, and a table of the raster
  steps (see `merge_slot_raster`).

Known limitations
-----------------
* Overlapping lines are not separated: the Mg VIII 772.3 slot image enters
  the red third of every Ne VIII 770 slot (~5% of the signal there), and
  O I 1027.4 the red third of every Ly Beta 1025 slot.
* With steps equal to the slot width there is no redundancy, so a
  non-uniform transmission across the slot cannot be told from solar
  structure; ``R`` assumes a flat plateau.
* Seams between steps fix the dispersion axis only relative to the raster
  direction. Overlap tests prefer it rotated a few degrees from the raster
  direction towards Solar X; at most ~1.3 arcsec at the slot edges, below
  the resolution and kept as an option (``dispersion_angle``).

Reading the output
------------------
The primary HDU is the merged radiance; ``WEIGHT``, ``UNCERTAINTY`` and
``TIME`` are image extensions with the same WCS and metadata, so a bare
``sunpy.map.Map(file)`` returns a list of four maps. Select one with::

    smap = sunpy.map.Map(file, hdus=0)          # radiance
    weight = sunpy.map.Map(file, hdus=1)

Conventions
-----------
* Slot images are arrays ``(ny, nk)`` = (slit row ``j``, dispersion pixel
  ``k``); a raster is ``(n_steps, ny, nk)``.
* Pixel indices are 0-based unless a name says FITS/1-based.
* Angles in headers and WCS objects built here are in arcsec.

Command-line examples
---------------------
Merge every window of every file in a directory. The slot responses are
fitted from all the files first and saved to ``<output>/slot_response``::

    python spice_slot_merge.py /data/spice/ -o /data/spice_slot_merged

Only some windows (exact ``EXTNAME`` or an unambiguous part of it), with a
quick-look PNG per input file::

    python spice_slot_merge.py /data/spice/*.fits -o merged -w 'Ne VIII' 'C III' -p

Reuse previously fitted responses (e.g. from a longer series) instead of
refitting them::

    python spice_slot_merge.py new_files/*.fits -o merged --response-dir old/slot_response

North-up 1 arcsec grid instead of the instrument-aligned native grid::

    python spice_slot_merge.py /data/spice/ -o merged --grid north --pixel-size 1

Correct the raster step (as measured with
`euispice_coalign.fit_slot_step_scale`) and then co-align with FSI::

    python spice_slot_merge.py /data/spice/ -o merged -c1 1.036
    python euispice_coalign.py merged/<stem>_NeVIII770_slotmerge.fits '/data/fsi174/*.fits'

Python examples
---------------
::

    from glob import glob
    import sunpy.map
    from spice_slot_merge import build_slot_response, merge_slot_raster

    files = sorted(glob('/data/spice/*.fits'))
    resp = build_slot_response(files, 'Ne VIII 770 (Merged)')
    resp.to_fits('slot_response_NeVIII770.fits')
    hdul = merge_slot_raster(files[0], resp, step_scale=1.036)
    hdul.writeto('merged.fits')
    smap = sunpy.map.Map(hdul[0].data, hdul[0].header)
"""

import argparse
import re
import warnings
from dataclasses import dataclass, field
from glob import glob
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.special import erf

__all__ = ['SlotRaster', 'SlotResponse', 'read_slot_raster', 'list_slot_windows',
           'build_slot_response', 'fit_background', 'slot_variance', 'slot_wcs',
           'merge_slot_raster', 'merged_filename', 'window_tag', 'plot_slot_response',
           'plot_merged']

#: Rest wavelengths (Angstrom) of the main line of common SPICE windows. Only
#: used to fill ``WAVELNTH`` of the merged maps.
LINE_WAVELENGTHS = {
    'Ne VIII 770': 770.428,
    'C III 977': 977.020,
    'Ly Beta 1025': 1025.722,
    'O VI 1032': 1031.912,
    'N IV 765': 765.152,
    'O III 703': 703.850,
    'Mg IX 706': 706.060,
    'S V 786': 786.468,
    'O IV 787': 787.711,
    'Ne VIII 780': 780.385,
    'C II 1036': 1036.337,
    'O V 760': 760.446,
}

#: Header keywords copied from the L2 window header into the merged maps.
_COPY_KEYWORDS = (
    'DATE-OBS', 'DATE-BEG', 'DATE-AVG', 'DATE-END', 'TIMESYS', 'TELAPSE', 'XPOSURE',
    'OBSRVTRY', 'INSTRUME', 'TELESCOP', 'DETECTOR', 'LEVEL', 'CREATOR', 'VERSION',
    'SOAR_DR', 'STUDYTYP', 'STUDYDES', 'STUDY', 'OBS_MODE', 'OBS_ID', 'SPIOBSID',
    'RASTERNO', 'NRASTERS', 'OBS_DESC', 'PURPOSE', 'SOOPNAME', 'TARGET', 'MISOSTUD',
    'WIN_TYPE', 'WINNO', 'SLIT_ID', 'SLIT_WID', 'WAVEMIN', 'WAVEMAX', 'WAVEREF',
    'BTYPE', 'BUNIT', 'RADCAL', 'RSUN_REF', 'RSUN_ARC', 'DSUN_OBS', 'DSUN_AU',
    'HGLN_OBS', 'HGLT_OBS', 'CRLN_OBS', 'CRLT_OBS', 'SOLAR_B0', 'SOLAR_P0', 'SOLAR_EP',
    'CAR_ROT', 'HEEX_OBS', 'HEEY_OBS', 'HEEZ_OBS', 'HCIX_OBS', 'HCIY_OBS', 'HCIZ_OBS',
    'OBS_VR', 'EAR_TDEL', 'SUN_TIME', 'DATE_EAR', 'DATE_SUN',
)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

@dataclass
class SlotRaster:
    """
    One window of one SPICE wide-slit raster file, as a stack of slot images.

    Attributes
    ----------
    filename : str
    header : `astropy.io.fits.Header`
        Header of the window HDU.
    data : `numpy.ndarray`
        ``(n_steps, ny, nk)`` radiance in ``BUNIT`` (W m-2 sr-1 for slot data);
        index 0 is raster pixel 1 of the file. NaN where there is no data.
    times : `astropy.time.Time`
        Mid-exposure time of each raster step, ``(n_steps,)``.
    jitter : `numpy.ndarray`
        ``(n_steps, 2)`` per-step pointing correction (Solar X, Solar Y) in
        arcsec from the ``WCSDVARR`` extensions, to be *added* to the world
        coordinates. Zero if the file has none.
    radcal : `numpy.ndarray`
        ``(nk,)`` radiometric calibration factor, DN per ``BUNIT``.
    n_spikes_filled : int
        Number of despiked pixels restored from ``SPIKPIXLIST``.
    """
    filename: str
    header: fits.Header
    data: np.ndarray
    times: Time
    jitter: np.ndarray
    radcal: np.ndarray
    n_spikes_filled: int = 0

    @property
    def window(self):
        return self.header['EXTNAME']

    @property
    def n_steps(self):
        return self.data.shape[0]


def list_slot_windows(filename):
    """
    ``EXTNAME`` of the observational wide-slit windows of a SPICE L2 file.
    """
    names = []
    with fits.open(filename) as hdul:
        for hdu in hdul:
            hdr = hdu.header
            if hdr.get('NAXIS', 0) == 4 and hdr.get('CTYPE1', '').startswith('HPLN') \
                    and 'wide' in str(hdr.get('WIN_TYPE', '')).lower():
                names.append(hdr['EXTNAME'])
    return names


def _find_window_hdu(hdul, window):
    """
    Return the HDU of ``window``: an index into the observational windows,
    an exact ``EXTNAME``, or a case-insensitive substring matching one window.
    """
    obs = [hdu for hdu in hdul
           if hdu.header.get('NAXIS', 0) == 4 and hdu.header.get('CTYPE1', '').startswith('HPLN')]
    names = [hdu.header['EXTNAME'] for hdu in obs]
    if isinstance(window, (int, np.integer)):
        try:
            return obs[window]
        except IndexError:
            raise IndexError(f'window index {window} out of range; windows are {names}') from None
    if window in names:
        return obs[names.index(window)]
    matches = [i for i, n in enumerate(names) if str(window).lower() in n.lower()]
    if len(matches) == 1:
        return obs[matches[0]]
    raise KeyError(f'window {window!r} matches {len(matches)} of the windows {names}')


def _step_times(header):
    """
    Mid-exposure time of each raster step from the time axis of the WCS.

    ``PC4_1`` ties time to raster step; it is negative for SPICE rasters, which
    scan from the last pixel to the first.
    """
    n_steps = header['NAXIS1']
    ref = Time(header.get('DATEREF', header['DATE-BEG']), scale='utc')
    p1 = np.arange(n_steps) + 1.0
    t = header.get('CRVAL4', 0.0) + header.get('CDELT4', 1.0) * (
        header.get('PC4_1', 0.0) * (p1 - header['CRPIX1'])
        + header.get('PC4_4', 1.0) * (1.0 - header.get('CRPIX4', 1.0)))
    return ref + t * u.Unit(header.get('CUNIT4', 's'))


def _read_jitter(hdul, header, n_steps):
    """
    Per-step pointing correction (arcsec) from the ``WCSDVARR`` lookup tables.

    SPICE stores the measured pointing drift as a SOLARNET 'Lookup' distortion
    applied to the *world* coordinates (``CWDISi``, ``DWi``; DPDD
    SPICE-UIO-DPDD-0002 issue 2.2, p. 47). astropy does not implement it, so
    the values are added by hand, with the same sign as ``euispice_coalign``.
    """
    jitter = np.zeros((n_steps, 2))
    for axis in (1, 2):
        if str(header.get(f'CWDIS{axis}', '')).strip().lower() != 'lookup':
            continue
        extver = int(header.get(f'DW{axis}.EXTVER', axis))
        try:
            dhdu = hdul['WCSDVARR', extver]
        except KeyError:
            warnings.warn(f'CWDIS{axis} = Lookup but WCSDVARR extension {extver} is missing; '
                          'no jitter correction applied for this axis.')
            continue
        table = np.asarray(dhdu.data, dtype=float).ravel()
        dh = dhdu.header
        # Lookup-table index (1-based) of each raster step (image pixel, 1-based)
        idx = (np.arange(n_steps) + 1 - dh.get('CRVAL1', 1.0)) / dh.get('CDELT1', 1.0) \
            + dh.get('CRPIX1', 1.0)
        jitter[:, axis - 1] = np.interp(idx, np.arange(table.size) + 1.0, table)
    return jitter


def _read_radcal(hdul, header, nk):
    """
    Per-dispersion-pixel calibration factor (DN per BUNIT), falling back to
    the scalar header ``RADCAL``.
    """
    column = f"RADCAL[{header['EXTNAME']}]"
    if 'VARIABLE_KEYWORDS' in hdul:
        table = hdul['VARIABLE_KEYWORDS'].data
        if column in table.columns.names:
            radcal = np.asarray(table[column][0], dtype=float).ravel()
            if radcal.size == nk:
                return radcal
    return np.full(nk, float(header['RADCAL']))


def _fill_spikes(hdul, extname, data):
    """
    Restore despiked pixels from the ``ESTIMATED`` column of the window's
    ``SPIKPIXLIST`` table. ``data`` is the FITS-order array ``(nk, ny, n_steps)``
    of the single exposure and is modified in place.

    Returns
    -------
    int
        Number of pixels filled.
    """
    name = f'SPIKPIXLIST[{extname}]'
    if name not in hdul:
        return 0
    table = hdul[name].data
    i = np.round(table['DIMENSION1']).astype(int) - 1
    j = np.round(table['DIMENSION2']).astype(int) - 1
    k = np.round(table['DIMENSION3']).astype(int) - 1
    t = np.round(table['DIMENSION4']).astype(int) - 1
    est = np.asarray(table['ESTIMATED'], dtype=float)
    ok = (t == 0) & np.isfinite(est)
    ok &= (k >= 0) & (k < data.shape[0]) & (j >= 0) & (j < data.shape[1]) \
        & (i >= 0) & (i < data.shape[2])
    ok[ok] &= np.isnan(data[k[ok], j[ok], i[ok]])
    data[k[ok], j[ok], i[ok]] = est[ok]
    return int(ok.sum())


def _erode_rows(cube, n):
    """
    Blank the first and last ``n`` finite pixels of every row (along the
    dispersion axis) of ``cube`` ``(..., nk)``, in place.

    The L2 distortion correction shears each window into a parallelogram; the
    pixels on its boundary are partial resamples and are biased by 5-40%.
    """
    if n <= 0:
        return cube
    finite = np.isfinite(cube)
    nk = cube.shape[-1]
    has = finite.any(axis=-1)
    first = np.where(has, np.argmax(finite, axis=-1), nk)
    last = np.where(has, nk - 1 - np.argmax(finite[..., ::-1], axis=-1), -1)
    k = np.arange(nk)
    blank = (k < first[..., None] + n) | (k > last[..., None] - n)
    cube[blank] = np.nan
    return cube


def read_slot_raster(filename, window, fill_spikes=True, erode=1):
    """
    Read one window of a SPICE L2 wide-slit raster as slot images.

    Parameters
    ----------
    filename : str or `pathlib.Path`
    window : int or str
        Index into the observational windows, exact ``EXTNAME``, or an
        unambiguous case-insensitive part of it (e.g. ``'Ne VIII'``).
    fill_spikes : bool, optional
        Restore despiked pixels from the ``SPIKPIXLIST`` ``ESTIMATED`` values.
    erode : int, optional
        Number of pixels dropped at each end of every row of valid data.

    Returns
    -------
    `SlotRaster`
    """
    with fits.open(filename) as hdul:
        hdu = _find_window_hdu(hdul, window)
        header = hdu.header.copy()
        if header.get('NAXIS4', 1) != 1:
            raise NotImplementedError('only one exposure per raster step is supported '
                                      f"(NAXIS4 = {header['NAXIS4']})")
        if abs(header.get('PC1_3', 0.0)) + abs(header.get('PC2_3', 0.0)) == 0:
            raise ValueError(f"{header['EXTNAME']} has no PC1_3/PC2_3 coupling between the "
                             'dispersion axis and Solar X; is this wide-slit data?')
        data = np.array(hdu.data[0], dtype=float)          # (nk, ny, n_steps)
        n_filled = _fill_spikes(hdul, header['EXTNAME'], data) if fill_spikes else 0
        jitter = _read_jitter(hdul, header, data.shape[2])
        radcal = _read_radcal(hdul, header, data.shape[0])
    cube = np.ascontiguousarray(data.transpose(2, 1, 0))   # (n_steps, ny, nk)
    _erode_rows(cube, erode)
    return SlotRaster(filename=str(filename), header=header, data=cube,
                      times=_step_times(header), jitter=jitter, radcal=radcal,
                      n_spikes_filled=n_filled)


# ---------------------------------------------------------------------------
# Slot response
# ---------------------------------------------------------------------------

def _edge_profile(x, lo, hi, sigma_lo, sigma_hi):
    """
    Top hat between ``lo`` and ``hi`` (half-maximum points) with Gaussian-
    blurred edges; 1 on the plateau.
    """
    return 0.5 * (erf((x - lo) / (np.sqrt(2) * sigma_lo))
                  - erf((x - hi) / (np.sqrt(2) * sigma_hi)))


@dataclass
class SlotResponse:
    """
    Slot response function of one window: the slot image of a uniform source,
    normalised to 1 on its plateau,

        R(j, k) = E(k - tilt * (j - j_ref); k_left, k_right, sigma_left, sigma_right)
                  * E(j; j_bottom, j_top, sigma_bottom, sigma_top),

    where ``E`` is a top hat with Gaussian edges (`_edge_profile`). All
    positions are 0-based pixel indices, widths are Gaussian sigmas in pixels.

    ``profile_k`` and ``profile_j`` are the empirical profiles the model was
    fitted to, kept for diagnostics.
    """
    window: str
    shape: tuple
    k_left: float
    k_right: float
    sigma_left: float
    sigma_right: float
    j_bottom: float
    j_top: float
    sigma_bottom: float
    sigma_top: float
    tilt: float = 0.0
    j_ref: float = 0.0
    profile_k: np.ndarray = None
    profile_j: np.ndarray = None
    files: list = field(default_factory=list)
    n_rows_used: int = 0

    @property
    def k_center(self):
        """Dispersion pixel of the slot centre at row ``j_ref``."""
        return 0.5 * (self.k_left + self.k_right)

    @property
    def width(self):
        """Half-maximum width of the slot image, in dispersion pixels."""
        return self.k_right - self.k_left

    def k_edges(self, j):
        """Left and right half-maximum edges at row(s) ``j``."""
        shift = self.tilt * (np.asarray(j, dtype=float) - self.j_ref)
        return self.k_left + shift, self.k_right + shift

    def array(self, shape=None):
        """
        The response sampled on the detector grid, ``(ny, nk)``.
        """
        ny, nk = self.shape if shape is None else shape
        j = np.arange(ny, dtype=float)[:, None]
        k = np.arange(nk, dtype=float)[None, :]
        kk = k - self.tilt * (j - self.j_ref)
        return (_edge_profile(kk, self.k_left, self.k_right, self.sigma_left, self.sigma_right)
                * _edge_profile(j, self.j_bottom, self.j_top, self.sigma_bottom, self.sigma_top))

    # FITS keyword <-> attribute
    _KEYS = (('KLEFT', 'k_left', 'Left half-max edge, 0-based dispersion pixel'),
             ('KRIGHT', 'k_right', 'Right half-max edge, 0-based dispersion pixel'),
             ('SIGLEFT', 'sigma_left', '[pixel] Gaussian sigma of left edge'),
             ('SIGRIGHT', 'sigma_right', '[pixel] Gaussian sigma of right edge'),
             ('JBOTTOM', 'j_bottom', 'Bottom half-max end, 0-based slit row'),
             ('JTOP', 'j_top', 'Top half-max end, 0-based slit row'),
             ('SIGBOT', 'sigma_bottom', '[pixel] Gaussian sigma of bottom end'),
             ('SIGTOP', 'sigma_top', '[pixel] Gaussian sigma of top end'),
             ('TILT', 'tilt', '[pixel/row] Slope of the slot edges dk/dj'),
             ('JREF', 'j_ref', 'Row about which TILT is defined, 0-based'))

    def to_fits(self, filename, overwrite=True):
        """
        Save the response: primary HDU is `array`, the parameters are header
        keywords, plus the empirical profiles and the list of input files.
        """
        hdr = fits.Header()
        hdr['EXTNAME'] = 'SLOT_RESPONSE'
        hdr['WINDOW'] = (self.window, 'SPICE window (EXTNAME)')
        for key, attr, comment in self._KEYS:
            hdr[key] = (float(getattr(self, attr)), comment)
        hdr['NROWS'] = (int(self.n_rows_used), 'Slot-image rows in the mean profile')
        hdr['NFILES'] = (len(self.files), 'Number of L2 files used')
        hdr.add_history('spice_slot_merge.build_slot_response')
        hdus = [fits.PrimaryHDU(self.array().astype(np.float32), header=hdr)]
        for name, profile in (('PROFILE_K', self.profile_k), ('PROFILE_J', self.profile_j)):
            if profile is not None:
                hdus.append(fits.ImageHDU(np.asarray(profile, dtype=np.float32), name=name))
        if self.files:
            hdus.append(fits.BinTableHDU(Table({'FILENAME': [Path(f).name for f in self.files]}),
                                         name='FILES'))
        fits.HDUList(hdus).writeto(filename, overwrite=overwrite)

    @classmethod
    def from_fits(cls, filename):
        with fits.open(filename) as hdul:
            hdr = hdul[0].header
            kwargs = {attr: hdr[key] for key, attr, _ in cls._KEYS}
            profile_k = hdul['PROFILE_K'].data.astype(float) if 'PROFILE_K' in hdul else None
            profile_j = hdul['PROFILE_J'].data.astype(float) if 'PROFILE_J' in hdul else None
            files = list(hdul['FILES'].data['FILENAME']) if 'FILES' in hdul else []
            shape = hdul[0].data.shape
            return cls(window=hdr['WINDOW'], shape=shape, profile_k=profile_k,
                       profile_j=profile_j, files=files, n_rows_used=hdr.get('NROWS', 0),
                       **kwargs)


def _nan_running_median(x, size):
    """Running median of a 1-D array ignoring NaN; NaN where a window is all-NaN."""
    if size <= 1:
        return x.copy()
    half = size // 2
    padded = np.pad(x, half, mode='constant', constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, size)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.nanmedian(windows, axis=-1)


def fit_background(img, k_left, k_right, margin_left, margin_right, band_width=8,
                   smooth_rows=7, min_pixels=3):
    """
    Pedestal under a slot image, linear in dispersion pixel, fitted per row.

    For each row the medians of two bands, ``[k_left - margin_left -
    band_width, k_left - margin_left]`` and the mirror band right of the slot,
    are taken, smoothed along the slit with a running median, and joined by a
    straight line. Where only one band has data (the distortion-corrected
    window is a parallelogram, so one band is cut off at the top or bottom of
    the slot) the other is predicted from the median right/left ratio of the
    image.

    Parameters
    ----------
    img : `numpy.ndarray`
        Slot image ``(ny, nk)``, NaN where there is no data.
    k_left, k_right : float or `numpy.ndarray`
        Half-maximum edges of the slot, scalar or per row ``(ny,)``.
    margin_left, margin_right : float
        Gap between each edge and its band, in pixels; ~2 sigma of the edge.
    band_width : float, optional
    smooth_rows : int, optional
        Length of the running median along the slit.
    min_pixels : int, optional
        Minimum number of valid pixels for a band median in a row.

    Returns
    -------
    `numpy.ndarray`
        ``(ny, nk)`` background, finite everywhere.
    """
    ny, nk = img.shape
    k = np.arange(nk, dtype=float)[None, :]
    kl = np.broadcast_to(np.asarray(k_left, dtype=float), (ny,))[:, None]
    kr = np.broadcast_to(np.asarray(k_right, dtype=float), (ny,))[:, None]
    left = (k >= kl - margin_left - band_width) & (k <= kl - margin_left)
    right = (k >= kr + margin_right) & (k <= kr + margin_right + band_width)

    def band_median(mask):
        vals = np.where(mask, img, np.nan)
        count = np.sum(np.isfinite(vals), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            med = np.nanmedian(vals, axis=1)
        med[count < min_pixels] = np.nan
        return med

    b_left = _nan_running_median(band_median(left), smooth_rows)
    b_right = _nan_running_median(band_median(right), smooth_rows)
    # Nominal band centres (where the straight line is anchored)
    k_left_c = (kl - margin_left - band_width / 2)[:, 0]
    k_right_c = (kr + margin_right + band_width / 2)[:, 0]

    both = np.isfinite(b_left) & np.isfinite(b_right) & (b_left > 0) & (b_right > 0)
    if both.sum() >= 5:
        ratio = np.median(b_right[both] / b_left[both])
    else:
        ratio = 1.0
    b_left = np.where(np.isfinite(b_left), b_left, b_right / ratio)
    b_right = np.where(np.isfinite(b_right), b_right, b_left * ratio)

    rows = np.arange(ny)
    for b in (b_left, b_right):
        ok = np.isfinite(b)
        if not ok.any():
            b[:] = 0.0
        elif not ok.all():
            b[~ok] = np.interp(rows[~ok], rows[ok], b[ok])
    slope = (b_right - b_left) / (k_right_c - k_left_c)
    return b_left[:, None] + slope[:, None] * (k - k_left_c[:, None])


def _half_max_crossings(profile):
    """
    Sub-pixel positions where a slot profile crosses half way between its
    5th and 95th percentiles, outermost on each side of the maximum.
    """
    x = np.arange(profile.size, dtype=float)
    ok = np.isfinite(profile)
    lo, hi = np.nanpercentile(profile, [5, 95])
    half = 0.5 * (lo + hi)
    above = ok & (profile > half)
    idx = np.where(above)[0]
    if idx.size < 3:
        raise ValueError('could not locate the slot image: no plateau above half maximum')
    i0, i1 = idx[0], idx[-1]
    left = np.interp(half, [profile[i0 - 1], profile[i0]], [x[i0 - 1], x[i0]]) if i0 > 0 else x[i0]
    right = np.interp(half, [profile[i1 + 1], profile[i1]], [x[i1 + 1], x[i1]]) \
        if i1 < profile.size - 1 else x[i1]
    return float(left), float(right)


def _mean_profile(rows, level, clip=5.0):
    """
    Level-weighted mean of normalised rows, ``sum(rows) / sum(level)`` per
    column, after rejecting pixels whose normalised value is more than
    ``clip`` robust sigmas from the column median.

    The plain median of ``rows / level`` is biased at the slot edges for a
    structured scene: the ratio of an edge pixel to the core mean has a skewed
    distribution, which narrows the fitted slot by ~0.7 pixel in tests. The
    ratio of sums is unbiased as long as the scene is on average as bright at
    the edges as in the core.

    Parameters
    ----------
    rows : `numpy.ndarray`
        ``(M, nk)`` background-subtracted rows.
    level : `numpy.ndarray`
        ``(M,)`` core level of each row.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        norm = rows / level[:, None]
        med = np.nanmedian(norm, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(norm - med), axis=0)
        ok = np.isfinite(norm) & (np.abs(norm - med) <= clip * np.maximum(mad, 1e-6))
        return np.where(ok, rows, 0).sum(axis=0) / np.where(ok, level[:, None], 0).sum(axis=0)


def _fit_k_profile(k, prof, p0, fix_sigma=None):
    """
    Fit ``A * E(k; kl, kr, sl, sr) + c`` to a normalised dispersion profile.
    Returns the parameter vector (A, kl, kr, sl, sr, c).
    """
    ok = np.isfinite(prof)
    if fix_sigma is None:
        def resid(p):
            return p[0] * _edge_profile(k[ok], p[1], p[2], p[3], p[4]) + p[5] - prof[ok]
        lower = [0.1, k[0] - 10, k[0], 0.3, 0.3, -1]
        upper = [10, k[-1], k[-1] + 10, 20, 20, 1]
        res = least_squares(resid, p0, bounds=(lower, upper), loss='soft_l1', f_scale=0.05)
        return res.x
    sl, sr = fix_sigma

    def resid(p):
        return p[0] * _edge_profile(k[ok], p[1], p[2], sl, sr) + p[3] - prof[ok]
    res = least_squares(resid, [p0[0], p0[1], p0[2], p0[5]], loss='soft_l1', f_scale=0.05)
    a, kl, kr, c = res.x
    return np.array([a, kl, kr, sl, sr, c])


def _slot_ends(profile):
    """
    Rough slot ends along the slit: the steepest rise and fall of the log of
    the row profile in the lower and upper halves.

    The level inside the slot follows the scene, so a half-maximum criterion
    finds dark patches; at the real ends the level drops by 10-30x within a
    few rows, which no solar structure does along the slit.
    """
    # Floor at 10% of the median so that the step from NaN rows to the dark
    # level outside the slot does not count as an edge.
    floor = 0.1 * np.nanmedian(profile[profile > 0])
    p = np.convolve(np.nan_to_num(profile, nan=floor), np.ones(3) / 3, mode='same')
    logp = np.log(np.clip(p, floor, None))
    d = np.gradient(logp)
    mid = profile.size // 2
    return float(np.argmax(d[:mid])), float(mid + np.argmin(d[mid:]))


def _fit_end(j, prof, j0, side, max_shift=6.0):
    """
    Refine one end of the slot along the slit: ``(a + b (j - j0)) * step + c``
    where ``step`` rises (``side='bottom'``) or falls (``'top'``) through 0.5
    at ``j0`` with Gaussian width sigma. ``j0`` may move at most ``max_shift``
    rows from the initial guess. Returns (j0, sigma).
    """
    sgn = 1.0 if side == 'bottom' else -1.0
    ok = np.isfinite(prof)

    def model(p, jj):
        step = 0.5 * (1 + erf(sgn * (jj - p[0]) / (np.sqrt(2) * p[1])))
        return (p[2] + p[3] * (jj - p[0])) * step + p[4]

    inside = ok & (sgn * (j - j0) > 4)
    level = np.nanmedian(prof[inside]) if inside.any() else 1.0
    p0 = [j0, 2.0, level, 0.0, 0.0]
    res = least_squares(lambda p: model(p, j[ok]) - prof[ok], p0,
                        bounds=([j0 - max_shift, 0.3, 0, -1, -1],
                                [j0 + max_shift, 10, 10 * max(level, 0.1), 1, 1]),
                        loss='soft_l1', f_scale=0.05)
    return float(res.x[0]), float(res.x[1])


def build_slot_response(files, window, n_iter=3, band_width=8, margin_sigma=2.0,
                        smooth_rows=7, fit_tilt=True, row_bin=64, verbose=True):
    """
    Fit the slot response of one window to the slot images of many rasters.

    Each slot image row is background-subtracted (`fit_background`) and its
    mean over the central half of the slot taken as its level; the level-
    weighted mean of the normalised rows of all raster steps of all files
    (`_mean_profile`) is the empirical cross-slot profile, to which an
    error-function top hat is fitted. Along the slit, the two ends of the slot
    are located on the median row profile (`_slot_ends`) and fitted the same
    way. Background bands depend on the edges, so the steps are iterated.

    Parameters
    ----------
    files : list of str
        SPICE L2 wide-slit raster files, all with the same window layout.
    window : int or str
        See `read_slot_raster`.
    n_iter : int, optional
        Background/edge iterations.
    band_width, margin_sigma, smooth_rows : optional
        Background bands, see `fit_background`; ``margin = margin_sigma *
        sigma`` of the corresponding edge.
    fit_tilt : bool, optional
        Fit a linear drift of the slot centre with slit row from profiles of
        ``row_bin``-row bins. It is kept only if it amounts to more than 0.5
        pixel over the slot and is significant at 3 sigma.
    verbose : bool, optional

    Returns
    -------
    `SlotResponse`
    """
    files = [str(f) for f in files]
    rasters = [read_slot_raster(f, window) for f in files]
    window_name = rasters[0].window
    stack = np.concatenate([r.data for r in rasters])          # (N, ny, nk)
    n_img, ny, nk = stack.shape
    k = np.arange(nk, dtype=float)
    j = np.arange(ny, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # Initial guesses from the raw median profile of the middle of the slit
        rows0 = slice(ny // 6, 5 * ny // 6)
        k_left, k_right = _half_max_crossings(np.nanmedian(stack[:, rows0], axis=(0, 1)))
        core0 = slice(int(round(0.5 * (k_left + k_right) - (k_right - k_left) / 4)),
                      int(round(0.5 * (k_left + k_right) + (k_right - k_left) / 4)) + 1)
        yprof0 = np.nanmedian(np.nanmean(stack[..., core0], axis=-1), axis=0)
        j_bottom, j_top = _slot_ends(yprof0)
    sigma_left = sigma_right = 3.0
    sigma_bottom = sigma_top = 2.0
    tilt, j_ref = 0.0, 0.5 * (j_bottom + j_top)

    for it in range(n_iter):
        shift = tilt * (j - j_ref)
        kl_rows, kr_rows = k_left + shift, k_right + shift
        sub = np.empty_like(stack)
        for n in range(n_img):
            sub[n] = stack[n] - fit_background(
                stack[n], kl_rows, kr_rows, margin_sigma * sigma_left,
                margin_sigma * sigma_right, band_width=band_width, smooth_rows=smooth_rows)
        width = k_right - k_left
        core = (np.abs(k[None, :] - (0.5 * (kl_rows + kr_rows))[:, None]) <= width / 4)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            level = np.nanmean(np.where(core[None], sub, np.nan), axis=-1)   # (N, ny)
            # Rows well inside the slot. Only near-empty rows are rejected:
            # selecting bright cores would select rows whose core is brighter
            # than their edges and bias the edges low (by ~10% for a
            # strongly structured scene); the level weighting of
            # `_mean_profile` already makes faint rows count little.
            inside = (j > j_bottom + 3 * sigma_bottom + 5) & (j < j_top - 3 * sigma_top - 5)
            good = inside[None, :] & np.isfinite(level) \
                & (level > 0.1 * np.nanmedian(level[:, inside]))
        rows_sub = sub[good]                                               # (M, nk)
        rows_level = level[good]                                           # (M,)
        rows_j = np.broadcast_to(j[None, :], good.shape)[good]
        if tilt != 0:
            # Shift every row to the frame of j_ref before averaging
            coords = np.array([np.repeat(np.arange(rows_sub.shape[0])[:, None], nk, 1),
                               k[None, :] + tilt * (rows_j[:, None] - j_ref)])
            shifted = map_coordinates(np.nan_to_num(rows_sub, nan=0.0), coords, order=1,
                                      mode='nearest', prefilter=False)
            profile_k = _mean_profile(np.where(np.isfinite(rows_sub), shifted, np.nan),
                                      rows_level)
        else:
            profile_k = _mean_profile(rows_sub, rows_level)

        # Fit the cross-slot profile around the slot, away from the window ends
        # (where the boundary of the distortion-corrected window is).
        fit_k = (k >= max(2, k_left - 4 * sigma_left - 3)) \
            & (k <= min(nk - 3, k_right + 4 * sigma_right + 3))
        p = _fit_k_profile(k[fit_k], profile_k[fit_k],
                           [1.0, k_left, k_right, sigma_left, sigma_right, 0.0])
        k_left, k_right, sigma_left, sigma_right = p[1], p[2], p[3], p[4]

        # Drift of the slot centre along the slit, measured in raw detector
        # coordinates on bins of rows, so the fitted slope is the total tilt.
        if fit_tilt:
            centres, rows_c = [], []
            for j0 in range(int(j_bottom + 10), int(j_top - 10) - row_bin + 1, row_bin):
                sel = (rows_j >= j0) & (rows_j < j0 + row_bin)
                if sel.sum() < 50:
                    continue
                pb = _mean_profile(rows_sub[sel], rows_level[sel])
                pbf = _fit_k_profile(k[fit_k], pb[fit_k], p, fix_sigma=(sigma_left, sigma_right))
                centres.append(0.5 * (pbf[1] + pbf[2]))
                rows_c.append(j0 + row_bin / 2)
            tilt = 0.0
            if len(centres) >= 4:
                rows_c, centres = np.array(rows_c), np.array(centres)
                fit = least_squares(lambda q: q[0] + q[1] * (rows_c - j_ref) - centres,
                                    [np.median(centres), 0.0], loss='soft_l1', f_scale=0.3)
                slope = fit.x[1]
                r = centres - (fit.x[0] + slope * (rows_c - j_ref))
                scatter = 1.4826 * np.median(np.abs(r - np.median(r)))
                slope_err = scatter / np.sqrt(np.sum((rows_c - rows_c.mean()) ** 2)) \
                    if scatter > 0 else np.inf
                if abs(slope) * (j_top - j_bottom) > 0.5 and abs(slope) > 3 * slope_err:
                    tilt = slope

        # Ends of the slot along the slit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            profile_j = np.nanmedian(level, axis=0)
            profile_j = profile_j / np.nanmedian(profile_j[inside])
        # Re-locate the ends on the background-subtracted profile, then refine
        j_bottom, j_top = _slot_ends(profile_j)
        win = 15
        jb = (j > j_bottom - win) & (j < j_bottom + win)
        jt = (j > j_top - win) & (j < j_top + win)
        j_bottom, sigma_bottom = _fit_end(j[jb], profile_j[jb], j_bottom, 'bottom')
        j_top, sigma_top = _fit_end(j[jt], profile_j[jt], j_top, 'top')
        if verbose:
            print(f'[{window_name}] iteration {it + 1}: k edges {k_left:.2f}-{k_right:.2f} '
                  f'(sigma {sigma_left:.2f}, {sigma_right:.2f}), width {k_right - k_left:.2f} px, '
                  f'tilt {tilt:.2e} px/row, rows {j_bottom:.1f}-{j_top:.1f} '
                  f'(sigma {sigma_bottom:.2f}, {sigma_top:.2f}), {int(good.sum())} rows')

    return SlotResponse(window=window_name, shape=(ny, nk), k_left=k_left, k_right=k_right,
                        sigma_left=sigma_left, sigma_right=sigma_right, j_bottom=j_bottom,
                        j_top=j_top, sigma_bottom=sigma_bottom, sigma_top=sigma_top,
                        tilt=tilt, j_ref=j_ref, profile_k=profile_k, profile_j=profile_j,
                        files=files, n_rows_used=int(good.sum()))


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def slot_variance(raster):
    """
    Per-pixel noise variance of the slot images from the `sospice` model.

    `sospice.calibrate.spice_error` requires data in W m-2 sr-1 nm-1, but
    wide-slit L2 data are line radiances in W m-2 sr-1 with ``RADCAL`` in DN
    per W m-2 sr-1. Calling `sospice.instrument_modelling.Observation`
    directly on DN avoids relabelling the units: shot, dark and read noise are
    computed in DN and converted back with the per-pixel ``RADCAL``. On-board
    JPEG compression noise is not modelled.

    Returns
    -------
    `numpy.ndarray`
        ``(n_steps, ny, nk)`` variance in ``BUNIT**2``.
    """
    from sospice.instrument_modelling import Observation, Spice, Study

    study = Study()
    study.init_from_header(raster.header)
    observation = Observation(Spice(), study)
    radcal = raster.radcal[None, None, :]
    signal = np.nan_to_num(raster.data * radcal, nan=0.0) * u.ct / u.pix
    with np.errstate(invalid='ignore'):
        _, sigma = observation.noise_effects(signal, study.av_wavelength)
    var = (sigma['Total'].to_value(u.ct / u.pix) / radcal) ** 2
    var[~np.isfinite(raster.data)] = np.nan
    return var


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def dispersion_vector(header, roll_corrected=True, scale=None, angle=0.0):
    """
    Solar (X, Y) displacement, in arcsec, of one dispersion pixel.

    The L2 header gives it as ``(CDELT1 * PC1_3, CDELT2 * PC2_3)`` with
    ``PC2_3`` absent, i.e. purely along Solar X. But the dispersion axis is
    the instrument X axis, the same axis the raster steps along, and the
    raster steps are rotated by the roll (``PC1_1``, ``PC2_1``). With
    ``roll_corrected=True`` the header's plate scale is kept and the vector is
    turned onto the raster direction.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
    roll_corrected : bool, optional
    scale : float, optional
        Plate scale (arcsec per pixel), overriding ``|CDELT1 * PC1_3|``.
    angle : float, optional
        Further counter-clockwise rotation of the vector, in degrees, e.g. to
        model a dispersion axis that is not exactly perpendicular to the slit.
    """
    d = np.array([header['CDELT1'] * header.get('PC1_3', 0.0),
                  header['CDELT2'] * header.get('PC2_3', 0.0)])
    header_scale = np.hypot(*d)
    if header_scale == 0:
        raise ValueError('header has no dispersion-to-space coupling (PC1_3, PC2_3)')
    if scale is not None:
        d = d / header_scale * scale
    scale = np.hypot(*d)
    if roll_corrected:
        raster_dir = np.array([header['CDELT1'] * header['PC1_1'],
                               header['CDELT2'] * header['PC2_1']])
        raster_dir /= np.hypot(*raster_dir)
        d = np.sign(d @ raster_dir) * scale * raster_dir
    if angle:
        a = np.deg2rad(angle)
        d = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ d
    return d


def _celestial_header(crval, cdelt, crpix, pc, shape=None):
    """
    Minimal 2-D helioprojective header; angles in arcsec, ``crpix`` 1-based.
    """
    hdr = fits.Header()
    if shape is not None:
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = int(shape[1])
        hdr['NAXIS2'] = int(shape[0])
    hdr['CTYPE1'] = 'HPLN-TAN'
    hdr['CTYPE2'] = 'HPLT-TAN'
    hdr['CUNIT1'] = 'arcsec'
    hdr['CUNIT2'] = 'arcsec'
    hdr['CRVAL1'] = float(crval[0])
    hdr['CRVAL2'] = float(crval[1])
    hdr['CDELT1'] = float(cdelt[0])
    hdr['CDELT2'] = float(cdelt[1])
    hdr['CRPIX1'] = float(crpix[0])
    hdr['CRPIX2'] = float(crpix[1])
    hdr['PC1_1'] = float(pc[0][0])
    hdr['PC1_2'] = float(pc[0][1])
    hdr['PC2_1'] = float(pc[1][0])
    hdr['PC2_2'] = float(pc[1][1])
    return hdr


def slot_wcs(header, step, k_center, roll_corrected=True, jitter=(0.0, 0.0), tilt=0.0,
             j_ref=0.0, dispersion_scale=None, dispersion_angle=0.0, step_scale=1.0):
    """
    2-D WCS of the slot image of one raster step, pixel axes (k, j).

    Intermediate world coordinates (arcsec) of pixel ``(k, j)`` of step ``i``
    are

        C1 (i - i0) + C2 (j - j0) + d (k - k_c(j)),

    with ``C1``/``C2`` the raster-step and slit-row columns of ``CDELT * PC``,
    ``d`` the `dispersion_vector`, and ``k_c(j) = k_center + tilt (j -
    j_ref)`` the slot centre. ``k_center`` replaces ``CRPIX3``: the header
    puts the slot centre at the window centre, the fitted response puts it
    where the slot image actually is. The per-step jitter is added to
    ``CRVAL``, which equals adding it to the world coordinates to within
    0.001 arcsec over a SPICE field of view.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        L2 window header.
    step : int
        0-based raster step.
    k_center : float
        0-based dispersion pixel of the slot centre at ``j_ref``.
    roll_corrected : bool, optional
        See `dispersion_vector`.
    jitter : (float, float), optional
        Pointing correction for this step, arcsec.
    tilt, j_ref : float, optional
        See `SlotResponse`.
    dispersion_scale : float, optional
        Plate scale along the dispersion axis (arcsec/pixel), overriding
        ``|CDELT1 * PC1_3|``.
    dispersion_angle : float, optional
        Rotation (deg) of the dispersion axis away from the raster direction,
        see `dispersion_vector`.
    step_scale : float, optional
        Multiplier of the raster step ``CDELT1`` (cf. ``cdelt1_multiplier`` of
        euispice_coalign), leaving the dispersion plate scale alone.

    Returns
    -------
    `astropy.wcs.WCS`
    """
    c1 = step_scale * np.array([header['CDELT1'] * header['PC1_1'],
                                header['CDELT2'] * header['PC2_1']])
    c2 = np.array([header['CDELT1'] * header['PC1_2'], header['CDELT2'] * header['PC2_2']])
    d = dispersion_vector(header, roll_corrected, dispersion_scale, dispersion_angle)
    i0, j0 = header['CRPIX1'] - 1.0, header['CRPIX2'] - 1.0
    # Linear map from (k - k_center, j - j0) to intermediate coordinates
    m = np.column_stack([d, c2 - tilt * d])
    offset = c1 * (step - i0) - tilt * d * (j0 - j_ref)
    ref = np.array([k_center, j0]) - np.linalg.solve(m, offset)
    hdr = _celestial_header((header['CRVAL1'] + jitter[0], header['CRVAL2'] + jitter[1]),
                            (1.0, 1.0), ref + 1.0, m)
    return WCS(hdr)


def _grid_frame(header, grid, pixel_size, roll_corrected, dispersion_scale=None):
    """
    Orientation (PC) and pixel size (CDELT) of the output grid.
    """
    if grid == 'native':
        # Columns along the raster-step direction, rows along the slit
        c1 = np.array([header['CDELT1'] * header['PC1_1'], header['CDELT2'] * header['PC2_1']])
        c2 = np.array([header['CDELT1'] * header['PC1_2'], header['CDELT2'] * header['PC2_2']])
        pc = np.column_stack([c1 / np.hypot(*c1), c2 / np.hypot(*c2)])
        if pixel_size is None:
            cdelt = (np.hypot(*dispersion_vector(header, roll_corrected, dispersion_scale)),
                     header['CDELT2'])
        else:
            cdelt = (pixel_size, pixel_size) if np.isscalar(pixel_size) else tuple(pixel_size)
    elif grid == 'north':
        pc = np.eye(2)
        size = 1.0 if pixel_size is None else pixel_size
        cdelt = (size, size) if np.isscalar(size) else tuple(size)
    else:
        raise ValueError(f"grid must be 'native' or 'north', not {grid!r}")
    return pc, cdelt


def _output_grid(header, wcs_list, response, grid, pixel_size, roll_corrected,
                 threshold=0.1, margin=2, dispersion_scale=None):
    """
    Header and shape of an output grid covering every slot pixel whose
    response exceeds ``threshold`` (must be below the merge ``min_weight``).
    """
    pc, cdelt = _grid_frame(header, grid, pixel_size, roll_corrected, dispersion_scale)
    crval = (header['CRVAL1'], header['CRVAL2'])
    probe = WCS(_celestial_header(crval, cdelt, (0.0, 0.0), pc))
    resp = response.array()
    jj, kk = np.nonzero(resp > threshold)
    xs, ys = [], []
    for w in wcs_list:
        lon, lat = w.pixel_to_world_values(kk, jj)
        x, y = probe.world_to_pixel_values(lon, lat)
        xs.append(x)
        ys.append(y)
    xs, ys = np.concatenate(xs), np.concatenate(ys)
    x0, x1 = np.floor(xs.min()) - margin, np.ceil(xs.max()) + margin
    y0, y1 = np.floor(ys.min()) - margin, np.ceil(ys.max()) + margin
    shape = (int(y1 - y0) + 1, int(x1 - x0) + 1)
    # probe pixel 0-based p maps to output pixel p - x0; CRPIX (1-based) of
    # the probe was 0.0, i.e. 0-based -1.
    crpix = (-x0, -y0)
    return _celestial_header(crval, cdelt, crpix, pc, shape=shape), shape


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------

def window_tag(window):
    """
    Short filename-safe tag for a window, e.g. 'Ne VIII 770 (Merged)' -> 'NeVIII770'.
    """
    name = re.sub(r'\(.*?\)', '', str(window))
    return re.sub(r'[^A-Za-z0-9]', '', name)


def merged_filename(l2_filename, window):
    """Output filename for one L2 file and window."""
    return f'{Path(l2_filename).stem}_{window_tag(window)}_slotmerge.fits'


def _line_wavelength(header):
    name = re.sub(r'\(.*?\)', '', header['EXTNAME']).strip()
    for key, wvl in LINE_WAVELENGTHS.items():
        if name.lower().startswith(key.lower()):
            return wvl
    return header['CRVAL3'] * 10 ** (header.get('WAVEUNIT', -9) + 10)


def merge_slot_raster(raster, response, grid='native', pixel_size=None, background=True,
                      roll_corrected=True, slot_center='response', min_weight=0.5,
                      uncertainty=True, margin_sigma=2.0, band_width=8, smooth_rows=7,
                      fill_spikes=True, erode=1, dispersion_scale=None, dispersion_angle=0.0,
                      step_scale=1.0):
    """
    Merge the slot images of one raster into a map.

    Every output pixel is projected into the slot image of each raster step
    (`slot_wcs`), where the background-subtracted image ``I`` and the response
    ``R`` -- both zero where there is no data -- are interpolated bilinearly
    and summed over steps; the map is ``sum(I) / sum(R)`` where ``sum(R) >=
    min_weight``.

    Parameters
    ----------
    raster : `SlotRaster` or str
        Slot raster or L2 filename (then read with ``response.window``).
    response : `SlotResponse`
    grid : {'native', 'north'}, optional
        'native' (default) is aligned with the instrument axes -- columns along
        the raster/dispersion direction, rows along the slit -- with the native
        0.98 x 1.098 arcsec sampling, so each slot image maps onto it by a
        sub-pixel shift. 'north' is aligned with Solar north.
    pixel_size : float or (float, float), optional
        Output pixel size in arcsec. Defaults to the native sampling, or 1
        arcsec for 'north'.
    background : bool, optional
        Subtract the out-of-slot pedestal (`fit_background`).
    roll_corrected : bool, optional
        Rotate the dispersion axis with the roll (`dispersion_vector`).
    slot_center : {'response', 'header'}, optional
        Slot centre from the fitted response (default) or the header
        ``CRPIX3`` (the window centre).
    min_weight : float, optional
        Minimum ``sum(R)`` for a valid output pixel. 0.5 puts the field edges
        at the half-maximum of the outermost slot edges.
    uncertainty : bool, optional
        Propagate the `sospice` noise model (`slot_variance`).
    margin_sigma, band_width, smooth_rows : optional
        See `fit_background`.
    fill_spikes, erode : optional
        See `read_slot_raster`.

    Returns
    -------
    `astropy.io.fits.HDUList`
        * primary: merged radiance (``BUNIT``), NaN outside the coverage;
        * ``WEIGHT``: ``sum(R)``, the effective number of exposures per pixel
          (~1 everywhere inside the field when steps equal the slot width);
        * ``UNCERTAINTY``: 1-sigma noise of the primary, from the per-pixel
          variances interpolated with the same weights, a conservative
          (single-native-pixel) estimate;
        * ``TIME``: response-weighted observation time of each pixel in seconds
          after ``DATEREF`` (= ``DATE-BEG``);
        * ``STEPS``: table of raster steps (time, jitter, slot centre).
    """
    if not isinstance(raster, SlotRaster):
        raster = read_slot_raster(raster, response.window, fill_spikes=fill_spikes, erode=erode)
    header = raster.header
    n_steps, ny, nk = raster.data.shape
    if (ny, nk) != tuple(response.shape):
        raise ValueError(f'raster slot images are {(ny, nk)} but the response is '
                         f'{tuple(response.shape)}')
    if response.window != raster.window:
        warnings.warn(f'response fitted on {response.window!r} applied to {raster.window!r}')

    resp = response.array()
    kl_rows, kr_rows = response.k_edges(np.arange(ny))
    images = raster.data.copy()
    if background:
        for i in range(n_steps):
            images[i] -= fit_background(raster.data[i], kl_rows, kr_rows,
                                        margin_sigma * response.sigma_left,
                                        margin_sigma * response.sigma_right,
                                        band_width=band_width, smooth_rows=smooth_rows)
    variance = slot_variance(raster) if uncertainty else None

    if slot_center == 'response':
        k_center = response.k_center
    elif slot_center == 'header':
        k_center = header['CRPIX3'] - 1.0
    else:
        raise ValueError("slot_center must be 'response' or 'header'")
    wcs_list = [slot_wcs(header, i, k_center, roll_corrected=roll_corrected,
                         jitter=raster.jitter[i], tilt=response.tilt, j_ref=response.j_ref,
                         dispersion_scale=dispersion_scale, dispersion_angle=dispersion_angle,
                         step_scale=step_scale)
                for i in range(n_steps)]

    out_header, shape = _output_grid(header, wcs_list, response, grid, pixel_size,
                                     roll_corrected, threshold=min(0.1, 0.5 * min_weight),
                                     dispersion_scale=dispersion_scale)
    out_wcs = WCS(out_header)
    yy, xx = np.indices(shape)
    lon, lat = out_wcs.pixel_to_world_values(xx, yy)

    t_ref = Time(header['DATE-BEG'], scale='utc')
    dt = (raster.times - t_ref).to_value(u.s)
    num = np.zeros(shape)
    den = np.zeros(shape)
    var_sum = np.zeros(shape)
    t_sum = np.zeros(shape)
    step_rows = []
    for i, w in enumerate(wcs_list):
        k_pix, j_pix = w.world_to_pixel_values(lon, lat)
        coords = np.array([j_pix, k_pix])
        valid = np.isfinite(images[i])

        def sample(arr):
            return map_coordinates(np.where(valid, arr, 0.0), coords, order=1,
                                   mode='constant', cval=0.0, prefilter=False)
        weight = sample(resp)
        num += sample(images[i])
        den += weight
        t_sum += weight * dt[i]
        if variance is not None:
            var_sum += sample(variance[i])
        centre = w.pixel_to_world_values(k_center, response.j_ref)
        step_rows.append((i, raster.times[i].isot, dt[i], raster.jitter[i, 0],
                          raster.jitter[i, 1], float(np.mod(centre[0] + 180, 360) - 180) * 3600,
                          float(centre[1]) * 3600))

    good = den >= min_weight
    with np.errstate(invalid='ignore', divide='ignore'):
        merged = np.where(good, num / den, np.nan)
        sigma = np.where(good, np.sqrt(var_sum) / den, np.nan) if variance is not None else None
        tmap = np.where(den > 0, t_sum / den, np.nan)

    # ---- header -----------------------------------------------------------
    hdr = fits.Header()
    for key in _COPY_KEYWORDS:
        if key in header:
            hdr[key] = (header[key], header.comments[key])
    hdr['EXTNAME'] = (raster.window, 'SPICE window')
    hdr['WAVELNTH'] = (_line_wavelength(header), '[Angstrom] Rest wavelength of the main line')
    hdr['WAVEUNIT'] = ('Angstrom', 'Unit of WAVELNTH (L2 WAVEUNIT is a power of 10)')
    hdr.update(out_header)
    hdr['SLOTMRG'] = (True, 'Slot raster merged by spice_slot_merge.py')
    hdr['PARENT'] = (Path(raster.filename).name, 'SPICE L2 file')
    hdr['NSTEPS'] = (n_steps, 'Number of raster steps merged')
    hdr['GRIDTYPE'] = (grid, 'native: instrument-aligned; north: north-up')
    hdr['BKGSUB'] = (bool(background), 'Out-of-slot pedestal subtracted')
    hdr['ROLLDISP'] = (bool(roll_corrected), 'Dispersion axis rotated with the roll')
    disp_scale = np.hypot(*dispersion_vector(header, roll_corrected, dispersion_scale))
    hdr['DISPSCL'] = (float(disp_scale), '[arcsec] Plate scale along the dispersion axis')
    hdr['DISPANG'] = (float(dispersion_angle),
                      '[deg] Dispersion axis rotation from raster direction')
    hdr['STEPSCL'] = (float(step_scale), 'Multiplier applied to the raster step CDELT1')
    hdr['SLOTCEN'] = (slot_center, 'Source of the slot centre (response or header)')
    hdr['SLOTKC'] = (float(k_center), 'Slot centre, 0-based dispersion pixel at SLOTJREF')
    hdr['SLOTJREF'] = (float(response.j_ref), 'Reference row of SLOTKC, 0-based')
    hdr['SLOTWID'] = (float(response.width), '[pixel] Half-max width of the slot image')
    hdr['SLOTSGL'] = (float(response.sigma_left), '[pixel] Gaussian sigma, left slot edge')
    hdr['SLOTSGR'] = (float(response.sigma_right), '[pixel] Gaussian sigma, right slot edge')
    hdr['SLOTTILT'] = (float(response.tilt), '[pixel/row] Drift of slot centre with row')
    # Resolution of the merged map along the instrument axes, arcsec; used to
    # blur reference images to the same resolution for co-alignment.
    hdr['SLOTSGX'] = (float(0.5 * (response.sigma_left + response.sigma_right)
                            * hdr['DISPSCL']), '[arcsec] Blur sigma along the raster axis')
    hdr['SLOTSGY'] = (float(0.5 * (response.sigma_bottom + response.sigma_top)
                            * abs(header['CDELT2'])), '[arcsec] Blur sigma along the slit')
    hdr['MINWGHT'] = (float(min_weight), 'Minimum sum of slot response for valid pixels')
    hdr['NSPIKFIL'] = (raster.n_spikes_filled, 'Despiked pixels restored from SPIKPIXLIST')
    hdr['ORIGCRV1'] = (header['CRVAL1'], '[arcsec] L2 CRVAL1 before any co-alignment')
    hdr['ORIGCRV2'] = (header['CRVAL2'], '[arcsec] L2 CRVAL2 before any co-alignment')
    hdr.add_history('spice_slot_merge.merge_slot_raster: S = sum(I) / sum(R) over raster steps')

    # Every image extension carries the full observer/time/WCS metadata, so
    # that each one is a valid sunpy map on its own.
    def aux_header(name, bunit, btype):
        h = hdr.copy()
        h['EXTNAME'] = name
        h['BUNIT'] = bunit
        h['BTYPE'] = btype
        return h

    hdus = [fits.PrimaryHDU(merged.astype(np.float32), header=hdr)]
    hdus.append(fits.ImageHDU(den.astype(np.float32), name='WEIGHT',
                              header=aux_header('WEIGHT', '', 'Sum of slot response')))
    if sigma is not None:
        hdus.append(fits.ImageHDU(sigma.astype(np.float32), name='UNCERTAINTY',
                                  header=aux_header('UNCERTAINTY', header.get('BUNIT', ''),
                                                    '1-sigma uncertainty')))
    time_hdr = aux_header('TIME', 's', 'Observation time after DATEREF')
    time_hdr['DATEREF'] = (header['DATE-BEG'], '[UTC] Zero point of the time values')
    hdus.append(fits.ImageHDU(tmap.astype(np.float32), header=time_hdr, name='TIME'))
    steps = Table(rows=step_rows, names=('STEP', 'DATE', 'TIME', 'JITTERX', 'JITTERY',
                                         'SLOT_TX', 'SLOT_TY'),
                  dtype=(int, 'U23', float, float, float, float, float))
    steps['TIME'].unit = 's'
    for col in ('JITTERX', 'JITTERY', 'SLOT_TX', 'SLOT_TY'):
        steps[col].unit = 'arcsec'
    hdus.append(fits.table_to_hdu(steps))
    hdus[-1].name = 'STEPS'
    return fits.HDUList(hdus)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_slot_response(response, filename=None):
    """
    Diagnostic figure: empirical vs fitted profiles across and along the slot.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    ny, nk = response.shape
    k = np.arange(nk)
    j = np.arange(ny)
    model_k = _edge_profile(k, response.k_left, response.k_right, response.sigma_left,
                            response.sigma_right)
    ax = axes[0]
    if response.profile_k is not None:
        ax.plot(k, response.profile_k, 'k.', ms=4, label='median of normalised rows')
    ax.plot(k, model_k, 'C3-', label='model')
    for x in (response.k_left, response.k_right):
        ax.axvline(x, color='C3', lw=0.5, ls=':')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set(xlabel='dispersion pixel k', ylabel='normalised radiance',
           title=f'{response.window}: width {response.width:.2f} px')
    ax.legend(fontsize=8)
    ax = axes[1]
    if response.profile_k is not None:
        ax.plot(k, response.profile_k - model_k, 'k.-', ms=3)
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set(xlabel='dispersion pixel k', ylabel='data - model', title='residual across the slot')
    ax = axes[2]
    if response.profile_j is not None:
        ax.plot(j, response.profile_j, 'k-', lw=0.7, label='median row level')
    ax.plot(j, _edge_profile(j, response.j_bottom, response.j_top, response.sigma_bottom,
                             response.sigma_top), 'C3-', label='model')
    ax.set(xlabel='slit row j', ylabel='normalised level',
           title=f'rows {response.j_bottom:.1f}-{response.j_top:.1f}, tilt '
                 f'{response.tilt:.1e} px/row')
    ax.legend(fontsize=8)
    if filename is not None:
        fig.savefig(filename, dpi=110)
        plt.close(fig)
    return fig


def plot_merged(hdul_or_files, filename=None, titles=None):
    """
    Quick-look of one or more merged maps side by side (WCS axes).
    """
    import matplotlib.pyplot as plt
    from astropy.visualization import AsinhStretch, ImageNormalize

    items = hdul_or_files if isinstance(hdul_or_files, (list, tuple)) else [hdul_or_files]
    hduls = [fits.open(x) if isinstance(x, (str, Path)) else x for x in items]
    fig = plt.figure(figsize=(3.6 * len(hduls), 8), layout='constrained')
    for n, hdul in enumerate(hduls):
        data = hdul[0].data
        ax = fig.add_subplot(1, len(hduls), n + 1, projection=WCS(hdul[0].header))
        norm = ImageNormalize(vmin=np.nanpercentile(data, 0.5), vmax=np.nanpercentile(data, 99.7),
                              stretch=AsinhStretch(0.1))
        ax.imshow(data, origin='lower', cmap='magma', norm=norm)
        ax.set_title(titles[n] if titles else hdul[0].header.get('EXTNAME', ''), fontsize=9)
        ax.coords[0].set_axislabel('Solar X')
        ax.coords[1].set_axislabel('Solar Y' if n == 0 else ' ')
        ax.coords.grid(color='w', alpha=0.25, ls=':')
    if filename is not None:
        fig.savefig(filename, dpi=110)
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _expand_inputs(inputs):
    files = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(str(f) for f in p.glob('*.fits')))
        else:
            files.extend(sorted(glob(str(item))) or [str(item)])
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Merge SPICE 30" slot rasters into one map per L2 file and window.')
    parser.add_argument('inputs', nargs='+', help='SPICE L2 wide-slit files, directories or globs')
    parser.add_argument('-o', '--output-dir', default='.', help='Output directory')
    parser.add_argument('-w', '--windows', nargs='*', default=None,
                        help='Windows (EXTNAME or unambiguous part of it); default: all')
    parser.add_argument('--response-dir', default=None,
                        help='Directory of slot_response_<window>.fits to reuse; fitted from '
                             'the inputs and saved to <output-dir>/slot_response if absent')
    parser.add_argument('--refit-response', action='store_true',
                        help='Refit the responses even if saved ones exist')
    parser.add_argument('--grid', choices=('native', 'north'), default='native')
    parser.add_argument('--pixel-size', type=float, nargs='+', default=None,
                        help='Output pixel size in arcsec (one value, or x and y)')
    parser.add_argument('--no-background', dest='background', action='store_false',
                        help='Do not subtract the out-of-slot pedestal')
    parser.add_argument('--header-dispersion', dest='roll_corrected', action='store_false',
                        help='Use the L2 PC1_3/PC2_3 as they are (dispersion along Solar X)')
    parser.add_argument('--header-slot-center', dest='slot_center', action='store_const',
                        const='header', default='response',
                        help='Put the slot centre at CRPIX3 instead of the fitted centre')
    parser.add_argument('-c1', '--step-scale', type=float, default=1.0,
                        help='Multiplier of the raster step CDELT1 (measure it with '
                             'euispice_coalign.fit_slot_step_scale)')
    parser.add_argument('--dispersion-scale', type=float, default=None,
                        help='Plate scale along the dispersion axis, arcsec/pixel '
                             '(default: |CDELT1 * PC1_3| of the header)')
    parser.add_argument('--dispersion-angle', type=float, default=0.0,
                        help='Rotation (deg) of the dispersion axis away from the raster direction')
    parser.add_argument('--min-weight', type=float, default=0.5)
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Save a quick-look PNG per input file')
    args = parser.parse_args(argv)

    files = _expand_inputs(args.inputs)
    if not files:
        parser.error('no input files')
    windows = list_slot_windows(files[0])
    if args.windows:
        with fits.open(files[0]) as hdul:
            windows = [_find_window_hdu(hdul, w).header['EXTNAME'] for w in args.windows]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    resp_dir = Path(args.response_dir) if args.response_dir else out_dir / 'slot_response'
    resp_dir.mkdir(parents=True, exist_ok=True)
    pixel_size = None
    if args.pixel_size:
        pixel_size = args.pixel_size[0] if len(args.pixel_size) == 1 else tuple(args.pixel_size[:2])

    outputs = {f: [] for f in files}
    for window in windows:
        resp_file = resp_dir / f'slot_response_{window_tag(window)}.fits'
        if resp_file.exists() and not args.refit_response:
            response = SlotResponse.from_fits(resp_file)
            print(f'[{window}] using {resp_file}')
        else:
            response = build_slot_response(files, window)
            response.to_fits(resp_file)
            plot_slot_response(response, resp_file.with_suffix('.png'))
            print(f'[{window}] saved {resp_file}')
        for f in files:
            hdul = merge_slot_raster(f, response, grid=args.grid, pixel_size=pixel_size,
                                     background=args.background,
                                     roll_corrected=args.roll_corrected,
                                     slot_center=args.slot_center, min_weight=args.min_weight,
                                     step_scale=args.step_scale,
                                     dispersion_scale=args.dispersion_scale,
                                     dispersion_angle=args.dispersion_angle)
            out = out_dir / merged_filename(f, window)
            hdul.writeto(out, overwrite=True)
            outputs[f].append(out)
        print(f'[{window}] merged {len(files)} files into {out_dir}')

    if args.plot:
        for f, outs in outputs.items():
            plot_merged([str(o) for o in outs], out_dir / f'{Path(f).stem}_slotmerge.png')


if __name__ == '__main__':
    main()
