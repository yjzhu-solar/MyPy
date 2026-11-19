r"""
Co-align Solar Orbiter/SPICE rasters to near-simultaneous EUI images.

A SPICE raster is built up one slit position at a time over a few hours, so it
has no single exposure time and no single observer position. This module
synthesises what EUI would have seen *on the SPICE pixel grid* -- sampling, for
each raster column, the EUI frame closest in time and differentially rotating
to that column's observation time -- then cross-correlates that synthetic
raster against the SPICE line intensity map to solve for the SPICE pointing
offset, and writes a corrected copy of the SPICE file.

Conventions used throughout
---------------------------
* Arrays are ``(ny, nx)`` = (along-slit, raster-step); ``ii`` always indexes the
  raster step / column, which is also the time axis.
* Pixel coordinates are 0-based (astropy ``world_to_pixel`` convention), not the
  1-based FITS ``CRPIX`` convention.
* The fitted shift is a correction to be *added* to the SPICE ``CRVAL``.

Command-line examples
---------------------
The second argument is a directory of EUI images or a glob pattern. The
corrected SPICE file is always written before any plotting, so a closed figure
window can never cost you the result.

Default run: 11 trial roll angles, per-column observer and time, no plot. The
output goes to ``<input stem>_coalign.fits`` next to the input::

    python euispice_coalign.py spice.fits /data/eui/ -w 'Ne VIII 770 - Peak'

Choose the output name and directory::

    python euispice_coalign.py spice.fits /data/eui/ \
        -w 'Ne VIII 770 - Peak' -o /work/coaligned -s my_raster_coalign.fits

Translation only, skipping the roll search. Builds 1 synthetic raster instead
of 11, so roughly an order of magnitude faster::

    python euispice_coalign.py spice.fits /data/eui/ -nr

Reuse the synthetic raster from an earlier run instead of rebuilding it. The
.npz records its own trial angles, so ``-nr`` does not need repeating here::

    python euispice_coalign.py spice.fits /data/eui/ \
        -sf /data/eui/eui_syn_raster_image_for_spice.npz

Blink the synthetic raster against the corrected SPICE map to eyeball the fit.
Blocks until the window is closed. The frame titles are built from the data
(EUI channel, fitted roll, SPICE window), or can be set explicitly::

    python euispice_coalign.py spice.fits /data/eui/ -p
    python euispice_coalign.py spice.fits /data/eui/ -p \
        -t1 'EUI 174 synthetic' -t2 'SPICE Ne VIII'

Treat the raster as instantaneous, i.e. one observer position and one time for
the whole scan. Faster, and a useful check on how much the per-column
treatment is actually buying you::

    python euispice_coalign.py spice.fits /data/eui/ -nsr

Correct a known error in the raster step size while co-aligning. This scales
CDELT1 and compensates the PC matrix so that only the step direction changes::

    python euispice_coalign.py spice.fits /data/eui/ -c1 1.03

Select the EUI files with a glob rather than a directory. Quote it so the
shell does not expand it first::

    python euispice_coalign.py spice.fits '/data/eui/*fsi174*.fits'

Everything together, as a batch job::

    python euispice_coalign.py spice.fits '/data/eui/*fsi174*.fits' \
        -w 'Ne VIII 770 - Peak' -c1 1.03 \
        -o /work/coaligned -s my_raster_coalign.fits -p

Python examples
---------------
The two stages are separate functions so the expensive one can be cached::

    from glob import glob
    from euispice_coalign import create_syn_rasters, calculate_eui_spice_shift

    eui_files = sorted(glob('/data/eui/*.fits'))

    syn = create_syn_rasters('spice.fits', eui_files, 'Ne VIII 770 - Peak',
                             save_filename='syn.npz')
    dx, dy, pc, roll = calculate_eui_spice_shift(
        'spice.fits', 'Ne VIII 770 - Peak', syn,
        save_filename='spice_coalign.fits', output_dir='/work')
    print(dx, dy, roll)          # arcsec, arcsec, degrees

Pass the .npz path instead of the array to skip the rebuild entirely::

    dx, dy, pc, roll = calculate_eui_spice_shift(
        'spice.fits', 'Ne VIII 770 - Peak', 'syn.npz')

Unlike ``-w`` on the command line, the Python API also takes a window index::

    syn = create_syn_rasters('spice.fits', eui_files, 3)   # 4th window

Fit the translation only, and show the blink comparison::

    syn = create_syn_rasters('spice.fits', eui_files, 'Ne VIII 770 - Peak',
                             rotation=False)
    dx, dy, pc, roll = calculate_eui_spice_shift(
        'spice.fits', 'Ne VIII 770 - Peak', syn, rotation=False, plot=True)

``rotation`` must agree between the two calls; a mismatch raises rather than
silently reporting the wrong roll. Passing a .npz makes that impossible, since
the file carries its own angles.

Merged slot rasters
-------------------
Files written by ``spice_slot_merge.py`` (30 arcsec slot rasters merged into
one map per window, header ``SLOTMRG = T``) are recognised automatically.
Each pixel is sampled in the EUI frame closest to the time of its own raster
step, with differential rotation to that frame, and EUI is blurred to the
SPICE resolution (``--no-blur`` to disable). ``-w`` and ``-c1`` do not apply;
the raster step is corrected when merging::

    python euispice_coalign.py merged_NeVIII770_slotmerge.fits '/data/fsi174/*.fits' -p

In Python, many files of one window can share one cached EUI frame set::

    from euispice_coalign import coalign_slot_files, fit_slot_step_scale
    table = coalign_slot_files(sorted(glob('merged/*_CIII977_slotmerge.fits')),
                               sorted(glob('/data/fsi304/*.fits')),
                               output_dir='coaligned')

and the raster-step multiplier can be measured from the agreement of the two
halves of a raster::

    best, table = fit_slot_step_scale(l2_file, 'C III 977', response,
                                      eui_files=sorted(glob('/data/fsi304/*.fits')))

Notes
-----
As a rough guide, a 192 x 832 raster against 35 EUI FSI frames takes about
15 s for a single angle and about two minutes for the full 11-angle sweep.
EUI frames are read on demand with a small cache (see `_EUIFrameSet`) rather
than all held in memory.
"""

import argparse
import os
import warnings
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import ImageNormalize, AsinhStretch
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from skimage.feature import match_template
from skimage.registration import phase_cross_correlation
import sunpy.map
from sunpy.coordinates import propagate_with_solar_surface, get_horizons_coord
from sunraster.instr.spice import read_spice_l2_fits
from sun_blinker import SunBlinker

# Private sunkit_image helper: sub-pixel parabolic refinement of the
# cross-correlation peak. It moved out of `sunkit_image.coalignment` into this
# submodule in sunkit_image 0.6/0.7 and now returns plain floats rather than
# Quantities in pix, so pin sunkit_image if this import ever breaks.
from sunkit_image.coalignment.match_template import _find_best_match_location


#: Trial roll angles in radians, searched when ``rotation=True``. The +/-1 deg
#: span is a convenience range around the nominal SPICE roll, not a physical
#: bound on the pointing error. Defined once so that layer ``i`` of a saved
#: synthetic-raster file always means the same angle; the angles are also
#: written into the .npz so a cached file can never be misinterpreted.
TRIAL_ROLL_ANGLES = np.deg2rad(np.linspace(-1.0, 1.0, 11))


def _select_window(spice_dataset, spice_window):
    """
    Resolve a window specifier to a SPICE window.

    Parameters
    ----------
    spice_dataset : dict-like
        Output of `sunraster.instr.spice.read_spice_l2_fits`.
    spice_window : int or str
        Index into, or ``EXTNAME`` of, the window to use.

    Returns
    -------
    `sunraster.spectrogram.SpectrogramCube`
    """
    keys = list(spice_dataset.keys())
    if isinstance(spice_window, (int, np.integer)):
        try:
            return spice_dataset[keys[spice_window]]
        except IndexError:
            raise IndexError(f'window index {spice_window} is out of range; the file '
                             f'has {len(keys)} windows: {keys}') from None
    if isinstance(spice_window, str):
        try:
            return spice_dataset[spice_window]
        except KeyError:
            raise KeyError(f'{spice_window} not found in the SPICE file. '
                           f'Available windows are {keys}') from None
    raise ValueError('spice_window must be either an int or a string')


def _prepare_spice_wcs(spice_window, cdelt1_multiplier=1):
    """
    Return the spatial+time SPICE WCS, with the raster step size rescaled.

    The wavelength axis (WCS axis 3) is dropped. The time axis is kept, because
    ``PC4_1`` ties observation time to raster step and dropping it would discard
    that relationship.

    ``cdelt1_multiplier`` must rescale only the raster-step *column* of the CD
    matrix. ``CDELT1`` scales both ``CD[0,0]`` and ``CD[0,1]``, and ``CDELT2``
    scales ``CD[1,0]``, so ``PC1_2`` is divided and ``PC2_1`` multiplied to
    leave everything but ``CD[:, 0]`` untouched.

    Parameters
    ----------
    spice_window : `sunraster.spectrogram.SpectrogramCube`
    cdelt1_multiplier : float, optional

    Returns
    -------
    `astropy.wcs.WCS`
        3-axis: (helioprojective longitude, latitude, time).
    """
    spice_wcs = spice_window.wcs.dropaxis(2)
    if spice_wcs.wcs.has_cd():
        raise NotImplementedError(
            'this SPICE file uses a CDi_j matrix; astropy then ignores writes to '
            'wcs.cdelt and wcs.pc, so cdelt1_multiplier and the roll search would '
            'be silently dropped. Convert the header to the PC + CDELT form first.')
    if cdelt1_multiplier != 1:
        spice_wcs.wcs.cdelt[0] = spice_wcs.wcs.cdelt[0] * cdelt1_multiplier
        spice_wcs.wcs.pc[0, 1] = spice_wcs.wcs.pc[0, 1] / cdelt1_multiplier
        spice_wcs.wcs.pc[1, 0] = spice_wcs.wcs.pc[1, 0] * cdelt1_multiplier
    return spice_wcs


def _trial_roll_angles(rotation):
    """
    Trial roll angles in radians: the full grid, or a single zero angle.
    """
    return TRIAL_ROLL_ANGLES if rotation else np.zeros(1)


def _roll_matrix(rot_angle, spice_wcs):
    """
    Matrix to post-multiply into the SPICE ``PC`` matrix for a trial roll.

    The off-diagonal terms carry the ``CDELT2``/``CDELT1`` ratio because SPICE
    pixels are strongly non-square (4.0 x 1.098 arcsec for a typical raster):
    a rotation in world space is not a rotation in pixel space.

    Parameters
    ----------
    rot_angle : float
        Roll angle in radians.
    spice_wcs : `astropy.wcs.WCS`
        Used only for its ``CDELT`` ratio, so it must already have any
        ``cdelt1_multiplier`` applied.

    Returns
    -------
    `numpy.ndarray`
        Shape ``(2, 2)``.
    """
    cdelt1, cdelt2 = spice_wcs.wcs.cdelt[:2]
    return np.array([[np.cos(rot_angle), -np.sin(rot_angle) * cdelt2 / cdelt1],
                     [np.sin(rot_angle) * cdelt1 / cdelt2, np.cos(rot_angle)]])


def _spacecraft_track(spice_time_obs):
    """
    Solar Orbiter's position at each SPICE raster step.

    Parameters
    ----------
    spice_time_obs : `astropy.time.Time`
        Observation time of each raster column, in raster-step order. This may
        run backwards in time: the scan direction is set by the sign of
        ``PC4_1`` and both directions occur.

    Returns
    -------
    `astropy.coordinates.SkyCoord`
        One position per raster step, in the same order as ``spice_time_obs``.
    """
    n_steps = len(spice_time_obs)
    # Horizons requires start < stop, so query in time order and un-flip after.
    reverse = spice_time_obs[-1] < spice_time_obs[0]
    t_start, t_stop = ((spice_time_obs[-1], spice_time_obs[0]) if reverse
                       else (spice_time_obs[0], spice_time_obs[-1]))
    # Horizons' 'step' counts *intervals*, not epochs, so n-1 returns n points
    # at exactly the raster cadence. Passing n returns n+1 points at a slightly
    # short cadence, which drifts by a full step across the raster.
    track = get_horizons_coord('solar orbiter',
                               {'start': t_start, 'stop': t_stop, 'step': f'{n_steps - 1}'})
    return np.flip(track) if reverse else track


def _spice_intensity_map(spice_window):
    """
    Wavelength-integrated intensity for a SPICE window.

    Parameters
    ----------
    spice_window : `sunraster.spectrogram.SpectrogramCube`

    Returns
    -------
    `numpy.ndarray`
        Shape ``(ny, nx)``.

    Notes
    -----
    `numpy.nansum` returns 0 for an all-NaN spectrum -- the SPICE dumbbells and
    off-detector regions -- which is indistinguishable from a genuinely dark
    pixel and would quietly bias the cross-correlation if such a pixel ever fell
    inside the template. Those pixels are restored to NaN.
    """
    data = spice_window.data
    intensity = np.nansum(data, axis=(0, 1))
    intensity[np.all(np.isnan(data), axis=(0, 1))] = np.nan
    return intensity


class _EUIFrameSet:
    """
    Lazy, bounded-memory access to a set of EUI image files.

    Loading every frame up front costs ~36 MB each (3072 x 3040 float32), i.e.
    over a gigabyte for a full-day FSI sequence, and rebuilding an interpolator
    per raster column repeats that work ~200 times per trial roll angle. Frames
    are instead read on demand and the most recently used ones kept together
    with their interpolators. Because the raster is swept in time order and
    EUI's cadence (~10 min) is much coarser than the SPICE slit step (~60 s),
    consecutive columns almost always reuse a cached frame.

    Parameters
    ----------
    eui_files : list of str
    max_cached : int, optional
        Number of decoded frames to keep. Set to ``len(eui_files)`` to hold the
        whole sequence in memory and never re-read a file.
    blur : (float, float), optional
        Gaussian sigma in arcsec along the image x and y axes, applied to each
        frame (NaN-aware) before its interpolator is built, to bring EUI to
        the resolution of the SPICE data it is compared with. ``None`` (the
        default) leaves the frames untouched.

    Attributes
    ----------
    time_obs : `astropy.time.Time`
        Exposure midpoint of each frame.
    rsun_ref : `astropy.units.Quantity`
        ``RSUN_REF`` of the first frame, used as the ``rsun`` of the
        helioprojective frames built against these images.
    label : str
        Channel(s) in the set, e.g. ``'EUI/FSI 174'``, for plot titles.
    pixel_scale : float
        ``CDELT1`` of the first frame in arcsec.
    """

    def __init__(self, eui_files, max_cached=4, blur=None):
        self.files = list(eui_files)
        if not self.files:
            raise ValueError('no EUI files supplied')
        # Headers only; this does not decompress the image extensions.
        headers = [fits.getheader(f, ext=1) for f in self.files]
        # DATE-AVG is the exposure midpoint. sunpy's Map.date would return
        # DATE-OBS, the *start* of the exposure (5 s earlier for a 10 s FSI
        # exposure).
        self.time_obs = Time([h.get('DATE-AVG', h['DATE-OBS']) for h in headers])
        self.rsun_ref = headers[0]['RSUN_REF'] * u.m
        channels = sorted({f"{h.get('DETECTOR', '?')} {h.get('WAVELNTH', '?')}" for h in headers})
        self.label = 'EUI/' + ' + '.join(channels)
        self.pixel_scale = abs(headers[0]['CDELT1']) * u.Unit(headers[0].get('CUNIT1', 'arcsec')
                                                              ).to(u.arcsec)
        self.blur = None if blur is None else tuple(float(b) for b in blur)
        self._max_cached = max(1, int(max_cached))
        self._cache = OrderedDict()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        """
        Return ``(map, interpolator)`` for frame ``index``.

        The interpolator is defined on a 0-based ``(row, col)`` grid, matching
        the output of `astropy.wcs.WCS.world_to_pixel`, and returns NaN outside
        the frame.
        """
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        eui_map = sunpy.map.Map(self.files[index])
        data = eui_map.data
        if self.blur is not None and max(self.blur) > 0:
            scale = (eui_map.scale.axis1.to_value(u.arcsec / u.pix),
                     eui_map.scale.axis2.to_value(u.arcsec / u.pix))
            data = _nan_gaussian(data, (self.blur[1] / scale[1], self.blur[0] / scale[0]))
        interpolator = RegularGridInterpolator(
            (np.arange(data.shape[0]), np.arange(data.shape[1])),
            data, bounds_error=False, method='linear')
        self._cache[index] = (eui_map, interpolator)
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return self._cache[index]


def create_syn_rasters(spice_file, eui_files, spice_window,
                       save_filename=None, rotation=True,
                       solar_rotation=True, cdelt1_multiplier=1):
    """
    Build synthetic EUI rasters sampled onto the SPICE pixel grid.

    Parameters
    ----------
    spice_file : str or `pathlib.Path`
        Path to a SPICE L2 raster FITS file.
    eui_files : list of str
        EUI image files spanning the SPICE raster's time range. Need not be
        sorted; the closest frame in time is chosen per raster column.
    spice_window : int or str
        Index into, or ``EXTNAME`` of, the SPICE window to use.
    save_filename : str, optional
        If given, the synthetic rasters and their trial roll angles are written
        here as a ``.npz``.
    rotation : bool, optional
        If True, produce one synthetic raster per angle in `TRIAL_ROLL_ANGLES`
        so a roll offset can be fitted alongside the translation. If False,
        produce a single raster at the nominal roll.
    solar_rotation : bool, optional
        If True, each raster column is assigned its own observation time and
        observer position. If False, all columns use the values of the first
        column, i.e. the raster is treated as instantaneous.
    cdelt1_multiplier : float, optional
        Correction factor applied to the SPICE raster step size ``CDELT1``.

    Returns
    -------
    `numpy.ndarray`
        Shape ``(n_angles, ny, nx)``: EUI intensity on the SPICE grid.
    """
    spice_dataset = read_spice_l2_fits(spice_file)

    # Measured per-slit-position pointing drift, one value per raster step, in
    # the units of CUNIT1/CUNIT2 (arcsec). EXTVER=1 is Solar X, EXTVER=2 Solar Y.
    with fits.open(spice_file) as hduls:
        spice_solarx_shift = hduls['WCSDVARR', 1].data.copy()
        spice_solary_shift = hduls['WCSDVARR', 2].data.copy()

    spice_window = _select_window(spice_dataset, spice_window)
    spice_wcs = _prepare_spice_wcs(spice_window, cdelt1_multiplier)

    spice_ny, spice_nx = (int(n) for n in spice_window.shape[-2:])
    spice_time_obs = spice_window.time[0]
    solar_orbiter_loc = _spacecraft_track(spice_time_obs)

    eui_frames = _EUIFrameSet(eui_files)

    rot_angles = _trial_roll_angles(rotation)
    eui_syn_raster_images = np.zeros((len(rot_angles), spice_ny, spice_nx))

    for ii, rot_angle in enumerate(rot_angles):
        spice_wcs_rotated = deepcopy(spice_wcs)
        spice_wcs_rotated.wcs.pc[:2, :2] = np.dot(spice_wcs.wcs.pc[:2, :2],
                                                  _roll_matrix(rot_angle, spice_wcs))
        eui_syn_raster_images[ii] = make_single_syn_raster(
            spice_wcs_rotated, (spice_ny, spice_nx), spice_time_obs,
            solar_orbiter_loc, eui_frames, spice_solarx_shift, spice_solary_shift,
            solar_rotation=solar_rotation)

    if save_filename is not None:
        # Store the angles alongside the images so a cached file can never be
        # paired with a different trial grid; the channel label is for titles.
        np.savez_compressed(save_filename,
                            eui_syn_raster_images=eui_syn_raster_images,
                            rot_angles=rot_angles,
                            eui_label=eui_frames.label)

    return eui_syn_raster_images


def make_single_syn_raster(spice_wcs, shape, spice_time_obs, solar_orbiter_loc,
                           eui_frames, spice_solarx_shift, spice_solary_shift,
                           solar_rotation=True):
    """
    Sample EUI onto the SPICE grid for one fixed trial WCS.

    Parameters
    ----------
    spice_wcs : `astropy.wcs.WCS`
        3-axis (lon, lat, time) SPICE WCS, wavelength already dropped and any
        trial roll already folded into ``PC``.
    shape : tuple of int
        ``(ny, nx)`` of the SPICE raster.
    spice_time_obs : `astropy.time.Time`
        Observation time of each raster column, length ``nx``.
    solar_orbiter_loc : `astropy.coordinates.SkyCoord`
        Spacecraft position at each raster column, length ``nx``, in the same
        order as ``spice_time_obs``.
    eui_frames : `_EUIFrameSet`
    spice_solarx_shift, spice_solary_shift : `numpy.ndarray`
        Per-column pointing corrections from the ``WCSDVARR`` extensions, in
        arcsec, length ``nx``.
    solar_rotation : bool, optional
        See `create_syn_rasters`.

    Returns
    -------
    `numpy.ndarray`
        Shape ``(ny, nx)``. NaN wherever the SPICE grid falls outside the EUI
        frame; a warning is issued if that happens.
    """
    spice_ny, spice_nx = shape
    eui_syn_raster_image = np.zeros((spice_ny, spice_nx))

    spice_pixy, spice_pixx, spice_pixt = np.indices((spice_ny, spice_nx, 1))

    # Rough because this WCS gives every pixel the same global obstime
    # (DATE-AVG) and no observer; both are replaced per column below. The time
    # pixel is held at 0 -- PC1_4 = PC2_4 = 0, so Tx/Ty do not depend on it.
    spice_skycoord_rough = spice_wcs.pixel_to_world(spice_pixx, spice_pixy,
                                                    spice_pixt)[0][:, :, 0]

    for ii in range(spice_nx):
        # SPICE records the measured pointing drift as a SOLARNET 'Lookup'
        # distortion of the *world* coordinates (CWDISi/CWERRi). astropy does
        # not implement that convention -- wcs.cpdis1 is None -- so
        # pixel_to_world above returns undistorted coordinates, and the DPDD
        # (SPICE-UIO-DPDD-0002 issue 2.2, p. 47) instructs Python users to add
        # element ii of each WCSDVARR array by hand, which is what happens here.
        #
        # Do not flip this to a minus sign. gpelouze/spice_jitter_correction
        # subtracts WCSDVARR, but on real data that doubles the jitter rather
        # than removing it: in solo_L2_spice-n-ras_20210914T025031_V24_67109159
        # (the DPDD's own jitter example) the minus sign turns the ~+/-13"
        # Solar Y wobble at the right of the raster into a ~+/-25-30" zigzag,
        # while the plus sign straightens it. The plus sign also correlates
        # better with EUI on the 20221024T231535 V22 raster.
        obs_index = ii if solar_rotation else 0
        spice_skycoord = SkyCoord(
            spice_skycoord_rough[:, ii].Tx.to(u.arcsec) + spice_solarx_shift[ii] * u.arcsec,
            spice_skycoord_rough[:, ii].Ty.to(u.arcsec) + spice_solary_shift[ii] * u.arcsec,
            frame='helioprojective',
            obstime=spice_time_obs[obs_index],
            observer=solar_orbiter_loc[obs_index],
            rsun=eui_frames.rsun_ref)

        eui_map_index = find_closest_frame(spice_time_obs[ii], eui_frames.time_obs)
        eui_map, eui_map_interpolator = eui_frames[eui_map_index]

        # Differentially rotate this column's coordinates to the EUI frame's own
        # observation time; 'rigid' avoids latitude-dependent shear across the
        # FOV. Frame-matching leaves up to half an EUI cadence of drift to undo.
        with propagate_with_solar_surface(rotation_model='rigid'):
            spice_skycoord_pixel = eui_map.wcs.world_to_pixel(spice_skycoord)

        # world_to_pixel returns (x, y); the interpolator grid is indexed
        # (row, col), hence the swap. Both are 0-based.
        eui_syn_raster_image[:, ii] = eui_map_interpolator(
            (spice_skycoord_pixel[1], spice_skycoord_pixel[0]))

    n_bad = np.count_nonzero(~np.isfinite(eui_syn_raster_image))
    if n_bad:
        warnings.warn(
            f'{n_bad} of {eui_syn_raster_image.size} synthetic-raster pixels fall '
            'outside the EUI frames and are NaN. match_template cannot handle '
            'these and will return a spurious corner match.', stacklevel=2)

    return eui_syn_raster_image


def calculate_eui_spice_shift(spice_file, spice_window, eui_syn_raster_images,
                              rotation=True, cdelt1_multiplier=1, save_filename=None,
                              output_dir=None, plot=False, title1=None, title2=None):
    """
    Fit the SPICE pointing offset against a synthetic EUI raster.

    The central quarter of the SPICE intensity map is used as a template and
    located within each synthetic raster by normalised cross-correlation. The
    trial roll angle with the highest correlation peak wins. A corrected copy of
    the SPICE file is written before any plotting, so the result survives a
    closed figure window.

    Parameters
    ----------
    spice_file : str or `pathlib.Path`
    spice_window : int or str
    eui_syn_raster_images : `numpy.ndarray` or str
        Output of `create_syn_rasters`, or a path to the saved ``.npz``. Its
        leading dimension must match the number of trial angles.
    rotation : bool, optional
        Must match the value used to build ``eui_syn_raster_images``. Ignored
        when loading a ``.npz`` that records its own trial angles.
    cdelt1_multiplier : float, optional
    save_filename : str, optional
        Filename for the corrected SPICE FITS file.
    output_dir : str, optional
    plot : bool, optional
        Show a blink comparison of the synthetic raster against the corrected
        SPICE intensity map. Blocks until the window is closed.
    title1, title2 : str, optional
        Titles for the synthetic-raster and SPICE frames of the blink
        comparison. By default they are built from the data: the EUI channel
        (known only when ``eui_syn_raster_images`` is a ``.npz`` path, else
        just 'EUI') and fitted roll, and the SPICE window name.

    Returns
    -------
    xshift, yshift : `astropy.units.Quantity`
        Helioprojective correction to *add* to the SPICE ``CRVAL1``/``CRVAL2``.
    rot_matrix : `numpy.ndarray`
        2x2 matrix post-multiplied into the SPICE ``PC`` matrix.
    rot_angle : float
        The corresponding roll correction, in degrees.
    """
    spice_dataset = read_spice_l2_fits(spice_file)
    spice_window = _select_window(spice_dataset, spice_window)
    spice_wcs = _prepare_spice_wcs(spice_window, cdelt1_multiplier)

    rot_angles = None
    eui_label = 'EUI'
    if isinstance(eui_syn_raster_images, (str, Path)):
        with np.load(eui_syn_raster_images) as npz:
            if 'rot_angles' in npz.files:
                rot_angles = npz['rot_angles']
            if 'eui_label' in npz.files:
                eui_label = str(npz['eui_label'])
            eui_syn_raster_images = npz['eui_syn_raster_images']
    if rot_angles is None:
        rot_angles = _trial_roll_angles(rotation)

    # Layer i of eui_syn_raster_images means rot_angles[i]. If the two disagree
    # the fit would report the wrong roll, or index off the end below.
    if len(rot_angles) != len(eui_syn_raster_images):
        raise ValueError(
            f'{len(eui_syn_raster_images)} synthetic rasters supplied but '
            f'{len(rot_angles)} trial roll angles (rotation={rotation}); both must '
            'be produced with the same setting.')

    spice_ny, spice_nx = (int(n) for n in spice_window.shape[-2:])
    spice_int_img = _spice_intensity_map(spice_window)

    # Central quarter: keeps the template clear of the dumbbells and of the
    # raster edges, and leaves room for the template to be found off-centre.
    spice_int_img_cut = spice_int_img[spice_ny // 4:3 * spice_ny // 4,
                                      spice_nx // 4:3 * spice_nx // 4]

    rotation_matrices = [_roll_matrix(rot_angle, spice_wcs) for rot_angle in rot_angles]

    yshifts = []
    xshifts = []
    max_ccs = []

    for syn_image in eui_syn_raster_images:
        xshift, yshift, max_cc = coalign_shift_pixel(syn_image, spice_int_img_cut)
        yshifts.append(yshift)
        xshifts.append(xshift)
        max_ccs.append(max_cc)

    max_cc_index = int(np.argmax(max_ccs))
    yshift_optimal, xshift_optimal, rot_matrix_optimal, rot_angle_optimal = \
        yshifts[max_cc_index], xshifts[max_cc_index], \
        rotation_matrices[max_cc_index], rot_angles[max_cc_index]

    spice_wcs_optimal = deepcopy(spice_wcs)
    spice_wcs_optimal.wcs.pc[:2, :2] = np.dot(spice_wcs.wcs.pc[:2, :2], rot_matrix_optimal)

    # The template was cut from SPICE at (ny//4, nx//4). match_template reports
    # where it actually sits in the synthetic (correctly-pointed) raster, so the
    # correction is world(found) - world(nominal), to be *added* to CRVAL.
    # Differencing in world rather than pixel space keeps the raster's PC shear
    # and the TAN projection correct.
    shift_reference_world_coord = spice_wcs_optimal.pixel_to_world(xshift_optimal,
                                                                   yshift_optimal, 0)[0]
    reference_pixel_world_coord = spice_wcs_optimal.pixel_to_world(spice_nx // 4,
                                                                   spice_ny // 4, 0)[0]

    xshift_optimal_world = shift_reference_world_coord.Tx - reference_pixel_world_coord.Tx
    yshift_optimal_world = shift_reference_world_coord.Ty - reference_pixel_world_coord.Ty

    spice_int_map = sunpy.map.Map(spice_int_img, spice_wcs_optimal)
    spice_int_map = spice_int_map.shift_reference_coord(xshift_optimal_world,
                                                        yshift_optimal_world)
    new_crval1, new_crval2 = (spice_int_map.reference_coordinate.Tx,
                              spice_int_map.reference_coordinate.Ty)
    new_rotation_matrix = spice_int_map.rotation_matrix

    save_new_spice_file(spice_file, new_crval1, new_crval2, new_rotation_matrix,
                        cdelt1_multiplier=cdelt1_multiplier,
                        outdir=output_dir, filename=save_filename)

    if plot:
        # The winning layer, used for both the image and its stretch. Reading the
        # loop variable above instead would silently scale to the *last* trial
        # roll angle rather than the fitted one.
        eui_syn_raster_best = eui_syn_raster_images[max_cc_index]
        eui_syn_raster_map = sunpy.map.Map(eui_syn_raster_best, spice_wcs_optimal)
        # SPICE pixels are strongly non-square (4.0 x 1.098 arcsec here); force
        # the display aspect so the blink comparison is not misleading.
        eui_syn_raster_map.plot_settings['aspect'] = \
            eui_syn_raster_map.scale.axis2 / eui_syn_raster_map.scale.axis1
        for key in ('CROTA1', 'CROTA2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
            spice_int_map.meta.pop(key, None)
        if title1 is None:
            title1 = f'{eui_label} synthetic raster'
            if len(rot_angles) > 1:
                title1 += f'\nroll {np.rad2deg(rot_angle_optimal):+.2f}\N{DEGREE SIGN}'
        if title2 is None:
            window_name = spice_window.meta.get('EXTNAME', 'SPICE window')
            title2 = f'SPICE {window_name}\nco-aligned'
        SunBlinker(eui_syn_raster_map, spice_int_map, reproject=True, fps=1,
                   title1=title1, title2=title2,
                   norm1=ImageNormalize(vmin=np.nanpercentile(eui_syn_raster_best, 0.2),
                                        vmax=np.nanpercentile(eui_syn_raster_best, 99.8),
                                        stretch=AsinhStretch(0.1)),
                   norm2=ImageNormalize(vmin=np.nanpercentile(spice_int_img, 0.2),
                                        vmax=np.nanpercentile(spice_int_img, 99.8),
                                        stretch=AsinhStretch(0.1)),)
        plt.show()

    return (xshift_optimal_world, yshift_optimal_world, rot_matrix_optimal,
            np.rad2deg(rot_angle_optimal))


def save_new_spice_file(spice_file, crval1, crval2, rotation_matrix, cdelt1_multiplier,
                        outdir=None, filename=None):
    """
    Write a copy of the SPICE file with corrected pointing keywords.

    Pixel data is copied unchanged; only the spatial WCS keywords of the science
    windows are rewritten.

    Parameters
    ----------
    spice_file : str or `pathlib.Path`
    crval1, crval2 : `astropy.units.Quantity`
        The corrected reference coordinate.
    rotation_matrix : `numpy.ndarray`
        The corrected 2x2 ``PC`` matrix.
    cdelt1_multiplier : float
    outdir : str or `pathlib.Path`, optional
        Defaults to the directory of ``spice_file``.
    filename : str, optional
        Defaults to the input stem with ``_coalign.fits`` appended.
    """
    with fits.open(spice_file) as hduls:
        for hdul in hduls:
            # Science windows only. The WCSDVARR distortion extensions also
            # carry CRVAL1/CRPIX1/CDELT1, but as the lookup array's own index
            # mapping -- rewriting those destroys the pointing correction.
            if hdul.header.get('CTYPE1', '').startswith('HPLN'):
                hdul.header = update_header(hdul.header, crval1, crval2,
                                            rotation_matrix, cdelt1_multiplier)
        outdir = Path(spice_file).parent if outdir is None else Path(outdir)
        if filename is None:
            filename = Path(spice_file).stem + '_coalign.fits'
        hduls.writeto(outdir / filename, overwrite=True)


def update_header(hdr, crval1, crval2, rotation_matrix, cdelt1_multiplier):
    """
    Return a copy of ``hdr`` with the co-aligned pointing solution applied.

    Parameters
    ----------
    hdr : `astropy.io.fits.Header`
    crval1, crval2 : `astropy.units.Quantity`
    rotation_matrix : `numpy.ndarray`
        Shape ``(2, 2)``.
    cdelt1_multiplier : float

    Returns
    -------
    `astropy.io.fits.Header`
    """
    hdr_new = hdr.copy()
    # Write in whatever units the header declares rather than assuming arcsec.
    # SPICE L2 uses CUNIT1 = 'arcsec', but sunraster hands the same WCS back in
    # degrees, and both conventions are live in this module.
    hdr_new['CRVAL1'] = crval1.to_value(u.Unit(hdr_new.get('CUNIT1', 'arcsec')))
    hdr_new['CRVAL2'] = crval2.to_value(u.Unit(hdr_new.get('CUNIT2', 'arcsec')))
    hdr_new['PC1_1'] = rotation_matrix[0, 0]
    hdr_new['PC1_2'] = rotation_matrix[0, 1]
    hdr_new['PC2_1'] = rotation_matrix[1, 0]
    hdr_new['PC2_2'] = rotation_matrix[1, 1]
    hdr_new['CDELT1'] = hdr_new['CDELT1'] * cdelt1_multiplier

    # CROTA and the CDi_j form are removed so that the PC + CDELT form written
    # here is unambiguously the one a reader picks up; a stale CROTA next to a
    # new PC matrix would be silently inconsistent.
    for key in ('CROTA', 'CROTA1', 'CROTA2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2'):
        hdr_new.pop(key, None)

    # Assigning a list to 'COMMENT' stringifies the list itself into one long
    # CONTINUE-ed card; add_comment writes one card per line.
    for comment in ('Co-aligned with EUI synoptic raster',
                    'CRVAL1 and CRVAL2 updated',
                    'PC1_1, PC1_2, PC2_1, PC2_2 updated',
                    'CROTA, CROTA1, CROTA2, CD1_1, CD1_2, CD2_1, CD2_2 removed'):
        hdr_new.add_comment(comment)
    hdr_new.add_history('euispice_coalign.py')

    return hdr_new


def find_closest_frame(select_time, time_sequence, light_travel_corr=0 * u.s):
    """
    Index of the entry in ``time_sequence`` closest to ``select_time``.

    Parameters
    ----------
    select_time : `astropy.time.Time`
    time_sequence : `astropy.time.Time`
    light_travel_corr : `astropy.units.Quantity`, optional
        Added to the time difference before minimising. Zero is correct for
        SPICE against EUI, which share a spacecraft; it exists for comparisons
        against instruments at a different distance from the Sun.

    Returns
    -------
    int
    """
    return np.argmin(np.abs(select_time - time_sequence + light_travel_corr))


def _calculate_shift(this_layer, template):
    """
    An improved version of the _calculate_shift function in sunkit_image.coalignment
    that also returns the maximum cross-correlation value.
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
        to the input array, plus the peak correlation value.

    Raises
    ------
    ValueError
        If either input contains non-finite values. `skimage.feature.match_template`
        does not propagate NaN to the whole correlation array -- it returns a
        finite array whose maximum sits in a corner -- so a NaN would otherwise
        produce a confident, silently wrong shift.
    """
    if not (np.all(np.isfinite(this_layer)) and np.all(np.isfinite(template))):
        raise ValueError(
            'non-finite entries in the layer or the template. For the layer this '
            'means the SPICE grid extends past the EUI frame; for the template it '
            'means the cut region includes the SPICE dumbbells or off-detector '
            'pixels. Narrow the cut region or supply EUI images covering the whole '
            'raster.')
    # Calculate the correlation array matching the template to this layer
    corr = match_template(this_layer, template)
    # Calculate the y and x shifts in pixels
    best_match = _find_best_match_location(corr)
    # Calculate the maximum cross-correlation value
    max_cc = np.max(corr)

    return *best_match, max_cc


def coalign_shift_pixel(layer, template):
    """
    Locate ``template`` within ``layer`` by normalised cross-correlation.

    Note the transposed return order relative to `_calculate_shift`.

    Parameters
    ----------
    layer : `numpy.ndarray`
        The larger array to search, shape ``(ny, nx)``. A plain array, not a
        `sunpy.map.GenericMap`.
    template : `numpy.ndarray`
        The smaller array to locate.

    Returns
    -------
    xshift, yshift : float
        0-based pixel position of the template's origin within ``layer``, with
        sub-pixel refinement.
    max_cc : float
        Peak correlation, used to rank trial roll angles against each other.
    """
    yshift, xshift, max_cc = _calculate_shift(layer, template)
    return float(xshift), float(yshift), max_cc


def coalign_shift_pixel_new(map1, map2):
    """
    Phase-correlation alternative to `coalign_shift_pixel`. Not interchangeable.

    Two conventions differ from `coalign_shift_pixel` and both must be handled
    before this can be swapped in:

    * `skimage.registration.phase_cross_correlation` requires ``map1`` and
      ``map2`` to have the same shape, so the caller must trim rather than cut a
      sub-region as a template.
    * It returns the shift needed to register ``map2`` onto ``map1``, not the
      position of a template's origin, so the world-shift arithmetic in
      `calculate_eui_spice_shift` does not apply unchanged.

    The returned figure of merit is ``-abs(error)``, which is larger for a
    better match but is not on the same scale as the peak correlation returned
    by `coalign_shift_pixel`; the two cannot be compared across trials.

    Parameters
    ----------
    map1, map2 : `numpy.ndarray`
        Same shape.

    Returns
    -------
    xshift, yshift : float
    figure_of_merit : float
    """
    (yshift, xshift), error, diffphase = phase_cross_correlation(map1, map2, upsample_factor=10)
    return xshift, yshift, -np.sqrt(np.sum(error ** 2))


# ---------------------------------------------------------------------------
# Merged slot rasters (output of spice_slot_merge.py)
# ---------------------------------------------------------------------------

def _nan_gaussian(data, sigma):
    """
    Gaussian filter that ignores NaN (normalised convolution); NaN where less
    than half of the kernel weight had data. ``sigma`` is (rows, cols) in pixels.
    """
    finite = np.isfinite(data)
    num = gaussian_filter(np.where(finite, data, 0.0).astype(float), sigma, mode='constant')
    den = gaussian_filter(finite.astype(float), sigma, mode='constant')
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0.5, num / den, np.nan)


def is_slot_merged(filename):
    """True if ``filename`` is a merged slot raster written by spice_slot_merge."""
    return bool(fits.getheader(filename, 0).get('SLOTMRG', False))


def slot_blur(slot_header, eui_pixel_scale, psf_pixels=0.85):
    """
    Gaussian sigma (arcsec, along the raster and slit axes) that brings an EUI
    image to the resolution of a merged slot raster.

    The merged map is blurred along the raster axis by the line profile and
    PSF (``SLOTSGX``, the sigma of the fitted slot edges) and along the slit
    by the PSF (``SLOTSGY``). EUI's own blur, taken as ``psf_pixels`` EUI
    pixels (FWHM ~2 pixels), is subtracted in quadrature. For FSI this is
    ~0; for HRIEUV it is nearly the full SPICE blur.
    """
    sigma_eui = psf_pixels * eui_pixel_scale
    return tuple(float(np.sqrt(max(slot_header.get(key, 0.0) ** 2 - sigma_eui ** 2, 0.0)))
                 for key in ('SLOTSGX', 'SLOTSGY'))


def _slot_step_map(hdul):
    """
    Index of the raster step each pixel of a merged slot raster belongs to
    (the step closest in time to the pixel's TIME value), and the step times.

    Pixels without coverage take the time of their column, or of the nearest
    column with coverage, so the synthetic image is defined everywhere.
    """
    header = hdul[0].header
    tmap = np.asarray(hdul['TIME'].data, dtype=float)
    steps = hdul['STEPS'].data
    step_dt = np.asarray(steps['TIME'], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        col = np.nanmedian(tmap, axis=0)
    ok = np.isfinite(col)
    if not ok.any():
        raise ValueError('TIME extension has no finite values')
    cols = np.arange(col.size)
    col = np.interp(cols, cols[ok], col[ok])
    tfilled = np.where(np.isfinite(tmap), tmap, col[None, :])
    index = np.argmin(np.abs(tfilled[..., None] - step_dt[None, None, :]), axis=-1)
    t_ref = Time(hdul['TIME'].header.get('DATEREF', header['DATE-BEG']))
    return index, t_ref + step_dt * u.s


def _open_slot(slot):
    """Return an open HDUList for a merged slot raster given as a path or HDUList."""
    return slot if isinstance(slot, fits.HDUList) else fits.open(slot)


def _world_to_eui_pixel(eui_map, lon, lat, obstime, observer, rsun):
    """
    EUI pixel coordinates of helioprojective (lon, lat) in degrees, observed
    at ``obstime`` from ``observer``, rotated with the solar surface to the
    time of ``eui_map``.
    """
    coords = SkyCoord(lon * u.deg, lat * u.deg, frame='helioprojective', obstime=obstime,
                      observer=observer, rsun=rsun)
    with propagate_with_solar_surface(rotation_model='rigid'):
        return eui_map.wcs.world_to_pixel(coords)


class _SmoothPixelMap:
    """
    Interpolation of `_world_to_eui_pixel` from a coarse grid.

    Within a SPICE field of view the mapping (two TAN projections plus a few
    minutes of rigid rotation) is nearly linear -- second differences across a
    16 x 40 grid are ~1e-5 pixel -- so it is evaluated exactly on a ``n_lon`` x
    ``n_lat`` grid and interpolated bilinearly (~1e-5 pixel error). Do not
    switch to ``method='cubic'``: scipy fits that spline with an iterative
    solver whose default tolerance leaves ~0.04 pixel errors on values of
    ~1000 pixels, even at the nodes. The interpolation is checked at the cell
    centres; `ok` is False if it is worse than ``tolerance`` pixels or the grid
    has non-finite values (e.g. off-disk points), and the caller should then
    use the exact path.
    """

    def __init__(self, eui_map, lon_range, lat_range, obstime, observer, rsun,
                 n_lon=16, n_lat=40, tolerance=0.01):
        lon = np.linspace(*lon_range, n_lon)
        lat = np.linspace(*lat_range, n_lat)
        glat, glon = np.meshgrid(lat, lon, indexing='ij')
        px, py = _world_to_eui_pixel(eui_map, glon, glat, obstime, observer, rsun)
        self.ok = bool(np.all(np.isfinite(px)) and np.all(np.isfinite(py)))
        if not self.ok:
            return
        self._px = RegularGridInterpolator((lat, lon), px, method='linear')
        self._py = RegularGridInterpolator((lat, lon), py, method='linear')
        # Check halfway between grid nodes, where interpolation is worst
        clat, clon = np.meshgrid(0.5 * (lat[1:] + lat[:-1]), 0.5 * (lon[1:] + lon[:-1]),
                                 indexing='ij')
        ex, ey = _world_to_eui_pixel(eui_map, clon, clat, obstime, observer, rsun)
        ix, iy = self(clon, clat)
        self.max_error = float(np.nanmax(np.hypot(ix - ex, iy - ey)))
        self.ok = self.max_error <= tolerance

    def __call__(self, lon, lat):
        points = (lat, lon)
        return self._px(points), self._py(points)


def create_syn_slot_images(slot_file, eui_files, rotation=True, solar_rotation=True,
                           blur='auto', save_filename=None, eui_frames=None, roll_angles=None,
                           fast=True):
    """
    Synthetic EUI images on the grid of a merged slot raster.

    Every pixel of the merged map is assigned the raster step it was observed
    in (from the ``TIME`` extension) and sampled in the EUI frame closest in
    time to that step, after differential rotation from the step time to the
    frame time -- the slot counterpart of `create_syn_rasters`, with one
    group of pixels per raster step instead of one column.

    Parameters
    ----------
    slot_file : str, `pathlib.Path` or `astropy.io.fits.HDUList`
        Merged slot raster (``SLOTMRG = T``).
    eui_files : list of str
        EUI L2 images around the raster time. Ignored if ``eui_frames`` given.
    rotation : bool, optional
        One image per angle in `TRIAL_ROLL_ANGLES` (True) or the nominal roll.
    roll_angles : array-like, optional
        Trial roll angles in radians, overriding ``rotation``.
    solar_rotation : bool, optional
        If False, every pixel uses the raster midpoint ``DATE-AVG``.
    blur : 'auto', (float, float) or None, optional
        Gaussian sigma (arcsec, x and y) applied to the EUI frames; 'auto'
        uses `slot_blur`. Ignored if ``eui_frames`` is given.
    save_filename : str, optional
        Save the images, angles and EUI label to this .npz.
    eui_frames : `_EUIFrameSet`, optional
        Reuse a frame set (and its cache) across many slot files.
    fast : bool, optional
        Evaluate the sky-to-EUI-pixel transform exactly on a coarse grid per
        raster step and interpolate it (`_SmoothPixelMap`, checked to 0.01
        pixel) instead of transforming every pixel for every trial angle.
        ~4x faster for 21 angles, same result to ~1e-6 of the image range;
        falls back to the exact path if the check fails.

    Returns
    -------
    `numpy.ndarray`
        ``(n_angles, ny, nx)``; NaN where the EUI frames do not reach.
    """
    hdul = _open_slot(slot_file)
    header = hdul[0].header.copy()
    shape = hdul[0].data.shape
    step_index, step_times = _slot_step_map(hdul)
    if hdul is not slot_file:
        hdul.close()
    if eui_frames is None:
        if blur == 'auto':
            probe = fits.getheader(eui_files[0], ext=1)
            scale = abs(probe['CDELT1']) * u.Unit(probe.get('CUNIT1', 'arcsec')).to(u.arcsec)
            blur = slot_blur(header, scale)
        eui_frames = _EUIFrameSet(eui_files, blur=blur)
    if not solar_rotation:
        step_index = np.zeros(shape, dtype=int)
        step_times = Time([header['DATE-AVG']])

    wcs = WCS(header).celestial
    # Observer position from the header (Solar Orbiter moves < 0.002 deg in
    # Stonyhurst longitude during a slot raster). It is re-stamped with each
    # step's time below: propagate_with_solar_surface rotates from the
    # *observer's* obstime, not the frame's, so an observer fixed at DATE-AVG
    # would rotate every step as if it had been observed at DATE-AVG.
    observer = sunpy.map.Map(np.zeros((2, 2)), header).observer_coordinate
    yy, xx = np.indices(shape)
    rot_angles = _trial_roll_angles(rotation) if roll_angles is None \
        else np.atleast_1d(np.asarray(roll_angles, dtype=float))
    syn = np.full((len(rot_angles),) + shape, np.nan)
    groups = [(g, step_index == g) for g in np.unique(step_index)]

    # World coordinates of every pixel for every trial roll
    world = []
    for angle in rot_angles:
        trial = deepcopy(wcs)
        trial.wcs.pc = np.dot(wcs.wcs.pc, _roll_matrix(angle, wcs))
        lon, lat = trial.pixel_to_world_values(xx, yy)
        world.append(((lon + 180) % 360 - 180, lat))

    for g, sel in groups:
        observer_g = SkyCoord(observer.lon, observer.lat, observer.radius,
                              frame='heliographic_stonyhurst', obstime=step_times[g])
        eui_map, interpolator = eui_frames[find_closest_frame(step_times[g],
                                                              eui_frames.time_obs)]
        mapper = None
        if fast:
            lons = np.concatenate([w[0][sel] for w in world])
            lats = np.concatenate([w[1][sel] for w in world])
            pad = 2.0 / 3600
            mapper = _SmoothPixelMap(eui_map, (lons.min() - pad, lons.max() + pad),
                                     (lats.min() - pad, lats.max() + pad), step_times[g],
                                     observer_g, eui_frames.rsun_ref)
            if not mapper.ok:
                mapper = None
        for ia, (lon, lat) in enumerate(world):
            if mapper is not None:
                px, py = mapper(lon[sel], lat[sel])
            else:
                px, py = _world_to_eui_pixel(eui_map, lon[sel], lat[sel], step_times[g],
                                             observer_g, eui_frames.rsun_ref)
            syn[ia][sel] = interpolator((py, px))

    if save_filename is not None:
        np.savez_compressed(save_filename, eui_syn_raster_images=syn, rot_angles=rot_angles,
                            eui_label=eui_frames.label, blur=np.asarray(eui_frames.blur or (0, 0)))
    return syn


def _finite_box(arr, max_bad=0.01, min_size=16):
    """
    Row and column slices of a box inside ``arr`` whose edge rows/columns are
    at least ``1 - max_bad`` finite, found by repeatedly trimming the worst
    edge. Remaining isolated NaN inside are left for the caller to fill.
    """
    bad = ~np.isfinite(arr)
    r0, r1, c0, c1 = 0, arr.shape[0], 0, arr.shape[1]
    while True:
        sub = bad[r0:r1, c0:c1]
        frac = {'r0': sub[0].mean(), 'r1': sub[-1].mean(),
                'c0': sub[:, 0].mean(), 'c1': sub[:, -1].mean()}
        side = max(frac, key=frac.get)
        if frac[side] <= max_bad:
            return slice(r0, r1), slice(c0, c1)
        if side == 'r0':
            r0 += 1
        elif side == 'r1':
            r1 -= 1
        elif side == 'c0':
            c0 += 1
        else:
            c1 -= 1
        if r1 - r0 < min_size or c1 - c0 < min_size:
            raise ValueError('no sufficiently large finite region')


def _fill_nan(arr):
    arr = np.array(arr, dtype=float)
    arr[~np.isfinite(arr)] = np.nanmedian(arr)
    return arr


def calculate_eui_spice_slot_shift(slot_file, eui_syn_images, rotation=True, margin=0.25,
                                   save_filename=None, output_dir=None, plot=False,
                                   title1=None, title2=None, roll_angles=None):
    """
    Fit the pointing of a merged slot raster against synthetic EUI images and
    write a co-aligned copy.

    The central part of the merged map -- inside its finite region, inset by
    ``margin`` of its size on each side, which sets the search range -- is
    located in each synthetic image by normalised cross-correlation; the roll
    with the highest peak wins. As in `calculate_eui_spice_shift`, the
    correction is world(found) - world(nominal) on the rolled WCS and is
    *added* to ``CRVAL``.

    Parameters
    ----------
    slot_file : str or `pathlib.Path`
    eui_syn_images : `numpy.ndarray` or str
        Output of `create_syn_slot_images`, or its .npz.
    rotation : bool, optional
        Must match ``eui_syn_images`` unless it is a .npz.
    roll_angles : array-like, optional
        Trial angles (radians) the images were built with, if not the default.
    margin : float, optional
        Fraction of the finite region trimmed from each side for the template.
    save_filename, output_dir : str, optional
        Name and directory of the co-aligned file; default
        ``<input stem>_coalign.fits`` next to the input.
    plot : bool, optional
        Blink the synthetic image against the co-aligned map.
    title1, title2 : str, optional

    Returns
    -------
    dict
        ``dx``, ``dy`` (arcsec, added to CRVAL1/2), ``roll`` (deg), ``max_cc``,
        ``all_cc`` (per trial angle) and ``output`` (written filename).
    """
    with fits.open(slot_file) as hdul:
        data = np.asarray(hdul[0].data, dtype=float)
        header = hdul[0].header.copy()
    wcs = WCS(header).celestial

    rot_angles = None
    eui_label = 'EUI'
    if isinstance(eui_syn_images, (str, Path)):
        with np.load(eui_syn_images) as npz:
            rot_angles = npz['rot_angles']
            eui_label = str(npz['eui_label']) if 'eui_label' in npz.files else eui_label
            eui_syn_images = npz['eui_syn_raster_images']
    if rot_angles is None:
        rot_angles = _trial_roll_angles(rotation) if roll_angles is None \
            else np.atleast_1d(np.asarray(roll_angles, dtype=float))
    if len(rot_angles) != len(eui_syn_images):
        raise ValueError(f'{len(eui_syn_images)} synthetic images but {len(rot_angles)} '
                         f'trial roll angles (rotation={rotation})')

    # Template: centre of the region where SPICE and every synthetic image are finite
    rs, cs = _finite_box(data)
    common = np.isfinite(data[rs, cs])
    for syn in eui_syn_images:
        common &= np.isfinite(syn[rs, cs])
    sub_r, sub_c = _finite_box(np.where(common, 1.0, np.nan))
    r0, r1 = rs.start + sub_r.start, rs.start + sub_r.stop
    c0, c1 = cs.start + sub_c.start, cs.start + sub_c.stop
    my, mx = int(margin * (r1 - r0)), int(margin * (c1 - c0))
    t_r0, t_c0 = r0 + my, c0 + mx
    template = _fill_nan(data[t_r0:r1 - my, t_c0:c1 - mx])

    shifts, max_ccs = [], []
    for syn in eui_syn_images:
        lr, lc = _finite_box(syn)
        layer = _fill_nan(syn[lr, lc])
        xs, ys, cc = coalign_shift_pixel(layer, template)
        shifts.append((xs + lc.start, ys + lr.start))
        max_ccs.append(cc)
    best = int(np.argmax(max_ccs))
    angle = rot_angles[best]
    rot_matrix = _roll_matrix(angle, wcs)
    wcs_best = deepcopy(wcs)
    wcs_best.wcs.pc = np.dot(wcs.wcs.pc, rot_matrix)

    found = wcs_best.pixel_to_world_values(*shifts[best])
    nominal = wcs_best.pixel_to_world_values(t_c0, t_r0)
    dx = ((found[0] - nominal[0] + 180) % 360 - 180) * 3600.0
    dy = (found[1] - nominal[1]) * 3600.0

    output = apply_slot_correction(
        slot_file, dx, dy, np.rad2deg(angle), output_dir=output_dir,
        save_filename=save_filename, reference=eui_label,
        extra={'COALCC': (float(max_ccs[best]), 'Peak normalised cross-correlation')},
        history='euispice_coalign.calculate_eui_spice_slot_shift')

    if plot:
        syn_best = eui_syn_images[best]
        syn_map = sunpy.map.Map(syn_best, wcs_best)
        # hdus=0: the merged file also holds WEIGHT/UNCERTAINTY/TIME images
        new_map = sunpy.map.Map(output, hdus=0)
        for m in (syn_map, new_map):
            m.plot_settings['aspect'] = m.scale.axis2 / m.scale.axis1
        if title1 is None:
            title1 = f'{eui_label} synthetic'
            if len(rot_angles) > 1:
                title1 += f'\nroll {np.rad2deg(angle):+.2f}\N{DEGREE SIGN}'
        if title2 is None:
            title2 = f"SPICE {header.get('EXTNAME', '')}\nco-aligned"
        SunBlinker(syn_map, new_map, reproject=True, fps=1, title1=title1, title2=title2,
                   norm1=ImageNormalize(vmin=np.nanpercentile(syn_best, 0.2),
                                        vmax=np.nanpercentile(syn_best, 99.8),
                                        stretch=AsinhStretch(0.1)),
                   norm2=ImageNormalize(vmin=np.nanpercentile(data, 0.2),
                                        vmax=np.nanpercentile(data, 99.8),
                                        stretch=AsinhStretch(0.1)))
        plt.show()

    return dict(dx=float(dx), dy=float(dy), roll=float(np.rad2deg(angle)),
                max_cc=float(max_ccs[best]), all_cc=np.array(max_ccs), output=str(output))


def apply_slot_correction(slot_file, dx, dy, roll, output_dir=None, save_filename=None,
                          reference='', extra=None, history='euispice_coalign'):
    """
    Write a copy of a merged slot raster with a pointing correction applied.

    The roll rotates the PC matrix about the reference pixel (post-multiplied
    by `_roll_matrix`, as in the fits) and the shift is added to ``CRVAL``; all
    image extensions (radiance, WEIGHT, UNCERTAINTY, TIME) are updated.

    Parameters
    ----------
    slot_file : str
    dx, dy : float
        Arcsec added to ``CRVAL1``/``CRVAL2``.
    roll : float
        Degrees.
    output_dir, save_filename : str, optional
        Default ``<input stem>_coalign.fits`` next to the input.
    reference : str, optional
        Written to ``COALREF``.
    extra : dict, optional
        More primary-header cards, ``{key: (value, comment)}``.
    history : str, optional

    Returns
    -------
    str
        The written filename.
    """
    header = fits.getheader(slot_file, 0)
    wcs = WCS(header).celestial
    rot_matrix = _roll_matrix(np.deg2rad(roll), wcs)
    new_pc = np.dot(np.array([[header['PC1_1'], header['PC1_2']],
                              [header['PC2_1'], header['PC2_2']]]), rot_matrix)
    unit1 = u.Unit(header.get('CUNIT1', 'arcsec')).to(u.arcsec)
    unit2 = u.Unit(header.get('CUNIT2', 'arcsec')).to(u.arcsec)
    new_crval = (header['CRVAL1'] + dx / unit1, header['CRVAL2'] + dy / unit2)

    outdir = Path(slot_file).parent if output_dir is None else Path(output_dir)
    if save_filename is None:
        save_filename = Path(slot_file).stem + '_coalign.fits'
    output = outdir / save_filename
    with fits.open(slot_file) as hdul:
        for hdu in hdul:
            if hdu.header.get('CTYPE1', '').startswith('HPLN'):
                h = hdu.header
                h['CRVAL1'], h['CRVAL2'] = new_crval
                h['PC1_1'], h['PC1_2'] = new_pc[0]
                h['PC2_1'], h['PC2_2'] = new_pc[1]
        h0 = hdul[0].header
        h0['COALREF'] = (reference, 'Reference of the co-alignment')
        h0['COALDX'] = (float(dx), '[arcsec] Shift added to CRVAL1')
        h0['COALDY'] = (float(dy), '[arcsec] Shift added to CRVAL2')
        h0['COALROLL'] = (float(roll), '[deg] Roll correction applied to PC')
        for key, value in (extra or {}).items():
            h0[key] = value
        h0.add_history(history)
        hdul.writeto(output, overwrite=True)
    return str(output)


def _region_shift(data, syn, wcs, rows, cols, margin=0.2):
    """
    Translation (arcsec) that puts part ``rows`` x ``cols`` of a merged map
    onto a synthetic EUI image of the same grid, and the peak correlation.
    """
    rs, cs = _finite_box(data[rows, cols])
    r0, r1 = rows.start + rs.start, rows.start + rs.stop
    c0, c1 = cols.start + cs.start, cols.start + cs.stop
    my, mx = int(margin * (r1 - r0)), int(margin * (c1 - c0))
    template = _fill_nan(data[r0 + my:r1 - my, c0 + mx:c1 - mx])
    lr, lc = _finite_box(syn)
    xs, ys, cc = coalign_shift_pixel(_fill_nan(syn[lr, lc]), template)
    found = wcs.pixel_to_world_values(xs + lc.start, ys + lr.start)
    nominal = wcs.pixel_to_world_values(c0 + mx, r0 + my)
    return (((found[0] - nominal[0] + 180) % 360 - 180) * 3600.0,
            (found[1] - nominal[1]) * 3600.0, cc)


def fit_slot_step_scale(l2_file, window, response, eui_files=None, eui_frames=None,
                        scales=(1.0, 1.02, 1.04, 1.06), **merge_kwargs):
    """
    Measure the raster-step multiplier of a slot raster against EUI.

    The raster is merged with each trial multiplier of ``CDELT1``; the left
    and right halves of the map are then co-aligned separately (translation
    only) with a synthetic EUI image, and the difference of their x shifts is
    recorded. The step is right when both halves need the same shift: the
    zero of a straight line through the differences is returned. The halves
    are ~4 raster steps apart, so a 1% step error shows up as ~1.2 arcsec,
    well within reach of FSI, whereas the peak correlation itself barely
    changes with the step.

    Parameters
    ----------
    l2_file : str
        SPICE L2 wide-slit raster.
    window : int or str
    response : `spice_slot_merge.SlotResponse`
    eui_files : list of str, optional
    eui_frames : `_EUIFrameSet`, optional
        Reuse a frame set; one of ``eui_files``/``eui_frames`` is required.
    scales : sequence of float, optional
    **merge_kwargs
        Passed to `spice_slot_merge.merge_slot_raster`.

    Returns
    -------
    best : float
        Multiplier at which the halves agree.
    table : `numpy.ndarray`
        ``(n_scales, 2)``: multiplier, x(left) - x(right) in arcsec.
    """
    import spice_slot_merge as ssm

    if eui_frames is None:
        eui_frames = _EUIFrameSet(eui_files)
    raster = ssm.read_slot_raster(l2_file, window)
    table = []
    for scale in scales:
        hdul = ssm.merge_slot_raster(raster, response, uncertainty=False, step_scale=scale,
                                     **merge_kwargs)
        syn = create_syn_slot_images(hdul, None, rotation=False, eui_frames=eui_frames)[0]
        data = np.asarray(hdul[0].data, dtype=float)
        wcs = WCS(hdul[0].header).celestial
        ny, nx = data.shape
        left = _region_shift(data, syn, wcs, slice(0, ny), slice(0, nx // 2))
        right = _region_shift(data, syn, wcs, slice(0, ny), slice(nx // 2, nx))
        table.append((scale, left[0] - right[0]))
    table = np.array(table)
    slope, intercept = np.polyfit(table[:, 0], table[:, 1], 1)
    return float(-intercept / slope), table


def coalign_slot_files(slot_files, eui_files, rotation=True, blur='auto', output_dir=None,
                       syn_dir=None, max_cached=8, verbose=True, roll_angles=None):
    """
    Co-align many merged slot rasters of one window against one EUI channel,
    sharing a single cached EUI frame set.

    Parameters
    ----------
    slot_files : list of str
    eui_files : list of str
    rotation : bool, optional
    blur : 'auto', (float, float) or None, optional
        'auto' derives the blur from the first slot file (`slot_blur`).
    output_dir : str, optional
        Directory for the co-aligned files (default: next to each input).
    syn_dir : str, optional
        If given, the synthetic images are saved there as .npz.
    max_cached : int, optional

    Returns
    -------
    `astropy.table.Table`
        One row per file: file, date, dx, dy, roll, max_cc, output.
    """
    from astropy.table import Table

    slot_files = [str(f) for f in slot_files]
    if blur == 'auto':
        probe = fits.getheader(eui_files[0], ext=1)
        scale = abs(probe['CDELT1']) * u.Unit(probe.get('CUNIT1', 'arcsec')).to(u.arcsec)
        blur = slot_blur(fits.getheader(slot_files[0], 0), scale)
    frames = _EUIFrameSet(eui_files, max_cached=max_cached, blur=blur)
    rows = []
    for f in slot_files:
        npz = None if syn_dir is None else str(Path(syn_dir) / (Path(f).stem + '_syn.npz'))
        syn = create_syn_slot_images(f, None, rotation=rotation, eui_frames=frames,
                                     save_filename=npz, roll_angles=roll_angles)
        res = calculate_eui_spice_slot_shift(f, syn, rotation=rotation, output_dir=output_dir,
                                             roll_angles=roll_angles)
        date = fits.getheader(f, 0).get('DATE-AVG', '')
        rows.append((Path(f).name, date, res['dx'], res['dy'], res['roll'], res['max_cc'],
                     res['output']))
        if verbose:
            print(f"{Path(f).name}: dx {res['dx']:+.2f}\" dy {res['dy']:+.2f}\" "
                  f"roll {res['roll']:+.2f} deg cc {res['max_cc']:.3f}")
    return Table(rows=rows, names=('file', 'date', 'dx', 'dy', 'roll', 'max_cc', 'output'))


if __name__ == '__main__':
    # Usage examples for every option are in the module docstring at the top.
    parser = argparse.ArgumentParser(description='Co-align EUI synoptic raster with SPICE data')
    parser.add_argument('spice_file', type=str, help='SPICE file')
    parser.add_argument('eui_files', type=str, help='EUI file directory, or a glob pattern')
    parser.add_argument('-w', '--spice_window', type=str, default='Ne VIII 770 - Peak',
                        help='SPICE window')
    parser.add_argument('-s', '--save_filename', type=str, default=None,
                        help='Filename of the co-aligned SPICE file')
    parser.add_argument('-nr', '--no-rotation', dest='rotation', action='store_false',
                        help='Disable the trial roll-angle search')
    parser.add_argument('-nsr', '--no-solar-rotation', dest='solar_rotation',
                        action='store_false',
                        help='Treat the raster as instantaneous (no per-column time)')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('-sf', '--synthetic_raster_filename', type=str, default=None,
                        help='Filename of a cached synthetic raster .npz to reuse')
    parser.add_argument('-c1', '--cdelt1', type=float, default=1, help='CDELT1 multiplier')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Show a blink comparison after saving')
    parser.add_argument('-t1', '--title1', type=str, default=None,
                        help='Title of the EUI frame in the blink comparison')
    parser.add_argument('-t2', '--title2', type=str, default=None,
                        help='Title of the SPICE frame in the blink comparison')

    parser.add_argument('--no-blur', dest='blur', action='store_false',
                        help='Merged slot rasters only: do not blur EUI to the SPICE resolution')

    args = parser.parse_args()

    if os.path.isdir(args.eui_files):
        eui_files = sorted(glob(os.path.join(args.eui_files, '*.fits')))
    else:
        eui_files = sorted(glob(args.eui_files))

    if is_slot_merged(args.spice_file):
        # Merged slot raster: one window per file, so -w and -c1 do not apply.
        if args.cdelt1 != 1:
            parser.error('-c1 is not supported for merged slot rasters')
        if args.synthetic_raster_filename is None:
            syn_file = os.path.join(os.path.dirname(eui_files[0]),
                                    Path(args.spice_file).stem + '_syn.npz')
            syn = create_syn_slot_images(args.spice_file, eui_files, rotation=args.rotation,
                                         solar_rotation=args.solar_rotation,
                                         blur='auto' if args.blur else None,
                                         save_filename=syn_file)
        else:
            syn = args.synthetic_raster_filename
        result = calculate_eui_spice_slot_shift(
            args.spice_file, syn, rotation=args.rotation, save_filename=args.save_filename,
            output_dir=args.output_dir, plot=args.plot, title1=args.title1, title2=args.title2)
        print(f"dx {result['dx']:+.2f} arcsec, dy {result['dy']:+.2f} arcsec, roll "
              f"{result['roll']:+.2f} deg, cc {result['max_cc']:.3f} -> {result['output']}")
        raise SystemExit(0)

    if args.synthetic_raster_filename is None:
        synthetic_raster_filename = os.path.join(os.path.dirname(eui_files[0]),
                                                 'eui_syn_raster_image_for_spice.npz')
        eui_syn_raster_images = create_syn_rasters(
            args.spice_file, eui_files, args.spice_window, synthetic_raster_filename,
            solar_rotation=args.solar_rotation, cdelt1_multiplier=args.cdelt1,
            rotation=args.rotation)
    else:
        # The .npz also carries its trial roll angles, so --no-rotation need not
        # be repeated to match how the cache was built.
        eui_syn_raster_images = args.synthetic_raster_filename

    xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal = \
        calculate_eui_spice_shift(args.spice_file, args.spice_window, eui_syn_raster_images,
                                  rotation=args.rotation, cdelt1_multiplier=args.cdelt1,
                                  save_filename=args.save_filename,
                                  output_dir=args.output_dir, plot=args.plot,
                                  title1=args.title1, title2=args.title2)

    print(xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal)
