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
Blocks until the window is closed::

    python euispice_coalign.py spice.fits /data/eui/ -p

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
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
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

    Attributes
    ----------
    time_obs : `astropy.time.Time`
        Exposure midpoint of each frame.
    rsun_ref : `astropy.units.Quantity`
        ``RSUN_REF`` of the first frame, used as the ``rsun`` of the
        helioprojective frames built against these images.
    """

    def __init__(self, eui_files, max_cached=4):
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
        interpolator = RegularGridInterpolator(
            (np.arange(eui_map.data.shape[0]), np.arange(eui_map.data.shape[1])),
            eui_map.data, bounds_error=False, method='linear')
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
        # paired with a different trial grid.
        np.savez_compressed(save_filename,
                            eui_syn_raster_images=eui_syn_raster_images,
                            rot_angles=rot_angles)

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
                              output_dir=None, plot=False):
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
    if isinstance(eui_syn_raster_images, (str, Path)):
        with np.load(eui_syn_raster_images) as npz:
            if 'rot_angles' in npz.files:
                rot_angles = npz['rot_angles']
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
        SunBlinker(eui_syn_raster_map, spice_int_map, reproject=True, fps=1,
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

    args = parser.parse_args()

    if os.path.isdir(args.eui_files):
        eui_files = sorted(glob(os.path.join(args.eui_files, '*.fits')))
    else:
        eui_files = sorted(glob(args.eui_files))

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
                                  output_dir=args.output_dir, plot=args.plot)

    print(xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal)
