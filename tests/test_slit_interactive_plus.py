"""
Tests for slit_interactive_plus.py. Run with:  python -m pytest tests/test_slit_interactive_plus.py

Straight slits are checked against slit_interactive.py, which samples the same points.
Everything the plus version changes on purpose (splines, spacing, backgrounds) is checked
against its own definition instead.
"""
import os
import sys
import glob
import warnings

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib.backend_bases import NavigationToolbar2
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
import sunpy.map
from sunpy.coordinates import frames

import slit_interactive_plus as sip

warnings.filterwarnings('ignore', message='.*non-interactive.*')
HRI = '/Users/yjzhu/Desktop/Solar/Weak_Riblets/data/hrieuv/raw'


def _cube(ny=64, nx=80, nt=12, seed=0, nan_edge=0):
    rng = np.random.default_rng(seed)
    base = gaussian_filter(rng.normal(size=(ny, nx)), 3)
    base = (base - base.min()) * 500 + 100
    yy, xx = np.mgrid[0:ny, 0:nx]
    cube = np.empty((ny, nx, nt))
    for t in range(nt):
        cube[:, :, t] = base * (1 + 0.05 * np.sin(t)) + 300 * np.exp(-((xx - 10 - 3 * t) ** 2 + (yy - 30) ** 2) / 20)
    if nan_edge:
        cube[:nan_edge] = np.nan
        cube[:, -nan_edge:] = np.nan
    return cube


def _maps(nt=6, ny=48, nx=64, rotation=7 * u.deg, cadence=10 * u.s):
    cube = _cube(ny, nx, nt)
    out = []
    for t in range(nt):
        obstime = Time('2023-04-10T22:35:00') + t * cadence
        ref = SkyCoord(100 * u.arcsec, -50 * u.arcsec, obstime=obstime, observer='earth',
                       frame=frames.Helioprojective)
        header = sunpy.map.make_fitswcs_header(cube[:, :, t].astype(np.float32), ref,
                                               scale=[0.5, 0.5] * u.arcsec / u.pix,
                                               rotation_angle=rotation, unit=u.DN / u.s)
        out.append(sunpy.map.Map(cube[:, :, t].astype(np.float32), header))
    return sunpy.map.Map(out, sequence=True)


@pytest.fixture(scope='module')
def old():
    return pytest.importorskip('slit_interactive')


# ----------------------------------------------------------------------------- ImageStack
def test_image_stack_inputs_and_errors():
    cube = _cube()
    assert sip.ImageStack(cube).shape == cube.shape
    seq = _maps()
    st = sip.ImageStack(seq)
    assert st.is_map and st.shape == (48, 64, 6) and st.unit == u.DN / u.s
    assert len(st.times) == 6
    assert sip.ImageStack(list(seq.maps)).nt == 6
    assert sip.ImageStack(seq[0]).nt == 1
    with pytest.raises(TypeError):
        sip.ImageStack(cube.tolist())
    with pytest.raises(ValueError):
        sip.ImageStack(np.zeros((5, 5)))


def test_crop_and_apply_do_not_modify_input():
    cube = _cube()
    original = cube.copy()
    st = sip.ImageStack(cube)
    for bl, tr in [([10, 5], [49, 34]), ([10, 5] * u.pix, [49, 34] * u.pix)]:
        c = st.crop(bl, tr)
        assert c.shape == (30, 40, cube.shape[2])
        np.testing.assert_array_equal(c.frame(3), cube[5:35, 10:50, 3])
    doubled = st.apply(lambda f: f * 2)
    np.testing.assert_allclose(doubled.frame(2), 2 * cube[:, :, 2])
    np.testing.assert_array_equal(cube, original)
    seq = _maps()
    cropped = sip.ImageStack(seq).crop([5, 6] * u.pix, [44, 35] * u.pix)
    assert cropped.shape == (30, 40, 6)


def test_sigma_over_mean_matches_numpy():
    cube = _cube(nan_edge=3)
    expected = np.nanstd(cube, axis=2) / np.nanmean(cube, axis=2)
    got = sip.ImageStack(cube).sigma_over_mean(stride=1)
    np.testing.assert_allclose(got, expected, rtol=1e-9, equal_nan=True)


def test_cache_gives_same_result():
    cube = _cube(nan_edge=4)
    geom = sip.SlitGeometry([5, 40, 75], [3, 50, 20], line_width=3)
    a = sip.extract_slits(sip.ImageStack(cube), [geom])[0].intensity
    b = sip.extract_slits(sip.ImageStack(cube, cache=True), [geom])[0].intensity
    np.testing.assert_allclose(a, b, rtol=1e-6, equal_nan=True)   # cache is float32
    c = sip.extract_slits(sip.ImageStack(cube, cache=True), [geom], frames=[1, 5, 7])[0]
    np.testing.assert_allclose(c.intensity, a[:, [1, 5, 7]], rtol=1e-6, equal_nan=True)


# ----------------------------------------------------------------------------- geometry
def test_straight_geometry_matches_skimage_profile_coordinates():
    from skimage.measure import profile
    for (x0, y0, x1, y1), lw in [((3.2, 4.1, 70.7, 50.3), 5), ((60, 10, 5, 55), 4), ((10, 10, 10, 60), 1)]:
        g = sip.SlitGeometry([x0, x1], [y0, y1], line_width=lw)
        rows, cols = profile._line_profile_coordinates((y0, x0), (y1, x1), linewidth=lw)
        assert g.x.shape == cols.shape
        np.testing.assert_allclose(g.x, cols, atol=1e-9)
        np.testing.assert_allclose(g.y, rows, atol=1e-9)


@pytest.mark.parametrize('n', [3, 4, 6, 10, 19, 40])
def test_spline_passes_through_points_with_uniform_spacing(n):
    t = np.linspace(0, 1, n)
    xs = 20 + 200 * t
    ys = 100 + 40 * np.sin(3 * np.pi * t)
    g = sip.SlitGeometry(xs, ys, line_width=5)
    assert g.method == 'spline'
    d = np.min(np.hypot(g.centre_x[:, None] - xs, g.centre_y[:, None] - ys), axis=0)
    assert d.max() < 0.6          # within half a sample of every clicked point
    step = np.hypot(np.diff(g.centre_x), np.diff(g.centre_y))
    assert step.max() <= 1.0 + 1e-3 and step.min() > 0.97 * step.mean()


@pytest.mark.parametrize('lw', [1, 2, 3, 4, 7])
def test_line_width_gives_that_many_lines(lw):
    for xs, ys in [([5, 60], [5, 40]), ([5, 30, 60], [5, 40, 10]), ([5, 20, 40, 60], [5, 30, 10, 40])]:
        g = sip.SlitGeometry(xs, ys, line_width=lw)
        assert g.x.shape == (g.n_dist, lw)
        across = np.hypot(g.x[:, -1] - g.x[:, 0], g.y[:, -1] - g.y[:, 0])
        np.testing.assert_allclose(across, lw - 1, atol=1e-6)


def test_geometry_validation_and_from_center():
    with pytest.raises(ValueError):
        sip.SlitGeometry([1, 1], [2, 2])
    with pytest.raises(ValueError):
        sip.SlitGeometry([1, 5], [2, 9], line_width=0)
    with pytest.raises(ValueError):
        sip.SlitGeometry([1, 5, 8], [2, 9, 3], method='bezier')
    g = sip.SlitGeometry.from_center(50, 40, 20, 30)
    np.testing.assert_allclose([g.control_x[1] - g.control_x[0], g.control_y[1] - g.control_y[0]],
                               [20 * np.cos(np.pi / 6), 20 * np.sin(np.pi / 6)])
    assert sip.SlitGeometry([1, 2, 5, 9], [1, 3, 2, 8], method='polyline').method == 'polyline'


# ----------------------------------------------------------------------------- extraction
def test_sampler_matches_regular_grid_interpolator():
    from scipy.interpolate import RegularGridInterpolator
    frame = _cube(nan_edge=4)[:, :, 0]
    rng = np.random.default_rng(3)
    x = rng.uniform(-3, 82, 4000)
    y = rng.uniform(-3, 66, 4000)
    ref = RegularGridInterpolator((np.arange(64), np.arange(80)), frame, bounds_error=False,
                                  fill_value=np.nan)(np.c_[y, x])
    got = sip.SlitSampler(x, y, 64, 80).sample(frame)
    finite = np.isfinite(ref)
    np.testing.assert_allclose(got[finite], ref[finite], rtol=1e-12)
    # NaN only where the reference is NaN, or fewer (zero-weight NaN corners are ignored)
    assert not np.any(np.isnan(got) & finite)
    on_grid = sip.SlitSampler([10.0, 79.0], [20.0, 63.0], 64, 80).sample(frame)
    assert np.isfinite(on_grid[0])


def test_straight_slit_matches_old_module_ndarray(old):
    cube = _cube(nan_edge=4)
    for xs, ys in [([5, 70], [8, 55]), ([2, 78], [1, 40])]:
        ref = old.generate_slit_data_from_points(xs, ys, cube, 'NDArray', line_width=5)
        new = sip.make_slit(cube, xs, ys, line_width=5)
        ref_i = np.asarray(ref['slit_intensity'])
        assert new.intensity.shape == ref_i.shape
        # the old code also turned samples NaN through zero-weight NaN neighbours
        assert not np.any(np.isnan(new.intensity) & np.isfinite(ref_i))
        both = np.isfinite(new.intensity) & np.isfinite(ref_i)
        assert both.mean() > 0.8
        np.testing.assert_allclose(new.intensity[both], ref_i[both], rtol=1e-12)
        np.testing.assert_allclose(new.distance.to_value(u.pix), ref['pixel_distance_interp'], rtol=1e-12)


def test_straight_slit_matches_old_module_sunpy(old):
    seq = _maps()
    xs, ys = [4, 58], [6, 40]
    ref = old.generate_slit_data_from_points(xs, ys, seq, 'SunpyMap', line_width=5, map_wcs=seq[0].wcs)
    new = sip.make_slit(seq, xs, ys, line_width=5)
    np.testing.assert_allclose(new.intensity, np.asarray(ref['slit_intensity']), rtol=1e-12)
    np.testing.assert_allclose(new.distance.to_value(u.m),
                               ref['world_coord_center_distance_interp'].to_value(u.m), rtol=1e-12)
    assert new.unit == u.DN / u.s
    assert new.distance.unit == u.Mm
    np.testing.assert_allclose(new.time_seconds, np.arange(6) * 10.0)


def test_world_distance_matches_plate_scale():
    seq = _maps()
    res = sip.make_slit(seq, [4, 30, 58], [6, 40, 20], line_width=3)
    scale = seq[0].scale.axis1 * u.pix
    expected = (res.geometry.length * scale).to_value(u.rad) * seq[0].dsun
    assert abs((res.distance[-1] - expected).to_value(u.km)) < 1e-3 * expected.to_value(u.km)
    assert res.centre_coord is not None and len(res.centre_coord) == res.geometry.n_dist


def test_wcs_shift_changes_world_coords_only():
    seq = _maps()
    a = sip.make_slit(seq, [4, 58], [6, 40])
    b = sip.make_slit(seq, [4, 58], [6, 40], wcs_shift=[3, -2] * u.arcsec)
    np.testing.assert_allclose(a.intensity, b.intensity)
    assert abs(a.centre_coord[0].Tx - b.centre_coord[0].Tx) > 2 * u.arcsec
    assert b.meta['wcs_shift'] == [3.0, -2.0]


def test_extract_slits_batch_equals_single():
    cube = _cube()
    geoms = [sip.SlitGeometry.from_center(30, 30, 15, a) for a in (0, 45, 90)]
    batch = sip.extract_slits(cube, geoms)
    for g, res in zip(geoms, batch):
        np.testing.assert_allclose(res.intensity, sip.extract_slits(cube, [g])[0].intensity)


def test_reducers():
    cube = _cube()
    a = sip.make_slit(cube, [5, 70], [8, 55], reducer='median')
    b = sip.make_slit(cube, [5, 70], [8, 55], reducer=lambda arr, axis: np.nanmax(arr, axis=axis))
    assert np.all(b.intensity >= a.intensity - 1e-9)
    with pytest.raises(ValueError):
        sip.make_slit(cube, [5, 70], [8, 55], reducer='mode')


# ----------------------------------------------------------------------------- background
def test_background_matches_old_without_nans(old):
    st = np.random.default_rng(5).normal(100, 5, size=(40, 90)) + np.linspace(0, 20, 90)
    pairs = [('median', {}, {}), ('percentile', dict(percentile=10), dict(percentile=10)),
             ('morphological', dict(window=5), dict(morph_size=5)),
             ('running_min', dict(window=15), dict(min_window=15)),
             ('running_percentile', dict(window=15, percentile=10), dict(percentile_window=15, percentile=10)),
             ('running_mean', dict(window=15), dict(mean_window=15))]
    for method, new_kw, old_kw in pairs:
        new_out, new_bg = sip.remove_background(st, method, **new_kw)
        old_out, old_bg = old.remove_background(st, method=method, **old_kw)
        np.testing.assert_allclose(new_bg, np.broadcast_to(old_bg, st.shape), rtol=1e-12, err_msg=method)


def test_background_nans_stay_local():
    st = np.random.default_rng(1).normal(100, 5, size=(30, 50))
    st[3, 7] = np.nan
    for method in sip.BACKGROUND_METHODS:
        out, bg = sip.remove_background(st, method)
        assert np.isnan(out).sum() == 1, method
        assert np.isfinite(out[3, 8:]).all(), method


def test_gaussian_background_smooths_along_time():
    st = np.zeros((40, 60))
    st[20, 30] = 1.0
    _, bg = sip.remove_background(st, 'gaussian', sigma=3)
    assert np.count_nonzero(bg[:, 30] > 1e-8) == 1        # no spread along distance
    assert np.count_nonzero(bg[20, :] > 1e-8) > 10         # spread along time


def test_register_background():
    @sip.register_background('test_row_min', offset=0.0)
    def row_min(st, offset=0.0):
        return np.nanmin(st, axis=1, keepdims=True) + offset
    try:
        st = np.arange(12.0).reshape(3, 4)
        out, bg = sip.remove_background(st, 'test_row_min', offset=1.0)
        np.testing.assert_allclose(bg[:, 0], [1, 5, 9])
    finally:
        sip.BACKGROUND_METHODS.pop('test_row_min')
        sip.BACKGROUND_DEFAULTS.pop('test_row_min')
    with pytest.raises(ValueError):
        sip.remove_background(st, 'nope')


# ----------------------------------------------------------------------------- fitting
def test_relocation_is_safe_and_accurate():
    flat = np.tile(np.linspace(0, 1, 50)[:, None], (1, 40))      # monotonic columns
    r = sip.fit_curve_on_image([5, 15, 25, 35], [20, 20, 20, 20], flat, poly_order=1, relocate=True)
    np.testing.assert_array_equal(r['ydata_fitted'], 20)           # no ridge: points stay
    assert not r['relocation_ok'].any()
    convex = np.tile((np.linspace(1, 0, 50) ** 2)[:, None], (1, 40))
    r = sip.fit_curve_on_image([5, 15, 25, 35], [20, 20, 20, 20], convex, poly_order=1, relocate=True)
    np.testing.assert_array_equal(r['ydata_fitted'], 20)
    edge = np.zeros((50, 40))
    edge[0] = 1.0                                                   # maximum on the image edge
    r = sip.fit_curve_on_image([5, 15], [1, 1], edge, poly_order=1, relocate=True)
    np.testing.assert_array_equal(r['ydata_fitted'], 0)
    t = np.arange(80)
    ridge = 20 + 0.4 * t
    img = np.exp(-(np.arange(60)[:, None] - ridge[None, :]) ** 2 / 4.0)
    xs = np.array([5, 20, 35, 50, 65, 75.0])
    r = sip.fit_curve_on_image(xs, np.interp(xs, t, ridge) + 0.8, img, poly_order=1, relocate=True,
                               window_half_size=2)
    np.testing.assert_allclose(r['fit_params'], [0.4, 20], atol=0.05)
    assert r['relocation_ok'].all()


def test_spacetime_fit_world_velocity():
    seq = _maps(nt=10, cadence=12 * u.s)
    res = sip.make_slit(seq, [4, 58], [6, 40], line_width=3)
    step_km = res.distance_step.to_value(u.km)
    rows = 5 + 2.5 * np.arange(10)                      # 2.5 rows per frame
    fit = sip.SpacetimeFit.from_points(res, np.arange(10), rows, poly_order=1)
    np.testing.assert_allclose(fit.velocity(), 2.5 * step_km / 12.0, rtol=1e-9)
    assert fit.world_units == ('s', 'km')
    arr = sip.make_slit(_cube(), [5, 70], [8, 55])
    fit2 = sip.SpacetimeFit.from_points(arr, [0, 5, 10], [3, 13, 23], poly_order=1)
    assert fit2.world_units == ('frame', 'pix')
    np.testing.assert_allclose(fit2.velocity(), 2.0 * arr.distance_step.to_value(u.pix))


def test_save_load_roundtrip(tmp_path):
    seq = _maps()
    res = sip.make_slit(seq, [4, 30, 58], [6, 40, 20], line_width=3, wcs_shift=[1, 2] * u.arcsec)
    fits = [sip.SpacetimeFit.from_points(res, [0, 2, 4, 5], [3, 9, 15, 19], poly_order=2),
            sip.SpacetimeFit.from_points(res, [0, 5], [30, 20], poly_order=1)]
    path = res.save(tmp_path / 'slit.h5', fits)
    loaded, lfits = sip.SlitResult.load(path)
    np.testing.assert_array_equal(loaded.intensity, res.intensity)
    np.testing.assert_allclose(loaded.distance, res.distance)
    assert loaded.unit == res.unit and loaded.meta['wcs_shift'] == [1.0, 2.0]
    assert np.all(loaded.times == res.times)
    np.testing.assert_allclose(loaded.geometry.x, res.geometry.x)
    assert [f.poly_order for f in lfits] == [2, 1]
    np.testing.assert_allclose(lfits[0].params_world, fits[0].params_world)
    cube_res = sip.make_slit(_cube(), [5, 70], [8, 55])
    loaded2, _ = sip.SlitResult.load(cube_res.save(tmp_path / 'arr.h5'))
    assert loaded2.times is None and loaded2.distance.unit == u.pix


def test_to_ndcube():
    res = sip.make_slit(_maps(), [4, 58], [6, 40])
    cube = res.to_ndcube()
    assert cube.data.shape == res.intensity.shape
    assert cube.unit == u.DN / u.s


# ----------------------------------------------------------------------------- preview
def test_preview_grid(tmp_path):
    cube = _cube(ny=50, nx=60, nt=8)
    results = sip.preview_grid(cube, n_x=2, n_y=2, n_angles=2, length=10, save_dir=tmp_path, dpi=50)
    assert len(results) == 8
    centres = {(round(r.geometry.centre_x.mean()), round(r.geometry.centre_y.mean())) for r in results}
    assert centres == {(20, 16), (20, 33), (39, 16), (39, 33)}   # interior grid, no edge slits
    assert all(np.isfinite(r.intensity).all() for r in results)
    assert len(list(tmp_path.glob('*.png'))) == 8
    assert sip.preview_grid(cube, n_x=1, n_y=1, n_angles=1)[0].geometry.method == 'polyline'


# ----------------------------------------------------------------------------- GUI (Agg)
class _Event:
    def __init__(self, ax, x, y, button=1):
        self.inaxes, self.xdata, self.ydata, self.button = ax, x, y, button
        self.x, self.y = ax.transData.transform((x, y))


def _open(data, **kw):
    pick = sip.SlitPickPlus(data)
    pick(backend=None, **kw)
    return pick


def _click(pick, ax, x, y, button=1):
    ev = _Event(ax, x, y, button)
    pick._on_press(ev)
    pick._on_release(ev)


def test_gui_slit_frame_and_fits(tmp_path):
    home = NavigationToolbar2.home
    pick = _open(_cube(), line_width=7, save_dir=str(tmp_path))
    assert NavigationToolbar2.home is home                 # no global monkeypatch
    assert pick.text_lw.text == '7'
    pick._start_selection()
    for x, y in [(5, 10), (30, 40), (70, 30)]:
        _click(pick, pick.ax1, x, y)
    _click(pick, pick.ax2, 50, 50)                         # extra point
    _click(pick, pick.ax2, 50, 50, button=3)               # and delete it again
    assert len(pick.points) == 3
    # drag the middle point
    ev = _Event(pick.ax1, 30, 40)
    pick._on_press(ev)
    pick._on_move(_Event(pick.ax1, 32, 44))
    pick._on_release(_Event(pick.ax1, 32, 44))
    assert pick.points[1] == (32.0, 44.0)
    pick._end_selection()
    assert pick.slit is not None and pick.slit.geometry.line_width == 7
    assert pick.slit.geometry.method == 'spline'

    draws = []
    pick.fig.canvas.mpl_connect('draw_event', lambda e: draws.append(1))
    pick.text_frame.set_val('0')
    assert pick.frame_index == 0
    pick.set_frame(5)
    pick.fig.canvas.draw()
    assert pick.frame_index == 5 and pick.text_frame.text == '5'
    assert list(pick._timeline.get_xdata()) == [5, 5]

    def add_fit(xs, ys):
        pick._start_fit()
        pick._on_press(_Event(pick.ax3, xs[0], ys[0]))
        for x, y in zip(xs[1:], ys[1:]):
            pick._on_move(_Event(pick.ax3, x, y))
        pick._on_release(_Event(pick.ax3, xs[-1], ys[-1]))
        pick._end_fit()

    add_fit([1, 3, 5, 7, 9], [10, 20, 28, 35, 40])
    pick.text_order.set_val('1')
    add_fit([1, 4, 8], [50, 55, 62])
    assert [f.poly_order for f in pick.fits] == [2, 1]
    pick._delete_fit()
    assert len(pick.fits) == 1 and len(pick.ax3.lines) == 3   # stroke, timeline, one fit
    add_fit([1, 4, 8], [50, 55, 62])
    pick._clean_fits_gui()
    assert pick.fits == [] and len(pick.ax3.lines) == 2

    for method in pick.bg_methods:                           # every registered method reachable
        pick.check_bg.set_active(0) if not pick.bg_on else None
        pick._cycle_background()
    assert set(pick.bg_methods) >= {'running_percentile', 'running_mean', 'gaussian'}

    add_fit([1, 3, 5, 7], [10, 20, 28, 35])
    path = pick.save()
    loaded, fits = sip.SlitResult.load(path)
    assert len(fits) == 1 and os.path.exists(tmp_path / 'slit_plot.png')
    np.testing.assert_array_equal(loaded.intensity, pick.slit.intensity)
    pick.clean()
    assert pick.slit is None and pick.fits == [] and pick.im3 is None
    pick.close()


def test_gui_sunpy_crop_and_time_box(tmp_path):
    seq = _maps(nt=8)
    pick = _open(seq, bottom_left=[4, 4] * u.pix, top_right=[59, 43] * u.pix, save_dir=str(tmp_path),
                 display_bin=2)
    assert pick.stack.shape == (40, 56, 8)
    pick.text_time.set_val(pick.stack.times[6].isot)
    assert pick.frame_index == 6 and pick.text_frame.text == '6'
    res = pick.make_slit([3, 30, 50], [5, 30, 12])
    assert res.distance.unit == u.Mm and pick.im3 is not None
    pick.save()
    loaded, _ = sip.SlitResult.load(tmp_path / 'slit.h5')
    assert loaded.meta['bottom_left']['unit'] == 'pix'
    pick.close()


def test_gui_actions_and_callbacks():
    seen = []
    pick = sip.SlitPickPlus(_cube())
    pick.add_action('Count', lambda p: seen.append(p.slit))
    pick.on_slit.append(lambda res: seen.append('made'))
    pick(backend=None)
    pick.make_slit([5, 70], [8, 55])
    pick.action_buttons[0]._observers.process('clicked', None)
    assert seen[0] == 'made' and isinstance(seen[1], sip.SlitResult)
    pick.close()


def test_gui_interactions_avoid_full_redraws():
    pick = _open(_cube(), save_dir='.')
    canvas = pick.fig.canvas
    canvas.draw()
    draws = []
    canvas.mpl_connect('draw_event', lambda e: draws.append(1))
    pick._start_selection()
    for x, y in [(5, 10), (30, 40), (70, 30)]:
        _click(pick, pick.ax1, x, y)                       # every TextBox sees every click
    ev = _Event(pick.ax1, 30, 40)
    pick._on_press(ev)
    pick._on_move(_Event(pick.ax1, 35, 45))
    pick._on_release(_Event(pick.ax1, 35, 45))
    pick.set_frame(3)
    pick.text_frame.set_val('4')
    pick.slider_asinh.set_val(0.3)
    pick.check_reloc.set_active(0)
    assert draws == [] and pick.frame_index == 4
    pick._end_selection()                                  # new slit: one full redraw
    assert len(draws) == 1
    pick.close()


@pytest.mark.parametrize('size', [(8.0, 7.0), (6.4, 5.2)])
def test_gui_layout_has_no_overlaps(size):
    pick = _open(_maps(nt=4))
    pick.make_slit([3, 30, 50], [5, 30, 12])
    pick.fig.set_size_inches(*size)
    pick._layout()
    pick.fig.canvas.draw()
    renderer = pick.fig.canvas.get_renderer()
    boxes = []
    for i, row in enumerate(pick._rows):
        items = [ax.bbox for ax, _, _ in row['cells']]
        if row['title'] is not None and row['title'].get_text():
            items.append(row['title'].get_window_extent(renderer))
        boxes += [(i, b) for b in items]
    for i, a in boxes:
        for j, b in boxes:
            if i < j:
                assert not a.overlaps(b), f'panel rows {i} and {j} overlap at {size}'
    panel_left = min(b.x0 for _, b in boxes)
    for ax in (pick.ax1, pick.ax2, pick.ax3):
        assert ax.get_tightbbox(renderer).x1 < panel_left, f'{ax} runs into the panel at {size}'
    assert not pick.ax1.get_tightbbox(renderer).overlaps(pick.ax3.get_tightbbox(renderer))
    assert not pick.slider_asinh.valtext.get_visible()
    pick.close()


def test_make_slit_without_gui():
    pick = sip.SlitPickPlus(_cube())
    res = pick.make_slit([5, 30, 70], [8, 40, 55], line_width=3)
    assert pick.fig is None and res.geometry.line_width == 3


# ----------------------------------------------------------------------------- real data (optional)
@pytest.mark.skipif(not os.path.isdir(HRI), reason='HRIEUV test data not available')
def test_hrieuv_straight_slit_matches_old(old):
    files = sorted(glob.glob(os.path.join(HRI, '*.fits')))[:5]
    seq = sunpy.map.Map(files, sequence=True)
    sub = sunpy.map.Map([m.submap([800, 800] * u.pix, top_right=[1199, 1099] * u.pix) for m in seq],
                        sequence=True)
    xs, ys = [20, 380], [30, 270]
    ref = old.generate_slit_data_from_points(xs, ys, sub, 'SunpyMap', line_width=5, map_wcs=sub[0].wcs)
    new = sip.make_slit(sub, xs, ys, line_width=5)
    np.testing.assert_allclose(new.intensity, np.asarray(ref['slit_intensity']), rtol=1e-10)
