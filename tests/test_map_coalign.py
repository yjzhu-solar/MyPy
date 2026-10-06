"""
Tests for map_coalign.py. Run with:  python -m pytest tests/test_map_coalign.py
"""
import os
import sys
import glob

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')
from scipy.ndimage import gaussian_filter, shift

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sunpy.map
from astropy.io import fits
from map_coalign import (MapSequenceCoalign, _calculate_shift, _find_best_match_location,
                         _shift_with_nan)


def _random_image(shape=(300, 300), seed=0):
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.normal(size=shape), 3)
    return (img - img.min()) * 1000 + 100


def test_find_best_match_location_near_edges():
    layer = _random_image((60, 60))
    tpl = layer[10:30, 12:32]
    cases = {
        'interior': layer,
        'peak on second-to-last row/col': layer[0:31, 0:33],   # was off by one pixel
        'peak on last row/col': layer[0:30, 0:32],
        'peak on first row/col': layer[10:40, 12:42],
    }
    for name, lay in cases.items():
        y, x = _calculate_shift(lay, tpl)
        yexp = 10 if lay.shape[0] > 30 else (10 if name != 'peak on first row/col' else 0)
        xexp = 12 if lay.shape[1] > 32 else (12 if name != 'peak on first row/col' else 0)
        assert abs(y.value - yexp) < 0.1, name
        assert abs(x.value - xexp) < 0.1, name


def test_find_best_match_location_synthetic_corr():
    corr = np.zeros((7, 9))
    corr[5, 7] = 1.0          # second-to-last row and column
    corr[4, 7] = corr[6, 7] = 0.5
    corr[5, 6] = corr[5, 8] = 0.5
    y, x = _find_best_match_location(corr)
    assert y.value == pytest.approx(5.0)
    assert x.value == pytest.approx(7.0)


def test_shift_with_nan_mask_and_values():
    data = np.arange(100, dtype=float).reshape(10, 10)
    out = _shift_with_nan(data, 1.0, -2.0, order=1)
    assert np.all(np.isnan(out[0, :]))          # first row came from outside
    assert np.all(np.isnan(out[:, -2:]))        # last two columns came from outside
    assert np.isfinite(out[1:, :-2]).all()
    assert out[1, 0] == pytest.approx(data[0, 2])
    out = _shift_with_nan(data, 0.3, -0.3, order=1)
    assert np.all(np.isnan(out[0, :])) and np.all(np.isnan(out[:, -1]))
    assert np.isfinite(out[1:, :-1]).all()


def _make_sequence(nt, shifts, shape=(256, 256)):
    base = _random_image((shape[0] + 40, shape[1] + 40), seed=1)
    maps = []
    for i in range(nt):
        dy, dx = shifts[i]
        frame = shift(base, (dy, dx), order=3, mode='nearest')[20:20 + shape[0], 20:20 + shape[1]]
        meta = {'cdelt1': 1.0, 'cdelt2': 1.0, 'crota2': 0.0, 'crpix1': 1, 'crpix2': 1,
                'crval1': 0, 'crval2': 0, 'ctype1': 'HPLN-TAN', 'ctype2': 'HPLT-TAN',
                'cunit1': 'arcsec', 'cunit2': 'arcsec', 'naxis1': shape[1], 'naxis2': shape[0],
                'date-obs': f'2020-01-01T00:00:{i:02d}.000', 'rsun_ref': 696000000.0,
                'hgln_obs': 0.0, 'hglt_obs': 0.0, 'dsun_obs': 1.5e11}
        maps.append(sunpy.map.Map(frame.astype(np.float32), meta))
    return MapSequenceCoalign(maps)


@pytest.mark.parametrize('method', ['match_template', 'phase'])
def test_coalign_recovers_known_shifts_partial_last_segment(method):
    rng = np.random.default_rng(3)
    nt = 13                                   # 13 % 5 != 0 -> partial last chunk
    shifts = rng.uniform(-2.5, 2.5, size=(nt, 2))
    shifts[0] = 0
    ms = _make_sequence(nt, shifts)
    ms.coalign(reference_index=0, bottom_left=[40, 40], top_right=[216, 216],
               check_header=True, nframes=5, iter=3, method=method, search_margin=10)
    # measured shift = structure displacement relative to frame 0
    assert np.allclose(ms.yshift_total, shifts[:, 0], atol=0.03), method
    assert np.allclose(ms.xshift_total, shifts[:, 1], atol=0.03), method
    # last frame must have been aligned too (regression for the nt-1 bug)
    assert abs(ms.yshift_total[-1] - shifts[-1, 0]) < 0.03
    # reference frame untouched, others carry XSHIFT/YSHIFT and NaN borders
    assert ms[0].meta['xshift'] == 0 and np.isfinite(ms[0].data).all()
    assert ms[5].meta['xshift'] == pytest.approx(ms.xshift_total[5])
    assert np.isnan(ms[5].data).any()
    # aligned frames agree with the reference in the interior
    for i in range(1, nt):
        d = ms[i].data[20:-20, 20:-20] - ms[0].data[20:-20, 20:-20]
        assert np.nanstd(d) < 0.02 * np.nanstd(ms[0].data), i


def test_coalign_argument_validation():
    ms = _make_sequence(3, np.zeros((3, 2)))
    with pytest.raises(ValueError):
        ms.coalign(bottom_left=[10, 10], check_header=False)
    with pytest.raises(ValueError):
        ms.coalign(bottom_left=[10, 10], top_right=[5, 5], check_header=False)
    with pytest.raises(ValueError):
        ms.coalign(bottom_left=[10, 10], top_right=[50, 50], check_header=False, method='nope')
    with pytest.raises(TypeError):
        ms[0] = np.zeros((3, 3))


def test_plot_no_wcs_cmap_and_axes():
    import matplotlib.pyplot as plt
    ms = _make_sequence(3, np.zeros((3, 2)))
    fig, ax = plt.subplots()
    ani = ms.plot(axes=ax, no_wcs=True, cmap='gray')
    im = ax.images[0]
    ani._func(1, *ani._args)
    assert im.get_cmap().name == 'gray'
    plt.close('all')
    ani = ms.plot()
    assert ani is not None
    plt.close('all')


def test_save_compressed_lossless(tmp_path):
    ms = _make_sequence(2, np.zeros((2, 2)))
    ms.coalign(bottom_left=[40, 40], top_right=[216, 216], check_header=False,
               nframes=5, iter=1, search_margin=5)
    ms[1] = sunpy.map.Map(_shift_with_nan(ms[1].data, 0.3, -0.7), ms[1].meta)
    out = str(tmp_path / 'c_{index:02}.fits')
    ms.save(out, overwrite=True, compress=True, quantize_level=0)
    with fits.open(tmp_path / 'c_01.fits') as h:
        assert isinstance(h[1], fits.CompImageHDU)
        assert h[1].compression_type == 'GZIP_1'
        back = h[1].data
    np.testing.assert_array_equal(back, ms[1].data)       # lossless incl. NaN
    assert fits.getheader(tmp_path / 'c_01.fits', 1)['XSHIFT'] == pytest.approx(ms.xshift_total[1])
    assert (tmp_path / 'coalign_info.h5').exists()
    ms.save(str(tmp_path / 'u_{index:02}.fits'), overwrite=True)
    np.testing.assert_array_equal(fits.getdata(tmp_path / 'u_01.fits'), ms[1].data)
