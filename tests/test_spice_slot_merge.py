"""
Round-trip tests for spice_slot_merge: simulate slot images of a known scene
with the SPICE geometry (roll, dispersion coupling, jitter, pedestal), merge
them, and compare with the scene.
"""
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
import astropy.units as u

import spice_slot_merge as ssm

ROLL = np.deg2rad(7.584)


def _header(n_steps=8, ny=300, nk=60, step=30.0, scale=0.98, roll=ROLL,
            crval=(-2358.0, -1268.0)):
    h = fits.Header()
    h['EXTNAME'] = 'Test 770 (Merged)'
    h['NAXIS'] = 4
    h['NAXIS1'], h['NAXIS2'], h['NAXIS3'], h['NAXIS4'] = n_steps, ny, nk, 1
    h['CTYPE1'], h['CTYPE2'] = 'HPLN-TAN', 'HPLT-TAN'
    h['CUNIT1'], h['CUNIT2'] = 'arcsec', 'arcsec'
    h['CDELT1'], h['CDELT2'] = step, 1.098
    h['CRPIX1'], h['CRPIX2'], h['CRPIX3'] = (n_steps + 1) / 2, (ny + 1) / 2, (nk + 1) / 2
    h['CRVAL1'], h['CRVAL2'] = crval
    h['PC1_1'] = np.cos(roll)
    h['PC1_2'] = -np.sin(roll) * h['CDELT2'] / h['CDELT1']
    h['PC2_1'] = np.sin(roll) * h['CDELT1'] / h['CDELT2']
    h['PC2_2'] = np.cos(roll)
    h['PC1_3'] = scale / step           # as in the L2 files: no roll, no PC2_3
    h['CRVAL3'], h['CDELT3'], h['WAVEUNIT'] = 77.0, 0.00975, -9
    h['CRVAL4'], h['CDELT4'], h['CRPIX4'], h['PC4_1'] = 23.5, 1.0, 1.0, -6.0
    h['DATE-BEG'] = h['DATEREF'] = h['DATE-OBS'] = '2023-04-10T22:39:24.410'
    h['DATE-AVG'] = '2023-04-10T22:39:48.298'
    h['BUNIT'] = 'W m-2 sr-1'
    return h


def _scene(lon_deg, lat_deg):
    """Smooth structured test scene in helioprojective coordinates."""
    x = (np.mod(lon_deg + 180, 360) - 180) * 3600 + 2358.0
    y = lat_deg * 3600 + 1268.0
    return (2.0 + np.sin(x / 9.0) * np.cos(y / 13.0) + 0.5 * np.sin((x + 2 * y) / 21.0)
            + 0.002 * x)


def _simulate(header, response, roll_corrected=True, jitter=None, pedestal=0.0):
    n_steps, ny, nk = header['NAXIS1'], header['NAXIS2'], header['NAXIS3']
    jitter = np.zeros((n_steps, 2)) if jitter is None else jitter
    resp = response.array()
    jj, kk = np.indices((ny, nk))
    cube = np.empty((n_steps, ny, nk))
    for i in range(n_steps):
        w = ssm.slot_wcs(header, i, response.k_center, roll_corrected=roll_corrected,
                         jitter=jitter[i], tilt=response.tilt, j_ref=response.j_ref)
        lon, lat = w.pixel_to_world_values(kk, jj)
        cube[i] = _scene(lon, lat) * resp + pedestal * (1 + 0.3 * (jj / ny))
    return ssm.SlotRaster(filename='synthetic', header=header, data=cube,
                          times=ssm._step_times(header), jitter=jitter,
                          radcal=np.ones(nk))


def _response(header, k_left=14.7, k_right=45.3, sigma=2.5, tilt=0.0):
    ny, nk = header['NAXIS2'], header['NAXIS3']
    return ssm.SlotResponse(window=header['EXTNAME'], shape=(ny, nk), k_left=k_left,
                            k_right=k_right, sigma_left=sigma, sigma_right=sigma,
                            j_bottom=20.0, j_top=ny - 21.0, sigma_bottom=1.5, sigma_top=1.5,
                            tilt=tilt, j_ref=(ny - 1) / 2)


def _compare(hdul, min_weight=0.95):
    data = hdul[0].data
    weight = hdul['WEIGHT'].data
    w = WCS(hdul[0].header)
    yy, xx = np.indices(data.shape)
    lon, lat = w.pixel_to_world_values(xx, yy)
    truth = _scene(lon, lat)
    inside = weight >= min_weight
    return np.nanmax(np.abs(data[inside] / truth[inside] - 1)), inside.sum()


@pytest.mark.parametrize('grid', ['native', 'north'])
def test_round_trip(grid):
    header = _header()
    resp = _response(header)
    jitter = np.column_stack([np.linspace(-0.4, 0.3, 8), np.linspace(0.3, -0.2, 8)])
    raster = _simulate(header, resp, jitter=jitter)
    hdul = ssm.merge_slot_raster(raster, resp, grid=grid, background=False, uncertainty=False)
    err, n = _compare(hdul)
    assert n > 1000
    # Bilinear interpolation of a scene with ~9 arcsec structure
    assert err < 0.03
    # The coverage is ~1 across seams: steps equal the slot width
    weight = hdul['WEIGHT'].data
    assert np.nanmedian(weight[weight > 0.5]) == pytest.approx(1.0, abs=0.02)


def test_header_dispersion_is_wrong_for_rolled_data():
    """Merging roll-corrected data with the L2 coupling leaves 4" seams."""
    header = _header()
    resp = _response(header)
    raster = _simulate(header, resp, roll_corrected=True)
    good = ssm.merge_slot_raster(raster, resp, background=False, uncertainty=False)
    bad = ssm.merge_slot_raster(raster, resp, background=False, uncertainty=False,
                                roll_corrected=False)
    assert _compare(bad)[0] > 3 * _compare(good)[0]


def test_pedestal_subtraction():
    header = _header()
    resp = _response(header)
    raster = _simulate(header, resp, pedestal=0.6)
    with_bg = ssm.merge_slot_raster(raster, resp, background=True, uncertainty=False)
    without = ssm.merge_slot_raster(raster, resp, background=False, uncertainty=False)
    assert _compare(with_bg)[0] < 0.04
    assert _compare(without)[0] > 0.2


def test_tilt_round_trip():
    header = _header()
    resp = _response(header, tilt=0.004)          # 1 px drift over 250 rows
    raster = _simulate(header, resp)
    hdul = ssm.merge_slot_raster(raster, resp, background=False, uncertainty=False)
    assert _compare(hdul)[0] < 0.03


def test_step_times():
    header = _header()
    t = ssm._step_times(header)
    # last raster pixel is observed first, 6 s per step, mid-exposure 2.5 s
    assert (t[-1] - Time(header['DATE-BEG'])).to_value(u.s) == pytest.approx(2.5)
    assert (t[0] - t[1]).to_value(u.s) == pytest.approx(6.0)


def test_build_response_recovers_edges(tmp_path, monkeypatch):
    header = _header(ny=400)
    truth = _response(header, k_left=15.2, k_right=45.8, sigma=3.0)
    rng = np.random.default_rng(1)
    rasters = []
    for n in range(6):
        hdr = header.copy()
        hdr['CRVAL1'] += 40 * n       # different scenes
        r = _simulate(hdr, truth, pedestal=0.4)
        r.data += rng.normal(0, 0.02, r.data.shape)
        rasters.append(r)
    monkeypatch.setattr(ssm, 'read_slot_raster', lambda f, w: rasters[int(f)])
    fitted = ssm.build_slot_response([str(n) for n in range(6)], 0, verbose=False)
    assert fitted.k_left == pytest.approx(truth.k_left, abs=0.4)
    assert fitted.k_right == pytest.approx(truth.k_right, abs=0.4)
    assert fitted.sigma_left == pytest.approx(3.0, abs=0.5)
    assert fitted.j_bottom == pytest.approx(truth.j_bottom, abs=1.0)
    assert fitted.j_top == pytest.approx(truth.j_top, abs=1.0)
    assert fitted.tilt == 0.0
    # Round trip through FITS
    fitted.to_fits(tmp_path / 'resp.fits')
    again = ssm.SlotResponse.from_fits(tmp_path / 'resp.fits')
    assert again.k_left == pytest.approx(fitted.k_left)
    np.testing.assert_allclose(again.array(), fitted.array(), atol=1e-6)


# ---------------------------------------------------------------------------
# End-to-end: co-alignment of a merged slot raster (euispice_coalign)
# ---------------------------------------------------------------------------

def _observer_keywords(h):
    h['DATE-END'] = '2023-04-10T22:40:11.409'
    h['RSUN_REF'] = 6.957e8
    h['DSUN_OBS'] = 43791609915.3
    h['HGLN_OBS'] = 67.9449213917
    h['HGLT_OBS'] = -2.53164541797
    h['XPOSURE'] = 5.0
    return h


def _fake_eui(path, scene, date, crval=(-2358.0, -1268.0), n=700, scale=1.0):
    """EUI-L2-like file (image in HDU 1) of ``scene`` on a north-up grid."""
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'] = 'HPLN-TAN', 'HPLT-TAN'
    h['CUNIT1'], h['CUNIT2'] = 'arcsec', 'arcsec'
    h['CDELT1'] = h['CDELT2'] = scale
    h['CRPIX1'] = h['CRPIX2'] = (n + 1) / 2
    h['CRVAL1'], h['CRVAL2'] = crval
    h['DATE-OBS'] = h['DATE-AVG'] = date
    h['RSUN_REF'] = 6.957e8
    h['DSUN_OBS'] = 43791609915.3
    h['HGLN_OBS'] = 67.9449213917
    h['HGLT_OBS'] = -2.53164541797
    h['DETECTOR'], h['WAVELNTH'], h['WAVEUNIT'] = 'FAKE', 174, 'Angstrom'
    w = WCS(h)
    yy, xx = np.indices((n, n))
    lon, lat = w.pixel_to_world_values(xx, yy)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(scene(lon, lat).astype(np.float32),
                                                   header=h)]).writeto(path)
    return str(path)


def test_slot_coalignment_recovers_pointing(tmp_path):
    ec = pytest.importorskip('euispice_coalign')
    from scipy.ndimage import gaussian_filter, map_coordinates
    rng = np.random.default_rng(7)
    field = gaussian_filter(rng.normal(size=(1600, 1600)), 4)

    def scene(lon, lat):
        x = (np.mod(lon + 180, 360) - 180) * 3600 + 2358.0 + 800
        y = lat * 3600 + 1268.0 + 800
        return 5 + map_coordinates(field, [y, x], order=1, mode='nearest')

    header = _observer_keywords(_header(ny=500))
    resp = _response(header, k_left=14.4, k_right=45.0, sigma=2.0)
    old = _scene
    try:
        globals()['_scene'] = scene
        raster = _simulate(header, resp)
    finally:
        globals()['_scene'] = old
    hdul = ssm.merge_slot_raster(raster, resp, background=False, uncertainty=False)
    # Mis-point the merged map: the truth is CRVAL - (12, -7) arcsec
    for hdu in hdul:
        if hdu.header.get('CTYPE1', '').startswith('HPLN'):
            hdu.header['CRVAL1'] += 12.0
            hdu.header['CRVAL2'] -= 7.0
    slot_file = tmp_path / 'slot.fits'
    hdul.writeto(slot_file)
    eui = _fake_eui(tmp_path / 'eui.fits', scene, '2023-04-10T22:39:48.298')

    syn = ec.create_syn_slot_images(slot_file, [eui], rotation=False, blur=None)
    res = ec.calculate_eui_spice_slot_shift(slot_file, syn, rotation=False,
                                            output_dir=tmp_path)
    # Rotation over <= 24 s moves features by <= 0.15 arcsec at this distance
    assert res['dx'] == pytest.approx(-12.0, abs=0.5)
    assert res['dy'] == pytest.approx(7.0, abs=0.5)
    out = fits.getheader(res['output'], 0)
    assert out['CRVAL1'] == pytest.approx(header['CRVAL1'], abs=0.5)
    assert out['CRVAL2'] == pytest.approx(header['CRVAL2'], abs=0.5)
    # every image extension carries the corrected WCS
    for ext in ('WEIGHT', 'TIME'):
        assert fits.getheader(res['output'], ext)['CRVAL1'] == pytest.approx(out['CRVAL1'])
