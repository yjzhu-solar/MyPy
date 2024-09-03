import numpy as np
import astropy
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.io import fits
from sunraster.instr.spice import read_spice_l2_fits
import sunpy.coordinates
from sunpy.coordinates import (propagate_with_solar_surface, 
                               Helioprojective, 
                               get_horizons_coord)
import sunpy.map
from scipy.interpolate import RegularGridInterpolator
from skimage.feature import match_template
import sunkit_image.coalignment as coalignment
import scipy.ndimage as ndimage
from glob import glob
import os
from pathlib import Path
import argparse
from sun_blinker import SunBlinker, ImageBlinker
import matplotlib.pyplot as plt
from astropy.visualization import (ImageNormalize, AsinhStretch)
from copy import deepcopy


def create_syn_rasters(spice_file, eui_files, spice_window,
                      save_filename=None, rotation=True,
                      solar_rotation=True,
                      cdelt1_multiplier=1):
    
    spice_dataset = read_spice_l2_fits(spice_file)

    # copy the WCSDVARR
    with fits.open(spice_file) as hduls:
        spice_solarx_shift = hduls[-2].data.copy()
        spice_solary_shift = hduls[-1].data.copy()
    
    if isinstance(spice_window, int):
        spice_window = spice_dataset[list(spice_dataset.keys())[0]]
    elif isinstance(spice_window, str):
        try:
            spice_window = spice_dataset[spice_window]
        except KeyError:
            raise KeyError(f'{spice_window} not found in the SPICE file. Available windows are {list(spice_dataset.keys())}')
    else:
        raise ValueError('spice_window must be either an int or a string')
    

    spice_wcs = spice_window.wcs.dropaxis(2)
    if cdelt1_multiplier != 1:
        spice_wcs.wcs.cdelt[0] = spice_wcs.wcs.cdelt[0]*cdelt1_multiplier
        spice_wcs.wcs.pc[0,1] = spice_wcs.wcs.pc[0,1]/cdelt1_multiplier
        spice_wcs.wcs.pc[1,0] = spice_wcs.wcs.pc[1,0]*cdelt1_multiplier

    spice_nx = spice_window.dimensions[-1].value.astype(int)
    spice_ny = spice_window.dimensions[-2].value.astype(int)
    spice_nt = spice_nx

    spice_time_obs = spice_window.time[0]
    solar_orbiter_loc = np.flip(get_horizons_coord('solar orbiter',
                                                   {'start':spice_time_obs[-1],
                                                    'stop':spice_time_obs[0],
                                                    'step':f'{spice_nt}'}))

    eui_syn_raster_image = np.zeros((spice_ny, spice_nx))

    eui_maps = sunpy.map.Map(eui_files)
    eui_time_obs = Time([eui_map.date for eui_map in eui_maps])

    if rotation:
        rot_angles = np.deg2rad(np.linspace(-1.0,1.0,11))
        eui_syn_raster_images = np.zeros((len(rot_angles), spice_ny, spice_nx))
        rot_matrices = [np.array([[np.cos(rot_angle), -np.sin(rot_angle)*spice_wcs.wcs.cdelt[1]/spice_wcs.wcs.cdelt[0]],
                                [np.sin(rot_angle)*spice_wcs.wcs.cdelt[0]/spice_wcs.wcs.cdelt[1], np.cos(rot_angle)]]) for rot_angle in rot_angles]
        
        for ii, rot_matrix in enumerate(rot_matrices):
            spice_wcs_rotated = deepcopy(spice_wcs)
            spice_wcs_rotated.wcs.pc[:2,:2] = np.dot(spice_wcs.wcs.pc[:2,:2], rot_matrix)

            eui_syn_raster_images[ii] = make_single_syn_raster(spice_wcs_rotated, eui_syn_raster_image, spice_time_obs, solar_orbiter_loc,
                                                                eui_maps, eui_time_obs, spice_solarx_shift, spice_solary_shift, 
                                                                solar_rotation=solar_rotation)
    else:
        eui_syn_raster_images = np.zeros((1, spice_ny, spice_nx))
        eui_syn_raster_images[0] = make_single_syn_raster(spice_wcs, eui_syn_raster_image, spice_time_obs, solar_orbiter_loc,
                                                          eui_maps, eui_time_obs, spice_solarx_shift, spice_solary_shift, 
                                                          solar_rotation=solar_rotation)
    
    if save_filename is not None:
        np.savez_compressed(save_filename, eui_syn_raster_images=eui_syn_raster_images)
    
    return eui_syn_raster_images


def make_single_syn_raster(spice_wcs, eui_syn_raster_image, spice_time_obs, solar_orbiter_loc,
                           eui_maps, eui_time_obs, spice_solarx_shift, spice_solary_shift, 
                           solar_rotation=True):
    
    eui_syn_raster_image = np.zeros_like(eui_syn_raster_image)
    spice_ny, spice_nx = eui_syn_raster_image.shape

    spice_pixy, spice_pixx, spice_pixt = np.indices((spice_ny, spice_nx, 1))

    spice_skycoord_rough = spice_wcs.pixel_to_world(spice_pixx, spice_pixy, spice_pixt)[0][:,:,0]

    for ii in range(eui_syn_raster_image.shape[1]):
        if solar_rotation:
            spice_skycoord = SkyCoord(spice_skycoord_rough[:,ii].Tx.to(u.arcsec) + spice_solarx_shift[ii]*u.arcsec, 
                                    spice_skycoord_rough[:,ii].Ty.to(u.arcsec) + spice_solary_shift[ii]*u.arcsec,
                                    frame='helioprojective',obstime=spice_time_obs[ii], 
                                    observer=solar_orbiter_loc[ii], 
                                    rsun=eui_maps[0].meta['rsun_ref']*u.m,)
        else:
            spice_skycoord = SkyCoord(spice_skycoord_rough[:,ii].Tx.to(u.arcsec) + spice_solarx_shift[ii]*u.arcsec, 
                                    spice_skycoord_rough[:,ii].Ty.to(u.arcsec) + spice_solary_shift[ii]*u.arcsec,
                                    frame='helioprojective',obstime=spice_time_obs[0], 
                                    observer=solar_orbiter_loc[0], 
                                    rsun=eui_maps[0].meta['rsun_ref']*u.m,)

        eui_map_index = find_closest_frame(spice_time_obs[ii], eui_time_obs)

        with propagate_with_solar_surface(rotation_model='rigid'):
            spice_skycoord_pixel = eui_maps[eui_map_index].wcs.world_to_pixel(spice_skycoord)

            eui_map_interpolator = RegularGridInterpolator((np.arange(eui_maps[eui_map_index].data.shape[0]),
                                                            np.arange(eui_maps[eui_map_index].data.shape[1])),
                                                            eui_maps[eui_map_index].data, bounds_error=False, 
                                                            method="linear")
            eui_syn_raster_image[:,ii] = eui_map_interpolator((spice_skycoord_pixel[1], spice_skycoord_pixel[0]))
        
    return eui_syn_raster_image

def calculate_eui_spice_shift(spice_file, eui_files, spice_window, eui_syn_raster_images,
                              rotation=True, cdelt1_multiplier=1):
    spice_dataset = read_spice_l2_fits(spice_file)
    
    if isinstance(spice_window, int):
        spice_window = spice_dataset[list(spice_dataset.keys())[0]]
    elif isinstance(spice_window, str):
        try:
            spice_window = spice_dataset[spice_window]
        except KeyError:
            raise KeyError(f'{spice_window} not found in the SPICE file. Available windows are {list(spice_dataset.keys())}')
    else:
        raise ValueError('spice_window must be either an int or a string')
    
    if isinstance(eui_syn_raster_images, str):
        eui_syn_raster_images = np.load(eui_syn_raster_image)['eui_syn_raster_images']
    
    spice_wcs = spice_window.wcs.dropaxis(2)
    if cdelt1_multiplier != 1:
        spice_wcs.wcs.cdelt[0] = spice_wcs.wcs.cdelt[0]*cdelt1_multiplier
        spice_wcs.wcs.pc[0,1] = spice_wcs.wcs.pc[0,1]/cdelt1_multiplier
        spice_wcs.wcs.pc[1,0] = spice_wcs.wcs.pc[1,0]*cdelt1_multiplier

    spice_nx = spice_window.dimensions[-1].value.astype(int)
    spice_ny = spice_window.dimensions[-2].value.astype(int)
    spice_nt = spice_nx

    spice_cdelt1 = spice_window.meta['CDELT1']*cdelt1_multiplier
    spice_cdelt2 = spice_window.meta['CDELT2']

    spice_time_obs = spice_window.time[0]

    spice_int_img = np.nansum(spice_window.data, axis=(0,1))

    if rotation:
        rot_angles = np.deg2rad(np.linspace(-1.0,1.0,11))
    else:
        rot_angles = [0]

    rotation_matrices = [np.array([[np.cos(rot_angle), -np.sin(rot_angle)*spice_cdelt2/spice_cdelt1],
                                    [np.sin(rot_angle)*spice_cdelt1/spice_cdelt2, np.cos(rot_angle)]]) for rot_angle in rot_angles]
    
    yshifts = []
    xshifts = []
    max_ccs = []
    
    for ii, eui_syn_raster_image in enumerate(eui_syn_raster_images[:,:,:]):
        spice_int_img_cut = spice_int_img[spice_ny//4:3*spice_ny//4, spice_nx//4:3*spice_nx//4]

        xshift, yshift, max_cc = coalign_shift_pixel(eui_syn_raster_image, spice_int_img_cut)


        yshifts.append(yshift)
        xshifts.append(xshift)
        max_ccs.append(max_cc)

    max_cc_index = np.argmax(max_ccs)
    yshift_optimal, xshift_optimal, rot_matrix_optimal, rot_angle_optimal = \
        yshifts[max_cc_index], xshifts[max_cc_index], rotation_matrices[max_cc_index], rot_angles[max_cc_index]
    
    spice_wcs_optimal = deepcopy(spice_wcs)
    spice_wcs_optimal.wcs.pc[:2,:2] = np.dot(spice_wcs.wcs.pc[:2,:2], rot_matrix_optimal)
    
    shift_reference_world_coord = spice_wcs_optimal.pixel_to_world(xshift_optimal, yshift_optimal, 0)[0]
    reference_pixel_world_coord = spice_wcs_optimal.pixel_to_world(spice_nx//4,spice_ny//4,0)[0]

    print(xshift_optimal, yshift_optimal, spice_nx//4, spice_ny//4)

    xshift_optimal_world = shift_reference_world_coord.Tx - reference_pixel_world_coord.Tx
    yshift_optimal_world = shift_reference_world_coord.Ty - reference_pixel_world_coord.Ty

    eui_syn_raster_map = sunpy.map.Map(eui_syn_raster_images[max_cc_index,:,:], spice_wcs_optimal)
    eui_syn_raster_map.plot_settings['aspect'] = eui_syn_raster_map.scale.axis2/eui_syn_raster_map.scale.axis1
    spice_int_map = sunpy.map.Map(spice_int_img, spice_wcs_optimal)
    spice_int_map = spice_int_map.shift_reference_coord(xshift_optimal_world, yshift_optimal_world)
    new_crval1, new_crval2 = spice_int_map.reference_coordinate.Tx, spice_int_map.reference_coordinate.Ty
    new_rotation_matrix = spice_int_map.rotation_matrix
    
    # new_rotation_matrix = np.dot(spice_int_map.rotation_matrix,rot_matrix_optimal)
    # new_crval1 = spice_int_map.reference_coordinate.Tx + xshift_optimal_world
    # new_crval2 = spice_int_map.reference_coordinate.Ty + yshift_optimal_world

    # spice_int_map.meta['CRVAL1'] = new_crval1.to_value(u.deg)
    # spice_int_map.meta['CRVAL2'] = new_crval2.to_value(u.deg)
    # spice_int_map.meta['PC1_1'] = new_rotation_matrix[0,0]
    # spice_int_map.meta['PC1_2'] = new_rotation_matrix[0,1]
    # spice_int_map.meta['PC2_1'] = new_rotation_matrix[1,0]
    # spice_int_map.meta['PC2_2'] = new_rotation_matrix[1,1]
    # spice_int_map.meta['CDELT1'] = spice_cdelt1/3600

    spice_int_map.meta.pop('CROTA1', None)
    spice_int_map.meta.pop('CROTA2', None)
    spice_int_map.meta.pop('CD1_1', None)
    spice_int_map.meta.pop('CD1_2', None)
    spice_int_map.meta.pop('CD2_1', None)
    spice_int_map.meta.pop('CD2_2', None)

    SunBlinker(eui_syn_raster_map, spice_int_map, reproject=True, fps=1, 
               norm1=ImageNormalize(vmin=np.nanpercentile(eui_syn_raster_image, 0.2),
                                    vmax=np.nanpercentile(eui_syn_raster_image, 99.8),
                                    stretch=AsinhStretch(0.1)),
               norm2=ImageNormalize(vmin=np.nanpercentile(spice_int_img, 0.2),
                                    vmax=np.nanpercentile(spice_int_img, 99.8),
                                    stretch=AsinhStretch(0.1)),)
    plt.show()

    save_new_spice_file(spice_file, new_crval1, new_crval2, new_rotation_matrix, cdelt1_multiplier=cdelt1_multiplier)

    return xshift_optimal_world, yshift_optimal_world, rot_matrix_optimal, np.rad2deg(rot_angle_optimal) 

def save_new_spice_file(spice_file, crval1, crval2, rotation_matrix, cdelt1_multiplier, outdir=None, filename=None):
    with fits.open(spice_file) as hduls:
        for hdul in hduls:
            if 'CRVAL1' in hdul.header.keys():
                hdul.header = update_header(hdul.header, crval1, crval2, rotation_matrix, cdelt1_multiplier)
        if outdir is None:
            outdir = Path(spice_file).parent
        if filename is None:
            filename = Path(spice_file).stem + '_coalign.fits'

        hduls.writeto(outdir/filename, overwrite=True)

def update_header(hdr, crval1, crval2, rotation_matrix, cdelt1_multiplier):
    hdr_new = hdr.copy()
    hdr_new['CRVAL1'] = crval1.to_value(u.arcsec)
    hdr_new['CRVAL2'] = crval2.to_value(u.arcsec)
    hdr_new['PC1_1'] = rotation_matrix[0,0]
    hdr_new['PC1_2'] = rotation_matrix[0,1]
    hdr_new['PC2_1'] = rotation_matrix[1,0]
    hdr_new['PC2_2'] = rotation_matrix[1,1]
    hdr_new['CDELT1'] = hdr_new['CDELT1']*cdelt1_multiplier

    hdr_new.pop('CROTA', None)
    hdr_new.pop('CROTA1', None)
    hdr_new.pop('CROTA2', None)
    hdr_new.pop('CD1_1', None)
    hdr_new.pop('CD1_2', None)
    hdr_new.pop('CD2_1', None)
    hdr_new.pop('CD2_2', None)

    hdr_new['COMMENT'] = ['Co-aligned with EUI synoptic raster',
                            'CRVAL1 and CRVAL2 updated',
                            'PC1_1, PC1_2, PC2_1, PC2_2 updated',
                            'CROTA, CROTA1, CROTA2, CD1_1, CD1_2, CD2_1, CD2_2 removed']
    hdr_new['HISTORY'] = 'euispice_coalign.py'

    return hdr_new

def find_closest_frame(select_time, time_seqence, light_travel_corr = 0*u.s):
    return np.argmin(np.abs(select_time - time_seqence + light_travel_corr))

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
        to the input array.
    """
    # Warn user if any NANs, Infs, etc are present in the layer or the template
    coalignment._check_for_nonfinite_entries(this_layer, template)
    # Calculate the correlation array matching the template to this layer
    corr = match_template(this_layer, template)
    # Calculate the y and x shifts in pixels
    best_match = coalignment._find_best_match_location(corr)
    # Calculate the maximum cross-correlation value
    max_cc = np.max(corr)

    return *best_match, max_cc


def coalign_shift_pixel(big_map, small_map):
    yshift, xshift, max_cc = _calculate_shift(big_map, small_map)
    return xshift.to_value(u.pix), yshift.to_value(u.pix), max_cc
    

if __name__ == '__main__':
    # eui_files = sorted(glob('../../Solar/EIS_DKIST_SolO/src/EUI/FSI/euv174/20221024/for_spice/*.fits'))
    # # create_syn_raster('../../Solar/EIS_DKIST_SolO/src/SPICE/20221024/solo_L2_spice-n-ras_20221024T231535_V07_150995398-000.fits',
    # #                   eui_files,'Ne VIII 770 - Peak','/home/yjzhu/Downloads/test.npz')
    # xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal = \
    #     calculate_eui_spice_shift('../../Solar/EIS_DKIST_SolO/src/SPICE/20221024/solo_L2_spice-n-ras_20221024T231535_V07_150995398-000.fits',
    #                                 eui_files,'Ne VIII 770 - Peak','/home/yjzhu/Downloads/test.npz', rotation=True)
    # print(xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal)

    '''
    example:
    python euispice_coalign.py path_to_spice path_to_eui_file_dir -w 'Ne VIII 770 - Peak' -s 'save_filename'
    '''

    parser = argparse.ArgumentParser(description='Co-align EUI synoptic raster with SPICE data')
    parser.add_argument('spice_file', type=str, help='SPICE file')
    parser.add_argument('eui_files', type=str, help='EUI files')
    parser.add_argument('-w','--spice_window', type=str, default='Ne VIII 770 - Peak', help='SPICE window')
    parser.add_argument('-s','--save_filename', type=str, default=None, help='Save filename')
    parser.add_argument('-r','--rotation', action='store_false', help='Rotation')
    parser.add_argument('-sr','--solar_rotation', action='store_false', help='Solar rotation')
    parser.add_argument('-o','--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('-sf','--synthetic_rater_filename', type=str, default=None, help='Filename of synthetic raster')
    parser.add_argument('-c1','--cdelt1', type=float, default=1, help='CDELT1 Multiplier')

    args = parser.parse_args()

    if "fits" in args.eui_files:
        eui_files = sorted(glob(args.eui_files))
    else:
        eui_files = sorted(glob(os.path.join(args.eui_files, '*.fits')))

    if args.synthetic_rater_filename is None:
        synthetic_rater_filename = os.path.join(os.path.dirname(eui_files[0]), 'eui_syn_raster_image_for_spice.npz')
        eui_syn_raster_images = create_syn_rasters(args.spice_file, eui_files, args.spice_window, synthetic_rater_filename,
                                                 solar_rotation=args.solar_rotation, cdelt1_multiplier=args.cdelt1)
    
    xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal = \
        calculate_eui_spice_shift(args.spice_file, eui_files, args.spice_window, eui_syn_raster_images, rotation=args.rotation,
                                  cdelt1_multiplier=args.cdelt1)
    
    print(xshift_optimal, yshift_optimal, rot_matrix_optimal, rot_angle_optimal)











    

    





    