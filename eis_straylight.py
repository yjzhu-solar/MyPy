import numpy as np
import matplotlib.pyplot as plt
import sunpy
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from regions import (EllipseAnnulusPixelRegion, PixCoord,
                     EllipsePixelRegion)
import sunkit_image.coalignment as coalignment
import warnings
from PyDynamic.uncertainty.interpolate import interp1d_unc

def calc_short_range_stray_light(eismap, center_coord, inner_radius=30*u.arcsec,
                                 outer_radius=50*u.arcsec, return_region=False,
                                 alpha=6.6):
    
    mask, region = calc_short_range_annulus_mask(eismap, center_coord, inner_radius, outer_radius)

    intensity_annulus = np.nanmean(eismap.data[mask])
    int_sr_straylight = intensity_annulus/alpha

    if return_region:
        return int_sr_straylight, region
    else:
        return int_sr_straylight
    
def calc_short_range_stray_light_profiles(eismap, eis_cube, center_coord, 
                                          inner_radius=30*u.arcsec,
                                          outer_radius=50*u.arcsec, 
                                          return_region=False,
                                          return_center_profile=False,
                                          alpha=6.6):
    
    mask, region = calc_short_range_annulus_mask(eismap, center_coord, inner_radius, outer_radius)
    profiles_annulus = eis_cube.data[mask]
    profiles_wavelength = eis_cube.wavelength[mask]
    profiles_err = eis_cube.uncertainty.array[mask]

    # find the best common wavelength range to interpolate
    profiles_wavelength_min = np.nanmax(profiles_wavelength[:,0], axis=0)
    profiles_wavelength_max = np.nanmin(profiles_wavelength[:,-1], axis=0)

    center_pixel_ix, center_pixel_iy = eismap.world_to_pixel(center_coord)
    center_pixel_ix, center_pixel_iy = center_pixel_ix.to_value(u.pix), center_pixel_iy.to_value(u.pix)
    center_pixel_ix, center_pixel_iy = np.round(center_pixel_ix).astype(int), np.round(center_pixel_iy).astype(int)

    center_profile = eis_cube.data[center_pixel_iy, center_pixel_ix, :]
    center_wavelength = eis_cube.wavelength[center_pixel_iy, center_pixel_ix, :]
    center_profile_err = eis_cube.uncertainty.array[center_pixel_iy, center_pixel_ix, :]

    wvl_min_index = np.where(center_wavelength >= profiles_wavelength_min)[0][0]
    wvl_max_index = np.where(center_wavelength <= profiles_wavelength_max)[0][-1]

    common_wavelength = center_wavelength[wvl_min_index:wvl_max_index+1]
    center_profile = center_profile[wvl_min_index:wvl_max_index+1]
    center_profile_err = center_profile_err[wvl_min_index:wvl_max_index+1]

    # interpolate the profiles to the common wavelength range
    profiles_interp = np.zeros((profiles_annulus.shape[0], common_wavelength.size))
    profiles_interp_err = np.zeros((profiles_annulus.shape[0], common_wavelength.size))
    for ii in range(profiles_annulus.shape[0]):
        _, profiles_interp[ii,:], profiles_interp_err[ii,:] = interp1d_unc(common_wavelength,
                                                                        profiles_wavelength[ii,:],
                                                                       profiles_annulus[ii,:],
                                                                       profiles_err[ii,:],
                                                                       kind='cubic')

    profiles_annulus_mean = np.nanmean(profiles_interp, axis=0)/alpha
    profiles_annulus_mean_err = np.sqrt(np.nansum(profiles_interp_err**2, axis=0))/np.sum(np.isfinite(profiles_interp), axis=0)/alpha

    if return_region or return_center_profile:
        aux_dict = {}
        if return_region:
            aux_dict['region'] = region
        if return_center_profile:
            aux_dict['center_profile'] = center_profile
            aux_dict['center_profile_err'] = center_profile_err
        
        return profiles_annulus_mean, profiles_annulus_mean_err, common_wavelength, aux_dict
    else:
        return profiles_annulus_mean, profiles_annulus_mean_err, common_wavelength

def calc_short_range_annulus_mask(eismap, center_coord, inner_radius=30*u.arcsec,
                                  outer_radius=50*u.arcsec):
    if eismap.measurement != 'intensity':
        raise ValueError('EIS map must be an intensity map')

    if isinstance(center_coord, SkyCoord):
        center_coord_pix = PixCoord.from_sky(center_coord, eismap.wcs)

        if eismap.date - center_coord.obstime > 1800*u.s:
            warnings.warn('Center coordinate is more than 30 minutes from the EIS map time, '
                        'Please consider using the sunpy.coordinate.propagate_with_solar_surface() '
                        'context manager to account for solar rotation')
    elif isinstance(center_coord, PixCoord):
        center_coord_pix = center_coord
    elif isinstance(center_coord, u.Quantity):
        if center_coord.unit == u.pix:
            center_coord_pix = PixCoord(center_coord)
        elif center_coord.unit == u.arcsec:
            center_coord_pix = PixCoord.from_sky(SkyCoord(*center_coord,frame=eismap.coordinate_frame), eismap.wcs)
    else:
        raise ValueError('center_coord must be a SkyCoord, PixCoord, tuple, list or u.Quantity')
    
    if not (isinstance(inner_radius, u.Quantity) and isinstance(outer_radius, u.Quantity)):
        raise ValueError('inner_radius and outer_radius must be astropy Quantities')
    
    region = EllipseAnnulusPixelRegion(center_coord_pix, 
                                    inner_width=(inner_radius/eismap.scale.axis1).to_value(u.pix),
                                        outer_width=(outer_radius/eismap.scale.axis1).to_value(u.pix),
                                        inner_height=(inner_radius/eismap.scale.axis2).to_value(u.pix),
                                        outer_height=(outer_radius/eismap.scale.axis2).to_value(u.pix),
                                        )
    
    mask = region.contains(PixCoord.from_sky(sunpy.map.all_coordinates_from_map(eismap), eismap.wcs))

    return mask, region


def calc_long_range_stray_light_aia_eis(eismap, aiamap, center_coord, region_radius=30*u.arcsec,
                                        return_region=False, beta=34.):
    
    if eismap.measurement != 'intensity':
        raise ValueError('EIS map must be an intensity map')
    
    if isinstance(center_coord, SkyCoord):
        center_coord_pix_eis = PixCoord.from_sky(center_coord, eismap.wcs)
        center_coord_pix_aia = PixCoord.from_sky(center_coord, aiamap.wcs)

        if eismap.date - center_coord.obstime > 1800*u.s:
            warnings.warn('Center coordinate is more than 30 minutes from the EIS map time, '
                        'Please consider using the sunpy.coordinate.propagate_with_solar_surface() '
                        'context manager to account for solar rotation')
    elif isinstance(center_coord, PixCoord):
        center_coord_pix_eis = center_coord
        center_coord_pix_aia = PixCoord.from_sky(eismap.pixel_to_world(center_coord), aiamap.wcs)
    elif isinstance(center_coord, u.Quantity):
        if center_coord.unit == u.pix:
            center_coord_pix_eis = PixCoord(*center_coord)
            center_coord_pix_aia = PixCoord.from_sky(eismap.wcs.pixel_to_world(*center_coord), aiamap.wcs)
        elif center_coord.unit == u.arcsec:
            center_coord_pix_eis = PixCoord.from_sky(SkyCoord(*center_coord,frame=eismap.coordinate_frame), eismap.wcs)
            center_coord_pix_aia = PixCoord.from_sky(SkyCoord(*center_coord,frame=eismap.coordinate_frame), aiamap.wcs)
    else:
        raise ValueError('center_coord must be a SkyCoord, PixCoord, tuple, list or u.Quantity')
    
    if not isinstance(region_radius, u.Quantity):
        raise ValueError('region_radius must be an astropy Quantity')
    
    if eismap.date - aiamap.date > 1800*u.s:
        warnings.warn('AIA map is more than 30 minutes from the EIS map time, '
                    'Please consider using the sunpy.coordinate.propagate_with_solar_surface() '
                    'context manager to account for solar rotation')
        

    eis_region = EllipsePixelRegion(center_coord_pix_eis, 
                                    width=(region_radius/eismap.scale.axis1).to_value(u.pix),
                                    height=(region_radius/eismap.scale.axis2).to_value(u.pix))
    aia_region = EllipsePixelRegion(center_coord_pix_aia,
                                    width=(region_radius/aiamap.scale.axis1).to_value(u.pix),
                                    height=(region_radius/aiamap.scale.axis2).to_value(u.pix))
    
    aiamap_all_coords = sunpy.map.all_coordinates_from_map(aiamap)
    mask_eis = eis_region.contains(PixCoord.from_sky(sunpy.map.all_coordinates_from_map(eismap), eismap.wcs))
    mask_aia = aia_region.contains(PixCoord.from_sky(aiamap_all_coords, aiamap.wcs))

    mask_aia_full_disk = np.arccos(np.cos(aiamap_all_coords.Tx) * np.cos(aiamap_all_coords.Ty)) <= 1.05 * aiamap.rsun_obs

    intensity_eis = np.nanmean(eismap.data[mask_eis])
    intensity_aia = np.nanmean(aiamap.data[mask_aia])

    int_aia_full_disk = np.nanmean(aiamap.data[mask_aia_full_disk])
    int_eis_full_disk = int_aia_full_disk/intensity_aia*intensity_eis
    int_lr_straylight = int_eis_full_disk/beta

    if return_region:
        return int_lr_straylight, eis_region, aia_region
    else:
        return int_lr_straylight


def coalign_eis_aia(eismap, aiamap):
    aia_resample_nx = (aiamap.scale.axis1 * aiamap.dimensions.x) / eismap.scale.axis1
    aia_resample_ny = (aiamap.scale.axis2 * aiamap.dimensions.y) / eismap.scale.axis2
    aia_map_resample = aiamap.resample(u.Quantity([aia_resample_nx, aia_resample_ny]))

    eis_to_aia_Txshift, eis_to_aia_Tyshift = coalign_shift(aia_map_resample, eismap)

    return eis_to_aia_Txshift, eis_to_aia_Tyshift



def coalign_shift(big_map, small_map):
    yshift, xshift = coalignment._calculate_shift(big_map.data, small_map.data)
    reference_coord = big_map.pixel_to_world(xshift, yshift)
    Txshift = reference_coord.Tx - small_map.bottom_left_coord.Tx
    Tyshift = reference_coord.Ty - small_map.bottom_left_coord.Ty

    return Txshift, Tyshift

if __name__ == '__main__':
    import eispac

    eis_fitres = eispac.read_fit('/home/yjzhu/Solar/EIS_DKIST_SolO/src/EIS/DHB_007_v2/20221025T0023/eis_20221025_014811.fe_12_195_119.1c-0.fit.h5')
    eis_map = eis_fitres.get_map(component=0, measurement='intensity')

    int_annulus, annulus_region = calc_short_range_stray_light(eis_map, eis_map.center, return_region=True)

    fig = plt.figure(layout='constrained')
    ax = fig.add_subplot(projection=eis_map)
    eis_map.plot(axes=ax, cmap='plasma')

    annulus_region.plot(ax=ax, edgecolor='red', facecolor='none')
    ax.plot_coord(eis_map.center, 'x', color='red')

    print(int_annulus)

    plt.show()

    aiamap = sunpy.map.Map('/home/yjzhu/Solar/EIS_DKIST_SolO/src/AIA/20221025/193/lvl15/aia.lev1_euv_12s.2022-10-25T022003Z.193.image.fits')

    int_lr_straylight, reference_region_eis, reference_region_aia \
          = calc_long_range_stray_light_aia_eis(eis_map, aiamap, eis_map.center, return_region=True)

    fig = plt.figure(figsize=(10,4),layout='constrained')
    ax1 = fig.add_subplot(121,projection=eis_map)
    eis_map.plot(axes=ax1, cmap='plasma')

    reference_region_eis.plot(ax=ax1, edgecolor='red', facecolor='none')
    ax1.plot_coord(eis_map.center, 'x', color='red')

    ax2 = fig.add_subplot(122,projection=aiamap)
    aiamap.plot(axes=ax2, cmap='plasma')

    reference_region_aia.plot(ax=ax2, edgecolor='red', facecolor='none')
    ax2.plot_coord(eis_map.center, 'x', color='red')

    print(int_lr_straylight)
    plt.show()

        
