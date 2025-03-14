# This is a naive reader for the L2 IRIS/SJI data. It keeps the original FITS-WCS with SunPy Maps,
# instead of gWCS used by IRISpy-LMSAL.  

import numpy as np
import sunpy 
import sunpy.map
import astropy
from astropy.wcs import WCS
import astropy.units as u
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
from datetime import datetime, timedelta, date

def read_iris_sji(filename, index=None, sdo_rsun=False, mask=True, nbin=1, **kwargs):
    with fits.open(filename) as hdul:
        data = hdul[0].data.copy()
        prim_header = hdul[0].header.copy()
        aux_data = hdul[1].data.copy()
        aux_header = hdul[1].header.copy()

        n_image = prim_header["NAXIS3"]

        if index is None:
            maps = []
            for ii in range(n_image): #
                maps.append(get_generic_map(ii, data, prim_header, aux_data, aux_header, nbin=nbin,
                                            sdo_rsun=sdo_rsun, mask=mask))
            
            return sunpy.map.Map(maps, **kwargs)
        
        elif isinstance(index, (int, np.integer)):
            return get_generic_map(index, data, prim_header, aux_data, aux_header, nbin=nbin,
                                   sdo_rsun=sdo_rsun, mask=mask, **kwargs)
        
        elif isinstance(index, (list, np.ndarray)):
            maps = []
            for ii in index:
                maps.append(get_generic_map(ii, data, prim_header, aux_data, aux_header,
                                            nbin=nbin, sdo_rsun=sdo_rsun, mask=mask))
            return sunpy.map.Map(maps, **kwargs)
        
        elif isinstance(index, slice):
            maps = []
            for ii in range(n_image)[index]:
                maps.append(get_generic_map(ii, data, prim_header, aux_data, aux_header,
                                            nbin=nbin, sdo_rsun=sdo_rsun, mask=mask))
            return sunpy.map.Map(maps, **kwargs)
        
        elif isinstance(index, (date,Time)):
            if isinstance(index, Time):
                index = index.to_datetime()
            
            date_obs_all = datetime.strptime(prim_header["DATE_OBS"],'%Y-%m-%dT%H:%M:%S.%f') + \
                            aux_data[:, aux_header["TIME"]]*timedelta(seconds=1)
            
            index = np.abs(date_obs_all - index).argmin()
            return get_generic_map(index, data, prim_header, aux_data, aux_header, 
                                   nbin=nbin, sdo_rsun=sdo_rsun, mask=mask, **kwargs)
        
        else:
            raise ValueError("index must be an integer, list, slice, date or Time object")

def get_generic_map(ii, data, prim_header, aux_data, aux_header, nbin=1, sdo_rsun=False, mask=True, **kwargs):
    date_obs_start = Time(prim_header["DATE_OBS"])
    date_exposure_start = date_obs_start + aux_data[ii, aux_header["TIME"]]*u.s
    exptime = aux_data[ii, aux_header["EXPTIMES"]]*u.s

    if sdo_rsun:
        rsun = 696000000*u.m
    else:
        rsun = None
    
    if nbin > 1:
        ii_start = ii - nbin//2
        ii_end = ii + nbin//2 + 1

        if ii_start < 0:
            ii_start = 0
        if ii_end > data.shape[0]:
            ii_end = data.shape[0]
        map_data = np.nanmean(data[ii_start:ii_end,:,:], axis=0)
    elif nbin == 1:
        map_data = data[ii,:,:]
    else:
        raise ValueError("nbin must be an integer greater than or equal to 1")
    map_fitswcs_header = sunpy.map.make_fitswcs_header(map_data,
                                                       SkyCoord(aux_data[ii, aux_header["XCENIX"]]*u.arcsec,
                                                                aux_data[ii, aux_header["YCENIX"]]*u.arcsec,
                                                                frame="helioprojective",observer="earth",
                                                                obstime=date_exposure_start,
                                                                rsun=rsun),
                                                        reference_pixel=u.Quantity([prim_header["CRPIX1"]-1,prim_header["CRPIX2"]-1],u.pix),
                                                        scale=u.Quantity([prim_header["CDELT1"],prim_header["CDELT2"]],u.arcsec/u.pix),
                                                        rotation_matrix=np.array([[aux_data[ii, aux_header["PC1_1IX"]],aux_data[ii, aux_header["PC1_2IX"]]],
                                                                                  [aux_data[ii, aux_header["PC2_1IX"]],aux_data[ii, aux_header["PC2_2IX"]]]]),
                                                        exposure=exptime,
                                                        telescope=prim_header["TELESCOP"],
                                                        instrument=prim_header["INSTRUME"],
                                                        wavelength=prim_header["TWAVE1"]*u.angstrom,)
    if mask:
        data_mask = np.isclose(data[ii,:,:],-200)
        return sunpy.map.Map(data[ii,:,:], map_fitswcs_header, mask=data_mask, **kwargs)
    else:
        return sunpy.map.Map(data[ii,:,:], map_fitswcs_header, **kwargs)



            

