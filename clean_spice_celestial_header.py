# The WCS header of SPICE include a temporal axis, which is good for 
# some perposes, for example, tracking the time of each exposure. 
# However, it creates a non-diagonal term in the PCi_j matrix, which 
# makes it difficult to separate the celestial part of the legacy WCS.
# This scripts removes the temporal axis from the WCS header

# example usage 
# from sunraster.instr.spice import read_spice_l2_fits
# spice_raster = read_spice_l2_fits('spice_l2.fits')
# spice_window_header = spice_raster[0].meta.original_header
# spice_window_header_cleaned = remove_temporal_axis(spice_window_header)

from copy import deepcopy

def remove_temporal_axis(hdr):
    hdr = deepcopy(hdr)
    hdr['NAXIS'] = 3

    keys_to_remove = ['NAXIS4', 'PXBEG4', 'PXEND4', 'CRPIX4', 'CRVAL4',
                      'CDELT4', 'CTYPE4', 'CUNIT4', 'NBIN4', 'CNAME4',
                      'PC4_1', 'PC4_2', 'PC4_3', 'PC4_4',]
    for key in keys_to_remove:
        hdr.pop(key, 0)
    
    return hdr
