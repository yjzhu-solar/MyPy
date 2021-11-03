import numpy as np
import sys

def eis_ccd_offset(wvl):
    grating_tilt = -0.0792

    if isinstance(wvl,(int,float,np.integer,np.floating)):
        wvl= np.array([wvl],dtype=np.float64)
    elif isinstance(wvl,list):
        wvl = np.array(wvl,dtype=np.float64)

    offset= np.zeros_like(wvl)

    index_sw = np.where(wvl<230)[0]
    index_lw = np.where(wvl>=230)[0]

    offset[index_lw] = grating_tilt*(wvl[index_lw]-256.32)
    offset[index_sw] = grating_tilt*(wvl[index_sw]-185.21) + 18.5 + \
                       grating_tilt*(275.35-256.32)
    
    return offset

def eis_slit_width(yip,slit_ind=0):
    if slit_ind == 0:
        poly_para = np.array([0.059358,-1.5094e-5,-5.6848e-9,
                              8.3517e-11,-4.5428e-14])
    elif slit_ind == 2:
        poly_para = np.array([0.067721,-2.2514e-5,
                              2.5833e-8,4.0998e-11,-2.9560e-14])
    else:
        sys.exit("Slit index must be either 0 (1 arcsec) or 2 (2 arcsec).")
    
    slit_width_poly = np.poly1d(np.flip(poly_para))
    return slit_width_poly(yip)

def eis_slit_width_offset(yip,slit_ind,wvl,wvl0):
    wvl_offset = eis_ccd_offset(wvl)
    wvl_offset_0 = eis_ccd_offset(wvl0)
    wvl_offset = wvl_offset_0 - wvl_offset
    
    return eis_slit_width(yip-wvl_offset,slit_ind=slit_ind)

if __name__ == "__main__":
    # print(eis_ccd_offset(195.1))
    # print(eis_ccd_offset(276))
    # print(eis_ccd_offset([195.1,276]))
    # print(eis_ccd_offset([195.1,276,198,184,257,262]))

    # print(eis_slit_width([22]))
    # print(eis_slit_width([22],slit_ind=2))
    # print(eis_slit_width([234,213,256,500,1000,2]))
    # print(eis_slit_width([234,213,256,500,1000,2],slit_ind=2))

    print(eis_slit_width_offset(400,2,194,180))