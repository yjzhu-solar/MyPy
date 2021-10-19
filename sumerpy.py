from matplotlib.pyplot import locator_params
import numpy as np
import scipy.io 
from scipy.interpolate import interp1d
import sys
from warnings import warn

def con_width_funct_4(slit, wavelength, order,fwhm_out, FWHM=True, 
                        DET_B=False):
    
    if (slit>=3) and (slit<=5):
        slit = 4
    elif slit>= 6:
        slit = 7
    
    width = []
    
    if DET_B is True:
        limit_1 = 661.0
        limit_2 = 750.0
    else:
        limit_1 = 780.0
        limit_2 = 805.0

    if isinstance(wavelength,(int,float,np.integer,np.floating)):
        wavelength = np.array([wavelength],dtype=np.float64)
    elif isinstance(wavelength,list):
        wavelength = np.array(wavelength,dtype=np.float64)
    if isinstance(fwhm_out,(int,float,np.integer,np.floating)):
        fwhm_out = np.array([fwhm_out],dtype=np.float64)
    elif isinstance(fwhm_out,list):
        fwhm_out = np.array(fwhm_out,dtype=np.float64)

    wavelength_c = wavelength * order
    fwhm_out_m = fwhm_out * order

    if (np.min(wavelength_c) < limit_1) or (np.max(wavelength_c) > 2*limit_2):
        sys.exit("Wavelength must be between " + \
                str(limit_1/order) + " and " + str(limit_2/order))

    if DET_B is True:
        for ii in range(9):
            w = np.where((wavelength_c > (650+ii*100)) \
                         & (wavelength_c < (750+ii*100)))[0]
            
            if w.size > 0:
                wave = 700 + ii*100
                fwhm_out = fwhm_out_m[w]

                fwhm_fname = "/usr/local/ssw/soho/sumer/idl/contrib/wilhelm/corr/" + \
                            "con_" + str(slit) + "det_b" + str(wave) + ".rst"

                fwhm_save = scipy.io.readsav(fwhm_fname)
                fwhm_out_c = fwhm_save["fwhm_out_c"]
                fwhm_in = fwhm_save["fwhm_in"]
                d_lam_d_in = fwhm_save["d_lam_d_in"]

                if FWHM is True:
                    width_in = d_lam_d_in/0.6006
                else:
                    width_in = d_lam_d_in

                result = np.zeros_like(fwhm_out)
                width_i = np.zeros_like(fwhm_out)

                InRange = np.where((fwhm_out >= np.min(fwhm_out_c)) & \
                                    (fwhm_out <= np.max(fwhm_out_c)))[0]
                
                OutRange = np.where((fwhm_out < np.min(fwhm_out_c)) & \
                                    (fwhm_out > np.max(fwhm_out_c)))[0]

                if InRange.size == 0:
                    warn("FWHM must be between " + \
                        str(np.min(fwhm_out_c)/order) + " and " + \
                        str(np.max(fwhm_out_c)/order) + " mA")
                if OutRange.size > 0:
                    warn("Some of the widths entered are outside the limits " + \
                        str(np.min(fwhm_out_c)/order) + " and " + \
                        str(np.max(fwhm_out_c)/order) + " mA. " + \
                        "These values will be returned as negative values.")

                interp_func = interp1d(fwhm_out_c,width_in)
                result[InRange] = interp_func(fwhm_out[InRange])
                
                if OutRange.size != 0:
                    result[OutRange] = -1 
                
                NonZero = np.where(result != 0)[0]
                if NonZero.size > 0:
                    width_i[NonZero] = result[NonZero]
                else:
                    print("You are outside the available FWHM range.")
                
                width = np.concatenate((width,width_i),axis=None)
    
    else:
        wave = 1300

        fwhm_fname = "/usr/local/ssw/soho/sumer/idl/contrib/wilhelm/corr/" + \
                    "con_" + str(slit) + "det_a" + str(wave) + ".rst"

        fwhm_save = scipy.io.readsav(fwhm_fname)
        fwhm_out_c = fwhm_save["fwhm_out_c"]
        fwhm_in = fwhm_save["fwhm_in"]
        d_lam_d_in = fwhm_save["d_lam_d_in"]

        if FWHM is True:
            width_in = d_lam_d_in/0.6006
        else:
            width_in = d_lam_d_in

        result = np.zeros_like(fwhm_out_m)
        width_i = np.zeros_like(fwhm_out_m)

        InRange = np.where((fwhm_out_m >= np.min(fwhm_out_c)) & \
                            (fwhm_out_m <= np.max(fwhm_out_c)))[0]
        
        OutRange = np.where((fwhm_out_m < np.min(fwhm_out_c)) & \
                            (fwhm_out_m > np.max(fwhm_out_c)))[0]

        if InRange.size == 0:
            warn("FWHM must be between " + \
                str(np.min(fwhm_out_c)/order) + " and " + \
                str(np.max(fwhm_out_c)/order) + " mA")
        if OutRange.size > 0:
            warn("Some of the widths entered are outside the limits " + \
                str(np.min(fwhm_out_c)/order) + " and " + \
                str(np.max(fwhm_out_c)/order) + " mA. " + \
                "These values will be returned as negative values.")

        interp_func = interp1d(fwhm_out_c,width_in)
        result[InRange] = interp_func(fwhm_out_m[InRange])
        
        if OutRange.size != 0:
            result[OutRange] = -1 
        
        NonZero = np.where(result != 0)[0]
        if NonZero.size > 0:
            width_i[NonZero] = result[NonZero]
        else:
            print("You are outside the available FWHM range.")
        
        width = np.concatenate((width,width_i),axis=None)
    
    return width/order

# if __name__ == "__main__":
#     test_wvl_detb = np.linspace(680,1480,9)
#     test_wvl_deta = np.linspace(880,1580,8)
#     test_fwhm = 300.
#     print(con_width_funct_4(1,[680.,780.,1023.],1,[300.,280.,290.],DET_B=True))
#     print(con_width_funct_4(1,[1000.,980.,1023.],1,[300.,280.,290.],DET_B=False))
#     print(con_width_funct_4(2,609,2,300,DET_B=False))
#     for ii in range(8):
#         print(con_width_funct_4(1,test_wvl_detb[ii],1,test_fwhm,DET_B=True))
#         print(con_width_funct_4(2,test_wvl_deta[ii],1,test_fwhm,DET_B=False))