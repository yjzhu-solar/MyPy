from matplotlib.pyplot import locator_params
import numpy as np
from numpy.core.fromnumeric import nonzero
import scipy.io 
from scipy.interpolate import interp1d
import sys
from warnings import warn

ssw_dir = "/usr/local/ssw/"
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

                fwhm_fname = ssw_dir + "soho/sumer/idl/contrib/wilhelm/corr/" + \
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

        fwhm_fname = ssw_dir + "soho/sumer/idl/contrib/wilhelm/corr/" + \
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

def read_radiometry_file_old(filename):
    sav_dict = scipy.io.readsav(filename)
    return sav_dict["sensit"], sav_dict["lambda"]

def read_radiometry_file_new(filename):
    sav_dict = scipy.io.readsav(filename)
    return sav_dict["sensit"], sav_dict["lamb"]
def get_calibration(detector,epoch,attitude,evaluation,order):
    calibration = np.zeros((10000,12))
    sen_kbr_1st = np.zeros(10000)
    sen_bare_1st = np.zeros(10000)
    sen_kbr_2nd = np.zeros(10000)
    sen_bare_2nd = np.zeros(10000)
    sen_kbr_3rd = np.zeros(10000)
    sen_bare_3rd = np.zeros(10000)
    lam_kbr_1st = np.zeros(10000)
    lam_bare_1st = np.zeros(10000)
    lam_kbr_2nd = np.zeros(10000)
    lam_bare_2nd = np.zeros(10000)
    lam_kbr_3rd = np.zeros(10000)
    lam_bare_3rd = np.zeros(10000)
    dir = ssw_dir + "soho/sumer/idl/contrib/wilhelm/rad/"

    if epoch == 0:
        if detector == "det_a":
            sensit, wvl = read_radiometry_file_old(dir + "a_kbr_1st_old.rst")
            sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl 
            sensit, wvl = read_radiometry_file_old(dir + "a_kbr_2nd.rst")
            sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl 
            sensit, wvl = read_radiometry_file_old(dir + "a_bare_1st.rst")
            sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl 
            sensit, wvl = read_radiometry_file_old(dir + "a_bare_2nd.rst")
            sen_bare_2nd[:sensit.shape[0]], lam_bare_2nd[:wvl.shape[0]] = sensit, wvl    
        else:
            sys.exit("No calibration could havee been performed for Detector B at this time.")
    elif epoch == 9:
        if order < 3:
            if evaluation == 0:
                if attitude == 0:
                    if detector == "det_a":
                        sensit, wvl = read_radiometry_file_new(dir + "a_kbr_1st_before.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "a_bare_1st_before_2002.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "a_kbr_2nd_before.rst")
                        sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "a_bare_2nd_before_2002.rst")
                        sen_bare_2nd[:sensit.shape[0]], lam_bare_2nd[:wvl.shape[0]] = sensit, wvl
                    elif detector == "det_b":
                        sensit, wvl = read_radiometry_file_new(dir + "b_kbr_1st_before.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "b_bare_1st_before.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "b_kbr_2nd_before.rst")
                        sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "b_bare_2nd_before.rst")
                        sen_bare_2nd[:sensit.shape[0]], lam_bare_2nd[:wvl.shape[0]] = sensit, wvl
                else:
                    if detector == "det_a":
                        sensit, wvl = read_radiometry_file_new(dir + "a_kbr_1st_after.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "a_bare_1st_after_2002.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "a_kbr_2nd_after.rst")
                        sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "a_bare_2nd_after_2002.rst")
                        sen_bare_2nd[:sensit.shape[0]], lam_bare_2nd[:wvl.shape[0]] = sensit, wvl
                    elif detector == "det_b":
                        sensit, wvl = read_radiometry_file_new(dir + "b_kbr_1st_after.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "b_bare_1st_after.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "b_kbr_2nd_after.rst")
                        sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "b_bare_2nd_after.rst")
                        sen_bare_2nd[:sensit.shape[0]], lam_bare_2nd[:wvl.shape[0]] = sensit, wvl
            else:
                if attitude == 0:
                    if detector == "det_a":
                        sensit, wvl = read_radiometry_file_new(dir + "kbr_1st_a_before.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "bare_1st_a_before_2002.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl

                    elif detector == "det_b":
                        sensit, wvl = read_radiometry_file_new(dir + "kbr_1st_b_before.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "bare_1st_b_before_2002.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl
                    sensit, wvl = read_radiometry_file_new(dir + "kbr_2nd_before.rst")
                    sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl
                else:
                    if detector == "det_a":
                        sensit, wvl = read_radiometry_file_new(dir + "kbr_1st_a_after.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "bare_1st_a_after_2002.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl

                    elif detector == "det_b":
                        sensit, wvl = read_radiometry_file_new(dir + "kbr_1st_b_after.rst")
                        sen_kbr_1st[:sensit.shape[0]], lam_kbr_1st[:wvl.shape[0]] = sensit, wvl
                        sensit, wvl = read_radiometry_file_new(dir + "bare_1st_b_after_2002.rst")
                        sen_bare_1st[:sensit.shape[0]], lam_bare_1st[:wvl.shape[0]] = sensit, wvl
                    sensit, wvl = read_radiometry_file_new(dir + "kbr_2nd_after.rst")
                    sen_kbr_2nd[:sensit.shape[0]], lam_kbr_2nd[:wvl.shape[0]] = sensit, wvl
        else:
            sensit, wvl = read_radiometry_file_new(dir + "kbr_3rd_after.rst")
            sen_kbr_3rd[:sensit.shape[0]], lam_kbr_3rd[:wvl.shape[0]] = sensit, wvl
            sensit, wvl = read_radiometry_file_new(dir + "bare_3rd_after.rst")
            sen_bare_3rd[:sensit.shape[0]], lam_bare_3rd[:wvl.shape[0]] = sensit, wvl

    calibration[:,0] = lam_kbr_1st
    calibration[:,1] = sen_kbr_1st
    calibration[:,2] = lam_bare_1st
    calibration[:,3] = sen_bare_1st
    calibration[:,4] = lam_kbr_2nd
    calibration[:,5] = sen_kbr_2nd
    calibration[:,6] = lam_bare_2nd
    calibration[:,7] = sen_bare_2nd
    calibration[:,8] = lam_kbr_3rd
    calibration[:,9] = sen_kbr_3rd
    calibration[:,10] = lam_bare_3rd
    calibration[:,11] = sen_bare_3rd

    del wvl, sensit,lam_kbr_1st,sen_kbr_1st,lam_bare_1st, sen_bare_1st,lam_kbr_2nd,
    sen_kbr_2nd,lam_bare_2nd,sen_bare_2nd,lam_kbr_3rd,sen_kbr_3rd,lam_bare_3rd,sen_bare_3rd

    return calibration

def vignetting(slit,wavelength):
    if slit == 1:
        vig = 6.0
        slit_width = 4.122
    elif slit == 2:
        vig=11.688-2.9773e-3*wavelength+2.80308e-6*wavelength**2       
        slit_width=0.986
    elif  3 <= slit <= 5:
        vig=11.688-2.9773e-3*wavelength+2.80308e-6*wavelength**2       
        slit_width=0.993
    elif 6<= slit <=8:
        vig=12.921+3.3522e-3*wavelength+4.46974e-6*wavelength**2
        slit_width=0.278
    else:
        sys.exit("Wrong slit selection.")
    diffr_cor = 100.0/(100.0-vig)
    return diffr_cor, slit_width

def photocathode(surface,order,calibration):
    if order == 1:
        if surface == "kbr":
            wvl = calibration[:,0]
            sensit = calibration[:,1]
        elif surface == "bare":
            wvl = calibration[:,2]
            sensit = calibration[:,3]
    elif order == 2:
        if surface == "kbr":
            wvl = calibration[:,4]
            sensit = calibration[:,5]
        elif surface == "bare":
            wvl = calibration[:,6]
            sensit = calibration[:,7]
    elif order == 3:
        if surface == "kbr":
            wvl = calibration[:,8]
            sensit = calibration[:,9]
        elif surface == "bare":
            wvl = calibration[:,10]
            sensit = calibration[:,11]

    valid = (np.where(sensit > 0))[0]
    if len(valid) > 0:
        sensit = sensit[valid]
        wvl = wvl[valid]*10.
    else:
        sensit = 0
        wvl = 0

    return wvl, sensit

def param_corr(lambda_n,a,b):
    return a + b*(1600. - lambda_n)/800.

def get_lambda_n(lambda_b_511):
    px_s_b = 0.0265
    m = 1
    d = 2777.45e0
    ra = 3200.78

    lambda_n_return = lambda_b_511
    sin_theta = m*lambda_n_return/d
    cos_theta = np.sqrt(1 - sin_theta**2)
    f_n = ra/(1 + cos_theta)
    d_lam = d*px_s_b/m/f_n
    for ii in range(4):
        diff = d_lam*(2653-2)
        sin_theta = m*(lambda_n_return + diff)/d
        cos_theta = np.sqrt(1 - sin_theta**2)
        f_n = ra/(1 + cos_theta)
        delta_x = param_corr(lambda_n_return+diff,-1.68,1.0)

        f_n = f_n - delta_x
        d_lam = d*px_s_b/m/f_n
    lambda_n_return = lambda_n_return + diff

    return lambda_n_return

def grating_b_511(lambda_n):
    ra = 3200.78
    px = 0.0265
    d = 2777.45e0
    dist = 70.3045
    fc = 399.60

    sin_theta = lambda_n/d
    cos_theta = np.sqrt(1 - sin_theta**2)
    f_n = ra/(1 + cos_theta)
    delta_x = param_corr(lambda_n,-1.68,1.0)
    f_n = f_n - delta_x
    beta = np.arctan(px*2653/f_n)
    fs = ra/(np.cos(beta) + cos_theta)

    l_2 = 1327*px
    alpha_0 = np.arctan(dist/2./f_n)
    zeta = 86.8*3.1416/180.+ alpha_0
    f = np.sqrt(f_n**2 + (dist/2.)**2)
    f_lam = np.sqrt(l_2**2 + f**2 - 2*l_2*f*np.cos(zeta))
    mag_b = f_lam/fc

    px_proj = px*np.cos(beta - 3.2*3.1416/180.)
    d_lambda_b_511 = d*np.cos(beta)*px_proj/f_lam

    return d_lambda_b_511, mag_b

def magnification(wavelength,order):
    d = 2777.45e0
    ra = 3200.78
    fc = 399.60

    px_s_a = 0.0266
    sin_theta = order*wavelength/d
    cos_theta = np.sqrt(1 - sin_theta**2)
    f_lambda = ra/(1+cos_theta)
    mag_a = f_lambda/fc
    d_lam_a = d*px_s_a/order/f_lambda
    lambda_n = get_lambda_n(order*wavelength)
    d_lam_b, mag_b = grating_b_511(lambda_n)
    d_lam_b = d_lam_b/order

    return mag_a, d_lam_a, mag_b, d_lam_b


def radiometry(slit0,wavelength,order,count_rate,bare=False,kbr=True,px=True,line=False,sun_line=False,
                arcsec=False,photons=False,watts=True,det_a=True,det_b=False,epoch=9,before=False,after=True,
                irrad_au=False,test=False,separate=True,joint=False,attenuator=False,quiet=False):

    h = 6.626e-34
    c = 2.998e8
    phot_energy = h*c/wavelength*1.e10
    solid = 2.350e-11
    mirror = 0.0117
    mag = 26.5/6.316
    if isinstance(wavelength,(int,float,np.integer,np.floating)):
        wavelength = np.array([wavelength],dtype=np.float64)
    elif isinstance(wavelength,list):
        wavelength = np.array(wavelength,dtype=np.float64)
    if isinstance(count_rate,(int,float,np.integer,np.floating)):
        count_rate = np.array([count_rate],dtype=np.float64)
    elif isinstance(count_rate,list):
        count_rate = np.array(count_rate,dtype=np.float64)
    n_c_r = len(count_rate)
    n_w = len(wavelength)

    if (n_c_r != n_w) and (n_c_r != 1) and (n_w != 1):
        sys.exit("Wavelength and count-rate arrays not of same dimension")
    res = count_rate * 0.0

    if sum([photons,watts]) in (0,2):
        sys.exit("Set either photons=True or watts=True.")
    if sum([px,line,sun_line,arcsec]) in (0,2,3,4):
        sys.exit("Set only one of px, line, sun_line, and arcsec equals True.")
    if (sun_line is True) or (arcsec is True):
        slit = 2
        warn("Slit selection has no effect.")
    else:
        slit = slit0
    
    if (sum([kbr,bare]) == 2) or (sum([kbr,attenuator]) == 2):
        sys.exit("kbr =True cannot be used with bare=True or attenuator=True.")
    elif (bare is True) or (attenuator is True):
        surface = "bare"
    else:
        surface = "kbr"
    
    if sum([det_a,det_b]) == 2:
        sys.exit("det_a=True cannot be used with det_b=True.")
    elif det_a is True:
        detector = "det_a"
    else:
        detector = "det_b"

    if (test is True) and (detector == "det_b"):
        test_phase = 0.82
    elif (test is True) and (detector == "det_a"):
        test_phase = 1.0
        warn("Test phase only with detector B. No test phase set.")
    else:
        test_phase = 1.0
        warn("No test phase set by default.")
    
    if (order == 3) and (detector == "det_b"):
        sys.exit("3rd order calibration only available for detector A.")
    
    if (sum([before,after]) in (0,2)):
        sys.exit("Set either before=True or after=True.")
    elif (order == 3) and (before is True):
        warn("after=True used by default in 3rd order.")
        attitude = 1
    elif before is True:
        attitude = 0
    else:
        attitude = 1
    
    if (sum([joint,bare]) == 2) and (order == 2):
        sys.exit("joint=True cannot be set with bare=True in second order.")
    elif (joint is True) and (epoch < 8):
        sys.exit("joint=True cannot be set with epooch less than 8.")
    elif (sum([joint,separate]) in (0,2)) and (order != 3):
        sys.exit("Set either separate=True or joint=True.")
    elif joint is True:
        evaluation = 1
    else:
        evaluation = 0
    
    calibration = get_calibration(detector,epoch,attitude,evaluation,order)
    wvl, sensit = photocathode(surface,order,calibration)
    mag_a, d_lam_a, mag_b, d_lam_b = magnification(wavelength,order)

    if detector == "det_a":
        mag_a_b = mag_a
        d_lam = d_lam_a
    else:
        mag_a_b = mag_b
        d_lam = d_lam_b
    
    if np.min(wvl) != 0:
        minwave = np.min(wvl[np.where(wvl > 0)])
        if (detector == "det_a") and (order == 1):
            minwave = 770
        maxwave = np.max(wvl)
        inrange = (np.where((wavelength>=minwave) | (wavelength<=maxwave)))[0]
        outrange = (np.where((wavelength>=maxwave) | (wavelength<= minwave)))[0]

        if maxwave > 1000:
            order_message = "(in first order)"
        elif maxwave > 700:
            order_message = "(in second order)"
        elif maxwave < 700:
            order_message = "(in third order)"
        
        if len(inrange) == 0:
            sys.exit("Wavelength must be between {:.2f} and {:.2f} angstrom".format(minwave,maxwave) + order_message)
        elif len(outrange) > 0:
            warn("Some of the wavelengths entered are outside the limits {:.2f} and {:.2f} angstrom".format(minwave,maxwave) + \
                "These values will be returned as -1.")
    else:
        sys.exit("There was no calibration for this detector at this epoch.")
    
    res[inrange] = np.interp(xp=wvl,fp=sensit,x=wavelength[inrange])

    if line is True:
        factor = 1.0e0
    elif sun_line is True:
        factor = mag_a_b/mag/solid
    elif arcsec:
        factor = mag_a_b/mag
    else:
        factor = d_lam
    
    if photons is True:
        res = res*factor
    else:
        res = res*factor/phot_energy
    
    diffr_cor, slit_width = vignetting(slit,wavelength)
    nonzero = (np.where(res != 0))[0]
    response = 1
    if (after is True) and (epoch < 6):
        sys.exit("after=True cannot be used with an epoch less than 6.")
    elif (after is True) and (epoch == 7):
        response = 0.69
    elif (after is True) and (epoch == 6):
        response = 0.57
    
    if attenuator is True:
        response = response * 0.1
    result = np.zeros_like(res)

    if len(nonzero) > 0:
        result[nonzero] = (count_rate*diffr_cor*mag_a_b)[nonzero] \
            /mag/slit_width/solid/mirror/res[nonzero]/response/test_phase
    else:
        warn("Wavelengths are outside the calibrated range.")
    if len(outrange) != 0:
        result[outrange] = -1
    interval = (np.where((wavelength >= 750)&(wavelength<=880)))[0]
    if (joint is True) and (len(interval)>0):
        warn("It is recommend to use separate=True for wavelength between 750 and 880 angstrom.")
    
    if (irrad_au is True) and (sun_line is True):
        print("Solar radius from Earth at date in")
        arcmin = np.float64(input("(1) mm/arcmin "))
        arcsec = np.float64(input("(2) ss.ss/arcsec"))
        radius = arcmin*60 + arcsec
        conv = 0.98*(959.6/radius)**2
        result = result*conv
    elif irrad_au is True:
        sys.exit("irrad_au=True is only effective together with sun_line=True")

    return result

def detector_bottom_top(wvl, det="A"):
    bbot=np.flip(np.array([-503.342,4.651,-0.0168628,3.03637e-05,-2.64783e-08,8.77005e-12]))
    btop=np.flip(np.array([-1420.08,15.2641,-0.0535999,9.29212e-05,-7.86775e-08,2.58698e-11]))
    c=np.flip(np.array([0.023036093, -9.0158920e-07, -1.6460661e-09, -1.4244883e-12]))

    poly_c = np.poly1d(c)
    if det == "A":
        pass
    elif det == "B":
        wvl=wvl+poly_c(wvl)*2653
    else:
        sys.exit("Invalid detector name")

    poly_bbot = np.poly1d(bbot)
    poly_btop = np.poly1d(btop)

    detector_bot = poly_bbot(wvl)
    detecotr_top = poly_btop(wvl)

    return detector_bot, detecotr_top

def delta_pixel(wvl1,wvl2,pixel,det="A",det_bot1=None,det_top1=None,det_bot2=None,det_top2=None):
    if (det_bot1 is None) and (det_top1 is None):
        det_bot1, det_top1 = detector_bottom_top(wvl1,det)
    if (det_bot2 is None) and (det_top2 is None):
        det_bot2, det_top2 = detector_bottom_top(wvl2,det)

    new_pixel = (pixel-det_bot1)*(det_top2-det_bot2)/(det_top1-det_bot1) + det_bot2

    return new_pixel



if __name__ == "__main__":
    # pass
    # print(get_lambda_n(680))
    # print(grating_b_511(900))
    # print(magnification(700,2))
    # print(vignetting(1,780))
    print(radiometry(1,780,1,1,kbr=False,bare=True))
#     test_wvl_detb = np.linspace(680,1480,9)
#     test_wvl_deta = np.linspace(880,1580,8)
#     test_fwhm = 300.
#     print(con_width_funct_4(1,[680.,780.,1023.],1,[300.,280.,290.],DET_B=True))
#     print(con_width_funct_4(1,[1000.,980.,1023.],1,[300.,280.,290.],DET_B=False))
#     print(con_width_funct_4(2,609,2,300,DET_B=False))
#     for ii in range(8):
#         print(con_width_funct_4(1,test_wvl_detb[ii],1,test_fwhm,DET_B=True))
#         print(con_width_funct_4(2,test_wvl_deta[ii],1,test_fwhm,DET_B=False))