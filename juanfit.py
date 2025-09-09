#!/usr/bin/env python3

'''
    File Name: juanfit.py
    Author: Yingjie Zhu
    Institute: University of Michigan
    Email: yjzhu(at)umich(dot)edu
    Date create: 07/20/2021
    Date Last Modified: 2/1/2023
    Python Version: 3.8
    Statues: In development
    Version: 0.0.3
    Description: This is a naive and dirty python script 
    to perform single/multiple Gaussian fitting to the 
    spectral lines.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, 
                                AutoLocator, MaxNLocator)
from IPython.display import display, Math
import emcee
import scipy
from scipy.special import wofz, voigt_profile
from scipy.optimize import curve_fit
from num2tex import num2tex
from warnings import warn
import sys
import multiprocessing as mp
import copy
import astropy.constants as const
from astropy.visualization import ZScaleInterval, ImageNormalize, LogStretch, AsymmetricPercentileInterval,\
         ManualInterval, SqrtStretch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


#plot setting 
from matplotlib import rcParams, rc_context
rc_context_dict = {'text.usetex': True,
                   'font.family': 'serif',
                   'axes.linewidth': 2,
                   'xtick.major.width': 1,
                   'xtick.major.size': 4,
                   'xtick.minor.width': 1,
                   'xtick.minor.size':2,
                   'ytick.major.width': 1,
                   'ytick.major.size': 4,
                   'ytick.minor.width': 1,
                   'ytick.minor.size': 2,
                   'text.latex.preamble': r'\usepackage[T1]{fontenc} \usepackage{amsmath} ' + \
                     r'\usepackage{siunitx} \sisetup{detect-all=true}'}

# rcParams['text.usetex'] = True
# rcParams['font.family'] = 'serif'
# rcParams['axes.linewidth'] = 2
# rcParams['xtick.major.width'] = 1
# rcParams['xtick.major.size'] = 4
# rcParams['xtick.minor.width'] = 1
# rcParams['xtick.minor.size'] =2 
# rcParams['ytick.major.width'] = 1
# rcParams['ytick.major.size'] = 4
# rcParams['ytick.minor.width'] = 1
# rcParams['ytick.minor.size'] = 2 
# rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc} \usepackage{amsmath} ' + \
#     r'\usepackage{siunitx} \sisetup{detect-all=true}'

class SpectrumFitSingle:
    '''
        SpectrumFitSingle performs single/multiple Gaussian fitting 
        to the input spectra.
    '''
    def __init__(self, data, wvl, line_number=None, line_wvl_init=None, int_max_init=None, \
                fwhm_init=None, int_cont_init=None, err=None, err_percent=None, same_width=False, \
                stray_light=False, stray_wvl_init=None, stray_int_total=None, stray_fwhm=None, \
                mask=None,custom_func=None,custom_init=None) -> None:

        '''
            Initialize the SpectrumFitSingle class.

            Parameters
            ----------
            data : 1-D array
                Input 1-D spectra (intensity) to fit.
            wvl : 1-D array
                1-D wavelength grid of the spectra.
            line_number: integer
                The desired number of lines to fit.
            line_wvl_init : scalar or 1-D array
                Initial wavelength(s) of the spectral line core(s).
            int_max_init : scalar or 1-D array
                Initial value(s) of the peak intensity.
            fwhm_init : scalar or 1-D array
                Initial value(s) of the full width at half maximum (FWHM).
            err : 1-D array , optional 
                Errors in the intensity at different wavelengths. If provided,
                will be used to calculate the likelihood function. Default is None.
            err_percent : scalar or 1-D array (0-100), optional 
                If provided, multiply this percentage to the data to create the errors used 
                in fittings. Conflict with the err parameter. Default is None. 
            same_width : bool or a list of bool, optional 
                If True, forces the fitted spectral lines have the same width. If provided 
                as a list of bools, only forces the spectral lines corresponding to True value
                have the same width in the fitting. Default is False.
            stray_light : bool, optional 
                If True, adds the stray light profile to the fitting. Default is False.
            stray_wvl_init : scalar or 1-D array, optional
                Initial wavelength(s) of the stray light line core(s). Default is None.
            stray_int_total: scalar or 1-D array, optional 
                Integrated intensity of the stray light profile(s). Default is None.
            stray_fwhm: scalar or 1-D array, optional 
                Full width at half maximum (FWHM) of the stray light profile(s).
                Default is None. 
            mask: [N,2] array, optional 
                If provided, will mask the data between the N intervals in the fitting. 
                For example, [[2,4],[10,12]] will mask the data points within [2,4] and 
                [10,12]. Default is None.  
            custom_func : function, optional 
                If provided, fit the spectra using this custom function. Default is None.
            custom_init : scalar, 1-D array like, optional 
                Initial values for the custom fitting function. 


        '''
        
        #input parameters
        self.data = data
        self.wvl = wvl
        if err_percent is None:
            self.err = err
        else:
            if err is None:
                self.err = self.data*err_percent/100
            else:
                warn("Both the err and err_percent parameters are used. Use err instead of err_percent.")
                self.err = err
        self.mask = mask
        self.stray_light = stray_light
        self.stray_wvl_init = stray_wvl_init
        self.stray_int_total = stray_int_total
        self.stray_fwhm = stray_fwhm
        #instance properties
        self.shape = data.shape
        self.custom_func = custom_func
        self.custom_init = custom_init

        #Create masked wavelength, intensity, and error by deleting the masked 
        #values, since scipy.curvefit does not support masked arrays... 
        if mask is None:
            self.wvl_tofit = self.wvl
            self.data_tofit = self.data
            self.err_tofit = self.err
        else:
            delete_index_all = []
            for ii in range(len(self.mask)):
                delete_index = np.where((self.wvl>=self.mask[ii][0]) & (self.wvl<=self.mask[ii][1]))[0]
                delete_index_all = delete_index_all + delete_index.tolist()
            self.wvl_tofit = np.delete(self.wvl,delete_index_all)
            self.data_tofit = np.delete(self.data,delete_index_all)
            if self.err is not None:
                self.err_tofit = np.delete(self.err,delete_index_all)
            else:
                self.err_tofit = self.err

        #If the custom fitting function is not provided, read the initial 
        #values for the embedded multi-gaussian functions. And create the 
        #Chi2 fitted parameters.
        if self.custom_func is None:
            self.line_number = line_number
            self.line_wvl_init = np.array(line_wvl_init)
            self.int_max_init = np.array(int_max_init)
            self.fwhm_init = np.array(fwhm_init)
            self.same_width = same_width

            if int_cont_init is None:
                self.int_cont_init = np.min((np.mean(self.data[:2]),np.mean(self.data[-2:])))
            else:
                self.int_cont_init = int_cont_init

            #fitted parameters
            self.line_wvl_fit = np.zeros(self.line_number)
            self.line_wvl_err = np.zeros(self.line_number)

            self.int_total_fit = np.zeros(self.line_number)
            self.int_total_err = np.zeros(self.line_number)

            if type(same_width) is list:
                self.fwhm_fit = np.zeros(self.line_number)
                self.fwhm_err = np.zeros(self.line_number)
                self.line_same_width_index = np.where(self.same_width)[0]
            elif same_width is True:
                self.fwhm_fit = np.float64(0.0)
                self.fwhm_err = np.float64(0.0)
            else:
                self.fwhm_fit = np.zeros(self.line_number)
                self.fwhm_err = np.zeros(self.line_number)

            self.int_cont_fit = np.float64(0.0)
            self.int_cont_err = np.float64(0.0)

            if self.line_number == 1:
                self.int_max_init = np.max(self.data_tofit)
            else:
                self.int_max_init = self.int_max_init*np.max(self.data_tofit)/np.max(self.int_max_init)

        else:
            self.custom_fit = np.zeros_like(custom_init)
            self.custom_err = np.zeros_like(custom_init)
        


    def run_lse(self,ignore_err=False,absolute_sigma=True,maxfev=2800):
        '''
            Performs least square estimation (Chi square fitting)
            to the spectral line(s).

            Parameters
            ----------
            ignore_err : bool, optional
            If True, ignores the input error. Default is False. 
            absolute_sigma: bool, optional
            If True, the errors have the same unit as data. Default is True.  
        '''

        if ignore_err is True:
            err_lse = None
        else:
            err_lse = self.err_tofit 

        if (err_lse is None) and (absolute_sigma is True):
            absolute_sigma = False
            warn("No input errors, absolute_sigma=False will be used in the Chi2 fitting.")

        if self.stray_light is False:
            if self.custom_func is None:
                if type(self.same_width) is list:
                    new_fwhm_init = np.concatenate((np.delete(self.fwhm_init,self.line_same_width_index),
                                                    np.average(self.fwhm_init[self.line_same_width_index])),
                                                    axis=None)
                    popt = np.concatenate((self.line_wvl_init,
                                        self.int_max_init*np.sqrt(2.*np.pi)*self.fwhm_init/2.355,
                                        new_fwhm_init,self.int_cont_init),axis = None)
                else:
                    popt = np.concatenate((self.line_wvl_init,
                                        self.int_max_init*np.sqrt(2.*np.pi)*self.fwhm_init/2.355,
                                        self.fwhm_init,self.int_cont_init),axis = None)
                self.dof = len(self.wvl_tofit) - len(popt)
                if type(self.same_width) is list:
                    popt, pcov = curve_fit(self.multi_gaussian_mixture_width, self.wvl_tofit, self.data_tofit,
                                        p0=popt,sigma=err_lse,absolute_sigma=absolute_sigma,maxfev=maxfev) 

                    self.line_wvl_fit = popt[:self.line_number]
                    self.int_total_fit = popt[self.line_number:self.line_number*2]
                    self.int_cont_fit = popt[-1]

                    perr = np.sqrt(np.diagonal(pcov))
                    self.line_wvl_err = perr[:self.line_number]
                    self.int_total_err = perr[self.line_number:self.line_number*2]
                    self.int_cont_err = perr[-1]

                    self.fwhm_fit = np.zeros(self.line_number)
                    self.fwhm_err = np.zeros(self.line_number)
                    fwhm_arg_index_cont = 0 
                    for ii in range(self.line_number):
                        if self.same_width[ii] is True:
                            self.fwhm_fit[ii] = popt[-2]
                            self.fwhm_err[ii] = perr[-2]
                        else:
                            self.fwhm_fit[ii] = popt[self.line_number*2+fwhm_arg_index_cont]
                            self.fwhm_err[ii] = perr[self.line_number*2+fwhm_arg_index_cont]
                            fwhm_arg_index_cont += 1

                elif self.same_width is True:
                    popt, pcov = curve_fit(self.multi_gaussian_same_width, self.wvl_tofit, self.data_tofit,
                                        p0=popt,sigma=err_lse,absolute_sigma=absolute_sigma,maxfev=maxfev)
                    
                    self.line_wvl_fit = popt[:self.line_number]
                    self.int_total_fit = popt[self.line_number:self.line_number*2]
                    self.fwhm_fit = popt[-2]
                    self.int_cont_fit = popt[-1]

                    perr = np.sqrt(np.diagonal(pcov))
                    self.line_wvl_err = perr[:self.line_number]
                    self.int_total_err = perr[self.line_number:self.line_number*2]
                    self.fwhm_err = perr[-2]
                    self.int_cont_err = perr[-1]
                else:
                    popt, pcov = curve_fit(self.multi_gaussian_diff_width, self.wvl_tofit, self.data_tofit,
                                        p0=popt,sigma=err_lse,absolute_sigma=absolute_sigma,maxfev=maxfev)
                    
                    self.line_wvl_fit = popt[:self.line_number]
                    self.int_total_fit = popt[self.line_number:self.line_number*2]
                    self.fwhm_fit = popt[self.line_number*2:self.line_number*3]
                    self.int_cont_fit = popt[-1]

                    perr = np.sqrt(np.diagonal(pcov))
                    self.line_wvl_err = perr[:self.line_number]
                    self.int_total_err = perr[self.line_number:self.line_number*2]
                    self.fwhm_err = perr[self.line_number*2:self.line_number*3]
                    self.int_cont_err = perr[-1]     
            else:
                self.dof = len(self.wvl_tofit) - len(self.custom_init)
                popt, pcov = curve_fit(self.custom_func, self.wvl_tofit, self.data_tofit,
                    p0=self.custom_init,sigma=err_lse,absolute_sigma=absolute_sigma,maxfev=maxfev) 
                
                self.custom_fit = popt
                self.custom_err = np.sqrt(np.diagonal(pcov))
        else:
            print("Fitting with stray light is not supported in this version.")

    def run_HahnMC(self,ignore_err=False,n_chain=10000,cred_lvl=None,absolute_sigma=True,save_chain=False):
        '''
            Fit line profiles using the Monte-Carlo method described 
            in Hahn et al. 2012, ApJ, 753, 36.

            Parameters
            ----------
            ignore_err : bool, optional 
            If True, ignore the input error in fitting. Default is False.
            n_chain : int, optional 
            The number of Monte Carlo iteration. Default is 10,000.
            cred_lvl: None or integer or float between 0 - 100, optional 
            If None, calculate from standard deviation. If not None, retrieve uncertainty from 
            the credible levels. Default is None.   
        '''

        self.run_lse(ignore_err=ignore_err,absolute_sigma=absolute_sigma)
        if type(self.same_width) is list:
            new_fwhm_fit = np.concatenate((np.delete(self.fwhm_fit,self.line_same_width_index),
                                        np.average(self.fwhm_fit[self.line_same_width_index])),
                                        axis=None)
            p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,new_fwhm_fit,
                                    self.int_cont_fit),axis=None)
            spec_fit = self.multi_gaussian_mixture_width(self.wvl_tofit,*p_fit)
        elif self.same_width is True:
            p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,self.fwhm_fit,
                                    self.int_cont_fit),axis=None)
            spec_fit = self.multi_gaussian_same_width(self.wvl_tofit,*p_fit)
        else:
            p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,self.fwhm_fit,
                                    self.int_cont_fit),axis=None)
            spec_fit = self.multi_gaussian_diff_width(self.wvl_tofit,*p_fit)  

        if self.err_tofit is None:                        
            err_diff = np.abs(self.data_tofit - spec_fit)
        else:
            err_diff = np.maximum(self.err_tofit,np.abs(self.data_tofit - spec_fit))
        
        random_err = np.zeros((n_chain,len(self.wvl_tofit)))
        for ii in range(len(self.wvl_tofit)):
            random_err[:,ii] = np.random.normal(0, err_diff[ii], n_chain)

        if self.stray_light is False:
            popt = np.concatenate((self.line_wvl_fit,self.int_total_fit,
                        self.fwhm_fit,self.int_cont_fit),axis = None)
        else:
            print("Fitting with stray light is not supported in this version.")

        popt_chain = np.zeros((n_chain,popt.shape[0]))
        if ignore_err is True: # no error 
            err_hmc = None
        else:
            err_hmc = err_diff
        if type(self.same_width) is list:
            for ii in range(n_chain):
                popt_chain[ii,:], _ = curve_fit(self.multi_gaussian_mixture_width, self.wvl_tofit,
                                            self.data_tofit+random_err[ii,:],p0=popt,sigma=err_hmc,
                                            absolute_sigma=absolute_sigma)
            
            popt_result = np.zeros_like(popt)
            popt_err = np.zeros((2,popt.shape[0]))
            for jj in range(popt.shape[0]):
                if cred_lvl is not None:
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
                else:
                    popt_result[jj] = np.percentile(popt_chain[:, jj],50)
                    popt_err[:,jj] = np.array([np.std(popt_chain[:, jj]),np.std(popt_chain[:, jj])])
            
            self.line_wvl_fit_hmc = popt_result[:self.line_number]
            self.int_total_fit_hmc = popt_result[self.line_number:self.line_number*2]
            self.int_cont_fit_hmc = popt_result[-1]

            self.line_wvl_err_hmc = popt_err[:,:self.line_number]
            self.int_total_err_hmc = popt_err[:,self.line_number:self.line_number*2]
            self.int_cont_err_hmc = popt_err[:,-1]


            self.fwhm_fit_hmc = np.zeros(self.line_number)
            self.fwhm_err_hmc = np.zeros((2,self.line_number))
            fwhm_arg_index_cont = 0 
            for ii in range(self.line_number):
                if self.same_width[ii] is True:
                    self.fwhm_fit_hmc[ii] = popt_result[-2]
                    self.fwhm_err_hmc[:,ii] = popt_err[:,-2]
                else:
                    self.fwhm_fit_hmc[ii] = popt_result[self.line_number*2+fwhm_arg_index_cont]
                    self.fwhm_err_hmc[:,ii] = popt_err[:,self.line_number*2+fwhm_arg_index_cont]
                    fwhm_arg_index_cont += 1

        elif self.same_width is True:
            for ii in range(n_chain):
                popt_chain[ii,:], _ = curve_fit(self.multi_gaussian_same_width, self.wvl_tofit,
                                            self.data_tofit+random_err[ii,:],p0=popt,sigma=err_hmc,
                                            absolute_sigma=absolute_sigma)
            
            popt_result = np.zeros_like(popt)
            popt_err = np.zeros((2,popt.shape[0]))
            for jj in range(popt.shape[0]):
                if cred_lvl is not None:
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
                else:
                    popt_result[jj] = np.percentile(popt_chain[:, jj],50)
                    popt_err[:,jj] = np.array([np.std(popt_chain[:, jj]),np.std(popt_chain[:, jj])])

            
            self.line_wvl_fit_hmc = popt_result[:self.line_number]
            self.int_total_fit_hmc = popt_result[self.line_number:self.line_number*2]
            self.fwhm_fit_hmc = popt_result[-2]
            self.int_cont_fit_hmc = popt_result[-1]

            self.line_wvl_err_hmc = popt_err[:,:self.line_number]
            self.int_total_err_hmc = popt_err[:,self.line_number:self.line_number*2]
            self.fwhm_err_hmc = popt_err[:,-2]
            self.int_cont_err_hmc = popt_err[:,-1]
        else:
            for ii in range(n_chain):
                popt_chain[ii,:], _ = curve_fit(self.multi_gaussian_diff_width, self.wvl_tofit,
                                            self.data_tofit+random_err[ii,:],p0=popt,sigma=err_hmc,
                                            absolute_sigma=absolute_sigma)
            
            popt_result = np.zeros_like(popt)
            popt_err = np.zeros((2,popt.shape[0]))
            for jj in range(popt.shape[0]):
                if cred_lvl is not None:
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    #print(type(np.array(mcmc[1])))
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
                else:
                    popt_result[jj] = np.percentile(popt_chain[:, jj],50)
                    popt_err[:,jj] = np.array([np.std(popt_chain[:, jj]),np.std(popt_chain[:, jj])])
        
            self.line_wvl_fit_hmc = popt_result[:self.line_number]
            self.int_total_fit_hmc = popt_result[self.line_number:self.line_number*2]
            self.fwhm_fit_hmc = popt_result[self.line_number*2:self.line_number*3]
            self.int_cont_fit_hmc = popt_result[-1]

            self.line_wvl_err_hmc = popt_err[:,:self.line_number]
            self.int_total_err_hmc = popt_err[:,self.line_number:self.line_number*2]
            self.fwhm_err_hmc = popt_err[:,self.line_number*2:self.line_number*3]
            self.int_cont_err_hmc = popt_err[:,-1]

        if save_chain:
            self.HahnMC_chain = popt_chain

    def get_fit_params(self,method="lse"):
        if self.custom_func is not None:
            return self.custom_fit, self.custom_err
        elif method == "hmc":
            return {"wvl_fit": self.line_wvl_fit_hmc,
                "int_fit": self.int_total_fit_hmc,
                "fwhm_fit": self.fwhm_fit_hmc,
                "cont_fit": self.int_cont_fit_hmc,
                "wvl_err": self.line_wvl_err_hmc,
                "int_err": self.int_total_err_hmc,
                "fwhm_err": self.fwhm_err_hmc,
                "cont_err": self.int_cont_err_hmc,
                "method": method}
        elif method == "lse":
            return  {"wvl_fit": self.line_wvl_fit,
                "int_fit": self.int_total_fit,
                "fwhm_fit": self.fwhm_fit,
                "cont_fit": self.int_cont_fit,
                "wvl_err": self.line_wvl_err,
                "int_err": self.int_total_err,
                "fwhm_err": self.fwhm_err,
                "cont_err": self.int_cont_err,
                "method": method}
        else:
            sys.exit("No such method.")

    def get_fit_profile(self,method="lse"):
        fit_params = self.get_fit_params(method=method)
        if self.custom_func is not None:
            spec_fit = self.custom_func(self.wvl,*self.custom_fit)
            res_fit = self.data - self.custom_func(self.wvl,*self.custom_fit)                
        elif self.same_width is True:
            p_fit = np.concatenate((fit_params["wvl_fit"],fit_params["int_fit"],fit_params["fwhm_fit"],
                                    fit_params["cont_fit"]),axis=None)
            spec_fit = self.multi_gaussian_same_width(self.wvl,*p_fit)
            res_fit = self.data - self.multi_gaussian_same_width(self.wvl,*p_fit) 
        else:
            p_fit = np.concatenate((fit_params["wvl_fit"],fit_params["int_fit"],fit_params["fwhm_fit"],
                                    fit_params["cont_fit"]),axis=None)
            spec_fit = self.multi_gaussian_diff_width(self.wvl,*p_fit) 
            res_fit = self.data - self.multi_gaussian_diff_width(self.wvl,*p_fit)    
        return spec_fit, res_fit

    def plot(self, plot_fit=True,plot_params=True, plot_mcmc=False,plot_hmc=False,
                color_style="Red",plot_title=None,xlabel=None,ylabel=None,xlim=None,
                line_caption=None,figsize_scale=1,save_fig=False,
                params_prec = {"int":2,"wvl":1,"fwhm":2,"cont":1},
                save_fname="./fit_result.pdf",save_fmt="pdf",save_dpi=300):
        '''
            Plot the input spectra and fitting results. 

            Parameters
            ----------
            plot_fit : bool, optional
            If True, plots the fitting results as well. Default is True.
            plot_params : bool, optional 
            If True, plot the fitted parameters by the figure. Default is True. 
            plot_mcmc : bool, optional 
            If True, plots the MCMC results. Default is False.
            plot_hmc: bool, optional 
            If True, plots the Monte Carlo fitting results using the method described in 
            Hahn et al. 2012, ApJ, 753, 36. Default is False. 
            xlim : [left limit, right_lim], optional 
            If provided, set the left and right limit of the x-axis. Default is None. 
            color_style : {"Red","Yellow","Green","Blue","Purple"}, optional
            Color style of the plot. Default is "Red".
            plot_title : string, optional 
            Set to be the title of the plot. Default is None.
            xlabel: string, optional 
            Set to be the label of the x-axis. Default is None.
            ylabel: string, optional 
            Set to be the label of the y-axis. Default is None. 
            save_fig: bool, optional 
            If True, save the plot to local directory. Default is False.
            save_fname: string, optional 
            The filename of the saved plot. Default is "./fit_result.pdf"
            save_fmt: string, optional 
            Format of the saved file, e.g., "pdf", "png", "svg"... Default is "pdf".
            save_dpi: int, optional
            Dots per inch (DPI) of the saved plot. Default is 300. 

        '''
        with rc_context(rc_context_dict):
            if (self.custom_func is not None) and (plot_params is True):
                warn("Use custom function in the fitting. Will not plot fitted parameters.")
                plot_params = False

            self.wvl_plot = np.linspace(self.wvl[0],self.wvl[-1],5*len(self.wvl))
            if (plot_fit is True) and (plot_params is True):
                fig = plt.figure(figsize=((8+3*np.ceil(self.line_number/2))*figsize_scale,8*figsize_scale),
                                constrained_layout=True)
                gs_fig = fig.add_gridspec(1, 2,figure=fig,width_ratios=[8., 3*np.ceil(self.line_number/2)])
                gs_plot = gs_fig[0].subgridspec(2, 1,height_ratios=[5,2])
                ax = fig.add_subplot(gs_plot[0])
                ax_res = fig.add_subplot(gs_plot[1])
            elif (plot_fit is True) and (plot_params is False):
                fig = plt.figure(figsize=(8*figsize_scale,8*figsize_scale))
                gs_plot = fig.add_gridspec(2, 1,height_ratios=[5,2])
                ax = fig.add_subplot(gs_plot[0])
                ax_res = fig.add_subplot(gs_plot[1])
            else:
                fig, ax = plt.subplots(figsize=(8*figsize_scale,6*figsize_scale),constrained_layout=True)
            ax.tick_params(which="both",labelsize=18,right=True,top=True)
            #ax.set_xlabel("Wavelength",fontsize=18)
            if ylabel is None:
                ax.set_ylabel("Intensity",fontsize=18)
            else:
                ax.set_ylabel(ylabel,fontsize=18)
            ax.tick_params(which="major",width=1.2,length=8,direction="in")
            ax.tick_params(which="minor",width=1.2,length=4,direction="in")
            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax.yaxis.get_offset_text().set_fontsize(18)
            ax.yaxis.get_offset_text().set_y(1.05)
            #print("Wavelength:",self.line_wvl_fit)
            #print("Width:",self.fwhm_fit)
            #print("Width Error:",self.fwhm_err)

            if color_style == "Red":
                colors = ["#E87A90","#FEDFE1","black","#E9002D","#DBD0D0"]
            elif color_style == "Green":
                colors = ["#00896C","#A8D8B9","black","#33A6B8","#DBD0D0"]
            elif color_style == "Yellow":
                colors = ["#FFBA84","#FAD689","black","#FC9F4D","#DBD0D0"]
            elif color_style == "Blue":
                colors = ["#3A8FB7","#A5DEE4","black","#58B2DC","#DBD0D0"]
            elif color_style == "Purple":
                colors = ["#8F77B5","#B28FCE","black","#6A4C9C","#DBD0D0"]


            if self.err is None:
                ln1, = ax.step(self.wvl,self.data,where="mid",color=colors[0],label = r"$I_{\rm obs}$",lw=2,zorder=15)
            else:
                ln1 = ax.errorbar(self.wvl,self.data,yerr = self.err,ds='steps-mid',color=colors[0],capsize=2,
                label = r"$I_{\rm obs}$",lw=2,zorder=15)
            
            ax.fill_between(self.wvl,np.ones_like(self.wvl)*np.min(self.data),self.data,
                            step='mid',color=colors[1],alpha=0.6)

            if self.mask is not None:
                for ii, mask_ in enumerate(self.mask):
                    ax.axvspan(mask_[0],mask_[1],color=colors[4],alpha=0.4)


            if plot_fit is True:
                if self.custom_func is not None:
                    pass
                elif plot_hmc is True:
                    line_wvl_plot = self.line_wvl_fit_hmc
                    int_total_plot = self.int_total_fit_hmc
                    fwhm_plot = self.fwhm_fit_hmc
                    int_cont_plot = self.int_cont_fit_hmc
                    line_wvl_err_plot = self.line_wvl_err_hmc
                    int_total_err_plot = self.int_total_err_hmc
                    fwhm_err_plot = self.fwhm_err_hmc
                    int_cont_err_plot = self.int_cont_err_hmc

                    int_total_text_fmt = r'$I_0 = {{{:#.{int_data_prec}g}}}' + \
                    r'_{{-{:#.{int_err_prec}g}}}^{{+{:#.{int_err_prec}g}}}$'
                    line_wvl_text_fmt = r'$\lambda_0 = {{{:#.{wvl_data_prec}g}}}' + \
                    r'_{{-{:#.{wvl_err_prec}g}}}^{{+{:#.{wvl_err_prec}g}}}$'
                    fwhm_text_fmt = r'$\Delta \lambda = {{{:#.{fwhm_data_prec}g}}}' + \
                    r'_{{-{:#.{fwhm_err_prec}g}}}^{{+{:#.{fwhm_err_prec}g}}}$'
                    int_cont_text_fmt = r'$I_{{\rm bg}} = {{{:#.{cont_data_prec}g}}}' + \
                    r'_{{-{:#.{cont_err_prec}g}}}^{{+{:#.{cont_err_prec}g}}}$'
                else:
                    line_wvl_plot = self.line_wvl_fit
                    int_total_plot = self.int_total_fit
                    fwhm_plot = self.fwhm_fit
                    int_cont_plot = self.int_cont_fit
                    line_wvl_err_plot = self.line_wvl_err
                    int_total_err_plot = self.int_total_err
                    fwhm_err_plot = self.fwhm_err
                    int_cont_err_plot = self.int_cont_err

                    int_total_text_fmt = r'$I_0 = {:#.{int_data_prec}g}\pm{:#.{int_err_prec}g}$'
                    line_wvl_text_fmt = r'$\lambda_0 = {:#.{wvl_data_prec}g}\pm{:#.{wvl_err_prec}g}$'
                    fwhm_text_fmt = r'$\Delta \lambda = {:#.{fwhm_data_prec}g}\pm{:#.{fwhm_err_prec}g}$'
                    int_cont_text_fmt = r'$I_{{\rm bg}} = {:#.{cont_data_prec}g}\pm{:#.{cont_err_prec}g}$'

                if self.custom_func is not None:
                    spec_fit = self.custom_func(self.wvl_plot,*self.custom_fit)
                    res_fit = self.data_tofit - self.custom_func(self.wvl_tofit,*self.custom_fit)                
                elif self.same_width is True:
                    p_fit = np.concatenate((line_wvl_plot,int_total_plot,fwhm_plot,
                                            int_cont_plot),axis=None)
                    spec_fit = self.multi_gaussian_same_width(self.wvl_plot,*p_fit)
                    res_fit = self.data_tofit - self.multi_gaussian_same_width(self.wvl_tofit,*p_fit) 
                else:
                    p_fit = np.concatenate((line_wvl_plot,int_total_plot,fwhm_plot,
                                            int_cont_plot),axis=None)
                    spec_fit = self.multi_gaussian_diff_width(self.wvl_plot,*p_fit) 
                    res_fit = self.data_tofit - self.multi_gaussian_diff_width(self.wvl_tofit,*p_fit)                           

                ln2, = ax.plot(self.wvl_plot,spec_fit,color=colors[2],ls="-",label = r"$I_{\rm fit}$",lw=2,
                                zorder=16,alpha=0.7)

                if self.custom_func is None:
                    if self.line_number > 1:
                        if self.same_width is True:
                            for jj in range(self.line_number):
                                line_profile = gaussian(self.wvl_plot, line_wvl_plot[jj],
                                                        int_total_plot[jj], fwhm_plot) \
                                                + int_cont_plot
                                ax.plot(self.wvl_plot,line_profile,color=colors[3],ls="--",lw=2,alpha=0.8)
                        else:
                            for jj in range(self.line_number):
                                line_profile = gaussian(self.wvl_plot, line_wvl_plot[jj],
                                                        int_total_plot[jj], fwhm_plot[jj]) \
                                                + int_cont_plot
                                ax.plot(self.wvl_plot,line_profile,color=colors[3],ls="--",lw=2,alpha=0.8)   
    

                if self.err is None:
                    ax_res.scatter(self.wvl_tofit,res_fit,marker="o",s=15,color=colors[3])
                else:
                    ax_res.errorbar(self.wvl_tofit,res_fit,self.err_tofit,ds='steps-mid',color=colors[3],capsize=3,
                                    lw=2,ls="none",marker="o",markersize=5)
                    chi2_fit = np.sum((res_fit/self.err_tofit)**2)/self.dof
                    ax_res.text(0.98,0.9,r"$\chi^2 = {:.1f}$".format(chi2_fit),fontsize=18,
                                ha="right",va="top",transform=ax_res.transAxes)

                ax_res.axhline(0,ls="--",lw=2,color="#91989F",alpha=0.7) 
                if xlabel is None:
                    ax_res.set_xlabel(r"$\textrm{Wavelength}$",fontsize=18)
                else:
                    ax_res.set_xlabel(xlabel,fontsize=18)
                ax_res.set_ylabel(r"$r$",fontsize=18)
                ax_res.tick_params(which="both",labelsize=18,top=True,right=True)
                ax_res.tick_params(which="major",width=1.2,length=8,direction="in")
                ax_res.tick_params(which="minor",width=1.2,length=4,direction="in")
                ax_res.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                ax_res.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

                if xlim is not None:
                    ax_res.set_xlim(xlim)
                if self.mask is not None:
                    for ii, mask_ in enumerate(self.mask):
                        ax_res.axvspan(mask_[0],mask_[1],color=colors[4],alpha=0.4)


            if xlim is not None:
                ax.set_xlim(xlim)

            if plot_title is not None:
                ax.set_title(plot_title,fontsize=18)

            if (plot_params and plot_fit) is True:
                gs_text = gs_fig[1].subgridspec(2, 1,height_ratios=[5,2])
                text_ncol = np.ceil(self.line_number/2)
                ax_text = fig.add_subplot(gs_text[0])
                ax_text.axis("off")
                if plot_mcmc or plot_hmc:
                    for ii in range(self.line_number):
                        int_data_prec = np.ceil(np.log10(np.abs(int_total_plot[ii]))).astype("int") - \
                            np.min(np.ceil(np.log10(int_total_err_plot[:,ii]))).astype("int") + params_prec["int"]
                        

                        ax_text.text(0.05+(ii//2)/text_ncol,0.87-(ii%2)*0.45,int_total_text_fmt.format(num2tex(int_total_plot[ii]),
                        num2tex(int_total_err_plot[0,ii]),num2tex(int_total_err_plot[1,ii]),int_data_prec = int_data_prec,
                        int_err_prec = params_prec["int"]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)

                        wvl_data_prec = np.ceil(np.log10(np.abs(line_wvl_plot[ii]))).astype("int") - \
                            np.min(np.ceil(np.log10(line_wvl_err_plot[:,ii])).astype("int")) + params_prec["wvl"]


                        ax_text.text(0.05+(ii//2)/text_ncol,0.78-(ii%2)*0.45,line_wvl_text_fmt.format(num2tex(line_wvl_plot[ii]),
                        num2tex(line_wvl_err_plot[0,ii]),num2tex(line_wvl_err_plot[1,ii]),wvl_data_prec = wvl_data_prec,
                        wvl_err_prec = params_prec["wvl"]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)

                        if self.same_width is True:

                            fwhm_data_prec = np.ceil(np.log10(np.abs(fwhm_plot))).astype("int") - \
                                np.min(np.ceil(np.log10(fwhm_err_plot)).astype("int")) + params_prec["fwhm"]

                            ax_text.text(0.05+(ii//2)/text_ncol,0.69-(ii%2)*0.45,fwhm_text_fmt.format(num2tex(fwhm_plot),
                            num2tex(fwhm_err_plot[0]),num2tex(fwhm_err_plot[1]),fwhm_data_prec = fwhm_data_prec,
                            fwhm_err_prec = params_prec["fwhm"]),ha = 'left',va = 'center', 
                            color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)
                        else:
                            fwhm_data_prec = np.ceil(np.log10(np.abs(fwhm_plot[ii]))).astype("int") - \
                                np.min(np.ceil(np.log10(fwhm_err_plot[:,ii]))).astype("int") + params_prec["fwhm"]

                            ax_text.text(0.05+(ii//2)/text_ncol,0.69-(ii%2)*0.45,fwhm_text_fmt.format(num2tex(fwhm_plot[ii]),
                            num2tex(fwhm_err_plot[0,ii]),num2tex(fwhm_err_plot[1,ii]),fwhm_data_prec = fwhm_data_prec,
                            fwhm_err_prec = params_prec["fwhm"]),ha = 'left',va = 'center', 
                            color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)
                        
                        cont_data_prec = np.ceil(np.log10(np.abs(int_cont_plot))).astype("int") - \
                            np.min(np.ceil(np.log10(int_cont_err_plot))).astype("int") + params_prec["cont"]

                        ax_text.text(0.05+(ii//2)/text_ncol,0.60-(ii%2)*0.45,int_cont_text_fmt.format(num2tex(int_cont_plot),
                            num2tex(int_cont_err_plot[0]),num2tex(int_cont_err_plot[1]),cont_data_prec = cont_data_prec,
                            cont_err_prec = params_prec["cont"]),ha = 'left',va = 'center', 
                            color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes) 
                else:
                    for ii in range(self.line_number):
                        int_data_prec = np.ceil(np.log10(np.abs(int_total_plot[ii]))).astype("int") - \
                            np.ceil(np.log10(int_total_err_plot[ii])).astype("int") + params_prec["int"]

                        ax_text.text(0.05+(ii//2)/text_ncol,0.87-(ii%2)*0.45,int_total_text_fmt.format(num2tex(int_total_plot[ii]),
                        num2tex(int_total_err_plot[ii]),int_data_prec = int_data_prec,int_err_prec = params_prec["int"]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)

                        wvl_data_prec = np.ceil(np.log10(np.abs(line_wvl_plot[ii]))).astype("int") - \
                            np.ceil(np.log10(line_wvl_err_plot[ii])).astype("int") + params_prec["wvl"]

                        ax_text.text(0.05+(ii//2)/text_ncol,0.78-(ii%2)*0.45,line_wvl_text_fmt.format(num2tex(line_wvl_plot[ii]),
                        num2tex(line_wvl_err_plot[ii]),wvl_data_prec = wvl_data_prec,wvl_err_prec = params_prec["wvl"]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)

                        if self.same_width is True:
                            fwhm_data_prec = np.ceil(np.log10(np.abs(fwhm_plot))).astype("int") - \
                                np.ceil(np.log10(fwhm_err_plot)).astype("int") + params_prec["fwhm"]
                            ax_text.text(0.05+(ii//2)/text_ncol,0.69-(ii%2)*0.45,fwhm_text_fmt.format(num2tex(fwhm_plot),
                            num2tex(fwhm_err_plot),fwhm_data_prec = fwhm_data_prec,fwhm_err_prec = params_prec["fwhm"]),ha = 'left',va = 'center', 
                            color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)
                        else:
                            fwhm_data_prec = np.ceil(np.log10(np.abs(fwhm_plot[ii]))).astype("int") - \
                                np.ceil(np.log10(fwhm_err_plot[ii])).astype("int") + params_prec["fwhm"]
                            ax_text.text(0.05+(ii//2)/text_ncol,0.69-(ii%2)*0.45,fwhm_text_fmt.format(num2tex(fwhm_plot[ii]),
                            num2tex(fwhm_err_plot[ii]),fwhm_data_prec = fwhm_data_prec,fwhm_err_prec = params_prec["fwhm"]),ha = 'left',va = 'center', 
                            color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)
                        
                        cont_data_prec = np.ceil(np.log10(np.abs(int_cont_plot))).astype("int") - \
                            np.ceil(np.log10(int_cont_err_plot)).astype("int") + params_prec["cont"]
                        ax_text.text(0.05+(ii//2)/text_ncol,0.60-(ii%2)*0.45,int_cont_text_fmt.format(num2tex(int_cont_plot),
                            num2tex(int_cont_err_plot),cont_data_prec = cont_data_prec,cont_err_prec = params_prec["cont"]),ha = 'left',va = 'center', 
                            color = 'black',fontsize = 18,linespacing=1.5,transform=ax_text.transAxes)
                
                if self.line_number > 1:
                    if line_caption is None:
                        line_caption = []
                        for ii in range(self.line_number):
                            line_caption.append(r"\textbf{\textsc{"+num_to_roman(ii+1,uppercase=False)+r"}}")
                    
                    for ii in range(self.line_number):
                        ax_text.text(0.05+(ii//2)/text_ncol,0.95-(ii%2)*0.45,line_caption[ii],
                        ha = 'left',va = 'center',color = 'black',fontsize = 18,linespacing=1.5,
                        transform=ax_text.transAxes)

                        ax_text.axhline(0.915-(ii%2)*0.45,xmin=0.05+(ii//2)/text_ncol,xmax=(ii//2+0.8)/text_ncol,color="#787878",alpha=0.7,lw=4)
                        if self.same_width is True:
                            ax.text(self.line_wvl_fit[ii],2.355*self.int_total_fit[ii]/self.fwhm_fit/np.sqrt(2*np.pi)+\
                                0.05*np.diff(ax.get_ylim()) + self.int_cont_fit,line_caption[ii],
                            ha = 'center',va = 'bottom',color = colors[3],fontsize = 18,linespacing=1.5)

                        else:
                            ax.text(self.line_wvl_fit[ii],2.355*self.int_total_fit[ii]/self.fwhm_fit[ii]/np.sqrt(2*np.pi)+\
                                0.05*np.diff(ax.get_ylim()) + self.int_cont_fit,line_caption[ii],
                            ha = 'center',va = 'bottom',color = colors[3],fontsize = 18,linespacing=1.5)

                        

                    ax.set_ylim(top=ax.get_ylim()[1]*1.07)
                else:
                    if line_caption is not None:
                        ax_text.text(0.05+(ii//2)/text_ncol,0.95-(ii%2)*0.45,line_caption,
                        ha = 'left',va = 'center',color = 'black',fontsize = 18,linespacing=1.5,
                        transform=ax_text.transAxes)

            if save_fig is True:
                plt.savefig(fname=save_fname,format=save_fmt,dpi=save_dpi)

            plt.show()

        return ax

    def multi_gaussian_same_width(self,wvl,*args):
        '''
        Generate the spectra which consists of multiple Gaussian spectral lines
        with the same width. 

        Parameters
        ----------
        wvl : 1-D array
        Wavelength points of the spectra. 
        *args : 
        Parameters of the Gaussian profiles, which follows the order: 
        wvl_line1, wvl_line2, ..., int_total_line1, int_total_line2, ...,
        fwhm_all_line, int_cont 
        '''
        spec_syn = np.zeros_like(wvl)

        for ii in range(self.line_number):
            spec_syn = spec_syn \
                    + gaussian(wvl,line_wvl=args[ii], 
                                int_total=args[ii + self.line_number], 
                                fwhm=args[-2])
            
        spec_syn = spec_syn + args[-1]

        return spec_syn

    def multi_gaussian_diff_width(self,wvl,*args):
        '''
        Generate the spectra which consists of multiple Gaussian spectral lines
        with different widths. 

        Parameters
        ----------
        wvl : 1-D array
        Wavelength points of the spectra. 
        *args : 
        Parameters of the Gaussian profiles, which follows the order: 
        wvl_line1, wvl_line2, ..., int_total_line1, int_total_line2, ...,
        fwhm_line1, fwhm_line2, ..., int_cont 
        '''
        spec_syn = np.zeros_like(wvl)

        for ii in range(self.line_number):
            spec_syn = spec_syn \
                    + gaussian(wvl,line_wvl=args[ii], 
                                int_total=args[ii + self.line_number], 
                                fwhm=args[ii + self.line_number*2])
            
        spec_syn = spec_syn + args[-1]

        return spec_syn
    
    def multi_gaussian_mixture_width(self,wvl,*args):
        '''
        Generate the spectra which consists of multiple Gaussian spectral lines,
        some of which have the same widths (defined in self.same_width). 

        Parameters
        ----------
        wvl : 1-D array
        Wavelength points of the spectra. 
        *args : 
        Parameters of the Gaussian profiles, which follows the order: 
        wvl_line1, wvl_line2, ..., int_total_line1, int_total_line2, ...,
        fwhm_line_diff_width1, fwhm_line_diff_width2, ..., fwhm_line_same_width,
        int_cont 
        '''
        spec_syn = np.zeros_like(wvl)
        fwhm_arg_index_cont = 0 

        for ii in range(self.line_number):
            if self.same_width[ii] is True:
                spec_syn = spec_syn \
                   + gaussian(wvl,line_wvl=args[ii], 
                              int_total=args[ii + self.line_number], 
                              fwhm=args[-2])
            else:
                spec_syn = spec_syn \
                   + gaussian(wvl,line_wvl=args[ii], 
                              int_total=args[ii + self.line_number], 
                              fwhm=args[self.line_number*2+fwhm_arg_index_cont])

                fwhm_arg_index_cont += 1

        spec_syn = spec_syn + args[-1]
        return spec_syn
                
class SpectrumFitRow:
    '''
        SpectrumFitRow fits spectral lines in a row. (e.g., along the slit of a slit-jaw
        spectrograph)
    '''
    def __init__(self, data, wvl, line_number, line_wvl_init, int_max_init, \
                fwhm_init, err=None, same_width=False, stray_light=False, \
                stray_wvl_init=None, stray_int_total=None, stray_fwhm=None, \
                mask=None,custom_func=None,custom_init=None) -> None:

        '''
            Initialize the SpectrumFitRow class
            
            Parameters
            ----------
            data : 2-D array
                Input 2-D spectra (intensity) to fit.
            wvl : 1-D array
                1-D wavelength grid of the spectra.
            line_number: integer
                The desired number of lines to fit.
            line_wvl_init : scalar or 1-D array
                Initial wavelength(s) of the spectral line core(s).
            int_max_init : scalar or 1-D array
                Initial value(s) of the peak intensity.
            fwhm_init : scalar or 1-D array
                Initial value(s) of the full width at half maximum (FWHM).
            err : 1-D array , optional 
                Errors in the intensity at different wavelengths. If provided,
                will be used to calculate the likelihood function. Default is None.
            same_width : bool or a list of bool, optional 
                If True, forces the fitted spectral lines have the same width. If provided 
                as a list of bools, only forces the spectral lines corresponding to True value
                have the same width in the fitting. Default is False.
            stray_light : bool, optional 
                If True, adds the stray light profile to the fitting. Default is False.
            stray_wvl_init : scalar or 1-D array, optional
                Initial wavelength(s) of the stray light line core(s). Default is None.
            stray_int_total: scalar or 1-D array, optional 
                Integrated intensity of the stray light profile(s). Default is None.
            stray_fwhm: scalar or 1-D array, optional 
                Full width at half maximum (FWHM) of the stray light profile(s).
                Default is None. 
            mask: [N,2] array, optional 
                If provided, will mask the data between the N intervals in the fitting. 
                For example, [[2,4],[10,12]] will mask the data points within [2,4] and 
                [10,12]. Default is None.  

        '''
        
        #input parameters
        self.data = data
        self.wvl = wvl
        self.line_number = line_number
        self.line_wvl_init = np.array(line_wvl_init)
        self.int_max_init = np.array(int_max_init)
        self.fwhm_init = np.array(fwhm_init)
        self.err = err
        self.same_width = same_width
        self.stray_light = stray_light
        self.stray_wvl_init = stray_wvl_init
        self.stray_int_total = stray_int_total
        self.stray_fwhm = stray_fwhm
        self.mask = mask

        #instance properties
        self.shape = data.shape
        self.wvl_plot = np.linspace(self.wvl[0],self.wvl[-1],5*len(self.wvl))
        if len(data.shape) == 1:
            self.frame_number = 1
        else:
            self.frame_number = self.shape[0]

        self.custom_func = custom_func
        self.custom_init = custom_init

        #fitted parameters
        if self.custom_func is None:        
            self.line_wvl_fit = np.zeros((self.frame_number,self.line_number))
            self.line_wvl_err = np.zeros((self.frame_number,self.line_number))

            self.int_total_fit = np.zeros((self.frame_number,self.line_number))
            self.int_total_err = np.zeros((self.frame_number,self.line_number))

            if same_width is True:
                self.fwhm_fit = np.zeros(self.frame_number)
                self.fwhm_err = np.zeros(self.frame_number)
            else:
                self.fwhm_fit = np.zeros((self.frame_number,self.line_number))
                self.fwhm_err = np.zeros((self.frame_number,self.line_number))

            self.int_cont_fit = np.zeros(self.frame_number)
            self.int_cont_err = np.zeros(self.frame_number)
        else:
            self.custom_fit = np.zeros((self.frame_number,len(custom_init)))
            self.custom_err = np.zeros((self.frame_number,len(custom_init)))

        #create single fit object
        self.single_fit_list = []
        if self.err is None:
            for ii in range(self.frame_number):
                self.single_fit_list.append(SpectrumFitSingle(data=self.data[ii,:], wvl=self.wvl,
                                                    line_number=self.line_number,line_wvl_init=self.line_wvl_init,
                                                    int_max_init=self.int_max_init,fwhm_init=self.fwhm_init,
                                                    err=self.err,same_width=self.same_width,stray_light=self.stray_light,
                                                    stray_wvl_init=self.stray_wvl_init,stray_int_total=self.stray_int_total,
                                                    stray_fwhm=self.stray_fwhm,mask=self.mask,custom_func=self.custom_func,
                                                    custom_init=self.custom_init))
        else:
            for ii in range(self.frame_number):
                self.single_fit_list.append(SpectrumFitSingle(data=self.data[ii,:], wvl=self.wvl,
                                                    line_number=self.line_number,line_wvl_init=self.line_wvl_init,
                                                    int_max_init=self.int_max_init,fwhm_init=self.fwhm_init,
                                                    err=self.err[ii,:],same_width=self.same_width,stray_light=self.stray_light,
                                                    stray_wvl_init=self.stray_wvl_init,stray_int_total=self.stray_int_total,
                                                    stray_fwhm=self.stray_fwhm,mask=self.mask,custom_func=self.custom_func,
                                                    custom_init=self.custom_init))


    def run_lse(self,ignore_err=False,absolute_sigma=True,prev_init=True,clean_obj_list=False):
        '''
            Performs least square estimation (Chi square fitting)
            to the spectral line(s).

            Parameters
            ----------
            ignore_err : bool, optional
            If True, ignores the input error. Default is False. 
            absolute_sigma: bool, optional
            If True, the errors have the same unit as data. Default is True.  
        '''

        for ii in range(self.frame_number):
            if (ii == 0) or (prev_init is False):
                pass
            else:
                if self.custom_func is None:
                    self.single_fit_list[ii].line_wvl_init = self.single_fit_list[ii-1].line_wvl_fit
                    self.single_fit_list[ii].fwhm_init = self.single_fit_list[ii-1].fwhm_fit
                    self.single_fit_list[ii].int_max_init = 2.355*self.single_fit_list[ii-1].int_total_fit/ \
                        np.sqrt(2.*np.pi)/self.single_fit_list[ii-1].fwhm_fit
                else:
                    self.single_fit_list[ii].custom_init = self.single_fit_list[ii-1].custom_fit
            
            try:
                self.single_fit_list[ii].run_lse(ignore_err=ignore_err,absolute_sigma=absolute_sigma)
            except:
                pass
            
            if self.custom_func is None:
                self.line_wvl_fit[ii:,] = self.single_fit_list[ii].line_wvl_fit
                self.line_wvl_err[ii:,] = self.single_fit_list[ii].line_wvl_err
                self.int_total_fit[ii,:] = self.single_fit_list[ii].int_total_fit
                self.int_total_err[ii,:] = self.single_fit_list[ii].int_total_err
                self.fwhm_fit[ii] = self.single_fit_list[ii].fwhm_fit
                self.fwhm_err[ii] = self.single_fit_list[ii].fwhm_err
                self.int_cont_fit[ii] = self.single_fit_list[ii].int_cont_fit
                self.int_cont_err[ii] = self.single_fit_list[ii].int_cont_err
            else:
                self.custom_fit[ii,:] = self.single_fit_list[ii].custom_fit
                self.custom_err[ii,:] = self.single_fit_list[ii].custom_err
        
        if clean_obj_list:
            delattr(self,"single_fit_list")

    def plot_fit(self, plot_fit=True, plot_hmc=False,plot_mcmc=False,
                color_style="Red",plot_title=None,xlabel=None,ylabel=None,xlim=None,
                save_fig=False,save_fname="./fit_row_result.pdf",save_fmt="pdf",save_dpi=300):

        '''
            Plot the input spectra and fitting results of all the points

            Parameters
            ----------
            plot_fit : bool, optional
            If True, plots the fitting results as well. Default is True.
            plot_params : bool, optional 
            If True, plot the fitted parameters by the figure. Default is True. 
            plot_mcmc : bool, optional 
            If True, plots the MCMC results. Default is False.
            plot_hmc: bool, optional 
            If True, plots the Monte Carlo fitting results using the method described in 
            Hahn et al. 2012, ApJ, 753, 36. Default is False. 
            xlim : [left limit, right_lim], optional 
            If provided, set the left and right limit of the x-axis. Default is None. 
            color_style : {"Red","Yellow","Green","Blue","Purple"}, optional
            Color style of the plot. Default is "Red".
            plot_title : string, optional 
            Set to be the title of the plot. Default is None.
            xlabel: string, optional 
            Set to be the label of the x-axis. Default is None.
            ylabel: string, optional 
            Set to be the label of the y-axis. Default is None. 
            save_fig: bool, optional 
            If True, save the plot to local directory. Default is False.
            save_fname: string, optional 
            The filename of the saved plot. Default is "./fit_result.pdf"
            save_fmt: string, optional 
            Format of the saved file, e.g., "pdf", "png", "svg"... Default is "pdf".
            save_dpi: int, optional
            Dots per inch (DPI) of the saved plot. Default is 300. 

        '''

        with rc_context(rc_context_dict):
            nrows = int(np.ceil(self.frame_number/4.))
            fig, axes = plt.subplots(nrows,4,figsize=(16,nrows*3),constrained_layout=True)

            if color_style == "Red":
                colors = ["#E87A90","#FEDFE1","black","#E9002D","#DBD0D0"]
            elif color_style == "Green":
                colors = ["#00896C","#A8D8B9","black","#33A6B8","#DBD0D0"]
            elif color_style == "Yellow":
                colors = ["#FFBA84","#FAD689","black","#FC9F4D","#DBD0D0"]
            elif color_style == "Blue":
                colors = ["#3A8FB7","#A5DEE4","black","#58B2DC","#DBD0D0"]
            elif color_style == "Purple":
                colors = ["#8F77B5","#B28FCE","black","#6A4C9C","#DBD0D0"]

            for ii, ax_ in enumerate(axes.flatten()):
                if ii < self.frame_number:
                    if self.err is None:
                        ln1, = ax_.step(self.wvl,self.data[ii],where="mid",color=colors[0],label = r"$I_{\rm obs}$",lw=2)
                    else:
                        ln1 = ax_.errorbar(self.wvl,self.data[ii],yerr = self.err[ii],ds='steps-mid',color=colors[0],capsize=2,
                        label = r"$I_{\rm obs}$",lw=1.5)
                    
                    ax_.fill_between(self.wvl,np.ones_like(self.wvl)*np.min(self.data[ii]),self.data[ii],
                        step='mid',color=colors[1],alpha=0.6)

                    if plot_fit is True:
                        if self.same_width is True:
                            p_fit = np.concatenate((self.line_wvl_fit[ii],self.int_total_fit[ii],self.fwhm_fit[ii],
                                                    self.int_cont_fit[ii]),axis=None)
                            spec_fit = self.single_fit_list[ii].multi_gaussian_same_width(self.wvl_plot,*p_fit)
                        else:
                            p_fit = np.concatenate((self.line_wvl_fit[ii],self.int_total_fit[ii],self.fwhm_fit[ii],
                                                    self.int_cont_fit[ii]),axis=None)
                            spec_fit = self.single_fit_list[ii].multi_gaussian_diff_width(self.wvl_plot,*p_fit)                            

                        ln2, = ax_.plot(self.wvl_plot,spec_fit,color=colors[2],ls="-",label = r"$I_{\rm fit}$",lw=1.5,
                        zorder=16,alpha=0.7)

                        if self.line_number > 1:
                            if self.same_width is True:
                                for jj in range(self.line_number):
                                    line_profile = gaussian(self.wvl_plot, self.line_wvl_fit[ii,jj],
                                                            self.int_total_fit[ii,jj], self.fwhm_fit[ii]) \
                                                    + self.int_cont_fit[ii]
                                    ax_.plot(self.wvl_plot,line_profile,color=colors[3],ls="--",lw=1.5,alpha=0.7)
                            else:
                                for jj in range(self.line_number):
                                    line_profile = gaussian(self.wvl_plot, self.line_wvl_fit[ii,jj],
                                                            self.int_total_fit[ii,jj], self.fwhm_fit[ii,jj]) \
                                                    + self.int_cont_fit[ii]
                                    ax_.plot(self.wvl_plot,line_profile,color=colors[3],ls="--",lw=1.5,alpha=0.7)   
                    ax_.tick_params(which="major",length=4,direction="in")
                    if xlim is not None:
                        ax_.set_xlim(xlim)  
                    if self.mask is not None:
                        for jj, mask_ in enumerate(self.mask):
                            ax_.axvspan(mask_[0],mask_[1],color=colors[4],alpha=0.4)
                else:
                    ax_.axis("off")
            for ii in range(-4-(4*nrows-self.frame_number),0-(4*nrows-self.frame_number)):
                if xlabel is None:
                    axes.flatten()[ii].set_xlabel(r"$\textrm{Wavelength}$",fontsize=12)
                else:
                    axes.flatten()[ii].set_xlabel(xlabel,fontsize=12)
            if nrows == 1:
                if ylabel is None:
                    axes[0].set_ylabel("Intensity",fontsize=12)
                else:
                    axes[0].set_ylabel(ylabel,fontsize=12)
            else:
                if ylabel is None:
                    for ii in range(nrows):
                        axes[ii,0].set_ylabel("Intensity",fontsize=12)
                else:
                    for ii in range(nrows):
                        axes[ii,0].set_ylabel(ylabel,fontsize=12)
            if save_fig is True:
                plt.savefig(fname=save_fname,format=save_fmt,dpi=save_dpi)

            plt.show()
        return axes

    def plot_single(self,frame_index,*args,**kwargs):
        ax = self.single_fit_list[frame_index].plot(*args,**kwargs)
        return ax
                                    
    def plot_variation(self,var="fwhm",plot_hmc=False,plot_mcmc=False,
                        xdata=None,xlabel=None,ylabel=None,xlim=None,
                        ylim=None,line_label=None):
        

        with rc_context(rc_context_dict):
            if xdata is None:
                xdata = np.arange(self.frame_number)

            if ylabel is None:
                if var == "fwhm":
                    ylabel = r"FWHM $\Delta \lambda$"
                if var == "int":
                    ylabel = r"Total Intensity $I_0$"
                if var == "wvl":
                    ylabel = r"Line Core Wavelength $\lambda_0$" 
            
            if line_label is None:
                line_label = []
                for ii in range(self.line_number):
                    line_label.append("{:.1f}".format(self.line_wvl_fit[0,ii]))
            
            if var == "fwhm":
                if plot_hmc:
                    pass
                elif plot_mcmc:
                    pass
                else:
                    ydata = np.squeeze(self.fwhm_fit)
                    yerr = np.squeeze(self.fwhm_err)
            elif var == "int":
                if plot_hmc:
                    pass
                elif plot_mcmc:
                    pass
                else:
                    ydata = np.squeeze(self.int_total_fit)
                    yerr = np.squeeze(self.int_total_err)
            elif var == "wvl":  
                if plot_hmc:
                    pass
                elif plot_mcmc:
                    pass
                else:
                    ydata = np.squeeze(self.line_wvl_fit)
                    yerr = np.squeeze(self.line_wvl_err)
            fig, ax = plt.subplots(figsize=(8,6),constrained_layout=True)

            if (self.same_width is True) or (self.line_number == 1):
                ax.errorbar(xdata,ydata,yerr=yerr,lw=2,capsize=3,marker="o",markersize=5)
            else:
                for ii in range(self.line_number):
                    ax.errorbar(xdata,ydata[:,ii],yerr=yerr[:,ii],lw=2,capsize=3,marker="o",markersize=5,
                                label=line_label[ii])
            
            ax.tick_params(labelsize=18,direction="in")
            if xlabel is not None:
                ax.set_xlabel(xlabel,fontsize=18)
            ax.set_ylabel(ylabel,fontsize=18)
            ax.legend(fontsize=18,frameon=False)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            
            plt.show()
            
        return ax 

    def return_dict(self):
        if self.custom_func is None:
            return {"line_wvl_fit":self.line_wvl_fit, "line_wvl_err":self.line_wvl_err,
                    "int_total_fit":self.int_total_fit,"int_total_err":self.int_total_err,
                    "fwhm_fit":self.fwhm_fit,"fwhm_err":self.fwhm_err,
                    "int_cont_fit":self.int_cont_fit,"int_cont_err":self.int_cont_err}
        else:
            return {"custom_fit":self.custom_fit, "custom_err":self.custom_err}

class SpectrumFit2D:
    '''
        SpectrumFit2D performs single/multiple Gaussian fitting 
        to the input spectra with two spatial axes (slit+raster).
        Some codes are copied from EISPAC.  
    '''
    def __init__(self, data, wvl, line_number=None, line_wvl_init=None, int_max_init=None, \
                fwhm_init=None, int_cont_init=None, err=None, err_percent=None, same_width=False, \
                stray_light=False, stray_wvl_init=None, stray_int_total=None, stray_fwhm=None, \
                mask=None,custom_func=None,custom_init=None) -> None:
        

        '''
            Initialize the SpectrumFitSingle class.

            Parameters
            ----------
            data : 3-D array
                Input 3-D spectra[slit,raster,wavelength] to fit. 
            wvl : 1-D array
                1-D wavelength grid of the spectra.
            line_number: integer
                The desired number of lines to fit.
            line_wvl_init : scalar or 1-D array
                Initial wavelength(s) of the spectral line core(s).
            int_max_init : scalar or 1-D array
                Initial value(s) of the peak intensity.
            fwhm_init : scalar or 1-D array
                Initial value(s) of the full width at half maximum (FWHM).
            err : 3-D array , optional 
                Errors in the intensity at different wavelengths. If provided,
                will be used to calculate the likelihood function. Default is None.
            err_percent : scalar or 1-D array (0-100), optional 
                If provided, multiply this percentage to the data to create the errors used 
                in fittings. Conflict with the err parameter. Default is None. 
            same_width : bool or a list of bool, optional 
                If True, forces the fitted spectral lines have the same width. If provided 
                as a list of bools, only forces the spectral lines corresponding to True value
                have the same width in the fitting. Default is False.
            stray_light : bool, optional 
                If True, adds the stray light profile to the fitting. Default is False.
            stray_wvl_init : scalar or 1-D array, optional
                Initial wavelength(s) of the stray light line core(s). Default is None.
            stray_int_total: scalar or 1-D array, optional 
                Integrated intensity of the stray light profile(s). Default is None.
            stray_fwhm: scalar or 1-D array, optional 
                Full width at half maximum (FWHM) of the stray light profile(s).
                Default is None. 
            mask: [N,2] array, optional 
                If provided, will mask the data between the N intervals in the fitting. 
                For example, [[2,4],[10,12]] will mask the data points within [2,4] and 
                [10,12]. Default is None.  
            custom_func : function, optional 
                If provided, fit the spectra using this custom function. Default is None.
            custom_init : scalar, 1-D array like, optional 
                Initial values for the custom fitting function. 
        '''

        self.npix_slit = data.shape[0]
        self.npix_raster = data.shape[1]
        self.npix_wvl = data.shape[2]

        self.data = data
        self.wvl = wvl
        if err_percent is None:
            self.err = err
            if (self.err is not None) and (self.data.shape != self.err.shape):
                warn("Intensity and error do not have the same shape.")
        else:
            if err is None:
                self.err = self.data*err_percent/100
            else:
                warn("Both the err and err_percent parameters are used. Use err instead of err_percent.")
                self.err = err
        self.mask = mask
        self.stray_light = stray_light
        self.stray_wvl_init = stray_wvl_init
        self.stray_int_total = stray_int_total
        self.stray_fwhm = stray_fwhm

        #instance properties
        self.custom_func = custom_func
        self.custom_init = custom_init

        #Create masked wavelength, intensity, and error by deleting the masked 
        #values, since scipy.curvefit does not support masked arrays... 
        if mask is None:
            self.wvl_tofit = self.wvl
            self.data_tofit = self.data
            self.err_tofit = self.err
        elif len(self.wvl.shape) == 1:
            delete_index_all = []
            for ii in range(len(self.mask)):
                delete_index = np.where((self.wvl>=self.mask[ii][0]) & (self.wvl<=self.mask[ii][1]))[0]
                delete_index_all = delete_index_all + delete_index.tolist()
            self.wvl_tofit = np.delete(self.wvl,delete_index_all)
            self.data_tofit = np.delete(self.data,delete_index_all,axis=2)
            if self.err is not None:
                self.err_tofit = np.delete(self.err,delete_index_all,axis=2)
            else:
                self.err_tofit = self.err
        elif len(self.wvl.shape) == 2:
            self.wvl_tofit = np.copy(self.wvl)
            self.data_tofit = np.copy(self.wvl)
            if self.err is not None:
                self.err_tofit = np.copy(self.err)
            else:
                self.err_tofit = self.err
            for jj in range(self.npix_raster):
                delete_index_all = []
                for ii in range(len(self.mask)):
                    delete_index = np.where((self.wvl[jj,:]>=self.mask[ii][0]) & (self.wvl[jj,:]<=self.mask[ii][1]))[0]
                    delete_index_all = delete_index_all + delete_index.tolist()
                self.wvl_tofit[jj,:] = np.delete(self.wvl[jj,:],delete_index_all)
                self.data_tofit[:,jj,:] = np.delete(self.data[:,jj,:],delete_index_all,axis=1)
                if self.err is not None:
                    self.err_tofit[:,jj,:] = np.delete(self.err[:,jj,:],delete_index_all,axis=1)
              

        self.line_number = line_number
        self.line_wvl_init = np.array(line_wvl_init)
        self.int_max_init = np.array(int_max_init)
        self.fwhm_init = np.array(fwhm_init)
        self.same_width = same_width

        self.int_cont_init = int_cont_init
        #If the custom fitting function is not provided, read the initial 
        #values for the embedded multi-gaussian functions. And create the 
        #Chi2 fitted parameters.
        if self.custom_func is None:
            #fitted parameters
            self.line_wvl_fit = np.zeros((self.npix_slit,self.npix_raster,self.line_number))
            self.line_wvl_err = np.zeros((self.npix_slit,self.npix_raster,self.line_number))

            self.int_total_fit = np.zeros((self.npix_slit,self.npix_raster,self.line_number))
            self.int_total_err = np.zeros((self.npix_slit,self.npix_raster,self.line_number))

            if type(same_width) is list:
                self.fwhm_fit = np.zeros((self.npix_slit,self.npix_raster,self.line_number))
                self.fwhm_err = np.zeros((self.npix_slit,self.npix_raster,self.line_number))
                self.line_same_width_index = np.where(self.same_width)[0]
            elif same_width is True:
                self.fwhm_fit = np.zeros((self.npix_slit,self.npix_raster))
                self.fwhm_err = np.zeros((self.npix_slit,self.npix_raster))
            else:
                self.fwhm_fit = np.zeros((self.npix_slit,self.npix_raster,self.line_number))
                self.fwhm_err = np.zeros((self.npix_slit,self.npix_raster,self.line_number))

            self.int_cont_fit = np.zeros((self.npix_slit,self.npix_raster))
            self.int_cont_err = np.zeros((self.npix_slit,self.npix_raster))

        else:
            self.custom_fit = np.zeros((self.npix_slit,self.npix_raster,len(self.custom_init)))
            self.custom_err = np.zeros((self.npix_slit,self.npix_raster,len(self.custom_init)))  

    def run_lse_along_slit(self,index_raster,ignore_err,absolute_sigma,prev_init):
        if len(self.wvl_tofit.shape) == 1:
            fit_model = SpectrumFitRow(self.data_tofit[:,index_raster,:],self.wvl_tofit,
                            self.line_number,self.line_wvl_init,self.int_max_init,self.fwhm_init,
                            self.err_tofit,self.same_width,self.stray_light,self.stray_wvl_init,
                            self.stray_int_total,self.stray_fwhm,mask=None,custom_func=self.custom_func,
                            custom_init=self.custom_init)
        elif len(self.wvl_tofit.shape) == 2:
            fit_model = SpectrumFitRow(self.data_tofit[:,index_raster,:],self.wvl_tofit[index_raster,:],
                            self.line_number,self.line_wvl_init,self.int_max_init,self.fwhm_init,
                            self.err_tofit,self.same_width,self.stray_light,self.stray_wvl_init,
                            self.stray_int_total,self.stray_fwhm,mask=None,custom_func=self.custom_func,
                            custom_init=self.custom_init)
        fit_model.run_lse(ignore_err=ignore_err,absolute_sigma=absolute_sigma,prev_init=prev_init,clean_obj_list=True)
        fit_model_dist = copy.deepcopy(fit_model.return_dict())
        del fit_model
        return fit_model_dist

    def run_lse_mp(self,ncpu="max",ignore_err=False,absolute_sigma=True,prev_init=True):

        if str(ncpu).lower() == 'max' or str(ncpu).lower() == 'none':
            ncpu = mp.cpu_count()
        else:
            ncpu = int(ncpu)
        
        if ncpu == 1:
            for ii in range(self.npix_raster):
                fit_model_dist = self.run_lse_along_slit(ii,ignore_err,absolute_sigma,prev_init)

                if self.custom_func is not None:
                    self.custom_fit[:,ii,:] = fit_model_dist["custom_fit"]
                    self.custom_err[:,ii,:] = fit_model_dist["custom_err"]
                else:
                    self.line_wvl_fit[:,ii,:] = fit_model_dist["line_wvl_fit"]
                    self.line_wvl_err[:,ii,:] = fit_model_dist["line_wvl_err"]
                    self.int_total_fit[:,ii,:] = fit_model_dist["int_total_fit"]
                    self.int_total_err[:,ii,:] = fit_model_dist["int_total_err"]

                    if type(self.same_width) is list:
                        self.fwhm_fit[:,ii,:] = fit_model_dist["fwhm_fit"]
                        self.fwhm_err[:,ii,:] = fit_model_dist["fwhm_err"]
                    elif self.same_width is True:
                        self.fwhm_fit[:,ii] = fit_model_dist["fwhm_fit"]
                        self.fwhm_err[:,ii] = fit_model_dist["fwhm_err"]
                    else:
                        self.fwhm_fit[:,ii,:] = fit_model_dist["fwhm_fit"]
                        self.fwhm_err[:,ii,:] = fit_model_dist["fwhm_err"]

                    self.int_cont_fit[:,ii] = fit_model_dist["int_cont_fit"]
                    self.int_cont_err[:,ii] = fit_model_dist["int_cont_err"]
                
        else:
            if ncpu > self.npix_raster:
                ncpu = self.npix_raster
            
            with mp.Pool(processes=ncpu) as pool:

                args = [(ii,ignore_err,absolute_sigma,prev_init) for ii in range(self.npix_raster)]
                pool_out = pool.starmap(self.run_lse_along_slit,args)

            for ii in range(self.npix_raster):
                if self.custom_func is not None:
                    self.custom_fit[:,ii,:] = pool_out[ii]["custom_fit"]
                    self.custom_err[:,ii,:] = pool_out[ii]["custom_err"]
                else:
                    self.line_wvl_fit[:,ii,:] = pool_out[ii]["line_wvl_fit"]
                    self.line_wvl_err[:,ii,:] = pool_out[ii]["line_wvl_err"]
                    self.int_total_fit[:,ii,:] = pool_out[ii]["int_total_fit"]
                    self.int_total_err[:,ii,:] = pool_out[ii]["int_total_err"]

                    if type(self.same_width) is list:
                        self.fwhm_fit[:,ii,:] = pool_out[ii]["fwhm_fit"]
                        self.fwhm_err[:,ii,:] = pool_out[ii]["fwhm_err"]
                    elif self.same_width is True:
                        self.fwhm_fit[:,ii] = pool_out[ii]["fwhm_fit"]
                        self.fwhm_err[:,ii] = pool_out[ii]["fwhm_err"]
                    else:
                        self.fwhm_fit[:,ii,:] = pool_out[ii]["fwhm_fit"]
                        self.fwhm_err[:,ii,:] = pool_out[ii]["fwhm_err"]

                    self.int_cont_fit[:,ii] = pool_out[ii]["int_cont_fit"]
                    self.int_cont_err[:,ii] = pool_out[ii]["int_cont_err"]

    def plot_fit2d(self,param,line_index=0,xcoord=None,ycoord=None,xmesh=None,ymesh=None,
                    extent=None,vmin=None,vmax=None,scale=None,ax=None,title=None,cmap=None,
                    xlabel=None,ylabel=None,ref_wvl=None,vel_corr="column",inst_width=None,
                    aspect=None,return_data=False):
        with rc_context(rc_context_dict):
            if param == "int":
                data_to_plot = self.int_total_fit[:,:,line_index]
            elif param == "wvl":
                data_to_plot = self.line_wvl_fit[:,:,line_index]
            elif param == "vel":
                data_to_plot = (ref_wvl - self.line_wvl_fit[:,:,line_index])/ref_wvl*const.c.cgs.value/1e5
                if vel_corr == "column":
                    data_to_plot = data_to_plot - np.nanmedian(data_to_plot,axis=0)[np.newaxis,:]
                    print("Doppler velocity corrected by the median of each raster.")
                elif vel_corr == "image":
                    data_to_plot = data_to_plot - np.nanmedian(data_to_plot)
                    print("Doppler velocity corrected by the median of the image.")
            elif (param == "fwhm") or (param == "veff"):
                if type(self.same_width) is list:
                    data_to_plot = self.fwhm_fit[:,:,line_index]
                elif self.same_width is True:
                    data_to_plot = self.fwhm_fit
                else:
                    data_to_plot = self.fwhm_fit[:,:,line_index]

                if inst_width is not None:
                    if isinstance(inst_width, np.ndarray):
                        data_to_plot = np.sqrt(data_to_plot**2 - inst_width[:,np.newaxis]**2)
                    else:
                        data_to_plot = np.sqrt(data_to_plot**2 - inst_width**2)
                    print("Instrumental width corrected.")

                if param == "veff":
                    data_to_plot = data_to_plot/ref_wvl/np.sqrt(4*np.log(2))*const.c.cgs.value/1e5
                    
            elif param == "cont":
                data_to_plot = self.int_cont_fit

            if ax is None:
                fig, ax = plt.subplots(figsize=(7,7),constrained_layout=True)

            if scale is None:
                norm = ImageNormalize(data_to_plot,vmin=vmin,vmax=vmax)
            elif scale == "sqrt":
                norm = ImageNormalize(data_to_plot,vmin=vmin,vmax=vmax,stretch=SqrtStretch())
            elif scale == "log":
                norm = ImageNormalize(data_to_plot,vmin=vmin,vmax=vmax,stretch=LogStretch())

            if cmap is None:
                if param == "vel":
                    cmap = "bwr_r"
                else:
                    cmap = "viridis"

            if (xcoord is not None) and (ycoord is not None):
                im = ax.pcolormesh(xcoord, ycoord, data_to_plot, rasterized=True,cmap=cmap,norm=norm)
            elif (extent is not None):
                im = ax.imshow(data_to_plot, origin="lower",cmap=cmap,extent=extent,norm=norm)
            elif (aspect is not None):
                im = ax.imshow(data_to_plot, origin="lower",cmap=cmap,norm=norm,aspect=aspect)
            
            plot_colorbar(im, ax, width="5%")

            ax.set_title(title,fontsize=16)
            if xlabel is not None:
                ax.set_xlabel(xlabel,fontsize=16)
            if ylabel is not None:
                ax.set_ylabel(ylabel,fontsize=16)
            ax.tick_params(labelsize=16)

            if return_data:
                return data_to_plot

            if ax is None:
                plt.show()




def gaussian(wvl,line_wvl,int_total,fwhm):
    line_profile = 2.355*int_total/np.sqrt(2.*np.pi)/fwhm \
                   *np.exp((-(1. / (2. * (fwhm/2.355)**2)) * (wvl - line_wvl)**2))

    return line_profile

def voigt(wvl, line_wvl,int_total,g_fwhm,l_fwhm):
    line_profile = voigt_profile(wvl - line_wvl,g_fwhm/2.355,l_fwhm/2. )*int_total

    return line_profile


def num_to_roman(number,uppercase=True):
    num = [1, 4, 5, 9, 10, 40, 50, 90,
        100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
        "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12
    
    roman_number=""
    while number:
        div = number // num[i]
        number %= num[i]
  
        while div:
            roman_number += sym[i]
            div -= 1
        i -= 1
    
    if uppercase:
        return roman_number
    else:
        return roman_number.lower()

def plot_colorbar(im, ax, width="3%", height="100%",loc="lower left",fontsize=16):
    clb_ax = inset_axes(ax,width=width,height=height,loc=loc,
                bbox_to_anchor=(1.02, 0., 1, 1),
                 bbox_transform=ax.transAxes,
                 borderpad=0)
    clb = plt.colorbar(im,pad = 0.05,orientation='vertical',ax=ax,cax=clb_ax)
    clb_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    clb_ax.yaxis.get_offset_text().set_fontsize(fontsize)
    clb_ax.tick_params(labelsize=fontsize)
    return clb, clb_ax