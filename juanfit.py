#!/usr/bin/env python3

'''
    File Name: juanfit.py
    Author: Yingjie Zhu
    Institute: University of Michigan
    Email: yjzhu(at)umich(dot)edu
    Date create: 07/20/2021
    Date Last Modified: 10/06/2021
    Python Version: 3.8
    Statues: In development
    Version: 0.0.1
    Description: This is a naive and dirty python script 
    to perform single/multiple Gaussian fitting to the 
    spectral lines.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import ticker
from IPython.display import display, Math
import emcee
from scipy.special import wofz
from scipy.optimize import curve_fit
from num2tex import num2tex

#plot setting 
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
rcParams['axes.linewidth'] = 2
rcParams['xtick.major.width'] = 1
rcParams['xtick.major.size'] = 4
rcParams['xtick.minor.width'] = 1
rcParams['xtick.minor.size'] =2 
rcParams['ytick.major.width'] = 1
rcParams['ytick.major.size'] = 4
rcParams['ytick.minor.width'] = 1
rcParams['ytick.minor.size'] = 2 
rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc} \usepackage{amsmath}'


class SpectrumFit:
    def __init__(self, data, wvl, line_number, line_wvl_init, int_max_init, \
                fwhm_init, err=None, same_width=False, stray_light=False, \
                stray_light_wvl_fixed=True, stray_wvl_init=None, \
                stray_int_total=None, stray_fwhm=None):
        
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
        self.stray_light_wvl_fixed = stray_light_wvl_fixed
        self.stray_wvl_init = stray_wvl_init
        self.stray_int_total = stray_int_total
        self.stray_fwhm = stray_fwhm

        #instance properties
        self.shape = data.shape
        self.wvl_plot = np.linspace(self.wvl[0],self.wvl[-1],101)
        if len(data.shape) == 1:
            self.frame_number = 1
        else:
            self.frame_number = self.shape[0]

    
        
        #fitted parameters
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



    def run_lse(self,ignore_err=False):
        if self.stray_light is False:
            popt = np.concatenate((self.line_wvl_init,
                                   self.int_max_init*np.sqrt(2.*np.pi)*self.fwhm_init/2.355,
                                   self.fwhm_init,np.mean(self.data[0,:2])),axis = None)
            
            if (self.err is None) or (ignore_err is True):
                if self.same_width is True:
                    for ii in range(self.frame_number):
                        popt, pcov = curve_fit(multi_gaussian_same_width, self.wvl, self.data[ii,:],
                                            p0=popt)
                        
                        
                        self.line_wvl_fit[ii,:] = popt[:self.line_number]
                        self.int_total_fit[ii,:] = popt[self.line_number:self.line_number*2]
                        self.fwhm_fit[ii] = popt[-2]
                        self.int_cont_fit[ii] = popt[-1]

                        perr = np.sqrt(np.diagonal(pcov))
                        self.line_wvl_err[ii,:] = perr[:self.line_number]
                        self.int_total_err[ii,:] = perr[self.line_number:self.line_number*2]
                        self.fwhm_err[ii] = perr[-2]
                        self.int_cont_err[ii] = perr[-1]
                else:
                    for ii in range(self.frame_number):
                        popt, pcov = curve_fit(multi_gaussian_diff_width, self.wvl, self.data[ii,:],
                                            p0=popt)
                        
                        self.line_wvl_fit[ii,:] = popt[:self.line_number]
                        self.int_total_fit[ii,:] = popt[self.line_number:self.line_number*2]
                        self.fwhm_fit[ii] = popt[self.line_number*2:self.line_number*3]
                        self.int_cont_fit[ii] = popt[-1]

                        perr = np.sqrt(np.diagonal(pcov))
                        self.line_wvl_err[ii,:] = perr[:self.line_number]
                        self.int_total_err[ii,:] = perr[self.line_number:self.line_number*2]
                        self.fwhm_err[ii] = perr[self.line_number*2:self.line_number*3]
                        self.int_cont_err[ii] = perr[-1]                  


    
    def plot(self, plot_fit=True, mcmc=False):
            nrows = int(np.ceil(self.frame_number/4.))
            fig, axes = plt.subplots(nrows,4,figsize=(16,nrows*3))

            for ii, ax_ in enumerate(axes.flatten()):
                if ii < self.frame_number:
                    if self.err is None:
                        ln1, = ax_.step(self.wvl,self.data[ii,:],where="mid",color="#E87A90",label = r"$I_{\rm obs}$",lw=2)
                    else:
                        ln1 = ax_.errorbar(self.wvl,self.data[ii,:],yerr = self.err[ii,:],ds='steps-mid',color="#E87A90",capsize=2,
                        label = r"$I_{\rm obs}$",lw=1.5)

                    if plot_fit is True:
                        if self.same_width is True:
                            p_fit = np.concatenate((self.line_wvl_fit[ii],self.int_total_fit[ii],self.fwhm_fit[ii],
                                                    self.int_cont_fit[ii]),axis=None)
                            spec_fit = multi_gaussian_same_width(self.wvl_plot,*p_fit)
                        else:
                            p_fit = np.concatenate((self.line_wvl_fit[ii],self.int_total_fit[ii],self.fwhm_fit[ii],
                                                    self.int_cont_fit[ii]),axis=None)
                            spec_fit = multi_gaussian_diff_width(self.wvl_plot,*p_fit)                            

                        ln2, = ax_.plot(self.wvl_plot,spec_fit,color="#FC9F40",ls="-",label = r"$I_{\rm fit}$",lw=1.5)

                        if self.line_number > 1:
                            if self.same_width is True:
                                for jj in range(self.line_number):
                                    line_profile = gaussian(self.wvl_plot, self.line_wvl_fit[ii,jj],
                                                            self.int_total_fit[ii,jj], self.fwhm_fit[ii]) \
                                                    + self.int_cont_fit[ii]
                                    ax_.plot(self.wvl_plot,line_profile,color="#E9002D",ls="-",lw=1.5,alpha=0.7)
                            else:
                                for jj in range(self.line_number):
                                    line_profile = gaussian(self.wvl_plot, self.line_wvl_fit[ii,jj],
                                                            self.int_total_fit[ii,jj], self.fwhm_fit[ii,jj]) \
                                                    + self.int_cont_fit[ii]
                                    ax_.plot(self.wvl_plot,line_profile,color="#E9002D",ls="-",lw=1.5,alpha=0.7)                                
                                    


class SpectrumFitSingle:
    '''
        SpectrumFitSingle performs single/multiple Gaussian fitting 
        to the input spectra.
    '''
    def __init__(self, data, wvl, line_number, line_wvl_init, int_max_init, \
                fwhm_init, err=None, same_width=False, stray_light=False, \
                stray_light_wvl_fixed=True, stray_wvl_init=None, \
                stray_int_total=None, stray_fwhm=None):

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
            same_width : bool, optional 
                If True, forces the fitted spectral lines have the same width.
                Default is False.
            stray_light : bool, optional 
                If True, adds the stray light profile to the fitting. Default is False.
            stray_light_wvl_fixed : True, optional 
                If True, fixed the line centroid wavelength of the stray light 
                profile. Default is True. 
            stray_wvl_init : scalar or 1-D array, optional
                Initial wavelength(s) of the stray light line core(s). Default is None.
            stray_int_total: scalar or 1-D array, optional 
                Integrated intensity of the stray light profile(s). Default is None.
            stray_fwhm: scalar or 1-D array, optional 
                Full width at half maximum (FWHM) of the stray light profile(s).
                Default is None. 


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
        self.stray_light_wvl_fixed = stray_light_wvl_fixed
        self.stray_wvl_init = stray_wvl_init
        self.stray_int_total = stray_int_total
        self.stray_fwhm = stray_fwhm

        #instance properties
        self.shape = data.shape
        self.wvl_plot = np.linspace(self.wvl[0],self.wvl[-1],101)

        #fitted parameters
        self.line_wvl_fit = np.zeros(self.line_number)
        self.line_wvl_err = np.zeros(self.line_number)

        self.int_total_fit = np.zeros(self.line_number)
        self.int_total_err = np.zeros(self.line_number)

        if same_width is True:
            self.fwhm_fit = np.float64(0.0)
            self.fwhm_err = np.float64(0.0)
        else:
            self.fwhm_fit = np.zeros(self.line_number)
            self.fwhm_err = np.zeros(self.line_number)

        self.int_cont_fit = np.float64(0.0)
        self.int_cont_err = np.float64(0.0)

    def run_lse(self,ignore_err=False):
        '''
            Performs least square estimation (Chi square fitting)
            to the spectral line(s).

            Parameters
            ----------
            ignore_err : bool, optional
            If True, ignores the input error. Default is False. 
        '''
        if self.stray_light is False:
            popt = np.concatenate((self.line_wvl_init,
                                   self.int_max_init*np.sqrt(2.*np.pi)*self.fwhm_init/2.355,
                                   self.fwhm_init,np.mean(self.data[:2])),axis = None)
            
            if (self.err is None) or (ignore_err is True): # no error 
                if self.same_width is True:
                    popt, pcov = curve_fit(multi_gaussian_same_width, self.wvl, self.data,
                                        p0=popt)
                    
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
                    popt, pcov = curve_fit(multi_gaussian_diff_width, self.wvl, self.data,
                                        p0=popt)
                    
                    self.line_wvl_fit = popt[:self.line_number]
                    self.int_total_fit = popt[self.line_number:self.line_number*2]
                    self.fwhm_fit = popt[self.line_number*2:self.line_number*3]
                    self.int_cont_fit = popt[-1]

                    perr = np.sqrt(np.diagonal(pcov))
                    self.line_wvl_err = perr[:self.line_number]
                    self.int_total_err = perr[self.line_number:self.line_number*2]
                    self.fwhm_err = perr[self.line_number*2:self.line_number*3]
                    self.int_cont_err = perr[-1]        
            else: # with error
                if self.same_width is True:
                    popt, pcov = curve_fit(multi_gaussian_same_width, self.wvl, self.data,
                                        p0=popt,sigma=self.err)
                    
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
                    popt, pcov = curve_fit(multi_gaussian_diff_width, self.wvl, self.data,
                                        p0=popt,sigma=self.err)
                    
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
            print("Fitting with stray light is not supported in this version.")

    def run_HahnMC(self,ignore_err=False,n_chain=10000,cred_lvl=90):
        '''
            Fit line profiles using the Monte-Carlo method described 
            in Hahn et al. 2012, ApJ, 753, 36.
        '''

        self.run_lse(ignore_err=ignore_err)
        if self.same_width is True:
            p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,self.fwhm_fit,
                                    self.int_cont_fit),axis=None)
            spec_fit = multi_gaussian_same_width(self.wvl,*p_fit)
        else:
            p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,self.fwhm_fit,
                                    self.int_cont_fit),axis=None)
            spec_fit = multi_gaussian_diff_width(self.wvl,*p_fit)  

        if self.err is None:                        
            err_diff = np.abs(self.data - spec_fit)
        else:
            err_diff = np.maximum(self.err,np.abs(self.data - spec_fit))
        
        random_err = np.zeros((n_chain,len(self.wvl)))
        for ii in range(len(self.wvl)):
            random_err[:,ii] = np.random.normal(0, err_diff[ii], n_chain)

        if self.stray_light is False:
            popt = np.concatenate((self.line_wvl_fit,self.int_total_fit,
                        self.fwhm_fit,self.int_cont_fit),axis = None)
        else:
            print("Fitting with stray light is not supported in this version.")

        popt_chain = np.zeros((n_chain,popt.shape[0]))
        if (self.err is None) or (ignore_err is True): # no error 
            if self.same_width is True:
                for ii in range(n_chain):
                    popt_chain[ii,:], _ = curve_fit(multi_gaussian_same_width, self.wvl,
                                                self.data+random_err[ii,:],p0=popt)
                
                popt_result = np.zeros_like(popt)
                popt_err = np.zeros((2,popt.shape[0]))
                for jj in range(popt.shape[0]):
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
                
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
                    popt_chain[ii,:], _ = curve_fit(multi_gaussian_diff_width, self.wvl,
                                                self.data+random_err[ii,:],p0=popt)
                
                popt_result = np.zeros_like(popt)
                popt_err = np.zeros((2,popt.shape[0]))
                for jj in range(popt.shape[0]):
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    #print(type(np.array(mcmc[1])))
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
            
                self.line_wvl_fit_hmc = popt_result[:self.line_number]
                self.int_total_fit_hmc = popt_result[self.line_number:self.line_number*2]
                self.fwhm_fit_hmc = popt_result[self.line_number*2:self.line_number*3]
                self.int_cont_fit_hmc = popt_result[-1]

                self.line_wvl_err_hmc = popt_err[:,:self.line_number]
                self.int_total_err_hmc = popt_err[:,self.line_number:self.line_number*2]
                self.fwhm_err_hmc = popt_err[:,self.line_number*2:self.line_number*3]
                self.int_cont_err_hmc = popt_err[:,-1]
        else: # with error
            if self.same_width is True:
                for ii in range(n_chain):
                    popt_chain[ii,:], _ = curve_fit(multi_gaussian_same_width, self.wvl,
                                                self.data+random_err[ii,:],p0=popt,sigma=err_diff)
                
                popt_result = np.zeros_like(popt)
                popt_err = np.zeros((2,popt.shape[0]))
                for jj in range(popt.shape[0]):
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
                
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
                    popt_chain[ii,:], _ = curve_fit(multi_gaussian_diff_width, self.wvl,
                                                self.data+random_err[ii,:],p0=popt,sigma=err_diff)
                
                popt_result = np.zeros_like(popt)
                popt_err = np.zeros((2,popt.shape[0]))
                for jj in range(popt.shape[0]):
                    mcmc_result = np.percentile(popt_chain[:, jj], [50-cred_lvl/2, 50, 50+cred_lvl/2])
                    residual = np.diff(mcmc_result)
                    #print(type(np.array(mcmc[1])))
                    popt_result[jj] = mcmc_result[1]
                    popt_err[:,jj] = np.array([residual[0],residual[1]])
            
                self.line_wvl_fit_hmc = popt_result[:self.line_number]
                self.int_total_fit_hmc = popt_result[self.line_number:self.line_number*2]
                self.fwhm_fit_hmc = popt_result[self.line_number*2:self.line_number*3]
                self.int_cont_fit_hmc = popt_result[-1]

                self.line_wvl_err_hmc = popt_err[:,:self.line_number]
                self.int_total_err_hmc = popt_err[:,self.line_number:self.line_number*2]
                self.fwhm_err_hmc = popt_err[:,self.line_number*2:self.line_number*3]
                self.int_cont_err_hmc = popt_err[:,-1]


    
    def plot(self, plot_fit=True,plot_params=True, plot_mcmc=False,plot_hmc=False,
                xlim=None,color_style="Warm",plot_title=None):
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
            xlim : [left limit, right_lim], optional 
            If provided, set the left and right limit of the x-axis. Default is None. 
            color_style : {"Warm","Cold"}, optional
            Color style of the plot. Default is "Warm".
        '''
        if plot_fit is True:
            fig = plt.subplots(figsize=(8,10))
            ax = plt.subplot2grid((8,1),(0,0),rowspan = 5,colspan = 1)
            ax_res = plt.subplot2grid((8,1),(6,0),rowspan = 2,colspan = 1)
        else:
            fig, ax = plt.subplots(figsize=(8,6))
        ax.tick_params(labelsize=18)
        #ax.set_xlabel("Wavelength",fontsize=18)
        ax.set_ylabel("Intensity",fontsize=18)
        ax.tick_params(which="major",width=1.2,length=8,direction="in")
        ax.tick_params(which="minor",width=1.2,length=4,direction="in")
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        #print("Wavelength:",self.line_wvl_fit)
        #print("Width:",self.fwhm_fit)
        #print("Width Error:",self.fwhm_err)

        if color_style == "Warm":
            colors= ["#E87A90","#FEDFE1","black","#E9002D"]
        elif color_style == "Cold":
            colors = ["#00896C","#A8D8B9","black","#33A6B8"]

        if self.err is None:
            ln1, = ax.step(self.wvl,self.data,where="mid",color=colors[0],label = r"$I_{\rm obs}$",lw=2,zorder=15)
        else:
            ln1 = ax.errorbar(self.wvl,self.data,yerr = self.err,ds='steps-mid',color=colors[0],capsize=2,
            label = r"$I_{\rm obs}$",lw=2,zorder=15)
        
        ax.fill_between(self.wvl,np.ones_like(self.wvl)*np.min(self.data),self.data,
                        step='mid',color=colors[1],alpha=0.6)

        if plot_fit is True:
            if plot_hmc is True:
                line_wvl_plot = self.line_wvl_fit_hmc
                int_total_plot = self.int_total_fit_hmc
                fwhm_plot = self.fwhm_fit_hmc
                int_cont_plot = self.int_cont_fit_hmc
                line_wvl_err_plot = self.line_wvl_err_hmc
                int_total_err_plot = self.int_total_err_hmc
                fwhm_err_plot = self.fwhm_err_hmc
                int_cont_err_plot = self.int_cont_err_hmc

                int_total_text_fmt = r'$I_0 = {:.3g}_{{-{:.1g}}}^{{+{:.1g}}}$'
                line_wvl_text_fmt = r'$\lambda_0 = {:.6g}_{{-{:.1g}}}^{{+{:.1g}}}$'
                fwhm_text_fmt = r'$\Delta \lambda = {:.3f}_{{-{:.1g}}}^{{+{:.1g}}}$'
                int_cont_text_fmt = r'$I_{{\rm bg}} = {:.2g}_{{-{:.1g}}}^{{+{:.1g}}}$'
            else:
                line_wvl_plot = self.line_wvl_fit
                int_total_plot = self.int_total_fit
                fwhm_plot = self.fwhm_fit
                int_cont_plot = self.int_cont_fit
                line_wvl_err_plot = self.line_wvl_err
                int_total_err_plot = self.int_total_err
                fwhm_err_plot = self.fwhm_err
                int_cont_err_plot = self.int_cont_err

                int_total_text_fmt = r'$I_0 = {:.3g}\pm{:.1g}$'
                line_wvl_text_fmt = r'$\lambda_0 = {:.6g}\pm{:.1g}$'
                fwhm_text_fmt = r'$\Delta \lambda = {:.3f}\pm{:.1g}$'
                int_cont_text_fmt = r'$I_{{\rm bg}} = {:.2g}\pm{:.1g}$'

            if self.same_width is True:
                p_fit = np.concatenate((line_wvl_plot,int_total_plot,fwhm_plot,
                                        int_cont_plot),axis=None)
                spec_fit = multi_gaussian_same_width(self.wvl_plot,*p_fit)
                res_fit = self.data - multi_gaussian_same_width(self.wvl,*p_fit) 
            else:
                p_fit = np.concatenate((line_wvl_plot,int_total_plot,fwhm_plot,
                                        int_cont_plot),axis=None)
                spec_fit = multi_gaussian_diff_width(self.wvl_plot,*p_fit) 
                res_fit = self.data - multi_gaussian_diff_width(self.wvl,*p_fit)                           

            ln2, = ax.plot(self.wvl_plot,spec_fit,color=colors[2],ls="-",label = r"$I_{\rm fit}$",lw=2,
                            zorder=16,alpha=0.7)

            if self.line_number > 1:
                if self.same_width is True:
                    for jj in range(self.line_number):
                        line_profile = gaussian(self.wvl_plot, line_wvl_plot[jj],
                                                int_total_plot[jj], fwhm_plot) \
                                        + int_cont_plot
                        ax.plot(self.wvl_plot,line_profile,color=colors[3],ls="--",lw=2,alpha=0.7)
                else:
                    for jj in range(self.line_number):
                        line_profile = gaussian(self.wvl_plot, line_wvl_plot[jj],
                                                int_total_plot[jj], fwhm_plot[jj]) \
                                        + int_cont_plot
                        ax.plot(self.wvl_plot,line_profile,color=colors[3],ls="--",lw=2,alpha=0.7)   

            if self.err is None:
                ax_res.scatter(self.wvl,res_fit,marker="o",s=15,color=colors[3])
            else:
                ax_res.errorbar(self.wvl,res_fit,self.err,ds='steps-mid',color=colors[3],capsize=3,
                                lw=2,ls="none",marker="o",markersize=5)

            ax_res.axhline(0,ls="--",lw=2,color="#91989F",alpha=0.7) 
            ax_res.set_xlabel(r"$\textrm{Wavelength}$",fontsize=18)
            ax_res.set_ylabel(r"$r$",fontsize=18)
            ax_res.tick_params(labelsize=18)
            ax_res.tick_params(which="major",width=1.2,length=8,direction="in")
            ax_res.tick_params(which="minor",width=1.2,length=4,direction="in")
            ax_res.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
            ax_res.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

        if xlim is not None:
            ax.set_xlim(xlim)

        if plot_title is not None:
            ax.set_title(plot_title,fontsize=18)

        if plot_params is True:
            if plot_mcmc or plot_hmc:
                for ii in range(self.line_number):
                    ax.text(1.05+(ii//2)*0.5,0.9-(ii%2)*0.5,int_total_text_fmt.format(num2tex(int_total_plot[ii]),
                    num2tex(int_total_err_plot[0,ii]),num2tex(int_total_err_plot[1,ii])),ha = 'left',va = 'center', 
                    color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)

                    ax.text(1.05+(ii//2)*0.5,0.8-(ii%2)*0.5,line_wvl_text_fmt.format(line_wvl_plot[ii],
                    line_wvl_err_plot[0,ii],line_wvl_err_plot[1,ii]),ha = 'left',va = 'center', 
                    color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)

                    if self.same_width is True:
                        ax.text(1.05+(ii//2)*0.5,0.7-(ii%2)*0.5,fwhm_text_fmt.format(fwhm_plot,
                        fwhm_err_plot[0],fwhm_err_plot[1]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)
                    else:
                        ax.text(1.05+(ii//2)*0.5,0.7-(ii%2)*0.5,fwhm_text_fmt.format(fwhm_plot[ii],
                        fwhm_err_plot[0,ii],fwhm_err_plot[1,ii]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)
                    
                    ax.text(1.05+(ii//2)*0.5,0.6-(ii%2)*0.5,int_cont_text_fmt.format(num2tex(int_cont_plot),
                        num2tex(int_cont_err_plot[0]),num2tex(int_cont_err_plot[1])),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes) 
            else:
                for ii in range(self.line_number):
                    ax.text(1.05+(ii//2)*0.5,0.9-(ii%2)*0.5,int_total_text_fmt.format(num2tex(int_total_plot[ii]),
                    num2tex(int_total_err_plot[ii])),ha = 'left',va = 'center', 
                    color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)

                    ax.text(1.05+(ii//2)*0.5,0.8-(ii%2)*0.5,line_wvl_text_fmt.format(line_wvl_plot[ii],
                    line_wvl_err_plot[ii]),ha = 'left',va = 'center', 
                    color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)

                    if self.same_width is True:
                        ax.text(1.05+(ii//2)*0.5,0.7-(ii%2)*0.5,fwhm_text_fmt.format(fwhm_plot,
                        fwhm_err_plot),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)
                    else:
                        ax.text(1.05+(ii//2)*0.5,0.7-(ii%2)*0.5,fwhm_text_fmt.format(fwhm_plot[ii],
                        fwhm_err_plot[ii]),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)
                    
                    ax.text(1.05+(ii//2)*0.5,0.6-(ii%2)*0.5,int_cont_text_fmt.format(num2tex(int_cont_plot),
                        num2tex(int_cont_err_plot)),ha = 'left',va = 'center', 
                        color = 'black',fontsize = 18,linespacing=1.5,transform=ax.transAxes)

        #plt.tight_layout()


def multi_gaussian_same_width(wvl,*args):
    line_number = (len(args)-2)//2

    spec_syn = np.zeros_like(wvl)

    for ii in range(line_number):
        spec_syn = spec_syn \
                   + gaussian(wvl,line_wvl=args[ii], 
                              int_total=args[ii + line_number], 
                              fwhm=args[-2])
        
    spec_syn = spec_syn + args[-1]

    return spec_syn

def multi_gaussian_diff_width(wvl,*args):
    line_number = (len(args)-1)//3

    spec_syn = np.zeros_like(wvl)

    for ii in range(line_number):
        spec_syn = spec_syn \
                   + gaussian(wvl,line_wvl=args[ii], 
                              int_total=args[ii + line_number], 
                              fwhm=args[ii + line_number*2])
        
    spec_syn = spec_syn + args[-1]

    return spec_syn


def gaussian(wvl,line_wvl,int_total,fwhm):
    line_profile = 2.355*int_total/np.sqrt(2.*np.pi)/fwhm \
                   *np.exp((-(1. / (2. * (fwhm/2.355)**2)) * (wvl - line_wvl)**2))

    return line_profile


#plt.plot(np.linspace(0,10,301),multi_gaussian_same_width(np.linspace(0,10,301),3,8,5,2,2,0.5))
#plt.show()

