#Version 2.0 of Yingjie's naive and dirty fitting code

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rcParams
rcParams['text.usetex'] = True
rcParams['font.family'] = 'serif'
from IPython.display import display, Math
import emcee
from scipy.special import wofz
from scipy.optimize import curve_fit


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
        self.wvl_plot = np.linspace(self.wvl[0],self.wvl[-1],51)
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
        if self.stray_light is False:
            popt = np.concatenate((self.line_wvl_init,
                                   self.int_max_init*np.sqrt(2.*np.pi)*self.fwhm_init/2.355,
                                   self.fwhm_init,np.mean(self.data[:2])),axis = None)
            
            if (self.err is None) or (ignore_err is True):
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


    
    def plot(self, plot_fit=True, mcmc=False):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.tick_params(labelsize=18)
        ax.set_xlabel("Wavelength",fontsize=18)
        ax.set_ylabel("Intensity",fontsize=18)
        ax.tick_params(which="major",width=1.2,length=8)
        ax.tick_params(which="minor",width=1.2,length=4)
        print("Wavelength:",self.line_wvl_fit)
        print("Width:",self.fwhm_fit)
        print("Width Error:",self.fwhm_err)

        if self.err is None:
            ln1, = ax.step(self.wvl,self.data,where="mid",color="#E87A90",label = r"$I_{\rm obs}$",lw=2)
        else:
            ln1 = ax.errorbar(self.wvl,self.data,yerr = self.err,ds='steps-mid',color="#E87A90",capsize=2,
            label = r"$I_{\rm obs}$",lw=2)

        if plot_fit is True:
            if self.same_width is True:
                p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,self.fwhm_fit,
                                        self.int_cont_fit),axis=None)
                spec_fit = multi_gaussian_same_width(self.wvl_plot,*p_fit)
            else:
                p_fit = np.concatenate((self.line_wvl_fit,self.int_total_fit,self.fwhm_fit,
                                        self.int_cont_fit),axis=None)
                spec_fit = multi_gaussian_diff_width(self.wvl_plot,*p_fit)                            

            ln2, = ax.plot(self.wvl_plot,spec_fit,color="#FC9F40",ls="-",label = r"$I_{\rm fit}$",lw=2)

            if self.line_number > 1:
                if self.same_width is True:
                    for jj in range(self.line_number):
                        line_profile = gaussian(self.wvl_plot, self.line_wvl_fit[jj],
                                                self.int_total_fit[jj], self.fwhm_fit) \
                                        + self.int_cont_fit
                        ax.plot(self.wvl_plot,line_profile,color="#E9002D",ls="-",lw=2,alpha=0.7)
                else:
                    for jj in range(self.line_number):
                        line_profile = gaussian(self.wvl_plot, self.line_wvl_fit[jj],
                                                self.int_total_fit[jj], self.fwhm_fit[jj]) \
                                        + self.int_cont_fit
                        ax.plot(self.wvl_plot,line_profile,color="#E9002D",ls="-",lw=2,alpha=0.7)       


    
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

