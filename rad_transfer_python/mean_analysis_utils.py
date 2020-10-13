import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import CubicSpline, BSpline
from scipy.optimize import leastsq


def _subset_k_to_instrument_grid(ds):
    return ds.where((ds.nu > 491.79016) &
                 (ds.nu < 1799.8556 ),
                 drop = True)



def sin_func(t, A, phase):
    return A*np.sin(((2*np.pi/12) - phase)*t)

def weighted_emission_height(da_weight, 
                             da_emission_height,):
    '''Calculate emission height, weighting by `weight_field_name`.
    Appropriately handles nans. '''
    weighted_num = da_weight * da_emission_height
    weighted_arr_masked = da_weight.where(~da_emission_height.isnull(), np.nan)


    sum_num = xr.apply_ufunc(np.nansum, 
                   weighted_num,
                   input_core_dims = [["nu"]], 
                   vectorize = True)
    sum_denom = xr.apply_ufunc(np.nansum, 
                   weighted_arr_masked,
                   input_core_dims = [["nu"]], 
                   vectorize = True)
    weighted_eh = sum_num/sum_denom
    
    return weighted_eh


def plot_forcing_spectral_residual(rad_diff, 
                                   nu, 
                                   title = None,
                                   xlabel = None,
                                   ylabel = None,
                                   xlims = None,
                                   ylims = None,
                                   alpha = 0.9,
                                   linewidth = 0.4,
                                   figsize = (11,6)):
    
    '''Plot spectral residual in mW. '''
    
    wavelength = 2*np.pi/nu
    fig = plt.figure(figsize = figsize)
    
    ax1 = fig.add_subplot(111)    
    ax2 = ax1.twiny()
    
    
    
    pos_rad = np.where(rad_diff>0, rad_diff, np.nan)
    neg_rad = np.where(rad_diff<=0, rad_diff, np.nan)

    ax1.plot(nu, 1e3*pos_rad, 
             alpha=0.7, 
             c = 'r',
             linewidth = linewidth)
    
    ax1.plot(nu, 1e3*neg_rad,
             alpha=0.7, 
             c = 'b',
             linewidth = linewidth)
    
    
    if not xlims is None:
        ax1.set_xlim(xlims)
    if not ylims is None:
        ax1.set_ylim(ylims)
    ax1.set_xlabel('Wavenumber [$cm^{-1}$]', 
                   weight = 'bold',
                   fontsize = 10)
    
    def tick_function(X):
        wnum = (1/X)*1e4
        return ["%.1f" % z for z in wnum]

    ax1Ticks = ax1.get_xticks()
    ax2Ticks = ax1Ticks
    ax2.set_xticks(ax2Ticks)
    ax2.set_xbound(ax1.get_xbound())
    ax2.set_xlabel(r'Wavelength $[\mu m]$', 
                   fontsize = 8,
                   weight = 'bold')
    ax2.set_xticklabels(tick_function(ax2Ticks))
#     ax2.xaxis.set_ticks_position('bottom') 
#     ax2.spines['bottom'].set_position(('outward', 40))
    
    ax1.set_ylabel(r'$\Delta R$ [$mW m^{-2} sr^{-1} cm^{-1}$]', weight = 'bold')
    # plt.xlim((4,30))
    if not title is None:
        ax1.set_title(title, weight = 'bold')
    ax1.grid()