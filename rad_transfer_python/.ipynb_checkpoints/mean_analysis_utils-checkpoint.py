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



def sin_func(t, A, phase, frequency):
    return A*np.sin(frequency*t - phase)

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