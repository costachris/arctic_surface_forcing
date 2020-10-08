import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob
import pandas as pd

from matplotlib.font_manager import FontProperties
from scipy.interpolate import CubicSpline, BSpline



def _create_dataarray(data,
                      nu, 
                      time_val,
                      name = None):
    return xr.DataArray(data = np.expand_dims(data, 1), 
             name = name,
             dims = ('nu', 'time'), 
             coords = {
                 'nu': nu, 
                 'time' : np.expand_dims(np.array(time_val),0)
             })