#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from glob import glob
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from hapi import *
from joblib import Parallel, delayed

from matplotlib.font_manager import FontProperties

from simulate_radiances_utils import *
from locations import *
# %run simulate_radiances_utils.py


font = {'weight' : 'bold',
        'size'   : 22}
label_fontdict = {'weight' : 'bold',
        'size'   : 22}
title_fontdict = {'weight' : 'bold',
        'size'   : 21}

matplotlib.rc('font', **font)


# In[ ]:





# In[2]:


# Configs
# loc_label = 'n_greenland'
# loc_label = 's_greenland'
loc_label = 'summit'
# loc_label = 'e_greenland'
# loc_label = 'w_greenland'


# In[17]:


# timestamps to process
# timestamps = pd.date_range('2015-09-01 01:30:00', '2015-09-07 22:30:00', freq = '3h')
timestamps = pd.date_range('2015-03-01 01:30:00', '2015-03-07 22:30:00', freq = '3h')


# In[12]:


# timestamps


# In[3]:


rel_data_dir = '/export/data2/groupMembers/cchristo/'
output_cs_matrix_rel_path = rel_data_dir + 'cs_matrices/' + loc_label + '/'
output_cs_matrix_path_format = output_cs_matrix_rel_path + '{year}/{month:02d}/cs_matrix_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.nc'


# In[4]:


print('Location: ', loc_label)
print('Writing output to: ' + output_cs_matrix_rel_path)

reanlysis_dir = rel_data_dir + 'reanalysis_3d/merra2/2015/'
file_type = '*.nc4'

all_file_paths = [y for x in os.walk(reanlysis_dir) for y in glob(os.path.join(x[0], file_type))]

# set lat, lon
loc_lat =  loc_lat_lon_map[loc_label][0] #72.5796
loc_lon = loc_lat_lon_map[loc_label][1] #-38.4588


# In[5]:


ds_3d = xr.open_mfdataset(all_file_paths, combine='by_coords')


# In[20]:


# ds_single = ds_3d.sel(lat = loc_lat, lon = loc_lon , time = 0, method = 'nearest')
# ds_single.load()


# In[ ]:





# In[6]:


# %%capture
# Min wavenumber
xmin = 400
# Maximum wavenumber
xmax = 2100
# Actually downloading the data 
# (we have to know the HITRAN molecule numbers, given in http://hitran.org/docs/molec-meta/)
fetch('H2O_S',1,1,xmin,xmax)
fetch('CO2_S',2,1,xmin,xmax)
fetch('CH4_S',6,1,xmin,xmax)
# fetch('O2',7,1,xmin,xmax)
nu_,sw_ = getColumns('CO2_S',['nu','sw'])


# In[7]:


# Let us hust get line position nu and line strength sw for the different molecules:
nu_H2O,sw_H2O = getColumns('H2O_S',['nu','sw'])
nu_CH4,sw_CH4 = getColumns('CH4_S',['nu','sw'])
nu_CO2,sw_CO2 = getColumns('CO2_S',['nu','sw'])


# In[54]:


NLEV = len(ds_3d['lev'])

def make_cs_matrix_time_i(ds_3d, time_i):
    '''
    
    Args
        time_i - pd.Timestamp of date to process
    '''
    
    ds_single = ds_3d.sel(lat = loc_lat, lon = loc_lon, method = 'nearest')
    ds_single = ds_single.sel(time = time_i)
    time_i = ds_single.time
    ds_single.load()
    p_prof, T_prof, dz_prof,  vmr_h2o_prof, VCD_dry_prof, rho_N_h2o_prof, rho_N_prof = compute_profile_properties_merra2(ds_single, verbose=False)
    
    out_cs_matrix_path = output_cs_matrix_path_format.format(year = time_i['time.year'].item(), 
                                                             month = time_i['time.month'].item(), 
                                                             day = time_i['time.day'].item(),
                                                             hour = time_i['time.hour'].item(), 
                                                             minute = time_i['time.minute'].item())
    # make dirs for file if they don't already exist
    out_cs_matrix_dir = os.path.dirname(out_cs_matrix_path)
    if not os.path.exists(out_cs_matrix_dir):
        os.makedirs(out_cs_matrix_dir)
    
    
    # if file already exists, skip
    if not os.path.exists(out_cs_matrix_path):
        create_cross_section_matrix_hapi(p_prof, T_prof, 
                                     xmin, xmax, 
                                     time_i = time_i,
                                     output_path=out_cs_matrix_path)
    else:
        print('File already exists: ', out_cs_matrix_path)


# In[17]:


# os.path.dirname('dir1/dir2/file.nc')


# In[8]:


def make_cs_matrix_timeseries(ds_3d, timestamps):
    '''Given a list of pd.Timestamps, create cs matrices and save.'''
    for time_i in timestamps:
        print(time_i)
        try:
            make_cs_matrix_time_i(ds_3d, time_i)
        except Exception as e: 
            print('Could not process', time_i)
            print('Reason: ', str(e))
    
    


# In[60]:


# where the magic happens
make_cs_matrix_timeseries(ds_3d, timestamps)


# In[43]:


# dates


# In[ ]:





# In[12]:


# Parallel(n_jobs=30)(delayed(make_cs_matrices_time_i)(ds_3d, time_ii) for time_ii in range(len(ds_3d.time)))


# In[13]:


# make_cs_matrices_time_i(ds_3d, -1)


# In[14]:


# dd = xr.open_dataset('/home/cchristo/proj_christian/data/cs_matrices/summit/2015/07/cs_matrix_20150707_2230.nc')


# In[16]:


# dd


# In[26]:


# create_cross_section_matrix_hapi()


# In[ ]:


# # nu_, cs_o2 = absorptionCoefficient_Voigt(SourceTables='O2', Environment={'p':1,'T':270}, WavenumberStep=1)

# cs_matrix_co2 = np.zeros((len(nu_),NLEV))
# cs_matrix_ch4 = np.zeros((len(nu_),NLEV))
# cs_matrix_h2o = np.zeros((len(nu_),NLEV))
# # Loop over each layer 
# for i in range(NLEV - 69):
#     print(str(i)+'/'+str(NLEV), end='\r')
#     p_ = p_prof[i]/101325
# #     print(p_)
#     T_ = T_prof[i]
#     nu_, cs_co2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
#     nu_, cs_ch4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
#     nu_, cs_h2o = absorptionCoefficient_Voigt(SourceTables='H2O_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
#     cs_matrix_co2[:,i] = cs_co2
#     cs_matrix_ch4[:,i] = cs_ch4
#     cs_matrix_h2o[:,i] = cs_h2o


# In[ ]:




