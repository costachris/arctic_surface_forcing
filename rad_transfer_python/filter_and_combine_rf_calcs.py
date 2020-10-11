#!/usr/bin/env python
# coding: utf-8

# In[2]:


### Filter surface radiative calculations for sky-free conditions and 
### and combine into monthly files. 


# In[4]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from joblib import Parallel, delayed

from matplotlib.font_manager import FontProperties

from simulate_radiances_utils import *
# %run simulate_radiances_utils.py
import urllib 


# In[5]:


path_to_cloud_mask = '/net/fluo/data2/groupMembers/cchristo/misc_data/cloud_bool_rad_29.nc'

# rel_dir_to_all_surface_spec = '/net/fluo/data2/groupMembers/cchristo/results/rt_results/'
# output_rel_dir_to_all_surface_spec = '/net/fluo/data2/groupMembers/cchristo/results/rt_results_monthly/'

rel_dir_to_all_surface_spec = '/net/fluo/data2/groupMembers/cchristo/results/rt_results_preindus_CO2/'
output_rel_dir_to_all_surface_spec = '/net/fluo/data2/groupMembers/cchristo/results/rt_results_preindus_CO2_monthly/'


rel_dir_format = '{year}/{month:02d}/'
# fname_format = out_data_path_format = '{year}/{month:02d}/surface_fields_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.nc'


# In[8]:


all_months = pd.date_range(start = '2011-01',
                          end = '2016-01',
                          freq = 'M')


# In[14]:


all_months[-1]


# In[6]:


cloud_mask_ds = xr.open_dataset(path_to_cloud_mask)


# In[ ]:


def process_time_parallel(time_i):
    print(time_i)
    out_fpath = output_rel_dir_to_all_surface_spec + 'surface_fields_{year}{month:02d}.nc'.format(year = time_i.year,
                                                         month = time_i.month)
    if not os.path.exists(out_fpath):
        rel_dir = rel_dir_format.format(year = time_i.year, month = time_i.month)
        file_names = os.listdir(rel_dir_to_all_surface_spec + rel_dir)
        file_paths_i = [rel_dir_to_all_surface_spec + rel_dir + filename for filename in file_names]

        # get cloud-free mask for month
        cloud_mask_ds_i = cloud_mask_ds.sel(time = '{year}-{month:02d}'.format(year = time_i.year, 
                                                        month = time_i.month))
        # find only cloud-free times 
        cloud_free_ds = cloud_mask_ds_i.where(cloud_mask_ds_i['cloud_free_bool'] == 1, drop = True)

        # open surface spectrums for month 
        ds_i = xr.open_mfdataset(file_paths_i, combine='by_coords')

        # select and save only cloud_free times
        ds_i_cloud_free = ds_i.sel(time = cloud_free_ds.time)

        out_fpath = output_rel_dir_to_all_surface_spec + 'surface_fields_{year}{month:02d}.nc'.format(year = time_i.year,
                                                             month = time_i.month)

        ds_i_cloud_free.to_netcdf(out_fpath)


# In[ ]:


Parallel(n_jobs=20)(delayed(process_time_parallel)(time_i) for time_i in all_months)
print('DONE!')


# # Not parallel

# In[7]:


# for time_i in all_months:
#     print(time_i)
#     out_fpath = output_rel_dir_to_all_surface_spec + 'surface_fields_{year}{month:02d}.nc'.format(year = time_i.year,
#                                                          month = time_i.month)
#     if not os.path.exists(out_fpath):
#         rel_dir = rel_dir_format.format(year = time_i.year, month = time_i.month)
#         file_names = os.listdir(rel_dir_to_all_surface_spec + rel_dir)
#         file_paths_i = [rel_dir_to_all_surface_spec + rel_dir + filename for filename in file_names]

#         # get cloud-free mask for month
#         cloud_mask_ds_i = cloud_mask_ds.sel(time = '{year}-{month:02d}'.format(year = time_i.year, 
#                                                         month = time_i.month))
#         # find only cloud-free times 
#         cloud_free_ds = cloud_mask_ds_i.where(cloud_mask_ds_i['cloud_free_bool'] == 1, drop = True)

#         # open surface spectrums for month 
#         ds_i = xr.open_mfdataset(file_paths_i, combine='by_coords')

#         # select and save only cloud_free times
#         ds_i_cloud_free = ds_i.sel(time = cloud_free_ds.time)

#         out_fpath = output_rel_dir_to_all_surface_spec + 'surface_fields_{year}{month:02d}.nc'.format(year = time_i.year,
#                                                              month = time_i.month)

#         ds_i_cloud_free.to_netcdf(out_fpath)


# In[72]:





# In[75]:


# cloud_free_ds


# In[ ]:





# In[5]:


# cloud_mask_ds


# In[27]:


# plt.plot(cloud_mask_ds['cloud_free_bool'].values[250:500])


# In[28]:


# plt.plot(cloud_mask_ds['800_cm_mean_rad'].values[250:500])


# In[ ]:




