#!/usr/bin/env python
# coding: utf-8

# In[69]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import requests
import pandas as pd


# In[ ]:


dates = pd.date_range('2015-01-01', '2015-01-07', freq = 'D')


# In[98]:


# output_data_dir = '/export/data1/cchristo/merra2/merra2_pres_levels/'
output_data_dir = '/home/cchristo/proj_christian/data/reanalysis_3d/merra2/'

# string formats for saving locally
output_rel_subpath_format = '{year}/{month:02d}/'
output_file_path_format = output_data_dir + output_rel_subpath_format + 'MERRA2_400.tavg3_3d_asm_Nv.{year}{month:02d}{day:02d}.nc4'
output_rel_path_format = output_data_dir + output_rel_subpath_format

# string format for MERRA2 data requests
input_url_format = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T3NVASM.5.12.4/{year}/{month:02d}/MERRA2_400.tavg3_3d_asm_Nv.{year}{month:02d}{day:02d}.nc4'


# In[102]:


for date_i in dates:
    print('Downloading:  ', date_i)
    URL = input_url_format.format(year = date_i.year, month = date_i.month, day = date_i.day)
    output_rel_path = output_rel_path_format.format(year = date_i.year, month = date_i.month, day = date_i.day)
    output_file_path = output_file_path_format.format(year = date_i.year, month = date_i.month, day = date_i.day)
    try:
#       if file aleady exists move on
        if not os.path.exists(output_file_path):
            wget_string = 'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --auth-no-challenge=on --keep-session-cookies --content-disposition ' + URL + ' --directory-prefix ' + output_rel_path       
            os.system(wget_string)
    except:
        print('Could not download ', date_i)

