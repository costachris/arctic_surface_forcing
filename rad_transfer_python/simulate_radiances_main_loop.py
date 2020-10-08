#!/usr/bin/env python
# coding: utf-8

# ## For running radiative transfer on saved cross section matrices. 
# 

# In[ ]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from glob import glob
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd

from matplotlib.font_manager import FontProperties
from scipy.interpolate import CubicSpline, BSpline

# %run simulate_radiances_utils.py
# %run locations.py
from simulate_radiances_utils import *
from radiance_loop_helper import *

font = {'weight' : 'bold',
        'size'   : 12}
label_fontdict = {'weight' : 'bold',
        'size'   : 12}
title_fontdict = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

from joblib import Parallel, delayed

# unit conversions 
W_m_mW_cm = 1e2*1e3


# In[2]:


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


# In[3]:


# path to profile timeseries over summit
profile_ts_file_path = "/net/fluo/data2/groupMembers/cchristo/profiles/summit_merra/summit_all.nc"

# path to julia-generation cross sections
input_cs_matrix_rel_path = '/net/fluo/data2/groupMembers/cchristo/cs_matrices/summit/julia_generated/'

# path formatting 
input_cs_matrix_path_format = input_cs_matrix_rel_path + '{year}/cs_matrix_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.nc'


# output data rel dir 
# output_data_rel_dir_top = '/net/fluo/data2/groupMembers/cchristo/results/rt_results/'
output_data_rel_dir_top = '/net/fluo/data2/groupMembers/cchristo/results/rt_results_preindus_CO2/'

out_data_path_format = '{year}/{month:02d}/surface_fields_{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}.nc'


# In[280]:


# all_times = pd.date_range(start = '2011-01-01 1:30:00',
#                           end = '2011-02-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-06-01 1:30:00',
#                           end = '2012-01-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-03-01 1:30:00',
#                           end = '2011-04-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-04-01 1:30:00',
#                           end = '2011-05-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-05-01 1:30:00',
#                           end = '2011-06-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-06-01 1:30:00',
#                           end = '2011-07-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-07-01 1:30:00',
#                           end = '2011-08-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-08-01 1:30:00',
#                           end = '2011-09-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-09-01 1:30:00',
#                           end = '2011-10-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-10-01 1:30:00',
#                           end = '2011-11-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-11-01 1:30:00',
#                           end = '2011-12-01 1:30:00',
#                           freq = '3H')

# all_times = pd.date_range(start = '2011-12-01 1:30:00',
#                           end = '2012-01-01 1:30:00',
#                           freq = '3H')


# # Set parameters 
# 

# In[4]:


# CO2_mr = 395.e-6
CO2_mr = 280.e-6
CH4_mr = 1.89e-6
AMF = 1.0


# ## Open summit profilesm

# In[5]:


# %%time
# all_profiles = xr.open_dataset(profile_ts_file_path)


# In[194]:


# all_profiles


# ### Feldman et al. looked at 2011 - 2015 over summit

# In[6]:


# all_times = pd.date_range(start = '2011-01-01 1:30:00',
#                           end = '2016-01-01 1:30:00',
#                           freq = '3H')

all_times = pd.date_range(start = '2011-01-01 1:30:00',
                          end = '2016-01-01 1:30:00',
                          freq = '3H')


# In[13]:


# for time_i in all_times:
#     print(time_i)
#     output_data_rel_dir = output_data_rel_dir_top + '{year}/{month:02d}/'.format(year = time_i.year, month = time_i.month)
    
#     output_path = output_data_rel_dir_top + out_data_path_format.format(year = time_i.year, 
#                                                      month = time_i.month, 
#                                                      day = time_i.day,
#                                                      hour = time_i.hour, 
#                                                      minute = time_i.minute)
#     if not os.path.exists(output_data_rel_dir):
#         os.makedirs(output_data_rel_dir)
# #         skip_bool = False
#     if os.path.exists(output_path):
#         skip_bool = True
#     else: 
#         skip_bool = False
# #     else:
# #         skip_bool = True
    
#     if not skip_bool:
#         # profile properties
#         prof_i_ds = all_profiles.sel(time = time_i)
#         p_prof, T_prof, dz_prof,  vmr_h2o_prof, VCD_dry_prof, rho_N_h2o_prof, rho_N_prof = compute_profile_properties_merra2(prof_i_ds, 
#                                                                                                                              verbose=False)

#         # open cs_matrix 
#         input_cs_matrix_path = input_cs_matrix_path_format.format(year = time_i.year, 
#                                                                      month = time_i.month, 
#                                                                      day = time_i.day,
#                                                                      hour = time_i.hour, 
#                                                                      minute = time_i.minute)
#         cs_matrix_ds = xr.open_dataset(input_cs_matrix_path)
#         cs_matrix_co2 = cs_matrix_ds['cs_matrix_co2'].values
#         cs_matrix_ch4 = cs_matrix_ds['cs_matrix_ch4'].values
#         cs_matrix_h2o = cs_matrix_ds['cs_matrix_h2o'].values

#         # nu = cs_matrix_ds['nu'].values
#         nu_ = np.linspace(400,2099.99, 170000)
#         NLEV = len(cs_matrix_ds['pressure'])

#         surface_spectrums = compute_downwelling_radiation(cs_matrix_co2,
#                                       cs_matrix_h2o,
#                                       cs_matrix_ch4,
#                                       T_prof,
#                                       VCD_dry_prof, 
#                                       vmr_h2o_prof,
#                                       nu_,
#                                       CO2_mr = CO2_mr, 
#                                       CH4_mr = CH4_mr,
#                                       AMF=AMF)
#         Surface_Down_CO2, Surface_Down_CH4, Surface_Down_H2O, Surface_Down= surface_spectrums

#         # compute emission height and dT/dz
#         tau_co2 = cs_matrix_co2*VCD_dry_prof*CO2_mr*AMF 
#         tau_ch4 = cs_matrix_ch4*VCD_dry_prof*CH4_mr*AMF 
#         tau_h2o = cs_matrix_h2o*VCD_dry_prof*vmr_h2o_prof*AMF 

#         # interpolate temperature profile to use for dT/dz plot
#         z_prof_0, interp_prof = interpolate_T_prof(T_prof, dz_prof, num_vertical_points = 3000)


#         tau_wl_co2, dT_dz_co2 =  calc_emission_and_dT_dz(tau_matrix = tau_co2[:,::-1].copy(),
#                                              T_interpolator = interp_prof,
#                                              z_prof_0 = z_prof_0)

#         tau_wl_ch4, dT_dz_ch4 =  calc_emission_and_dT_dz(tau_matrix = tau_ch4[:,::-1].copy(),
#                                              T_interpolator = interp_prof,
#                                              z_prof_0 = z_prof_0)

#         tau_wl_h2o, dT_dz_h2o =  calc_emission_and_dT_dz(tau_matrix = tau_h2o[:,::-1].copy(),
#                                              T_interpolator = interp_prof,
#                                              z_prof_0 = z_prof_0)

#         ds_out = xr.Dataset()
#         ds_out['lw_down_CO2']  = _create_dataarray(Surface_Down_CO2, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['lw_down_CH4']  = _create_dataarray(Surface_Down_CH4, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['lw_down_H2O']  = _create_dataarray(Surface_Down_H2O, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['lw_down_total']  = _create_dataarray(Surface_Down, 
#                           nu = nu_, 
#                           time_val = time_i)

#         ds_out['eh_CO2']  = _create_dataarray(tau_wl_co2, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['eh_CH4']  = _create_dataarray(tau_wl_ch4, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['eh_H2O']  = _create_dataarray(tau_wl_h2o, 
#                           nu = nu_, 
#                           time_val = time_i)

#         ds_out['dT_CO2']  = _create_dataarray(dT_dz_co2, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['dT_CH4']  = _create_dataarray(dT_dz_ch4, 
#                           nu = nu_, 
#                           time_val = time_i)
#         ds_out['dT_H2O']  = _create_dataarray(dT_dz_h2o, 
#                           nu = nu_, 
#                           time_val = time_i)

#         ds_out.to_netcdf(output_path)


# In[181]:


# %%time
# %run simulate_radiances_utils.py
# compute tau matricies 


# In[ ]:





# In[14]:


def process_time_parallel(time_i):
    try:
        output_data_rel_dir = output_data_rel_dir_top + '{year}/{month:02d}/'.format(year = time_i.year, month = time_i.month)

        output_path = output_data_rel_dir_top + out_data_path_format.format(year = time_i.year, 
                                                         month = time_i.month, 
                                                         day = time_i.day,
                                                         hour = time_i.hour, 
                                                         minute = time_i.minute)
        if not os.path.exists(output_data_rel_dir):
            os.makedirs(output_data_rel_dir, exist_ok=True)
    #         skip_bool = False
        if os.path.exists(output_path):
            skip_bool = True
        else: 
            skip_bool = False
    #     else:
    #         skip_bool = True

        if not skip_bool:
            # profile properties
            all_profiles = xr.open_dataset(profile_ts_file_path)
            prof_i_ds = all_profiles.sel(time = time_i)
            p_prof, T_prof, dz_prof,  vmr_h2o_prof, VCD_dry_prof, rho_N_h2o_prof, rho_N_prof = compute_profile_properties_merra2(prof_i_ds, 
                                                                                                                                 verbose=False)

            # open cs_matrix 
            input_cs_matrix_path = input_cs_matrix_path_format.format(year = time_i.year, 
                                                                         month = time_i.month, 
                                                                         day = time_i.day,
                                                                         hour = time_i.hour, 
                                                                         minute = time_i.minute)
            cs_matrix_ds = xr.open_dataset(input_cs_matrix_path)
            cs_matrix_co2 = cs_matrix_ds['cs_matrix_co2'].values
            cs_matrix_ch4 = cs_matrix_ds['cs_matrix_ch4'].values
            cs_matrix_h2o = cs_matrix_ds['cs_matrix_h2o'].values

            # nu = cs_matrix_ds['nu'].values
            nu_ = np.linspace(400,2099.99, 170000)
            NLEV = len(cs_matrix_ds['pressure'])

            surface_spectrums = compute_downwelling_radiation(cs_matrix_co2,
                                          cs_matrix_h2o,
                                          cs_matrix_ch4,
                                          T_prof,
                                          VCD_dry_prof, 
                                          vmr_h2o_prof,
                                          nu_,
                                          CO2_mr = CO2_mr, 
                                          CH4_mr = CH4_mr,
                                          AMF=AMF)
            Surface_Down_CO2, Surface_Down_CH4, Surface_Down_H2O, Surface_Down= surface_spectrums

            # compute emission height and dT/dz
            tau_co2 = cs_matrix_co2*VCD_dry_prof*CO2_mr*AMF 
            tau_ch4 = cs_matrix_ch4*VCD_dry_prof*CH4_mr*AMF 
            tau_h2o = cs_matrix_h2o*VCD_dry_prof*vmr_h2o_prof*AMF 

            # interpolate temperature profile to use for dT/dz plot
            z_prof_0, interp_prof = interpolate_T_prof(T_prof, dz_prof, num_vertical_points = 3000)


            tau_wl_co2, dT_dz_co2 =  calc_emission_and_dT_dz(tau_matrix = tau_co2[:,::-1].copy(),
                                                 T_interpolator = interp_prof,
                                                 z_prof_0 = z_prof_0)

            tau_wl_ch4, dT_dz_ch4 =  calc_emission_and_dT_dz(tau_matrix = tau_ch4[:,::-1].copy(),
                                                 T_interpolator = interp_prof,
                                                 z_prof_0 = z_prof_0)

            tau_wl_h2o, dT_dz_h2o =  calc_emission_and_dT_dz(tau_matrix = tau_h2o[:,::-1].copy(),
                                                 T_interpolator = interp_prof,
                                                 z_prof_0 = z_prof_0)

            ds_out = xr.Dataset()
            ds_out['lw_down_CO2']  = _create_dataarray(Surface_Down_CO2, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['lw_down_CH4']  = _create_dataarray(Surface_Down_CH4, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['lw_down_H2O']  = _create_dataarray(Surface_Down_H2O, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['lw_down_total']  = _create_dataarray(Surface_Down, 
                              nu = nu_, 
                              time_val = time_i)

            ds_out['eh_CO2']  = _create_dataarray(tau_wl_co2, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['eh_CH4']  = _create_dataarray(tau_wl_ch4, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['eh_H2O']  = _create_dataarray(tau_wl_h2o, 
                              nu = nu_, 
                              time_val = time_i)

            ds_out['dT_CO2']  = _create_dataarray(dT_dz_co2, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['dT_CH4']  = _create_dataarray(dT_dz_ch4, 
                              nu = nu_, 
                              time_val = time_i)
            ds_out['dT_H2O']  = _create_dataarray(dT_dz_h2o, 
                              nu = nu_, 
                              time_val = time_i)

            ds_out.to_netcdf(output_path)
    except Exception as e:
        print('Failed on ', str(time_i), e)
            


# In[15]:


Parallel(n_jobs=25)(delayed(process_time_parallel)(time_i) for time_i in all_times)


# # sanity checks

# In[161]:


plt.plot(1e3*Surface_Down)
# plt.xlim([0,25000])


# In[154]:


wl_nu = 1.e7/nu_*1.e-9
wavenum_m = nu_*1e2


# In[117]:


# wavenum_m


# In[164]:


plt.figure(figsize = (10,7))
plt.plot(nu_, 1e3*Surface_Down_CO2,label='Rdown CO2', alpha=0.7 ,linewidth = 0.5)
plt.plot(nu_, 1e3*Surface_Down_CH4,label='Rdown CH4', alpha=0.7 ,linewidth = 0.5)
plt.plot(nu_, 1e3*Surface_Down_H2O,label='Rdown H2O', alpha=0.7 ,linewidth = 0.5)

# plt.plot(wl_nu*1e6,Surface_Down_dH2O,label='downwelling radiance at surface (double H$_2$O', alpha=0.7)#, wl_nu*1e6, np.sum(R,axis=1), wl_nu*1e6,R_surf)

# plt.plot(nu_, W_m_mW_cm*planck_wavenumber(wavenum_m,300),label='BB @ 300K',alpha=0.63)
plt.plot(nu_, W_m_mW_cm*planck_wavenumber(wavenum_m,270),label='BB @ 270K',alpha=0.63)
plt.plot(nu_, W_m_mW_cm*planck_wavenumber(wavenum_m,260),label='BB @ 260K',alpha=0.63)
plt.plot(nu_, W_m_mW_cm*planck_wavenumber(wavenum_m,250),label='BB @ 250K',alpha=0.63)
plt.plot(nu_, W_m_mW_cm*planck_wavenumber(wavenum_m,240),label='BB @ 240K',alpha=0.63)
plt.plot(nu_, W_m_mW_cm*planck_wavenumber(wavenum_m,220),label='BB @ 220K',alpha=0.63)
plt.legend(loc=0)

# plt.xlim((491,1799))
plt.xlim((500, 1800))
plt.xlabel('Wavenumber ($cm^{-1}$)')
plt.ylabel(r'Downwelling Radiance ($mW m^{-2} sr^{-1} cm^{-1}$)')
# plt.xlim((4,30))
plt.title('Downwelling Thermal Radiance in Nadir at surface')
plt.grid()
# plt.savefig('figs/christian_update_9_14/Rdown_gas_components_zoom.png', dpi = 300)


# In[199]:


plt.plot(nu_, tau_wl_ch4)
# plt.xlim([])
# plt.xlim([640, 800])
# plt.ylim([0, 3000])


# In[183]:


plt.title(r'Summit Greenland $\frac{dT}{dz}$ @ $\tau = 1$')
plt.plot(nu_, dT_dz_co2, linewidth = 0.5)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlim([550, 750])
plt.grid()


# In[ ]:





# In[ ]:





# In[77]:


plt.imshow(cs_matrix_h2o, vmax = 1e-19)
plt.gca().set_aspect(0.0002)
plt.colorbar()

