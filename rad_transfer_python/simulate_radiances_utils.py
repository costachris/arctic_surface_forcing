import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import os
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from hapi import *


from matplotlib.font_manager import FontProperties

### Plot settings
font = {'weight' : 'bold',
        'size'   : 12}
label_fontdict = {'weight' : 'bold',
        'size'   : 12}
title_fontdict = {'weight' : 'bold',
        'size'   : 12}

matplotlib.rc('font', **font)

##### Define constants

go = 9.8196 #(m/s**2) 
Rd = 287.04 # specific gas constant for dry air
R_universal = 8.314472
Na = 6.0221415e23

# Some constants and the planck function (as radiance!)
pi = np.pi
h = 6.62607004e-34 # m^2/kg/s
c = 299792458 # m/s
k = 1.380649e-23 # J/K

###### Helper function for fundamental equations
def planck(wav, T):
    c1 = 2.0*h*c**2
    c2 = h*c/(wav*k*T)
    intensity = c1/ ( (wav**5)*(np.exp(c2) - 1.0) )
    # Convert to W/sr/m^2/µm here directly (it was W/sr/m^2/m)
    return intensity*1.e-6
#     return intensity*1.e-2

def planck_wavenumber(wavenum, T):
    c1 = 2.0*h*(c**2)*(wavenum**3)
    c2 = (h*c*wavenum)/(k*T)
    intensity = c1/(np.exp(c2) - 1.0)
    return intensity
#     return (c1, c2, intensity)
###### Helper function for calculating profile properties
def compute_profile_properties_merra2(ds, verbose=True):
    ''' Given single profile from merra2 meteorlogical reanalysis, compute pressure levels, VMR 
    for water vapor. Profile should contain variables PS, PL, QV, T, and DELP'''
    # Surface pressure at location
    ps_local = ds['PS'].values
    p_local = ds['PL'].values
    # q and T profiles at location
    q_local = ds['QV'].values
    T_local = ds['T'].values

    NLEV = len(T_local)

    dz = np.divide(ds['DELP'].values,ds['PL'].values)*(Rd*T_local*(1+0.608*q_local))/go
    rho_N =  ds['PL'].values*(1-q_local*1.6068)/(R_universal*T_local)*Na/10000.0
    rho_N_h2o =  ds['PL'].values*(q_local*1.6068)/(R_universal*T_local)*Na/10000.0
    vmr_h2o = q_local*1.6068

    if verbose:
        print('Total column density of dry air: ' +str(np.sum(dz*rho_N))+' molec/cm^2')
        print('Total column density of water vapor: ' + str(np.sum(dz*rho_N_h2o))+' molec/cm^2')
    VCD_dry = dz*rho_N
    
    return(p_local, T_local, dz,  vmr_h2o, VCD_dry, rho_N_h2o, rho_N)



def create_cross_section_matrix_hapi(p_prof, T_prof, xmin, xmax, time_i=None, output_path=None):
    '''Given temperature/pressure profile, create cross-section matrix (w/ option to save)
    Args:
        output_path - str
            If not None, save cs matrices as netcdf to specified path.
    
    Returns:
    cs_matrix - xr.Dataset [number of levels, number of wavelengths]
    '''
    nu_, cs_co2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[xmin,xmax],Environment={'p':1,'T':270},IntensityThreshold=1e-27)
    
    NLEV = len(p_prof)
    
    cs_matrix_co2 = np.zeros((len(nu_),NLEV))
    cs_matrix_ch4 = np.zeros((len(nu_),NLEV))
    cs_matrix_h2o = np.zeros((len(nu_),NLEV))
    

    # Loop over each layer 
    for i in range(NLEV):
        print(str(i)+'/'+str(NLEV), end='\r')
        p_ = p_prof[i]/101325
    #     print(p_)”
        T_ = T_prof[i]
        nu_, cs_co2 = absorptionCoefficient_Voigt(SourceTables='CO2_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
        nu_, cs_ch4 = absorptionCoefficient_Voigt(SourceTables='CH4_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
        nu_, cs_h2o = absorptionCoefficient_Voigt(SourceTables='H2O_S', WavenumberRange=[xmin,xmax],Environment={'p':p_,'T':T_},IntensityThreshold=1e-27)
        cs_matrix_co2[:,i] = cs_co2
        cs_matrix_ch4[:,i] = cs_ch4
        cs_matrix_h2o[:,i] = cs_h2o
        
        
    cs_matrix_ds = xr.Dataset()
    cs_matrix_co2_da = xr.DataArray(cs_matrix_co2, coords = [nu_, p_prof], dims = ['nu','pressure'])
    cs_matrix_ch4_da = xr.DataArray(cs_matrix_ch4, coords = [nu_, p_prof], dims = ['nu','pressure'])
    cs_matrix_h2o_da = xr.DataArray(cs_matrix_h2o, coords = [nu_, p_prof], dims = ['nu','pressure'])


    cs_matrix_ds['cs_matrix_co2'] = cs_matrix_co2_da
    cs_matrix_ds['cs_matrix_ch4'] = cs_matrix_ch4_da
    cs_matrix_ds['cs_matrix_h2o'] = cs_matrix_h2o_da
    if not (time_i is None):
        cs_matrix_ds['time'] = time_i
    cs_matrix_ds = cs_matrix_ds.assign_coords(time = cs_matrix_ds['time'])

    if output_path:
        cs_matrix_ds.to_netcdf(output_path)
    return cs_matrix_ds

######## Functions for performing RT calculations


def compute_downwelling_radiation(cs_matrix_co2,
                                  cs_matrix_h2o,
                                  cs_matrix_ch4,
                                  T_prof,
                                  VCD_dry_prof, 
                                  vmr_h2o_prof,
                                  nu,
                                  CO2_mr = 400.e-6, 
                                  CH4_mr = 1.8e-6,
                                  AMF=1):
    '''Compute downwelling radiation from an atmosphere containing 
    3 greenhouse gasses (CO2, CH4, and water vapor).
    
    CO2 and CH4 are assumed to be well-mixed, whereas the vmr of water vapor
    can vary. 
    
    '''
    NLEV = cs_matrix_co2.shape[1]

    # Generate matrices of optical thickness per layer now for each gas: 
    tau_co2 = cs_matrix_co2*VCD_dry_prof*CO2_mr*AMF 
    tau_h2o = cs_matrix_h2o*VCD_dry_prof*vmr_h2o_prof*AMF 
    tau_ch4 = cs_matrix_ch4*VCD_dry_prof*CH4_mr*AMF 
    
    # total transmission
    T = np.exp(-tau_co2)*np.exp(-tau_h2o)*np.exp(-tau_ch4)
    
    # component-by-component transmission 
    T_CO2 = np.exp(-tau_co2)
    T_H2O = np.exp(-tau_h2o)
    T_CH4 = np.exp(-tau_ch4)
    
    # Generate Planck curve per layer + surface:
    wl_nu = 1.e7/nu*1.e-9
    wavenum_m = nu*1e2
    # Use skin temperature of 300K
#     B = np.zeros((len(nu_),NLEV))

    B = np.zeros(T.shape)
    for i in range(NLEV):
        B[:,i] = planck_wavenumber(wavenum_m,T_prof[i])*1e2
    
    # compute downwelling IR radiation 
    Rdown = np.zeros(cs_matrix_co2.shape)
    Rdown_CO2 = np.empty_like(Rdown)
    Rdown_CH4 = np.empty_like(Rdown)
    Rdown_H2O = np.empty_like(Rdown)



    for i in range(NLEV):
        Rdown[:,i] = B[:,i]*(1-T[:,i])*np.prod(T[:,i+1:],axis=1)
        # component-by-component
        Rdown_CO2[:,i] = B[:,i]*(1-T_CO2[:,i])*np.prod(T_CO2[:,i+1:],axis=1)
        Rdown_CH4[:,i] = B[:,i]*(1-T_CH4[:,i])*np.prod(T_CH4[:,i+1:],axis=1)
        Rdown_H2O[:,i] = B[:,i]*(1-T_H2O[:,i])*np.prod(T_H2O[:,i+1:],axis=1)
    
    Surface_Down = np.sum(Rdown,axis=1)

    Surface_Down_CO2 = np.sum(Rdown_CO2,axis=1)
    Surface_Down_CH4 = np.sum(Rdown_CH4,axis=1)
    Surface_Down_H2O = np.sum(Rdown_H2O,axis=1)
        
    return (Surface_Down_CO2, Surface_Down_CH4, Surface_Down_H2O, Surface_Down)

####### Plotting functions

def plot_profile(v_coord, temp,
                 v_coord_type = 'pressure',
                 plot_kind = 'line',
                 min_pres = 10, xlabel = "Temperature [C]", newfig_bool = True,
                 xlim = None,
                 label = None, rotation = 0):
    '''Given xr.dataset of single profile, plot vertical profile w/ log(p)
    
    Args
    -------
        v_coord - array-like
            vertical coordinate
        v_coord_type - str {'pressure', 'height'}
            Use pressure or height as vertical coordinate
        plot_kind - str {'line', 'scatter'}
            Kind of plot to use.
    
    Returns
    --------
        plt.axis
    
    '''

    
    if newfig_bool:
        plt.figure(figsize = (6,6))
    if plot_kind == 'line':
        plt.plot(temp, v_coord, linewidth = 2, label = label)
    elif plot_kind == 'scatter':
        plt.scatter(temp, v_coord, label = label)
    
    if v_coord_type == 'pressure':
        plt.gca().invert_yaxis()
        plt.ylim([np.nanmax(v_coord), min_pres])
        plt.gca().set_yscale('log')
        plt.ylabel("Pressure [Pa]")
    elif v_coord_type == 'height':
        plt.ylim([np.nanmin(v_coord), np.nanmax(v_coord)])
        plt.ylabel("Height [m]")
        
            
    plt.grid()
    plt.xlabel(xlabel)
   
    if xlim: 
        plt.xlim(xlim)
    if rotation != 0: 
        plt.xticks(rotation=rotation)
#     plt.locator_params(nbins=8)
#     plt.yticks(np.arange(min_pres, pres.max(), 100.0))

    return plt.gca()


def plot_emission_height(wl_nm, tau_wl, T_prof, p_prof, label, 
                         ylim = None , 
                         xlim = [5,30],
                         ave_emmission_pres = None):
#     wl_nm = wl_nu*1e6

    plt.figure(figsize = (15,5))
    ax0 = plt.subplot(121)
    plt.plot(wl_nm, tau_wl, label = label)
    
    plt.grid()
#     plt.gca().set_yscale('log')
#     plt.gca().invert_yaxis()

    plt.xlabel(r'Wavenumber $[cm^{-1}]$')
    plt.ylabel(r'$\tau = 1$ Height [m]')
    plt.legend()
    if ylim:
#         plt.ylim([p_full.max(), 4*10**4])
        plt.ylim(ylim)
#     plt.xlim((12,18))
    if xlim:
        plt.xlim(xlim)
    else:
        plt.xlim([wl_nm.min(), wl_nm.max()])
    plt.subplot(122) #, sharey = ax0)

    plt.plot(T_prof, p_prof, '.-')
#     plt.gca().set_yscale('log')
#     plt.gca().invert_yaxis()
    if ave_emmission_pres:
        plt.axhline(y = ave_emmission_pres, color = 'r', linestyle = '--')
    if ylim:
        plt.ylim(ylim)        
    plt.grid()