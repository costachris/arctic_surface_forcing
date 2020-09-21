using Pkg
Pkg.activate("rad_transfer"; shared=true)
using RadiativeTransfer
using RadiativeTransfer.CrossSection
using KernelAbstractions
using ..Architectures
using Pkg.Artifacts
using ProgressMeter
using DataStructures
using PyPlot
using CUDA
using Flux
using NetCDF
using NCDatasets
using Dates

include("julia_cs_matrix_helper.jl")

profile_ts_file_path = "/net/fluo/data2/groupMembers/cchristo/profiles/summit_merra/summit_all.nc"
rel_path_name = "/net/fluo/data2/groupMembers/cchristo/cs_matrices/summit/julia_generated/"

# open hitran tables
hitran_data_CO2 = read_hitran("hitran_molec_id_2_CO2.tar", mol=2, iso=1, ν_min=400, ν_max=2100)
hitran_data_H2O = read_hitran("hitran_molec_id_1_H2O.tar", mol=1, iso=1, ν_min=400, ν_max=2100)
hitran_data_CH4 = read_hitran("hitran_molec_id_6_CH4.tar", mol=6, iso=1, ν_min=400, ν_max=2100)

# open profile timeseries
ds = Dataset(profile_ts_file_path) ;




for time_i in 1:length(ds["time"])
    
    T = ds["T"][:,time_i]
    P = ds["PL"][:,time_i]
    time_ii = ds["time"][time_i]
    
    display(time_ii)
    # (lon, lat, z, time)
    var_array_sizes = size(T)
    
    # define grids 
    ν_grid = 400:0.01:2099.99
    p_grid = P[:] / 100
    t_grid = T[:]
    
    # compute cs matrix
    CO2_cs_matrix_cuda = get_cross_section_matrix(hitran_data_CO2, ν_grid, p_grid, t_grid)
    H2O_cs_matrix_cuda = get_cross_section_matrix(hitran_data_H2O, ν_grid, p_grid, t_grid)
    CH4_cs_matrix_cuda = get_cross_section_matrix(hitran_data_CH4, ν_grid, p_grid, t_grid)
    
    # convert to regular array
    CO2_cs_matrix = convert(Array, CO2_cs_matrix_cuda)
    H2O_cs_matrix = convert(Array, H2O_cs_matrix_cuda)
    CH4_cs_matrix = convert(Array, CH4_cs_matrix_cuda)
    
    
    fname = string(year(time_ii),"/cs_matrix_",year(time_ii), lpad(month(time_ii),2, "0"), lpad(day(time_ii),2,"0"), 
    "_", lpad(hour(time_ii),2,"0"), lpad(minute(time_ii),2,"0"), ".nc")


    out_file_path = string(rel_path_name, fname)


    ds_out = Dataset(out_file_path,"c")

    nu_dim = defDim(ds_out,"nu", size(ν_grid)[1])
    pres_dim = defDim(ds_out,"pressure", size(p_grid)[1])




    # Define the variables temperature with the attribute units
    CO2_cs_matrix_var = defVar(ds_out,"cs_matrix_co2", Float32, ("pressure", "nu"))
    H2O_cs_matrix_var = defVar(ds_out, "cs_matrix_h2o", Float32, ("pressure", "nu"))
    CH4_cs_matrix_var = defVar(ds_out,"cs_matrix_ch4", Float32, ("pressure", "nu"))


    CO2_cs_matrix_var[:,:] = CO2_cs_matrix
    H2O_cs_matrix_var[:,:] = H2O_cs_matrix
    CH4_cs_matrix_var[:,:] = CH4_cs_matrix



    close(ds_out)

end