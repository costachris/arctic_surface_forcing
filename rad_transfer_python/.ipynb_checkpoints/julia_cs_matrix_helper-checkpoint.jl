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
using Flux


function get_cross_section_matrix(hitran, ν_grid, p_grid, t_grid)
    cs_matrix = zeros(length(p_grid), length(ν_grid)) |> gpu ;

    # Calculate all the cross-sections at the pressure and temperature grids
    @showprogress 1 "Computing Cross Sections for Interpolation..." for i in 1:length(p_grid)
            cs_matrix[i,:] = compute_absorption_cross_section(hitran, 
                                                                Voigt(), 
                                                                collect(ν_grid), 
                                                                p_grid[i], 
                                                                t_grid[i],
                                                                CEF=HumlicekWeidemann32SDErrorFunction(),
                                                                wing_cutoff = 100,
                                                                architecture=Architectures.GPU())
    end

    return cs_matrix
    
end



# function save_cs_matrix_to_netcdf(p_grid,ν_grid,out_file_path)
    
#     ds = Dataset(out_file_path,"c")
#     nu_dim = defDim(ds,"nu", size(ν_grid)[1])
#     pres_dim = defDim(ds,"pressure", size(p_grid)[1])



#     # Define the variables temperature with the attribute units
#     CO2_cs_matrix_var = defVar(ds,"cs_matrix_co2", Float32, ("pressure", "nu"))
#     H2O_cs_matrix_var = defVar(ds, "cs_matrix_h2o", Float32, ("pressure", "nu"))
#     CH4_cs_matrix_var = defVar(ds,"cs_matrix_ch4", Float32, ("pressure", "nu"))
#     # nu_coord = defVar(ds,"nu", Float64, ("nu"))
#     # pres_coord = defVar(ds,"pressure", Float64, ("pressure"))
#     # nu_coord = coord(ds,"nu", Float64, ("nu"))

#     CO2_cs_matrix_var[:,:] = CO2_cs_matrix
#     H2O_cs_matrix_var[:,:] = H2O_cs_matrix
#     CH4_cs_matrix_var[:,:] = CH4_cs_matrix

#     # nu_coord = collect(ν_grid)
#     # pres_coord = p_grid


# close(ds)
