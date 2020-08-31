using Pkg
Pkg.activate("rad_transfer"; shared=true)
using RadiativeTransfer
using RadiativeTransfer.CrossSection




function get_cross_section_matrix(hitran, ν_grid, p_grid, t_grid)
    cs_matrix = zeros(length(p_grid), length(ν_grid));

    # Calculate all the cross-sections at the pressure and temperature grids
    @showprogress 1 "Computing Cross Sections for Interpolation..." for i in 1:length(p_grid)
            cs_matrix[i,:] = compute_absorption_cross_section(hitran, 
                                                                Voigt(), 
                                                                collect(ν_grid), 
                                                                p_grid[i], 
                                                                t_grid[i])
    end

    return cs_matrix
    
end
