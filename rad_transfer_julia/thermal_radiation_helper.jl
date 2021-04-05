using Pkg.Artifacts
using NCDatasets
using ProgressMeter
using Plots
using ImageFiltering
using Distributions
using Interpolations
using Polynomials

using RadiativeTransfer.CrossSection
using Parameters
using CSV
using DelimitedFiles
using ForwardDiff, DiffResults

using NumericalIntegration

# using RadiativeTransfer
# using RadiativeTransfer.Architectures


# function load_interpolation_model(filepath::String)
#     itp_model::InterpolationModel = @load filepath itp_model
#     return itp_model
# end


function pressure_filter(p_array::Array, t_array::Array, p_threshold::Real)
    p_thres_inds = findall(p_array -> (p_array > p_threshold), p_array)
    return p_array[p_thres_inds], t_array[p_thres_inds] 
end


## existing 

abstract type InstrumentOperator end

"Struct for an atmospheric profile"
struct AtmosphericProfile{FT}
    lat::FT
    lon::FT
    psurf::FT
    T::Array{FT,1}
    q::Array{FT,1}
    p::Array{FT,1}
    p_levels::Array{FT,1}
    vmr_h2o::Array{FT,1}
    vcd_dry::Array{FT,1}
    vcd_h2o::Array{FT,1}
end;

"Struct for Kernel Instrument Function"
@with_kw struct KernelInstrument{FT} <: InstrumentOperator 
    kernel::FT
    ν_out::Array
end;


function planck(wavenum, T)
    wavenum_m = wavenum*1.0e2
    h = 6.62607004e-34 # m^2/kg/s
    c = 299792458 # m/s
    k = 1.380649e-23 # J/K
    c1 = 2.0*h*(c^2)*(wavenum_m^3)
    c2 = (h*c*wavenum_m)/(k*T)
    intensity = c1/(exp(c2) - 1.0)
    return intensity*1.0e2
end



function loop_over_array(file, 
                         lat_array, 
                         lon_array,
                         timeIndex,
                         hitran_array_interp)
    
    global profile_filt
    profile_filt = zeros(1,2); 
    rf_down_array = zeros(size(lat_array)[1], size(lon_array)[1])
    display(size(rf_down_array))
    for lat_ii = 1:length(lat_array)
        for lon_ii = 1:length(lon_array)
            
            lat = lat_array[lat_ii]
            lon = lon_array[lon_ii]
            
            display(lat)
            
            profile = read_atmos_profile_new(file, lat, lon, timeIndex);

            p_array = profile.p
            p_filt_inds = findall(p_array -> (p_array > 15000), p_array)

            profile_filt = AtmosphericProfile(profile.lat, 
                                              profile.lon, 
                                              profile.psurf, 
                                              profile.T[p_filt_inds], 
                                              profile.q[p_filt_inds], 
                                              profile.p[p_filt_inds], 
                                              profile.p_levels[p_filt_inds], 
                                              profile.vmr_h2o[p_filt_inds], 
                                              profile.vcd_dry[p_filt_inds], 
                                              profile.vcd_h2o[p_filt_inds])


            # Minimum wavenumber
            ν_min  = 491.0
            # Maximum wavenumber
            ν_max = 1799.0

            # define grid
            res = 0.01
            ν = ν_min:res:ν_max

            σ_matrix = compute_profile_crossSections_interp(profile_filt, hitran_array_interp , ν);


            # Define concentration profile:
            nL = length(profile_filt.T)
            vmr_co2 = zeros(nL) .+ 400e-6
            vmr_co2_pre = zeros(nL) .+ 270e-6
            vmr_ch4 = zeros(nL) .+ 2e-6
            vmr_h2o = profile_filt.vmr_h2o
            vmrs_pre = [vmr_co2_pre, vmr_h2o, vmr_ch4 ];

            vmrs = [vmr_co2, vmr_h2o, vmr_ch4 ];


            σ_matrix_red_cu = CuArray(σ_matrix);

            T, rad_down_pre, rad_up = forward_model_thermal_gpu(vmrs_pre,
                                                             σ_matrix_red_cu,
                                                             profile_filt,
                                                             0.05, 280.0, 1.0, ν);

            T, rad_down_today, rad_up = forward_model_thermal_gpu(vmrs,
                                                             σ_matrix_red_cu,
                                                             profile_filt,
                                                             0.05, 280.0, 1.0, ν);


            diff = Array((rad_down_today[:,end] .- rad_down_pre[:,end]));
            sus = NumericalIntegration.integrate(ν, diff) * 3.14159265358
            display(sus)
            rf_down_array[lat_ii, lon_ii] = sus[1]
            
            
        end
    end
    
    return rf_down_array
end


function compute_profile_properties_merra(T::Array, 
                                          q::Array,
                                          plocal::Array, 
                                          psurf::Array, 
                                          delp::Array)
   """Given profile, compute profile properties and return AtmosphericProfile object"""
    Na = 6.0221415e23;
    # Dry and wet mass
    dryMass = 28.9647e-3  / Na  # in kg/molec, weighted average for N2 and O2
    wetMass = 18.01528e-3 / Na  # just H2O
    ratio = dryMass / wetMass 
    
    Rd = 287.04 # specific gas constant for dry air
    R_universal = 8.314472
    go = 9.8196 #(m/s**2) 
    
    n_layers = length(T)
    # also get a VMR vector of H2O (volumetric!)
    vmr_h2o = zeros(FT, n_layers, )
    vcd_dry = zeros(FT, n_layers, )
    vcd_h2o = zeros(FT, n_layers, )
    #################################################################
    
    dz = (delp./plocal).*(Rd.*T.*(1 .+ 0.608*q))./go
    
    rho_N = (plocal .* (1 .- q .* ratio)) ./ (R_universal .* T) .* (Na/10000.0)
    rho_N_h2o = (plocal .* (q .* ratio)) ./ (R_universal .* T) .* (Na/10000.0)
    
    vmr_h2o = q .* ratio
    vcd_dry = dz .* rho_N
    
    # Total column density of water vapor:
#     display(sum(dz.*rho_N_h2o))
    
    return AtmosphericProfile(NaN, NaN, psurf, T, q, plocal, plocal, vmr_h2o, vcd_dry, vcd_h2o)
end;
    

function subset_variables_merra(ds::NCDataset, 
                                lat::Real,
                                lon::Real,
                                timeIndex)
    
    
    # get nearest lat/lon index
    lat_   = ds["lat"][:]
    lon_   = ds["lon"][:]
    
    FT = eltype(lat_)
    lat = FT(lat)
    lon = FT(lon)
    
    # Find index (nearest neighbor, one could envision interpolation in space and time!):
    iLat = argmin(abs.(lat_ .- lat))
    iLon = argmin(abs.(lon_ .- lon))
    
    # Temperature profile
    T    = convert(Array, ds["T"][:,:,:, timeIndex])
    # specific humidity profile
    q    = convert(Array, ds["QV"][:,:,:, timeIndex])

    # 3d pressure
    plocal = convert(Array, ds["PL"][:,:,:, timeIndex])
    
    # delta pressure 
    delp = convert(Array, ds["DELP"][:,:,:, timeIndex])
    
    # Surafce pressure
    psurf = convert(Array, ds["PS"][:,:,timeIndex])
    
    return (T, q, plocal, delp, psurf)
end


function read_atmos_profile_new(file::String, lat::Real, lon::Real, timeIndex; g₀=9.8196)
    @assert 1 <= timeIndex <= 4
    


    ds = Dataset(file)

    # See how easy it is to actually extract data? Note the [:] in the end reads in ALL the data in one step
#     lat_   = ds["YDim"][:]
#     lon_   = ds["XDim"][:]
    
    lat_   = convert(Array{Float64}, ds["lat"][:])
    lon_   = convert(Array{Float64}, ds["lon"][:])
    
    FT = eltype(lat_)
    lat = FT(lat)
    lon = FT(lon)
    
    # Find index (nearest neighbor, one could envision interpolation in space and time!):
    iLat = argmin(abs.(lat_ .- lat))
    iLon = argmin(abs.(lon_ .- lon))
#     @show ds["T"]
    # Temperature profile
    T    = convert(Array{FT,1}, ds["T"][ iLon,iLat, :, timeIndex])
    # specific humidity profile
    q    = convert(Array{FT,1}, ds["QV"][iLon,iLat,  :, timeIndex])

    # 3d pressure
    plocal = convert(Array{FT,1}, ds["PL"][iLon, iLat, :, timeIndex])
    
    # delta pressure 
    delp = convert(Array{FT,1}, ds["DELP"][iLon, iLat, :, timeIndex])
    
    # Surafce pressure
    psurf = convert(FT, ds["PS"][iLon, iLat, timeIndex])
    
    close(ds)
    
    prof_size = size(T)
    #################################################################
    # Define required constants 
    # Avogradro's number:
    Na = 6.0221415e23;
    # Dry and wet mass
    dryMass = 28.9647e-3  / Na  # in kg/molec, weighted average for N2 and O2
    wetMass = 18.01528e-3 / Na  # just H2O
    ratio = dryMass / wetMass 
    
    Rd = 287.04 # specific gas constant for dry air
    R_universal = 8.314472
    go = 9.8196 #(m/s**2) 
    
    n_layers = length(T)
    # also get a VMR vector of H2O (volumetric!)
    vmr_h2o = zeros(FT, n_layers, )
    vcd_dry = zeros(FT, n_layers, )
    vcd_h2o = zeros(FT, n_layers, )
    #################################################################
    
    dz = (delp./plocal).*(Rd.*T.*(1 .+ 0.608*q))./go
    
#     rho_N =  ds['PL'].values*(1-q_local*1.6068)/(R_universal*T_local)*Na/10000.0
    
#     rho_N_h2o =  ds['PL'].values*(q_local*1.6068)/(R_universal*T_local)*Na/10000.0
    
#     vmr_h2o = q_local*1.6068
    rho_N = (plocal .* (1 .- q .* ratio)) ./ (R_universal .* T) .* (Na/10000.0)
    rho_N_h2o = (plocal .* (q .* ratio)) ./ (R_universal .* T) .* (Na/10000.0)
    
    vmr_h2o = q .* ratio
    vcd_dry = dz .* rho_N
    
    # Total column density of water vapor:
#     display(sum(dz.*rho_N_h2o))
    
    return AtmosphericProfile(lat, lon, psurf, T, q, plocal, plocal, vmr_h2o, vcd_dry, vcd_h2o)
end;


# struct AtmosphericProfile{FT}
#     lat::FT
#     lon::FT
#     psurf::FT
#     T::Array{FT,1}
#     q::Array{FT,1}
#     p::Array{FT,1}
#     p_levels::Array{FT,1}
#     vmr_h2o::Array{FT,1}
#     vcd_dry::Array{FT,1}
#     vcd_h2o::Array{FT,1}
# end;
"Read atmospheric profile (just works for our file, can be generalized"
function read_atmos_profile(file::String, lat::Real, lon::Real, timeIndex; g₀=9.8196)
    @assert 1 <= timeIndex <= 4

    ds = Dataset(file)

    # See how easy it is to actually extract data? Note the [:] in the end reads in ALL the data in one step
    lat_   = ds["YDim"][:]
    lon_   = ds["XDim"][:]
    
#     lat_   = ds["lat"][:]
#     lon_   = ds["lon"][:]
    
    FT = eltype(lat_)
    lat = FT(lat)
    lon = FT(lon)
    
    # Find index (nearest neighbor, one could envision interpolation in space and time!):
    iLat = argmin(abs.(lat_ .- lat))
    iLon = argmin(abs.(lon_ .- lon))
    @show ds["T"]
    # Temperature profile
    T    = convert(Array{FT,1}, ds["T"][ iLon,iLat, :, timeIndex])
    # specific humidity profile
    q    = convert(Array{FT,1}, ds["QV"][iLon,iLat,  :, timeIndex])
    
    # Surafce pressure
    psurf = convert(FT, ds["PS"][iLon, iLat, timeIndex])
    
    # AK and BK global attributes (important to calculate pressure half-levels)
    ak = ds.attrib["HDF_GLOBAL.ak"][:]
    bk = ds.attrib["HDF_GLOBAL.bk"][:]

    p_half = (ak + bk * psurf)
    p_full = (p_half[2:end] + p_half[1:end - 1]) / 2
    close(ds)
    
    # Avogradro's number:
    Na = 6.0221415e23;
    # Dry and wet mass
    dryMass = 28.9647e-3  / Na  # in kg/molec, weighted average for N2 and O2
    wetMass = 18.01528e-3 / Na  # just H2O
    ratio = dryMass / wetMass 
    n_layers = length(T)
    # also get a VMR vector of H2O (volumetric!)
    vmr_h2o = zeros(FT, n_layers, )
    vcd_dry = zeros(FT, n_layers, )
    vcd_h2o = zeros(FT, n_layers, )

    # Now actually compute the layer VCDs
    for i = 1:n_layers 
        Δp = p_half[i + 1] - p_half[i]
        vmr_h2o[i] = q[i] * ratio
        vmr_dry = 1 - vmr_h2o[i]
        M  = vmr_dry * dryMass + vmr_h2o[i] * wetMass
        vcd_dry[i] = vmr_dry * Δp / (M * g₀ * 100.0^2)   # includes m2->cm2
        vcd_h2o[i] = vmr_h2o[i] * Δp / (M * g₀ * 100^2)
    end
    
    return AtmosphericProfile(lat, lon, psurf, T, q, p_full, p_half, vmr_h2o, vcd_dry, vcd_h2o)
end;


"Computes cross section matrix for arbitrary number of absorbers"
function compute_profile_crossSections_interp(profile::AtmosphericProfile, hitranModels::Array{InterpolationModel}, ν::AbstractRange{<:Real})
    nGases   = length(hitranModels)
    nProfile = length(profile.p)
    FT = eltype(profile.T)
    n_layers = length(profile.T)

#     σ_matrix = zeros(FT, (length(ν), n_layers, nGases))
    σ_matrix = zeros(FT, (length(ν), n_layers, nGases))
    @showprogress for i = 1:n_layers
        p_ = profile.p[i] / 100 # in hPa
        T_ = profile.T[i]
        
#         display(p_)
#         display(T_)
        
        for j = 1:nGases
            σ_matrix[:,i,j] = Array(absorption_cross_section(hitranModels[j], 
                                                       ν,
                                                       p_,
                                                       T_))
        end
    end
    return σ_matrix
end;



function forward_model_thermal_gpu(vmrs, σ_matrix, profile, albedo, Tsurf, μ, ν)
    nProfile = length(profile.T)
    nSpec    = size(σ_matrix,1)
    nGas     = length(vmrs)
    
    # wavelength in meter
#     wl = CuArray((1e7*1e-9)./collect(ν))
#     wl = CuArray((1e7*1e-9)./collect(ν)
#     display(wl)
    # Total sum of τ
    ∑τ       = CuArray(zeros(nSpec,nProfile))
#     ∑τ       = zeros(nSpec,nProfile)
    
    # Planck Source function
    S        = CuArray(zeros(nSpec,nProfile))
    
    # Layer boundary up and down fluxes:
    rad_down = CuArray(zeros(nSpec,nProfile+1))
    rad_up   = CuArray(zeros(nSpec,nProfile+1))
    
    for i=1:nProfile
        S[:,i] = planck.(ν, profile.T[i])
    end
    
    # sum up total layer τ:
    for i=1:nGas,j=1:nProfile
        ∑τ[:,j] .+= σ_matrix[:,j,i] .* (vmrs[i][j] .* profile.vcd_dry[j])'
    end
    
    
    # Transmission per layer
    T = exp.(-∑τ/μ)
    
    # Atmosphere from top to bottom:
    for i=1:nProfile
        rad_down[:,i+1] = rad_down[:,i].*T[:,i] .+ (1 .-T[:,i]).*S[:,i]
    end
#     display(rad_down)
    # Upward flux at surface:
    rad_up[:,end] = (1-albedo)*planck.(ν,Tsurf) .+ Array(rad_down[:,end])*albedo
#     display("past")   
    # Atmosphere from bottom to top
    for i=nProfile:-1:1
        rad_up[:,i] = rad_up[:,i+1].*T[:,i] .+ (1 .-T[:,i]) .* S[:,i]
    end
    
    return T, rad_down, rad_up
end




"Computes cross section matrix for arbitrary number of absorbers"
function compute_profile_crossSections(profile::AtmosphericProfile, hitranModels::Array{HitranModel}, ν::AbstractRange{<:Real})
    nGases   = length(hitranModels)
    nProfile = length(profile.p)
    FT = eltype(profile.T)
    n_layers = length(profile.T)

#     σ_matrix = zeros(FT, (length(ν), n_layers, nGases))
    σ_matrix = zeros(FT, (length(ν), n_layers, nGases))
    @showprogress for i = 1:n_layers
        p_ = profile.p[i] / 100 # in hPa
        T_ = profile.T[i]
        for j = 1:nGases
            σ_matrix[:,i,j] = Array(absorption_cross_section(hitranModels[j], 
                                                       ν,
                                                       p_,
                                                       T_))
        end
    end
    return σ_matrix
end;

"Creates a Gaussian Kernel"
function gaussian_kernel(FWHM, res; fac=5)
    co = 2.355
    width = FWHM / res / co
    extent = ceil(fac * width)
    d = Normal(0, width)
    kernel = centered(pdf.(d, -extent:extent))
    return kernel ./ sum(kernel)
end;

"Rescales x-axis to go from [-1,1]"
function rescale_x(a)
    a = a .- mean(a);
    a = a / (a[end] - a[1]) * 2;
end;

"Convolve and resample"
function conv_spectra(m::KernelInstrument, ν, spectrum)
    s = imfilter(spectrum, m.kernel)
    interp_cubic = CubicSplineInterpolation(ν, s)
    return interp_cubic(m.ν_out)
end;
    

"Reduce profile dimensions"
function reduce_profile(n::Int, profile::AtmosphericProfile, σ_matrix)
    @assert n < length(profile.T)
    @unpack lat, lon, psurf = profile
    # New rough half levels (boundary points)
    a = range(0, maximum(profile.p), length=n + 1)
    dims = size(σ_matrix)
    FT = eltype(σ_matrix)
    σ_matrix_lr = zeros(FT, dims[1], n, dims[3])
    T = zeros(FT, n);
    q = zeros(FT, n);
    p_full = zeros(FT, n);
    p_levels = zeros(FT, n + 1);
    vmr_h2o  = zeros(FT, n);
    vcd_dry  = zeros(FT, n);
    vcd_h2o  = zeros(FT, n);

    for i = 1:n
        ind = findall(a[i] .< profile.p .<= a[i + 1]);
        σ_matrix_lr[:,i,:] = mean(σ_matrix[:,ind,:], dims=2);
        p_levels[i] = profile.p_levels[ind[1]]
        p_levels[i + 1] = profile.p_levels[ind[end]]
        p_full[i] = mean(profile.p_levels[ind])
        T[i] = mean(profile.T[ind])
        q[i] = mean(profile.q[ind])
        vmr_h2o[i] = mean(profile.vmr_h2o[ind])
        vcd_dry[i] = sum(profile.vcd_dry[ind])
        vcd_h2o[i] = sum(profile.vcd_h2o[ind])
    end

    return AtmosphericProfile(lat, lon, psurf, T, q, p_full, p_levels, vmr_h2o, vcd_dry, vcd_h2o), σ_matrix_lr
end;

function artifact_ESE_folder(uuid::Union{String,Nothing}=nothing)
    name = "ESE156_2020"
    artifacts_toml = find_artifacts_toml(pwd())
    artifact_dict = Artifacts.load_artifacts_toml(artifacts_toml)
    Artifacts.do_artifact_str(name, artifact_dict, artifacts_toml, uuid) 
end


    
    
    




    
