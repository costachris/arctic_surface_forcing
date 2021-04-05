# Use our tools (you might need to add packages, see file)
using CUDA
device!(2)
# 0 indexed

include("thermal_radiation_helper.jl")
using RadiativeTransfer
using RadiativeTransfer.Architectures
using RadiativeTransfer.CrossSection
using ..Architectures: GPU

using JLD2
using NumericalIntegration
# using PyPlot
# use float32

W_m_mW_cm = 1e2*1e3

file = "/net/fluo/data2/groupMembers/cchristo/reanalysis_3d/merra2_averages/seas_means_renamed.nc"

output_path = "/net/fluo/data2/groupMembers/cchristo/rf_maps/summer_sh_rf"



# loadloop_over_arrayodel 
# interp_model_co2 = load_interpolation_model("/net/fluo/data2/groupMembers/cchristo/interp_models/test/co2_p10_t_5")
interp_model_co2 = @load "/net/fluo/data2/groupMembers/cchristo/interp_models/test/co2_p10_t_5" 
interp_model_co2  = eval(interp_model_co2[1])

interp_model_h2o = @load "/net/fluo/data2/groupMembers/cchristo/interp_models/test/h2o_p10_t_5"
interp_model_h2o = eval(interp_model_h2o[1])

interp_model_ch4 = @load "/net/fluo/data2/groupMembers/cchristo/interp_models/test/ch4_p10_t_5"
interp_model_ch4 = eval(interp_model_ch4[1])

hitran_array_interp = [interp_model_co2, interp_model_h2o, interp_model_ch4];

# northern hemis
# lats = collect(30.0:1:90.0) ; 

# southern hemis
lats = collect(-90.0:1:-45.0);


lons = collect(-179.5:3:179.5) ; 
# lons = Array([-38.47]) ; 
display("sh summer")

result = loop_over_array(file, lats, lons, 2, hitran_array_interp) ;

@save output_path result