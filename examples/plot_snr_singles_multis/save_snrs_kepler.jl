#using CSV
using PyPlot

include("../../SysSimExClusters/src/clusters.jl")
include("../../SysSimExClusters/src/planetary_catalog.jl")





##### To get the SNR of each planet:

snr_singles = Float64[]
snr_multis = Float64[]
snr_multis_weakest = Float64[]

#=
# No cuts at all (except only keeping confirmed and candidate planets):
###f = open("snrs_kepler_all.out", "w")
planets_keep = planet_catalog[(planet_catalog[!,:koi_disposition] .== "CONFIRMED") .| (planet_catalog[!,:koi_disposition] .== "CANDIDATE"),:]
KOI_systems = [x[1:6] for x in planets_keep[!,:kepoi_name]]
checked_bools = zeros(size(planets_keep,1))
for i in 1:length(KOI_systems)
    if checked_bools[i] == 0
        system_i = (1:length(KOI_systems))[KOI_systems .== KOI_systems[i]]
        checked_bools[system_i] .= 1
        system_snr = planets_keep[!,:koi_model_snr][system_i]
        ###println(f, system_snr)
        if length(system_snr) > 1
            append!(snr_multis, system_snr[.~ismissing.(system_snr)])
            append!(snr_multis_weakest, minimum(system_snr[.~ismissing.(system_snr)]))
        elseif length(system_snr) == 1
            append!(snr_singles, system_snr[.~ismissing.(system_snr)])
        end
    end
end
###close(f)
=#

#=
# Only applying our period and radius cuts (3-300 d, 0.5-10 R_earth):
# WARNING: actually the cuts in radius here are based on the old radius values (i.e. without revised stellar properties from Gaia DR2)
###f = open("snrs_kepler_PR_cuts.out", "w")
planets_keep = planet_catalog[(planet_catalog[!,:koi_disposition] .== "CONFIRMED") .| (planet_catalog[!,:koi_disposition] .== "CANDIDATE"),:]
planets_keep = planets_keep[(planets_keep[!,:koi_period] .> 3.) .& (planets_keep[!,:koi_period] .< 300.) .& (planets_keep[!,:koi_prad] .> 0.5) .& (planets_keep[!,:koi_prad] .< 10.) .& (.~ismissing.(planets_keep[!,:koi_prad])), :]
KOI_systems = [x[1:6] for x in planets_keep[!,:kepoi_name]]
checked_bools = zeros(size(planets_keep,1))
for i in 1:length(KOI_systems)
    if checked_bools[i] == 0
        system_i = (1:length(KOI_systems))[KOI_systems .== KOI_systems[i]]
        checked_bools[system_i] .= 1
        system_snr = planets_keep[!,:koi_model_snr][system_i]
        ###println(f, system_snr)
        if length(system_snr) > 1
            append!(snr_multis, system_snr[.~ismissing.(system_snr)])
            append!(snr_multis_weakest, minimum(system_snr[.~ismissing.(system_snr)]))
        elseif length(system_snr) == 1
            append!(snr_singles, system_snr[.~ismissing.(system_snr)])
        end
    end
end
###close(f)
=#

#=
# Only applying our stellar cuts (stellar sample from HFR2020b):
###f = open("snrs_kepler_HFR2020b_stars.out", "w")
add_param_fixed(sim_param,"min_period", 0.)
add_param_fixed(sim_param,"max_period", Inf)
add_param_fixed(sim_param,"min_radius", 0.)
add_param_fixed(sim_param,"max_radius", Inf)
@time planets_keep = keep_planet_candidates_given_sim_param(planet_catalog; sim_param=sim_param, stellar_catalog=stellar_catalog, recompute_radii=true)
KOI_systems = [x[1:6] for x in planets_keep[!,:kepoi_name]]
checked_bools = zeros(size(planets_keep,1))
for i in 1:length(KOI_systems)
    if checked_bools[i] == 0
        system_i = (1:length(KOI_systems))[KOI_systems .== KOI_systems[i]]
        checked_bools[system_i] .= 1
        system_snr = planets_keep[!,:koi_model_snr][system_i]
        ###println(f, system_snr)
        if length(system_snr) > 1
            append!(snr_multis, system_snr[.~ismissing.(system_snr)])
            append!(snr_multis_weakest, minimum(system_snr[.~ismissing.(system_snr)]))
        elseif length(system_snr) == 1
            append!(snr_singles, system_snr[.~ismissing.(system_snr)])
        end
    end
end
###close(f)
=#

#
# Using all our cuts in HFR2020b:
###f = open("snrs_kepler_HFR2020b_all_cuts.out", "w")
KOI_systems = [x[1:6] for x in planets_cleaned[!,:kepoi_name]]
checked_bools = zeros(size(planets_cleaned,1))
for i in 1:length(KOI_systems)
    if checked_bools[i] == 0
        system_i = (1:length(KOI_systems))[KOI_systems .== KOI_systems[i]]
        checked_bools[system_i] .= 1
        system_snr = planets_cleaned[!,:koi_model_snr][system_i]
        ###println(f, system_snr)
        if length(system_snr) > 1
            append!(snr_multis, system_snr[.~ismissing.(system_snr)])
            append!(snr_multis_weakest, minimum(system_snr[.~ismissing.(system_snr)]))
        elseif length(system_snr) == 1
            append!(snr_singles, system_snr[.~ismissing.(system_snr)])
        end
    end
end
###close(f)
#




##### To make some plots:

logrange(x1, x2, n) = collect(10^logx for logx in range(log10(x1), log10(x2), length=n))

# Histogram of mass ratios:
hist([snr_singles, snr_multis, snr_multis_weakest], bins=logrange(5., 1e4, 101), histtype="step", label=["Singles","Multis","Multis weakest"])
ax = gca()
ax.set_xscale("log")
xlabel("SNR", fontsize=20)
legend(fontsize=20)

