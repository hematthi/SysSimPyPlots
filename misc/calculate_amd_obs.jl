using ExoplanetsSysSim

using CSV
using DataFrames
using Glob
using PyPlot

include("../../SysSimExClusters/src/AMD_stability/solar_system.jl")




##### To load a table of the physical catalog with observed planets only:

dir = "/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/"
file = joinpath(dir, "physical_catalog_obs_only.csv")
table_planets = CSV.read(file)





##### To evaluate AMD stability for each system:

tid_all = unique(table_planets[!,:target_id])
n_sys = length(tid_all)

n_pl_all = Vector{Float64}(undef, n_sys)
amd_tot_all = Vector{Float64}(undef, n_sys)
amd_crit_all = Vector{Float64}(undef, n_sys)

for (i,tid) in enumerate(tid_all)
    id_sys = (1:size(table_planets,1))[table_planets[!,:target_id] .== tid]
    planets = table_planets[id_sys,:]
    n_pl_all[i] = size(planets, 1)

    m = planets[!,:planet_mass]
    μ = m ./ planets[!,:star_mass]
    P = planets[!,:period]
    a = semimajor_axis.(P, m .+ planets[!,:star_mass])
    e = planets[!,:ecc]
    i_m = planets[!,:incl_mut]

    AMD_tot = total_AMD_system(μ, a, e, i_m)[1]
    AMD_crit = critical_AMD_system(μ, a)

    amd_tot_all[i] = AMD_tot
    amd_crit_all[i] = AMD_crit
end

amd_ratio_crit_all = amd_tot_all ./ amd_crit_all





##### To make some plots:

logrange(x1, x2, n) = collect(10^logx for logx in range(log10(x1), log10(x2), length=n))

# Histogram of planet masses:
hist(table_planets[!,:planet_mass] ./ ExoplanetsSysSim.earth_mass, bins=logrange(1e-1, 1e3, 50), histtype="step")
ax = gca()
ax.set_xscale("log")
xlabel(L"$M_p$ ($M_\oplus$)", fontsize=20)

# Histogram of periods:
hist(table_planets[!,:period], bins=logrange(3, 300, 50), histtype="step")
ax = gca()
ax.set_xscale("log")
xlabel(L"$P$ (days)", fontsize=20)

# Histogram of eccentricities:
hist(table_planets[!,:ecc], bins=logrange(1e-3, 1, 50), histtype="step")
ax = gca()
ax.set_xscale("log")
xlabel(L"$e$", fontsize=20)

# Histogram of mutual inclinations (ignoring true singles which have zero mutual inclination):
im_all = table_planets[!,:incl_mut]
hist(im_all .* 180 ./π, bins=logrange(1e-2, 1e2, 50), histtype="step")
ax = gca()
ax.set_xscale("log")
xlabel(L"$i_m$ ($^\circ$)", fontsize=20)

# Histogram of total AMD/AMD_crit:
plot = PyPlot.figure(figsize=(8,5))
hist(amd_ratio_crit_all, bins=logrange(1e-5, 1, 50), histtype="step")
ax = gca()
ax.tick_params(axis="both", labelsize=20)
ax.set_xscale("log")
xlabel(L"AMD$_{tot}$/AMD$_{crit}$", fontsize=20)
ylabel("Systems", fontsize=20)
plot.tight_layout()

# Histogram of total AMD/AMD_crit split by (observed) multiplicity:
plot = PyPlot.figure(figsize=(8,5))
labels_m = [L"$m = 1$", L"$m = 2$", L"$m = 3$", L"$m = 4+$"]
bools_m = [n_pl_all .== 1, n_pl_all .== 2, n_pl_all .== 3, n_pl_all .>= 4]
hist([amd_ratio_crit_all[bools] for bools in bools_m], bins=logrange(1e-5, 1, 50), histtype="step", label=labels_m)
ax = gca()
ax.tick_params(axis="both", labelsize=20)
ax.set_xscale("log")
xlabel(L"AMD$_{tot}$/AMD$_{crit}$", fontsize=20)
ylabel("Systems", fontsize=20)
legend(loc="upper left", bbox_to_anchor=(0,1), frameon=false, fontsize=16)
plot.tight_layout()
