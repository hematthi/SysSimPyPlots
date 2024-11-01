# To import required modules:
import numpy as np
import time
import os
import sys
import copy

##### For calling Julia functions:
# Import Julia:
from julia.api import Julia
jl = Julia(compiled_modules=False)

# Import Julia modules:
from julia import Main
Main.include("/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/src/models_test.jl") # can now call functions in this script using "jl.eval(f"name_of_function({a},{b})")" or just "Main.name_of_function(a,b)"!
Main.include("/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/src/clusters.jl")
#####





##### NR20 model: best-fit parameters (medians of posteriors) for Model 2 from Table 1 of NR20:
'''
μ_M, σ_M = 1.00, 1.65 # ln(Earth masses)
C = 2.37 # Earth radii
M_break1 = 17.4 # Earth masses
M_break2 = 175.7 # Earth masses
γ0, γ1, γ2 = 0.00, 0.74, 0.04
σ0, σ1, σ2 = 0.18, 0.34, 0.10
R_min, R_max = 0.4, 20. # minimum and maximum planet radii to truncate at (Earth radii)
P_min, P_break, P_max = 0.3, 7.16, 100. # days
β1, β2 = 0.88, -0.76
α = 7.98 # fudge factor for mass-loss timescale
'''
##### Hybrid model: best-fit parameters (medians of posteriors) from ABC:
#'''
μ_M, σ_M = 0.95, 1.60 # ln(Earth masses)
C = 1.79 # Earth radii
M_break1 = 61.2 # Earth masses
M_break2 = 1e3 # Earth masses # NOTE: no M_break2 in this model so set a high value
γ0, γ1, γ2 = 0.10, 0.37, 0. # NOTE: only one break, so γ2 is arbitrary
σ0, σ1, σ2 = 0.25, 0.26, 0.1 # NOTE: only one break, so σ2 is arbitrary
R_min, R_max = 0.4, 20. # minimum and maximum planet radii to truncate at (Earth radii)
P_min, P_break, P_max = 3., 30., 300. # days # NOTE: no P_break in this model so value is arbitrary
β1, β2 = 0.06, 0.06 # NOTE: no P_break in this model so both of these are the same
α = 7.98 # fudge factor for mass-loss timescale # NOTE: fixed for some runs (to what value?)
#'''

M_min, M_max = 0.1, 1e3
M_array = np.logspace(np.log10(M_min), np.log10(M_max), 1000)
μσ_R_array = [Main.mean_radius_and_scatter_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=M_break2, γ0=γ0, γ1=γ1, γ2=γ2, σ0=σ0, σ1=σ1, σ2=σ2) for M in M_array]

μ_R_array = np.array([μσ_R[0] for μσ_R in μσ_R_array])
σ_R_array = np.array([μσ_R[1] for μσ_R in μσ_R_array])

R_S07_silicate_array = np.array([Main.radius_given_mass_pure_silicate_fit_seager2007(M) for M in M_array])



##### Mocking up a simple population:
#'''
N_pl = 10000 # number of planets to draw
print('Simulating a simple underlying population with %s planets...' % N_pl)

M_init_all = np.random.lognormal(μ_M, σ_M, N_pl) # initial planet masses (Earth masses)
R_init_all = np.array([Main.draw_radius_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=M_break2, γ0=γ0, γ1=γ1, γ2=γ2, σ0=σ0, σ1=σ1, σ2=σ2) for M in M_init_all]) # initial planet radii (Earth radii)
R_init_all[R_init_all < R_min] = R_min

M_env_all = np.array([Main.envelope_mass_smoothed_low_high_neil_rogers2020(M) for M in M_init_all]) # initial envelope masses (Earth masses)

P_all = jl.eval(f"ExoplanetsSysSim.draw_broken_power_law({β1}, {β2}, {P_min}, {P_max}, {P_break}, {N_pl})") # orbital periods (days) #NOTE: need "jl.eval(f"func({args})")" instead of "Main.func(args)" to call Julia functions in packages here
F_p_all = np.array([Main.bolometric_flux_at_planet_period(P) for P in P_all]) # assuming all planets are around Solar mass and luminosity stars (erg s^-1 cm^-2)

t_loss_all = np.array([Main.mass_loss_timescale_lopez2012(M_env_all[i], R_init_all[i], F_p_all[i]) for i in range(N_pl)]) # mass-loss timescales (Gyrs)
p_ret_all = np.array([Main.prob_retain_envelope_neil_rogers2020(5., t, α) for t in t_loss_all]) # probabilities of retaining envelope
bools_ret_all = np.random.rand(N_pl) < p_ret_all # True = retain envelope, False = lose envelope

M_final_all = copy.deepcopy(M_init_all)
M_final_all[~bools_ret_all] = M_final_all[~bools_ret_all] - M_env_all[~bools_ret_all] # final planet masses (Earth masses)

R_final_all = copy.deepcopy(R_init_all)
R_final_all[~bools_ret_all] = np.array([Main.draw_radius_given_mass_neil_rogers2020(Main.radius_given_mass_pure_silicate_fit_seager2007(M), 0.05) for M in M_final_all[~bools_ret_all]]) # final planet radii (Earth radii)
#'''
