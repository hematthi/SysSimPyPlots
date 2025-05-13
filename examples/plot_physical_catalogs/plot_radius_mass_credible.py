# To import required modules:
import numpy as np
import time
from tqdm import tqdm # for progress bar
import copy
import matplotlib
import matplotlib.cm as cm # for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
from matplotlib import ticker # for setting contour plots to log scale

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *

from mass_radius_models import *

##### For calling Julia functions:
# Import Julia:
from julia.api import Julia
jl = Julia(compiled_modules=False)

# Import Julia modules:
from julia import Main
Main.include("/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/src/models_test.jl") # can now call functions in this script using "jl.eval(f"name_of_function({a},{b})")" or just "Main.name_of_function(a,b)"!
Main.include("/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/src/clusters.jl")
#####





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Underlying/Fit_some8_KS_params9/'





##### To compute the credible regions for the median radius-mass relation from the model posterior distribution:
##### This will call Julia functions from Python

mass_min, mass_max = 0.1, 1e3
radii_min, radii_max = 0.5, 10.

M_array = np.logspace(np.log10(mass_min), np.log10(mass_max), 500)
qtls = [0.16,0.5,0.84] # for 1-sigma

##### (1) From the model parameters drawn from the posterior distribution:

#run_directory = 'Hybrid_NR20_AMD_model1/Fit_all_KS/Params12/GP_files/'
#loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory
run_directory = 'Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/GP_files/'
loadfiles_directory = '/Users/hematthi/Documents/NPP_ARC_Modernize_Kepler/Personal_research/SysSim/Model_Optimization/' + run_directory

n_params = 9
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 37.65, 1425.6 # 12 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 67.65, 141134.4 # 13 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 16.05, 14.26 # 8 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 25.0, 2.7, 2.45, 112.9 # fit some, 8 params
n_train, mean_f, sigma_f, lscales, vol = 2000, 30.0, 2.7, 3.03, 240.84 # fit some, 9 params
n_points, max_mean, max_std, max_post = 10000, 'Inf', 'Inf', -13.0 #100000, 'Inf', 'Inf', 'Inf'
file_name = 'GP_train%s_meanf%s_sigmaf%s_lscales%s_vol%s_points%s_mean%s_std%s_post%s.csv' % (n_train, mean_f, sigma_f, lscales, vol, n_points, max_mean, max_std, max_post)
xprior_accepted_table = load_GP_table_prior_draws(file_name, file_name_path=loadfiles_directory)
active_params_names = np.array(xprior_accepted_table.dtype.names[:n_params])

# To compute the quantiles for some model parameters:
μ_M_qtls = np.quantile(xprior_accepted_table['mean_ln_mass'], qtls)
σ_M_qtls = np.quantile(xprior_accepted_table['sigma_ln_mass'], qtls)
Minit_pdf_array_med = scipy.stats.norm(μ_M_qtls[1], σ_M_qtls[1]).pdf(np.log(M_array)) # pdf of initial mass distribution evaluated on M_array with the median param values of μ_M and σ_M

# To compute the median radius-mass prediction for each set of model parameters:
points_to_use = min(1000, len(xprior_accepted_table))
Minit_pdf_array_all = np.zeros((points_to_use, len(M_array))) # pdf's evaluated on M_array for each set of model params
μ_R_array_all = np.zeros((points_to_use, len(M_array)))
σ_R_array_all = np.zeros((points_to_use, len(M_array)))
print('Calculating radius-mass relation from each posterior draw...')
for i in tqdm(range(points_to_use)):
    params = xprior_accepted_table[i]
    
    μ_M, σ_M = params['mean_ln_mass'], params['sigma_ln_mass']
    C = params['norm_radius']
    M_break1 = 20. #params['break1_mass'] # 20.
    γ0, γ1 = params['power_law_γ0'], 0.5 #params['power_law_γ1'] # 0.5
    σ0, σ1 = params['power_law_σ0'], 0.3 #params['power_law_σ1'] # 0.3
    
    Minit_pdf_array_all[i] = scipy.stats.norm(μ_M, σ_M).pdf(np.log(M_array))
    
    μσ_R_array = np.array([Main.mean_radius_and_scatter_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=1e4, γ0=γ0, γ1=γ1, γ2=0., σ0=σ0, σ1=σ1, σ2=0.1) for M in M_array])
    μ_R_array_all[i] = np.array([μσ_R[0] for μσ_R in μσ_R_array])
    σ_R_array_all[i] = np.array([μσ_R[1] for μσ_R in μσ_R_array])

Minit_pdf_array_qtls = np.quantile(Minit_pdf_array_all, qtls, axis=0)
μ_R_array_qtls = np.quantile(μ_R_array_all, qtls, axis=0)

##### (2) From the model parameters drawn from the posterior distribution that have simulated catalogs (that still pass the distance threshold after simulation):

loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/GP_best_models_100/'

params_all = []

runs = 100
for i in range(runs):
    run_number = i+1
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    params_all.append(params_i)

# To compute the quantiles for some model parameters:
μ_M_qtls_sim = np.quantile([params['mean_ln_mass (ln M_earth)'] for params in params_all], qtls)
σ_M_qtls_sim = np.quantile([params['sigma_ln_mass (ln M_earth)'] for params in params_all], qtls)
Minit_pdf_array_all_sim = np.zeros((runs, len(M_array))) # pdf's evaluated on M_array for each set of model params
Minit_pdf_array_med_sim = scipy.stats.norm(μ_M_qtls_sim[1], σ_M_qtls_sim[1]).pdf(np.log(M_array)) # pdf of initial mass distribution evaluated on M_array with the median param values of μ_M and σ_M

# To compute the median radius-mass prediction for each set of model parameters:
μ_R_array_all_sim = np.zeros((runs, len(M_array)))
σ_R_array_all_sim = np.zeros((runs, len(M_array)))
print('Calculating radius-mass relation from each simulated catalog...')
for i,params in enumerate(tqdm(params_all)):
    μ_M, σ_M = params['mean_ln_mass (ln M_earth)'], params['sigma_ln_mass (ln M_earth)']
    C = params['norm_radius (R_earth)']
    M_break1 = params['break_mass (M_earth)']
    γ0, γ1 = params['power_law_γ0'], params['power_law_γ1']
    σ0, σ1 = params['power_law_σ0'], params['power_law_σ1']
    
    Minit_pdf_array_all_sim[i] = scipy.stats.norm(μ_M, σ_M).pdf(np.log(M_array))
    
    μσ_R_array = np.array([Main.mean_radius_and_scatter_given_mass_neil_rogers2020(M, C=C, M_break1=M_break1, M_break2=1e4, γ0=γ0, γ1=γ1, γ2=0., σ0=σ0, σ1=σ1, σ2=0.1) for M in M_array])
    μ_R_array_all_sim[i] = np.array([μσ_R[0] for μσ_R in μσ_R_array])
    σ_R_array_all_sim[i] = np.array([μσ_R[1] for μσ_R in μσ_R_array])

Minit_pdf_array_qtls_sim = np.quantile(Minit_pdf_array_all_sim, qtls, axis=0)
μ_R_array_qtls_sim = np.quantile(μ_R_array_all_sim, qtls, axis=0)



R_S07_silicate_array = np.array([Main.radius_given_mass_pure_silicate_fit_seager2007(M) for M in M_array])





#####

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 12 #legend labels font size

bins = 100



##### To plot radius vs. mass from several models:

plot_color_gradient_by_pdf = True # whether to color the credible regions with a gradient colormap corresponding to the pdf of the initial mass distribution (with median params)
also_plot_posterior = False # whether to also plot the credible regions from the posterior draws (if False, will just plot the credible region from the simulated catalogs)

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5, 5, left=0.15, bottom=0.1, right=0.93, top=0.95, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:]) # main panel
# Plot the mass-radius/radius-mass relations:
plt.plot(mass_evals_med_H20, radius_evals_H20, '-', color='k') #, label='H20, mean prediction'
plt.fill_betweenx(radius_evals_H20, mass_evals_016_H20, mass_evals_084_H20, color='k', alpha=0.2, label=r'H20 model, 1$\sigma$ scatter around median') # 16%-84% region of H20 model
#'''
if also_plot_posterior:
    if plot_color_gradient_by_pdf:
        c_norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
        cmap = cm.get_cmap('Purples')
        plt.plot(M_array, μ_R_array_qtls[1], '-', color='m')
        plt.plot(M_array, μ_R_array_qtls[0], '--', color='m')
        plt.plot(M_array, μ_R_array_qtls[2], '--', color='m')
        for i in range(len(M_array)-1):
            plt.fill_between(M_array[i:i+1], μ_R_array_qtls[2][i:i+1], μ_R_array_qtls[0][i:i+1], color=cmap(c_norm(Minit_pdf_array_med[i]/np.max(Minit_pdf_array_med))), alpha=0.5, label='Hybrid model, 16%-84% of median (posterior draws)' if i==np.argmax(Minit_pdf_array_med) else None)
    else:
        plt.plot(M_array, μ_R_array_qtls[1], '-', color='m')
        plt.fill_between(M_array, μ_R_array_qtls[2], μ_R_array_qtls[0], color='m', alpha=0.2, label='Hybrid model, 16%-84% of median (posterior draws)')
#'''
if plot_color_gradient_by_pdf:
    c_norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
    cmap = cm.get_cmap('Blues')
    plt.plot(M_array, μ_R_array_qtls_sim[1], '-', color='b')
    plt.plot(M_array, μ_R_array_qtls_sim[0], '--', color='b')
    plt.plot(M_array, μ_R_array_qtls_sim[2], '--', color='b')
    for i in range(len(M_array)-1):
        plt.fill_between(M_array[i:i+1], μ_R_array_qtls_sim[2][i:i+1], μ_R_array_qtls_sim[0][i:i+1], color=cmap(c_norm(Minit_pdf_array_med_sim[i]/np.max(Minit_pdf_array_med_sim))), alpha=0.5, label='Hybrid model, 16%-84% of median' if i==np.argmax(Minit_pdf_array_med_sim) else None)
else:
    plt.plot(M_array, μ_R_array_qtls_sim[1], '-', color='b')
    plt.fill_between(M_array, μ_R_array_qtls_sim[2], μ_R_array_qtls_sim[0], color='b', alpha=0.2, label='Hybrid model, 16%-84% of median (simulated)')
#'''
plt.plot(M_array, R_S07_silicate_array, color='g') #, label='S07, pure-silicate'
plt.fill_between(M_array, 0.95*R_S07_silicate_array, 1.05*R_S07_silicate_array, color='g', alpha=0.2, label='S07 pure-silicate model with 5% scatter') #, label='NR20, 5% scatter around S07 pure-silicate (final)'
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([1e-1, 1., 10., 1e2, 1e3])
ytick_vals = np.array([0.5, 1., 2., 4., 10.])
plt.xticks(xtick_vals, xtick_vals)
plt.yticks(ytick_vals, ytick_vals)
plt.xlim([mass_min, mass_max])
plt.ylim([radii_min, radii_max])
plt.xlabel(r'Planet mass, $M_p$ [$M_\oplus$]', fontsize=tfs)
plt.ylabel(r'Planet radius, $R_p$ [$R_\oplus$]', fontsize=tfs)
plt.legend(loc='lower right', bbox_to_anchor=(0.99,0.01), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[0,:]) # top histogram
# Just plotting either the posterior or the simulated catalogs for this panel:
if also_plot_posterior:
    str_μ_M = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(μ_M_qtls[1]), '{:0.2f}'.format(μ_M_qtls[1]-μ_M_qtls[0]), '{:0.2f}'.format(μ_M_qtls[2]-μ_M_qtls[1]))
    str_σ_M = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(σ_M_qtls[1]), '{:0.2f}'.format(σ_M_qtls[1]-σ_M_qtls[0]), '{:0.2f}'.format(σ_M_qtls[2]-σ_M_qtls[1]))
    plt.plot(M_array, Minit_pdf_array_qtls[1], '-', color='m')
    plt.fill_between(M_array, Minit_pdf_array_qtls[2], Minit_pdf_array_qtls[0], color='m', alpha=0.2, label=r'Hybrid model, $\ln{M_{p,\rm init}} \sim$' '\n' r'$\mathcal{N}(\mu = %s, \sigma = %s)$' % (str_μ_M, str_σ_M))
else:
    str_μ_M = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(μ_M_qtls_sim[1]), '{:0.2f}'.format(μ_M_qtls_sim[1]-μ_M_qtls_sim[0]), '{:0.2f}'.format(μ_M_qtls_sim[2]-μ_M_qtls_sim[1]))
    str_σ_M = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(σ_M_qtls_sim[1]), '{:0.2f}'.format(σ_M_qtls_sim[1]-σ_M_qtls_sim[0]), '{:0.2f}'.format(σ_M_qtls_sim[2]-σ_M_qtls_sim[1]))
    plt.plot(M_array, Minit_pdf_array_qtls_sim[1], '-', color='b')
    plt.fill_between(M_array, Minit_pdf_array_qtls_sim[2], Minit_pdf_array_qtls_sim[0], color='b', alpha=0.2, label=r'Hybrid model, $\ln{M_{p,\rm init}} \sim$' '\n' r'$\mathcal{N}(\mu = %s, \sigma = %s)$' % (str_μ_M, str_σ_M))
plt.gca().set_xscale("log")
plt.xlim([mass_min, mass_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    save_fn = 'Models_underlying_radius_mass_credible_both.pdf' if also_plot_posterior else 'Models_underlying_radius_mass_credible.pdf'
    plt.savefig(savefigures_directory + save_fn)
    plt.close()
plt.show()
