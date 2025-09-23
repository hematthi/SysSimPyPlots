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
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/clustered_initial_masses/Underlying/Compare_to_hybrid_nonclustered/'





##### To compute the credible regions for the median radius-mass relation from the model posterior distribution:
##### This will call Julia functions from Python

mass_min, mass_max = 0.1, 1e3
radii_min, radii_max = 0.5, 10.

M_array = np.logspace(np.log10(mass_min), np.log10(mass_max), 500)
qtls = [0.16,0.5,0.84] # for 1-sigma

R_S07_silicate_array = np.array([Main.radius_given_mass_pure_silicate_fit_seager2007(M) for M in M_array])

##### From the model parameters drawn from the posterior distribution that have simulated catalogs (that still pass the distance threshold after simulation):

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_best_models_100/'

model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2]
model_names = ['Hybrid Model 2', 'Hybrid Model 1']
model_colors = ['g', 'b']
model_colormaps = ['Greens', 'Blues']
models = len(model_loadfiles_dirs)

runs = 100

model_params_all = []
model_μ_M_qtls_sim = []
model_σ_M_qtls_sim = []
model_Minit_pdf_array_med_sim = []
model_Minit_pdf_array_qtls_sim = []
model_μ_R_array_qtls_sim = []

# For the model with clustered initial masses only, also compute the initial mass pdf of a typical cluster (centered at the same 'mu' (e.g. median of the overall initial mass distribution), but with 'sigma' from the simulation params):
model_σ_M_cluster_qtls_sim = []
model_Minit_cluster_pdf_array_qtls_sim = []

for loadfiles_dir in model_loadfiles_dirs:
    params_all = []
    for i in range(runs):
        run_number = i+1
        params_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
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
    
    # Append the results we need from each model:
    model_params_all.append(params_all)
    model_μ_M_qtls_sim.append(μ_M_qtls_sim)
    model_σ_M_qtls_sim.append(σ_M_qtls_sim)
    model_Minit_pdf_array_med_sim.append(Minit_pdf_array_med_sim)
    model_Minit_pdf_array_qtls_sim.append(Minit_pdf_array_qtls_sim)
    model_μ_R_array_qtls_sim.append(μ_R_array_qtls_sim)
    
    # For the model with clustered initial masses:
    if 'sigma_ln_mass_in_cluster (ln M_earth)' in params_all[0]:
        σ_M_cluster_qtls_sim = np.quantile([params['sigma_ln_mass_in_cluster (ln M_earth)'] for params in params_all], qtls)
        Minit_cluster_pdf_array_all_sim = np.array([scipy.stats.norm(μ_M_qtls_sim[1], params['sigma_ln_mass_in_cluster (ln M_earth)']).pdf(np.log(M_array)) for params in params_all]) # pdf's evaluated on M_array for each 'sigma_ln_mass_in_cluster', at the median 'mean_ln_mass'
        
        Minit_cluster_pdf_array_qtls_sim = np.quantile(Minit_cluster_pdf_array_all_sim, qtls, axis=0)
        
        # Append the results:
        model_σ_M_cluster_qtls_sim.append(σ_M_cluster_qtls_sim)
        model_Minit_cluster_pdf_array_qtls_sim.append(Minit_cluster_pdf_array_qtls_sim)
    else:
        # Append empty lists to keep the same length for indexing the models:
        model_σ_M_cluster_qtls_sim.append([])
        model_Minit_cluster_pdf_array_qtls_sim.append([])





##### Optionally, also load a simulated catalog to plot the initial and final radii vs. masses for a sample of planets:

plot_sample = True

if plot_sample:
    loadfiles_directory_sample = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/clustered_initial_masses/Fit_some8p1_KS/Params10_fix_highM/GP_best_models_100/'
    run_number = 3

    has_extra_params = True # [NEW,TEMPORARY] whether the simulated catalog has extra planet params

    params = read_sim_params(loadfiles_directory_sample + 'periods%s.out' % run_number)
    sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory_sample, run_number=run_number, load_full_tables=False if has_extra_params else True)

    N_pl_plot = 1000 # number of planets to plot





#####

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

bins = 100



##### To plot radius vs. mass from several models:

fig = plt.figure(figsize=(12,12))
plot = GridSpec(5, 5, left=0.1, bottom=0.08, right=0.95, top=0.98, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:]) # main panel
# Plot the mass-radius/radius-mass relations:
for m in range(models):
    color = model_colors[m]
    μ_R_array_qtls_sim = model_μ_R_array_qtls_sim[m]
    Minit_pdf_array_med_sim = model_Minit_pdf_array_med_sim[m]
    
    plot_color_gradient_by_pdf = True if m==0 else False
    if plot_color_gradient_by_pdf:
        c_norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
        cmap = cm.get_cmap(model_colormaps[m])
        plt.plot(M_array, μ_R_array_qtls_sim[1], '-', color=color)
        plt.plot(M_array, μ_R_array_qtls_sim[0], '--', color=color)
        plt.plot(M_array, μ_R_array_qtls_sim[2], '--', color=color, label=model_names[m] + ', 16%-84% of median')
        for i in range(len(M_array)-1):
            plt.fill_between(M_array[i:i+1], μ_R_array_qtls_sim[2][i:i+1], μ_R_array_qtls_sim[0][i:i+1], color=cmap(c_norm(Minit_pdf_array_med_sim[i]/np.max(Minit_pdf_array_med_sim))), alpha=0.5) #, label=model_names[m] + ', 16%-84% of median' if i==np.argmax(Minit_pdf_array_med_sim) else None)
    else:
        plt.plot(M_array, μ_R_array_qtls_sim[1], '-', color=color)
        plt.plot(M_array, μ_R_array_qtls_sim[0], '--', color=color)
        plt.plot(M_array, μ_R_array_qtls_sim[2], '--', color=color, label=model_names[m] + ', 16%-84% of median')
        #plt.fill_between(M_array, μ_R_array_qtls_sim[2], μ_R_array_qtls_sim[0], color=color, alpha=0, label=model_names[m] + ', 16%-84% of median')
#
plt.plot(mass_evals_med_H20, radius_evals_H20, '-', color='k') #, label='H20, mean prediction'
plt.fill_betweenx(radius_evals_H20, mass_evals_016_H20, mass_evals_084_H20, color='k', alpha=0.2, label=r'H20 model, 1$\sigma$ scatter around median') # 16%-84% region of H20 model
#
plt.plot(M_array, R_S07_silicate_array, color='tab:brown') #, label='S07, pure-silicate'
plt.fill_between(M_array, 0.95*R_S07_silicate_array, 1.05*R_S07_silicate_array, color='tab:brown', alpha=0.5, label='S07 pure-silicate model with 5% scatter') #, label='NR20, 5% scatter around S07 pure-silicate (final)'
# Plot a sample population:
if plot_sample:
    plt.scatter(sssp['init_mass_all'][:N_pl_plot], sssp['init_radii_all'][:N_pl_plot], s=20, marker='o', edgecolors='k', facecolors='none', label='Sample population (initial)')
    plt.scatter(sssp['mass_all'][:N_pl_plot], sssp['radii_all'][:N_pl_plot], s=30, marker='.', c='r', label='Sample population (final)')
#
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
for m in range(models):
    color = model_colors[m]
    μ_M_qtls_sim = model_μ_M_qtls_sim[m]
    σ_M_qtls_sim = model_σ_M_qtls_sim[m]
    Minit_pdf_array_qtls_sim = model_Minit_pdf_array_qtls_sim[m]
    
    str_μ_M = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(μ_M_qtls_sim[1]), '{:0.2f}'.format(μ_M_qtls_sim[1]-μ_M_qtls_sim[0]), '{:0.2f}'.format(μ_M_qtls_sim[2]-μ_M_qtls_sim[1]))
    str_σ_M = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(σ_M_qtls_sim[1]), '{:0.2f}'.format(σ_M_qtls_sim[1]-σ_M_qtls_sim[0]), '{:0.2f}'.format(σ_M_qtls_sim[2]-σ_M_qtls_sim[1]))
    plt.plot(M_array, Minit_pdf_array_qtls_sim[1], '-', color=color)
    plt.fill_between(M_array, Minit_pdf_array_qtls_sim[2], Minit_pdf_array_qtls_sim[0], color=color, alpha=0.2, label=model_names[m] + '\n' r'$\mathcal{N}(\mu = %s, \sigma = %s)$' % (str_μ_M, str_σ_M)) # r', $\ln{M_{p,\rm init}} \sim$' '\n'
    
    # For the model with clustered initial masses:
    if 'sigma_ln_mass_in_cluster (ln M_earth)' in model_params_all[m][0]:
        σ_M_cluster_qtls_sim = model_σ_M_cluster_qtls_sim[m]
        Minit_cluster_pdf_array_qtls_sim = model_Minit_cluster_pdf_array_qtls_sim[m]
        
        str_μ_M_cluster = '{:0.2f}'.format(μ_M_qtls_sim[1])
        str_σ_M_cluster = '%s_{-%s}^{+%s}' % ('{:0.2f}'.format(σ_M_cluster_qtls_sim[1]), '{:0.2f}'.format(σ_M_cluster_qtls_sim[1]-σ_M_cluster_qtls_sim[0]), '{:0.2f}'.format(σ_M_cluster_qtls_sim[2]-σ_M_cluster_qtls_sim[1]))
        y_scale = np.max(Minit_pdf_array_qtls_sim[2])/np.max(Minit_cluster_pdf_array_qtls_sim[2]) # factor to scale the pdf values so the lognormal of the cluster is not too tall
        plt.plot(M_array, y_scale*Minit_cluster_pdf_array_qtls_sim[1], ':', color=color)
        plt.fill_between(M_array, y_scale*Minit_cluster_pdf_array_qtls_sim[2], y_scale*Minit_cluster_pdf_array_qtls_sim[0], color=color, alpha=0.1) # label=r'typical cluster, $\mathcal{N}(\mu = %s, \sigma = %s)$' % (str_μ_M_cluster, str_σ_M_cluster)
        plt.text(0.05, 0.1, r'typical cluster $(\sigma = %s)$' % str_σ_M_cluster, fontsize=lfs-2, transform=ax.transAxes)
plt.gca().set_xscale("log")
plt.xlim([mass_min, mass_max])
plt.xticks([])
plt.yticks([])
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs-2)

if savefigures:
    plt.savefig(savefigures_directory + 'Models_underlying_radius_mass_credible.pdf')
    plt.close()
plt.show()
