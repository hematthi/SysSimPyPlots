# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
from matplotlib import ticker #for setting contour plots to log scale
import scipy.integrate #for numerical integration
import scipy.misc #for factorial function
from scipy.special import erf #error function, used in computing CDF of normal distribution
import scipy.interpolate #for interpolation functions
import corner #corner.py package for corner plots
#matplotlib.rc('text', usetex=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *





savefigures = False
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Intrinsic_compare/Distribute_AMD/'
save_name = 'Models_Compare' #'Models_Compare_fswp_alphaP_bprp'





##### To load the underlying populations:

# Model 1:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
run_number1 = ''

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory1 + 'periods%s.out' % run_number1)

param_vals_all1 = read_sim_params(loadfiles_directory1 + 'periods%s.out' % run_number1)
sssp_per_sys1, sssp1 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory1, run_number=run_number1)

# Model 2:
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_equal/durations_norm_circ_singles_multis_GF2020_KS/GP_med/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_med/'
run_number2 = ''

param_vals_all2 = read_sim_params(loadfiles_directory2 + 'periods%s.out' % run_number2)
sssp_per_sys2, sssp2 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory2, run_number=run_number2)



model_sssp = [sssp1, sssp2]
model_sssp_per_sys = [sssp_per_sys1, sssp_per_sys2]
model_names = ['Maximum AMD model', 'Two-Rayleigh model'] #['Distribute AMD per mass', 'Distribute AMD equal'] #['Maximum AMD model', 'Two-Rayleigh model'] #[r'Linear $\alpha_P(b_p-r_p-E^*)$', r'Linear $f_{\rm swpa}(b_p-r_p-E^*)$'] #['Maximum AMD model', 'Two-Rayleigh model (Paper II)'] # Make sure this matches the models loaded!
model_linestyles = ['-', '--']
model_colors = ['g', 'b']
model_stagger_errorbars = [-0.05, 0.05] # offsets for plotting multiplicity counts in order to stagger errorbars





##### To plot the simulated catalog as marginal distributions:

subdirectory = '' #'Paper_Figures/Models/Underlying/Clustered_P_R/' #'Paper_Figures/'; 'Talk_Figures/'

fig_size = (8,3) #size of each panel (figure)
fig_lbrt = [0.15, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size





##### To load and compute the same statistics for a large number of models, computing the confidence intervals for each bin:

loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_equal/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/' #'/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/Split_stars/Clustered_P_R_fswp_bprp/Params13_KS/durations_KS/GP_best_models/'

model_loadfiles_dirs = [loadfiles_directory1, loadfiles_directory2]
models = len(model_loadfiles_dirs)

runs = 100

f_pairs_mmr_all = []
f_mmr_all = []

Mtot_bins = np.arange(23)-1.5 #include a -1 bin with all zeros for plotting purposes
Mtot_bins_mid = (Mtot_bins[:-1] + Mtot_bins[1:])/2.
Mtot_counts_all = []
Mtot_earth_counts_all = []

clustertot_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
clustertot_bins_mid = (clustertot_bins[:-1] + clustertot_bins[1:])/2.
clustertot_counts_all = []

pl_per_cluster_bins = np.arange(12)-0.5 #includes 0 bin but will not count them
pl_per_cluster_bins_mid = (pl_per_cluster_bins[:-1] + pl_per_cluster_bins[1:])/2.
pl_per_cluster_counts_all = []

P_bins = np.logspace(np.log10(P_min), np.log10(P_max), n_bins+1)
P_bins_mid = (P_bins[:-1] + P_bins[1:])/2.
P_counts_all = []

Rm_bins = np.logspace(np.log10(1.), np.log10(20.), n_bins+1)
Rm_bins_mid = (Rm_bins[:-1] + Rm_bins[1:])/2.
Rm_counts_all = []

e_bins = np.logspace(np.log10(3e-4), 0., n_bins+1)
e_bins_mid = (e_bins[:-1] + e_bins[1:])/2.
e_counts_all = []
e1_counts_all = [] # singles only
e2p_counts_all = [] # multis only

im_bins = np.logspace(-2., np.log10(180.), n_bins+1)
im_bins_mid = (im_bins[:-1] + im_bins[1:])/2.
im_counts_all = []

mass_bins = np.logspace(np.log10(0.09), 2., n_bins+1)
mass_bins_mid = (mass_bins[:-1] + mass_bins[1:])/2.
mass_counts_all = []

radii_bins = np.logspace(np.log10(radii_min), np.log10(radii_max), n_bins+1)
radii_bins_mid = (radii_bins[:-1] + radii_bins[1:])/2.
radii_counts_all = []

radii_ratio_bins = np.logspace(-1., 1., n_bins+1)
radii_ratio_bins_mid = (radii_ratio_bins[:-1] + radii_ratio_bins[1:])/2.
radii_ratio_counts_all = []

N_mH_bins = np.logspace(0., np.log10(200.), n_bins+1)
N_mH_bins_mid = (N_mH_bins[:-1] + N_mH_bins[1:])/2.
N_mH_counts_all = []

Rstar_bins = np.linspace(0.5, 2.5, n_bins+1)
Rstar_bins_mid = (Rstar_bins[:-1] + Rstar_bins[1:])/2.
Rstar_counts_all = []

dynamical_mass_bins = np.logspace(np.log10(2e-7), -3., n_bins+1)
dynamical_mass_bins_mid = (dynamical_mass_bins[:-1] + dynamical_mass_bins[1:])/2.
dynamical_mass_counts_all = []

radii_partitioning_bins = np.logspace(-5., 0., n_bins+1)
radii_partitioning_bins_mid = (radii_partitioning_bins[:-1] + radii_partitioning_bins[1:])/2.
radii_partitioning_counts_all = []

radii_monotonicity_bins = np.linspace(-0.6, 0.6, n_bins+1)
radii_monotonicity_bins_mid = (radii_monotonicity_bins[:-1] + radii_monotonicity_bins[1:])/2.
radii_monotonicity_counts_all = []

gap_complexity_bins = np.linspace(0., 1., n_bins+1)
gap_complexity_bins_mid = (gap_complexity_bins[:-1] + gap_complexity_bins[1:])/2.
gap_complexity_counts_all = []

# To also store median e and im per multiplicity for power-law fitting:
n_array = np.arange(2,11)
log_n = np.log10(n_array)
e_med_1_all = []
e_med_n_all = []
im_med_n_all = []

for loadfiles_dir in model_loadfiles_dirs:
    f_pairs_mmr = [] # fraction of planet pairs that are near an MMR for each catalog
    f_mmr = [] # fraction of planets that are near an MMR for each catalog
    N_sys_with_mmr = [] # number of systems with at least one planet pair near MMR for each catalog
    N_sys_1p = [] # number of systems with at least one planet for each catalog
    N_sys_2p = [] # number of systems with at least two planets for each catalog
    f_high = [] ##### for Two-Rayleigh model only ##### fraction of systems assigned to high inclinations for each catalog (from sim param)
    
    Mtot_counts = []
    Mtot_earth_counts = []
    clustertot_counts = []
    pl_per_cluster_counts = []
    P_counts = []
    Rm_counts = []
    e_counts = []
    e1_counts = []
    e2p_counts = []
    im_counts = []
    mass_counts = []
    radii_counts = []
    radii_ratio_counts = []
    N_mH_counts = []
    Rstar_counts = []
    
    dynamical_mass_counts = []
    radii_partitioning_counts = []
    radii_monotonicity_counts = []
    gap_complexity_counts = []
    
    e_med_1 = np.zeros(runs)
    e_med_n = np.zeros((runs, len(n_array)))
    im_med_n = np.zeros((runs, len(n_array)))

    for i in range(runs): #range(1,runs+1)
        run_number = i+1
        print(i)
        N_sim_i = read_targets_period_radius_bounds(loadfiles_dir + 'periods%s.out' % run_number)[0]
        param_vals_i = read_sim_params(loadfiles_dir + 'periods%s.out' % run_number)
        sssp_per_sys_i, sssp_i = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_dir, run_number=run_number)
        
        #####f_high.append(param_vals_i['f_high_incl']) ##### for Two-Rayleigh model only
        
        # Fraction of planet pairs near an MMR:
        '''
        count_mmr = 0
        for mmr in res_ratios:
            count_mmr += np.sum((sssp_i['Rm_all'] > mmr) & (sssp_i['Rm_all'] < mmr*(1.+res_width)))
        f_pairs_mmr.append(float(count_mmr)/len(sssp_i['Rm_all']))
        # Fraction of planets near an MMR:
        count_mmr = 0
        count_sys_with_mmr = 0
        for p_sys in sssp_per_sys_i['P_all']:
            p_sys = p_sys[p_sys > 0]
            mmr_sys = np.zeros(len(p_sys))
            pr_sys = p_sys[1:]/p_sys[:-1]
            pr_mmr_sys = np.zeros(len(pr_sys))
            
            for mmr in res_ratios:
                pr_mmr_sys[(pr_sys >= mmr) & (pr_sys <= mmr*(1.+res_width))] = 1
            for j,res in enumerate(pr_mmr_sys):
                if res == 1:
                    mmr_sys[j] = 1
                    mmr_sys[j+1] = 1
            count_mmr += np.sum(mmr_sys == 1)
            count_sys_with_mmr += any(mmr_sys)
        f_mmr.append(float(count_mmr)/len(sssp_i['P_all']))
        N_sys_with_mmr.append(count_sys_with_mmr)
        N_sys_1p.append(np.sum(sssp_per_sys_i['Mtot_all'] > 0))
        N_sys_2p.append(np.sum(sssp_per_sys_i['Mtot_all'] > 1))
        '''
        
        # Multiplicities:
        counts, bins = np.histogram(sssp_per_sys_i['Mtot_all'], bins=Mtot_bins)
        counts[1] = N_sim_i - len(sssp_per_sys_i['Mtot_all'])
        Mtot_counts.append(counts/float(np.sum(counts)))
        
        # Multiplicities for Earth-sized planets:
        Earth_bools_per_sys = (sssp_per_sys_i['radii_all'] > 0.75) & (sssp_per_sys_i['radii_all'] < 1.25)
        Earth_counts_per_sys = np.sum(Earth_bools_per_sys, axis=1)
        counts, bins = np.histogram(Earth_counts_per_sys, bins=Mtot_bins)
        counts[1] = N_sim_i - len(Earth_counts_per_sys)
        Mtot_earth_counts.append(counts/float(np.sum(counts)))
        
        # Numbers of clusters:
        counts, bins = np.histogram(sssp_i['clustertot_all'], bins=clustertot_bins)
        clustertot_counts.append(counts/float(np.sum(counts)))
        
        # Numbers of planets per cluster:
        counts, bins = np.histogram(sssp_i['pl_per_cluster_all'], bins=pl_per_cluster_bins)
        pl_per_cluster_counts.append(counts/float(np.sum(counts)))
        
        # Periods:
        counts, bins = np.histogram(sssp_i['P_all'], bins=P_bins)
        P_counts.append(counts/float(np.sum(counts)))
        
        # Period ratios:
        counts, bins = np.histogram(sssp_i['Rm_all'], bins=Rm_bins)
        Rm_counts.append(counts/float(np.sum(counts)))
        
        # Eccentricities:
        counts, bins = np.histogram(sssp_i['e_all'], bins=e_bins)
        e_counts.append(counts/float(np.sum(counts)))
        # Singles only:
        e1 = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == 1, 0]
        counts1, bins = np.histogram(e1, bins=e_bins)
        e1_counts.append(counts1/float(np.sum(counts))) # normalize counts to all planets
        # Multis only:
        e2p = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] > 1]
        e2p = e2p[sssp_per_sys_i['P_all'][sssp_per_sys_i['Mtot_all'] > 1] > 0]
        counts2p, bins = np.histogram(e2p, bins=e_bins)
        e2p_counts.append(counts2p/float(np.sum(counts))) # normalize counts to all planets
        
        # Mutual inclinations:
        counts, bins = np.histogram(sssp_i['inclmut_all']*(180./np.pi), bins=im_bins)
        im_counts.append(counts/float(np.sum(counts)))
        
        # Planet masses:
        counts, bins = np.histogram(sssp_i['mass_all'], bins=mass_bins)
        mass_counts.append(counts/float(np.sum(counts)))
        
        # Planet radii:
        counts, bins = np.histogram(sssp_i['radii_all'], bins=radii_bins)
        radii_counts.append(counts/float(np.sum(counts)))
        
        # Planet radii ratios:
        counts, bins = np.histogram(sssp_i['radii_ratio_all'], bins=radii_ratio_bins)
        radii_ratio_counts.append(counts/float(np.sum(counts)))
        
        # Separations:
        counts, bins = np.histogram(sssp_i['N_mH_all'], bins=N_mH_bins)
        N_mH_counts.append(counts/float(np.sum(counts)))
        
        # Stellar radii:
        counts, bins = np.histogram(sssp_i['Rstar_all'], bins=Rstar_bins)
        Rstar_counts.append(counts/float(np.sum(counts)))
        
        # Dynamical masses:
        counts, bins = np.histogram(sssp_per_sys_i['dynamical_mass'], bins=dynamical_mass_bins)
        dynamical_mass_counts.append(counts/float(np.sum(counts)))
        
        # Planet radii partitioning:
        counts, bins = np.histogram(sssp_per_sys_i['radii_partitioning'], bins=radii_partitioning_bins)
        radii_partitioning_counts.append(counts/float(np.sum(counts)))
        
        # Planet radii monotonicity:
        counts, bins = np.histogram(sssp_per_sys_i['radii_monotonicity'], bins=radii_monotonicity_bins)
        radii_monotonicity_counts.append(counts/float(np.sum(counts)))
        
        # Gap complexity:
        counts, bins = np.histogram(sssp_per_sys_i['gap_complexity'], bins=gap_complexity_bins)
        gap_complexity_counts.append(counts/float(np.sum(counts)))
    
        # Median eccentricity and mutual inclination per multiplicity:
        e_1 = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == 1,0]
        e_med_1[i] = np.median(e_1)
        for j,n in enumerate(n_array):
            e_n = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
            e_n = e_n.flatten()
            im_n = sssp_per_sys_i['inclmut_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
            im_n = im_n.flatten() * (180./np.pi)
            
            e_med_n[i,j] = np.median(e_n)
            im_med_n[i,j] = np.median(im_n)

    Mtot_counts_all.append(np.array(Mtot_counts))
    Mtot_earth_counts_all.append(np.array(Mtot_earth_counts))
    clustertot_counts_all.append(np.array(clustertot_counts))
    pl_per_cluster_counts_all.append(np.array(pl_per_cluster_counts))
    P_counts_all.append(np.array(P_counts))
    Rm_counts_all.append(np.array(Rm_counts))
    e_counts_all.append(np.array(e_counts))
    e1_counts_all.append(np.array(e1_counts))
    e2p_counts_all.append(np.array(e2p_counts))
    im_counts_all.append(np.array(im_counts))
    mass_counts_all.append(np.array(mass_counts))
    radii_counts_all.append(np.array(radii_counts))
    radii_ratio_counts_all.append(np.array(radii_ratio_counts))
    N_mH_counts_all.append(np.array(N_mH_counts))
    Rstar_counts_all.append(np.array(Rstar_counts))

    dynamical_mass_counts_all.append(np.array(dynamical_mass_counts))
    radii_partitioning_counts_all.append(np.array(radii_partitioning_counts))
    radii_monotonicity_counts_all.append(np.array(radii_monotonicity_counts))
    gap_complexity_counts_all.append(np.array(gap_complexity_counts))

    e_med_1_all.append(e_med_1)
    e_med_n_all.append(e_med_n)
    im_med_n_all.append(im_med_n)

f_pairs_mmr_all = np.array(f_pairs_mmr_all)
f_mmr_all = np.array(f_mmr_all)

Mtot_counts_all = np.array(Mtot_counts_all)
Mtot_earth_counts_all = np.array(Mtot_earth_counts_all)
clustertot_counts_all = np.array(clustertot_counts_all)
pl_per_cluster_counts_all = np.array(pl_per_cluster_counts_all)
P_counts_all = np.array(P_counts_all)
Rm_counts_all = np.array(Rm_counts_all)
e_counts_all = np.array(e_counts_all)
e1_counts_all = np.array(e1_counts_all)
e2p_counts_all = np.array(e2p_counts_all)
im_counts_all = np.array(im_counts_all)
mass_counts_all = np.array(mass_counts_all)
radii_counts_all = np.array(radii_counts_all)
radii_ratio_counts_all = np.array(radii_ratio_counts_all)
N_mH_counts_all = np.array(N_mH_counts_all)
Rstar_counts_all = np.array(Rstar_counts_all)

dynamical_mass_counts_all = np.array(dynamical_mass_counts_all)
radii_partitioning_counts_all = np.array(radii_partitioning_counts_all)
radii_monotonicity_counts_all = np.array(radii_monotonicity_counts_all)
gap_complexity_counts_all = np.array(gap_complexity_counts_all)





Mtot_counts_qtls = [np.zeros((len(Mtot_bins_mid),3)) for m in range(models)]
clustertot_counts_qtls = [np.zeros((len(clustertot_bins_mid),3)) for m in range(models)]
pl_per_cluster_counts_qtls = [np.zeros((len(pl_per_cluster_bins_mid),3)) for m in range(models)]

P_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
Rm_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
e_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
e1_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
e2p_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
im_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
mass_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
radii_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
radii_ratio_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
N_mH_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
Rstar_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]

dynamical_mass_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
radii_partitioning_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
radii_monotonicity_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]
gap_complexity_counts_qtls = [np.zeros((n_bins,3)) for m in range(models)]

for m in range(models):
    for b in range(len(Mtot_bins_mid)):
        counts_bin_sorted = np.sort(Mtot_counts_all[m][:,b])
        Mtot_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
    for b in range(len(clustertot_bins_mid)):
        counts_bin_sorted = np.sort(clustertot_counts_all[m][:,b])
        clustertot_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])
    for b in range(len(pl_per_cluster_bins_mid)):
        counts_bin_sorted = np.sort(pl_per_cluster_counts_all[m][:,b])
        pl_per_cluster_counts_qtls[m][b] = np.quantile(counts_bin_sorted, [0.16, 0.5, 0.84])

    for b in range(n_bins):
        # Periods:
        P_counts_qtls[m][b] = np.quantile(P_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Period ratios:
        Rm_counts_qtls[m][b] = np.quantile(Rm_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Eccentricities:
        e_counts_qtls[m][b] = np.quantile(e_counts_all[m][:,b], [0.16, 0.5, 0.84])
        e1_counts_qtls[m][b] = np.quantile(e1_counts_all[m][:,b], [0.16, 0.5, 0.84])
        e2p_counts_qtls[m][b] = np.quantile(e2p_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Mutual inclinations:
        im_counts_qtls[m][b] = np.quantile(im_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Planet masses:
        mass_counts_qtls[m][b] = np.quantile(mass_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Planet radii:
        radii_counts_qtls[m][b] = np.quantile(radii_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Planet radii ratios:
        radii_ratio_counts_qtls[m][b] = np.quantile(radii_ratio_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Separations:
        N_mH_counts_qtls[m][b] = np.quantile(N_mH_counts_all[m][:,b], [0.16, 0.5, 0.84])
        
        # Stellar radii:
        Rstar_counts_qtls[m][b] = np.quantile(Rstar_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Dynamical masses:
        dynamical_mass_counts_qtls[m][b] = np.quantile(dynamical_mass_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Planet radii partitioning:
        radii_partitioning_counts_qtls[m][b] = np.quantile(radii_partitioning_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Planet radii monotonicity:
        radii_monotonicity_counts_qtls[m][b] = np.quantile(radii_monotonicity_counts_all[m][:,b], [0.16, 0.5, 0.84])

        # Gap complexity:
        gap_complexity_counts_qtls[m][b] = np.quantile(gap_complexity_counts_all[m][:,b], [0.16, 0.5, 0.84])

#####





#'''
# Multiplicities:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(Mtot_bins_mid, Mtot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(Mtot_bins_mid + model_stagger_errorbars[m], Mtot_counts_qtls[m][:,1], yerr=[Mtot_counts_qtls[m][:,1]-Mtot_counts_qtls[m][:,0], Mtot_counts_qtls[m][:,2]-Mtot_counts_qtls[m][:,1]], fmt='none', color=model_colors[m], lw=lw, label='') #capsize=5 #label=r'16% and 84%' if m==0 else ''
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([-0.5, 10.5]) #[0, np.max(x)]
plt.ylim([0., 0.6])
plt.xlabel('Intrinsic planet multiplicity', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_multiplicities.pdf')
    plt.close()

# Number of clusters:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(clustertot_bins_mid, clustertot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(clustertot_bins_mid + model_stagger_errorbars[m], clustertot_counts_qtls[m][:,1], yerr=[clustertot_counts_qtls[m][:,1]-clustertot_counts_qtls[m][:,0], clustertot_counts_qtls[m][:,2]-clustertot_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 5.5])
plt.ylim([0., 1.])
plt.xlabel(r'Clusters per system $N_c$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_clusters.pdf')
    plt.close()

# Number of planets per cluster:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(pl_per_cluster_bins_mid + model_stagger_errorbars[m], pl_per_cluster_counts_qtls[m][:,1], yerr=[pl_per_cluster_counts_qtls[m][:,1]-pl_per_cluster_counts_qtls[m][:,0], pl_per_cluster_counts_qtls[m][:,2]-pl_per_cluster_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 7.5])
plt.ylim([0., 0.7])
plt.xlabel(r'Planets per cluster $N_p$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_planets_per_cluster.pdf')
    plt.close()

# Periods:
plot_fig_pdf_simple(fig_size, [sssp['P_all'] for sssp in model_sssp], [], x_min=P_min, x_max=P_max, n_bins=n_bins, log_x=True, log_y=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    label_this = r'16%-84%' if m==0 else ''
    plt.fill_between(P_bins_mid, P_counts_qtls[m][:,0], P_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha, label=label_this)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_periods.pdf')
    plt.close()

# Period ratios (all):
plot_fig_pdf_simple(fig_size, [sssp['Rm_all'] for sssp in model_sssp], [], x_min=Rm_bins[0], x_max=Rm_bins[-1], y_max=0.07, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(Rm_bins_mid, Rm_counts_qtls[m][:,0], Rm_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
plt.minorticks_off()
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_periodratios.pdf')
    plt.close()

# Eccentricities:
#plot_fig_pdf_simple(fig_size, [sssp['e_all'] for sssp in model_sssp], [], x_min=e_bins[0], x_max=e_bins[-1], log_x=True, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$e$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
#for m in range(models):
#    plt.fill_between(e_bins_mid, e_counts_qtls[m][:,0], e_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
#'''
plot_fig_pdf_simple(fig_size, [sssp2['e_all']], [], x_min=e_bins[0], x_max=e_bins[-1], log_x=True, n_bins=n_bins, c_sim=[model_colors[1]], lw=lw, ls_sim=[model_linestyles[1]], labels_sim=[''], xlabel_text=r'$e$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
plt.fill_between(e_bins_mid, e_counts_qtls[1][:,0], e_counts_qtls[1][:,2], color=model_colors[1], alpha=alpha)
e1 = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] == 1, 0]
e2p = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] > 1]
e2p = e2p[sssp_per_sys1['P_all'][sssp_per_sys1['Mtot_all'] > 1] > 0]
plt.hist(e1, bins=e_bins, histtype='step', weights=np.ones(len(e1))/len(sssp1['e_all']), color=model_colors[0], ls=':', lw=lw, label='Singles')
plt.hist(e2p, bins=e_bins, histtype='step', weights=np.ones(len(e2p))/len(sssp1['e_all']), color=model_colors[0], ls=model_linestyles[0], lw=lw, label='Multis')
plt.fill_between(e_bins_mid, e1_counts_qtls[0][:,0], e1_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
plt.fill_between(e_bins_mid, e2p_counts_qtls[0][:,0], e2p_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)
#'''
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_eccentricities.pdf')
    plt.close()

# Mutual inclinations:
plot_fig_pdf_simple(fig_size, [sssp['inclmut_all']*(180./np.pi) for sssp in model_sssp], [], x_min=im_bins[0], x_max=im_bins[-1], y_max=0.06, log_x=True, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$i_m$ (deg)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(im_bins_mid, im_counts_qtls[m][:,0], im_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_mutualinclinations.pdf')
    plt.close()

# Planet masses:
plot_fig_pdf_simple(fig_size, [sssp['mass_all'] for sssp in model_sssp], [], x_min=mass_bins[0], x_max=mass_bins[-1], n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$M_p$ ($M_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(mass_bins_mid, mass_counts_qtls[m][:,0], mass_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_masses.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple(fig_size, [sssp['radii_all'] for sssp in model_sssp], [], x_min=radii_min, x_max=radii_max, y_max=0.025, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_bins_mid, radii_counts_qtls[m][:,0], radii_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii.pdf')
    plt.close()

# Planet radii ratios:
plot_fig_pdf_simple(fig_size, [sssp['radii_ratio_all'] for sssp in model_sssp], [], x_min=radii_ratio_bins[0], x_max=radii_ratio_bins[-1], y_max=0.06, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_{p,i+1}/R_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_ratio_bins_mid, radii_ratio_counts_qtls[m][:,0], radii_ratio_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_ratios.pdf')
    plt.close()

# Separations in mutual Hill radii:
plot_fig_pdf_simple(fig_size, [sssp['N_mH_all'] for sssp in model_sssp], [], x_min=N_mH_bins[0], x_max=N_mH_bins[-1], y_max=0.05, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\Delta$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(N_mH_bins_mid, N_mH_counts_qtls[m][:,0], N_mH_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_deltas.pdf')
    plt.close()

# Stellar radii:
plot_fig_pdf_simple(fig_size, [sssp['Rstar_all'] for sssp in model_sssp], [], x_min=Rstar_bins[0], x_max=Rstar_bins[-1], n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_\star (R_\odot)$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(Rstar_bins_mid, Rstar_counts_qtls[m][:,0], Rstar_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_stellar_radii.pdf')
    plt.close()

### GF2020 metrics, but for the underlying systems:
# Dynamical masses CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['dynamical_mass'] for sssp_per_sys in model_sssp_per_sys], [], x_min=dynamical_mass_bins[0], x_max=dynamical_mass_bins[-1], y_max=0.05, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mu$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(dynamical_mass_bins_mid, dynamical_mass_counts_qtls[m][:,0], dynamical_mass_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_dynamical_masses.pdf')
    plt.close()

# Planet radii partitioning CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_partitioning'] for sssp_per_sys in model_sssp_per_sys], [], x_min=radii_partitioning_bins[0], x_max=radii_partitioning_bins[-1], y_max=0.05, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mathcal{Q}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_partitioning_bins_mid, radii_partitioning_counts_qtls[m][:,0], radii_partitioning_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_partitioning.pdf')
    plt.close()

# Planet radii monotonicity CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['radii_monotonicity'] for sssp_per_sys in model_sssp_per_sys], [], x_min=radii_monotonicity_bins[0], x_max=radii_monotonicity_bins[-1], y_max=0.03, n_bins=n_bins, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mathcal{M}_R$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_monotonicity_bins_mid, radii_monotonicity_counts_qtls[m][:,0], radii_monotonicity_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_radii_monotonicity.pdf')
    plt.close()

# Gap complexity CDFs:
plot_fig_pdf_simple(fig_size, [sssp_per_sys['gap_complexity'] for sssp_per_sys in model_sssp_per_sys], [], x_min=gap_complexity_bins[0], x_max=gap_complexity_bins[-1], y_max=0.06, n_bins=n_bins, log_x=False, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$\mathcal{C}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(gap_complexity_bins_mid, gap_complexity_counts_qtls[m][:,0], gap_complexity_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_gap_complexity.pdf')
    plt.close()

plt.show()
plt.close()
#'''





##### To compute some occurrence rates statistics:
'''
# Fraction of stars with no planets/at least 1 planet:
fswnp, fswnp_16, fswnp_84 = np.median(Mtot_counts_all[:,0]), Mtot_counts_16[0], Mtot_counts_84[0]
fswp, fswp_16, fswp_84 = 1.-fswnp, 1.-fswnp_84, 1.-fswnp_16

# Mean number of planets per star:
pl_tot_all = np.sum((Mtot_counts_all*N_sim_i)*np.arange(len(Mtot_bins_mid)), axis=1)
pl_per_star_all = np.sort(pl_tot_all/np.float(N_sim_i))
mps, mps_16, mps_84 = np.median(pl_per_star_all), pl_per_star_all[16], pl_per_star_all[84]

# Mean number of planets per planetary system (i.e. stars with at least 1 planet):
n_1plus_all = N_sim_i - (Mtot_counts_all*N_sim_i)[:,0]
pl_per_1plus_all = np.sort(pl_tot_all/n_1plus_all)
mpps, mpps_16, mpps_84 = np.median(pl_per_1plus_all), pl_per_1plus_all[16], pl_per_1plus_all[84]

# Fraction of stars with Earth-sized planets:
fswne, fswne_16, fswne_84 = np.median(Mtot_earth_counts_all[:,0]), np.sort(Mtot_earth_counts_all[:,0])[16], np.sort(Mtot_earth_counts_all[:,0])[84]
fswe, fswe_16, fswe_84 = 1.-fswne, 1.-fswne_84, 1.-fswne_16

# Mean number of Earth-sized planets per star:
ep_tot_all = np.sum((Mtot_earth_counts_all*N_sim_i)*np.arange(len(Mtot_bins_mid)), axis=1)
ep_per_star_all = np.sort(ep_tot_all/np.float(N_sim_i))
meps, meps_16, meps_84 = np.median(ep_per_star_all), ep_per_star_all[16], ep_per_star_all[84]

# Mean number of Earth-sized planets per planetary system (i.e. stars with at least 1 planet):
ep_per_1plus_all = np.sort(ep_tot_all/n_1plus_all)
mepps, mepps_16, mepps_84 = np.median(ep_per_1plus_all), ep_per_1plus_all[16], ep_per_1plus_all[84]

# Fraction of planetary systems with at least one Earth-sized planet (i.e. probability that a planet-hosting star contains an Earth-sized planet):
fswpwe_all = np.sort((1.-Mtot_earth_counts_all[:,0])/(1.-Mtot_counts_all[:,0]))
fswpwe, fswpwe_16, fswpwe_84 = np.median(fswpwe_all), fswpwe_all[16], fswpwe_all[84]
'''





##### To plot eccentricity vs mutual inclinations, with attached histograms:

persys_1d_1, perpl_1d_1 = convert_underlying_properties_per_planet_1d(sssp_per_sys1, sssp1)
persys_1d_2, perpl_1d_2 = convert_underlying_properties_per_planet_1d(sssp_per_sys2, sssp2)

ecc_min_max, incl_min_max = [e_bins[0], e_bins[-1]], [im_bins[0], im_bins[-1]]

fig = plt.figure(figsize=(16,9))
plot = GridSpec(5, 9, left=0.1, bottom=0.1, right=0.975, top=0.975, wspace=0, hspace=0)

ax = plt.subplot(plot[1:,:4]) # scatter i_m vs ecc (model 1)
corner.hist2d(np.log10(perpl_1d_1['e_all']), np.log10(perpl_1d_1['im_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([-3., -2., -1., 0.])
ytick_vals = np.array([-2., -1., 0., 1., 2.])
plt.xticks(xtick_vals, 10.**xtick_vals)
plt.yticks(ytick_vals, 10.**ytick_vals)
plt.xlim(np.log10(np.array(ecc_min_max)))
plt.ylim(np.log10(np.array(incl_min_max)))
plt.xlabel(r'$e$', fontsize=tfs)
plt.ylabel(r'$i_m$ ($^\circ$)', fontsize=tfs)
ax.text(x=0.95, y=0.05, s=model_names[0], ha='right', fontsize=tfs, color=model_colors[0], transform=ax.transAxes)

ax = plt.subplot(plot[0,:4]) # top histogram of ecc (model 1)
e1 = sssp_per_sys1['e_all'][sssp_per_sys1['Mtot_all'] == 1, 0]
#plt.hist(sssp1['e_all'], bins=e_bins, weights=np.ones(len(sssp1['e_all']))/len(sssp1['e_all']), histtype='step', color=model_colors[0], ls=model_linestyles[0], lw=lw)
#plt.fill_between(e_bins_mid, e_counts_qtls[0][:,0], e_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
#'''
plt.hist(e1, bins=e_bins, histtype='step', weights=np.ones(len(e1))/len(sssp1['e_all']), color=model_colors[0], ls=':', lw=lw, label=r'$n = 1$')
plt.hist(perpl_1d_1['e_all'], bins=e_bins, histtype='step', weights=np.ones(len(e2p))/len(sssp1['e_all']), color=model_colors[0], ls=model_linestyles[0], lw=lw, label=r'$n \geq 2$')
plt.fill_between(e_bins_mid, e1_counts_qtls[0][:,0], e1_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
plt.fill_between(e_bins_mid, e2p_counts_qtls[0][:,0], e2p_counts_qtls[0][:,2], color=model_colors[0], alpha=alpha)
#'''
plt.gca().set_xscale("log")
plt.xticks(10.**xtick_vals, [])
plt.yticks([])
plt.xlim(np.array(ecc_min_max))
plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1:,4:8]) # scatter i_m vs ecc (model 2)
corner.hist2d(np.log10(perpl_1d_2['e_all']), np.log10(perpl_1d_2['im_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
xtick_vals = np.array([-3., -2., -1., 0.])
plt.xticks(xtick_vals, 10.**xtick_vals)
plt.yticks([])
plt.xlim(np.log10(np.array(ecc_min_max)))
plt.ylim(np.log10(np.array(incl_min_max)))
plt.xlabel(r'$e$', fontsize=tfs)
ax.text(x=0.95, y=0.05, s=model_names[1], ha='right', fontsize=tfs, color=model_colors[1], transform=ax.transAxes)

ax = plt.subplot(plot[0,4:8]) # top histogram of ecc (model 2)
plt.hist(sssp2['e_all'], bins=e_bins, weights=np.ones(len(sssp2['e_all']))/len(sssp2['e_all']), histtype='step', color=model_colors[1], ls=model_linestyles[1], lw=lw)
plt.fill_between(e_bins_mid, e_counts_qtls[1][:,0], e_counts_qtls[1][:,2], color=model_colors[1], alpha=alpha)
plt.gca().set_xscale("log")
plt.xticks(10.**xtick_vals, [])
plt.yticks([])
plt.xlim(np.array(ecc_min_max))

ax = plt.subplot(plot[1:,8]) # side histogram of i_m (model 1+2)
plt.hist(perpl_1d_1['im_all'], bins=im_bins, weights=np.ones(len(perpl_1d_1['im_all']))/len(perpl_1d_1['im_all']), histtype='step', orientation='horizontal', color=model_colors[0], ls=model_linestyles[0], lw=lw)
plt.hist(perpl_1d_2['im_all'], bins=im_bins, weights=np.ones(len(perpl_1d_2['im_all']))/len(perpl_1d_2['im_all']), histtype='step', orientation='horizontal', color=model_colors[1], ls=model_linestyles[1], lw=lw)
for m in range(models):
    plt.fill_betweenx(im_bins_mid, im_counts_qtls[m][:,0], im_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
plt.gca().set_yscale("log")
plt.xticks([])
plt.yticks(10.**ytick_vals, [])
plt.ylim(np.array(incl_min_max))

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_underlying_ecc_vs_incl.pdf')
    plt.close()
plt.show()





##### To fit power-laws to the median eccentricities and mutual inclinations as a function of intrinsic multiplicity n:
##### NOTE: We are actually fitting a line to log(e) and log(im) vs log(n)

n_norm = 5 # n for normalization constant
f_linear = lambda p, x: p[0] + p[1]*x - p[1]*np.log10(n_norm) # extra term for normalizing
f_err = lambda p, x, y: y - f_linear(p,x)
e_p0 = [0.03, -2.] # initial guesses for log10(mu_e) and alpha_e
im_p0 = [1., -2.] # initial guesses for log10(mu_im) and alpha_im

mu_e_all, alpha_e_all = [np.zeros(runs), np.zeros(runs)], [np.zeros(runs), np.zeros(runs)]
mu_im_all, alpha_im_all = [np.zeros(runs), np.zeros(runs)], [np.zeros(runs), np.zeros(runs)]
for m in range(models):
    for i in range(runs):
        log_e_med = np.log10(e_med_n_all[m][i])
        log_im_med = np.log10(im_med_n_all[m][i])
        
        e_fit = scipy.optimize.leastsq(f_err, e_p0, args=(log_n, log_e_med), full_output=1)
        e_logmu_alpha = e_fit[0]
        mu_e, alpha_e = 10.**(e_logmu_alpha[0]), e_logmu_alpha[1]
        mu_e_all[m][i], alpha_e_all[m][i] = mu_e, alpha_e
        
        im_fit = scipy.optimize.leastsq(f_err, im_p0, args=(log_n, log_im_med), full_output=1)
        im_logmu_alpha = im_fit[0]
        mu_im, alpha_im = 10.**(im_logmu_alpha[0]), im_logmu_alpha[1]
        mu_im_all[m][i], alpha_im_all[m][i] = mu_im, alpha_im





##### To plot median eccentricities and mutual inclinations vs. intrinsic multiplicity n, along with power-law fits, and for models with other values of f_amd_crit:

def compute_power_law_at_n_quantiles(n_array, mu_all, alpha_all, n_norm=5, qtl=[0.16, 0.5, 0.84]):
    assert len(mu_all) == len(alpha_all)
    power_law_n_all = np.zeros((len(mu_all), len(n_array)))
    for i in range(len(mu_all)):
        power_law_n_all[i,:] = mu_all[i] * (n_array/n_norm)**alpha_all[i]
    return np.quantile(power_law_n_all, qtl, axis=0)

# To load other catalogs with different values of f_amd_crit:
loadfiles_directory = '../../ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/f_amd_crit_all/Params12_KS/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
sssp_per_sys_0p5, sssp_0p5 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory + 'f_amd_crit_0p5/')
sssp_per_sys_2, sssp_2 = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory + 'f_amd_crit_2/')

e_med_n_fcrits = np.zeros((2, len(n_array)))
im_med_n_fcrits = np.zeros((2, len(n_array)))
for i,sssp_per_sys_i in enumerate([sssp_per_sys_0p5, sssp_per_sys_2]):
    # Median eccentricity and mutual inclination per multiplicity:
    for j,n in enumerate(n_array):
        e_n = sssp_per_sys_i['e_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
        e_n = e_n.flatten()
        im_n = sssp_per_sys_i['inclmut_all'][sssp_per_sys_i['Mtot_all'] == n,:n]
        im_n = im_n.flatten() * (180./np.pi)
        
        e_med_n_fcrits[i,j] = np.median(e_n)
        im_med_n_fcrits[i,j] = np.median(im_n)



fig = plt.figure(figsize=(8,10))
plot = GridSpec(2,1,left=0.15,bottom=0.1,right=0.98,top=0.98,wspace=0,hspace=0)

ax = plt.subplot(plot[0,0]) # ecc vs n
e_med_1_qtls = np.quantile(e_med_1_all[0], [0.16,0.5,0.84])
e_med_n_qtls = np.quantile(e_med_n_all[0], [0.16,0.5,0.84], axis=0)
e_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[0], alpha_e_all[0], n_norm=n_norm)
e_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[1], alpha_e_all[1], n_norm=n_norm)
plt.errorbar(n_array, e_med_n_qtls[1,:], yerr=[e_med_n_qtls[1,:]-e_med_n_qtls[0,:], e_med_n_qtls[2,:]-e_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='Maximum AMD model')
plt.errorbar([1], [e_med_1_qtls[1]], yerr=[[e_med_1_qtls[1]-e_med_1_qtls[0]], [e_med_1_qtls[2]-e_med_1_qtls[1]]], fmt='o', capsize=5, ls='none', color='c', label='')
plt.plot(n_array, e_med_n_fcrits[0,:], 'o', lw=lw, color='k', label=r'$f_{\rm crit} = 0.5$')
plt.plot(n_array, e_med_n_fcrits[1,:], 'o', lw=lw, color='r', label=r'$f_{\rm crit} = 2$')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model1[0,:], e_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.031_{-0.003}^{+0.004}$, $\alpha_e = -1.74_{-0.07}^{+0.10}$')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model2[0,:], e_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.040_{-0.002}^{+0.003}$, $\alpha_e = -1.54_{-0.11}^{+0.10}$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.yticks([0.01, 0.1, 1.])
plt.xlim([0.5, 10.5])
plt.ylim([0.005, 1.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.ylabel('Median $e$', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[1,0]) # incl vs n
im_med_n_qtls = np.quantile(im_med_n_all[0], [0.16,0.5,0.84], axis=0)
im_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(n_array, mu_im_all[0], alpha_im_all[0], n_norm=n_norm)
im_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(n_array, mu_im_all[1], alpha_im_all[1], n_norm=n_norm)
plt.errorbar(n_array, im_med_n_qtls[1,:], yerr=[im_med_n_qtls[1,:]-im_med_n_qtls[0,:], im_med_n_qtls[2,:]-im_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='')
plt.plot(n_array, im_med_n_fcrits[0,:], 'o', lw=lw, color='k', label='')
plt.plot(n_array, im_med_n_fcrits[1,:], 'o', lw=lw, color='r', label='')
plt.fill_between(n_array, im_plfit_n_qtls_model1[0,:], im_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.10_{-0.11}^{+0.15}$, $\alpha_i = -1.73_{-0.08}^{+0.09}$')
plt.fill_between(n_array, im_plfit_n_qtls_model2[0,:], im_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.41_{-0.08}^{+0.10}$, $\alpha_i = -1.55_{-0.11}^{+0.11}$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='orange', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -3.5$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.arange(10)+1)
plt.yticks([0.1, 1., 10.])
plt.xlim([0.5, 10.5])
plt.ylim([0.1, 30.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Intrinsic planet multiplicity $n$', fontsize=tfs)
plt.ylabel('Median $i_m$ ($^\circ$)', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:] + [handles[0]]
labels = labels[1:] + [labels[0]]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(savefigures_directory + subdirectory + save_name + '_ecc_incl_vs_mult_fits.pdf')
    plt.close()
plt.show()





##### To remake some figures for proposals:

directory = '/Users/hematthi/Documents/GradSchool/Postdoctoral_Applications/Figures/'

fig_size = (6,3) #size of each panel (figure)
fig_lbrt = [0.2, 0.3, 0.95, 0.925]

n_bins = 100
lw = 3 #linewidth
alpha = 0.2

afs = 20 #axes labels font size
tfs = 20 #text labels font size
lfs = 16 #legend labels font size

# Multiplicities:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(Mtot_bins_mid, Mtot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(Mtot_bins_mid + model_stagger_errorbars[m], Mtot_counts_qtls[m][:,1], yerr=[Mtot_counts_qtls[m][:,1]-Mtot_counts_qtls[m][:,0], Mtot_counts_qtls[m][:,2]-Mtot_counts_qtls[m][:,1]], fmt='none', color=model_colors[m], lw=lw, label='') #capsize=5 #label=r'16% and 84%' if m==0 else ''
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([-0.5, 10.5]) #[0, np.max(x)]
plt.ylim([0., 0.6])
plt.xlabel('Intrinsic planet multiplicity', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs) #show the legend
if savefigures:
    plt.savefig(directory + save_name + '_underlying_multiplicities.pdf')
    plt.close()

# Number of clusters:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(clustertot_bins_mid, clustertot_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(clustertot_bins_mid + model_stagger_errorbars[m], clustertot_counts_qtls[m][:,1], yerr=[clustertot_counts_qtls[m][:,1]-clustertot_counts_qtls[m][:,0], clustertot_counts_qtls[m][:,2]-clustertot_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 5.5])
plt.ylim([0., 1.])
plt.xlabel(r'Clusters per system $N_c$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_clusters.pdf')
    plt.close()

# Number of planets per cluster:
ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
for m in range(models):
    plt.plot(pl_per_cluster_bins_mid, pl_per_cluster_counts_qtls[m][:,1], drawstyle='steps-mid', color=model_colors[m], ls=model_linestyles[m], lw=lw, label=model_names[m])
    plt.errorbar(pl_per_cluster_bins_mid + model_stagger_errorbars[m], pl_per_cluster_counts_qtls[m][:,1], yerr=[pl_per_cluster_counts_qtls[m][:,1]-pl_per_cluster_counts_qtls[m][:,0], pl_per_cluster_counts_qtls[m][:,2]-pl_per_cluster_counts_qtls[m][:,1]], fmt='.', color=model_colors[m], lw=lw, label='')
ax.tick_params(axis='both', labelsize=afs)
plt.xlim([0.5, 7.5])
plt.ylim([0., 0.7])
plt.xlabel(r'Planets per cluster $N_p$', fontsize=tfs)
plt.ylabel('Fraction', fontsize=tfs)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_planets_per_cluster.pdf')
    plt.close()

# Periods:
plot_fig_pdf_simple(fig_size, [sssp['P_all'] for sssp in model_sssp], [], x_min=P_min, x_max=P_max, n_bins=n_bins, log_x=True, log_y=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    label_this = r'16%-84%' if m==0 else ''
    plt.fill_between(P_bins_mid, P_counts_qtls[m][:,0], P_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha, label=label_this)
plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=lfs)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_periods.pdf')
    plt.close()

# Period ratios (all):
plot_fig_pdf_simple(fig_size, [sssp['Rm_all'] for sssp in model_sssp], [], x_min=Rm_bins[0], x_max=Rm_bins[-1], y_max=0.07, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(Rm_bins_mid, Rm_counts_qtls[m][:,0], Rm_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
plt.minorticks_off()
if savefigures:
    plt.savefig(directory + save_name + '_underlying_periodratios.pdf')
    plt.close()

# Planet radii:
plot_fig_pdf_simple(fig_size, [sssp['radii_all'] for sssp in model_sssp], [], x_min=radii_min, x_max=radii_max, y_max=0.025, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_bins_mid, radii_counts_qtls[m][:,0], radii_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_radii.pdf')
    plt.close()

# Planet radii ratios:
plot_fig_pdf_simple(fig_size, [sssp['radii_ratio_all'] for sssp in model_sssp], [], x_min=radii_ratio_bins[0], x_max=radii_ratio_bins[-1], y_max=0.06, n_bins=n_bins, log_x=True, c_sim=model_colors, lw=lw, ls_sim=model_linestyles, labels_sim=model_names, xlabel_text=r'$R_{p,i+1}/R_{p,i}$', afs=afs, tfs=tfs, lfs=lfs, fig_lbrt=fig_lbrt)
for m in range(models):
    plt.fill_between(radii_ratio_bins_mid, radii_ratio_counts_qtls[m][:,0], radii_ratio_counts_qtls[m][:,2], color=model_colors[m], alpha=alpha)
if savefigures:
    plt.savefig(directory + save_name + '_underlying_radii_ratios.pdf')
    plt.close()

plt.show()

# Eccentricity and mutual inclinations vs. intrinsic multiplicity:

fig = plt.figure(figsize=(12,6))
plot = GridSpec(1,2,left=0.1,bottom=0.15,right=0.98,top=0.95,wspace=0.3,hspace=0)

ax = plt.subplot(plot[0,0]) # ecc vs n
e_med_1_qtls = np.quantile(e_med_1_all[0], [0.16,0.5,0.84])
e_med_n_qtls = np.quantile(e_med_n_all[0], [0.16,0.5,0.84], axis=0)
e_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[0], alpha_e_all[0], n_norm=n_norm)
e_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(np.arange(10)+1, mu_e_all[1], alpha_e_all[1], n_norm=n_norm)
plt.errorbar(n_array, e_med_n_qtls[1,:], yerr=[e_med_n_qtls[1,:]-e_med_n_qtls[0,:], e_med_n_qtls[2,:]-e_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='Maximum AMD model')
plt.errorbar([1], [e_med_1_qtls[1]], yerr=[[e_med_1_qtls[1]-e_med_1_qtls[0]], [e_med_1_qtls[2]-e_med_1_qtls[1]]], fmt='o', capsize=5, ls='none', color='c', label='')
plt.plot(n_array, e_med_n_fcrits[0,:], 'o', lw=lw, color='k', label=r'$f_{\rm crit} = 0.5$')
plt.plot(n_array, e_med_n_fcrits[1,:], 'o', lw=lw, color='r', label=r'$f_{\rm crit} = 2$')
plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model1[0,:], e_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.031_{-0.003}^{+0.004}$,' + '\n' + r'$\alpha_e = -1.74_{-0.07}^{+0.10}$')
#plt.fill_between(np.arange(10)+1, e_plfit_n_qtls_model2[0,:], e_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{e,5} = 0.040_{-0.002}^{+0.003}$, $\alpha_e = -1.54_{-0.11}^{+0.10}$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.arange(10)+1)
plt.yticks([0.01, 0.1, 1.])
plt.xlim([0.5, 10.5])
plt.ylim([0.005, 1.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Intrinsic planet multiplicity $n$', fontsize=tfs)
plt.ylabel('Median eccentricity $e$', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = [handles[-1]] + handles[:-1]
labels = [labels[-1]] + labels[:-1]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

ax = plt.subplot(plot[0,1]) # incl vs n
im_med_n_qtls = np.quantile(im_med_n_all[0], [0.16,0.5,0.84], axis=0)
im_plfit_n_qtls_model1 = compute_power_law_at_n_quantiles(n_array, mu_im_all[0], alpha_im_all[0], n_norm=n_norm)
im_plfit_n_qtls_model2 = compute_power_law_at_n_quantiles(n_array, mu_im_all[1], alpha_im_all[1], n_norm=n_norm)
plt.errorbar(n_array, im_med_n_qtls[1,:], yerr=[im_med_n_qtls[1,:]-im_med_n_qtls[0,:], im_med_n_qtls[2,:]-im_med_n_qtls[1,:]], fmt='o', capsize=5, ls='none', color='g', label='')
plt.plot(n_array, im_med_n_fcrits[0,:], 'o', lw=lw, color='k', label='')
plt.plot(n_array, im_med_n_fcrits[1,:], 'o', lw=lw, color='r', label='')
plt.fill_between(n_array, im_plfit_n_qtls_model1[0,:], im_plfit_n_qtls_model1[2,:], color='b', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.10_{-0.11}^{+0.15}$,' + '\n' + r'$\alpha_i = -1.73_{-0.08}^{+0.09}$')
#plt.fill_between(n_array, im_plfit_n_qtls_model2[0,:], im_plfit_n_qtls_model2[2,:], color='g', alpha=0.25, label=r'$\tilde{\mu}_{i,5}[^\circ] = 1.41_{-0.08}^{+0.10}$, $\alpha_i = -1.55_{-0.11}^{+0.11}$')
plt.plot(n_array, incl_mult_power_law_Zhu2018(n_array)*np.sqrt(2.*np.log(2.)), 'x', ls='-', lw=2, ms=5, color='orange', label=r'$\sigma_{i,5}[^\circ] = 0.8$, $\alpha_i = -3.5$')
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=afs)
plt.xticks(np.arange(9)+2)
plt.yticks([0.1, 1., 10.])
plt.xlim([1.5, 10.5])
plt.ylim([0.1, 30.])
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
plt.xlabel('Intrinsic planet multiplicity $n$', fontsize=tfs)
plt.ylabel('Median mutual inclination $i_m$ ($^\circ$)', fontsize=tfs)
handles, labels = ax.get_legend_handles_labels()
handles = handles[1:] + [handles[0]]
labels = labels[1:] + [labels[0]]
plt.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

if savefigures:
    plt.savefig(directory + save_name + '_ecc_incl_vs_mult_fits.pdf')
    plt.close()
plt.show()
