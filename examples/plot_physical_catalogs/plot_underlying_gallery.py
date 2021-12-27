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





##### To load the underlying and observed populations:

savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/Best_models/GP_med/Systems_conditional/'
run_number = ''
model_name = 'Maximum_AMD_model' + run_number

compute_ratios = compute_ratios_adjacent
AD_mod = 'true' # 'true' or 'false'
weights_all = load_split_stars_weights_only()
dists_include = ['delta_f',
                 'mult_CRPD_r',
                 'periods_KS',
                 'period_ratios_KS',
                 #'durations_KS',
                 #'durations_norm_circ_KS',
                 'durations_norm_circ_singles_KS',
                 'durations_norm_circ_multis_KS',
                 'duration_ratios_nonmmr_KS',
                 'duration_ratios_mmr_KS',
                 'depths_KS',
                 'radius_ratios_KS',
                 'radii_partitioning_KS',
                 'radii_monotonicity_KS',
                 'gap_complexity_KS',
                 ]

N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)
N_factor = N_sim/N_Kep

param_vals_all = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
sssp_per_sys, sssp = compute_summary_stats_from_cat_phys(file_name_path=loadfiles_directory, run_number=run_number, load_full_tables=True)
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod, compute_ratios=compute_ratios)





from src.functions_compute_RVs import *

def plot_systems_gallery_conditional(sssp_per_sys, sssp, P_cond, Rp_cond, det=False, fig_size=(6,12), panels_per_fig=1, N_sys_sample=50, N_sys_per_plot=50, plot_line_per=1, afs=20, tfs=20, save_name_base='no_name_fig', save_fig=False):
    
    # P_cond = [P_lower, P_upper] for period of planet conditioned on
    # Rp_cond = [Rp_lower, Rp_upper] for radius of planet conditioned on
    P_lower, P_upper = P_cond
    Rp_lower, Rp_upper = Rp_cond
    assert P_lower < P_upper
    assert Rp_lower < Rp_upper
    
    n_per_sys = sssp_per_sys['Mtot_all']
    n_det_per_sys = np.sum(sssp_per_sys['det_all'], axis=1)
    
    if det:
        bools_in_P_Rp_cond = np.any((sssp_per_sys['det_all'] == 1) & (sssp_per_sys['P_all'] > P_lower) & (sssp_per_sys['P_all'] < P_upper) & (sssp_per_sys['radii_all'] > Rp_lower) & (sssp_per_sys['radii_all'] < Rp_upper), axis=1)
    else:
        bools_in_P_Rp_cond = np.any((sssp_per_sys['P_all'] > P_lower) & (sssp_per_sys['P_all'] < P_upper) & (sssp_per_sys['radii_all'] > Rp_lower) & (sssp_per_sys['radii_all'] < Rp_upper), axis=1)
    i_cond = np.arange(len(n_per_sys))[bools_in_P_Rp_cond]
    print('Number of systems that have a planet with period in %s (d) and radius in %s (R_earth): %s' % (P_cond, Rp_cond, len(i_cond)))
    
    i_sample = np.random.choice(i_cond, N_sys_sample, replace=False)
    i_sort = np.argsort(sssp['Mstar_all'][i_sample])

    Mstar_sample = sssp['Mstar_all'][i_sample][i_sort]
    P_sample = sssp_per_sys['P_all'][i_sample][i_sort]
    det_sample = sssp_per_sys['det_all'][i_sample][i_sort]
    Mp_sample = sssp_per_sys['mass_all'][i_sample][i_sort]
    Rp_sample = sssp_per_sys['radii_all'][i_sample][i_sort]
    e_sample = sssp_per_sys['e_all'][i_sample][i_sort]
    incl_sample = sssp_per_sys['incl_all'][i_sample][i_sort]
    clusterids_sample = sssp_per_sys['clusterids_all'][i_sample][i_sort]
    
    n_panels = int(np.ceil(float(N_sys_sample)/N_sys_per_plot))
    n_figs = int(np.ceil(float(n_panels)/panels_per_fig))
    print('Generating %s figures...' % n_figs)
    for h in range(n_figs):
        fig = plt.figure(figsize=fig_size)
        plot = GridSpec(1,panels_per_fig,left=0.1,bottom=0.075,right=0.9,top=0.85,wspace=0,hspace=0.1)
        for i in range(panels_per_fig):
            ax = plt.subplot(plot[0,i])
            j_start = (h*panels_per_fig + i)*N_sys_per_plot
            j_end = (h*panels_per_fig + i+1)*N_sys_per_plot
            for j in range(len(P_sample[j_start:j_end])):
                id_sys = (h*panels_per_fig + i)*N_sys_per_plot + j
                
                Mstar_sys = Mstar_sample[id_sys]
                P_sys = P_sample[id_sys]
                det_sys = det_sample[id_sys]
                Mp_sys = Mp_sample[id_sys]
                Rp_sys = Rp_sample[id_sys]
                e_sys = e_sample[id_sys]
                incl_sys = incl_sample[id_sys]
                clusterids_sys = clusterids_sample[id_sys]
                
                det_sys = det_sys[P_sys > 0]
                Mp_sys = Mp_sys[P_sys > 0]
                Rp_sys = Rp_sys[P_sys > 0]
                e_sys = e_sys[P_sys > 0]
                incl_sys = incl_sys[P_sys > 0]
                clusterids_sys = clusterids_sys[P_sys > 0]
                P_sys = P_sys[P_sys > 0]
                
                K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
                #print(K_sys)
                sc = plt.scatter(P_sys[det_sys == 1], np.ones(np.sum(det_sys == 1))+j, c=K_sys[det_sys == 1], s=10.*Rp_sys[det_sys == 1]**2., vmin=0., vmax=5.)
                plt.scatter(P_sys[det_sys == 0], np.ones(np.sum(det_sys == 0))+j, c=K_sys[det_sys == 0], edgecolors='r', s=10.*Rp_sys[det_sys == 0]**2., vmin=0., vmax=5.) #facecolors='none'
                #sc = plt.scatter(P_sys, np.ones(len(P_sys))+j, c=K_sys, s=10.*Rp_sys**2., vmin=0., vmax=10.)
                #plt.scatter(P_sys[det_sys == 0], np.ones(np.sum(det_sys == 0))+j, color='r', marker='x', s=8.*Rp_sys[det_sys == 0]**2.)
                
                if (j+1)%plot_line_per == 0:
                    plt.axhline(y=j+1, lw=0.5, ls='--', color='k')
                    plt.text(x=1.8, y=j+1, s='{:.2f}'.format(np.round(Mstar_sys,2)), va='center', ha='right', fontsize=8)
                    plt.text(x=450, y=j+1, s='{:.2f}'.format(np.round(np.max(K_sys),2)), va='center', ha='left', fontsize=8)
            plt.text(x=1.6, y=j+2.5, s=r'$M_\star$ ($M_\odot$)', va='center', ha='center', fontsize=8)
            plt.text(x=500, y=j+2.5, s=r'$K_{\rm max}$ (m/s)', va='center', ha='center', fontsize=8)
            #plt.fill_betweenx(y=[0,j+2], x1=P_lower, x2=P_upper, color='g', alpha=0.2)
            plt.gca().set_xscale("log")
            ax.tick_params(axis='both', labelsize=afs)
            ax.set_xticks([3,10,30,100,300])
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.set_yticks([])
            plt.xlim([2., 400.])
            plt.ylim([0., N_sys_per_plot+1])
            plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)

        plot = GridSpec(1,1,left=0.1,bottom=0.91,right=0.9,top=0.93,wspace=0,hspace=0)
        plt.figtext(0.5, 0.96, 'Systems conditioned on a planet with\n' r'$P = %s$d, $R_p = %s R_\oplus$' % (P_cond, Rp_cond), va='center', ha='center', fontsize=tfs)
        cax = plt.subplot(plot[0,0])
        cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label(r'RV semi-amplitude $K$ (m/s)', fontsize=12)
        
        save_name = save_name_base + '_%s.pdf' % i
        if save_fig:
            plt.savefig(save_name)
            plt.close()


##### To plot galleries of systems conditioned on a given planet:

afs = 16 #axes labels font size
tfs = 16 #text labels font size
lfs = 16 #legend labels font size

P_cond, Rp_cond = [9.5,10.5], [1.,1.5]
fig_name = savefigures_directory + model_name + '_systems_with_P%s_%s_R%s_%s_detected' % (P_cond[0], P_cond[1], Rp_cond[0], Rp_cond[1])
plot_systems_gallery_conditional(sssp_per_sys, sssp, P_cond, Rp_cond, det=True, fig_size=(5,10), afs=afs, tfs=tfs, save_name_base=fig_name, save_fig=savefigures)

P_cond, Rp_cond = [4.5,5.5], [1.8,2.2]
fig_name = savefigures_directory + model_name + '_systems_with_P%s_%s_R%s_%s_detected' % (P_cond[0], P_cond[1], Rp_cond[0], Rp_cond[1])
plot_systems_gallery_conditional(sssp_per_sys, sssp, P_cond, Rp_cond, det=True, fig_size=(5,10), afs=afs, tfs=tfs, save_name_base=fig_name, save_fig=savefigures)

P_cond, Rp_cond = [19.,21.], [1.,2.]
fig_name = savefigures_directory + model_name + '_systems_with_P%s_%s_R%s_%s_detected' % (P_cond[0], P_cond[1], Rp_cond[0], Rp_cond[1])
plot_systems_gallery_conditional(sssp_per_sys, sssp, P_cond, Rp_cond, det=True, fig_size=(5,10), afs=afs, tfs=tfs, save_name_base=fig_name, save_fig=savefigures)

P_cond, Rp_cond = [19.,21.], [2.,5.]
fig_name = savefigures_directory + model_name + '_systems_with_P%s_%s_R%s_%s_detected' % (P_cond[0], P_cond[1], Rp_cond[0], Rp_cond[1])
plot_systems_gallery_conditional(sssp_per_sys, sssp, P_cond, Rp_cond, det=True, fig_size=(5,10), afs=afs, tfs=tfs, save_name_base=fig_name, save_fig=savefigures)
