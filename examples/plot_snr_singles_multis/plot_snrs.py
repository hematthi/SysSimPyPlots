# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec #for specifying plot attributes
from matplotlib import ticker #for setting contour plots to log scale
from matplotlib.colors import LogNorm #for log color scales
from scipy.stats import ks_2samp

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

from src.functions_general import *
from src.functions_compare_kepler import *
from src.functions_load_sims import *
from src.functions_plot_catalogs import *
from src.functions_plot_params import *





savefigures = False

def load_snrs_singles_multis(snr_file):
    snr_per_sys = [] # list to be filled with lists of all SNRs per system
    with open(snr_file, 'r') as file:
        for line in file:
            if line[0] != '#':
                line = line[1:-2].split(', ')
                snr_sys = [float(i) for i in line]
                snr_per_sys.append(snr_sys)

    snr_singles = []
    snr_multis = []
    snr_multis_2 = []
    snr_multis_3 = []
    snr_multis_4p = []
    snr_multis_weakest = []
    snr_ratios_adj = []
    snr_ratios_adj_2 = []
    snr_ratios_adj_3 = []
    snr_ratios_adj_4p = []
    snr_ratios_all = []
    for x in snr_per_sys:
        if len(x) == 1:
            snr_singles += x
        else:
            ratios_adj = compute_ratios_adjacent(np.array(x))
            ratios_all = compute_ratios_all(np.array(x))
            
            snr_multis += x
            snr_multis_weakest.append(np.min(x))
            snr_ratios_adj += list(ratios_adj)
            snr_ratios_all += list(ratios_all)
            
            # To also save snr ratios as a function of multiplicity:
            if len(x) == 2:
                snr_multis_2 += x
                snr_ratios_adj_2 += list(ratios_adj)
            elif len(x) == 3:
                snr_multis_3 += x
                snr_ratios_adj_3 += list(ratios_adj)
            else: # if len(x) >= 4
                snr_multis_4p += x
                snr_ratios_adj_4p += list(ratios_adj)

    snr = {}
    snr['per_sys'] = snr_per_sys
    snr['singles'] = np.array(snr_singles)
    snr['multis'] = np.array(snr_multis)
    snr['multis_2'] = np.array(snr_multis_2)
    snr['multis_3'] = np.array(snr_multis_3)
    snr['multis_4p'] = np.array(snr_multis_4p)
    snr['multis_weakest'] = np.array(snr_multis_weakest)
    snr['ratios_adj'] = np.array(snr_ratios_adj)
    snr['ratios_adj_2'] = np.array(snr_ratios_adj_2)
    snr['ratios_adj_3'] = np.array(snr_ratios_adj_3)
    snr['ratios_adj_4p'] = np.array(snr_ratios_adj_4p)
    snr['ratios_all'] = np.array(snr_ratios_all)
    return snr

# To load a single simulated catalog:
#snr_sim = load_snrs_singles_multis('snrs_sim.out')

# Repeat for Kepler planets:
snr_kep_no_cuts = load_snrs_singles_multis('snrs_kepler_all.out')
snr_kep_pr_cuts = load_snrs_singles_multis('snrs_kepler_PR_cuts.out')
snr_kep_star_cuts = load_snrs_singles_multis('snrs_kepler_HFR2020b_stars.out')
snr_kep = load_snrs_singles_multis('snrs_kepler_HFR2020b_all_cuts.out')

ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(3., 300., 0.5, 10., compute_ratios=compute_ratios_adjacent)

# Load multiple simulated catalogs:
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/output_snr/GP_best_models/'

snr_sims_all = []
runs = 100
for i in range(1,100+1): #range(1,runs+1)
    print(i)
    snr_sim = load_snrs_singles_multis(loadfiles_directory + 'snrs%s.out' % i)
    snr_sims_all.append(snr_sim)

# To load the full catalog information for one simulated catalog:
snr_sim = load_snrs_singles_multis(loadfiles_directory + 'snrs1.out')
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number='1', compute_ratios=compute_ratios_adjacent)

snr_singles_all = np.array([])
snr_multis_all = np.array([])
snr_multis_weakest_all = np.array([])

n_bins = 100

snr_bins = np.logspace(np.log10(5.), np.log10(1e4), n_bins+1)
snr_bins_mid = (snr_bins[:-1] + snr_bins[1:])/2.
snr_counts_singles_all_normfield = [] # counts normalized by number of planets in this field only
snr_counts_singles_all = [] # counts normalized by number of total planets
snr_counts_multis_all_normfield = []
snr_counts_multis_all = []
snr_counts_multis_2_all_normfield = []
snr_counts_multis_2_all = []
snr_counts_multis_3_all_normfield = []
snr_counts_multis_3_all = []
snr_counts_multis_4p_all_normfield = []
snr_counts_multis_4p_all = []
snr_counts_multis_weakest_all_normfield = []
snr_counts_multis_weakest_all = []
pvals_ks_singles_all = []
pvals_ks_multis_all = []
pvals_ks_multis_weakest_all = []

snr_ratios_bins = np.logspace(np.log10(2e-2), 1., n_bins+1)
snr_ratios_bins_mid = (snr_ratios_bins[:-1] + snr_ratios_bins[1:])/2.
snr_ratios_adj_counts_all = []
snr_ratios_all_counts_all = []

for snr_dict in snr_sims_all:
    n_pl = len(snr_dict['singles']) + len(snr_dict['multis'])
    snr_singles_all = np.concatenate((snr_singles_all, snr_dict['singles']))
    snr_multis_all = np.concatenate((snr_multis_all, snr_dict['multis']))
    snr_multis_weakest_all = np.concatenate((snr_multis_weakest_all, snr_dict['multis_weakest']))
    
    counts, bins = np.histogram(snr_dict['singles'], bins=snr_bins)
    snr_counts_singles_all_normfield.append(counts/float(np.sum(counts)))
    snr_counts_singles_all.append(counts/float(n_pl))
    #snr_counts_singles_all.append(counts)
    counts, bins = np.histogram(snr_dict['multis'], bins=snr_bins)
    snr_counts_multis_all_normfield.append(counts/float(np.sum(counts)))
    snr_counts_multis_all.append(counts/float(n_pl))
    #snr_counts_multis_all.append(counts)
    counts, bins = np.histogram(snr_dict['multis_2'], bins=snr_bins)
    snr_counts_multis_2_all_normfield.append(counts/float(np.sum(counts)))
    snr_counts_multis_2_all.append(counts/float(n_pl))
    #snr_counts_multis_all.append(counts)
    counts, bins = np.histogram(snr_dict['multis_3'], bins=snr_bins)
    snr_counts_multis_3_all_normfield.append(counts/float(np.sum(counts)))
    snr_counts_multis_3_all.append(counts/float(n_pl))
    #snr_counts_multis_all.append(counts)
    counts, bins = np.histogram(snr_dict['multis_4p'], bins=snr_bins)
    snr_counts_multis_4p_all_normfield.append(counts/float(np.sum(counts)))
    snr_counts_multis_4p_all.append(counts/float(n_pl))
    #snr_counts_multis_all.append(counts)
    counts, bins = np.histogram(snr_dict['multis_weakest'], bins=snr_bins)
    snr_counts_multis_weakest_all_normfield.append(counts/float(np.sum(counts)))
    snr_counts_multis_weakest_all.append(counts/float(n_pl))
    #snr_counts_multis_weakest_all.append(counts)
    
    pvals_ks_singles_all.append(ks_2samp(snr_dict['singles'], snr_kep['singles']).pvalue)
    pvals_ks_multis_all.append(ks_2samp(snr_dict['multis'], snr_kep['multis']).pvalue)
    pvals_ks_multis_weakest_all.append(ks_2samp(snr_dict['multis_weakest'], snr_kep['multis_weakest']).pvalue)
    
    counts, bins = np.histogram(snr_dict['ratios_adj'], bins=snr_ratios_bins)
    snr_ratios_adj_counts_all.append(counts/float(np.sum(counts)))
    counts, bins = np.histogram(snr_dict['ratios_all'], bins=snr_ratios_bins)
    snr_ratios_all_counts_all.append(counts/float(np.sum(counts)))

snr_counts_singles_all_normfield = np.array(snr_counts_singles_all_normfield)
snr_counts_singles_all = np.array(snr_counts_singles_all)
snr_counts_multis_all_normfield = np.array(snr_counts_multis_all_normfield)
snr_counts_multis_all = np.array(snr_counts_multis_all)
snr_counts_multis_2_all_normfield = np.array(snr_counts_multis_2_all_normfield)
snr_counts_multis_2_all = np.array(snr_counts_multis_2_all)
snr_counts_multis_3_all_normfield = np.array(snr_counts_multis_3_all_normfield)
snr_counts_multis_3_all = np.array(snr_counts_multis_3_all)
snr_counts_multis_4p_all_normfield = np.array(snr_counts_multis_4p_all_normfield)
snr_counts_multis_4p_all = np.array(snr_counts_multis_4p_all)
snr_counts_multis_weakest_all_normfield = np.array(snr_counts_multis_weakest_all_normfield)
snr_counts_multis_weakest_all = np.array(snr_counts_multis_weakest_all)
pvals_ks_singles_all = np.array(pvals_ks_singles_all)
pvals_ks_multis_all = np.array(pvals_ks_multis_all)
pvals_ks_multis_weakest_all = np.array(pvals_ks_multis_weakest_all)

snr_ratios_adj_counts_all = np.array(snr_ratios_adj_counts_all)
snr_ratios_all_counts_all = np.array(snr_ratios_all_counts_all)

# To compute the quantiles:
qtls = [0.16,0.5,0.84]

snr_counts_singles_qtls_normfield = np.zeros((n_bins,3))
snr_counts_singles_qtls = np.zeros((n_bins,3))
snr_counts_multis_qtls_normfield = np.zeros((n_bins,3))
snr_counts_multis_qtls = np.zeros((n_bins,3))
snr_counts_multis_2_qtls_normfield = np.zeros((n_bins,3))
snr_counts_multis_2_qtls = np.zeros((n_bins,3))
snr_counts_multis_3_qtls_normfield = np.zeros((n_bins,3))
snr_counts_multis_3_qtls = np.zeros((n_bins,3))
snr_counts_multis_4p_qtls_normfield = np.zeros((n_bins,3))
snr_counts_multis_4p_qtls = np.zeros((n_bins,3))
snr_counts_multis_weakest_qtls_normfield = np.zeros((n_bins,3))
snr_counts_multis_weakest_qtls = np.zeros((n_bins,3))
snr_ratios_adj_counts_qtls = np.zeros((n_bins,3))
snr_ratios_all_counts_qtls = np.zeros((n_bins,3))
for b in range(n_bins):
    snr_counts_singles_qtls_normfield[b] = np.quantile(snr_counts_singles_all_normfield[:,b], qtls)
    snr_counts_singles_qtls[b] = np.quantile(snr_counts_singles_all[:,b], qtls)
    snr_counts_multis_qtls_normfield[b] = np.quantile(snr_counts_multis_all_normfield[:,b], qtls)
    snr_counts_multis_qtls[b] = np.quantile(snr_counts_multis_all[:,b], qtls)
    snr_counts_multis_2_qtls_normfield[b] = np.quantile(snr_counts_multis_2_all_normfield[:,b], qtls)
    snr_counts_multis_2_qtls[b] = np.quantile(snr_counts_multis_2_all[:,b], qtls)
    snr_counts_multis_3_qtls_normfield[b] = np.quantile(snr_counts_multis_3_all_normfield[:,b], qtls)
    snr_counts_multis_3_qtls[b] = np.quantile(snr_counts_multis_3_all[:,b], qtls)
    snr_counts_multis_4p_qtls_normfield[b] = np.quantile(snr_counts_multis_4p_all_normfield[:,b], qtls)
    snr_counts_multis_4p_qtls[b] = np.quantile(snr_counts_multis_4p_all[:,b], qtls)
    snr_counts_multis_weakest_qtls_normfield[b] = np.quantile(snr_counts_multis_weakest_all_normfield[:,b], qtls)
    snr_counts_multis_weakest_qtls[b] = np.quantile(snr_counts_multis_weakest_all[:,b], qtls)
    snr_ratios_adj_counts_qtls[b] = np.quantile(snr_ratios_adj_counts_all[:,b], qtls)
    snr_ratios_all_counts_qtls[b] = np.quantile(snr_ratios_all_counts_all[:,b], qtls)





# To compare different cuts in the Kepler catalog:

# All on the same plot:
plot_fig_pdf_simple((16,8), [], [], x_min=5., x_max=1e4, y_max=0.03, n_bins=100, normalize=False, log_x=True, xlabel_text='SNR', ylabel_text='Fraction', legend=False, fig_lbrt=[0.1, 0.125, 0.95, 0.95])
colors = ['r','g','b','k']
labels = ['No cuts', 'P-R cuts only', 'Star cuts only', 'All H2020b cuts']
labels_totals = []
for i,snr_kep_cuts in enumerate([snr_kep_no_cuts, snr_kep_pr_cuts, snr_kep_star_cuts, snr_kep]):
    n_pl = len(snr_kep_cuts['singles']) + len(snr_kep_cuts['multis'])
    print(n_pl)
    labels_totals.append(labels[i] + ' (%s planets)' % n_pl)
    plt.hist(snr_kep_cuts['singles'], bins=np.logspace(np.log10(5.), np.log10(1e4), 101), weights=np.ones(len(snr_kep_cuts['singles']))/n_pl, histtype='step', ls='-', color=colors[i], label=labels[i])
    plt.hist(snr_kep_cuts['multis'], bins=np.logspace(np.log10(5.), np.log10(1e4), 101), weights=np.ones(len(snr_kep_cuts['multis']))/n_pl, histtype='step', ls='--', color=colors[i])
    plt.hist(snr_kep_cuts['multis_weakest'], bins=np.logspace(np.log10(5.), np.log10(1e4), 101), weights=np.ones(len(snr_kep_cuts['multis_weakest']))/n_pl, histtype='step', ls=':', color=colors[i])
# Need custom proxy artists for multiple legends:
singles_line = mlines.Line2D([], [], ls='-', color='k', label='Observed singles')
multis_line = mlines.Line2D([], [], ls='--', color='k', label='Observed multis')
weakest_multis_line = mlines.Line2D([], [], ls=':', color='k', label='Weakest in observed multis')
legend1 = plt.legend(handles=[singles_line, multis_line, weakest_multis_line], loc='lower right', bbox_to_anchor=(1,0.2), ncol=1, frameon=False, fontsize=20)
plt.legend(handles=[mlines.Line2D([], [], ls='-', color=colors[i], label=labels_totals[i]) for i in range(len(labels))], loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=20)
plt.gca().add_artist(legend1)
if savefigures:
    plt.savefig('snr_singles_multis_kepler_cuts.pdf')
    plt.close()

# Separate panels:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(2,2, left=0.075, bottom=0.125, right=0.95, top=0.95, wspace=0.2, hspace=0.3)
for i,snr_kep_cuts in enumerate([snr_kep_no_cuts, snr_kep_pr_cuts, snr_kep_star_cuts, snr_kep]):
    n_pl = len(snr_kep_cuts['singles']) + len(snr_kep_cuts['multis'])
    print(n_pl)
    ax = plt.subplot(plot[int(i/2),i%2])
    plot_panel_pdf_simple(ax, [snr_kep_cuts['singles'], snr_kep_cuts['multis'], snr_kep_cuts['multis_weakest']], [], x_min=5., x_max=1e4, n_bins=100, normalize=False, log_x=True, c_sim=[colors[i]]*3, ls_sim=['-','--',':'], labels_sim=['Observed singles','Observed multis','Weakest in observed multis'], xlabel_text='SNR', ylabel_text='Counts', legend=True if i==0 else False)
    plt.text(x=0.98, y=0.9, s=labels[i], ha='right', fontsize=20, transform=ax.transAxes)
if savefigures:
    plt.savefig('snr_singles_multis_kepler_cuts_panels.pdf')
    plt.close()





##### To compare SNRs of singles vs multis for Kepler vs a simulated catalog:
'''
plot_fig_pdf_simple((16,8), [snr_sim['singles'], snr_sim['multis'], snr_sim['multis_weakest']], [], x_min=5., x_max=1e4, n_bins=100, normalize=False, log_x=True, c_sim=['b','r','r'], ls_sim=['-','-','--'], labels_sim=['Observed singles','Observed multis','Weakest in observed multis'], xlabel_text='SNR', ylabel_text='Counts', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_singles_multis_sim.pdf', save_fig=False)

plot_fig_pdf_simple((16,8), [], [snr_kep['singles'], snr_kep['multis'], snr_kep['multis_weakest']], x_min=5., x_max=1e4, n_bins=100, normalize=False, log_x=True, c_Kep=['b','r','m'], ls_Kep=['-','-','--'], labels_Kep=['Observed singles','Observed multis','Weakest in observed multis'], xlabel_text='SNR', ylabel_text='Counts', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_singles_multis_kepler.pdf', save_fig=False)

plot_fig_pdf_simple((16,8), [snr_sim['singles'], snr_sim['multis'], snr_sim['multis_weakest']], [snr_kep['singles'], snr_kep['multis'], snr_kep['multis_weakest']], x_min=5., x_max=1e4, n_bins=100, normalize=False, log_x=True, c_sim=['b','r','r'], c_Kep=['b','r','m'], ls_sim=['-','-','--'], ls_Kep=['-','-','--'], labels_sim=['Observed singles','Observed multis','Weakest in observed multis'], labels_Kep=['Observed singles (Kepler)','Observed multis (Kepler)','Weakest in observed multis (Kepler)'], xlabel_text='SNR', ylabel_text='Counts', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_singles_multis_compare.pdf', save_fig=False)

plt.show()
'''

##### To compare SNRs of singles vs multis for Kepler vs many simulated catalogs:

# Separate figures:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(3,2, left=0.1, bottom=0.125, right=0.95, top=0.95, wspace=0.3, hspace=0)
fields = ['singles', 'multis', 'multis_weakest']
colors = ['b', 'r', 'm']
labels = ['Observed singles','Observed multis','Weakest in observed multis']
for i,snr_qtls in enumerate([snr_counts_singles_qtls, snr_counts_multis_qtls, snr_counts_multis_weakest_qtls]):
    n_pl = len(snr_kep['singles']) + len(snr_kep['multis'])
    ax = plt.subplot(plot[i,0])
    plot_panel_pdf_simple(ax, [snr_kep[fields[i]]], [], x_min=5., x_max=1e4, n_bins=100, normalize=False, N_sim_Kep_factor=n_pl, log_x=True, c_sim=[colors[i]], labels_sim=['Kepler data'], xlabel_text='SNR', ylabel_text='Fraction', legend=False)
    if i != 2:
        plt.xticks([])
    plt.fill_between(snr_bins_mid, snr_qtls[:,0], snr_qtls[:,2], step='mid', color=colors[i], alpha=0.2, label='Model 16-84%')
    plt.text(x=0.98, y=0.85, s=labels[i], ha='right', fontsize=16, transform=ax.transAxes)
    if i == 0:
        plt.title('Bins normalized by total planets in catalog', fontsize=20)
        plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=16)
for i,snr_qtls in enumerate([snr_counts_singles_qtls_normfield, snr_counts_multis_qtls_normfield, snr_counts_multis_weakest_qtls_normfield]):
    n_pl = len(snr_kep['singles']) + len(snr_kep['multis'])
    ax = plt.subplot(plot[i,1])
    plot_panel_pdf_simple(ax, [snr_kep[fields[i]]], [], x_min=5., x_max=1e4, n_bins=100, normalize=False, N_sim_Kep_factor=len(snr_kep[fields[i]]), log_x=True, c_sim=[colors[i]], labels_sim=['Kepler data'], xlabel_text='SNR', ylabel_text='Fraction', legend=False)
    if i != 2:
        plt.xticks([])
    plt.fill_between(snr_bins_mid, snr_qtls[:,0], snr_qtls[:,2], step='mid', color=colors[i], alpha=0.2, label='Model 16-84%')
    plt.text(x=0.98, y=0.85, s=labels[i], ha='right', fontsize=16, transform=ax.transAxes)
    if i == 0:
        plt.title('Bins normalized by total planets in category', fontsize=20)
        plt.legend(loc='lower right', bbox_to_anchor=(1,0), ncol=1, frameon=False, fontsize=16)
if savefigures:
    plt.savefig('snr_singles_multis_compare_credible_panels.pdf')
    plt.close()

# All on the same figure:
plot_fig_pdf_simple((16,8), [snr_kep['singles'], snr_kep['multis'], snr_kep['multis_weakest']], [], x_min=5., x_max=1e4, n_bins=100, normalize=False, N_sim_Kep_factor=len(snr_kep['singles']) + len(snr_kep['multis']), log_x=True, c_sim=['b','r','m'], ls_sim=['-','-','-'], labels_sim=['Observed singles','Observed multis','Weakest in observed multis'], xlabel_text='SNR', ylabel_text='Fraction', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_singles_multis_kepler.pdf', save_fig=False)
plt.fill_between(snr_bins_mid, snr_counts_singles_qtls[:,0], snr_counts_singles_qtls[:,2], step='mid', color='b', alpha=0.2)
plt.fill_between(snr_bins_mid, snr_counts_multis_qtls[:,0], snr_counts_multis_qtls[:,2], step='mid', color='r', alpha=0.2)
plt.fill_between(snr_bins_mid, snr_counts_multis_weakest_qtls[:,0], snr_counts_multis_weakest_qtls[:,2], step='mid', color='m', alpha=0.2)
if savefigures:
    plt.savefig('snr_singles_multis_compare_credible.pdf')
    plt.close()

plt.show()

# All simulated catalogs combined:
plot_fig_pdf_simple((16,8), [snr_singles_all, snr_multis_all, snr_multis_weakest_all], [snr_kep['singles'], snr_kep['multis'], snr_kep['multis_weakest']], x_min=5., x_max=1e4, n_bins=100, normalize=False, N_sim_Kep_factor=100, log_x=True, c_sim=['b','r','m'], c_Kep=['b','r','m'], ls_sim=['-','-','--'], ls_Kep=['-','-','--'], labels_sim=['Observed singles','Observed multis','Weakest in observed multis'], labels_Kep=['Observed singles (Kepler)','Observed multis (Kepler)','Weakest in observed multis (Kepler)'], xlabel_text='SNR', ylabel_text='Counts', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_singles_multis_compare.pdf', save_fig=False)

print('p-val for singles: ', ks_2samp(snr_singles_all, snr_kep['singles']).pvalue)
print('p-val for multis: ', ks_2samp(snr_multis_all, snr_kep['multis']).pvalue)
print('p-val for weakest multis: ', ks_2samp(snr_multis_weakest_all, snr_kep['multis_weakest']).pvalue)

plt.show()

# SNRs of multis as a function of observed multiplicity:
plot_fig_pdf_simple((16,8), [snr_kep['multis'], snr_kep['multis_2'], snr_kep['multis_3'], snr_kep['multis_4p']], [], x_min=5., x_max=1e4, n_bins=100, normalize=False, N_sim_Kep_factor=len(snr_kep['singles']) + len(snr_kep['multis']), log_x=True, c_sim=['k','b','r','m'], ls_sim=['-','-','-','-'], labels_sim=['All observed multis','Observed doubles','Observed triples','Observed 4+'], xlabel_text='SNR', ylabel_text='Fraction', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_multis_per_mult_kepler.pdf', save_fig=False)
plt.fill_between(snr_bins_mid, snr_counts_multis_qtls[:,0], snr_counts_multis_qtls[:,2], step='mid', color='k', alpha=0.2)
plt.fill_between(snr_bins_mid, snr_counts_multis_2_qtls[:,0], snr_counts_multis_2_qtls[:,2], step='mid', color='b', alpha=0.2)
plt.fill_between(snr_bins_mid, snr_counts_multis_3_qtls[:,0], snr_counts_multis_3_qtls[:,2], step='mid', color='r', alpha=0.2)
plt.fill_between(snr_bins_mid, snr_counts_multis_4p_qtls[:,0], snr_counts_multis_4p_qtls[:,2], step='mid', color='m', alpha=0.2)
if savefigures:
    plt.savefig('snr_multis_per_mult_compare_credible.pdf')
    plt.close()

plt.show()





##### To plot SNR ratios for Kepler vs many simulated catalogs:

# All SNR ratios (adjacent vs all pairs):
plot_fig_pdf_simple((16,8), [snr_kep['ratios_adj'], snr_kep['ratios_all']], [], x_min=2e-2, x_max=10., n_bins=100, normalize=True, log_x=True, c_sim=['b','k'], ls_sim=['-','-'], labels_sim=['Adjacent pairs','All pairs'], xlabel_text='SNR ratio', ylabel_text='Fraction', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_ratios_compare.pdf', save_fig=False)
plt.fill_between(snr_ratios_bins_mid, snr_ratios_adj_counts_qtls[:,0], snr_ratios_adj_counts_qtls[:,2], step='mid', color='b', alpha=0.2)
plt.fill_between(snr_ratios_bins_mid, snr_ratios_all_counts_qtls[:,0], snr_ratios_all_counts_qtls[:,2], step='mid', color='k', alpha=0.2)
if savefigures:
    plt.savefig('snr_ratios_compare.pdf')
    plt.close()
plt.show()

# SNR ratios as a function of observed multiplicity:
plot_fig_pdf_simple((16,8), [snr_kep['ratios_adj'], snr_kep['ratios_adj_2'], snr_kep['ratios_adj_3'], snr_kep['ratios_adj_4p']], [], x_min=2e-2, x_max=10., n_bins=100, normalize=True, log_x=True, c_sim=['k','b','r','m'], ls_sim=['-','-','-','-'], labels_sim=['All adjacent pairs','Observed doubles','Observed triples','Observed 4+'], xlabel_text='SNR ratio', ylabel_text='Fraction', legend=True, fig_lbrt=[0.1, 0.15, 0.95, 0.95], save_name='snr_ratios_per_mult.pdf', save_fig=False)
#plt.fill_between(snr_ratios_bins_mid, snr_ratios_adj_counts_qtls[:,0], snr_ratios_adj_counts_qtls[:,2], step='mid', color='b', alpha=0.2)
#plt.fill_between(snr_ratios_bins_mid, snr_ratios_all_counts_qtls[:,0], snr_ratios_all_counts_qtls[:,2], step='mid', color='k', alpha=0.2)
if savefigures:
    plt.savefig('snr_ratios_per_mult.pdf')
    plt.close()
plt.show()

# Scatter plot of SNR_in vs SNR_out:
snr_in_out_adj_Kep = [] # [mult, SNR_in, SNR_out]
snr_in_out_adj_sim = []
for snr_sys in snr_kep['per_sys']:
    m = len(snr_sys)
    if m >= 2:
        for i in range(m-1):
            snr_in_out_adj_Kep.append([m, snr_sys[i], snr_sys[i+1]])
for snr_sys in snr_sim['per_sys']:
    m = len(snr_sys)
    if m >= 2:
        for i in range(m-1):
            snr_in_out_adj_sim.append([m, snr_sys[i], snr_sys[i+1]])
snr_in_out_adj_Kep = np.array(snr_in_out_adj_Kep)
snr_in_out_adj_sim = np.array(snr_in_out_adj_sim)

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1, left=0.125, bottom=0.125, right=0.95, top=0.95, wspace=0, hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(snr_in_out_adj_Kep[:,1], snr_in_out_adj_Kep[:,2], marker='.', color='k', label='Kepler')
plt.scatter(snr_in_out_adj_sim[:,1], snr_in_out_adj_sim[:,2], marker='.', color='r', label='Simulated')
plt.plot(np.logspace(-1, 4, 100), np.logspace(-1, 4, 100), '--', color='k', label='Equal SNR')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([5., 1e3])
plt.ylim([5., 1e3])
plt.xlabel(r'SNR$_{\rm in}$', fontsize=20)
plt.ylabel(r'SNR$_{\rm out}$', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=16)
plt.show()

fig = plt.figure(figsize=(16,5))
plot = GridSpec(1,3, left=0.075, bottom=0.15, right=0.975, top=0.95, wspace=0.3, hspace=0)

ax = plt.subplot(plot[0,0])
plt.text(x=6., y=600., s='Observed doubles', fontsize=16)
plt.scatter(snr_in_out_adj_Kep[snr_in_out_adj_Kep[:,0] == 2,1], snr_in_out_adj_Kep[snr_in_out_adj_Kep[:,0] == 2,2], marker='.', color='k', label='Kepler')
plt.scatter(snr_in_out_adj_sim[snr_in_out_adj_sim[:,0] == 2,1], snr_in_out_adj_sim[snr_in_out_adj_sim[:,0] == 2,2], marker='.', color='r', label='Simulated')
plt.plot(np.logspace(-1, 4, 100), np.logspace(-1, 4, 100), '--', color='k', label='Equal SNR')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([5., 1e3])
plt.ylim([5., 1e3])
plt.xlabel(r'SNR$_{\rm in}$', fontsize=20)
plt.ylabel(r'SNR$_{\rm out}$', fontsize=20)

ax = plt.subplot(plot[0,1])
plt.text(x=6., y=600., s='Observed triples', fontsize=16)
plt.scatter(snr_in_out_adj_Kep[snr_in_out_adj_Kep[:,0] == 3,1], snr_in_out_adj_Kep[snr_in_out_adj_Kep[:,0] == 3,2], marker='^', color='k', label='Kepler')
plt.scatter(snr_in_out_adj_sim[snr_in_out_adj_sim[:,0] == 3,1], snr_in_out_adj_sim[snr_in_out_adj_sim[:,0] == 3,2], marker='^', color='r', label='Simulated')
plt.plot(np.logspace(-1, 4, 100), np.logspace(-1, 4, 100), '--', color='k', label='Equal SNR')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([5., 1e3])
plt.ylim([5., 1e3])
plt.xlabel(r'SNR$_{\rm in}$', fontsize=20)
plt.ylabel(r'SNR$_{\rm out}$', fontsize=20)

ax = plt.subplot(plot[0,2])
plt.text(x=6., y=600., s='Observed 4+', fontsize=16)
plt.scatter(snr_in_out_adj_Kep[snr_in_out_adj_Kep[:,0] >= 4,1], snr_in_out_adj_Kep[snr_in_out_adj_Kep[:,0] >= 4,2], marker='+', color='k', label='Kepler')
plt.scatter(snr_in_out_adj_sim[snr_in_out_adj_sim[:,0] >= 4,1], snr_in_out_adj_sim[snr_in_out_adj_sim[:,0] >= 4,2], marker='+', color='r', label='Simulated')
plt.plot(np.logspace(-1, 4, 100), np.logspace(-1, 4, 100), '--', color='k', label='Equal SNR')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
plt.xlim([5., 1e3])
plt.ylim([5., 1e3])
plt.xlabel(r'SNR$_{\rm in}$', fontsize=20)
plt.ylabel(r'SNR$_{\rm out}$', fontsize=20)

plt.show()



# Scatter plot of SNR ratios vs. period ratios and radius ratios:
fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1, left=0.125, bottom=0.125, right=0.95, top=0.95, wspace=0, hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(ssk['Rm_obs'], snr_kep['ratios_adj'], marker='.', color='k', label='Kepler')
#plt.scatter(snr_in_out_adj_sim[:,1], snr_in_out_adj_sim[:,2], marker='.', color='r', label='Simulated')
#plt.plot(np.logspace(-1, 4, 100), np.logspace(-1, 4, 100), '--', color='k', label='Equal SNR')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
#plt.xlim([1., 20.])
#plt.ylim([5., 1e3])
plt.xlabel(r'$P_{\rm out}/P_{\rm in}$', fontsize=20)
plt.ylabel(r'SNR$_{\rm out}$/SNR$_{\rm in}$', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=16)
plt.show()

fig = plt.figure(figsize=(8,8))
plot = GridSpec(1,1, left=0.125, bottom=0.125, right=0.95, top=0.95, wspace=0, hspace=0)
ax = plt.subplot(plot[0,0])
plt.scatter(ssk['D_ratio_obs'], snr_kep['ratios_adj'], marker='.', color='k', label='Kepler')
#plt.scatter(snr_in_out_adj_sim[:,1], snr_in_out_adj_sim[:,2], marker='.', color='r', label='Simulated')
#plt.plot(np.logspace(-1, 4, 100), np.logspace(-1, 4, 100), '--', color='k', label='Equal SNR')
plt.gca().set_xscale("log")
plt.gca().set_yscale("log")
ax.tick_params(axis='both', labelsize=16)
#plt.xlim([1., 20.])
#plt.ylim([5., 1e3])
plt.xlabel(r'$\delta_{\rm out}/\delta_{\rm in}$', fontsize=20)
plt.ylabel(r'SNR$_{\rm out}$/SNR$_{\rm in}$', fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(0.,1.), ncol=1, frameon=False, fontsize=16)
plt.show()
