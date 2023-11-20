# To import required modules:
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes

from syssimpyplots.compare_kepler import *





##### To load the weights files:

sample_names = ['all', 'bluer', 'redder']
sample_colors = ['k','b','r']
Nmult_max = 8

Nmult_evals_jobs = []
d_all_vals_evals_jobs = []
weights_all_jobs = []
'''
job_seeds = [567, 1964, 2997, 3346]
for js in job_seeds:
    Nmult_evals, d_all_keys_evals, d_all_vals_evals, weights_all = load_split_stars_model_evaluations_and_weights('../../ACI/Clustered_P_R_split_stars_weights_ADmod_true_targs399675_evals1000_seed%s.txt' % js)
    Nmult_evals_jobs.append(Nmult_evals)
    d_all_vals_evals_jobs.append(d_all_vals_evals)
    weights_all_jobs.append(weights_all)
'''
Nmult_evals, d_all_keys_evals, d_all_vals_evals, weights_all = load_split_stars_model_evaluations_and_weights('../../../SysSimExClusters/src/Maximum_AMD_model_split_stars_weights_ADmod_true_targs86760_evals100_all_pairs.txt')
#Nmult_evals, d_all_keys_evals, d_all_vals_evals, weights_all = load_split_stars_model_evaluations_and_weights('../../src/syssimpyplots/data/Clustered_P_R_split_stars_weights_ADmod_true_targs88912_evals100_all_pairs.txt')
#Nmult_evals, d_all_keys_evals, d_all_vals_evals, weights_all = load_split_stars_model_evaluations_and_weights('../../ACI/Clustered_P_R_split_stars_weights_ADmod_true_targs79935_evals100_all_pairs.txt')
Nmult_evals_jobs.append(Nmult_evals)
d_all_vals_evals_jobs.append(d_all_vals_evals)
weights_all_jobs.append(weights_all)

jobs = len(Nmult_evals_jobs)





##### To plot the weights:

dists_keys_KS = ["delta_f", "mult_CRPD", "mult_CRPD_r", "periods_KS", "period_ratios_KS", "depths_KS", "radii_KS", "radius_ratios_KS", "durations_KS", "durations_norm_circ_KS", "durations_norm_circ_singles_KS", "durations_norm_circ_multis_KS", "duration_ratios_KS", "duration_ratios_nonmmr_KS", "duration_ratios_mmr_KS", "radii_partitioning_KS", "radii_monotonicity_KS", "gap_complexity_KS"]
dists_keys_AD = ["delta_f", "mult_CRPD", "mult_CRPD_r", "periods_AD", "period_ratios_AD", "depths_AD", "radii_AD", "radius_ratios_AD", "durations_AD", "durations_norm_circ_AD", "durations_norm_circ_singles_AD", "durations_norm_circ_multis_AD", "duration_ratios_AD", "duration_ratios_nonmmr_AD", "duration_ratios_mmr_AD", "radii_partitioning_AD", "radii_monotonicity_AD", "gap_complexity_AD"]

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.075,bottom=0.3,right=0.95,top=0.95,wspace=0.,hspace=0.)
ax = plt.subplot(plot[0,0])
for i in range(jobs):
    weights_all_job = weights_all_jobs[i]
    for j,key in enumerate(sample_names):
        plt.plot(range(len(dists_keys_KS)), [weights_all_job[key][dist_key] for dist_key in dists_keys_KS], drawstyle='steps-mid', color=sample_colors[j], ls='-', label=key)
plt.gca().set_yscale("log")
plt.xticks(range(len(dists_keys_KS)), dists_keys_KS, rotation='vertical')
plt.show()

fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.075,bottom=0.3,right=0.95,top=0.95,wspace=0.,hspace=0.)
ax = plt.subplot(plot[0,0])
for i in range(jobs):
    weights_all_job = weights_all_jobs[i]
    for j,key in enumerate(sample_names):
        plt.plot(range(len(dists_keys_AD)), [weights_all_job[key][dist_key] for dist_key in dists_keys_AD], drawstyle='steps-mid', color=sample_colors[j], ls='-', label=key)
plt.gca().set_yscale("log")
plt.xticks(range(len(dists_keys_AD)), dists_keys_AD, rotation='vertical')
plt.show()





##### To plot the individual distance terms (weighted and unweighted) from the model evaluations used to compute the weights:

dists_keys_plot = ["delta_f", "mult_CRPD", "mult_CRPD_r", "periods_KS", "period_ratios_KS", "depths_KS", "radii_KS", "radius_ratios_KS", "durations_KS", "durations_norm_circ_KS", "durations_norm_circ_singles_KS", "durations_norm_circ_multis_KS", "duration_ratios_KS", "duration_ratios_nonmmr_KS", "duration_ratios_mmr_KS", "radii_partitioning_KS", "radii_monotonicity_KS", "gap_complexity_KS", "periods_AD", "period_ratios_AD", "depths_AD", "radii_AD", "radius_ratios_AD", "durations_AD", "durations_norm_circ_AD", "durations_norm_circ_singles_AD", "durations_norm_circ_multis_AD", "duration_ratios_AD", "duration_ratios_nonmmr_AD", "duration_ratios_mmr_AD", "radii_partitioning_AD", "radii_monotonicity_AD", "gap_complexity_AD"]

# To first print the rms and weights in an easy to read format:
print('{:<30}: {:<8}, {:<8}; {:<8}, {:<8}; {:<8}, {:<8}'.format('Distance term', 'rms_all', 'w_all', 'rms_blue', 'w_blue', 'rms_red', 'w_red'))
for key in dists_keys_plot:
    w_all, w_blue, w_red = weights_all['all'][key], weights_all['bluer'][key], weights_all['redder'][key]
    rms_all, rms_blue, rms_red = 1./w_all, 1./w_blue, 1./w_red
    print('{:<30}: {:<8}, {:<8}; {:<8}, {:<8}; {:<8}, {:<8}'.format(key, np.round(rms_all,5), int(np.round(w_all)), np.round(rms_blue,5), int(np.round(w_blue)), np.round(rms_red,5), int(np.round(w_red))))

N_panels = len(dists_keys_plot)
cols = int(np.ceil(np.sqrt(N_panels))) #number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols #number of rows, such that rows*cols >= N_panels

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,dist_key in enumerate(dists_keys_plot):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    for j in range(jobs):
        d_all_vals_evals_job = d_all_vals_evals_jobs[j]
        weights_all_job = weights_all_jobs[j]

        plt.hist([d_all_vals_evals_job[key][dist_key] for key in sample_names], bins=50, histtype='step', color=sample_colors, label=sample_names)
        for s,key in enumerate(sample_names):
            plt.axvline(x=1./weights_all_job[key][dist_key], color=sample_colors[s])
    plt.xlabel(dist_key, fontsize=12)
    plt.ylabel('')
    #if i==0:
        #plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, fontsize=12)

fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0.3,hspace=0.4)
for i,dist_key in enumerate(dists_keys_plot):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])

    for j in range(jobs):
        d_all_vals_evals_job = d_all_vals_evals_jobs[j]
        weights_all_job = weights_all_jobs[j]

        plt.hist([d_all_vals_evals_job[key][dist_key]*weights_all_job[key][dist_key] for key in sample_names], bins=50, histtype='step', color=sample_colors, label=sample_names)
    plt.xlabel(dist_key, fontsize=12)
    plt.ylabel('')
    #if i==0:
        #plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, fontsize=12)

plt.show()





##### To plot the observed multiplicities from the model evaluations:
'''
Nmult_max_plot = Nmult_max-2

fig = plt.figure(figsize=(16,8))
plot = GridSpec(Nmult_max_plot,1,left=0.05,bottom=0.1,right=0.975,top=0.975,wspace=0,hspace=0)
for i in range(Nmult_max_plot):
    ax = plt.subplot(plot[i,0])

    for j in range(jobs):
        Nmult_evals_job = Nmult_evals_jobs[j]

        for s,key in enumerate(sample_names):
            x = Nmult_evals_job[key][str(i+1)]
            if np.min(x) > 0:
                plt.hist(x, histtype='step', bins=np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), 21), color=sample_colors[s], label=sample_names[s])
    plt.gca().set_xscale("log")
    if i!=Nmult_max_plot-1:
        ax.tick_params(labelbottom=False)
    plt.xlim([1,7000])
    plt.text(x=0.01, y=0.8, s='N = %s' % (str(i+1)), transform = ax.transAxes)
    plt.xlabel('Number of systems with N observed planets', fontsize=12)
    plt.ylabel('')
    #if i==Nmult_max_plot-1:
        #plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=3, fontsize=12)
plt.show()
'''





##### To compute the std of the total distances:
'''
dists_keys_include = ["delta_f", "mult_CRPD_r", "periods_AD", "period_ratios_AD", "depths_AD", "radius_ratios_AD", "durations_AD", "duration_ratios_nonmmr_AD", "duration_ratios_mmr_AD"]

evals = len(d_all_vals_evals_jobs[0]['all'])
dtot_samples = {}
for sample in sample_names:
    dtot_evals = np.zeros(evals)
    for dist_key in dists_keys_include:
        dtot_evals += d_all_vals_evals_jobs[0][sample][dist_key]*weights_all_jobs[0][sample][dist_key]
    dtot_samples[sample] = dtot_evals

dtot = dtot_samples['all'] + dtot_samples['bluer'] + dtot_samples['redder']
'''
