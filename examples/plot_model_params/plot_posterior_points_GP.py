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

from syssimpyplots.plot_params import *





##### To load the files with the GP evaluated points:

savefigures = False
transformed_rates = False
#run_directory = 'Hybrid_NR20_AMD_model1/Fit_all_KS/Params13_alpha1_100/GP_files/'
#loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory
run_directory = 'Hybrid_NR20_AMD_model1/Fit_some8p1_KS/Params9_fix_highM/GP_files/' #'Fit_all_KS/Params8/'
loadfiles_directory = '/Users/hematthi/Documents/NPP_ARC_Modernize_Kepler/Personal_research/SysSim/Model_Optimization/' + run_directory
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Model_Optimization/' + run_directory
model_name = 'Hybrid_NR20_AMD_model1'

active_params_symbols = [#r'$M_{\rm break,1}$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         r'$\ln{(\alpha_{\rm ret})}$',
                         r'$\mu_M$',
                         r'$R_{p,\rm norm}$',
                         r'$\alpha_P$',
                         r'$\gamma_0$',
                         #r'$\gamma_1$',
                         r'$\sigma_0$',
                         #r'$\sigma_1$',
                         r'$\sigma_M$',
                         #r'$\sigma_P$',
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!

dims = len(active_params_symbols)

#long_symbols = [False, True, False, False, False, False, False, False, False, False, False] # Clustered_P_R_fswp_bprp_AMD_sys with Delta_c + sigma_{e,1}
long_symbols = [False]*dims # Hybrid_NR20_AMD_model1

if transformed_rates:
    active_params_transformed_symbols = np.copy(active_params_symbols)
    i_transformed, j_transformed = 0, 1 # 1, 2
    active_params_transformed_symbols[i_transformed] = r'$\ln{(\lambda_c \lambda_p)}$' #'\n'
    active_params_transformed_symbols[j_transformed] = r'$\ln{(\lambda_p/\lambda_c)}$' #'\n'

# To load the training points:
data_train = load_training_points(dims, file_name_path=loadfiles_directory, file_name='Active_params_recomputed_distances_table_best10000_every1.txt')
active_params_names = np.array(data_train['active_params_names'])

# To load the tables of points drawn from the prior based on the GP model:
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 37.65, 1425.6 # 12 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 67.65, 141134.4 # 13 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 35.0, 2.7, 16.05, 74.25 # 8 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 25.0, 2.7, 2.45, 112.9 # fit some, 8 params
#n_train, mean_f, sigma_f, lscales, vol = 2000, 30.0, 2.7, 3.03, 240.84 # fit some8, 9 params
n_train, mean_f, sigma_f, lscales, vol = 2000, 30.0, 2.7, 2.22, 48.52 # fit some8+1, 9 params
#n_points, max_mean, max_std, max_post = 100000, 'Inf', 'Inf', 'Inf' # for initial testing
n_points, max_mean, max_std, max_post = 10000, 'Inf', 'Inf', -12.0
file_name = 'GP_train%s_meanf%s_sigmaf%s_lscales%s_vol%s_points%s_mean%s_std%s_post%s.csv' % (n_train, mean_f, sigma_f, lscales, vol, n_points, max_mean, max_std, max_post)
xprior_accepted_table = load_GP_table_prior_draws(file_name, file_name_path=loadfiles_directory)
active_params_transformed_names = np.array(xprior_accepted_table.dtype.names[:dims])

GP_model_name = '_GP_train%s_meanf%s_sigmaf%s_lscales%s' % (n_train, mean_f, sigma_f, lscales)
model_name = model_name + GP_model_name





##### To plot the mean, std, and posterior draws as histograms:

#plot_fig_hists_GP_draws((16,8), xprior_accepted_table, save_name=savefigures_directory + model_name + '_vol%s_prior%s_GP_mean%s_std%s_post%s_hists.pdf' % (vol, len(xprior_accepted_table), max_mean, max_std, max_post), save_fig=savefigures)
plt.show()

##### To make cuts for the posterior:

mean_cut, std_cut, post_cut = np.inf, np.inf, -12.0
#xprior_accepts = make_cuts_GP_mean_std_post(active_params_transformed_names, xprior_accepted_table, max_mean=mean_cut, max_std=std_cut, max_post=post_cut)
if transformed_rates:
    xprior_accepts_transformed = make_cuts_GP_mean_std_post(active_params_transformed_names, xprior_accepted_table, max_mean=mean_cut, max_std=std_cut, max_post=post_cut)
    xprior_accepts = transform_sum_diff_params_inverse(xprior_accepts_transformed, i_transformed, j_transformed)
else:
    xprior_accepts = make_cuts_GP_mean_std_post(active_params_names, xprior_accepted_table, max_mean=mean_cut, max_std=std_cut, max_post=post_cut)

##### To make corner plots for the GP draws:
'''
if transformed_rates:
    plot_cornerpy_wrapper(active_params_symbols, xprior_accepts, save_name=savefigures_directory + model_name + '_vol%s_prior%s_GP_mean%s_std%s_post%s_corner.pdf' % (vol, len(xprior_accepts), mean_cut, std_cut, post_cut), save_fig=savefigures)
    plot_cornerpy_wrapper(active_params_transformed_symbols, xprior_accepts_transformed, save_name=savefigures_directory + model_name + '_vol%s_prior%s_GP_mean%s_std%s_post%s_transformed_corner.pdf' % (vol, len(xprior_accepts_transformed), mean_cut, std_cut, post_cut), save_fig=savefigures)
else:
    plot_cornerpy_wrapper(active_params_symbols, xprior_accepts, save_name=savefigures_directory + model_name + '_vol%s_prior%s_GP_mean%s_std%s_post%s_corner.pdf' % (vol, len(xprior_accepts), mean_cut, std_cut, post_cut), save_fig=savefigures)
'''

fig = plot_cornerpy_wrapper(active_params_symbols, xprior_accepts, show_titles=False)
axes = np.array(fig.axes).reshape((dims, dims))
for i in range(dims):
    q = corner.quantile(xprior_accepts[:,i], [0.16, 0.5, 0.84])
    q_pm = np.diff(q)
    if long_symbols[i]:
        title = active_params_symbols[i] + '=\n' + r'$%s_{-%s}^{+%s}$' % ('{:0.2f}'.format(q[1]), '{:0.2f}'.format(q_pm[0]), '{:0.2f}'.format(q_pm[1]))
    else:
        title = active_params_symbols[i] + r'$=%s_{-%s}^{+%s}$' % ('{:0.2f}'.format(q[1]), '{:0.2f}'.format(q_pm[0]), '{:0.2f}'.format(q_pm[1]))

    #if active_params_names[i] == 'sigma_hk':
    #    title = active_params_symbols[i] + '=\n' + r'$%s_{-%s}^{+%s}$' % ('{:0.3f}'.format(q[1]), '{:0.3f}'.format(q_pm[0]), '{:0.3f}'.format(q_pm[1]))
    ax = axes[i,i]
    ax.set_title(title, fontsize=20)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_vol%s_prior%s_GP_mean%s_std%s_post%s_corner.pdf' % (vol, len(xprior_accepts), mean_cut, std_cut, post_cut))
    plt.close()

#plt.show()
plt.close()

#To compute the quantiles directly (and get more digits, say for the eccentricity scale):
for i in range(dims):
    q = corner.quantile(xprior_accepts[:,i], [0.16, 0.5, 0.84])
    q_pm = np.diff(q)
    if active_params_names[i] == 'sigma_hk':
        print('%s = %s_{-%s}^{+%s}' % ('{:<50}'.format(active_params_names[i]), '{:0.3f}'.format(q[1]), '{:0.3f}'.format(q_pm[0]), '{:0.3f}'.format(q_pm[1])))
    else:
        print('%s = %s_{-%s}^{+%s}' % ('{:<50}'.format(active_params_names[i]), '{:0.2f}'.format(q[1]), '{:0.2f}'.format(q_pm[0]), '{:0.2f}'.format(q_pm[1])))
    if (active_params_names[i] == 'log_rate_clusters') or (active_params_names[i] == 'log_rate_planets_per_cluster'):
        q_unlogged = np.exp(q)
        q_pm_unlogged = np.diff(q_unlogged)
        print('%s = %s_{-%s}^{+%s}' % ('{:<50}'.format(active_params_names[i][4:]), '{:0.2f}'.format(q_unlogged[1]), '{:0.2f}'.format(q_pm_unlogged[0]), '{:0.2f}'.format(q_pm_unlogged[1])))





##### To plot single corner plots for pairs of parameters:
'''
afs, tfs, lfs = 20, 20, 16
bins = 20

ix, iy = 2, 6
qx = corner.quantile(xprior_accepts[:,ix], [0.16, 0.5, 0.84])
qy = corner.quantile(xprior_accepts[:,iy], [0.16, 0.5, 0.84])
qx_pm, qy_pm = np.diff(qx), np.diff(qy)

fig = plt.figure(figsize=(8,8))
plot = GridSpec(5,5,left=0.17,bottom=0.15,right=0.92,top=0.9,wspace=0,hspace=0)

ax = plt.subplot(plot[1:,:4])
corner.hist2d(xprior_accepts[:,ix], xprior_accepts[:,iy], bins=bins, plot_density=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
ax.tick_params(axis='both', labelsize=afs)
plt.xlabel(active_params_symbols[ix], fontsize=tfs)
plt.ylabel(active_params_symbols[iy], fontsize=tfs)
ax.text(x=0.05, y=0.9, s='KS', ha='left', fontsize=tfs, transform=ax.transAxes)

ax = plt.subplot(plot[0,:4]) # top histogram
plt.title(active_params_symbols[ix] + r'$= %s_{-%s}^{+%s}$' % ('{:0.2f}'.format(qx[1]), '{:0.2f}'.format(qx_pm[0]), '{:0.2f}'.format(qx_pm[1])), fontsize=tfs)
plt.hist(xprior_accepts[:,ix], bins=bins, histtype='step', color='k', ls='-', lw=1)
plt.xticks([])
plt.yticks([])
#plt.xlim()

ax = plt.subplot(plot[1:,4]) # side histogram
plt.hist(xprior_accepts[:,iy], bins=bins, histtype='step', orientation='horizontal', color='k', ls='-', lw=1)
plt.xticks([])
plt.yticks([])
ax.yaxis.set_label_position('right')
plt.ylabel(active_params_symbols[iy] + r'$= %s_{-%s}^{+%s}$' % ('{:0.2f}'.format(qy[1]), '{:0.2f}'.format(qy_pm[0]), '{:0.2f}'.format(qy_pm[1])), fontsize=tfs)
#plt.ylim()

#plt.savefig(savefigures_directory + model_name + '_vol%s_prior%s_GP_mean%s_std%s_post%s_corner_%s_%s.pdf' % (vol, len(xprior_accepts), mean_cut, std_cut, post_cut, ix, iy))
plt.close()
'''





##### To make a custom plotting function for making 'corner' plots with contours based on an array of values instead of the density of points:
'''
grid_dims = 50
GP_grids = load_GP_2d_grids(dims, n_train, mean_f, sigma_f, lscales, file_name_path=loadfiles_directory, grid_dims=grid_dims)

dist_cut = -15. # after subtracting the mean function
GP_prob_below_dist_cut_2d_grids = cdf_normal(dist_cut, mu=GP_grids['mean_grids'], std=GP_grids['std_grids'])

xtrain_sample = data_train['xtrain'][np.random.choice(np.arange(len(data_train['xtrain'])), 1000)]

plot_contours_and_points_corner(active_params_symbols, GP_grids['xlower'], GP_grids['xupper'], GP_grids['mean_grids'], xpoints=xtrain_sample, points_alpha=0.1, save_name=savefigures_directory + model_name + '_grids2d_mean_corner.pdf', save_fig=savefigures)

plot_contours_and_points_corner(active_params_symbols, GP_grids['xlower'], GP_grids['xupper'], GP_grids['std_grids'], xpoints=xtrain_sample, points_alpha=0.1, save_name=savefigures_directory + model_name + '_grids2d_std_corner.pdf', save_fig=savefigures)

plot_contours_and_points_corner(active_params_symbols, GP_grids['xlower'], GP_grids['xupper'], GP_prob_below_dist_cut_2d_grids, xpoints=xtrain_sample, points_alpha=0.1, save_name=savefigures_directory + model_name + '_grids2d_frac_mean%s_corner.pdf' % dist_cut, save_fig=savefigures)

plt.close()
'''
