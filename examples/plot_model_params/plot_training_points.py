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
import copy
#matplotlib.rc('text', usetex=True)


from syssimpyplots.plot_params import *





##### To load the files with the GP evaluated points:

savefigures = False
run_directory = 'Hybrid_NR20_AMD_model1/Fit_split_KS/Params12/GP_files/' #'Hybrid_NR20_AMD_model1/Fit_all_KS/Params13_alpha1_100/GP_files/'
loadfiles_directory = '/Users/hematthi/Documents/NotreDame_Postdoc/CRC/Files/SysSim/Model_Optimization/' + run_directory
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Model_Optimization/' + run_directory
model_name = 'Hybrid_NR20_AMD_model1'

active_params_symbols = [r'$M_{\rm break,1}$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         r'$\mu_M$',
                         r'$R_{p,\rm norm}$',
                         r'$\alpha_P$',
                         r'$\gamma_0$',
                         r'$\gamma_1$',
                         r'$\sigma_0$',
                         r'$\sigma_1$',
                         r'$\sigma_M$',
                         r'$\sigma_P$',
                         #r'$\alpha_{\rm ret}$',
                         ] # this list of parameter symbols must match the order of parameters in 'active_params_names'!
dims = len(active_params_symbols)

active_params_transformed_symbols = copy.deepcopy(np.array(active_params_symbols, dtype=object))
i_transformed, j_transformed = 1, 2
active_params_transformed_symbols[i_transformed] = r'$\ln{(\lambda_c \lambda_p)}$'
active_params_transformed_symbols[j_transformed] = r'$\ln{(\lambda_p/\lambda_c)}$'

# To load the training points:
data_optim = load_training_points(dims, file_name_path=loadfiles_directory, file_name='Active_params_distances_table_best100000_every10.txt')
data_train = load_training_points(dims, file_name_path=loadfiles_directory, file_name='Active_params_recomputed_distances_table_best100000_every10.txt')
active_params_names = np.array(data_train['active_params_names'])





# To first plot histograms of the distances (from optimization, and after recomputing):
fig = plt.figure(figsize=(16,8))
plot = GridSpec(1,1,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0,hspace=0)
ax = plt.subplot(plot[0,0])
plt.hist([data_optim['ytrain'], data_train['ytrain']], histtype='step', bins=100, label=['Optimization', 'Recomputed'])
ax.tick_params(axis='both', labelsize=20)
plt.xlabel('Total weighted distance', fontsize=20)
plt.ylabel('Points', fontsize=20)
plt.legend(loc='upper right', ncol=1, frameon=False, fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_distances.pdf')
    plt.close()
plt.show()

##### If we want to compute and plot the un-logged rates (i.e. lambda_c, lambda_p):
#data_train['xtrain'][:,[1,2]] = np.exp(data_train['xtrain'][:,[1,2]])
#active_params_symbols[1], active_params_symbols[2] = r'$\lambda_c$', r'$\lambda_p$'
#active_params_transformed_symbols[1], active_params_transformed_symbols[2] = r'$\lambda_c \lambda_p$', r'$\lambda_p /\lambda_c$'

# To make corner plots for the GP training points:

plot_cornerpy_wrapper(active_params_symbols, data_train['xtrain'], save_name=savefigures_directory + model_name + '_training_corner.pdf', save_fig=savefigures)
plot_cornerpy_wrapper(active_params_transformed_symbols, transform_sum_diff_params(data_train['xtrain'], i_transformed, j_transformed), save_name=savefigures_directory + model_name + '_training_transformed_corner.pdf', save_fig=savefigures)
plt.show()





##### To load the table of points minimizing the GP mean and overplot them:
'''
n_points_min = 100
file_name = 'GP_train%s_meanf%s_sigmaf%s_lscales%s_minimize_mean_iterations%s.csv' % (n_train, mean_f, sigma_f, lscales, n_points_min)
xmin_table = load_table_points_min_GP(file_name, file_name_path=loadfiles_directory)
xmins = xmin_table[active_params_names].view((float, dims))

plot_cornerpy_wrapper(active_params_symbols, data_train['xtrain'], xpoints_extra=xmins, save_name=savefigures_directory + model_name + '_training_corner.pdf', save_fig=savefigures)
plot_cornerpy_wrapper(active_params_transformed_symbols, transform_sum_diff_params(data_train['xtrain'], 1, 2), xpoints_extra=transform_sum_diff_params(xmins, 1, 2), save_name=savefigures_directory + model_name + '_training_transformed_corner.pdf', save_fig=savefigures)
plt.close()
'''
