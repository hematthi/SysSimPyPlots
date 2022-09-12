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

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





##### To load the files with the GP evaluated points:

savefigures = False
transformed_rates = True
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters/Figures/Model_Optimization/Split_stars/Clustered_P_R_fswp_step/Params13_KS/durations_KS/GP_files/'
model_name = 'Step_fswp_compare'

active_params_symbols = [r'$f_{\sigma_{i,\rm high}}$',
                         #r'$f_{\rm swpa}$',
                         r'$f_{\rm swpa,bluer}$',
                         r'$f_{\rm swpa,redder}$',
                         #r'$f_{\rm swpa,med}$',
                         #r'$df_{\rm swpa}/d(b_p-r_p-E^*)$',
                         r'$\ln{(\lambda_c)}$',
                         r'$\ln{(\lambda_p)}$',
                         #r'$\Delta_c$',
                         r'$\alpha_P$',
                         #r'$\alpha_{P,\rm med}$',
                         #r'$d\alpha_P/d(b_p-r_p-E^*)$',
                         r'$\alpha_{R1}$',
                         r'$\alpha_{R2}$',
                         r'$\sigma_{e}$', #'$\sigma_{e,1}$'
                         r'$\sigma_{i,\rm high}$ ($^\circ$)', # '\n',
                         r'$\sigma_{i,\rm low}$ ($^\circ$)', # '\n',
                         r'$\sigma_R$',
                         r'$\sigma_P$'
                         ] #this list of parameter symbols must match the order of parameters in the loaded table!

dims = len(active_params_symbols)

active_params_transformed_symbols = np.copy(active_params_symbols)
i_transformed, j_transformed = 3, 4
if transformed_rates:
    active_params_transformed_symbols[i_transformed] = r'$\ln{(\lambda_c \lambda_p)}$' #'\n'
    active_params_transformed_symbols[j_transformed] = r'$\ln{(\lambda_p/\lambda_c)}$' #'\n'

# To load the tables of points drawn from the prior based on the GP model:
loadfiles_directory1 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/Split_stars/Clustered_P_R_fswp_step/Params13_KS/durations_KS/GP_files/'
file_name1 = 'GP_train2000_meanf75.0_sigmaf2.7_lscales38.42_vol37.62_points50000_meanInf_stdInf_post-28.0.csv'
xprior_accepted_table1 = load_GP_table_prior_draws(file_name1, file_name_path=loadfiles_directory1)
active_params_transformed_names = np.array(xprior_accepted_table1.dtype.names[:dims])

loadfiles_directory2 = '/Users/hematthi/Documents/GradSchool/Research/ACI/Model_Optimization/Split_stars/Clustered_P_R_fswp_step/Params13_AD/durations_AD/GP_files/'
file_name2 = 'GP_train2000_meanf150.0_sigmaf2.7_lscales38.42_vol37.62_points50000_meanInf_stdInf_post-50.0.csv'
xprior_accepted_table2 = load_GP_table_prior_draws(file_name2, file_name_path=loadfiles_directory2)
active_params_transformed_names = np.array(xprior_accepted_table2.dtype.names[:dims])





##### To make cuts for the posterior:

xprior_accepts_transformed1 = make_cuts_GP_mean_std_post(active_params_transformed_names, xprior_accepted_table1, max_post=-28.0)
xprior_accepts1 = transform_sum_diff_params_inverse(xprior_accepts_transformed1, i_transformed, j_transformed)

xprior_accepts_transformed2 = make_cuts_GP_mean_std_post(active_params_transformed_names, xprior_accepted_table2, max_post=-50.0)
xprior_accepts2 = transform_sum_diff_params_inverse(xprior_accepts_transformed2, i_transformed, j_transformed)





##### To plot a histogram of parameters:

fswp_diff1 = xprior_accepts1[:,2] - xprior_accepts1[:,1]
fswp_diff2 = xprior_accepts2[:,2] - xprior_accepts2[:,1]
qtls1 = np.quantile(fswp_diff1, [0.16, 0.5, 0.84])
qtls2 = np.quantile(fswp_diff2, [0.16, 0.5, 0.84])

figsize, fig_lbrt = (8,4), [0.15, 0.2, 0.95, 0.95]
afs, tfs, lfs = 20, 20, 16
bins = 100

plot_fig_pdf_simple(figsize, [fswp_diff1, fswp_diff2], [], x_min=-0.1, x_max=0.7, n_bins=bins, normalize=False, c_sim=['k','k'], ls_sim=['-','--'], lw=1, labels_sim=['KS','AD'], xlabel_text=r'$f_{\rm swpa,redder} - f_{\rm swpa,bluer}$', ylabel_text='Points', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_lbrt=fig_lbrt)
plt.axvline(x=0., color='r', ls='--', lw=1)
#plt.savefig(savefigures_directory + model_name + '_KS_AD.pdf')
plt.show()
