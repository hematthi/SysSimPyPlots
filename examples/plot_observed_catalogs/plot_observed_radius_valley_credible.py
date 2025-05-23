# To import required modules:
import numpy as np
import time
import os
import sys
import matplotlib
import matplotlib.cm as cm # for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec # for specifying plot attributes
import scipy.integrate # for numerical integration
import scipy.interpolate # for interpolation functions
from scipy.stats import gaussian_kde # for fitting KDE's
from scipy.stats import wasserstein_distance # the "Earth-mover's" distance

from syssimpyplots.general import *
from syssimpyplots.compare_kepler import *
from syssimpyplots.load_sims import *
from syssimpyplots.plot_catalogs import *
from syssimpyplots.plot_params import *





savefigures = False
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/SysSimExClusters/examples/test/'
#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_med/'
savefigures_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Figures/Hybrid_NR20_AMD_model1/Observed/' + 'Radius_valley_measures/Fit_some8_KS_params9/'
run_number = ''
model_name = 'Hybrid_NR20_AMD_model1' + run_number

compute_ratios = compute_ratios_adjacent
weights_all = load_split_stars_weights_only()
dists_include = ['depths_KS', 'radii_KS']





##### To load the files with the systems with observed planets:

# To first read the number of simulated targets and bounds for the periods and radii:
N_sim, cos_factor, P_min, P_max, radii_min, radii_max = read_targets_period_radius_bounds(loadfiles_directory + 'periods%s.out' % run_number)

# To read the simulation parameters from the file:
params = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)

# To load and process the simulated observed catalog of stars and planets:
sss_per_sys, sss = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)

# To load and process the observed Kepler catalog and compare with our simulated catalog:
ssk_per_sys, ssk = compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, compute_ratios=compute_ratios)

dists, dists_w = compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all['all'], dists_include, N_sim, cos_factor=cos_factor)





##### To define functions for measuring the significance of the radius valley:

##### TODO: move these functions to a module in src

def calculate_radius_valley_depth_using_global_bins(radii_sample, radius_valley_bounds=(1.8,2.2), x_min=0.5, x_max=5.5, n_bins=100, fractional_depth=True, verbose=False, plot_fig=False, save_name='no_name_fig.pdf', save_fig=False):
    """
    Create a histogram of a sample of planet radii and compute the "depth" of the radius valley.
    
    The "depth" of the radius valley is defined as the difference between the minimum value of the bins within 'radius_valley_bounds' and the minimum of the maximum values of the bins on each side of the valley.
    
    Parameters
    ----------
    radii_sample : array[float]
        The sample of planet radii (in Earth radii).
    radius_valley_bounds : (float,float), default=(1.8,2.2)
        The bounds for planet radii considered to be within the "radius valley" (in Earth radii).
    x_min : float, default=0.5
        The minimum planet radius to include (in Earth radii).
    x_max : float, default=5.5
        The maximum planet radius to include (in Earth radii).
    n_bins : int, default=100
        The number of bins to use between `x_min` and `x_max`.
    fractional_depth : bool, default=True
        Whether to compute the fractional depth (i.e. the depth as a fraction of the minimum of the maximum bins on either side; the max value of 1 means the valley drops to zero). Note that this can be negative if the "valley" is actually higher than one of peaks.
    verbose : bool, default=False
        Whether to print the computed values.
    plot_fig : bool, default=False
        Whether to also plot the histogram and the depth of the radius valley. If True, will call :py:func:`syssimpyplots.plot_catalogs.plot_fig_pdf_simple`.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure, if `plot_fig=True`.
    save_fig : bool, default=False
        Whether to save the figure, if `plot_fig=True`. If True, will save the figure in the working directory with the file name given by `save_name`.
    
    Returns
    -------
    depth : float
        The "depth" of the radius valley. This should be a positive value; if negative, it means that the radius "valley" is actually a peak (more precisely, it is higher than at least one of the peaks on the two sides)!
    """
    # Compute the normalized histogram:
    radii_inrange = radii_sample[(radii_sample >= x_min) & (radii_sample <= x_max)]
    N_inrange = len(radii_inrange) # number of samples in the range considered
    counts, bin_edges = np.histogram(radii_inrange, bins=n_bins, range=(x_min,x_max), weights=np.ones(N_inrange)/N_inrange) # the counts are normalized to sum to 1
    bins_mid = (bin_edges[:-1] + bin_edges[1:])/2.
    
    # Figure out which bins (whose midpoints) are within the radius valley bounds:
    iBins_valley = np.where((bins_mid >= radius_valley_bounds[0]) & (bins_mid <= radius_valley_bounds[1]))[0]
    
    # Find the minimum of the bins in the radius valley and the maximums of the bins on either side:
    # NOTE: the following 'iBin_' indices still index the full array of counts/bins
    iBin_min_valley = np.argmin(counts[iBins_valley]) + iBins_valley[0]
    iBin_max_before_valley = np.argmax(counts[:iBins_valley[0]]) # the peak before the valley
    iBin_max_after_valley = np.argmax(counts[iBins_valley[-1]:]) + iBins_valley[-1] # the peak after the valley
    min_valley = counts[iBin_min_valley]
    max_before_valley = counts[iBin_max_before_valley]
    max_after_valley = counts[iBin_max_after_valley]
    
    # Compute the "depth" of the radius valley:
    height = min(max_before_valley, max_after_valley)
    depth = (height - min_valley)/height if fractional_depth else height - min_valley
    if depth < 0:
        print(f'WARNING: the depth ({depth}) is negative! There is no valley in the given radius valley bounds.')
    
    if verbose:
        print(f'min_valley = {min_valley} at radius mid-bin = {bins_mid[iBin_min_valley]}')
        print(f'max_before_valley = {max_before_valley} at radius mid-bin = {bins_mid[iBin_max_before_valley]}')
        print(f'max_after_valley = {max_after_valley} at radius mid-bin = {bins_mid[iBin_max_after_valley]}')
        print('#')
        print(f'depth = {depth}')
        print('#####')
    
    # To also make a plot:
    if plot_fig:
        ax = plot_fig_pdf_simple([radii_sample], [], x_min=x_min, x_max=x_max, n_bins=n_bins, normalize=True, xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]')
        plt.axvline(radius_valley_bounds[0], ls=':', lw=1)
        plt.axvline(radius_valley_bounds[1], ls=':', lw=1)
        plt.hlines(height, radius_valley_bounds[0], radius_valley_bounds[1], linestyles='dashed', lw=1)
        ax.annotate('', xy=(bins_mid[iBin_min_valley], min_valley), xytext=(bins_mid[iBin_min_valley], height), arrowprops=dict(arrowstyle='->', lw=2, color='r'))
        plt.figtext(0.92, 0.9, r'Valley depth = {:.2f}'.format(depth), color='r', fontsize=16, ha='right', va='top')
        
        if save_fig:
            plt.savefig(save_name)
            plt.close()
    
    return depth

def calculate_radius_valley_depth_using_kde(radii_sample, radius_valley_bounds=(1.8,2.2), x_min=0.5, x_max=5.5, n_bins=100, bw='Scotts', bw_scotts_factor=1., fractional_depth=True, verbose=False, plot_fig=False, save_name='no_name_fig.pdf', save_fig=False):
    """
    Fit a KDE to a sample of planet radii and compute the "depth" of the radius valley.
    
    Same as `calculate_radius_valley_depth_using_global_bins` except the bins are replaced by a KDE.
    
    Parameters
    ----------
    radii_sample : array[float]
        The sample of planet radii (in Earth radii).
    radius_valley_bounds : (float,float), default=(1.8,2.2)
        The bounds for planet radii considered to be within the "radius valley" (in Earth radii).
    x_min : float, default=0.5
        The minimum planet radius to include (in Earth radii).
    x_max : float, default=5.5
        The maximum planet radius to include (in Earth radii).
    n_bins : int, default=100
        The number of bins to use between `x_min` and `x_max`, for plotting purposes.
    bw : {float, 'Scotts'}, default='Scotts'
        The bandwidth or method for computing the bandwidth. If 'Scotts', will compute the bandwidth using Scott's rule, which is 'n^(-1/5)' where 'n' is the number of data points.
    bw_scotts_factor : float, default=1.
        The factor to multiply the bandwidth from Scott's rule. Only used if `bw='Scotts'`.
    fractional_depth : bool, default=True
        Whether to compute the fractional depth (i.e. the depth as a fraction of the minimum of the maximum on either side; the max value of 1 means the valley drops to zero). Note that this can be negative if the "valley" is actually higher than one of peaks.
    verbose : bool, default=False
        Whether to print the computed values.
    plot_fig : bool, default=False
        Whether to also plot the histogram and the depth of the radius valley. If True, will call :py:func:`syssimpyplots.plot_catalogs.plot_fig_pdf_simple`.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure, if `plot_fig=True`.
    save_fig : bool, default=False
        Whether to save the figure, if `plot_fig=True`. If True, will save the figure in the working directory with the file name given by `save_name`.
    
    Returns
    -------
    depth : float
        The "depth" of the radius valley. This should be a positive value; if negative, it means that the radius "valley" is actually a peak (more precisely, it is higher than at least one of the peaks on the two sides)!
    """
    #radii_inrange = radii_sample[(radii_sample >= x_min) & (radii_sample <= x_max)]
    #N_inrange = len(radii_inrange) # number of samples in the range considered
    
    # Fit the KDE:
    # NOTE: we should fit the KDE to the full sample, not just the points in the range (x_min, x_max); thus also use the total number of data points in the calculation of 'bw'.
    if bw == 'Scotts':
        bw_Scotts = len(radii_sample)**(-1./5)
        bw = bw_Scotts*bw_scotts_factor
    kde = gaussian_kde(radii_sample, bw_method=bw)
    
    # Evaluate the KDE on an array of radii values:
    x_evals = np.linspace(x_min, x_max, 1001)
    kde_evals = kde(x_evals)
    
    i_valley = np.where((x_evals >= radius_valley_bounds[0]) & (x_evals <= radius_valley_bounds[1]))[0] # indices of the 'x_evals' that are in the radius valley

    # Find the minimum in the radius valley and the maximums on either side using the KDE:
    i_min_valley = np.argmin(kde_evals[i_valley]) + i_valley[0]
    i_max_before_valley = np.argmax(kde_evals[:i_valley[0]]) # the peak before the valley
    i_max_after_valley = np.argmax(kde_evals[i_valley[-1]:]) + i_valley[-1] # the peak after the valley
    min_valley = kde_evals[i_min_valley]
    max_before_valley = kde_evals[i_max_before_valley]
    max_after_valley = kde_evals[i_max_after_valley]
    
    # Compute the "depth" of the radius valley:
    height = min(max_before_valley, max_after_valley)
    depth = (height - min_valley)/height if fractional_depth else height - min_valley
    if depth < 0:
        print(f'WARNING: the depth ({depth}) is negative! There is no valley in the given radius valley bounds.')
    
    if verbose:
        print(f'min_valley = {min_valley} at radius = {x_evals[i_min_valley]}')
        print(f'max_before_valley = {max_before_valley} at radius = {x_evals[i_max_before_valley]}')
        print(f'max_after_valley = {max_after_valley} at radius = {x_evals[i_max_after_valley]}')
        print('#')
        print(f'depth = {depth}')
        print('#####')
    
    # To also make a plot:
    if plot_fig:
        ax = plot_fig_pdf_simple([radii_sample], [], x_min=x_min, x_max=x_max, n_bins=n_bins, normalize=True, labels_sim=[None], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]')
        bins = np.linspace(x_min, x_max, n_bins+1) # these are the bins the call above would be using to create the histogram; needed to normalize the KDE density
        fnorm = bins[1]-bins[0]
        plt.plot(x_evals, fnorm*kde(x_evals), color='b', label=r'KDE fit ($bw = {:0.2f}$)'.format(bw))
        plt.axvline(radius_valley_bounds[0], ls=':', lw=1)
        plt.axvline(radius_valley_bounds[1], ls=':', lw=1)
        plt.hlines(fnorm*height, radius_valley_bounds[0], radius_valley_bounds[1], linestyles='dashed', lw=1)
        ax.annotate('', xy=(x_evals[i_min_valley], fnorm*min_valley), xytext=(x_evals[i_min_valley], fnorm*height), arrowprops=dict(arrowstyle='->', lw=2, color='r'))
        plt.figtext(0.92, 0.9, r'Valley depth = {:.2f}'.format(depth), color='r', fontsize=16, ha='right', va='top')
        plt.legend(loc='upper right', bbox_to_anchor=(1,0.9), ncol=1, frameon=False, fontsize=16)
        
        if save_fig:
            plt.savefig(save_name)
            plt.close()
    
    return depth

def calculate_radius_valley_depth_using_two_kdes(radii_sample, radius_valley_bounds=(1.8,2.2), x_min=0.5, x_max=5.5, n_bins=100, bw_low='Scotts', bw_high='Scotts', bw_low_scotts_factor=0.25, bw_high_scotts_factor=2., fractional_depth=True, verbose=False, plot_fig=False, save_name='no_name_fig.pdf', save_fig=False):
    """
    Fit two KDEs (a low bandwith and a high bandwidth) to a sample of planet radii and compute the "depth" of the radius valley.
    
    The "depth" of the radius valley is defined as the difference between the KDEs, at the location of the minimum of the KDE with the low bandwidth, within 'radius_valley_bounds'.
    
    Parameters
    ----------
    radii_sample : array[float]
        The sample of planet radii (in Earth radii).
    radius_valley_bounds : (float,float), default=(1.8,2.2)
        The bounds for planet radii considered to be within the "radius valley" (in Earth radii).
    x_min : float, default=0.5
        The minimum planet radius to include (in Earth radii).
    x_max : float, default=5.5
        The maximum planet radius to include (in Earth radii).
    n_bins : int, default=100
        The number of bins to use between `x_min` and `x_max`, for plotting purposes.
    bw_low : {float, 'Scotts'}, default='Scotts'
        The bandwidth or method for computing the bandwidth, for the low-bandwidth KDE. If 'Scotts', will compute the bandwidth using Scott's rule, which is 'n^(-1/5)' where 'n' is the number of data points.
    bw_high : {float, 'Scotts'}, default='Scotts'
        The bandwidth or method for computing the bandwidth, for the high-bandwidth KDE.
    bw_low_scotts_factor : float, default=0.25
        The factor to multiply the bandwidth from Scott's rule for the low-bandwidth KDE. Only used if `bw_low='Scotts'`.
    bw_high_scotts_factor : float, default=2.
        The factor to multiply the bandwidth from Scott's rule for the high-bandwidth KDE. Only used if `bw_high='Scotts'`.
    fractional_depth : bool, default=True
        Whether to compute the fractional depth (i.e. the depth as a fraction of the minimum of the maximum on either side; the max value of 1 means the valley drops to zero). Note that this can be negative if the "valley" is actually higher than one of peaks.
    verbose : bool, default=False
        Whether to print the computed values.
    plot_fig : bool, default=False
        Whether to also plot the histogram and the depth of the radius valley. If True, will call :py:func:`syssimpyplots.plot_catalogs.plot_fig_pdf_simple`.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure, if `plot_fig=True`.
    save_fig : bool, default=False
        Whether to save the figure, if `plot_fig=True`. If True, will save the figure in the working directory with the file name given by `save_name`.
    
    Returns
    -------
    depth : float
        The "depth" of the radius valley. This should be a positive value; if negative, it means that the radius "valley" is actually a peak!
    """
    # Fit the KDEs:
    # NOTE: we should fit the KDEs to the full sample, not just the points in the range (x_min, x_max); thus also use the total number of data points in the calculation of the bandwidths.
    bw_Scotts = len(radii_sample)**(-1./5)
    bw_low = bw_Scotts*bw_low_scotts_factor if bw_low=='Scotts' else bw_low
    bw_high = bw_Scotts*bw_high_scotts_factor if bw_high=='Scotts' else bw_high
    assert bw_low < bw_high, 'bw_high must be larger than bw_low!'
    kde_low = gaussian_kde(radii_sample, bw_method=bw_low)
    kde_high = gaussian_kde(radii_sample, bw_method=bw_high)
    
    # Evaluate the KDEs on an array of radii values:
    x_evals = np.linspace(x_min, x_max, 1001)
    kde_low_evals = kde_low(x_evals)
    kde_high_evals = kde_high(x_evals)
    
    i_valley = np.where((x_evals >= radius_valley_bounds[0]) & (x_evals <= radius_valley_bounds[1]))[0] # indices of the 'x_evals' that are in the radius valley

    # Find the minimum in the radius valley of the low-bandwidth KDE:
    i_min_valley = np.argmin(kde_low_evals[i_valley]) + i_valley[0]
    min_valley = kde_low_evals[i_min_valley]
    
    # Compute the "depth" of the radius valley:
    height = kde_high_evals[i_min_valley]
    depth = (height - min_valley)/height if fractional_depth else height - min_valley
    if depth < 0:
        print(f'WARNING: the depth ({depth}) is negative! There is no valley in the given radius valley bounds.')
    
    if verbose:
        print(f'min_valley = {min_valley} at radius = {x_evals[i_min_valley]}')
        print('#')
        print(f'depth = {depth}')
        print('#####')
    
    # To also make a plot:
    if plot_fig:
        ax = plot_fig_pdf_simple([radii_sample], [], x_min=x_min, x_max=x_max, n_bins=n_bins, normalize=True, labels_sim=[None], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]')
        bins = np.linspace(x_min, x_max, n_bins+1) # these are the bins the call above would be using to create the histogram; needed to normalize the KDE density
        fnorm = bins[1]-bins[0]
        plt.plot(x_evals, fnorm*kde_low(x_evals), color='b', label=r'KDE fit ($bw_{{\rm low}} = {:0.2f}$)'.format(bw_low)) # should appear to capture the radius valley
        plt.plot(x_evals, fnorm*kde_high(x_evals), color='k', label=r'KDE fit ($bw_{{\rm high}} = {:0.2f}$)'.format(bw_high)) # should appear to smooth over the radius valley
        plt.axvline(radius_valley_bounds[0], ls=':', lw=1)
        plt.axvline(radius_valley_bounds[1], ls=':', lw=1)
        ax.annotate('', xy=(x_evals[i_min_valley], fnorm*min_valley), xytext=(x_evals[i_min_valley], fnorm*height), arrowprops=dict(arrowstyle='->', lw=2, color='r'))
        plt.figtext(0.92, 0.9, r'Valley depth = {:.2f}'.format(depth), color='r', fontsize=16, ha='right', va='top')
        plt.legend(loc='upper right', bbox_to_anchor=(1,0.9), ncol=1, frameon=False, fontsize=16)
        
        if save_fig:
            plt.savefig(save_name)
            plt.close()
    
    return depth





#'''
##### To plot...

n_bins = 100
lw = 2 # linewidth
alpha = 0.2 # transparency of histograms

afs = 20 # axes labels font size
tfs = 20 # text labels font size
lfs = 16 # legend labels font size

##### To load and compute the same statistics for a large number of models:

#loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/ACI/Simulated_Data/AMD_system/Split_stars/Singles_ecc/Params11_KS/Distribute_AMD_per_mass/durations_norm_circ_singles_multis_GF2020_KS/GP_best_models/'
loadfiles_directory = '/Users/hematthi/Documents/GradSchool/Research/SysSim/Simulated_catalogs/Hybrid_NR20_AMD_model1/Fit_some8_KS/Params9_fix_highM/GP_best_models_100/'
runs = 100

sss_all = []
sss_per_sys_all = []
params_all = []

radii_measures = {'KS_dist_w': [],
                  'AD_dist_w': [],
                  'EMD': [],
                  'depth_binned': [],
                  'depth_kde': [],
                  'depth_two_kdes': []}

bw_factor = 0.25 # factor for multiplying the KDE bandwidth from Scott's rule

for i in range(1,runs+1):
    run_number = i
    sss_per_sys_i, sss_i = compute_summary_stats_from_cat_obs(file_name_path=loadfiles_directory, run_number=run_number, compute_ratios=compute_ratios)
    params_i = read_sim_params(loadfiles_directory + 'periods%s.out' % run_number)
    dists_i, dists_w_i = compute_distances_sim_Kepler(sss_per_sys_i, sss_i, ssk_per_sys, ssk, weights_all['all'], dists_include, N_Kep, cos_factor=cos_factor)
    
    EMD_radii_i = wasserstein_distance(sss_i['radii_obs'], ssk['radii_obs'])
    depth_binned_i = calculate_radius_valley_depth_using_global_bins(sss_i['radii_obs'], n_bins=n_bins)
    depth_kde_i = calculate_radius_valley_depth_using_kde(sss_i['radii_obs'], bw_scotts_factor=bw_factor)
    depth_two_kdes_i = calculate_radius_valley_depth_using_two_kdes(sss_i['radii_obs'], bw_low_scotts_factor=bw_factor, bw_high_scotts_factor=2.)
    
    sss_all.append(sss_i)
    sss_per_sys_all.append(sss_per_sys_i)
    params_all.append(params_i)
    
    radii_measures['KS_dist_w'].append(dists_w_i['radii_KS'])
    radii_measures['AD_dist_w'].append(dists_w_i['radii_AD'])
    radii_measures['EMD'].append(EMD_radii_i)
    radii_measures['depth_binned'].append(depth_binned_i)
    radii_measures['depth_kde'].append(depth_kde_i)
    radii_measures['depth_two_kdes'].append(depth_two_kdes_i)

#####





##### To calculate the depth for the Kepler catalog first:

depth_binned_Kep = calculate_radius_valley_depth_using_global_bins(ssk['radii_obs'], plot_fig=True)
depth_kde_Kep = calculate_radius_valley_depth_using_kde(ssk['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=True)
depth_two_kdes_Kep = calculate_radius_valley_depth_using_two_kdes(ssk['radii_obs'], bw_low_scotts_factor=bw_factor, bw_high_scotts_factor=2., plot_fig=True)
print(f'Kepler catalog: depth (binned) = {depth_binned_Kep}')
print(f'Kepler catalog: depth (kde) = {depth_kde_Kep}')
print(f'Kepler catalog: depth (two kdes) = {depth_two_kdes_Kep}')
plt.show()

##### To sort the catalogs by the 'depth' of the radius valley and plot them:

sort_by = 'depth_kde' # 'KS_dist_w', 'AD_dist_w', 'EMD', 'depth_binned', 'depth_kde', 'depth_two_kdes'

print(f'Sorting the simulated catalogs by {sort_by}...')
iSort = np.argsort(radii_measures[sort_by])
if 'depth' in sort_by:
    # If sorting by a measure of depth, reverse so the indices are sorted by decreasing order:
    iSort = iSort[::-1]

for i in iSort[:10]:
    run_number = i+1 # the catalog/run numbers are 1-based
    save_name = savefigures_directory + model_name + '_radii_valley_%s_catalog%s.pdf' % (sort_by, run_number)
    if sort_by == 'depth_binned':
        depth = calculate_radius_valley_depth_using_global_bins(sss_all[i]['radii_obs'], plot_fig=True, save_name=save_name, save_fig=savefigures)
    elif sort_by == 'depth_kde':
        depth = calculate_radius_valley_depth_using_kde(sss_all[i]['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=True, save_name=save_name, save_fig=savefigures)
    elif sort_by == 'depth_two_kdes':
        depth = calculate_radius_valley_depth_using_two_kdes(sss_all[i]['radii_obs'], bw_low_scotts_factor=bw_factor, bw_high_scotts_factor=2., plot_fig=True, save_name=save_name, save_fig=savefigures)
    else:
        # Sorted by a distance to the Kepler catalog, so just use your favorite measure:
        depth = calculate_radius_valley_depth_using_kde(sss_all[i]['radii_obs'], bw_scotts_factor=bw_factor, plot_fig=True)
        print(f'i={i}: depth = {depth}, {sort_by} = {radii_measures[sort_by][i]}')
    print(f'i={i}: depth = {depth}')
plt.show()

##### To plot several catalogs with the largest depths along with the Kepler catalog:

N_plot = 10
norm = matplotlib.colors.Normalize(vmin=0., vmax=1.)
cmap = cm.ScalarMappable(norm=norm, cmap='Blues_r')
cmap.set_array([])

plot_fig_pdf_simple([sss_all[i]['radii_obs'] for i in iSort[:N_plot]] + [ssk['radii_obs']], [], x_min=radii_min, x_max=6., c_sim=[cmap.to_rgba(i) for i in np.linspace(0.1,0.6,N_plot)] + ['k'], ls_sim=['-']*N_plot + ['-'], lw=[1.]*N_plot + [2.], labels_sim=['Simulated catalogs \n''with largest valleys'] + [None]*(N_plot-1) + ['Kepler'], xlabel_text=r'Planet radius, $R_p$ [$R_\oplus$]', afs=afs, tfs=tfs, lfs=lfs, legend=True, fig_size=(10,6), fig_lbrt=[0.15,0.15,0.95,0.95]) # ls_sim=[(0,(1,1))]*N_plot + ['-']
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_compare_enlarged_draws_best%s.pdf' % N_plot)
    plt.close()
plt.show()





##### To plot the (active) model parameters of the simulated catalogs as a corner scatter plot, with the points colored by the depth:

# NOTE: when loading the parameters, the 'log' params are actually converted to unlogged values; see 'read_sim_params()':
active_params_names_symbols = {#'break_mass (M_earth)': r'$M_{p,\rm break}$ $(M_\oplus)$',
                               'log_rate_clusters': r'$\lambda_c$',
                               'log_rate_planets_per_cluster': r'$\lambda_p$',
                               'log_α_pret': r'$\alpha_{\rm ret}$',
                               'mean_ln_mass (ln M_earth)': r'$\mu_M$',
                               'norm_radius (R_earth)': r'$R_{p,\rm norm}$',
                               'power_law_P': r'$\alpha_P$',
                               'power_law_γ0': r'$\gamma_0$',
                               #'power_law_γ1': r'$\gamma_1$',
                               'power_law_σ0': r'$\sigma_0$',
                               #'power_law_σ1': r'$\sigma_1$',
                               'sigma_ln_mass (ln M_earth)': r'$\sigma_M$',
                               #'sigma_logperiod_per_pl_in_cluster': r'$\sigma_P$',
                               }
active_params_names = list(active_params_names_symbols.keys())
active_params_symbols = list(active_params_names_symbols.values())
active_params_all = np.array([[params[key] for key in active_params_names] for params in params_all]) # double list comprehension!

plot_points_corner(active_params_symbols, active_params_all, fpoints=radii_measures['depth_kde'], f_label='Depth (KDE)', cmap='Reds', points_size=10., fig_size=(16,16), save_name=savefigures_directory + model_name + '_params_depths_corner.pdf', save_fig=savefigures)
plt.show()





##### To plot the radii measures vs. each other to see how they correlate with each other:

radii_measures_pairs = [("KS_dist_w", "AD_dist_w"),
                        ("KS_dist_w", "EMD"),
                        #("KS_dist_w", "depth_binned"),
                        ("KS_dist_w", "depth_kde"),
                        #("KS_dist_w", "depth_two_kdes"),
                        ("AD_dist_w", "EMD"),
                        #("AD_dist_w", "depth_binned"),
                        ("AD_dist_w", "depth_kde"),
                        #("AD_dist_w", "depth_two_kdes"),
                        #("EMD", "depth_binned"),
                        ("EMD", "depth_kde"),
                        #("EMD", "depth_two_kdes"),
                        #("depth_binned", "depth_kde"),
                        #("depth_binned", "depth_two_kdes"),
                        #("depth_kde", "depth_two_kdes")
                        ]

N_panels = len(radii_measures_pairs)
cols = int(np.ceil(np.sqrt(N_panels))) # number of columns
rows = int(np.sqrt(N_panels)) if float(int(np.sqrt(N_panels)))*float(cols) >= N_panels else cols # number of rows, such that rows*cols >= N_panels

# To plot radii measures as scatter plots:
fig = plt.figure(figsize=(16,8))
plot = GridSpec(rows,cols,left=0.05,bottom=0.1,right=0.975,top=0.95,wspace=0.3,hspace=0.5)
for i,pair in enumerate(radii_measures_pairs):
    i_row, i_col = i//cols, i%cols
    ax = plt.subplot(plot[i_row,i_col])
    sc = plt.scatter(radii_measures[pair[0]], radii_measures[pair[1]], marker='.', c='k', s=10, alpha=1)
    ax.tick_params(axis='both', labelsize=12)
    plt.xlabel(pair[0], fontsize=16)
    plt.ylabel(pair[1], fontsize=16)
if savefigures:
    plt.savefig(savefigures_directory + model_name + '_radii_measures_2D.pdf')
    plt.close()
plt.show()
