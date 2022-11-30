# To import required modules:
import numpy as np
import matplotlib
#import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
from matplotlib import ticker #for setting contour plots to log scale
from matplotlib.colors import LogNorm #for log color scales
from mpl_toolkits.axes_grid1.inset_locator import inset_axes #for inset axes
import corner #corner.py package for corner plots
import scipy.stats

import syssimpyplots.general as gen
import syssimpyplots.compare_kepler as ckep
import syssimpyplots.load_sims as lsims





# Functions for plotting histograms and cumulative distributions:

def setup_fig_single(fig_size, left, bottom, right, top):
    """
    Set up a single-panel figure using `matplotlib.gridspec.GridSpec <https://matplotlib.org/stable/api/_as_gen/matplotlib.gridspec.GridSpec.html>`_.

    Parameters
    ----------
    fig_size : tuple
        The figure size, e.g. (16,8).
    left : float
        The left margin of the panel (between 0 and 1).
    bottom: float
        The bottom margin of the panel (between 0 and 1).
    right : float
        The right margin of the panel (between 0 and 1).
    top : float
        The top margin of the panel (between 0 and 1).

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The plotting axes for the figure.
    """
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1,1, left=left, bottom=bottom, right=right, top=top, wspace=0, hspace=0)
    ax = plt.subplot(plot[0,0])
    return ax

def plot_panel_counts_hist_simple(ax, x_sim, x_Kep, x_min=0, x_max=None, y_min=None, y_max=None, x_llim=None, x_ulim=None, normalize=False, N_sim_Kep_factor=1., log_y=False, c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ms_Kep=['x'], lines_Kep=False, lw=1, labels_sim=['Simulated'], labels_Kep=['Kepler'], xticks_custom=None, xlabel_text='x', ylabel_text='Number', afs=20, tfs=20, lfs=16, legend=False, show_counts_sim=False, show_counts_Kep=False):
    """
    Plot histogram(s) of discrete integer values on a given panel.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes to plot on.
    x_sim : list[array[int]]
        A list with sample(s) of integers (e.g. simulated data).
    x_Kep : list[array[int]]
        A list with sample(s) of integers (e.g. Kepler data).
    x_min : float, default=0
        The minimum value to include.
    x_max : float, optional
        The maximum value to include.
    y_min : float, optional
        The lower y-axis limit.
    y_max : float, optional
        The upper y-axis limit.
    x_llim : float, optional
        The lower x-axis limit.
    x_ulim : float, optional
        The upper x-axis limit.
    normalize : bool, default=False
        Whether to normalize the histograms. If True, each histogram will sum to one.
    N_sim_Kep_factor : float, default=1.
        The number of simulated targets divided by the number of Kepler targets. If `normalize=False`, will divide the bin counts for the simulated data (`x_sim`) by this value to provide an equivalent comparison to the Kepler counts.
    log_y : bool, default=False
        Whether to plot the y-axis on a log-scale.
    c_sim : list[str], default=['k']
        A list of plotting colors for the histograms of simulated data.
    c_Kep : list[str], default=['k']
        A list of plotting colors for the histograms of Kepler data.
    ls_sim : list[str], default=['-']
        A list of line styles for the histograms of simulated data.
    ms_Kep : list[str], default=['x']
        A list of marker shapes for the histograms of Kepler data.
    lines_Kep : bool, default=False
        Whether to also draw the histogram lines for the Kepler data. Default value (False) will just draw marker points (given by `ms_Kep`) for the Kepler counts in each bin.
    lw : float, default=1
        The line width for the histograms.
    labels_sim : list[str], default=['Simulated']
        A list of legend labels for the histograms of simulated data.
    labels_Kep : list[str], default=['Kepler']
        A list of legend labels for the histograms of Kepler data.
    xticks_custom : list or array[float], optional
        The x-values at which to plot ticks.
    xlabel_text : str, default='x'
        The x-axis label.
    ylabel_text : str, default='Number'
        The y-axis label.
    afs : int, default=20
        The axes fontsize.
    tfs : int, default=20
        The text fontsize.
    lfs : int, default=16
        The legend fontsize.
    legend : bool, default=False
        Whether to show the legend.
    show_counts_sim : bool, default=False
        Whether to show the individual bin counts for the simulated data.
    show_counts_Kep : bool, default=True
        Whether to show the individual bin counts for the Kepler data.
    """
    if y_min == None:
        y_min = 1e-4 if normalize==True else 1
    if x_max == None:
        #x_max = np.nanmax([np.max(x) if len(x) > 0 else np.nan for x in x_sim+x_Kep])
        x_max = max([np.max(x) for x in x_sim+x_Kep])
    bins = np.histogram([], bins=(x_max - x_min)+1, range=(x_min-0.5, x_max+0.5))[1]
    bins_mid = (bins[:-1] + bins[1:])/2.
    if x_llim == None:
        x_llim = x_min # x_min is the minimum for the bins, while x_llim is the minimum of the x-axis for plotting
    if x_ulim == None:
        x_ulim = x_max+0.5 # x_max is the minimum for the bins, while x_ulim is the maximum of the x-axis for plotting

    for i,x in enumerate(x_sim):
        counts = np.histogram(x, bins=bins)[0]/float(N_sim_Kep_factor)
        counts_normed = counts/float(np.sum(counts))
        if normalize:
            counts_plot = counts_normed
        else:
            counts_plot = counts
        plt.plot(bins_mid, counts_plot, drawstyle='steps-mid', color=c_sim[i], ls=ls_sim[i], lw=lw, label=labels_sim[i])
        if show_counts_sim:
            for j,count in enumerate(counts):
                if j>0:
                    plt.text(bins_mid[j], 2.*y_min*(4**(len(x_sim)-i-1.)), '{:0.1f}'.format(count), ha='center', color=c_sim[i], fontsize=lfs)
    for i,x in enumerate(x_Kep):
        counts = np.histogram(x, bins=bins)[0]
        counts_normed = counts/float(np.sum(counts))
        if normalize:
            counts_plot = counts_normed
        else:
            counts_plot = counts
        plt.scatter(bins_mid, counts_plot, marker=ms_Kep[i], color=c_Kep[i], label=labels_Kep[i])
        if lines_Kep:
            plt.plot(bins_mid, counts_plot, drawstyle='steps-mid', color=c_Kep[i], ls='-', lw=lw)
        if show_counts_Kep:
            for j,count in enumerate(counts):
                plt.text(bins_mid[j], 2.*(4**(len(x_Kep)-i-1.))*counts_plot[j], str(count), ha='center', color=c_Kep[i], fontsize=lfs)

    if log_y:
        plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    if xticks_custom is not None:
        ax.set_xticks(xticks_custom)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([x_llim, x_ulim]) # the extra 1 for the minimum is so we can calculate a zero-bin but not show it
    plt.ylim([y_min, y_max])
    plt.xlabel(xlabel_text, fontsize=tfs)
    plt.ylabel(ylabel_text, fontsize=tfs)
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs) #show the legend

def plot_fig_counts_hist_simple(fig_size, x_sim, x_Kep, x_min=0, x_max=None, y_min=None, y_max=None, x_llim=None, x_ulim=None, normalize=False, N_sim_Kep_factor=1., log_y=False, c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ms_Kep=['x'], lines_Kep=False, lw=1, labels_sim=['Simulated'], labels_Kep=['Kepler'], xticks_custom=None, xlabel_text='x', ylabel_text='Number', afs=20, tfs=20, lfs=16, legend=False, show_counts_sim=False, show_counts_Kep=False, fig_lbrt=[0.15, 0.2, 0.95, 0.925], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot a figure with histogram(s) of discrete integer values.

    Wrapper for the function :py:func:`syssimpyplots.plot_catalogs.plot_panel_counts_hist_simple`. Includes all of the parameters of that function, with the following additional parameters:

    Parameters
    ----------
    fig_size : tuple
        The figure size, e.g. '(16,8)'.
    fig_lbrt : list[float], default=[0.15, 0.2, 0.95, 0.925]
        The positions of the (left, bottom, right, and top) margins of the plotting panel (all values must be between 0 and 1).
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The plotting axes, if `save_fig=False`.
    """
    left, bottom, right, top = fig_lbrt
    ax = setup_fig_single(fig_size, left=left, bottom=bottom, right=right, top=top)

    plot_panel_counts_hist_simple(ax, x_sim, x_Kep, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, x_llim=x_llim, x_ulim=x_ulim, normalize=normalize, N_sim_Kep_factor=N_sim_Kep_factor, log_y=log_y, c_sim=c_sim, c_Kep=c_Kep, ls_sim=ls_sim, ms_Kep=ms_Kep, lines_Kep=lines_Kep, lw=lw, labels_sim=labels_sim, labels_Kep=labels_Kep, xticks_custom=xticks_custom, xlabel_text=xlabel_text, ylabel_text=ylabel_text, afs=afs, tfs=tfs, lfs=lfs, legend=legend, show_counts_sim=show_counts_sim, show_counts_Kep=show_counts_Kep)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax

def plot_panel_pdf_simple(ax, x_sim, x_Kep, x_min=None, x_max=None, y_min=None, y_max=None, n_bins=100, normalize=True, N_sim_Kep_factor=1., log_x=False, log_y=False, c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ls_Kep=['-'], lw=1, alpha=0.2, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text=None, xticks_custom=None, xlabel_text='x', ylabel_text='Fraction', afs=20, tfs=20, lfs=16, legend=False):
    """
    Plot histogram(s) of continuous distributions on a given panel.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes to plot on.
    x_sim : list[array[float]]
        A list with sample(s) of values (e.g. simulated data).
    x_Kep : list[array[float]]
        A list with sample(s) of values (e.g. Kepler data).
    x_min : float, optional
        The minimum value to include.
    x_max : float, optional
        The maximum value to include.
    y_min : float, optional
        The lower y-axis limit.
    y_max : float, optional
        The upper y-axis limit.
    n_bins : int, default=100
        The number of bins.
    normalize : bool, default=True
        Whether to normalize the histograms. If True, each histogram will sum to one.
    N_sim_Kep_factor : float, default=1.
        The number of simulated targets divided by the number of Kepler targets. If `normalize=False`, will divide the bin counts for the simulated data (`x_sim`) by this value to provide an equivalent comparison to the Kepler counts.
    log_x : bool, default=False
        Whether to plot the x-axis on a log-scale and use log-uniform bins.
    log_y : bool, default=False
        Whether to plot the y-axis on a log-scale.
    c_sim : list[str], default=['k']
        A list of plotting colors for the histograms of simulated data.
    c_Kep : list[str], default=['k']
        A list of plotting colors for the histograms of Kepler data.
    ls_sim : list[str], default=['-']
        A list of line styles for the histograms of simulated data.
    ls_Kep : list[str], default=['-']
        A list of line styles for the histograms of Kepler data.
    lw : float, default=1
        The line width for the histograms.
    alpha : float, default=0.2
        The transparency of the shading for the histograms of Kepler data (between 0 and 1).
    labels_sim : list[str], default=['Simulated']
        A list of legend labels for the histograms of simulated data.
    labels_Kep : list[str], default=['Kepler']
        A list of legend labels for the histograms of Kepler data.
    extra_text : str, optional
        Extra text to be displayed on the figure.
    xticks_custom : list or array[float], optional
        The x-values at which to plot ticks.
    xlabel_text : str, default='x'
        The x-axis label.
    ylabel_text : str, default='Fraction'
        The y-axis label.
    afs : int, default=20
        The axes fontsize.
    tfs : int, default=20
        The text fontsize.
    lfs : int, default=16
        The legend fontsize.
    legend : bool, default=False
        Whether to show the legend.
    """
    if x_min == None:
        x_min = np.nanmin([np.min(x) if len(x) > 0 else np.nan for x in x_sim+x_Kep])
    if x_max == None:
        x_max = np.nanmax([np.max(x) if len(x) > 0 else np.nan for x in x_sim+x_Kep])

    if log_x:
        bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
    else:
        bins = np.linspace(x_min, x_max, n_bins+1)

    for i,x in enumerate(x_sim):
        if normalize:
            weights = np.ones(len(x))/len(x)
        else:
            weights = np.ones(len(x))/N_sim_Kep_factor
        plt.hist(x, bins=bins, histtype='step', weights=weights, log=log_y, color=c_sim[i], ls=ls_sim[i], lw=lw, label=labels_sim[i])
    for i,x in enumerate(x_Kep):
        if normalize:
            weights = np.ones(len(x))/len(x)
        else:
            weights = np.ones(len(x))
        plt.hist(x, bins=bins, histtype='stepfilled', weights=weights, log=log_y, color=c_Kep[i], ls=ls_Kep[i], alpha=alpha, label=labels_Kep[i])

    if log_x:
        plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    if xticks_custom is not None:
        ax.set_xticks(xticks_custom)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel(xlabel_text, fontsize=tfs)
    plt.ylabel(ylabel_text, fontsize=tfs)
    plt.text(x=0.02, y=0.8, s=extra_text, ha='left', fontsize=lfs, transform=ax.transAxes)
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs) #show the legend

def plot_fig_pdf_simple(fig_size, x_sim, x_Kep, x_min=None, x_max=None, y_min=None, y_max=None, n_bins=100, normalize=True, N_sim_Kep_factor=1., log_x=False, log_y=False, c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ls_Kep=['-'], lw=1, alpha=0.2, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text=None, xticks_custom=None, xlabel_text='x', ylabel_text='Fraction', afs=20, tfs=20, lfs=16, legend=False, fig_lbrt=[0.15, 0.2, 0.95, 0.925], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot a figure with histogram(s) of continuous distributions.

    Wrapper for the function :py:func:`syssimpyplots.plot_catalogs.plot_panel_pdf_simple`. Includes all of the parameters of that function, with the following additional parameters:

    Parameters
    ----------
    fig_size : tuple
        The figure size, e.g. '(16,8)'.
    fig_lbrt : list[float], default=[0.15, 0.2, 0.95, 0.925]
        The positions of the (left, bottom, right, and top) margins of the plotting panel (all values must be between 0 and 1).
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The plotting axes, if `save_fig=False`.
    """
    left, bottom, right, top = fig_lbrt
    ax = setup_fig_single(fig_size, left=left, bottom=bottom, right=right, top=top)

    plot_panel_pdf_simple(ax, x_sim, x_Kep, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, n_bins=n_bins, normalize=normalize, N_sim_Kep_factor=N_sim_Kep_factor, log_x=log_x, log_y=log_y, c_sim=c_sim, c_Kep=c_Kep, ls_sim=ls_sim, ls_Kep=ls_Kep, lw=lw, alpha=alpha, labels_sim=labels_sim, labels_Kep=labels_Kep, extra_text=extra_text, xticks_custom=xticks_custom, xlabel_text=xlabel_text, ylabel_text=ylabel_text, afs=afs, tfs=tfs, lfs=lfs, legend=legend)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax

def plot_panel_cdf_simple(ax, x_sim, x_Kep, x_min=None, x_max=None, y_min=0., y_max=1., log_x=False, c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ls_Kep=['--'], lw=1, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text=None, xticks_custom=None, xlabel_text='x', ylabel_text='CDF', one_minus=False, afs=20, tfs=20, lfs=16, legend=False, label_dist=False):
    """
    Plot cumulative distribution functions (CDFs) for continuous distributions on a given panel.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes to plot on.
    x_sim : list[array[float]]
        A list with sample(s) of values (e.g. simulated data).
    x_Kep : list[array[float]]
        A list with sample(s) of values (e.g. Kepler data).
    x_min : float, optional
        The minimum value to include.
    x_max : float, optional
        The maximum value to include.
    y_min : float, default=0.
        The lower y-axis limit.
    y_max : float, default=1.
        The upper y-axis limit.
    log_x : bool, default=False
        Whether to plot the x-axis on a log-scale.
    c_sim : list[str], default=['k']
        A list of plotting colors for the CDFs of simulated data.
    c_Kep : list[str], default=['k']
        A list of plotting colors for the CDFs of Kepler data.
    ls_sim : list[str], default=['-']
        A list of line styles for the CDFs of simulated data.
    ls_Kep : list[str], default=['--']
        A list of line styles for the CDFs of Kepler data.
    lw : float, default=1
        The line width for the CDFs.
    labels_sim : list[str], default=['Simulated']
        A list of legend labels for the CDFs of simulated data.
    labels_Kep : list[str], default=['Kepler']
        A list of legend labels for the CDFs of Kepler data.
    extra_text : str, optional
        Extra text to be displayed on the figure.
    xticks_custom : list or array[float], optional
        The x-values at which to plot ticks.
    xlabel_text : str, default='x'
        The x-axis label.
    ylabel_text : str, default='CDF'
        The y-axis label.
    one_minus : bool, default=False
        Whether to plot one minus the CDF.
    afs : int, default=20
        The axes fontsize.
    tfs : int, default=20
        The text fontsize.
    lfs : int, default=16
        The legend fontsize.
    legend : bool, default=False
        Whether to show the legend.
    label_dist : bool, default=False
        Whether to compute and show the distances between the simulated and Kepler CDFs on the lower-right corner of the plot. If True, will compute the KS distance (using :py:func:`syssimpyplots.compare_kepler.KS_dist`) and the modified AD distance (using :py:func:`syssimpyplots.compare_kepler.AD_mod_dist`) between each pair of `x_sim` and `x_Kep` arrays.
    """
    if x_min == None:
        x_min = np.nanmin([np.min(x) if len(x) > 0 else np.nan for x in x_sim+x_Kep])
    if x_max == None:
        x_max = np.nanmax([np.max(x) if len(x) > 0 else np.nan for x in x_sim+x_Kep])

    for i,x in enumerate(x_sim):
        cdf = 1. - (np.arange(len(x))+1.)/float(len(x)) if one_minus else (np.arange(len(x))+1.)/float(len(x))
        x = np.sort(x)
        x = np.insert(x, 0, x[0])
        cdf = np.insert(cdf, 0, 1) if one_minus else np.insert(cdf, 0, 0) # to connect the first point to 0 (or 1 if 'one_minus' is True) so the CDF does not jump abruptly at the first data point
        plt.plot(x, cdf, drawstyle='steps-post', color=c_sim[i], ls=ls_sim[i], lw=lw, label=labels_sim[i])
    for i,x in enumerate(x_Kep):
        cdf = 1. - (np.arange(len(x))+1.)/float(len(x)) if one_minus else (np.arange(len(x))+1.)/float(len(x))
        x = np.sort(x)
        x = np.insert(x, 0, x[0])
        cdf = np.insert(cdf, 0, 1) if one_minus else np.insert(cdf, 0, 0) # to connect the first point to 0 (or 1 if 'one_minus' is True) so the CDF does not jump abruptly at the first data point
        plt.plot(x, cdf, drawstyle='steps-post', color=c_Kep[i], ls=ls_Kep[i], lw=lw, label=labels_Kep[i])
    if label_dist:
        if len(x_sim) == len(x_Kep):
            for i in range(len(x_sim)):
                dist_KS, dist_KS_pos = ckep.KS_dist(x_sim[i], x_Kep[i])
                dist_AD = ckep.AD_mod_dist(x_sim[i], x_Kep[i])
                plt.text(0.98, 0.2+(len(x_sim)-(i+1.))*0.3, r'$\mathcal{D}_{\rm KS} = %s$' % np.round(dist_KS, 3), color=c_Kep[i], ha='right', fontsize=lfs, transform=ax.transAxes)
                plt.text(0.98, 0.05+(len(x_sim)-(i+1.))*0.3, r'$\mathcal{D}_{\rm AD^\prime} = %s$' % np.round(dist_AD, 3), color=c_Kep[i], ha='right', fontsize=lfs, transform=ax.transAxes)
        else:
            print('Error: x_sim != x_Kep')

    if log_x:
        plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    if xticks_custom is not None:
        ax.set_xticks(xticks_custom)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel(xlabel_text, fontsize=tfs)
    plt.ylabel(ylabel_text, fontsize=tfs)
    plt.text(x=0.02, y=0.8, s=extra_text, ha='left', fontsize=lfs, transform=ax.transAxes)
    if legend:
        if one_minus:
            plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), ncol=1, frameon=False, fontsize=lfs) #show the legend
        else:
            plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs) #show the legend

def plot_fig_cdf_simple(fig_size, x_sim, x_Kep, x_min=None, x_max=None, y_min=0., y_max=1., log_x=False, c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ls_Kep=['--'], lw=1, labels_sim=['Simulated'], labels_Kep=['Kepler'], extra_text=None, xticks_custom=None, xlabel_text='x', ylabel_text='CDF', one_minus=False, afs=20, tfs=20, lfs=16, legend=False, label_dist=False, fig_lbrt=[0.15, 0.2, 0.95, 0.925], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot a figure with CDF(s) of continuous distributions.

    Wrapper for the function :py:func:`syssimpyplots.plot_catalogs.plot_panel_cdf_simple`. Includes all of the parameters of that function, with the following additional parameters:

    Parameters
    ----------
    fig_size : tuple
        The figure size, e.g. '(16,8)'.
    fig_lbrt : list[float], default=[0.15, 0.2, 0.95, 0.925]
        The positions of the (left, bottom, right, and top) margins of the plotting panel (all values must be between 0 and 1).
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.

    Returns
    -------
    ax : matplotlib.axes._subplots.AxesSubplot
        The plotting axes, if `save_fig=False`.
    """
    left, bottom, right, top = fig_lbrt
    ax = setup_fig_single(fig_size, left=left, bottom=bottom, right=right, top=top)

    plot_panel_cdf_simple(ax, x_sim, x_Kep, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, log_x=log_x, c_sim=c_sim, c_Kep=c_Kep, ls_sim=ls_sim, ls_Kep=ls_Kep, lw=lw, labels_sim=labels_sim, labels_Kep=labels_Kep, extra_text=extra_text, xticks_custom=xticks_custom, xlabel_text=xlabel_text, ylabel_text=ylabel_text, one_minus=one_minus, afs=afs, tfs=tfs, lfs=lfs, legend=legend, label_dist=label_dist)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax

def plot_fig_mult_cdf_simple(fig_size, x_sim, x_Kep, x_min=1, x_max=None, y_min=0., y_max=1., c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ls_Kep=['--'], lw=1, labels_sim=['Simulated'], labels_Kep=['Kepler'], xticks_custom=None, xlabel_text='x', ylabel_text='CDF', afs=20, tfs=20, lfs=16, legend=False, fig_lbrt=[0.15, 0.2, 0.95, 0.925], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot a figure with CDF(s) of discrete integer values.

    Analogous to the function :py:func:`syssimpyplots.plot_catalogs.plot_fig_cdf_simple` but for integer values; uses almost all of the same parameters.
    """
    left, bottom, right, top = fig_lbrt
    ax = setup_fig_single(fig_size, left=left, bottom=bottom, right=right, top=top)

    if x_max == None:
        #x_max = np.nanmax([np.max(x) if len(x) > 0 else np.nan for x in x_sim+x_Kep])
        x_max = max([np.max(x) for x in x_sim+x_Kep])

    for i,x in enumerate(x_sim):
        counts_cumu = np.array([sum(x <= xi) for xi in range(x_min, np.max(x)+1)])
        plt.plot(range(x_min, np.max(x)+1), counts_cumu/float(len(x)), drawstyle='steps-post', color=c_sim[i], ls=ls_sim[i], lw=lw, label=labels_sim[i])
    for i,x in enumerate(x_Kep):
        counts_cumu = np.array([sum(x <= xi) for xi in range(x_min, np.max(x)+1)])
        plt.plot(range(x_min, np.max(x)+1), counts_cumu/float(len(x)), drawstyle='steps-post', color=c_Kep[i], ls=ls_Kep[i], lw=lw, label=labels_Kep[i])

    ax.tick_params(axis='both', labelsize=afs)
    if xticks_custom is not None:
        ax.set_xticks(xticks_custom)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel(xlabel_text, fontsize=tfs)
    plt.ylabel(ylabel_text, fontsize=tfs)
    if legend:
        plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs) #show the legend

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax

def plot_fig_pdf_composite(x_sim_all, x_Kep_all, param_vals=None, x_mins=[None], x_maxs=[None], y_mins=[None], y_maxs=[None], n_bins=100, normalize=False, N_sim_Kep_factor=1., log_xs=[False], log_ys=[False], c_sim=['k'], c_Kep=['k'], ls_sim=['-'], ls_Kep=['-'], lw=1, alpha=0.2, labels_sim=['Simulated'], labels_Kep=['Kepler'], xticks_customs=[None], xlabel_texts=['x'], ylabel_texts=['Fraction'], afs=20, tfs=20, lfs=16, legend_panels=1, fig_size=(16,8), fig_lbrt=[0.15, 0.2, 0.95, 0.925], save_name='no_name_fig.pdf', save_fig=False):

    n = len(x_sim_all)
    if len(x_mins) < n:
        x_mins = x_mins*n
    if len(x_maxs) < n:
        x_maxs = x_maxs*n
    if len(y_mins) < n:
        y_mins = y_mins*n
    if len(y_maxs) < n:
        y_maxs = y_maxs*n
    if len(log_xs) < n:
        log_xs = log_xs*n
    if len(log_ys) < n:
        log_ys = log_ys*n
    if len(xticks_customs) < n:
        xticks_customs = xticks_customs*n
    if len(xlabel_texts) < n:
        xlabel_texts = xlabel_texts*n
    if len(ylabel_texts) < n:
        ylabel_texts = ylabel_texts*n

    extra_panels = 1 if param_vals!=None else 0 # extra 'panel' for listing param values
    panels = n + extra_panels
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(int(np.ceil(panels/2.)),2,left=fig_lbrt[0],bottom=fig_lbrt[1],right=fig_lbrt[2],top=fig_lbrt[3],wspace=0.15,hspace=0.6)

    #To print the parameter values:
    if param_vals is not None:
        nrows = 7
        for i,param in enumerate(param_vals):
            plt.figtext(x=0.55+0.16*int(i/float(nrows)), y=0.925-0.025*(i%nrows), s=r'%s = %s' % (lsims.param_symbols[param], np.round(param_vals[param],3)), fontsize=lfs)

    # Multiplicities: (assumed 0th index of "x_sim_all" and "x_Kep_all")
    ax = plt.subplot(plot[0,0])
    plot_panel_counts_hist_simple(ax, x_sim_all[0], x_Kep_all[0], x_min=x_mins[0], x_max=x_maxs[0], y_min=y_mins[0], y_max=y_maxs[0], x_llim=0.5, normalize=normalize, N_sim_Kep_factor=N_sim_Kep_factor, log_y=log_ys[0], c_sim=c_sim, c_Kep=c_Kep, ls_sim=ls_sim, ms_Kep=['x']*len(ls_Kep), lw=lw, labels_sim=labels_sim, labels_Kep=labels_Kep, xlabel_text=xlabel_texts[0], ylabel_text=ylabel_texts[0], afs=afs, tfs=tfs, lfs=lfs, legend=True, show_counts_Kep=True, show_counts_sim=True)

    for i in range(1,panels-1):
        row, col = int((i+1)/2), (i+1)%2
        ax = plt.subplot(plot[row,col])
        plot_panel_pdf_simple(ax, x_sim_all[i], x_Kep_all[i], x_min=x_mins[i], x_max=x_maxs[i], n_bins=n_bins, normalize=normalize, N_sim_Kep_factor=N_sim_Kep_factor, log_x=log_xs[i], log_y=log_ys[i], c_sim=c_sim, c_Kep=c_Kep, ls_sim=ls_sim, ls_Kep=ls_Kep, lw=lw, alpha=alpha, labels_sim=labels_sim, labels_Kep=labels_Kep, xticks_custom=xticks_customs[i], xlabel_text=xlabel_texts[i], ylabel_text=ylabel_texts[i], afs=afs, tfs=tfs, lfs=lfs, legend=True if legend_panels==i else False)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def load_cat_obs_and_plot_fig_pdf_composite(load_dir, weights, run_number='', Rstar_min=0., Rstar_max=10., Mstar_min=0., Mstar_max=10., teff_min=0., teff_max=1e4, bp_rp_min=-5., bp_rp_max=5., label_dist=True, AD_mod=True, dists_include=[], n_bins=100, lw=1, alpha=0.2, afs=12, tfs=12, lfs=12, save_name='no_name_fig.pdf', save_fig=False):

    #To load and analyze the simulated and Kepler observed catalogs:

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = lsims.read_targets_period_radius_bounds(load_dir + 'periods%s.out' % run_number)
    param_vals = lsims.read_sim_params(load_dir + 'periods%s.out' % run_number)

    sss_per_sys, sss = lsims.compute_summary_stats_from_cat_obs(file_name_path=load_dir, run_number=run_number, Rstar_min=Rstar_min, Rstar_max=Rstar_max, Mstar_min=Mstar_min, Mstar_max=Mstar_max, teff_min=teff_min, teff_max=teff_max, bp_rp_min=bp_rp_min, bp_rp_max=bp_rp_max)

    ssk_per_sys, ssk = ckep.compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, Rstar_min=Rstar_min, Rstar_max=Rstar_max, Mstar_min=Mstar_min, Mstar_max=Mstar_max, teff_min=teff_min, teff_max=teff_max, bp_rp_min=bp_rp_min, bp_rp_max=bp_rp_max)

    dists, dists_w = ckep.compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights, dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)



    #To plot the 'observed' distributions with the actual observed Kepler distributions:

    fig = plt.figure(figsize=(16,8))
    plot = GridSpec(4,3,left=0.075,bottom=0.075,right=0.975,top=0.975,wspace=0.15,hspace=0.5)

    #To print the parameter values:
    nrows = 7
    for i,param in enumerate(param_vals):
        plt.figtext(x=0.02+0.12*int(i/float(nrows)), y=0.95-0.025*(i%nrows), s=r'%s = %s' % (lsims.param_symbols[param], np.round(param_vals[param],3)), fontsize=tfs)

    ax = plt.subplot(plot[1,0])
    plt.title(r'$\mathcal{D}_W({\rm KS}) = %1.2f$; $\mathcal{D}_W({\rm AD}) = %1.2f$' % (dists_w['tot_dist_KS_default'], dists_w['tot_dist_AD_default']), fontsize=lfs)
    x = sss_per_sys['Mtot_obs'][sss_per_sys['Mtot_obs'] > 0]
    max_M = np.max((np.max(sss_per_sys['Mtot_obs']), np.max(ssk_per_sys['Mtot_obs'])))
    counts, bins = np.histogram(x, bins=max_M+1, range=(-0.5, max_M+0.5))
    bins_mid = (bins[:-1] + bins[1:])/2.
    plt.plot(bins_mid, counts/float(np.sum(counts)), 'o-', color='k', lw=lw, label='%s x5 pl (Sim)' % (sum(x)/5.))
    counts, bins = np.histogram(ssk_per_sys['Mtot_obs'], bins=bins)
    plt.plot(bins_mid, counts/float(np.sum(counts)), 'o--', color='k', alpha=alpha, label='%s pl (Kep)' % sum(ssk_per_sys['Mtot_obs']))
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim([1., max_M])
    #plt.xlim([1., 8.])
    plt.xlabel('Observed planets per system', fontsize=tfs)
    plt.ylabel('Fraction', fontsize=tfs)
    plt.legend(loc='lower left', bbox_to_anchor=(0.01,0.01), frameon=False, ncol=1, fontsize=lfs) #show the legend
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$|f_{\rm sim} - f_{\rm Kep}| = %1.4f$ ($%1.2f$)' % (dists['delta_f'], dists_w['delta_f']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'${\rm CRPD} = %1.4f$ ($%1.2f$)' % (dists['mult_CRPD'], dists_w['mult_CRPD']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.4, s=r'${\rm CRPD_r} = %1.4f$ ($%1.2f$)' % (dists['mult_CRPD_r'], dists_w['mult_CRPD_r']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[2,0])
    plot_panel_pdf_simple(ax, [sss['P_obs']], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, lw=lw, alpha=alpha, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['periods_KS'], dists_w['periods_KS']), ha='right', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['periods_AD'], dists_w['periods_AD']), ha='right', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[3,0])
    R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
    plot_panel_pdf_simple(ax, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, n_bins=n_bins, log_x=True, lw=lw, alpha=alpha, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['period_ratios_KS'], dists_w['period_ratios_KS']), ha='right', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['period_ratios_AD'], dists_w['period_ratios_AD']), ha='right', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[0,1])
    plot_panel_pdf_simple(ax, [sss['tdur_obs']], [ssk['tdur_obs']], x_max=15., n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['durations_KS'], dists_w['durations_KS']), ha='right', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['durations_AD'], dists_w['durations_AD']), ha='right', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[1,1])
    plot_panel_pdf_simple(ax, [np.log10(sss['xi_obs'])], [np.log10(ssk['xi_obs'])], x_min=-0.5, x_max=0.5, n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'$\log{\xi}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['duration_ratios_KS'], dists_w['duration_ratios_KS']), ha='right', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['duration_ratios_AD'], dists_w['duration_ratios_AD']), ha='right', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[2,1])
    plot_panel_pdf_simple(ax, [sss['D_obs']], [ssk['D_obs']], x_min=1e-5, x_max=1e-2, n_bins=n_bins, log_x=True, lw=lw, alpha=alpha, xlabel_text=r'$\delta$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['depths_KS'], dists_w['depths_KS']), ha='right', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['depths_AD'], dists_w['depths_AD']), ha='right', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[3,1])
    plot_panel_pdf_simple(ax, [sss['D_above_obs'], sss['D_below_obs']], [ssk['D_above_obs'], ssk['D_below_obs']], x_min=1e-5, x_max=1e-2, n_bins=n_bins, log_x=True, c_sim=['b','r'], c_Kep=['b','r'], ls_sim=['-','-'], ls_Kep=['-','-'], lw=lw, alpha=alpha, labels_sim=['Above', 'Below'], labels_Kep=[None, None], xlabel_text=r'$\delta$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=12) #show the legend
    if label_dist:
        plt.text(x=0.98, y=0.85, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['depths_above_KS'], dists_w['depths_above_KS']), ha='right', color='b', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.7, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['depths_above_AD'], dists_w['depths_above_AD']), ha='right', color='b', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.55, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['depths_below_KS'], dists_w['depths_below_KS']), ha='right', color='r', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.4, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['depths_below_AD'], dists_w['depths_below_AD']), ha='right', color='r', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[0,2])
    plot_panel_pdf_simple(ax, [sss['D_ratio_obs']], [ssk['D_ratio_obs']], x_min=0.1, x_max=10., n_bins=n_bins, log_x=True, lw=lw, alpha=alpha, xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_KS'], dists_w['radius_ratios_KS']), ha='right', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_AD'], dists_w['radius_ratios_AD']), ha='right', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[1,2])
    plot_panel_pdf_simple(ax, [sss['D_ratio_above_obs'], sss['D_ratio_below_obs'], sss['D_ratio_across_obs']], [ssk['D_ratio_above_obs'], ssk['D_ratio_below_obs'], ssk['D_ratio_across_obs']], x_min=0.1, x_max=10., n_bins=n_bins, log_x=True, c_sim=['b','r','k'], c_Kep=['b','r','k'], ls_sim=['-','-','-'], ls_Kep=['-','-','-'], lw=lw, alpha=alpha, labels_sim=['Above', 'Below', 'Across'], labels_Kep=[None, None, None], xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=12) #show the legend
    if label_dist:
        plt.text(x=0.98, y=0.85, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_above_KS'], dists_w['radius_ratios_above_KS']), ha='right', color='b', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.7, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_above_AD'], dists_w['radius_ratios_above_AD']), ha='right', color='b', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.55, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_below_KS'], dists_w['radius_ratios_below_KS']), ha='right', color='r', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.4, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_below_AD'], dists_w['radius_ratios_below_AD']), ha='right', color='r', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.25, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_across_KS'], dists_w['radius_ratios_across_KS']), ha='right', color='k', fontsize=12, transform = ax.transAxes)
        plt.text(x=0.98, y=0.1, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_across_AD'], dists_w['radius_ratios_across_AD']), ha='right', color='k', fontsize=12, transform = ax.transAxes)

    ax = plt.subplot(plot[2,2])
    plot_panel_pdf_simple(ax, [sss['radii_obs']], [ssk['radii_obs']], x_min=radii_min, x_max=radii_max, n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'$R_p (R_\oplus)$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[3,2])
    plot_panel_pdf_simple(ax, [sss['Rstar_obs']], [ssk['Rstar_obs']], x_max=3., n_bins=n_bins, lw=lw, alpha=alpha, xlabel_text=r'$R_\star (R_\odot)$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def load_cat_obs_and_plot_fig_pdf_composite_simple(load_dir, weights, run_number='', Rstar_min=0., Rstar_max=10., Mstar_min=0., Mstar_max=10., teff_min=0., teff_max=1e4, bp_rp_min=-5., bp_rp_max=5., label_dist=True, AD_mod=True, dists_include=[], n_bins=100, c_sim=['k'], lw=1, alpha=0.2, afs=12, tfs=12, lfs=12, fig_size=(16,8), save_name='no_name_fig.pdf', save_fig=False):

    #To load and analyze the simulated and Kepler observed catalogs:

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = lsims.read_targets_period_radius_bounds(load_dir + 'periods%s.out' % run_number)
    param_vals = lsims.read_sim_params(load_dir + 'periods%s.out' % run_number)

    sss_per_sys, sss = lsims.compute_summary_stats_from_cat_obs(file_name_path=load_dir, run_number=run_number, Rstar_min=Rstar_min, Rstar_max=Rstar_max, Mstar_min=Mstar_min, Mstar_max=Mstar_max, teff_min=teff_min, teff_max=teff_max, bp_rp_min=bp_rp_min, bp_rp_max=bp_rp_max)

    ssk_per_sys, ssk = ckep.compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, Rstar_min=Rstar_min, Rstar_max=Rstar_max, Mstar_min=Mstar_min, Mstar_max=Mstar_max, teff_min=teff_min, teff_max=teff_max, bp_rp_min=bp_rp_min, bp_rp_max=bp_rp_max)

    dists, dists_w = ckep.compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights, dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)



    #To plot the 'observed' distributions with the actual observed Kepler distributions:

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(4,3,left=0.075,bottom=0.1,right=0.975,top=0.95,wspace=0.2,hspace=0.5)

    #To print the parameter values:
    nrows = 8
    for i,param in enumerate(param_vals):
        plt.figtext(x=0.02+0.13*int(i/float(nrows)), y=0.95-0.025*(i%nrows), s=r'%s = %s' % (lsims.param_symbols[param], np.round(param_vals[param],3)), fontsize=lfs-2)

    ax = plt.subplot(plot[0,1])
    if label_dist:
        plt.title(r'$\mathcal{D}_W({\rm KS}) = %1.2f$; $\mathcal{D}_W({\rm AD}) = %1.2f$' % (dists_w['tot_dist_KS_default'], dists_w['tot_dist_AD_default']), fontsize=lfs)
    plot_panel_counts_hist_simple(ax, [sss_per_sys['Mtot_obs']], [ssk_per_sys['Mtot_obs']], x_min=0, y_min=1e-1, y_max=1e4, x_llim=0.5, N_sim_Kep_factor=float(N_sim)/ckep.N_Kep, log_y=True, c_sim=c_sim, lw=lw, xlabel_text='Observed planets per system', ylabel_text='', afs=afs, tfs=tfs-2, lfs=lfs, legend=True, show_counts_Kep=True, show_counts_sim=True)
    if label_dist:
        plt.text(x=0.02, y=0.45, s=r'$D_{f} = %1.4f$ ($%1.2f$)' % (dists['delta_f'], dists_w['delta_f']), ha='left', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.02, y=0.25, s=r'$\rho_{\rm CRPD} = %1.4f$ ($%1.2f$)' % (dists['mult_CRPD_r'], dists_w['mult_CRPD_r']), ha='left', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[1,0])
    plot_panel_pdf_simple(ax, [sss['P_obs']], [ssk['P_obs']], x_min=P_min, x_max=P_max, y_min=1e-3, y_max=0.1, n_bins=n_bins, log_x=True, log_y=True, c_sim=c_sim, lw=lw, alpha=alpha, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['periods_KS'], dists_w['periods_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['periods_AD'], dists_w['periods_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[1,1])
    R_max_cut = 30. #upper cut-off for plotting period ratios; np.max(sss['Rm_obs'])
    plot_panel_pdf_simple(ax, [sss['Rm_obs'][sss['Rm_obs'] < R_max_cut]], [ssk['Rm_obs'][ssk['Rm_obs'] < R_max_cut]], x_min=1., x_max=R_max_cut, y_max=0.075, n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, alpha=alpha, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['period_ratios_KS'], dists_w['period_ratios_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['period_ratios_AD'], dists_w['period_ratios_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[2,0])
    plot_panel_pdf_simple(ax, [sss['D_obs']], [ssk['D_obs']], x_min=1e-5, x_max=10.**(-2.), n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, alpha=alpha, xlabel_text=r'$\delta$', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['depths_KS'], dists_w['depths_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['depths_AD'], dists_w['depths_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[2,1])
    plot_panel_pdf_simple(ax, [sss['D_ratio_obs']], [ssk['D_ratio_obs']], x_min=0.1, x_max=10., y_max=0.05, n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, alpha=alpha, xlabel_text=r'$\delta_{i+1}/\delta_i$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_KS'], dists_w['radius_ratios_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['radius_ratios_AD'], dists_w['radius_ratios_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[3,0])
    plot_panel_pdf_simple(ax, [sss['tdur_obs']], [ssk['tdur_obs']], x_min=0., x_max=15., n_bins=n_bins, c_sim=c_sim, lw=lw, alpha=alpha, xlabel_text=r'$t_{\rm dur}$ (hrs)', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['durations_KS'], dists_w['durations_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['durations_AD'], dists_w['durations_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[3,1])
    plot_panel_pdf_simple(ax, [np.log10(sss['xi_res_obs'])], [np.log10(ssk['xi_res_obs'])], x_min=-0.5, x_max=0.5, y_max=0.1, n_bins=n_bins, c_sim=c_sim, lw=lw, labels_sim=['Near MMR'], labels_Kep=[None], alpha=alpha, xlabel_text=r'$\log{\xi}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['duration_ratios_mmr_KS'], dists_w['duration_ratios_mmr_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['duration_ratios_mmr_AD'], dists_w['duration_ratios_mmr_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[3,2])
    plot_panel_pdf_simple(ax, [np.log10(sss['xi_nonres_obs'])], [np.log10(ssk['xi_nonres_obs'])], x_min=-0.5, x_max=0.5, y_max=0.1, n_bins=n_bins, c_sim=c_sim, lw=lw, labels_sim=['Not near MMR'], labels_Kep=[None], alpha=alpha, xlabel_text=r'$\log{\xi}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    plt.legend(loc='upper left', bbox_to_anchor=(0.01,0.99), ncol=1, frameon=False, fontsize=lfs)
    if label_dist:
        plt.text(x=0.98, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['duration_ratios_nonmmr_KS'], dists_w['duration_ratios_nonmmr_KS']), ha='right', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.98, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['duration_ratios_nonmmr_AD'], dists_w['duration_ratios_nonmmr_AD']), ha='right', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[0,2])
    plot_panel_pdf_simple(ax, [sss['tdur_tcirc_obs']], [ssk['tdur_tcirc_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=c_sim, lw=lw, alpha=alpha, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    if label_dist:
        plt.text(x=0.02, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['durations_norm_circ_KS'], dists_w['durations_norm_circ_KS']), ha='left', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.02, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['durations_norm_circ_AD'], dists_w['durations_norm_circ_AD']), ha='left', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[1,2]) # observed singles
    plot_panel_pdf_simple(ax, [sss['tdur_tcirc_1_obs']], [ssk['tdur_tcirc_1_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=c_sim, lw=lw, labels_sim=['Singles'], labels_Kep=[''], alpha=alpha, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=True)
    if label_dist:
        plt.text(x=0.02, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['durations_norm_circ_singles_KS'], dists_w['durations_norm_circ_singles_KS']), ha='left', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.02, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['durations_norm_circ_singles_AD'], dists_w['durations_norm_circ_singles_AD']), ha='left', fontsize=lfs, transform = ax.transAxes)

    ax = plt.subplot(plot[2,2]) # observed multis
    plot_panel_pdf_simple(ax, [sss['tdur_tcirc_2p_obs']], [ssk['tdur_tcirc_2p_obs']], x_min=0., x_max=1.5, n_bins=n_bins, c_sim=c_sim, lw=lw, labels_sim=['Multis'], labels_Kep=[''], alpha=alpha, xlabel_text=r'$t_{\rm dur}/t_{\rm circ}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs, legend=True)
    if label_dist:
        plt.text(x=0.02, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists['durations_norm_circ_multis_KS'], dists_w['durations_norm_circ_multis_KS']), ha='left', fontsize=lfs, transform = ax.transAxes)
        plt.text(x=0.02, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists['durations_norm_circ_multis_AD'], dists_w['durations_norm_circ_multis_AD']), ha='left', fontsize=lfs, transform = ax.transAxes)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def load_cat_obs_and_plot_fig_pdf_split_bprp_GF2020_metrics(load_dir, weights_all, run_number='', label_dist=True, AD_mod=True, dists_include=[], n_bins=100, lw=1, alpha=0.2, afs=12, tfs=12, lfs=12, fig_size=(16,8), save_name='no_name_fig.pdf', save_fig=False):

    #To load and analyze the simulated and Kepler observed catalogs:

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = lsims.read_targets_period_radius_bounds(load_dir + 'periods%s.out' % run_number)
    param_vals = lsims.read_sim_params(load_dir + 'periods%s.out' % run_number)

    #To plot the 'observed' distributions with the actual observed Kepler distributions:

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(4,3,left=0.075,bottom=0.1,right=0.975,top=0.95,wspace=0.2,hspace=0.5)

    #To print the parameter values:
    nrows = 8
    for i,param in enumerate(param_vals):
        plt.figtext(x=0.02+0.13*int(i/float(nrows)), y=0.95-0.025*(i%nrows), s=r'%s = %s' % (lsims.param_symbols[param], np.round(param_vals[param],3)), fontsize=lfs-2)

    stars_cleaned = ckep.load_Kepler_stars_cleaned()
    #bp_rp_med = np.nanmedian(stars_cleaned['bp_rp'])
    bp_rp_corr_med = np.nanmedian(stars_cleaned['bp_rp'] - stars_cleaned['e_bp_rp_interp'])

    sample_names = ['all', 'bluer', 'redder']
    sample_colors = ['k', 'b', 'r']
    sample_bprp_min = [0., 0., bp_rp_corr_med]
    sample_bprp_max = [1e6, bp_rp_corr_med, 1e6]

    GF2020_metrics = ['radii_partitioning', 'radii_monotonicity', 'gap_complexity']
    x_mins = [1e-5, -0.5, 0.]
    x_maxs = [1., 0.6, 1.]
    log_xs = [True, False, False]
    xlabel_texts = [r'$\mathcal{Q}_R$', r'$\mathcal{M}_R$', r'$\mathcal{C}$']

    for i,sample in enumerate(sample_names):
        sss_per_sys, sss = lsims.compute_summary_stats_from_cat_obs(file_name_path=load_dir, run_number=run_number, bp_rp_min=sample_bprp_min[i], bp_rp_max=sample_bprp_max[i])
        ssk_per_sys, ssk = ckep.compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max, bp_rp_min=sample_bprp_min[i], bp_rp_max=sample_bprp_max[i])
        dists, dists_w = ckep.compute_distances_sim_Kepler(sss_per_sys, sss, ssk_per_sys, ssk, weights_all[sample], dists_include, N_sim, cos_factor=cos_factor, AD_mod=AD_mod)

        for j,key in enumerate(GF2020_metrics):
            ax = plt.subplot(plot[j+1,i])
            plot_panel_pdf_simple(ax, [sss_per_sys[GF2020_metrics[j]]], [ssk_per_sys[GF2020_metrics[j]]], x_min=x_mins[j], x_max=x_maxs[j], n_bins=n_bins, log_x=log_xs[j], c_sim=[sample_colors[i]], lw=lw, alpha=alpha, xlabel_text=xlabel_texts[j], afs=afs, tfs=tfs, lfs=lfs)
            if label_dist:
                plt.text(x=0.02, y=0.8, s=r'$\mathcal{D}_{\rm KS} = %1.4f$ ($%1.2f$)' % (dists[GF2020_metrics[j]+'_KS'], dists_w[GF2020_metrics[j]+'_KS']), ha='left', fontsize=lfs, transform = ax.transAxes)
                plt.text(x=0.02, y=0.6, s=r'$\mathcal{D}_{\rm AD} = %1.4f$ ($%1.2f$)' % (dists[GF2020_metrics[j]+'_AD'], dists_w[GF2020_metrics[j]+'_AD']), ha='left', fontsize=lfs, transform = ax.transAxes)

    if save_fig:
        plt.savefig(save_name)
        plt.close()



def load_cat_phys_and_plot_fig_pdf_composite_simple(load_dir, run_number='', n_bins=100, c_sim=['k'], lw=1, afs=12, tfs=12, lfs=12, fig_size=(16,8), save_name='no_name_fig.pdf', save_fig=False):

    #To load and analyze the simulated physical catalogs:

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = lsims.read_targets_period_radius_bounds(load_dir + 'periods%s.out' % run_number)
    param_vals = lsims.read_sim_params(load_dir + 'periods%s.out' % run_number)

    sssp_per_sys, sssp = lsims.compute_summary_stats_from_cat_phys(file_name_path=load_dir, run_number=run_number)



    #To plot the underlying distributions:

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(4,3,left=0.075,bottom=0.075,right=0.975,top=0.975,wspace=0.15,hspace=0.5)

    #To print the parameter values:
    nrows = 8
    for i,param in enumerate(param_vals):
        plt.figtext(x=0.02+0.13*int(i/float(nrows)), y=0.95-0.025*(i%nrows), s=r'%s = %s' % (lsims.param_symbols[param], np.round(param_vals[param],3)), fontsize=lfs-2)

    ax = plt.subplot(plot[1,0])
    x = np.concatenate((sssp_per_sys['Mtot_all'], np.zeros(N_sim - len(sssp_per_sys['Mtot_all']), dtype='int')))
    plot_panel_counts_hist_simple(ax, [x], [], x_min=-1, x_llim=-0.5, x_ulim=10.5, normalize=True, c_sim=c_sim, lw=lw, xlabel_text='Intrinsic planet multiplicity', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[2,0])
    plot_panel_counts_hist_simple(ax, [sssp['clustertot_all']], [], x_llim=0.5, x_ulim=5.5, normalize=True, c_sim=c_sim, lw=lw, xlabel_text=r'Clusters per system $N_c$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[3,0])
    plot_panel_counts_hist_simple(ax, [sssp['pl_per_cluster_all']], [], x_llim=0.5, x_ulim=7.5, normalize=True, c_sim=c_sim, lw=lw, xlabel_text=r'Planets per cluster $N_c$', ylabel_text='Fraction', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[0,1])
    plot_panel_pdf_simple(ax, [sssp['P_all']], [], x_min=P_min, x_max=P_max, n_bins=n_bins, log_x=True, log_y=True, c_sim=c_sim, lw=lw, xticks_custom=[3,10,30,100,300], xlabel_text=r'$P$ (days)', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[0,2])
    plot_panel_pdf_simple(ax, [sssp['Rm_all']], [], x_min=1., x_max=20., n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, xticks_custom=[1,2,3,4,5,10,20], xlabel_text=r'$P_{i+1}/P_i$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[1,1])
    plot_panel_pdf_simple(ax, [sssp['radii_all']], [], x_min=radii_min, x_max=radii_max, n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, xticks_custom=[0.5,1,2,4,10], xlabel_text=r'$R_p$ ($R_\oplus$)', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[1,2])
    plot_panel_pdf_simple(ax, [sssp['radii_ratio_all']], [], x_min=1e-1, x_max=10., n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, xlabel_text=r'$R_{p,i+1}/R_{p,i}$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[2,1])
    plot_panel_pdf_simple(ax, [sssp['mass_all']], [], x_min=0.07, x_max=1e3, n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, xlabel_text=r'$M_p$ ($M_\oplus$)', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[2,2])
    plot_panel_pdf_simple(ax, [sssp['N_mH_all']], [], x_min=1., x_max=200., n_bins=n_bins, log_x=True, c_sim=c_sim, lw=lw, xlabel_text=r'$\Delta$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)

    ax = plt.subplot(plot[3,1])
    plot_panel_pdf_simple(ax, [sssp['e_all']], [], x_min=0., x_max=1., n_bins=n_bins, c_sim=c_sim, lw=lw, xlabel_text=r'$e$', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    ax_in = inset_axes(ax, width='75%', height='60%')
    plot_panel_pdf_simple(ax_in, [sssp['e_all']], [], x_min=0., x_max=0.1, n_bins=n_bins, c_sim=c_sim, lw=lw, xlabel_text='', ylabel_text='', afs=afs-2, tfs=tfs-2, lfs=lfs-2)

    ax = plt.subplot(plot[3,2])
    plot_panel_pdf_simple(ax, [sssp['inclmut_all']*(180./np.pi)], [], x_min=0., x_max=90., n_bins=n_bins, c_sim=c_sim, lw=lw, xlabel_text=r'$i_m$ (deg)', ylabel_text='', afs=afs, tfs=tfs, lfs=lfs)
    ax_in = inset_axes(ax, width='75%', height='60%')
    plot_panel_pdf_simple(ax, [sssp['inclmut_all']*(180./np.pi)], [], x_min=0., x_max=5., n_bins=n_bins, c_sim=c_sim, lw=lw, xlabel_text='', ylabel_text='', afs=afs-2, tfs=tfs-2, lfs=lfs-2)

    if save_fig:
        plt.savefig(save_name)
        plt.close()





# Functions for plotting galleries of systems:

def plot_figs_systems_gallery(x_per_sys, s_per_sys, x_min=2., x_max=300., log_x=True, s_norm=2., alpha=None, color_by='k', colors_per_sys=None, det_per_sys=None, llabel_per_sys=None, llabel_text=None, llabel_fmt=r'{:.2f}', rlabel_per_sys=None, rlabel_text=None, rlabel_fmt=r'{:.2f}', xticks_custom=None, xlabel_text=r'Period $P$ (days)', title=None, legend=False, s_examples=[1.,3.,10.], s_units=r'$R_\oplus$', legend_fmt=r'${:.0f}$', afs=16, tfs=16, lfs=8, sys_per_fig=100, line_every=1, max_figs=5, fig_size=(4,8), fig_lbrt=[0.1,0.1,0.9,0.95], save_name_base='gallery', save_fmt='png', save_fig=False):
    """
    Plot a gallery of systems to visualize their architectures.

    Parameters
    ----------
    x_per_sys : list[array[float]] or 2-d array[float]
        The properties of the planets in each system to plot on the x-axis. This can be a list of 1-d arrays, or a 2-d array (where each row is a separate system, potentially with padded values), typically of the orbital period (days).
    s_per_sys : list[array[float]] or 2-d array[float]
        The properties of the planets in each system to plot as the sizes of the points (which will be proportional to the squared values of `s_per_sys`). This can be a list of 1-d arrays, or a 2-d array (where each row is a separate system, potentially with padded values), typically of the planet radii (Earth radii).
    x_min : float, default=2.
        The lower limit for the x-axis.
    x_max : float, default=300.
        The upper limit for the x-axis.
    log_x : bool, default=True
        Whether to plot the x-axis on a log-scale.
    s_norm : float, default=2.
        The normalization for the sizes of the points.
    alpha : float, optional
        The transparency of the points, between 0 (transparent) and 1 (opaque).
    color_by : {'k', 'size_order', 'custom'}
        A string indicating the way the points are colored. The options are: 'k' to color all points in black; 'size_order' to color the points by their size ordering; 'custom' to color the points by custom values given by `colors_per_sys`.
    colors_per_sys : array[float], optional
        A 2-d array for coloring the planets in each system, used if `color_by='custom'`.
    det_per_sys : array[float], optional
        A 2-d array indicating which planets in each system are detected. If provided, will mark undetected planets with red outlines.
    llabel_per_sys : array[float], optional
        A 1-d array for left-labeling each system.
    llabel_text : str, optional
        The label for the left-labelings of each system.
    llabel_fmt : str, optional
        The string formatting of the left-labels in `llabel_per_sys`.
    rlabel_per_sys : array[float], optional
        A 1-d array for right-labeling each system.
    rlabel_text : str, optional
        The label for the right-labelings of each system.
    rlabel_fmt : str, optional
        The string formatting of the right-labels in `rlabel_per_sys`.
    xticks_custom : list or array[float], optional
        The x-values at which to plot ticks.
    xlabel_text : str, default=r'Period $P$ (days)'
        The x-axis label.
    title : str, optional
        The title for the figures.
    legend : bool, default=False
        Whether to plot a legend. Note that this will not make an actual legend object, but will simply plot an additional "system" at the top with several example planets of sizes given by `s_examples`.
    s_examples : list or array[float], default=[1.,3.,10.]
        The example planet sizes to plot in the legend. Must have the same units as the values in `s_per_sys`.
    s_units : str, default=r'$R_\oplus$'
        The text for displaying the units in the legend.
    legend_fmt : str, default=r'${:.0f}$'
        The string formatter for the example values in the legend.
    afs : int, default=16
        The axes fontsize.
    tfs : int, default=16
        The text fontsize.
    lfs : int, default=8
        The labels fontsize.
    sys_per_fig : int, default=100
        The number of systems to plot on each figure.
    line_every : int, default=1
        The number of systems before a horizontal line is plotted.
    max_figs : int, default=5
        The maximum number of figures to generate.
    fig_size : tuple, default=(4,8)
        The figure size.
    fig_lbrt : list[float], default=[0.1, 0.1, 0.9, 0.95]
        The positions of the (left, bottom, right, and top) margins of the plotting panel (all values must be between 0 and 1).
    save_name_base : str, default='gallery'
        The start of the file names for saving each figure.
    save_fmt : str, default='png'
        The file format for saving each figure (e.g., 'png', 'pdf', 'eps', 'jpg', etc.).
    save_fig : bool, default=False
        Whether to save the figures. If True, will save each figure in the working directory with the file names given by `save_name_base` with an index appended.
    """
    assert max_figs < 20 # extra limit to prevent accidentally making too many figures
    N_sys = len(x_per_sys)
    assert N_sys == len(s_per_sys)
    if colors_per_sys is not None:
        assert N_sys == len(colors_per_sys)
    if det_per_sys is not None:
        assert N_sys == len(det_per_sys)
    if llabel_per_sys is not None:
        assert N_sys == len(llabel_per_sys)
    if rlabel_per_sys is not None:
        assert N_sys == len(rlabel_per_sys)

    # Set up the number of figures:
    n_figs = int(np.ceil(N_sys/sys_per_fig))
    if n_figs > max_figs:
        print('Warning: too many systems (%s) to plot! Limiting number of figures to %s.' % (N_sys, max_figs))
        print('If you want to increase the number of systems to plot, increase either `sys_per_fig` or `max_figs`.')
        n_figs = max_figs

    # Plot each figure:
    print('Generating %s figures each showing %s systems...' % (n_figs, sys_per_fig))
    for i in range(n_figs):
        ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
        if title is not None:
            plt.title(title, fontsize=tfs)

        for j in range(len(x_per_sys[i*sys_per_fig:(i+1)*sys_per_fig])):
            x_sys_padded = x_per_sys[i*sys_per_fig + j]
            x_sys = np.copy(x_sys_padded)[x_sys_padded > 0] # to remove zero-padding
            s_sys = s_per_sys[i*sys_per_fig + j]
            s_sys = s_sys[s_sys > 0] # to remove zero-padding

            # Set the color scheme for the planets:
            if color_by == 'size_order':
                c_sys = np.argsort(s_sys)
            elif color_by == 'custom' and colors_per_sys is not None:
                c_sys = colors_per_sys[i*sys_per_fig + j]
                c_sys = c_sys[x_sys_padded > 0]
            else:
                c_sys = np.array(['k']*len(x_sys))

            # Plot the planets:
            if det_per_sys is not None: # plot detected and undetected planets differently
                det_sys = det_per_sys[i*sys_per_fig + j][x_sys_padded > 0] # still have to remove zero-padding, but according to `x_sys` since `det_sys` contains zeros for undetected planets
                bools_det = det_sys == 1 # boolean array for indicating detected/undetected planets
                plt.scatter(x_sys[bools_det], np.ones(np.sum(bools_det))+j, c=c_sys[bools_det], s=s_norm*s_sys[bools_det]**2., alpha=alpha) # detected planets
                plt.scatter(x_sys[~bools_det], np.ones(np.sum([~bools_det]))+j, c=c_sys[~bools_det], edgecolors='r', s=s_norm*s_sys[~bools_det]**2., alpha=alpha) # undetected planets marked with red outlines
            else: # plot all planets the same way if no `det_per_sys` provided
                plt.scatter(x_sys, np.ones(len(x_sys))+j, c=c_sys, s=s_norm*s_sys**2., alpha=alpha)

            # Optional: label each system (e.g. with some quantity)
            if llabel_per_sys is not None: # to put a left-label on the system
                plt.text(x=0.9*x_min, y=j+1, s=llabel_fmt.format(llabel_per_sys[i*sys_per_fig + j]), va='center', ha='right', fontsize=lfs)
            if rlabel_per_sys is not None: # to put a right-label on the system
                plt.text(x=1.1*x_max, y=j+1, s=rlabel_fmt.format(rlabel_per_sys[i*sys_per_fig + j]), va='center', ha='left', fontsize=lfs)
            if (j+1)%line_every == 0: # to draw a line through every `line_every` systems
                plt.axhline(y=j+1, lw=0.05, color='k')

        # Optional: plot a "system" to serve as a legend
        if legend:
            # Plot three example planets of different sizes for reference:
            x_examples = np.logspace(np.log10(x_min), np.log10(x_max), len(s_examples)+2)[1:-1] if log_x else np.linspace(x_min, x_max, 5)[1:-1] # first and last elements are for buffer zones
            plt.scatter(x_examples, np.ones(len(s_examples))+j+2, c='k', s=s_norm*np.array(s_examples)**2., alpha=alpha)
            for s,size in enumerate(s_examples):
                plt.annotate(legend_fmt.format(size) + s_units, (1.2*x_examples[s], j+3), ha='left', va='center', fontsize=lfs)
            plt.text(1.1*x_min, j+3, s='Legend:', ha='left', va='center', fontsize=lfs)
            plt.fill_between([x_min, x_max], j+2, j+4, color='b', alpha=0.2)

        if llabel_per_sys is not None and llabel_text is not None: # to label the left-label
            plt.text(x=0.9*x_min, y=j+2, s=llabel_text, va='center', ha='right', fontsize=lfs)
        if rlabel_per_sys is not None and rlabel_text is not None: # to label the right-label
            plt.text(x=1.1*x_max, y=j+2, s=rlabel_text, va='center', ha='left', fontsize=lfs)
        if log_x:
            plt.gca().set_xscale("log")
        ax.tick_params(axis='both', labelsize=afs)
        if xticks_custom is not None:
            ax.set_xticks(xticks_custom)
            ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks([])
        plt.xlim([x_min, x_max])
        plt.ylim([0., sys_per_fig+3 if legend else sys_per_fig+1])
        plt.xlabel(xlabel_text, fontsize=tfs)

        if n_figs==1:
            save_name = save_name_base + '.%s' % save_fmt
        else:
            save_name = save_name_base + '_%s.%s' % (i, save_fmt)
        if save_fig:
            plt.savefig(save_name)
            plt.close()

def plot_figs_observed_systems_gallery_from_cat_obs(ss_per_sys, sort_by='inner', n_min=3, n_max=20, x_min=2., x_max=300., log_x=True, s_norm=2., color_by='k', llabel=None, llabel_text=None, llabel_fmt=r'{:.2f}', xticks_custom=None, xlabel_text=r'Period $P$ (days)', title=None, legend=False, s_examples=[1.,3.,10.], s_units=r'$R_\oplus$', legend_fmt=r'${:.0f}$', afs=16, tfs=16, lfs=8, max_sys=100, sys_per_fig=100, line_every=1, max_figs=5, fig_size=(4,8), fig_lbrt=[0.1,0.1,0.9,0.95], save_name_base='gallery', save_fmt='png', save_fig=False):
    """
    Plot a gallery of systems from an observed catalog.

    Filters, samples, and sorts the systems before plotting them using the function :py:func:`syssimpyplots.plot_catalogs.plot_figs_systems_gallery`.

    Parameters
    ----------
    ss_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in an observed catalog (2-d and 1-d arrays). Can be a simulated catalog (i.e. returned by :py:func:`syssimpyplots.load_sims.compute_summary_stats_from_cat_obs`) or a Kepler catalog (i.e. returned by :py:func:`syssimpyplots.compare_kepler.compute_summary_stats_from_Kepler_catalog`).
    sort_by : {'inner', 'multiplicity', 'stellar_mass'}, optional
        The property to sort the systems by. Default is None (no sorting).
    n_min : int, default=3
        The minimum number of observed planets a system must have to be included.
    n_max : int, default=20
        The maximum number of observed planets a system must have to be included.
    llabel : {'multiplicity', 'stellar_mass'}, optional
        The property to left-label each system with. Default is None (no left-labels).


    The remaining parameters are described in the documentation for the function  :py:func:`syssimpyplots.plot_catalogs.plot_figs_systems_gallery`.
    """
    # TODO: add param `plot_p_or_a` : {'p', 'a'}
    assert n_min <= n_max

    # Sample the systems passing the multiplicity requirements:
    i_keep = np.arange(len(ss_per_sys['Mtot_obs']))[(ss_per_sys['Mtot_obs'] >= n_min) & (ss_per_sys['Mtot_obs'] <= n_max)]
    N_sys = len(i_keep)
    print('Total number of systems with %s <= m <= %s observed planets: %s' % (n_min, n_max, N_sys))
    if N_sys > max_sys:
        print('Total number of systems exceeds `max_sys`; sampling just %s systems.' % max_sys)
        i_keep = np.random.choice(i_keep, max_sys, replace=False)

    # Set up and sort the arrays of planets to be plotted:
    x_per_sys = ss_per_sys['P_obs'][i_keep]
    s_per_sys = ss_per_sys['radii_obs'][i_keep]
    if sort_by=='inner':
        i_sort = np.argsort(x_per_sys[:,0])
    elif sort_by=='multiplicity':
        i_sort = np.argsort(ss_per_sys['Mtot_obs'][i_keep])
    elif sort_by=='stellar_mass':
        i_sort = np.argsort(ss_per_sys['Mstar_obs'][i_keep])
    else:
        i_sort = np.arange(N_sys)
    x_per_sys = x_per_sys[i_sort]
    s_per_sys = s_per_sys[i_sort]

    # Set up the color scheme for the planets:
    if color_by=='k' or color_by=='size_order':
        pass
    elif color_by=='custom':
        print('WARNING: other custom coloring is not supported by this function. Defaulting to black points.')
        color_by = 'k'
    else:
        print('WARNING: `color_by` argument does not match any built-in options. Defaulting to black points.')
        color_by = 'k'

    # Set up the left-labels for each system, if any:
    if llabel=='multiplicity':
        llabel_per_sys = ss_per_sys['Mtot_obs'][i_keep][i_sort]
        llabel_text = 'Mult.' if llabel_text is None else llabel_text # if the user did not set a llabel_text
        llabel_fmt = r'{:d}'
    elif llabel=='stellar_mass':
        llabel_per_sys = ss_per_sys['Mstar_obs'][i_keep][i_sort]
        llabel_text = r'$M_\star$ ($M_\odot$)' if llabel_text is None else llabel_text # if the user did not set a llabel_text
    elif llabel is None:
        llabel_per_sys = None
    else:
        print('No key matching "%s" for `llabel` argument; omitting the left-labels for each system.' % llabel)

    # NOTE: `alpha`, `colors_per_sys`, `det_per_sys`, `rlabel_per_sys`, `rlabel_text`, and `rlabel_fmt` for ``plot_figs_systems_gallery()`` are inaccessible/unused by this function
    plot_figs_systems_gallery(x_per_sys, s_per_sys, x_min=x_min, x_max=x_max, log_x=log_x, s_norm=s_norm, color_by=color_by, llabel_per_sys=llabel_per_sys, llabel_text=llabel_text, llabel_fmt=llabel_fmt, xticks_custom=xticks_custom, xlabel_text=xlabel_text, title=title, legend=legend, s_examples=s_examples, s_units=s_units, legend_fmt=legend_fmt, afs=afs, tfs=tfs, lfs=lfs, sys_per_fig=sys_per_fig, line_every=line_every, max_figs=max_figs, fig_size=fig_size, fig_lbrt=fig_lbrt, save_name_base=save_name_base, save_fmt=save_fmt, save_fig=save_fig)

def plot_figs_physical_systems_gallery_from_cat_phys(sssp_per_sys, sssp, sort_by='inner', n_min=1, n_max=20, n_det_min=1, n_det_max=10, x_min=2., x_max=300., log_x=True, s_norm=2., color_by='k', mark_det=True, llabel=None, llabel_text=None, llabel_fmt=r'{:.2f}', xticks_custom=None, xlabel_text=r'Period $P$ (days)', title=None, legend=False, s_examples=[1.,3.,10.], s_units=r'$R_\oplus$', legend_fmt=r'${:.0f}$', afs=16, tfs=16, lfs=8, max_sys=100, sys_per_fig=100, line_every=1, max_figs=5, fig_size=(4,8), fig_lbrt=[0.1,0.1,0.9,0.95], save_name_base='gallery', save_fmt='png', save_fig=False):
    """
    Plot a gallery of systems from a physical catalog.

    Filters, samples, and sorts the systems before plotting them using the function :py:func:`syssimpyplots.plot_catalogs.plot_figs_systems_gallery`.

    Parameters
    ----------
    sssp_per_sys : dict
        The dictionary containing the planetary and stellar properties for each system in a physical catalog (2-d and 1-d arrays), such as one returned by :py:func:`syssimpyplots.load_sims.compute_summary_stats_from_cat_phys`).
    sort_by : {'inner', 'multiplicity', 'stellar_mass'}, optional
        The property to sort the systems by. Default is None (no sorting).
    n_min : int, default=3
        The minimum number of physical planets a system must have to be included.
    n_max : int, default=20
        The maximum number of physical planets a system must have to be included.
    n_det_min : int, default=1
        The minimum number of detected planets a system must have to be included.
    n_det_max : int, default=10
        The maximum number of detected planets a system must have to be included.
    color_by : {'k', 'size_order', 'cluster', 'mass', 'eccentricity', 'AMD'}
        The way in which the planets are colored.
    llabel : {'multiplicity', 'stellar_mass'}, optional
        The property to left-label each system with.
    mark_det : bool, default=True
        Whether to mark the detected and undetected planets separately.


    The remaining parameters are described in the documentation for the function  :py:func:`syssimpyplots.plot_catalogs.plot_figs_systems_gallery`.

    Note
    ----
    The `mark_det=True` option requires the ``sssp_per_sys`` object to have been generated with `load_full_tables=True` and `match_observed=True`!
    """
    # TODO: add param `plot_p_or_a` : {'p', 'a'}
    assert n_min <= n_max
    assert n_det_min <= n_det_max

    # Sample the systems passing the multiplicity requirements:
    n_per_sys = sssp_per_sys['Mtot_all']
    n_det_per_sys = np.sum(sssp_per_sys['det_all'], axis=1)
    bools_n_inrange = (n_per_sys >= n_min) & (n_per_sys <= n_max)
    bools_n_det_inrange = (n_det_per_sys >= n_det_min) & (n_det_per_sys <= n_det_max)
    i_keep = np.arange(len(n_per_sys))[bools_n_inrange & bools_n_det_inrange]
    N_sys = len(i_keep)
    print('Total number of systems with %s <= n <= %s physical planets and %s <= m <= %s observed planets: %s' % (n_min, n_max, n_det_min, n_det_max, N_sys))
    if N_sys > max_sys:
        print('Total number of systems exceeds `max_sys`; sampling just %s systems.' % max_sys)
        i_keep = np.random.choice(i_keep, max_sys, replace=False)

    # Set up and sort the arrays of planets to be plotted:
    x_per_sys = sssp_per_sys['P_all'][i_keep]
    s_per_sys = sssp_per_sys['radii_all'][i_keep]
    if sort_by=='inner':
        i_sort = np.argsort(x_per_sys[:,0])
    elif sort_by=='multiplicity':
        i_sort = np.argsort(sssp_per_sys['Mtot_all'][i_keep])
    elif sort_by=='stellar_mass':
        i_sort = np.argsort(sssp['Mstar_all'][i_keep])
    else:
        i_sort = np.arange(N_sys)
    x_per_sys = x_per_sys[i_sort]
    s_per_sys = s_per_sys[i_sort]

    # Set up the color scheme for the planets:
    colors_per_sys = None
    if color_by=='k' or color_by=='size_order':
        pass
    elif color_by=='cluster':
        colors_per_sys = sssp_per_sys['clusterids_all'][i_keep][i_sort]
    elif color_by=='mass':
        colors_per_sys = sssp_per_sys['mass_all'][i_keep][i_sort]
    elif color_by=='eccentricity':
        colors_per_sys = sssp_per_sys['e_all'][i_keep][i_sort]
    elif color_by=='AMD':
        colors_per_sys = sssp_per_sys['AMD_all'][i_keep][i_sort]
    else:
        print('WARNING: `color_by` argument does not match any built-in options. Defaulting to black points.')
        color_by = 'k'
    color_by = 'custom' if colors_per_sys is not None else color_by

    det_per_sys = sssp_per_sys['det_all'][i_keep][i_sort] if mark_det else None

    # Set up the left-labels for each system, if any:
    if llabel=='multiplicity':
        llabel_per_sys = sssp_per_sys['Mtot_all'][i_keep][i_sort]
        llabel_text = 'Mult.' if llabel_text is None else llabel_text # if the user did not set a llabel_text
        llabel_fmt = r'{:d}'
    elif llabel=='stellar_mass':
        llabel_per_sys = sssp['Mstar_all'][i_keep][i_sort]
        llabel_text = r'$M_\star$ ($M_\odot$)' if llabel_text is None else llabel_text # if the user did not set a llabel_text
    elif llabel is None:
        llabel_per_sys = None
    else:
        print('No key matching "%s" for `llabel` argument; omitting the left-labels for each system.' % llabel)

    # NOTE: `alpha`, `rlabel_per_sys`, `rlabel_text`, and `rlabel_fmt` for ``plot_figs_systems_gallery()`` are inaccessible/unused by this function
    plot_figs_systems_gallery(x_per_sys, s_per_sys, x_min=x_min, x_max=x_max, log_x=log_x, s_norm=s_norm, color_by=color_by, colors_per_sys=colors_per_sys, det_per_sys=det_per_sys, llabel_per_sys=llabel_per_sys, llabel_text=llabel_text, llabel_fmt=llabel_fmt, xticks_custom=xticks_custom, xlabel_text=xlabel_text, title=title, legend=legend, s_examples=s_examples, s_units=s_units, legend_fmt=legend_fmt, afs=afs, tfs=tfs, lfs=lfs, sys_per_fig=sys_per_fig, line_every=line_every, max_figs=max_figs, fig_size=fig_size, fig_lbrt=fig_lbrt, save_name_base=save_name_base, save_fmt=save_fmt, save_fig=save_fig)





# Functions for plotting period-radius distributions and grids:

def load_cat_obs_and_plot_fig_period_radius(load_dir, run_number='', lw=1, save_name='no_name_fig.pdf', save_fig=False):

    #To load and analyze the simulated and Kepler observed catalogs:

    N_sim, cos_factor, P_min, P_max, radii_min, radii_max = lsims.read_targets_period_radius_bounds(load_dir + 'periods%s.out' % run_number)
    param_vals = lsims.read_sim_params(load_dir + 'periods%s.out' % run_number)

    sss_per_sys, sss = lsims.compute_summary_stats_from_cat_obs(file_name_path=load_dir, run_number=run_number)

    ssk_per_sys, ssk = ckep.compute_summary_stats_from_Kepler_catalog(P_min, P_max, radii_min, radii_max)



    #To plot a period vs radius scatter plot with binned statistics to compare the simulated and Kepler catalogs:
    fig = plt.figure(figsize=(16,8))
    plot = GridSpec(1,2,left=0.075,bottom=0.1,right=0.975,top=0.75,wspace=0.2,hspace=0)

    #To print the parameter values:
    nrows = 7
    for i,param in enumerate(param_vals):
        plt.figtext(x=0.02+0.12*int(i/float(nrows)), y=0.95-0.025*(i%nrows), s=r'%s = %s' % (lsims.param_symbols[param], np.round(param_vals[param],3)), fontsize=12)

    P_bins = 5
    P_lines, radii_lines = np.logspace(np.log10(P_min), np.log10(P_max), P_bins+1), np.array([0.5, 1., 2., 4., 6., 8., 10.])
    radii_bins = len(radii_lines)-1

    ax = plt.subplot(plot[0,0])
    N_sample = int(np.round(len(sss_per_sys['P_obs'])*cos_factor)) #number of simulated planets we would expect if we assigned orbits isotropically
    i_sample = np.random.choice(np.arange(len(sss_per_sys['P_obs'])), N_sample, replace=False)
    plt.scatter(sss['P_obs'][i_sample], sss['radii_obs'][i_sample], c='k', marker='o')
    plt.scatter(ssk['P_obs'], ssk['radii_obs'], c='r', marker='o')
    for x in P_lines:
        plt.axvline(x, lw=lw, color='g')
    for y in radii_lines:
        plt.axhline(y, lw=lw, color='g')
    plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xticks([3,10,30,100,300])
    ax.set_yticks([0.5,2,4,6,8,10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([P_min, P_max])
    plt.ylim([radii_min, radii_max])
    plt.xlabel(r'$P$ (days)', fontsize=20)
    plt.ylabel(r'$R_p (R_\oplus)$', fontsize=20)

    ax = plt.subplot(plot[0,1])
    N_obs_grid = np.zeros((radii_bins, P_bins))
    N_confirmed_grid = np.zeros((radii_bins, P_bins))
    for j in range(radii_bins):
        for i in range(P_bins):
            N_obs_cell = np.sum((sss['P_obs'] > P_lines[i]) & (sss['P_obs'] < P_lines[i+1]) & (sss['radii_obs'] > radii_lines[j]) & (sss['radii_obs'] < radii_lines[j+1]))
            N_confirmed_cell = np.sum((ssk['P_obs'] > P_lines[i]) & (ssk['P_obs'] < P_lines[i+1]) & (ssk['radii_obs'] > radii_lines[j]) & (ssk['radii_obs'] < radii_lines[j+1]))
            N_obs_grid[j,i] = N_obs_cell
            N_confirmed_grid[j,i] = N_confirmed_cell
            plt.text(x=0.02+i*(1./P_bins), y=(j+1)*(1./radii_bins)-0.025, s='%s' % np.round(N_obs_cell*cos_factor, 1), ha='left', va='top', color='k', fontsize=16, transform = ax.transAxes)
            plt.text(x=0.02+i*(1./P_bins), y=(j+1)*(1./radii_bins)-0.075, s='%s' % N_confirmed_cell, ha='left', va='top', color='r', fontsize=16, transform = ax.transAxes)
            plt.text(x=0.02+i*(1./P_bins), y=(j+1)*(1./radii_bins)-0.125, s='%s' % np.round((N_obs_cell*cos_factor)/float(N_confirmed_cell), 2), ha='left', va='top', color='b', fontsize=16, fontweight='bold', transform = ax.transAxes)
    N_obs_normed_grid = N_obs_grid*cos_factor
    plt.imshow(N_obs_normed_grid/N_confirmed_grid, cmap='coolwarm', aspect='auto', interpolation="nearest", origin='lower') #extent=(3, 300, 0.5, 10)
    cbar = plt.colorbar()
    cbar.set_label(r'$N_{\rm Sim}/N_{\rm Kep}$', rotation=270, va='bottom', fontsize=20)
    #plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=20)
    plt.xticks(np.linspace(-0.5, P_bins-0.5, P_bins), [3,10,30,100,300])
    plt.yticks(np.linspace(-0.5, radii_bins-0.5, radii_bins+1), radii_lines)
    plt.xlabel(r'$P$ (days)', fontsize=20)

    plot = GridSpec(1,1,left=0.83,bottom=0.8,right=0.895,top=0.93,wspace=0,hspace=0) #just for the 'legend'
    ax = plt.subplot(plot[0,0])
    plt.text(x=0.05, y=0.9, s=r'$N_{\rm Sim}$', ha='left', va='top', color='k', fontsize=14, transform = ax.transAxes)
    plt.text(x=0.05, y=0.7, s=r'$N_{\rm Kep}$', ha='left', va='top', color='r', fontsize=14, transform = ax.transAxes)
    plt.text(x=0.05, y=0.5, s=r'$N_{\rm Sim}/N_{\rm Kep}$', ha='left', va='top', color='b', fontsize=14, transform = ax.transAxes)
    plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off', right='off', left='off', labelleft='off')

    if save_fig:
        plt.savefig(save_name)
        plt.close()



def plot_fig_period_radius_fraction_multis(sss_per_sys, sss, P_bins, R_bins, fig_size=(10,8), fig_lbrt=[0.15, 0.15, 0.95, 0.95], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    n_P_bins, n_R_bins = len(P_bins)-1, len(R_bins)-1

    counts_pl_grid = np.zeros((n_R_bins, n_P_bins))
    counts_sys_grid = np.zeros((n_R_bins, n_P_bins))
    counts_singles_grid = np.zeros((n_R_bins, n_P_bins))
    counts_multis_grid = np.zeros((n_R_bins, n_P_bins))
    f_multis_sys4p_grid = np.zeros((n_R_bins, n_P_bins))
    for j in range(n_R_bins):
        for i in range(n_P_bins):
            pl_cell = np.sum((sss_per_sys['P_obs'] > P_bins[i]) & (sss_per_sys['P_obs'] < P_bins[i+1]) & (sss_per_sys['radii_obs'] > R_bins[j]) & (sss_per_sys['radii_obs'] < R_bins[j+1]))
            P_per_sys_cell = sss_per_sys['P_obs'][np.any((sss_per_sys['P_obs'] > P_bins[i]) & (sss_per_sys['P_obs'] < P_bins[i+1]) & (sss_per_sys['radii_obs'] > R_bins[j]) & (sss_per_sys['radii_obs'] < R_bins[j+1]), axis=1)]
            sys_cell = len(P_per_sys_cell)
            singles_cell = np.sum(P_per_sys_cell[:,1] < 0)
            #print('Systems in cell = %s; planets in cell = %s; observed singles in cell = %s' % (sys_cell, pl_cell, singles_cell))

            counts_pl_grid[j,i] = pl_cell
            counts_sys_grid[j,i] = sys_cell
            counts_singles_grid[j,i] = singles_cell
            counts_multis_grid[j,i] = pl_cell - singles_cell
            f_multis_sys4p_grid[j,i] = (pl_cell - singles_cell)/pl_cell if pl_cell >=4 else np.nan

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1,1, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)

    ax = plt.subplot(plot[:,:])
    plt.imshow(f_multis_sys4p_grid, cmap='coolwarm', norm=LogNorm(vmin=0.1, vmax=1.), aspect='auto', interpolation="nearest", origin='lower', extent=(np.log10(P_bins[0]),np.log10(P_bins[-1]),np.log10(R_bins[0]),np.log10(R_bins[-1]))) #cmap='coolwarm'
    cbar = plt.colorbar(ticks=np.linspace(0.1,1.,10), format=ticker.ScalarFormatter())
    cbar.set_label(r'Multi-planet Fraction', rotation=90, va='top', fontsize=tfs)
    cbar.ax.tick_params(labelsize=afs)
    cbar.ax.minorticks_off()
    plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=1, marker='.', c='k')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim([np.log10(P_bins[0]), np.log10(P_bins[-1])])
    plt.ylim([np.log10(R_bins[0]), np.log10(R_bins[-1])])
    plt.xticks(np.log10(P_bins), ['{:.1f}'.format(x) for x in P_bins])
    plt.yticks(np.log10(R_bins), ['{:.1f}'.format(x) for x in R_bins])
    plt.xlabel(r'Orbital Period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

    return f_multis_sys4p_grid

def plot_fig_period_radius_fraction_multis_higher(sss_per_sys, sss, P_bins, R_bins, fig_size=(10,8), fig_lbrt=[0.1, 0.1, 0.95, 0.95], afs=12, tfs=12, lfs=12, save_name='no_name_fig.pdf', save_fig=False):

    n_P_bins, n_R_bins = len(P_bins)-1, len(R_bins)-1

    counts_pl_grid = np.zeros((n_R_bins, n_P_bins))
    counts_sys_grid = np.zeros((n_R_bins, n_P_bins))
    counts_singles_grid = np.zeros((n_R_bins, n_P_bins))
    counts_multis_grid = np.zeros((n_R_bins, n_P_bins))
    f_multis_sys4p_grid = np.zeros((n_R_bins, n_P_bins))
    f_multis3p_sys4p_grid = np.zeros((n_R_bins, n_P_bins))
    f_multis4p_sys4p_grid = np.zeros((n_R_bins, n_P_bins))
    for j in range(n_R_bins):
        for i in range(n_P_bins):
            pl_cell = np.sum((sss_per_sys['P_obs'] > P_bins[i]) & (sss_per_sys['P_obs'] < P_bins[i+1]) & (sss_per_sys['radii_obs'] > R_bins[j]) & (sss_per_sys['radii_obs'] < R_bins[j+1]))
            P_per_sys_cell = sss_per_sys['P_obs'][np.any((sss_per_sys['P_obs'] > P_bins[i]) & (sss_per_sys['P_obs'] < P_bins[i+1]) & (sss_per_sys['radii_obs'] > R_bins[j]) & (sss_per_sys['radii_obs'] < R_bins[j+1]), axis=1)]
            sys_cell = len(P_per_sys_cell)
            singles_cell = np.sum(P_per_sys_cell[:,1] < 0)
            doubles_or_less_cell = np.sum(P_per_sys_cell[:,2] < 0)
            triples_or_less_cell = np.sum(P_per_sys_cell[:,3] < 0)
            #print('Systems in cell = %s; planets in cell = %s; observed singles in cell = %s' % (sys_cell, pl_cell, singles_cell))

            counts_pl_grid[j,i] = pl_cell
            counts_sys_grid[j,i] = sys_cell
            counts_singles_grid[j,i] = singles_cell
            counts_multis_grid[j,i] = pl_cell - singles_cell
            f_multis_sys4p_grid[j,i] = (pl_cell - singles_cell)/pl_cell if pl_cell >=4 else np.nan
            f_multis3p_sys4p_grid[j,i] = (pl_cell - doubles_or_less_cell)/pl_cell if pl_cell >=4 else np.nan
            f_multis4p_sys4p_grid[j,i] = (pl_cell - triples_or_less_cell)/pl_cell if pl_cell >=4 else np.nan

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(2,2, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0.4, hspace=0.4)

    ax = plt.subplot(plot[0,0])
    plt.imshow(1. - f_multis_sys4p_grid, cmap='coolwarm', norm=LogNorm(vmin=0.05, vmax=1.), aspect='auto', interpolation="nearest", origin='lower', extent=(np.log10(P_bins[0]),np.log10(P_bins[-1]),np.log10(R_bins[0]),np.log10(R_bins[-1]))) #cmap='coolwarm'
    cbar = plt.colorbar(ticks=np.linspace(0.1,1.,10), format=ticker.ScalarFormatter())
    cbar.set_label(r'Fraction in singles', rotation=90, va='top', fontsize=tfs)
    cbar.ax.tick_params(labelsize=afs)
    cbar.ax.minorticks_off()
    plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=1, marker='.', c='k')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim([np.log10(P_bins[0]), np.log10(P_bins[-1])])
    plt.ylim([np.log10(R_bins[0]), np.log10(R_bins[-1])])
    plt.xticks(np.log10(P_bins), ['{:.1f}'.format(x) for x in P_bins])
    plt.yticks(np.log10(R_bins), ['{:.1f}'.format(x) for x in R_bins])
    plt.xlabel(r'Orbital Period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

    ax = plt.subplot(plot[0,1])
    plt.imshow(f_multis_sys4p_grid, cmap='coolwarm', norm=LogNorm(vmin=0.05, vmax=1.), aspect='auto', interpolation="nearest", origin='lower', extent=(np.log10(P_bins[0]),np.log10(P_bins[-1]),np.log10(R_bins[0]),np.log10(R_bins[-1]))) #cmap='coolwarm'
    cbar = plt.colorbar(ticks=np.linspace(0.1,1.,10), format=ticker.ScalarFormatter())
    cbar.set_label(r'Fraction in multis (2+)', rotation=90, va='top', fontsize=tfs)
    cbar.ax.tick_params(labelsize=afs)
    cbar.ax.minorticks_off()
    plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=1, marker='.', c='k')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim([np.log10(P_bins[0]), np.log10(P_bins[-1])])
    plt.ylim([np.log10(R_bins[0]), np.log10(R_bins[-1])])
    plt.xticks(np.log10(P_bins), ['{:.1f}'.format(x) for x in P_bins])
    plt.yticks(np.log10(R_bins), ['{:.1f}'.format(x) for x in R_bins])
    plt.xlabel(r'Orbital Period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

    ax = plt.subplot(plot[1,0])
    plt.imshow(f_multis3p_sys4p_grid, cmap='coolwarm', norm=LogNorm(vmin=0.05, vmax=1.), aspect='auto', interpolation="nearest", origin='lower', extent=(np.log10(P_bins[0]),np.log10(P_bins[-1]),np.log10(R_bins[0]),np.log10(R_bins[-1]))) #cmap='coolwarm'
    cbar = plt.colorbar(ticks=np.linspace(0.1,1.,10), format=ticker.ScalarFormatter())
    cbar.set_label(r'Fraction in multis (3+)', rotation=90, va='top', fontsize=tfs)
    cbar.ax.tick_params(labelsize=afs)
    cbar.ax.minorticks_off()
    plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=1, marker='.', c='k')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim([np.log10(P_bins[0]), np.log10(P_bins[-1])])
    plt.ylim([np.log10(R_bins[0]), np.log10(R_bins[-1])])
    plt.xticks(np.log10(P_bins), ['{:.1f}'.format(x) for x in P_bins])
    plt.yticks(np.log10(R_bins), ['{:.1f}'.format(x) for x in R_bins])
    plt.xlabel(r'Orbital Period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

    ax = plt.subplot(plot[1,1])
    plt.imshow(f_multis4p_sys4p_grid, cmap='coolwarm', norm=LogNorm(vmin=0.05, vmax=1.), aspect='auto', interpolation="nearest", origin='lower', extent=(np.log10(P_bins[0]),np.log10(P_bins[-1]),np.log10(R_bins[0]),np.log10(R_bins[-1]))) #cmap='coolwarm'
    cbar = plt.colorbar(ticks=np.linspace(0.1,1.,10), format=ticker.ScalarFormatter())
    cbar.set_label(r'Fraction in multis (4+)', rotation=90, va='top', fontsize=tfs)
    cbar.ax.tick_params(labelsize=afs)
    cbar.ax.minorticks_off()
    plt.scatter(np.log10(sss['P_obs']), np.log10(sss['radii_obs']), s=1, marker='.', c='k')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim([np.log10(P_bins[0]), np.log10(P_bins[-1])])
    plt.ylim([np.log10(R_bins[0]), np.log10(R_bins[-1])])
    plt.xticks(np.log10(P_bins), ['{:.1f}'.format(x) for x in P_bins])
    plt.yticks(np.log10(R_bins), ['{:.1f}'.format(x) for x in R_bins])
    plt.xlabel(r'Orbital Period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'Planet radius $R_p$ ($R_\oplus$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()





# Functions for making various other plots:

def compute_pratio_in_out_and_plot_fig(p_per_sys_all, colors=['k'], labels=['Input'], xymax=50., xyticks_custom=None, afs=12, tfs=12, lfs=12, save_name='no_name_fig.pdf', save_fig=False):

    ax = setup_fig_single((8,8), 0.12, 0.12, 0.95, 0.95)

    for s in range(len(p_per_sys_all)):
        pr_in_out_3, pr_in_out_4, pr_in_out_5plus = [], [], []
        for i,p_sys in enumerate(p_per_sys_all[s]):
            p_sys = p_sys[p_sys > 0]
            pr_sys = p_sys[1:]/p_sys[:-1]
            if len(p_sys) == 3:
                pr_in_out_3.append([pr_sys[0], pr_sys[1]])
            elif len(p_sys) == 4:
                for j in range(len(pr_sys)-1):
                    pr_in_out_4.append([pr_sys[j], pr_sys[j+1]])
            elif len(p_sys) >= 5:
                for j in range(len(pr_sys)-1):
                    pr_in_out_5plus.append([pr_sys[j], pr_sys[j+1]])
        pr_in_out_3, pr_in_out_4, pr_in_out_5plus = np.array(pr_in_out_3), np.array(pr_in_out_4), np.array(pr_in_out_5plus)
        #print(np.shape(pr_in_out_3), np.shape(pr_in_out_4), np.shape(pr_in_out_5plus))

        ##### To plot the inner vs. outer period ratios of triplets (in 3+ systems) (similar to Fig 6 in Zhu et al. 2019 and Fig 7 in Weiss et al. 2018a):

        if s == 0:
            plt.scatter(pr_in_out_3[:,0], pr_in_out_3[:,1], facecolors='none', edgecolors=colors[s], marker='^', label='3 planets')
            plt.scatter(pr_in_out_4[:,0], pr_in_out_4[:,1], facecolors='none', edgecolors=colors[s], marker='s', label='4 planets')
            plt.scatter(pr_in_out_5plus[:,0], pr_in_out_5plus[:,1], facecolors='none', edgecolors=colors[s], marker='*', label='5+ planets')
        else:
            plt.scatter(pr_in_out_3[:,0], pr_in_out_3[:,1], facecolors='none', edgecolors=colors[s], marker='^')
            plt.scatter(pr_in_out_4[:,0], pr_in_out_4[:,1], facecolors='none', edgecolors=colors[s], marker='s')
            plt.scatter(pr_in_out_5plus[:,0], pr_in_out_5plus[:,1], facecolors='none', edgecolors=colors[s], marker='*')
    plt.axis('equal')
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    if xyticks_custom is not None:
        ax.set_xticks([1,2,3,4,5,10,20,40])
        ax.set_yticks([1,2,3,4,5,10,20,40])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.axis([1., xymax, 1., xymax])
    plt.xlabel(r'$\mathcal{P}_{\rm in}$', fontsize=tfs)
    plt.ylabel(r'$\mathcal{P}_{\rm out}$', fontsize=tfs)
    for s in range(len(p_per_sys_all)):
        plt.text(x=1.2, y=(0.85-(0.1*s))*xymax, s=labels[s], color=colors[s], fontsize=lfs)
    plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=lfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def compute_pratio_in_out_and_plot_fig_pdf(p_per_sys_all, last_is_Kep=False, fig_size=(10,6), fig_lbrt=[0.15, 0.15, 0.95, 0.95], n_bins=100, x_min=None, x_max=None, colors=['k'], ls=['-'], lw=1, labels=['Input'], afs=12, tfs=12, lfs=12, save_name='no_name_fig.pdf', save_fig=False):

    pr_in_out_3_all, pr_in_out_4_all, pr_in_out_5plus_all = [], [], []
    for s in range(len(p_per_sys_all)):
        pr_in_out_3, pr_in_out_4, pr_in_out_5plus = [], [], []
        for i,p_sys in enumerate(p_per_sys_all[s]):
            p_sys = p_sys[p_sys > 0]
            pr_sys = p_sys[1:]/p_sys[:-1]
            if len(p_sys) == 3:
                pr_in_out_3.append([pr_sys[0], pr_sys[1]])
            elif len(p_sys) == 4:
                for j in range(len(pr_sys)-1):
                    pr_in_out_4.append([pr_sys[j], pr_sys[j+1]])
            elif len(p_sys) >= 5:
                for j in range(len(pr_sys)-1):
                    pr_in_out_5plus.append([pr_sys[j], pr_sys[j+1]])
        pr_in_out_3, pr_in_out_4, pr_in_out_5plus = np.array(pr_in_out_3), np.array(pr_in_out_4), np.array(pr_in_out_5plus)

        pr_in_out_3_all.append(pr_in_out_3)
        pr_in_out_4_all.append(pr_in_out_4)
        pr_in_out_5plus_all.append(pr_in_out_5plus)

    prr_out_in_3_all = [pr_in_out_3_all[s][:,1]/pr_in_out_3_all[s][:,0] for s in range(len(p_per_sys_all))]
    prr_out_in_4_all = [pr_in_out_4_all[s][:,1]/pr_in_out_4_all[s][:,0] for s in range(len(p_per_sys_all))]
    prr_out_in_5plus_all = [pr_in_out_5plus_all[s][:,1]/pr_in_out_5plus_all[s][:,0] for s in range(len(p_per_sys_all))]
    prr_out_in_all_all = [np.concatenate((prr_out_in_3_all[s], prr_out_in_4_all[s], prr_out_in_5plus_all[s]), axis=0) for s in range(len(p_per_sys_all))]

    ##### To plot the ratio of outer to inner period ratios of triplets (in 3+ systems):

    ax = setup_fig_single(fig_size, fig_lbrt[0], fig_lbrt[1], fig_lbrt[2], fig_lbrt[3])
    if last_is_Kep:
        prr_out_in_all_sim = prr_out_in_all_all[:-1]
        prr_out_in_all_Kep = [prr_out_in_all_all[-1]]
    else:
        prr_out_in_all_sim = prr_out_in_all_all
        prr_out_in_all_Kep = []
    plot_panel_pdf_simple(ax, prr_out_in_all_sim, prr_out_in_all_Kep, n_bins=n_bins, x_min=x_min, x_max=x_max, log_x=True, c_sim=colors, ls_sim=ls, lw=lw, labels_sim=labels, xlabel_text=r'$\mathcal{P}_{\rm out}/\mathcal{P}_{\rm in} = (P_{j+2}/P_{j+1})/(P_{j+1}/P_j)$', afs=afs, tfs=tfs, lfs=lfs, legend=True)

    if save_fig:
        plt.savefig(save_name)
        plt.close()





def plot_fig_underlying_mult_vs_amd_ecc_incl(sssp_per_sys, sssp, fig_size=(16,8), fig_lbrt=[0.075, 0.1, 0.975, 0.975], n_min_max=[0.5, 10.5], amd_min_max=[None, None], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    # Planet multiplicity vs. AMD, eccentricity, and mutual inclination:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1, 3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0.2, hspace=0)

    ax = plt.subplot(plot[0,0]) # multiplicity vs total AMD
    lab = True # switches to False when just one n is labeled
    for n in range(1,np.max(sssp_per_sys['Mtot_all'])+1):
        AMD_n = sssp['AMD_tot_all'][sssp_per_sys['Mtot_all'] == n]
        #AMD_n = sssp_per_sys['AMD_all'][sssp_per_sys['Mtot_all'] == n, :n]
        if len(AMD_n) >= 10:
            #AMD_n = AMD_n.flatten()
            q01, q16, qmed, q84, q99 = np.quantile(AMD_n, [0.01, 0.16, 0.5, 0.84, 0.99])
            if n==1:
                plt.scatter(qmed, n, color='c', marker='x', s=100)
                plt.plot((q16, q84), (n,n), color='c', ls='-', lw=3)
                plt.plot((q01, q99), (n,n), color='c', ls='-', lw=1)
            else:
                plt.scatter(qmed, n, color='k', marker='x', s=100, label='Median' if lab else '')
                plt.plot((q16, q84), (n,n), color='k', ls='-', lw=3, label='68%' if lab else '')
                plt.plot((q01, q99), (n,n), color='k', ls='-', lw=1, label='98%' if lab else '')
                lab = False
    plt.gca().set_xscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim(amd_min_max)
    plt.ylim(n_min_max)
    plt.xlabel(r'${\rm AMD}_{\rm tot}$', fontsize=tfs)
    plt.ylabel(r'Intrinsic planet multiplicity $n$', fontsize=tfs)
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

    ax = plt.subplot(plot[0,1]) # multiplicity vs eccentricity
    for n in range(1,np.max(sssp_per_sys['Mtot_all'])+1):
        e_n = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == n,:n]
        if len(e_n) >= 10:
            e_n = e_n.flatten()
            q01, q16, qmed, q84, q99 = np.quantile(e_n, [0.01, 0.16, 0.5, 0.84, 0.99])
            color = 'c' if n==1 else 'k'
            plt.scatter(qmed, n, color=color, marker='x', s=100)
            plt.plot((q16, q84), (n,n), color=color, ls='-', lw=3)
            plt.plot((q01, q99), (n,n), color=color, ls='-', lw=1)
    plt.gca().set_xscale("log")
    #ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim(ecc_min_max)
    plt.ylim(n_min_max)
    plt.xlabel(r'$e$', fontsize=tfs)

    ax = plt.subplot(plot[0,2]) # multiplicity vs mutual inclination
    for n in range(2,np.max(sssp_per_sys['Mtot_all'])+1):
        im_n = sssp_per_sys['inclmut_all'][sssp_per_sys['Mtot_all'] == n,:n]
        if len(im_n) >= 10:
            im_n = im_n.flatten() * (180./np.pi)
            q01, q16, qmed, q84, q99 = np.quantile(im_n, [0.01, 0.16, 0.5, 0.84, 0.99])
            plt.scatter(qmed, n, color='k', marker='x', s=100)
            plt.plot((q16, q84), (n,n), color='k', ls='-', lw=3)
            plt.plot((q01, q99), (n,n), color='k', ls='-', lw=1)
    plt.gca().set_xscale("log")
    #ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlim(incl_min_max)
    plt.ylim(n_min_max)
    plt.xlabel(r'$i_m$ ($^\circ$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def convert_underlying_properties_per_planet_1d(sssp_per_sys, sssp, n_min=2, n_max=None):
    if n_max is None:
        n_max = np.max(sssp_per_sys['Mtot_all'])
    assert(1 <= n_min <= n_max)

    AMD_tot_n_all_once = [] # len = number of systems
    AMD_tot_n_all = [] # expanded to match number of planets, using np.kron()
    AMD_n_all = []
    e_n_all = []
    im_n_all = []
    mass_n_all = []
    pratio_min_n_all_once = [] # len = number of systems
    pratio_min_n_all = [] # expanded to match number of planets, using np.kron()
    for n in range(n_min,n_max+1):
        if np.sum(sssp_per_sys['Mtot_all'] == n) > 0:
            AMD_tot_n = sssp['AMD_tot_all'][sssp_per_sys['Mtot_all'] == n]
            AMD_n = sssp_per_sys['AMD_all'][sssp_per_sys['Mtot_all'] == n,:n]
            e_n = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == n,:n]
            im_n = sssp_per_sys['inclmut_all'][sssp_per_sys['Mtot_all'] == n,:n]
            mass_n = sssp_per_sys['mass_all'][sssp_per_sys['Mtot_all'] == n,:n]
            if n > 1:
                pratio_n = sssp_per_sys['Rm_all'][sssp_per_sys['Mtot_all'] == n,:n-1]
                pratio_min_n = np.min(pratio_n, axis=1)
            else: # for singles, there are no period ratios
                pratio_n = []
                pratio_min_n = []

            AMD_tot_n_all_once.append(AMD_tot_n)
            AMD_tot_n_all.append(np.kron(AMD_tot_n, np.ones(n))) # repeats each value of AMD to match number of planets
            AMD_n_all.append(AMD_n.flatten())
            e_n_all.append(e_n.flatten())
            im_n_all.append(im_n.flatten())
            mass_n_all.append(mass_n.flatten())
            if n > 1:
                pratio_min_n_all_once.append(pratio_min_n)
                pratio_min_n_all.append(np.kron(pratio_min_n, np.ones(n))) # repeats each value of pratio_min to match number of planets
            else: # for singles, there are no period ratios
                pratio_min_n_all_once.append([])
                pratio_min_n_all.append([]) # repeats each value of pratio_min to match number of planets

    AMD_tot_all_once_1d = np.concatenate(AMD_tot_n_all_once)
    AMD_tot_all_1d = np.concatenate(AMD_tot_n_all)
    AMD_all_1d = np.concatenate(AMD_n_all)
    e_all_1d = np.concatenate(e_n_all)
    im_all_1d = np.concatenate(im_n_all) * (180./np.pi)
    mass_all_1d = np.concatenate(mass_n_all)
    pratio_min_all_once_1d = np.concatenate(pratio_min_n_all_once)
    pratio_min_all_1d = np.concatenate(pratio_min_n_all)

    persys_1d = {'AMD_tot_all': AMD_tot_all_once_1d, 'pratio_min_all': pratio_min_all_once_1d}
    perpl_1d = {'AMD_tot_all': AMD_tot_all_1d, 'AMD_all': AMD_all_1d, 'e_all': e_all_1d, 'im_all': im_all_1d, 'mass_all': mass_all_1d, 'pratio_min_all': pratio_min_all_1d}
    return [persys_1d, perpl_1d]

def plot_fig_underlying_amd_vs_ecc_incl(sssp_per_sys, sssp, n_min=2, n_max=None, show_singles=True, limit_singles=1000, fig_size=(16,8), fig_lbrt=[0.1, 0.1, 0.975, 0.975], amd_min_max=[None, None], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    persys_1d, perpl_1d = convert_underlying_properties_per_planet_1d(sssp_per_sys, sssp, n_min=n_min, n_max=n_max)

    # AMD vs. eccentricity and mutual inclination:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1, 2, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)

    ax1 = plt.subplot(plot[0,0]) # AMD vs eccentricity
    corner.hist2d(np.log10(perpl_1d['e_all']), np.log10(perpl_1d['AMD_tot_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    if show_singles:
        AMD_1 = sssp_per_sys['AMD_all'][sssp_per_sys['Mtot_all'] == 1,:1]
        e_1 = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == 1,:1]
        plt.scatter(np.log10(e_1)[:limit_singles], np.log10(AMD_1)[:limit_singles], color='c', marker='x')
    ax1.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([-3., -2., -1., 0.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.xlim(np.log10(np.array(ecc_min_max)))
    plt.ylim(np.log10(np.array(amd_min_max)))
    plt.xlabel(r'$e$', fontsize=tfs)
    plt.ylabel(r'$\log_{10}({\rm AMD}_{\rm tot})$', fontsize=tfs)

    ax2 = plt.subplot(plot[0,1]) # AMD vs mutual inclination
    corner.hist2d(np.log10(perpl_1d['im_all']), np.log10(perpl_1d['AMD_tot_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    ax2.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([-2., -1., 0., 1., 2.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.yticks([])
    plt.xlim(np.log10(np.array(incl_min_max)))
    plt.ylim(np.log10(np.array(amd_min_max)))
    plt.xlabel(r'$i_m$ ($^\circ$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax1, ax2

def plot_fig_underlying_ecc_vs_incl(sssp_per_sys, sssp, n_min=2, n_max=None, fig_size=(8,8), fig_lbrt=[0.15, 0.1, 0.95, 0.95], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    persys_1d, perpl_1d = convert_underlying_properties_per_planet_1d(sssp_per_sys, sssp, n_min=n_min, n_max=n_max)

    # AMD vs. eccentricity and mutual inclination:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1, 1, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)

    ax = plt.subplot(plot[0,0])
    corner.hist2d(np.log10(perpl_1d['e_all']), np.log10(perpl_1d['im_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    ax.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([-3., -2., -1., 0.])
    ytick_vals = np.array([-2., -1., 0., 1., 2.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.yticks(ytick_vals, 10.**ytick_vals)
    plt.xlim(np.log10(np.array(ecc_min_max)))
    plt.ylim(np.log10(np.array(incl_min_max)))
    plt.xlabel(r'$e$', fontsize=tfs)
    plt.ylabel(r'$i_m$ ($^\circ$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax

def plot_fig_underlying_mass_vs_amd_ecc_incl(sssp_per_sys, sssp, n_min=2, n_max=None, show_singles=True, limit_singles=1000, fig_size=(16,8), fig_lbrt=[0.1, 0.1, 0.975, 0.975], mass_min_max=[None, None], amd_min_max=[None, None], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    persys_1d, perpl_1d = convert_underlying_properties_per_planet_1d(sssp_per_sys, sssp, n_min=n_min, n_max=n_max)

    # Planet mass vs. AMD, eccentricity, and mutual inclination:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1, 3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)

    if show_singles:
        AMD_1 = sssp_per_sys['AMD_all'][sssp_per_sys['Mtot_all'] == 1,:1]
        e_1 = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == 1,:1]
        mass_1 = sssp_per_sys['mass_all'][sssp_per_sys['Mtot_all'] == 1,:1]

    ax1 = plt.subplot(plot[0,0]) # mass vs AMD
    corner.hist2d(np.log10(perpl_1d['AMD_all']), np.log10(perpl_1d['mass_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    if show_singles:
        plt.scatter(np.log10(AMD_1)[:limit_singles], np.log10(mass_1)[:limit_singles], color='c', marker='x')
    ax1.tick_params(axis='both', labelsize=afs)
    ytick_vals = np.array([-2., -1., 0., 1., 2., 3.])
    plt.yticks(ytick_vals, 10.**ytick_vals)
    plt.xlim(np.log10(np.array(amd_min_max)))
    plt.ylim(np.log10(np.array(mass_min_max)))
    plt.xlabel(r'$\log_{10}({\rm AMD})$', fontsize=tfs)
    plt.ylabel(r'$M_p$ ($M_\oplus$)', fontsize=tfs)

    ax2 = plt.subplot(plot[0,1]) # mass vs eccentricity
    corner.hist2d(np.log10(perpl_1d['e_all']), np.log10(perpl_1d['mass_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    if show_singles:
        plt.scatter(np.log10(e_1)[:limit_singles], np.log10(mass_1)[:limit_singles], color='c', marker='x')
    ax2.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([-3., -2., -1., 0.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.yticks([]) #plt.yticks(ytick_vals, 10.**ytick_vals)
    plt.xlim(np.log10(np.array(ecc_min_max)))
    plt.ylim(np.log10(np.array(mass_min_max)))
    plt.xlabel(r'$e$', fontsize=tfs)

    ax3 = plt.subplot(plot[0,2]) # mass vs mutual inclination
    corner.hist2d(np.log10(perpl_1d['im_all']), np.log10(perpl_1d['mass_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    ax3.tick_params(axis='both', labelsize=afs)
    xtick_vals = np.array([-2., -1., 0., 1., 2.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.yticks([]) #plt.yticks(ytick_vals, 10.**ytick_vals)
    plt.xlim(np.log10(np.array(incl_min_max)))
    plt.ylim(np.log10(np.array(mass_min_max)))
    plt.xlabel(r'$i_m$ ($^\circ$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax1, ax2, ax3

def plot_fig_underlying_pratio_min_vs_amd_ecc_incl(sssp_per_sys, sssp, n_min=2, n_max=None, fig_size=(16,8), fig_lbrt=[0.1, 0.1, 0.975, 0.975], pratio_min_max=[None, None], amd_min_max=[None, None], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    persys_1d, perpl_1d = convert_underlying_properties_per_planet_1d(sssp_per_sys, sssp, n_min=n_min, n_max=n_max)

    # Minimum period ratio vs. AMD, eccentricity, and mutual inclination:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1, 3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0, hspace=0)

    ax1 = plt.subplot(plot[0,0]) # min period ratio vs AMD
    corner.hist2d(np.log10(persys_1d['AMD_tot_all']), np.log10(persys_1d['pratio_min_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    ax1.tick_params(axis='both', which='both', labelsize=afs)
    ytick_vals = np.array([1., 2., 3., 4., 5., 10.])
    plt.yticks(np.log10(ytick_vals), ytick_vals)
    plt.xlim(np.log10(np.array(amd_min_max)))
    plt.ylim(np.log10(np.array(pratio_min_max)))
    plt.xlabel(r'$\log_{10}({\rm AMD}_{\rm tot})$', fontsize=tfs)
    plt.ylabel(r'${\rm min}(\mathcal{P} = P_{i+1}/P_i)$', fontsize=tfs)

    ax2 = plt.subplot(plot[0,1]) # min period ratio vs eccentricity
    corner.hist2d(np.log10(perpl_1d['e_all']), np.log10(perpl_1d['pratio_min_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    ax2.tick_params(axis='both', which='both', labelsize=afs)
    xtick_vals = np.array([-3., -2., -1., 0.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.yticks([]) #plt.yticks(ytick_vals, 10.**ytick_vals)
    plt.xlim(np.log10(np.array(ecc_min_max)))
    plt.ylim(np.log10(np.array(pratio_min_max)))
    plt.xlabel(r'$e$', fontsize=tfs)

    ax3 = plt.subplot(plot[0,2]) # min period ratio vs mutual inclination
    corner.hist2d(np.log10(perpl_1d['im_all']), np.log10(perpl_1d['pratio_min_all']), bins=50, plot_density=False, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    ax3.tick_params(axis='both', which='both', labelsize=afs)
    xtick_vals = np.array([-2., -1., 0., 1., 2.])
    plt.xticks(xtick_vals, 10.**xtick_vals)
    plt.yticks([]) #plt.yticks(ytick_vals, 10.**ytick_vals)
    plt.xlim(np.log10(np.array(incl_min_max)))
    plt.ylim(np.log10(np.array(pratio_min_max)))
    plt.xlabel(r'$i_m$ ($^\circ$)', fontsize=tfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return ax1, ax2, ax3

def plot_fig_underlying_ecc_incl_per_mult(sssp_per_sys, sssp, n_min=1, n_max=None, n_bins=100, fit_dists=False, log_x=False, alpha=0.2, fig_size=(16,8), fig_lbrt=[0.03, 0.1, 0.97, 0.97], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    assert n_max > n_min
    n_mults = range(n_min, n_max+1)

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(len(n_mults), 2, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0.1, hspace=0)

    # Eccentricity distributions:
    x_min, x_max = ecc_min_max
    x = np.logspace(np.log10(x_min), np.log10(x_max), 100) if log_x else np.linspace(x_min, x_max, 100)
    for i in range(len(n_mults)): # plot from high n (top) to low n (bottom)
        n = n_mults[::-1][i]
        e_n = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == n,:n]
        e_n = e_n.flatten()

        if fit_dists:
            # Fit Rayleigh:
            loc_rl, scale_rl = scipy.stats.rayleigh.fit(e_n, floc=0)
            dist_rl = scipy.stats.rayleigh(scale=scale_rl)
            print('(n = %s) Rayleigh fit: scale = %s' % (n, scale_rl))
            # Fit Lognormal:
            shape_ln, loc_ln, scale_ln = scipy.stats.lognorm.fit(e_n, floc=0)
            dist_ln = scipy.stats.lognorm(s=shape_ln, scale=scale_ln)
            print('(n = %s) Lognormal fit: mu = %s, sigma = %s' % (n, np.log(scale_ln), shape_ln))
            # Fit von Mises Fisher:
            #kappa_vMF, loc_vMF, scale_vMF = scipy.stats.vonmises.fit(e_n, fscale=1)
            #dist_vMF = scipy.stats.vonmises(kappa=kappa_vMF, loc=loc_vMF)
            #print('(n = %s) von Mises Fisher fit: kappa = %s, loc = %s' % (n, kappa_vMF, loc_vMF))

        ax = plt.subplot(plot[i,0])
        if log_x:
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
        else:
            bins = np.linspace(x_min, x_max, n_bins+1)
        color = 'c' if n==1 else 'b' ### different color for singles (if they are drawing from a separate distribution)
        plt.hist(e_n, bins=bins, histtype='stepfilled', density=fit_dists, color=color, alpha=alpha)
        if fit_dists:
            # Rayleigh:
            plt.plot(x, dist_rl.pdf(x), color='r', label=r'Rayleigh($\sigma = %s$)' % np.round(scale_rl, 3))
            # Lognormal:
            plt.plot(x, dist_ln.pdf(x), color='g', label=r'LogN($\mu = %s, \sigma = %s$)' % (np.round(np.log(scale_ln), 2), np.round(shape_ln, 2)))
            # von Mises Fisher:
            #plt.plot(x, dist_vMF.pdf(x), color='k', label=r'vMF($\kappa = %s, {\rm loc} = %s$)' % (np.round(kappa_vMF, 2), np.round(loc_vMF)))
        if log_x:
            plt.gca().set_xscale("log")
        ax.tick_params(axis='both', which='both', labelsize=afs)
        if n != n_min:
            plt.xticks([])
        plt.yticks([])
        plt.xlim(np.array(ecc_min_max))
        #plt.ylim()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if n == n_min:
            plt.xlabel(r'$e$', fontsize=tfs)
        plt.text(x=0.01, y=0.6, s=r'$n = %s$' % n, fontsize=lfs, transform=ax.transAxes)
        plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=10)

    # Mutual inclinations distributions:
    x_min, x_max = incl_min_max
    x = np.logspace(np.log10(x_min), np.log10(x_max), 100) if log_x else np.linspace(x_min, x_max, 100)
    for i in range(len(n_mults)): # plot from high n (top) to low n (bottom)
        n = n_mults[::-1][i]
        if n == 1:
            break # can only plot mutual inclinations if n > 1
        im_n = sssp_per_sys['inclmut_all'][sssp_per_sys['Mtot_all'] == n,:n]
        im_n = im_n.flatten() * (180./np.pi)

        if fit_dists:
            # Fit Rayleigh:
            loc_rl, scale_rl = scipy.stats.rayleigh.fit(im_n, floc=0)
            dist_rl = scipy.stats.rayleigh( scale=scale_rl)
            scale_rl_Zhu = gen.incl_mult_power_law_Zhu2018(n, sigma_5=0.8, alpha=-3.5)
            print('(n = %s) Rayleigh fit: scale = %s; scale_Zhu = %s' % (n, scale_rl, scale_rl_Zhu))
            #scale_rl = scale_rl_Zhu # use Zhu2018 instead of our fit
            # Fit Lognormal:
            shape_ln, loc_ln, scale_ln = scipy.stats.lognorm.fit(im_n, floc=0)
            dist_ln = scipy.stats.lognorm(s=shape_ln, scale=scale_ln)
            print('(n = %s) Lognormal fit: mu = %s, sigma = %s' % (n, np.log(scale_ln), shape_ln))
            # Fit von Mises Fisher:
            #kappa_vMF, loc_vMF, scale_vMF = scipy.stats.vonmises.fit(im_n, fscale=1)
            #dist_vMF = scipy.stats.vonmises(kappa=kappa_vMF, loc=loc_vMF)
            #print('(n = %s) von Mises Fisher fit: kappa = %s, loc = %s' % (n, kappa_vMF, loc_vMF))

        ax = plt.subplot(plot[i,1])
        if log_x:
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
        else:
            bins = np.linspace(x_min, x_max, n_bins+1)
        plt.hist(im_n, bins=bins, histtype='stepfilled', density=fit_dists, color='b', alpha=alpha)
        if fit_dists:
            # Rayleigh:
            plt.plot(x, dist_rl.pdf(x), color='r', label=r'Rayleigh($\sigma = %s$)' % np.round(scale_rl, 3))
            # Lognormal:
            plt.plot(x, dist_ln.pdf(x), color='g', label=r'LogN($\mu = %s, \sigma = %s$)' % (np.round(np.log(scale_ln), 2), np.round(shape_ln, 2)))
            # von Mises Fisher:
            #plt.plot(x, dist_vMF.pdf(x), color='k', label=r'vMF($\kappa = %s, {\rm loc} = %s$)' % (np.round(kappa_vMF, 2), np.round(loc_vMF, 2)))
        if log_x:
            plt.gca().set_xscale("log")
        ax.tick_params(axis='both', which='both', labelsize=afs)
        if n != max(2,n_min):
            plt.xticks([])
        plt.yticks([])
        plt.xlim(np.array(incl_min_max))
        #plt.ylim()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if n == max(2,n_min):
            plt.xlabel(r'$i_m$ ($^\circ$)', fontsize=tfs)
        plt.text(x=0.01, y=0.6, s=r'$n = %s$' % n, fontsize=lfs, transform=ax.transAxes)
        plt.legend(loc='upper right', bbox_to_anchor=(0.99,0.99), ncol=1, frameon=False, fontsize=10)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def plot_fig_underlying_amd_ecc_incl_per_mult(sssp_per_sys, sssp, n_min=1, n_max=None, n_bins=100, fit_dists=False, log_x=False, alpha=0.2, fig_size=(16,8), fig_lbrt=[0.03, 0.1, 0.97, 0.97], amd_min_max=[None, None], ecc_min_max=[None, None], incl_min_max=[None, None], afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):

    assert n_max > n_min
    n_mults = range(n_min, n_max+1)

    n_draws = 1000000 # number of samples for plotting from a fitted distribution

    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(len(n_mults), 3, left=fig_lbrt[0], bottom=fig_lbrt[1], right=fig_lbrt[2], top=fig_lbrt[3], wspace=0.1, hspace=0)

    # AMD_tot distributions:
    x_min, x_max = amd_min_max
    x = np.logspace(np.log10(x_min), np.log10(x_max), 100) if log_x else np.linspace(x_min, x_max, 100)
    for i in range(len(n_mults)): # plot from high n (top) to low n (bottom)
        n = n_mults[::-1][i]
        AMD_n = sssp['AMD_tot_all'][sssp_per_sys['Mtot_all'] == n]
        AMD_n = AMD_n.flatten()

        ax = plt.subplot(plot[i,0])
        if log_x:
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
        else:
            bins = np.linspace(x_min, x_max, n_bins+1)
        color = 'c' if n==1 else 'b' ### different color for singles (if they are drawing from a separate distribution)
        plt.hist(AMD_n, bins=bins, histtype='stepfilled', color=color, alpha=alpha, label='Maximum AMD model')
        plt.axvline(x=np.median(AMD_n), ymax=0.2, color='k', label='Median')
        if log_x:
            plt.gca().set_xscale("log")
        ax.tick_params(axis='both', which='both', labelsize=afs)
        if n != n_min:
            plt.xticks([])
        plt.yticks([])
        plt.xlim(np.array(amd_min_max))
        #plt.ylim()
        if n == n_min:
            plt.xlabel(r'${\rm AMD}_{\rm tot}$', fontsize=tfs)
        plt.text(x=0.01, y=0.7, s=r'$n = %s$' % n, fontsize=lfs, transform=ax.transAxes)
        if n == n_max:
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = [handles[1], handles[0]], [labels[1], labels[0]]
            ax.legend(handles, labels, loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

    # Eccentricity distributions:
    x_min, x_max = ecc_min_max
    x = np.logspace(np.log10(x_min), np.log10(x_max), 100) if log_x else np.linspace(x_min, x_max, 100)
    for i in range(len(n_mults)): # plot from high n (top) to low n (bottom)
        n = n_mults[::-1][i]
        e_n = sssp_per_sys['e_all'][sssp_per_sys['Mtot_all'] == n,:n]
        e_n = e_n.flatten()
        q16, q50, q84 = np.quantile(e_n, [0.16, 0.5, 0.84])

        if fit_dists:
            # Fit Rayleigh:
            loc_rl, scale_rl = scipy.stats.rayleigh.fit(e_n, floc=0)
            dist_rl = scipy.stats.rayleigh(scale=scale_rl)
            ###print('(n = %s) Rayleigh fit: scale = %s' % (n, scale_rl))
            # Fit Lognormal:
            shape_ln, loc_ln, scale_ln = scipy.stats.lognorm.fit(e_n, floc=0)
            dist_ln = scipy.stats.lognorm(s=shape_ln, scale=scale_ln)
            ###print('(n = %s) Lognormal fit: mu = %s, sigma = %s' % (n, np.log(scale_ln), shape_ln))
            # Fit von Mises Fisher:
            #kappa_vMF, loc_vMF, scale_vMF = scipy.stats.vonmises.fit(e_n, fscale=1)
            #dist_vMF = scipy.stats.vonmises(kappa=kappa_vMF, loc=loc_vMF)
            #print('(n = %s) von Mises Fisher fit: kappa = %s, loc = %s' % (n, kappa_vMF, loc_vMF))
            print('(Ecc: n = {:<2}) med+/- = {:0.3f}_{{-{:0.3f}}}^{{+{:0.3f}}}, mu = {:<8}, sigma = {:<8}, sigma_ray = {:<8}'.format(n, np.round(q50, 3), np.round(q50-q16, 3), np.round(q84-q50, 3), np.round(scale_ln, 3), np.round(shape_ln, 3), np.round(scale_rl, 3)))

        ax = plt.subplot(plot[i,1])
        if log_x:
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
            bins_mid = 10.**((np.log10(bins[1:]) + np.log10(bins[:-1]))/2.)
        else:
            bins = np.linspace(x_min, x_max, n_bins+1)
            bins_mid = (bins[1:]+bins[:-1])/2.
        color = 'c' if n==1 else 'b' ### different color for singles (if they are drawing from a separate distribution)
        plt.hist(e_n, bins=bins, weights=np.ones(len(e_n))/len(e_n), histtype='stepfilled', color=color, alpha=alpha)
        plt.axvline(x=np.median(e_n), ymax=0.2, color='k')
        #plt.axvline(x=gen.incl_mult_power_law_Zhu2018(n, sigma_5=0.03, alpha=-2.), ymax=0.2, color='darkorange', label=r'$\sigma_{i,n} = 0.03(n/5)^{-2}$')
        if fit_dists:
            # Rayleigh:
            label = '' #r'Rayleigh($\sigma = %s$)' % np.round(scale_rl, 3)
            counts = np.histogram(dist_rl.rvs(size=n_draws), bins=bins)[0]
            plt.plot(bins_mid, counts/float(n_draws), c='r', label='Rayleigh fit')
            # Lognormal:
            label = '' #r'LogN($\mu = %s, \sigma = %s$)' % (np.round(np.log(scale_ln), 2), np.round(shape_ln, 2))
            counts = np.histogram(dist_ln.rvs(size=n_draws), bins=bins)[0]
            plt.plot(bins_mid, counts/float(n_draws), c='g', label='Lognormal fit')
        if log_x:
            plt.gca().set_xscale("log")
        ax.tick_params(axis='both', which='both', labelsize=afs)
        if n != n_min:
            plt.xticks([])
        plt.yticks([])
        plt.xlim(np.array(ecc_min_max))
        #plt.ylim()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if n == n_min:
            plt.xlabel(r'$e$', fontsize=tfs)
        plt.text(x=0.01, y=0.7, s=r'$n = %s$' % n, fontsize=lfs, transform=ax.transAxes)
        if n == n_max:
            plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

    # Mutual inclinations distributions:
    x_min, x_max = incl_min_max
    x = np.logspace(np.log10(x_min), np.log10(x_max), 100) if log_x else np.linspace(x_min, x_max, 100)
    for i in range(len(n_mults)): # plot from high n (top) to low n (bottom)
        n = n_mults[::-1][i]
        if n == 1:
            break # can only plot mutual inclinations if n > 1
        im_n = sssp_per_sys['inclmut_all'][sssp_per_sys['Mtot_all'] == n,:n]
        im_n = im_n.flatten() * (180./np.pi)
        q16, q50, q84 = np.quantile(im_n, [0.16, 0.5, 0.84])

        if fit_dists:
            # Fit Rayleigh:
            loc_rl, scale_rl = scipy.stats.rayleigh.fit(im_n, floc=0)
            dist_rl = scipy.stats.rayleigh( scale=scale_rl)
            scale_rl_Zhu = gen.incl_mult_power_law_Zhu2018(n, sigma_5=0.8, alpha=-3.5)
            ###print('(n = %s) Rayleigh fit: scale = %s; scale_Zhu = %s' % (n, scale_rl, scale_rl_Zhu))
            #scale_rl = scale_rl_Zhu # use Zhu2018 instead of our fit
            # Fit Lognormal:
            shape_ln, loc_ln, scale_ln = scipy.stats.lognorm.fit(im_n, floc=0)
            dist_ln = scipy.stats.lognorm(s=shape_ln, scale=scale_ln)
            ###print('(n = %s) Lognormal fit: mu = %s, sigma = %s' % (n, np.log(scale_ln), shape_ln))
            # Fit von Mises Fisher:
            #kappa_vMF, loc_vMF, scale_vMF = scipy.stats.vonmises.fit(im_n, fscale=1)
            #dist_vMF = scipy.stats.vonmises(kappa=kappa_vMF, loc=loc_vMF)
            #print('(n = %s) von Mises Fisher fit: kappa = %s, loc = %s' % (n, kappa_vMF, loc_vMF))
            print('(Incl: n = {:<2}) med+/- = {:0.3f}_{{-{:0.3f}}}^{{+{:0.3f}}}, mu = {:<8}, sigma = {:<8}, sigma_ray = {:<8}, Zhu = {:<8}'.format(n, np.round(q50, 2), np.round(q50-q16, 2), np.round(q84-q50, 2), np.round(scale_ln, 2), np.round(shape_ln, 2), np.round(scale_rl, 2), np.round(scale_rl_Zhu, 2)))

        ax = plt.subplot(plot[i,2])
        if log_x:
            bins = np.logspace(np.log10(x_min), np.log10(x_max), n_bins+1)
            bins_mid = 10.**((np.log10(bins[1:]) + np.log10(bins[:-1]))/2.)
        else:
            bins = np.linspace(x_min, x_max, n_bins+1)
            bins_mid = (bins[1:]+bins[:-1])/2.
        plt.hist(im_n, bins=bins, weights=np.ones(len(im_n))/len(im_n), histtype='stepfilled', color='b', alpha=alpha)
        plt.axvline(x=np.median(im_n), ymax=0.2, color='k')
        # To also plot a power-law fit to the median values:
        #plt.axvline(x=gen.incl_mult_power_law_Zhu2018(n, sigma_5=0.8, alpha=-3.5)*np.sqrt(2.*np.log(2.)), ymax=0.2, color='b', label=r'$\sigma_{i,n} = 0.8(n/5)^{-3.5}$')
        #plt.axvline(x=gen.incl_mult_power_law_Zhu2018(n, sigma_5=1., alpha=-2.)*np.sqrt(2.*np.log(2.)), ymax=0.2, color='darkorange', label=r'$\sigma_{i,n} = 1.0(n/5)^{-2}$')
        if fit_dists:
            # Rayleigh:
            counts = np.histogram(dist_rl.rvs(size=n_draws), bins=bins)[0]
            plt.plot(bins_mid, counts/float(n_draws), c='r', label='')
            # Lognormal:
            counts = np.histogram(dist_ln.rvs(size=n_draws), bins=bins)[0]
            plt.plot(bins_mid, counts/float(n_draws), c='g', label='')
        if log_x:
            plt.gca().set_xscale("log")
        ax.tick_params(axis='both', which='both', labelsize=afs)
        if n != max(2,n_min):
            plt.xticks([])
        plt.yticks([])
        plt.xlim(np.array(incl_min_max))
        #plt.ylim()
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        if n == max(2,n_min):
            plt.xlabel(r'$i_m$ ($^\circ$)', fontsize=tfs)
        plt.text(x=0.01, y=0.7, s=r'$n = %s$' % n, fontsize=lfs, transform=ax.transAxes)
        if n == n_max:
            plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=lfs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
