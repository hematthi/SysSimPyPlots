# To import required modules:
import numpy as np
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
#from matplotlib import ticker #for setting contour plots to log scale
import corner #corner.py package for corner plots





# Functions to analyze GP models/outputs and to make corner plots for visualizing the parameter space:

def load_training_points(dims, file_name_path='', file_name=''):
    """
    Load the table of model parameters and total weighted distances of the model iterations compared to the Kepler data, that was used for training the Gaussian process emulator.

    Parameters
    ----------
    dims : int
        The number of (free) model parameters.
    file_name_path : str, default=''
        The path to the file containing the results of the model iterations.
    file_name : str, default=''
        The name of the file containing the results of the model iterations (e.g., 'Active_params_recomputed_distances_table_best100000_every10.txt').

    Returns
    -------
    data_train : dict
        A dictionary of the model training points.


    The dictionary contains the following fields:

    - `active_params_names`: The names of the (free) model parameters.
    - `xtrain`: The (free) model parameter values for each iteration (2-d array).
    - `ytrain`: The total weighted distance of the model compared to the Kepler data for each iteration (1-d array).
    """
    data_table = np.genfromtxt(file_name_path + file_name, delimiter=' ', names=True, dtype='f8')
    cols = len(data_table.dtype.names)

    active_params_names = data_table.dtype.names[:dims]
    print('Active param names: ', active_params_names)

    #xtrain = data_table[np.array(active_params_names)]
    #xtrain = xtrain.view((float, dims))
    xtrain = data_table.view(np.float64).reshape(data_table.shape + (cols,))
    xtrain = xtrain[:,:dims]

    ytrain = data_table['dist_tot_weighted']

    data_train = {'active_params_names': active_params_names, 'xtrain': xtrain, 'ytrain': ytrain}
    return data_train

def load_GP_table_prior_draws(file_name, file_name_path=''):
    """
    Load a table of model parameter draws from the prior.
    """
    #Example file_name: 'GP_emulator_points%s_meanf%s_small_hparams_prior_draws%s.csv' % (n_train, mean_f, n_draws)
    xprior_accepted_table = np.genfromtxt(file_name_path + file_name, skip_header=4, delimiter=',', names=True, dtype='f8')
    return xprior_accepted_table

def load_table_points_min_GP(file_name, file_name_path=''):
    #Example file_name: 'GP_train2000_meanf50.0_minimize_mean_iterations10.csv'
    xmin_table = np.genfromtxt(file_name_path + file_name, skip_header=2, delimiter=',', names=True, dtype='f8')
    return xmin_table

def load_GP_2d_grids(dims, n_train, mean_f, sigma_f, lscales, file_name_path='', grid_dims=50):
    file_name_mean = 'GP_train%s_meanf%s_sigmaf%s_lscales%s_grids2d_%sx%s_mean.csv' % (n_train, mean_f, sigma_f, lscales, grid_dims, grid_dims)
    file_name_std = 'GP_train%s_meanf%s_sigmaf%s_lscales%s_grids2d_%sx%s_std.csv' % (n_train, mean_f, sigma_f, lscales, grid_dims, grid_dims)
    GP_mean_2d_grids = np.genfromtxt(file_name_path + file_name_mean, delimiter=',', skip_header=5, dtype='f8')
    GP_std_2d_grids = np.genfromtxt(file_name_path + file_name_std, delimiter=',', skip_header=5, dtype='f8')
    with open(file_name_path + file_name_mean, 'r') as f:
        for line in f:
            if line[0:8] == '# xlower':
                xlower = [float(x) for x in line[12:-2].split(' ')]
            if line[0:8] == '# xupper':
                xupper = [float(x) for x in line[12:-2].split(' ')]
            if line[0:12] == '# xmin_guess':
                xmin_guess = [float(x) for x in line[16:-2].split(' ')]

    if grid_dims != np.shape(GP_mean_2d_grids)[1]:
        print('PROBLEM: mismatch with grid_dims!')
    GP_mean_2d_grids = np.reshape(GP_mean_2d_grids, (sum(range(dims)), grid_dims, grid_dims))
    GP_std_2d_grids = np.reshape(GP_std_2d_grids, (sum(range(dims)), grid_dims, grid_dims))

    GP_grids = {'xlower': xlower, 'xupper': xupper, 'xmin_guess': xmin_guess, 'mean_grids': GP_mean_2d_grids, 'std_grids': GP_std_2d_grids}
    return GP_grids

def transform_sum_diff_params(xpoints, i, j):
    """
    Transform two parameters into their sum and difference.

    Parameters
    ----------
    xpoints : array[float]
        An array of parameters for each iteration (2-d array).
    i,j : int
        The index of a parameter to transform.

    Returns
    -------
    xpoints_transformed : array[float]
        The array of parameters with the x_i and x_j parameters replaced by their sum (x_i+x_j) and difference (x_j-x_i). 2-d array with the same shape as `xpoints`.

    See Also
    --------
    transform_sum_diff_params_inverse : Perform the inverse transformation (return the original parameters from their sum and difference).
    """
    print('Transforming columns (i,j) to (i+j,j-i).')
    xpoints_transformed = np.copy(xpoints)
    xpoints_transformed[:,i], xpoints_transformed[:,j] = xpoints_transformed[:,i] + xpoints_transformed[:,j], xpoints_transformed[:,j] - xpoints_transformed[:,i]
    return xpoints_transformed

def transform_sum_diff_params_inverse(xpoints, i, j):
    """
    Transform the sum and difference of two parameters back into the original parameters.

    Parameters
    ----------
    xpoints : array[float]
        An array of parameters for each iteration (2-d array), where the x_i and x_j parameters are the sum and difference of two original parameters.
    i,j : int
        The index of a transformed parameter.

    Returns
    -------
    xpoints_transformed : array[float]
        The array of original parameters (2-d array). Equivalent to the transformation of x_i and x_j to (x_i-x_j)/2 and (x_i+x_j)/2.

    See Also
    --------
    transform_sum_diff_params : Perform the sum and difference of two parameters (the inverse of this transformation).
    """
    print('Transforming columns (i,j) to ((i-j)/2,(i+j)/2).')
    xpoints_transformed = np.copy(xpoints)
    xpoints_transformed[:,i], xpoints_transformed[:,j] = (xpoints_transformed[:,i] - xpoints_transformed[:,j])/2., (xpoints_transformed[:,i] + xpoints_transformed[:,j])/2.
    return xpoints_transformed

def make_cuts_GP_mean_std_post(x_names, xprior_table, max_mean=np.inf, max_std=np.inf, max_post=np.inf):
    """
    Return a restricted sample of parameters given cuts on the Gaussian process (GP) emulator predictions at those parameters.

    Parameters
    ----------
    x_names : array[str]
        The names of the parameters.
    xprior_table : structured array
        The table of parameter values and GP predictions at each iteration, with parameter/column names corresponding to `x_names`.
    max_mean : float, default=inf
        The maximum mean prediction to include.
    max_std : float, default=inf
        The maximum standard deviation to include.
    max_post : float, default=inf
        The maximum posterior draw value to include.

    Returns
    -------
    xpoints_cut : structured array
        The table of parameter values at the iterations surviving all of the cuts.
    """
    dims = len(x_names)

    xpoints_all = xprior_table[np.array(x_names)]
    xpoints_cut = xpoints_all[(xprior_table['GP_mean'] < max_mean) & (xprior_table['GP_std'] < max_std) & (xprior_table['GP_posterior_draw'] < max_post)]
    #xpoints_cut = xpoints_cut.view((float, dims))
    xpoints_cut = xpoints_cut.view(np.float64).reshape(xpoints_cut.shape + (dims+3,))
    xpoints_cut = xpoints_cut[:,:dims]
    print('Total points: %s; points left: %s' % (len(xpoints_all), len(xpoints_cut)))

    return xpoints_cut

def plot_fig_hists_GP_draws(fig_size, xprior_table, bins=100, save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot figure for visualizing the Gaussian process emulator statistics.

    Contains four panels for the histograms of the Gaussian process mean predictions, prediction draws, and standard deviations, along with a scatter plot of mean prediction vs. standard deviation.

    Parameters
    ----------
    fig_size : tuple
        The figure size, e.g. (16,8).
    xprior_table : structured array
        The table of parameter values and GP predictions at each iteration.
    bins : int, default=100
        The number of bins to use for the histograms.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.
    """
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(2,2,left=0.1,bottom=0.1,right=0.95,top=0.95,wspace=0.2,hspace=0.2)

    ax = plt.subplot(plot[0,0])
    plt.hist(xprior_table['GP_mean'], bins=bins)
    plt.xlabel('GP mean prediction')
    plt.ylabel('Points')

    ax = plt.subplot(plot[0,1])
    plt.hist(xprior_table['GP_posterior_draw'], bins=bins)
    plt.xlabel('GP prediction')
    plt.ylabel('Points')

    ax = plt.subplot(plot[1,0])
    plt.hist(xprior_table['GP_std'], bins=bins)
    plt.xlabel('GP std of prediction')
    plt.ylabel('Points')

    ax = plt.subplot(plot[1,1])
    plt.scatter(xprior_table['GP_mean'], xprior_table['GP_std'], marker='.')
    plt.xlabel('GP mean prediction')
    plt.ylabel('GP std of prediction')

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def plot_cornerpy_wrapper(x_symbols, xpoints, xpoints_extra=None, c_extra='r', s_extra=1, quantiles=[0.16, 0.5, 0.84], verbose=False, fig=None, show_titles=True, label_kwargs={'fontsize': 20}, title_kwargs={'fontsize':20}, save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot a parameter space as a corner plot.

    Wrapper for the corner.corner function from the `corner.py package <https://corner.readthedocs.io/en/latest/>`_, with the option of over-plotting an additional set of points for comparison.

    Parameters
    ----------
    x_symbols : list or array[str]
        The list of the parameter names/symbols.
    xpoints : array[float]
        The sample of parameter values (2-d array).
    xpoints_extra : array[float], optional
        An additional sample of parameter values to plot (2-d array).
    c_extra : str, default='r'
        The color for plotting the additional sample of parameters.
    s_extra : float, default=1
        The marker size for plotting the additional sample of parameters.
    quantiles : list, default=[0.16, 0.5, 0.84]
        The quantiles to show on the histograms of the parameters as vertical lines.
    verbose : bool, default=False
        Whether to print the quantiles for the distribution of each parameter.
    fig : matplotlib.figure.Figure, optional
        An existing figure object to plot on.
    show_titles : bool, default=True
        Whether to show the quantiles for the distribution of each parameter above each histogram.
    label_kwargs : dict, default={'fontsize': 20}
        Extra parameters for setting the labels.
    title_kwargs : dict, default={'fontsize': 20}
        Extra parameters for setting the title.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure, if `save_fig=False`.
    """
    dims = len(x_symbols)

    fig = corner.corner(xpoints, labels=x_symbols, quantiles=quantiles, verbose=verbose, fig=fig, show_titles=show_titles, label_kwargs=label_kwargs, title_kwargs=title_kwargs)

    # If want to plot an additional set of points:
    if xpoints_extra is not None:
        axes = np.array(fig.axes).reshape((dims, dims))
        for i in range(dims):
            for j in range(i):
                ax = axes[i,j]
                ax.scatter(xpoints_extra[:,j], xpoints_extra[:,i], color=c_extra, s=s_extra)

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    else:
        return fig

def plot_contours_and_points_corner(x_symbols, xlower, xupper, contour_2d_grids, xpoints=None, points_size=1., points_alpha=1., afs=10, tfs=12, lfs=10, fig_size=(16,16), fig_lbrtwh=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot contours for the evaluations of a function over 2-d grids of an n-d parameter space.

    Parameters
    ----------
    x_symbols : list or array[str]
        The list of the parameter names/symbols.
    xlower : list or array[float]
        The lower bounds for each parameter.
    xupper : list or array[float]
        The upper bounds for each parameter.
    contour_2d_grids : array[float]
        The array of 2-d grids for the evaluations of a function (3-d array of dimensions '(N,gdim,gdim)' where 'N' is the number of unique pairs of parameters, and 'gdim' is the number of grid points along each grid dimension).
    xpoints : array[float], optional
        An additional sample of parameter values to plot (2-d array).
    points_size : float, default=1.
        The point size for plotting the additional sample of parameters.
    points_alpha : float, default=1.
        The transparency of the points for plotting the additional sample of parameters.
    afs : int, default=10
        The axes fontsize.
    tfs : int, default=12
        The text fontsize.
    lfs : int, default=10
        The legend fontsize.
    fig_size : tuple, default=(16,16)
        The figure size.
    fig_lbrtwh : list, default=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05]
        The positions of the (left, bottom, right, and top) margins of all the plotting panels (between 0 and 1), followed by the width and height space between the panels.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.
    """
    dims = len(x_symbols)
    grid_dims = np.shape(contour_2d_grids)[2]

    fig = plt.figure(figsize=fig_size)
    left, bottom, right, top, wspace, hspace = fig_lbrtwh
    plot = GridSpec(dims, dims, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    grid_index = 0
    for i in range(dims): # indexes the row or y-axis variable
        for j in range(i): # indexes the column or x-axis variable
            xaxis = np.linspace(xlower[j], xupper[j], grid_dims)
            yaxis = np.linspace(xlower[i], xupper[i], grid_dims)
            xgrid, ygrid = np.meshgrid(xaxis, yaxis)

            ax = plt.subplot(plot[i,j])
            cplot = plt.contour(xgrid, ygrid, contour_2d_grids[grid_index])
            plt.clabel(cplot, inline=1, fontsize=lfs)
            if xpoints is not None:
                plt.scatter(xpoints[:,j], xpoints[:,i], s=points_size, alpha=points_alpha, c='k')
            ax.tick_params(labelleft=False, labelbottom=False)
            if i == dims-1:
                ax.tick_params(labelbottom=True)
                plt.xlabel(x_symbols[j], fontsize=tfs)
            if j == 0:
                ax.tick_params(labelleft=True)
                plt.ylabel(x_symbols[i], fontsize=tfs)
            plt.xticks(rotation=45, fontsize=afs)
            plt.yticks(rotation=45, fontsize=afs)

            grid_index += 1

    for i in range(dims):
        ax = plt.subplot(plot[i,i])
        ax.tick_params(labelleft=False, labelbottom=False)
        if i == dims-1:
            #ax.tick_params(labelbottom=True)
            plt.xlabel(x_symbols[i], fontsize=tfs)
        if i == 0:
            #ax.tick_params(labelleft=True)
            plt.ylabel(x_symbols[i], fontsize=tfs)
        plt.xticks(rotation=45, fontsize=afs)
        plt.yticks(rotation=45, fontsize=afs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def plot_2d_points_and_contours_with_histograms(x, y, x_min=None, x_max=None, y_min=None, y_max=None, log_x=False, log_y=False, bins_hist=50, bins_cont=50, points_only=False, xlabel_text='x', ylabel_text='y', extra_text=None, plot_qtls=True, log_x_qtls=False, log_y_qtls=False, x_str_format='{:0.2f}', y_str_format='{:0.2f}', x_symbol=r'$x$', y_symbol=r'$y$', afs=20, tfs=20, lfs=16, fig_size=(8,8), fig_lbrtwh=[0.15,0.15,0.95,0.95,0.,0.], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot a pair of parameters as a 2-d scatter plot with attached histograms of each parameter.

    Parameters
    ----------
    x, y : array[float]
        The values of the parameters (1-d array).
    x_min : float, optional
        The minimum value of `x` to include.
    x_max : float, optional
        The maximum value of `x` to include.
    y_min : float, optional
        The minimum value of `y` to include.
    y_max : float, optional
        The maximum value of `y` to include.
    log_x : bool, default=False
        Whether to plot the x-axis on a log-scale (and take the log of the values in `x` for the histograms and quantiles)\*.
    log_y : bool, default=False
        Whether to plot the y-axis on a log-scale (and take the log of the values in `y` for the histograms and quantiles)\*.
    bins_hist : int, default=50
        The number of bins to use for the histograms.
    bins_cont : int, default=50
        The number of bins to use for the contour maps (along each parameter).
    points_only : bool, default=False
        Whether to plot only the parameter points instead of their contour maps.
    xlabel_text : str, default='x'
        The x-axis label.
    ylabel_text : str, default='y'
        The y-axis label.
    extra_text : str, optional
        Extra text to be displayed on the figure.
    plot_qtls : bool, default=True
        Whether to compute, print, and plot the quantiles of each parameter on the histograms.
    log_x_qtls : bool, default=False
        Whether to use the log of the `x` values for computing their quantiles. Only used if `log_x=True`.
    log_y_qtls : bool, default=False
        Whether to use the log of the `y` values for computing their quantiles. Only used if `log_y=True`.
    x_str_format : str, default='{:0.2f}'
        The string formatter for the `x` quantiles (e.g. to control how many significant figures are shown).
    y_str_format : str, default='{:0.2f}'
        The string formatter for the `y` quantiles (e.g. to control how many significant figures are shown).
    x_symbol : str, default=r'$x$'
        The symbol for the `x` parameter for labeling the quantiles.
    y_symbol : str, default=r'$y$'
        The symbol for the `y` parameter for labeling the quantiles.
    afs : int, default=20
        The axes fontsize.
    tfs : int, default=20
        The text fontsize.
    lfs : int, default=16
        The legend fontsize.
    fig_size : tuple, default=(8,8)
        The figure size.
    fig_lbrtwh : list, default=[0.15, 0.15, 0.95, 0.95, 0., 0.]
        The positions of the (left, bottom, right, and top) margins of all the plotting panels (between 0 and 1), followed by the width and height space between the panels (main plot and top/side histograms).
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.

    Returns
    -------
    ax_main : matplotlib.axes._subplots.AxesSubplot
        The plotting axes for the main panel (scatter plot/contour map).


    Note
    ----
    \*The `log_x` and `log_y` parameters are options to take the log of the `x` and `y` inputs, NOT to mean that the input values are already logged! The user should always pass unlogged values for `x` and `y` (and their limits).
    """
    assert len(x) == len(y)
    if log_x:
        x = np.log10(x)
        x_min, x_max = np.log10(x_min), np.log10(x_max)
    if log_y:
        y = np.log10(y)
        y_min, y_max = np.log10(y_min), np.log10(y_max)

    fig = plt.figure(figsize=fig_size)
    left, bottom, right, top, wspace, hspace = fig_lbrtwh
    plot = GridSpec(5, 5, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)

    ax_main = plt.subplot(plot[1:,:4])
    if points_only:
        plt.scatter(x, y, color='k')
    else:
        corner.hist2d(x, y, bins=bins_cont, plot_datapoints=True, plot_density=False, fill_contours=True, contour_kwargs={'colors': ['0.6','0.4','0.2','0']}, data_kwargs={'color': 'k'})
    plt.text(x=0.05, y=0.95, s=extra_text, ha='left', va='top', fontsize=lfs, transform=ax_main.transAxes)
    ax_main.tick_params(axis='both', labelsize=afs)
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.xlabel(xlabel_text, fontsize=tfs)
    plt.ylabel(ylabel_text, fontsize=tfs)

    ax = plt.subplot(plot[0,:4]) # top histogram
    xhist = plt.hist(x, bins=np.linspace(x_min, x_max, bins_hist+1), histtype='step', color='k', ls='-')
    if plot_qtls:
        x_qtls = np.quantile(x, q=[0.16,0.5,0.84])
        plt.vlines(x=x_qtls, ymin=0., ymax=1.1*np.max(xhist[0]), colors='k', linestyles=[':','--',':'])
        if log_x and not log_x_qtls: # if plotting contours/histograms in log(x), but want to report unlogged quantiles
            x_qtls = 10.**x_qtls

        qmed_str = x_str_format.format(x_qtls[1])
        q_m_str = x_str_format.format(x_qtls[1]-x_qtls[0])
        q_p_str = x_str_format.format(x_qtls[2]-x_qtls[1])
        print('%s = %s_{-%s}^{+%s}' % (x_symbol, qmed_str, q_m_str, q_p_str))
        plt.text(x=0.02, y=0.95, s=x_symbol + r'$= %s_{-%s}^{+%s}$' % (qmed_str, q_m_str, q_p_str), ha='left', va='top', fontsize=lfs, transform=ax.transAxes)
    plt.xlim([x_min, x_max])
    plt.ylim([0., 1.1*np.max(xhist[0])])
    plt.xticks([])
    plt.yticks([])

    ax = plt.subplot(plot[1:,4]) # side histogram
    yhist = plt.hist(y, bins=np.linspace(y_min, y_max, bins_hist+1), histtype='step', orientation='horizontal', color='k', ls='-')
    if plot_qtls:
        y_qtls = np.quantile(y, q=[0.16,0.5,0.84])
        plt.hlines(y=y_qtls, xmin=0., xmax=1.1*np.max(yhist[0]), colors='k', linestyles=[':','--',':'])
        if log_y and not log_y_qtls: # if plotting contours/histograms in log(y), but want to report unlogged quantiles
            y_qtls = 10.**y_qtls

        qmed_str = y_str_format.format(y_qtls[1])
        q_m_str = y_str_format.format(y_qtls[1]-y_qtls[0])
        q_p_str = y_str_format.format(y_qtls[2]-y_qtls[1])
        print('%s = %s_{-%s}^{+%s}' % (y_symbol, qmed_str, q_m_str, q_p_str))
        plt.text(x=0.95, y=0.98, s=y_symbol + r'$= %s_{-%s}^{+%s}$' % (qmed_str, q_m_str, q_p_str), rotation=270, ha='right', va='top', fontsize=lfs, transform=ax.transAxes)
    plt.xlim([0., 1.1*np.max(yhist[0])])
    plt.ylim([y_min, y_max])
    plt.xticks([])
    plt.yticks([])

    if save_fig:
        plt.savefig(save_name)
        plt.close()
    return ax_main

def plot_function_heatmap_contours_given_irregular_points_corner(x_symbols, xpoints, fpoints, xlower=None, xupper=None, show_points=True, points_size=1., points_alpha=1., afs=10, tfs=12, lfs=10, fig_size=(16,16), fig_lbrtwh=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot 2-d heat-maps and contours of a function evaluated on a set of irregularly spaced points in the n-d parameter space.

    Parameters
    ----------
    x_symbols : list or array[str]
        The list of the parameter names/symbols.
    xpoints : array[float]
        The sample of parameter values at which the function was evaluated (2-d array).
    fpoints : array[float]
        The function evaluations at the points given by `xpoints`.
    xlower : list or array[float]
        The lower bounds for each parameter.
    xupper : list or array[float]
        The upper bounds for each parameter.
    show_points : bool, default=True
        Whether to plot the individual parameter points.
    points_size : float, default=1.
        The point size for plotting the sample of parameters.
    points_alpha : float, default=1.
        The transparency of the points for plotting the sample of parameters.
    afs : int, default=10
        The axes fontsize.
    tfs : int, default=12
        The text fontsize.
    lfs : int, default=10
        The legend fontsize.
    fig_size : tuple, default=(16,16)
        The figure size.
    fig_lbrtwh : list, default=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05]
        The positions of the (left, bottom, right, and top) margins of all the plotting panels (between 0 and 1), followed by the width and height space between the panels.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.
    """
    dims = len(x_symbols)
    if xlower == None:
        xlower = np.min(xpoints, axis=0)
    if xupper == None:
        xupper = np.max(xpoints, axis=0)

    fig = plt.figure(figsize=fig_size)
    left, bottom, right, top, wspace, hspace = fig_lbrtwh
    plot = GridSpec(dims, dims, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for i in range(dims): # indexes the row or y-axis variable
        for j in range(i): # indexes the column or x-axis variable
            ax = plt.subplot(plot[i,j])
            #plt.tricontour(xpoints[:,j], xpoints[:,i], fpoints, linewidths=0.5, colors='k')
            contours = plt.tricontourf(xpoints[:,j], xpoints[:,i], fpoints, cmap="RdBu_r")
            #plt.colorbar(contours)
            if show_points:
                plt.scatter(xpoints[:,j], xpoints[:,i], s=points_size, alpha=points_alpha, c='k')
            ax.axis((xlower[j], xupper[j], xlower[i], xupper[i]))
            ax.tick_params(labelleft=False, labelbottom=False)
            if i == dims-1:
                ax.tick_params(labelbottom=True)
                plt.xlabel(x_symbols[j], fontsize=tfs)
            if j == 0:
                ax.tick_params(labelleft=True)
                plt.ylabel(x_symbols[i], fontsize=tfs)
            plt.xticks(rotation=45, fontsize=afs)
            plt.yticks(rotation=45, fontsize=afs)

    for i in range(dims):
        ax = plt.subplot(plot[i,i])
        plt.colorbar(contours)
        ax.tick_params(labelleft=False, labelbottom=False)
        if i == dims-1:
            #ax.tick_params(labelbottom=True)
            plt.xlabel(x_symbols[i], fontsize=tfs)
        if i == 0:
            #ax.tick_params(labelleft=True)
            plt.ylabel(x_symbols[i], fontsize=tfs)
            plt.xticks(rotation=45, fontsize=afs)
            plt.yticks(rotation=45, fontsize=afs)

    if save_fig:
        plt.savefig(save_name)
        plt.close()

def plot_function_heatmap_averaged_grid_given_irregular_points_corner(x_symbols, xpoints, fpoints, flabel='f', xlower=None, xupper=None, x_bins=20, show_cbar=True, show_points=True, points_size=1., points_alpha=1., afs=10, tfs=12, lfs=10, fig_size=(16,16), fig_lbrtwh=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05], save_name='no_name_fig.pdf', save_fig=False):
    """
    Plot 2-d heat-maps (using `matplotlib.pyplot.imshow <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html>`_) on grids based on the evaluations of a function on a set of irregularly spaced points in the n-d parameter space.

    The value of each grid bin is computed as the mean of the function evaluated at all of the points inside the bin.

    Parameters
    ----------
    x_symbols : list or array[str]
        The list of the parameter names/symbols.
    xpoints : array[float]
        The sample of parameter values at which the function was evaluated (2-d array).
    fpoints : array[float]
        The function evaluations at the points given by `xpoints`.
    flabel : str, default='f'
        The text label for the function.
    xlower : list or array[float]
        The lower bounds for each parameter.
    xupper : list or array[float]
        The upper bounds for each parameter.
    x_bins : int, default=20
        The number of bins to use for each dimension of each grid.
    show_cbar : bool, default=True
        Whether to show the colorbar for the function.
    show_points : bool, default=True
        Whether to plot the individual parameter points.
    points_size : float, default=1.
        The point size for plotting the sample of parameters.
    points_alpha : float, default=1.
        The transparency of the points for plotting the sample of parameters.
    afs : int, default=10
        The axes fontsize.
    tfs : int, default=12
        The text fontsize.
    lfs : int, default=10
        The legend fontsize.
    fig_size : tuple, default=(16,16)
        The figure size.
    fig_lbrtwh : list, default=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05]
        The positions of the (left, bottom, right, and top) margins of all the plotting panels (between 0 and 1), followed by the width and height space between the panels.
    save_name : str, default='no_name_fig.pdf'
        The file name for saving the figure.
    save_fig : bool, default=False
        Whether to save the figure. If True, will save the figure in the working directory with the file name given by `save_name`.
    """
    if any(np.isinf(fpoints)):
        bools_inf = np.isinf(fpoints)
        xpoints = xpoints[~bools_inf]
        fpoints = fpoints[~bools_inf]
        print('Infinite f values provided; discarding those points (n = %s).' % np.sum(bools_inf))

    dims = len(x_symbols)
    if xlower == None:
        xlower = np.min(xpoints, axis=0)
    if xupper == None:
        xupper = np.max(xpoints, axis=0)

    fig = plt.figure(figsize=fig_size)
    left, bottom, right, top, wspace, hspace = fig_lbrtwh
    plot = GridSpec(dims, dims, left=left, bottom=bottom, right=right, top=top, wspace=wspace, hspace=hspace)
    for i in range(dims): # indexes the row or y-axis variable
        for j in range(i): # indexes the column or x-axis variable
            ax = plt.subplot(plot[i,j])

            # Construct the grid of averaged f values from the points in each grid square:
            grid_f_avg = np.zeros((x_bins, x_bins))
            xj_bins = np.linspace(xlower[j], xupper[j], x_bins+1)
            xi_bins = np.linspace(xlower[i], xupper[i], x_bins+1)
            for ii in range(x_bins):
                for jj in range(x_bins):
                    bools_in_bin = (xpoints[:,i] >= xi_bins[ii]) & (xpoints[:,i] < xi_bins[ii+1]) & (xpoints[:,j] >= xj_bins[jj]) & (xpoints[:,j] < xj_bins[jj+1])
                    fpoints_in_bin = fpoints[bools_in_bin]
                    if len(fpoints_in_bin) > 0:
                        grid_f_avg[ii,jj] = np.mean(fpoints_in_bin)
            grid_f_avg[grid_f_avg == 0] = np.nan

            current_cmap = cm.get_cmap()
            current_cmap.set_bad(color='red')
            plt.imshow(grid_f_avg, aspect='auto', origin='lower', extent=(xlower[j], xupper[j], xlower[i], xupper[i]))

            if show_points:
                plt.scatter(xpoints[:,j], xpoints[:,i], s=points_size, alpha=points_alpha, c='k')
            ax.axis((xlower[j], xupper[j], xlower[i], xupper[i]))
            ax.tick_params(labelleft=False, labelbottom=False)
            if i == dims-1:
                ax.tick_params(labelbottom=True)
                plt.xlabel(x_symbols[j], fontsize=tfs)
            if j == 0:
                ax.tick_params(labelleft=True)
                plt.ylabel(x_symbols[i], fontsize=tfs)
            plt.xticks(rotation=45, fontsize=afs)
            plt.yticks(rotation=45, fontsize=afs)

    for i in range(dims):
        ax = plt.subplot(plot[i,i])
        plt.hist(xpoints[:,i], bins=x_bins, histtype='step')
        ax.tick_params(labelleft=False, labelbottom=False)
        if i == dims-1:
            #ax.tick_params(labelbottom=True)
            plt.xlabel(x_symbols[i], fontsize=tfs)
        if i == 0:
            #ax.tick_params(labelleft=True)
            plt.ylabel(x_symbols[i], fontsize=tfs)
            plt.xticks(rotation=45, fontsize=afs)
            plt.yticks(rotation=45, fontsize=afs)

    if show_cbar:
        ax = plt.subplot(plot[0:3,dims-3:dims])
        plt.hist(fpoints, bins=100, histtype='step')
        plt.xlabel(flabel, fontsize=tfs)
        plt.colorbar()

    if save_fig:
        plt.savefig(save_name)
        plt.close()
