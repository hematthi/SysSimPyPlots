# To import required modules:
import numpy as np
import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
#from matplotlib import ticker #for setting contour plots to log scale
import corner #corner.py package for corner plots





# Functions to analyze GP models/outputs and to make corner plots for visualizing the parameter space:

def load_training_points(dims, file_name_path='', file_name='Active_params_recomputed_distances_table_best100000_every10.txt'):
    # 'dims' is the number of dimensions (model parameters)

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
    print('Transforming columns (i,j) to (i+j,j-i).')
    xpoints_transformed = np.copy(xpoints)
    xpoints_transformed[:,i], xpoints_transformed[:,j] = xpoints_transformed[:,i] + xpoints_transformed[:,j], xpoints_transformed[:,j] - xpoints_transformed[:,i]
    return xpoints_transformed

def transform_sum_diff_params_inverse(xpoints, i, j):
    print('Transforming columns (i,j) to ((i-j)/2,(i+j)/2).')
    xpoints_transformed = np.copy(xpoints)
    xpoints_transformed[:,i], xpoints_transformed[:,j] = (xpoints_transformed[:,i] - xpoints_transformed[:,j])/2., (xpoints_transformed[:,i] + xpoints_transformed[:,j])/2.
    return xpoints_transformed

def make_cuts_GP_mean_std_post(x_names, xprior_table, max_mean=np.inf, max_std=np.inf, max_post=np.inf):
    dims = len(x_names)

    xpoints_all = xprior_table[np.array(x_names)]
    xpoints_cut = xpoints_all[(xprior_table['GP_mean'] < max_mean) & (xprior_table['GP_std'] < max_std) & (xprior_table['GP_posterior_draw'] < max_post)]
    #xpoints_cut = xpoints_cut.view((float, dims))
    xpoints_cut = xpoints_cut.view(np.float64).reshape(xpoints_cut.shape + (dims+3,))
    xpoints_cut = xpoints_cut[:,:dims]
    print('Total points: %s; points left: %s' % (len(xpoints_all), len(xpoints_cut)))

    return xpoints_cut

def plot_fig_hists_GP_draws(fig_size, xprior_table, bins=100, save_name='no_name_fig.pdf', save_fig=False):
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

def plot_contours_and_points_corner(x_symbols, xlower, xupper, contour_2d_grids, xpoints=None, points_size=1., points_alpha=1., fig_size=(16,16), fig_lbrtwh=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05], afs=10, tfs=12, lfs=10, save_name='no_name_fig.pdf', save_fig=False):

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
    # NOTE: The 'log_x' and 'log_y' are options to take the log of the x and y inputs, NOT to mean that the x and y are already logged; user should always pass unlogged values for x and y (and their limits).

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

def plot_function_heatmap_contours_given_irregular_points_corner(x_symbols, xpoints, fpoints, xlower=None, xupper=None, show_points=True, points_size=1., points_alpha=1., fig_size=(16,16), fig_lbrtwh=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05], afs=10, tfs=12, lfs=10, save_name='no_name_fig.pdf', save_fig=False):

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

def plot_function_heatmap_averaged_grid_given_irregular_points_corner(x_symbols, xpoints, fpoints, flabel='f', xlower=None, xupper=None, x_bins=20, show_cbar=True, show_points=True, points_size=1., points_alpha=1., fig_size=(16,16), fig_lbrtwh=[0.05, 0.05, 0.98, 0.98, 0.05, 0.05], afs=10, tfs=12, lfs=10, save_name='no_name_fig.pdf', save_fig=False):

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
