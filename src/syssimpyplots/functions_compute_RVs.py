# To import required modules:
import numpy as np
import matplotlib
#import matplotlib.cm as cm #for color maps
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec #for specifying plot attributes
from matplotlib import ticker #for setting contour plots to log scale
from matplotlib.colors import LogNorm #for log color scales
from mpl_toolkits.axes_grid1.inset_locator import inset_axes #for inset axes
from scipy.optimize import brentq # for solving roots of equations
import scipy.linalg # for linear algebra
import time

import src.functions_general as gen





# Functions to compute the RV signals of planets:

def M_anom(t, T0, P):
    # Compute the mean anomaly M (radians) at a time t (yrs) given a reference epoch T0 (yrs) and period P (yrs)
    M = (2.*np.pi*(t - T0))/P
    return M

def fzero_E_anom(E, t, T0, P, e):
    # Kepler's equation moved to one side (solve for zeros to get solutions)
    M = M_anom(t, T0, P)
    return E - e*np.sin(E) - M

def E_anom(t, T0, P, e):
    # Compute the eccentric anomaly E (radians) numerically at a time t (yrs) given a reference epoch T0 (yrs), period P (yrs), and eccentricity e
    
    #E = brentq(fzero_E_anom, 0., 2.*np.pi, args=(t, T0, P, e)) # doesn't always work!
    
    # Try to solve iteratively:
    M = M_anom(t, T0, P)
    E0 = 0
    dE = 1
    while dE > 1e-10:
        E = M + e*np.sin(E0)
        dE = E - E0
        E0 = E
    
    return E

def nu_anom(E, e):
    # Compute the true anomaly nu (radians) given an eccentric anomaly E (radians) and eccentricity e
    nu = 2.*np.arctan(np.sqrt((1.+e)/(1.-e))*np.tan(E/2.))
    return nu

def RV_true(t, K, P, T0=0., e=0., w=0., gamma=0.):
    # Compute the true radial velocity (m/s) at a time t (yrs) given the semi-amplitude K (m/s), period P (yrs), reference epoch T0 (yrs), eccentricity e, argument of pericenter w (radians), and reference zero-point gamma (m/s)
    E = E_anom(t, T0, P, e)
    nu = nu_anom(E, e)
    V_r = K*(np.cos(nu + w) + e*np.cos(w)) + gamma
    return V_r

def RV_true_sys(t, K_sys, P_sys, T0_sys, e_sys, w_sys, gamma=0.):
    # Compute the true radial velocity (m/s) at a time t (yrs) given all the planets in the system
    n_pl = len(K_sys)
    assert n_pl == len(P_sys) == len(T0_sys) == len(e_sys) == len(w_sys)
    V_r_per_pl = np.zeros(n_pl)
    for i in range(n_pl):
        V_r_per_pl[i] = RV_true(t, K_sys[i], P_sys[i], T0=T0_sys[i], e=e_sys[i], w=w_sys[i])
    V_r = np.sum(V_r_per_pl)
    return V_r

def rv_K(m, P, e=None, i=None, Mstar=1.0):
    # Compute the radial velocity semi-amplitude K of a system given arrays of the planet masses m (M_earth) and periods (days), and optionally eccentricities, sky-inclinations (rad), and stellar mass (Msun)
    n_pl = len(m)
    e = np.zeros(n_pl) if e is None else e
    i = (np.pi/2.)*np.ones(n_pl) if i is None else i
    
    assert all(m >= 0) & all(P > 0) & (Mstar > 0)
    assert all(0 <= e) & all(e < 1)
    
    K = (28.4329/np.sqrt(1. - e**2.)) * (m*np.sin(i)*gen.Mearth/gen.Mjup) * (np.sum(m)*(gen.Mearth/gen.Msun) + Mstar)**(-2./3.) * (P/365.25)**(-1./3.)
    return K # units of m/s





# Functions to fit RV time series using generalized least squares (GLS):

def fit_rv_K_single_planet_model_GLS(t_obs, RV_obs, covarsc, P, T0=0., e=0., w=0.):
    # Fit a radial velocity series RV_obs (m/s) using generalized least squares assuming a single planet model with known period P (days), epoch T0 (days), eccentricity e, and argument of pericenter w (radians), for the semi-amplitude K (m/s)
    # The array of observation times t_obs (days) must correspond to the array of RV observations RV_obs, and to the covariance matrix of observation uncertainties covarsc
    E_obs = np.array([E_anom(t, T0, P, e) for t in t_obs])
    nu_obs = nu_anom(E_obs, e)

    sincosphase = np.cos(nu_obs + w) + e*np.cos(w) # V_r = K*sincosphase

    Xt_inv_covar_X = sincosphase.transpose() @ scipy.linalg.solve(covarsc, sincosphase)
    X_inv_covar_y =  sincosphase.transpose() @ scipy.linalg.solve(covarsc, RV_obs)
    K_hat = scipy.linalg.solve(Xt_inv_covar_X, X_inv_covar_y) # estimated K (m/s)
    inv_covar_of_fit_params = sincosphase.transpose() @ scipy.linalg.solve(covarsc, sincosphase)
    sigma_K = 1./np.sqrt(inv_covar_of_fit_params) # estimated uncertainty in K (m/s)

    return K_hat, sigma_K
    
def fit_rv_Ks_multi_planet_model_GLS(t_obs, RV_obs, covarsc, P_all, T0_all, e_all, w_all):
    # Fit a radial velocity series RV_obs (m/s) using generalized least squares assuming a multi-planet model with known periods P_all (days), epochs T0_all (days), eccentricities e_all, and argument of pericenter w_all (radians), for the semi-amplitudes K (m/s) of each planet
    # The array of observation times t_obs (days) must correspond to the array of RV observations RV_obs, and to the covariance matrix of observation uncertainties covarsc
    n_pl = len(P_all)
    assert n_pl == len(T0_all) == len(e_all) == len(w_all)
    sincosphase = []
    for i in range(n_pl):
        E_obs = np.array([E_anom(t, T0_all[i], P_all[i], e_all[i]) for t in t_obs])
        nu_obs = nu_anom(E_obs, e_all[i])
        sincosphase.append(np.cos(nu_obs + w_all[i]) + e_all[i]*np.cos(w_all[i]))
    sincosphase = np.array(sincosphase).transpose()

    Xt_inv_covar_X = sincosphase.transpose() @ scipy.linalg.solve(covarsc, sincosphase)
    X_inv_covar_y =  sincosphase.transpose() @ scipy.linalg.solve(covarsc, RV_obs)
    K_hat_all = scipy.linalg.solve(Xt_inv_covar_X, X_inv_covar_y) # estimated K's (m/s)
    inv_covar_of_fit_params = sincosphase.transpose() @ scipy.linalg.solve(covarsc, sincosphase)
    symmetrized_inv_covar_of_fit_params = 0.5 * (inv_covar_of_fit_params + inv_covar_of_fit_params.transpose())
    sigma_K_all = 1./np.diag(scipy.linalg.cholesky(symmetrized_inv_covar_of_fit_params))
    
    return K_hat_all, sigma_K_all





# Functions to condition catalogs on a set of planet properties and simulate RV observations (fitting RVs to measure the amplitude of the transiting planet):

def conditionals_dict(P_cond_bounds=None, Rp_cond_bounds=None, Mp_cond_bounds=None, det=True):
    
    # P_cond_bounds = [P_lower, P_upper] for period of planet conditioned on
    # Rp_cond_bounds = [Rp_lower, Rp_upper] for radius of planet conditioned on
    # Mp_cond_bounds = [Mp_lower, Mp_upper] for mass of planet conditioned on
    P_lower, P_upper = [0., np.inf] if P_cond_bounds==None else P_cond_bounds
    Rp_lower, Rp_upper = [0., np.inf] if Rp_cond_bounds==None else Rp_cond_bounds
    Mp_lower, Mp_upper = [0., np.inf] if Mp_cond_bounds==None else Mp_cond_bounds
    assert P_lower < P_upper
    assert Rp_lower < Rp_upper
    assert Mp_lower < Mp_upper
    
    conds = {}
    conds['P_lower'] = P_lower
    conds['P_upper'] = P_upper
    conds['Rp_lower'] = Rp_lower
    conds['Rp_upper'] = Rp_upper
    conds['Mp_lower'] = Mp_lower
    conds['Mp_upper'] = Mp_upper
    conds['det'] = det
    return conds

def condition_planets_bools_per_sys(sssp_per_sys, conds):
    # Compute a 2d array of booleans where the conditioned planets are indicated by True values
    # Note: includes systems without any conditioned planets (i.e. rows with all False values)
    bools_cond_per_sys = (sssp_per_sys['P_all'] > conds['P_lower']) & (sssp_per_sys['P_all'] < conds['P_upper']) & (sssp_per_sys['radii_all'] > conds['Rp_lower']) & (sssp_per_sys['radii_all'] < conds['Rp_upper']) & (sssp_per_sys['mass_all'] > conds['Mp_lower']) & (sssp_per_sys['mass_all'] < conds['Mp_upper']) # 2d array with True for each conditioned planet (should be at least one True per row/system)
    if conds['det']:
        bools_cond_per_sys = bools_cond_per_sys & (sssp_per_sys['det_all'] == 1)
    return bools_cond_per_sys

def condition_systems_indices(sssp_per_sys, conds):
    bools_cond_per_sys = condition_planets_bools_per_sys(sssp_per_sys, conds)
    bools_in_bounds = np.any(bools_cond_per_sys, axis=1)
    n_per_sys = sssp_per_sys['Mtot_all']
    i_cond = np.arange(len(n_per_sys))[bools_in_bounds]
    print('Number of systems that have a planet with period in [%s,%s] d, radius in [%s,%s] R_earth, and mass in [%s,%s] M_earth: %s/%s' % (conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper'], conds['Mp_lower'], conds['Mp_upper'], len(i_cond), len(n_per_sys)))
    return i_cond

def plot_systems_gallery_with_RVseries_conditional(sssp_per_sys, sssp, conds, outputs_RVs=None, mark_undet=True, fit_RVs=False, N_obs_all=None, repeat=1000, σ_1obs=1., t_obs_σ=0.2, fig_size=(9,10), seed=None, N_sample=20, N_per_plot=20, afs=12, tfs=12, save_name_base='no_name_fig', save_fig=False):
    
    # outputs_RVs = None: if provided (a table of RV simulations of conditioned systems), will draw systems from these (must have system ids and match the catalog that is provided)
    
    if outputs_RVs is not None:
        i_cond = outputs_RVs['id_sys']
    else:
        i_cond = condition_systems_indices(sssp_per_sys, conds)
    
    np.random.seed(seed)
    #i_sample = np.random.choice(i_cond, N_sample, replace=False)
    i_sample = i_cond[:N_sample]
    
    # To sort by stellar mass:
    #i_sort = np.argsort(sssp['Mstar_all'][i_sample])
    
    # To sort by K_cond:
    K_cond_sample = np.zeros(N_sample)
    for i in range(N_sample):
        Mstar_sys = sssp['Mstar_all'][i_sample][i]
        P_sys = sssp_per_sys['P_all'][i_sample][i]
        det_sys = sssp_per_sys['det_all'][i_sample][i]
        Mp_sys = sssp_per_sys['mass_all'][i_sample][i]
        Rp_sys = sssp_per_sys['radii_all'][i_sample][i]
        e_sys = sssp_per_sys['e_all'][i_sample][i]
        incl_sys = sssp_per_sys['incl_all'][i_sample][i]
        clusterids_sys = sssp_per_sys['clusterids_all'][i_sample][i]
        
        det_sys = det_sys[P_sys > 0]
        Mp_sys = Mp_sys[P_sys > 0]
        Rp_sys = Rp_sys[P_sys > 0]
        e_sys = e_sys[P_sys > 0]
        incl_sys = incl_sys[P_sys > 0]
        clusterids_sys = clusterids_sys[P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        
        K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
        id_pl_cond = np.arange(len(P_sys))[(P_sys > conds['P_lower']) & (P_sys < conds['P_upper']) & (Rp_sys > conds['Rp_lower']) & (Rp_sys < conds['Rp_upper'])][0] # index of conditioned planet
        K_cond = K_sys[id_pl_cond]
        
        K_cond_sample[i] = K_cond
    i_sort = np.argsort(K_cond_sample)
    
    # To not sort (just plot systems from top to bottom):
    #i_sort = np.arange(N_sample)[::-1]

    Mstar_sample = sssp['Mstar_all'][i_sample][i_sort]
    P_sample = sssp_per_sys['P_all'][i_sample][i_sort]
    det_sample = sssp_per_sys['det_all'][i_sample][i_sort]
    Mp_sample = sssp_per_sys['mass_all'][i_sample][i_sort]
    Rp_sample = sssp_per_sys['radii_all'][i_sample][i_sort]
    e_sample = sssp_per_sys['e_all'][i_sample][i_sort]
    incl_sample = sssp_per_sys['incl_all'][i_sample][i_sort]
    clusterids_sample = sssp_per_sys['clusterids_all'][i_sample][i_sort]

    n_figs = int(np.ceil(float(N_sample)/N_per_plot))
    print('Generating %s figures...' % n_figs)
    for h in range(n_figs):
        fig = plt.figure(figsize=fig_size)
        plt.figtext(0.5, 0.96, r'Systems conditioned on a Venus-like planet', va='center', ha='center', fontsize=tfs)
        #plt.figtext(0.5, 0.96, r'Systems conditioned on a planet in $P = [%s,%s]$d, $R_p = [%s,%s] R_\oplus$' % (conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
        ###plt.figtext(0.5, 0.96, 'Systems conditioned on a planet in\n' r'$P = %s$d, $R_p = %s R_\oplus$' % (conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
        
        # For the gallery:
        plot = GridSpec(1,1,left=0.08,bottom=0.1,right=0.44,top=0.8,wspace=0,hspace=0)
        #plot = GridSpec(1,1,left=0.1,bottom=0.075,right=0.46,top=0.85,wspace=0,hspace=0)
        ax = plt.subplot(plot[:,:])
        for j in range(N_per_plot):
            id_sys = h*n_figs + j
            
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
            id_pl_cond = np.arange(len(P_sys))[(P_sys > conds['P_lower']) & (P_sys < conds['P_upper']) & (Rp_sys > conds['Rp_lower']) & (Rp_sys < conds['Rp_upper'])][0] # index of conditioned planet
            K_cond = K_sys[id_pl_cond]
            
            #print(K_sys)
            # Defaults: s=10*Rp^2, vmin=0, vmax=5
            ecolor = 'r' if mark_undet else None
            sc = plt.scatter(P_sys[det_sys == 1], np.ones(np.sum(det_sys == 1))+j, c=K_sys[det_sys == 1], s=20.*Rp_sys[det_sys == 1]**2., vmin=0., vmax=2.)
            plt.scatter(P_sys[det_sys == 0], np.ones(np.sum(det_sys == 0))+j, c=K_sys[det_sys == 0], edgecolors=ecolor, s=20.*Rp_sys[det_sys == 0]**2., vmin=0., vmax=2.) #facecolors='none'
            ###sc = plt.scatter(P_sys, np.ones(len(P_sys))+j, c=K_sys, s=10.*Rp_sys**2., vmin=0., vmax=10.)
            ###plt.scatter(P_sys[det_sys == 0], np.ones(np.sum(det_sys == 0))+j, color='r', marker='x', s=8.*Rp_sys[det_sys == 0]**2.)
            
            plt.axhline(y=j+0.5, lw=0.5, ls='-', color='k')
            plt.text(x=1.8, y=j+1, s='{:.2f}'.format(np.round(K_cond,2)), va='center', ha='right', fontsize=10)
        plt.text(x=1., y=j+1.6, s=r'$K_{\rm cond}$ (m/s)', va='bottom', ha='left', fontsize=10)
        #plt.text(x=1.5, y=j+1.6, s=r'$M_\star$ ($M_\odot$)', va='bottom', ha='center', fontsize=10)
        #plt.fill_betweenx(y=[0,j+2], x1=conds['P_lower'], x2=conds['P_upper'], color='g', alpha=0.2)
        plt.gca().set_xscale("log")
        ax.tick_params(axis='both', labelsize=afs)
        ax.set_xticks([3,10,30,100,300])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks([])
        plt.xlim([2., 400.])
        plt.ylim([0.5, N_per_plot+0.5])
        plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
        
        # For colorbar:
        plot = GridSpec(1,1,left=0.08,bottom=0.89,right=0.44,top=0.92,wspace=0,hspace=0)
        #plot = GridSpec(1,1,left=0.1,bottom=0.91,right=0.46,top=0.93,wspace=0,hspace=0)
        cax = plt.subplot(plot[0,0])
        cbar = plt.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label(r'RV semi-amplitude $K$ (m/s)', fontsize=10)
        
        # For the RV time series:
        plot = GridSpec(N_per_plot,1,left=0.52,bottom=0.1,right=0.88,top=0.8,wspace=0,hspace=0)
        #plot = GridSpec(N_per_plot,1,left=0.54,bottom=0.075,right=0.9,top=0.85,wspace=0,hspace=0)
        for j in range(N_per_plot):
            # NOTE: still starting from j=0 but will index the panels starting from the bottom to match the systems in the gallery
            ax = plt.subplot(plot[N_per_plot-j-1,:])
            id_sys = h*n_figs + j
            
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
            
            T0_sys = P_sys*np.random.random(len(P_sys)) # reference epochs for each planet
            omega_sys = 2.*np.pi*np.random.random(len(P_sys)) # WARNING: need to get from simulated physical catalogs, NOT re-drawn
            
            K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
            id_pl_cond = np.arange(len(P_sys))[(P_sys > conds['P_lower']) & (P_sys < conds['P_upper']) & (Rp_sys > conds['Rp_lower']) & (Rp_sys < conds['Rp_upper'])][0] # index of conditioned planet
            P_cond = P_sys[id_pl_cond] # period of conditioned planet (days)
            Rp_cond = Rp_sys[id_pl_cond] # radius of conditioned planet (Earth radii)
            K_cond = K_sys[id_pl_cond] # K of the conditioned planet (m/s)
            
            t_end = 150. # days
            t_array = np.linspace(0., t_end, 1001)
            RV_sys_per_pl = np.zeros((len(t_array),len(P_sys)))
            for i in range(len(P_sys)):
                T0 = P_sys[i]*np.random.uniform()
                omega = 2.*np.pi*np.random.uniform()
                RV_sys_per_pl[:,i] = [RV_true(t, K_sys[i], P_sys[i], T0=T0, e=e_sys[i], w=omega) for t in t_array]
            RV_sys = np.sum(RV_sys_per_pl, axis=1)
            RV_absmax = np.max([-np.min(RV_sys), np.max(RV_sys)])
            
            # To plot the true RV time series:
            plt.plot(t_array, RV_sys, '-', color='k', label='RV of total system')
            plt.plot(t_array, RV_sys_per_pl[:,id_pl_cond], ':', color='r', label='RV of conditioned planet')
            plt.xlim([t_array[0], t_array[-1]])
            plt.ylim([-RV_absmax, RV_absmax])
            plt.yticks([-0.6*RV_absmax, 0., 0.6*RV_absmax])
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
            if j == 0:
                ax.tick_params(axis='x', labelsize=afs)
                plt.xlabel('Time (days)', fontsize=tfs)
            else:
                plt.xticks([])
        
            # If want to also report the N_obs required for measuring K:
            if outputs_RVs is None and fit_RVs:
                start = time.time()
                
                K_cond_hat_daily = np.zeros((len(N_obs_all), repeat))
                K_cond_hat_daily_ideal = np.zeros((len(N_obs_all), repeat))
                sigma_K_cond_daily = np.zeros((len(N_obs_all), repeat))
                sigma_K_cond_daily_ideal = np.zeros((len(N_obs_all), repeat))
                for n in range(repeat):
                    t_obs_daily = []
                    for i,N_obs in enumerate(N_obs_all):
                        covarsc = σ_1obs**2. * np.identity(N_obs)
                        N_obs_add = N_obs - len(t_obs_daily) # number of observations to add
                        t_obs_daily = t_obs_daily + list(len(t_obs_daily) + np.arange(N_obs_add) + t_obs_σ*np.random.random(N_obs_add))
                        
                        RV_obs_daily = np.array([RV_true_sys(t, K_sys, P_sys, T0_sys, e_sys, omega_sys) for t in t_obs_daily]) + σ_1obs*np.random.randn(N_obs) # RV obs of the system
                        RV_obs_daily_ideal = np.array([RV_true(t, K_cond, P_cond, T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond]) for t in t_obs_daily]) + σ_1obs*np.random.randn(N_obs) # RV obs of the conditioned planet only (throwing out all other planets)
                        
                        K_cond_hat_daily[i,n], sigma_K_cond_daily[i,n] = fit_rv_K_single_planet_model_GLS(t_obs_daily, RV_obs_daily, covarsc, P_sys[id_pl_cond], T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond])
                        K_cond_hat_daily_ideal[i,n], sigma_K_cond_daily_ideal[i,n] = fit_rv_K_single_planet_model_GLS(t_obs_daily, RV_obs_daily_ideal, covarsc, P_sys[id_pl_cond], T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond])
                rmsd_K_daily_all = np.sqrt(np.mean((K_cond_hat_daily - K_cond)**2., axis=1))
                rmsd_K_daily_ideal_all = np.sqrt(np.mean((K_cond_hat_daily_ideal - K_cond)**2., axis=1))
                rms_sigma_K_daily_all = np.sqrt(np.mean(sigma_K_cond_daily**2., axis=1))
                rms_sigma_K_daily_ideal_all = np.sqrt(np.mean(sigma_K_cond_daily_ideal**2., axis=1))
                
                i_N_obs_20p = rmsd_K_daily_all/K_cond < 0.2
                i_N_obs_20p_ideal = rmsd_K_daily_ideal_all/K_cond < 0.2
                N_obs_min_20p = N_obs_all[i_N_obs_20p][0] if np.sum(i_N_obs_20p) > 0 else np.nan
                N_obs_min_20p_ideal = N_obs_all[i_N_obs_20p_ideal][0] if np.sum(i_N_obs_20p_ideal) > 0 else np.nan
                rms_sigma_K_20p = rms_sigma_K_daily_all[i_N_obs_20p][0] if np.sum(i_N_obs_20p) > 0 else np.nan
                rms_sigma_K_20p_ideal = rms_sigma_K_daily_ideal_all[i_N_obs_20p_ideal][0] if np.sum(i_N_obs_20p_ideal) > 0 else np.nan
                
                stop = time.time()
                print('{:d} ({:0.1f}s): K = {:0.3f} m/s --- N_obs for RMSD(K_cond)/K_cond <20%: {:.0f} ({:.0f} ideal)'.format(j, stop-start, K_cond, N_obs_min_20p, N_obs_min_20p_ideal))
                
                s1 = '{:d}'.format(N_obs_min_20p) if ~np.isnan(N_obs_min_20p) else '>{:d}'.format(N_obs_all[-1])
                s2 = '{:d}'.format(N_obs_min_20p_ideal) if ~np.isnan(N_obs_min_20p_ideal) else '>{:d}'.format(N_obs_all[-1])
                plt.text(x=1.02*t_end, y=0, s='%s (%s)' % (s1, s2), va='center', ha='left', fontsize=10)
                if j == N_per_plot-1:
                    plt.text(x=1.02, y=1.1, s=r'$N_{\rm obs}$ (ideal)', va='bottom', ha='left', fontsize=10, transform=ax.transAxes)
                    plt.legend(loc='lower left', bbox_to_anchor=(0,1.2), ncol=1, frameon=False, fontsize=10)
            elif outputs_RVs is not None:
                ##### WARNING: double check if indexing is right
                i_of_i_sort = i_sort[id_sys] #i_sort[-(id_sys+1)]
                N_obs_max = int(np.nanmax(outputs_RVs['N_obs_min_20p']))
                N_obs_min_20p = outputs_RVs['N_obs_min_20p'][i_of_i_sort]
                N_obs_min_20p_ideal = outputs_RVs['N_obs_min_20p_ideal'][i_of_i_sort]
                
                s1 = '{:d}'.format(int(N_obs_min_20p)) if ~np.isnan(N_obs_min_20p) else '>{:d}'.format(N_obs_max)
                s2 = '{:d}'.format(int(N_obs_min_20p_ideal)) if ~np.isnan(N_obs_min_20p_ideal) else '>{:d}'.format(N_obs_max)
                plt.text(x=1.02*t_end, y=0, s='%s (%s)' % (s1, s2), va='center', ha='left', fontsize=10)
                if j == N_per_plot-1:
                    plt.text(x=1.02, y=1.1, s=r'$N_{\rm obs}$ (ideal)', va='bottom', ha='left', fontsize=10, transform=ax.transAxes)
                    plt.legend(loc='lower left', bbox_to_anchor=(0,1.2), ncol=1, frameon=False, fontsize=10)
            else: # if outputs_RVs=None and fit_RVs=False
                plt.text(x=1.02*t_end, y=0, s='{:.2f}'.format(np.round(np.max(K_sys),2)), va='center', ha='left', fontsize=10)
                if j == N_per_plot-1:
                    plt.text(x=1.02, y=1.1, s=r'$K_{\rm max}$ (m/s)', va='bottom', ha='left', fontsize=10, transform=ax.transAxes)
                    plt.legend(loc='lower left', bbox_to_anchor=(0,1.2), ncol=1, frameon=False, fontsize=10)
        
        save_name = save_name_base + '_%s.pdf' % h
        if save_fig:
            plt.savefig(save_name)
            plt.close()

def fit_RVobs_systems_conditional(sssp_per_sys, sssp, conds, N_obs_all, cond_only=False, fit_sys_cond=True, fit_all_planets=False, N_sample=20, repeat=1000, σ_1obs=1., obs_mode='daily'):
    
    # cond_only = False: set True to simulate ideal case (RV signal without any other planets)
    # fit_sys_cond = True: set True to simulate real case (fitting conditioned planet only)
    # fit_all_planets = False: set True to simulate optimistic case (fitting all planets)
    
    # σ_1obs = 1. # single-measurement uncertainity in RVs (m/s)
    # obs_mode = 'daily' or 'random' # how the observation times are drawn
    t_obs_σ = 0.2 # variation in exact 'daily' observing time (days) to avoid aliasing
    
    i_cond = condition_systems_indices(sssp_per_sys, conds)
    i_sample = np.random.choice(i_cond, N_sample, replace=False)

    Mstar_sample = sssp['Mstar_all'][i_sample]
    P_sample = sssp_per_sys['P_all'][i_sample]
    det_sample = sssp_per_sys['det_all'][i_sample]
    Mp_sample = sssp_per_sys['mass_all'][i_sample]
    Rp_sample = sssp_per_sys['radii_all'][i_sample]
    e_sample = sssp_per_sys['e_all'][i_sample]
    incl_sample = sssp_per_sys['incl_all'][i_sample]
    clusterids_sample = sssp_per_sys['clusterids_all'][i_sample]

    outputs = []
    for id_sys in range(len(i_sample)):
        start = time.time()
        
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
        
        T0_sys = P_sys*np.random.random(len(P_sys)) # reference epochs for each planet
        omega_sys = 2.*np.pi*np.random.random(len(P_sys)) # WARNING: need to get from simulated physical catalogs, NOT re-drawn
        
        K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
        id_pl_cond = np.arange(len(P_sys))[(P_sys > conds['P_lower']) & (P_sys < conds['P_upper']) & (Rp_sys > conds['Rp_lower']) & (Rp_sys < conds['Rp_upper'])][0] # index of conditioned planet
        P_cond = P_sys[id_pl_cond] # period of conditioned planet (days)
        Rp_cond = Rp_sys[id_pl_cond] # radius of conditioned planet (Earth radii)
        Mp_cond = Mp_sys[id_pl_cond] # mass of conditioned planet (Earth masses)
        K_cond = K_sys[id_pl_cond] # K of the conditioned planet (m/s)
        K_max = np.max(K_sys)
        
        # To compute the true RV time series:
        t_end = 150. # days
        t_array = np.linspace(0., t_end, 1001)
        RV_sys_per_pl = np.zeros((len(t_array),len(P_sys)))
        for i in range(len(P_sys)):
            RV_sys_per_pl[:,i] = [RV_true(t, K_sys[i], P_sys[i], T0=T0_sys[i], e=e_sys[i], w=omega_sys[i]) for t in t_array]
        RV_sys = np.sum(RV_sys_per_pl, axis=1)
        RV_absmax = np.max([-np.min(RV_sys), np.max(RV_sys)])
        
        # To simulate and fit RV observations for measuring K of the conditioned planet:
        K_cond_hat_ideal = np.zeros((len(N_obs_all), repeat)) # cond_only
        K_cond_hat = np.zeros((len(N_obs_all), repeat)) # fit_sys_cond
        K_cond_hat_fitall = np.zeros((len(N_obs_all), repeat)) # fit_all_planets
        sigma_K_cond_ideal = np.zeros((len(N_obs_all), repeat))
        sigma_K_cond = np.zeros((len(N_obs_all), repeat))
        sigma_K_cond_fitall = np.zeros((len(N_obs_all), repeat))
        sigma_K_cond_theory = σ_1obs * np.sqrt(2./N_obs_all)
        #print('##### P_sys: ', P_sys)
        for n in range(repeat):
            t_obs = []
            for i,N_obs in enumerate(N_obs_all):
                covarsc = σ_1obs**2. * np.identity(N_obs)
                if obs_mode == 'daily':
                    N_obs_add = N_obs - len(t_obs) # number of observations to add
                    t_obs = t_obs + list(len(t_obs) + np.arange(N_obs_add) + t_obs_σ*np.random.random(N_obs_add))
                elif obs_mode == 'random':
                    t_obs = np.sort(t_end*np.random.random(N_obs))
                
                if cond_only:
                    # RV time series of conditioned planet only:
                    RV_obs = np.array([RV_true(t, K_cond, P_cond, T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond]) for t in t_obs]) + σ_1obs*np.random.randn(N_obs)
                    K_cond_hat_ideal[i,n], sigma_K_cond_ideal[i,n] = fit_rv_K_single_planet_model_GLS(t_obs, RV_obs, covarsc, P_sys[id_pl_cond], T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond])
                
                # RV time series of the system:
                RV_obs = np.array([RV_true_sys(t, K_sys, P_sys, T0_sys, e_sys, omega_sys) for t in t_obs]) + σ_1obs*np.random.randn(N_obs)
                
                if fit_sys_cond:
                    K_cond_hat[i,n], sigma_K_cond[i,n] = fit_rv_K_single_planet_model_GLS(t_obs, RV_obs, covarsc, P_sys[id_pl_cond], T0=T0_sys[id_pl_cond], e=e_sys[id_pl_cond], w=omega_sys[id_pl_cond])
                
                if fit_all_planets:
                    bools_fit = K_sys > 0. #σ_1obs/2.
                    id_pl_cond_of_fits = np.where(np.arange(len(K_sys))[bools_fit] == id_pl_cond)[0][0] # index of conditioned planet counting only fitted planets
                    # Do not attempt fitting if fewer observations than free parameters (planets)
                    if N_obs < 2*np.sum(bools_fit):
                        #print('Skipping N_obs = ', N_obs)
                        K_cond_hat_fitall[i,n] = np.inf
                        continue
                    else:
                        try:
                            K_hat_all, sigma_K_all = fit_rv_Ks_multi_planet_model_GLS(t_obs, RV_obs, covarsc, P_sys[bools_fit], T0_sys[bools_fit], e_sys[bools_fit], omega_sys[bools_fit])
                            K_cond_hat_fitall[i,n], sigma_K_cond_fitall[i,n] = K_hat_all[id_pl_cond_of_fits], sigma_K_all[id_pl_cond_of_fits]
                        except:
                            print('##### Possible singular/non-positive-definite matrices; skipping N_obs = ', N_obs)
                            K_cond_hat_fitall[i,n] = np.inf
        
        rmsd_K_all_ideal = np.sqrt(np.mean((K_cond_hat_ideal - K_cond)**2., axis=1))
        rmsd_K_all = np.sqrt(np.mean((K_cond_hat - K_cond)**2., axis=1))
        rmsd_K_all_fitall = np.sqrt(np.mean((K_cond_hat_fitall - K_cond)**2., axis=1))
        rms_sigma_K_all_ideal = np.sqrt(np.mean(sigma_K_cond_ideal**2., axis=1))
        rms_sigma_K_all = np.sqrt(np.mean(sigma_K_cond**2., axis=1))
        rms_sigma_K_all_fitall = np.sqrt(np.mean(sigma_K_cond_fitall**2., axis=1))
        
        stop = time.time()
        
        data_values = (i_sample[id_sys], len(K_sys), P_cond, Rp_cond, Mp_cond, K_cond, K_max, np.sum(K_sys))
        print('{:d} ({:0.1f}s): sys_id = {:d} --- n_pl = {:d} --- P_cond = {:0.3f} d --- Rp_cond = {:0.3f} R_earth --- Mp_cond = {:0.3f} M_earth --- K_cond = {:0.3f} m/s --- K_max = {:0.3f} m/s --- K_cond/sum(K) = {:0.3f}'.format(id_sys, stop-start, i_sample[id_sys], len(K_sys), P_cond, Rp_cond, Mp_cond, K_cond, K_max, K_cond/np.sum(K_sys)))
        
        rmsd_K_cond_ferr = [0.3, 0.2, 0.1, 0.05] # fraction error rmsd(K_cond)/K_cond thresholds
        if cond_only:
            N_obs_min_ferr = []
            rms_sigma_K_ferr = []
            for ferr in rmsd_K_cond_ferr:
                i_ferr = rmsd_K_all_ideal/K_cond < ferr
                N_obs_min_ferr.append(N_obs_all[i_ferr][0] if np.sum(i_ferr) > 0 else np.nan)
                rms_sigma_K_ferr.append(rms_sigma_K_all_ideal[i_ferr][0] if np.sum(i_ferr) > 0 else np.nan)
            data_values += tuple(N_obs_min_ferr + rms_sigma_K_ferr + [np.min(rmsd_K_all_ideal/K_cond)])
            print('Ideal case --- N_obs for RMSD(K_cond)/K_cond < %s: %s --- best error = %s' % (rmsd_K_cond_ferr, N_obs_min_ferr, np.round(np.min(rmsd_K_all_ideal/K_cond),3)))
        
        if fit_sys_cond:
            N_obs_min_ferr = []
            rms_sigma_K_ferr = []
            for ferr in rmsd_K_cond_ferr:
                i_ferr = rmsd_K_all/K_cond < ferr
                N_obs_min_ferr.append(N_obs_all[i_ferr][0] if np.sum(i_ferr) > 0 else np.nan)
                rms_sigma_K_ferr.append(rms_sigma_K_all[i_ferr][0] if np.sum(i_ferr) > 0 else np.nan)
            data_values += tuple(N_obs_min_ferr + rms_sigma_K_ferr + [np.min(rmsd_K_all/K_cond)])
            print('Fit cond case --- N_obs for RMSD(K_cond)/K_cond < %s: %s --- best error = %s' % (rmsd_K_cond_ferr, N_obs_min_ferr, np.round(np.min(rmsd_K_all/K_cond),3)))
        
        if fit_all_planets:
            N_obs_min_ferr = []
            rms_sigma_K_ferr = []
            for ferr in rmsd_K_cond_ferr:
                i_ferr = rmsd_K_all_fitall/K_cond < ferr
                N_obs_min_ferr.append(N_obs_all[i_ferr][0] if np.sum(i_ferr) > 0 else np.nan)
                rms_sigma_K_ferr.append(rms_sigma_K_all_fitall[i_ferr][0] if np.sum(i_ferr) > 0 else np.nan)
            data_values += tuple(N_obs_min_ferr + rms_sigma_K_ferr + [np.min(rmsd_K_all_fitall/K_cond)])
            print('Fit all case --- N_obs for RMSD(K_cond)/K_cond < %s: %s --- best error = %s' % (rmsd_K_cond_ferr, N_obs_min_ferr, np.round(np.min(rmsd_K_all_fitall/K_cond),3)))
        
        outputs.append(data_values)

    # Make sure the dtypes match the appended outputs!
    data_dtypes = [('id_sys','i4'), ('n_pl','i4'), ('P_cond','f8'), ('Rp_cond','f8'), ('Mp_cond','f8'), ('K_cond','f8'), ('K_max','f8'), ('K_sum','f8')]
    if cond_only:
        data_dtypes += [('N_obs_min_30p_ideal','f8'), ('N_obs_min_20p_ideal','f8'), ('N_obs_min_10p_ideal','f8'), ('N_obs_min_5p_ideal','f8'), ('rms_sigma_K_30p_ideal','f8'), ('rms_sigma_K_20p_ideal','f8'), ('rms_sigma_K_10p_ideal','f8'), ('rms_sigma_K_5p_ideal','f8'), ('rmsd_best_ideal','f8')]
    if fit_sys_cond:
        data_dtypes += [('N_obs_min_30p','f8'), ('N_obs_min_20p','f8'), ('N_obs_min_10p','f8'), ('N_obs_min_5p','f8'), ('rms_sigma_K_30p','f8'), ('rms_sigma_K_20p','f8'), ('rms_sigma_K_10p','f8'), ('rms_sigma_K_5p','f8'), ('rmsd_best','f8')]
    if fit_all_planets:
        data_dtypes += [('N_obs_min_30p_fitall','f8'), ('N_obs_min_20p_fitall','f8'), ('N_obs_min_10p_fitall','f8'), ('N_obs_min_5p_fitall','f8'), ('rms_sigma_K_30p_fitall','f8'), ('rms_sigma_K_20p_fitall','f8'), ('rms_sigma_K_10p_fitall','f8'), ('rms_sigma_K_5p_fitall','f8'), ('rmsd_best_fitall','f8')]
    
    outputs = np.array(outputs, dtype=data_dtypes)
    
    # To make some additional plots:
    '''
    fig = plt.figure(figsize=(16,8))
    plot = GridSpec(2,2,left=0.1,bottom=0.1,right=0.975,top=0.95,wspace=0.2,hspace=0.4)

    ax = plt.subplot(plot[0,0])
    plt.plot(outputs['N_obs_min_20p'], outputs['K_cond']/outputs['K_sum'], 'o')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlabel(r'Number of obs. needed for RMSD(K_cond)/K_cond < 0.2', fontsize=tfs)
    plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=tfs)

    ax = plt.subplot(plot[1,0])
    sc = plt.scatter(outputs['K_cond'], outputs['K_cond']/outputs['K_sum'], c=outputs['N_obs_min_20p'])
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlabel(r'$K_{\rm cond}$', fontsize=tfs)
    plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=tfs)
    cbar = plt.colorbar(sc, orientation='horizontal')
    cbar.set_label(r'Number of obs. needed for RMSD(K_cond)/K_cond < 0.2', fontsize=10)
    
    ax = plt.subplot(plot[0,1])
    plt.plot(outputs['N_obs_min_20p'], outputs['K_cond'], 'o', color='r', label=r'$K_{\rm cond}$')
    plt.plot(outputs['N_obs_min_20p'], outputs['K_max'], 'o', color='b', label=r'$K_{\rm max}$')
    plt.plot(outputs['N_obs_min_20p'], outputs['K_sum'], 'o', color='k', label=r'$\sum{K}$')
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlabel(r'Number of obs. needed for RMSD(K_cond)/K_cond < 0.2', fontsize=tfs)
    plt.ylabel(r'$K$ (m/s)', fontsize=tfs)
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, fontsize=lfs)
    
    ax = plt.subplot(plot[1,1])
    sc = plt.scatter(outputs['K_cond'], outputs['K_cond']/outputs['K_sum'], c=outputs['rmsd_best'])
    ax.tick_params(axis='both', labelsize=afs)
    plt.xlabel(r'$K_{\rm cond}$', fontsize=tfs)
    plt.ylabel(r'$K_{\rm cond}/\sum{K}$', fontsize=tfs)
    cbar = plt.colorbar(sc, orientation='horizontal')
    cbar.set_label(r'RMSD(K_cond)', fontsize=10)
    
    if show_fig:
        plt.show()
    else:
        plt.close()
    '''
    return outputs

def fit_RVobs_single_planets_vs_K(K_array, N_obs_all, P_bounds, alpha_P=0., sigma_ecc=0.25, repeat=1000, σ_1obs=1., t_obs_σ=0.2):

    # σ_1obs = 1. # single-measurement uncertainity in RVs (m/s)
    # t_obs_σ = 0.2 # variation in exact 'daily' observing time (days) to avoid aliasing
    P_lower, P_upper = P_bounds
    assert P_lower < P_upper
    
    outputs = []
    for j,K in enumerate(K_array):
        start = time.time()
        
        # To simulate and fit RV observations for measuring K:
        K_hat_random = np.zeros((len(N_obs_all), repeat))
        K_hat_daily = np.zeros((len(N_obs_all), repeat))
        sigma_K_random = np.zeros((len(N_obs_all), repeat))
        sigma_K_daily = np.zeros((len(N_obs_all), repeat))
        sigma_K_ideal_all = σ_1obs * np.sqrt(2./N_obs_all)
        
        for n in range(repeat):
            P = np.random.power(alpha_P + 1.)*(P_upper - P_lower) + P_lower
            ecc = np.random.rayleigh(sigma_ecc)
            while ecc >= 1:
                ecc = np.random.rayleigh(sigma_ecc)
            
            T0 = P*np.random.random() # reference epoch
            omega = 2.*np.pi*np.random.random()
            
            # To compute the true RV time series:
            t_end = 150. # days
            t_array = np.linspace(0., t_end, 1001)
            RV_sys = np.array([RV_true(t, K, P, T0=T0, e=ecc, w=omega) for t in t_array])
            
            t_obs_daily = []
            for i,N_obs in enumerate(N_obs_all):
                covarsc = σ_1obs**2. * np.identity(N_obs)
                t_obs_random = np.sort(t_end*np.random.random(N_obs))
                N_obs_add = N_obs - len(t_obs_daily) # number of observations to add
                t_obs_daily = t_obs_daily + list(len(t_obs_daily) + np.arange(N_obs_add) + t_obs_σ*np.random.random(N_obs_add))
                
                RV_obs_random = np.array([RV_true(t, K, P, T0, ecc, omega) for t in t_obs_random]) + σ_1obs*np.random.randn(N_obs)
                RV_obs_daily = np.array([RV_true(t, K, P, T0, ecc, omega) for t in t_obs_daily]) + σ_1obs*np.random.randn(N_obs)
                
                K_hat_random[i,n], sigma_K_random[i,n] = fit_rv_K_single_planet_model_GLS(t_obs_random, RV_obs_random, covarsc, P, T0=T0, e=ecc, w=omega)
                K_hat_daily[i,n], sigma_K_daily[i,n] = fit_rv_K_single_planet_model_GLS(t_obs_daily, RV_obs_daily, covarsc, P, T0=T0, e=ecc, w=omega)

        stop = time.time()
        
        rmsd_K_random_all = np.sqrt(np.mean((K_hat_random - K)**2., axis=1))
        rmsd_K_daily_all = np.sqrt(np.mean((K_hat_daily - K)**2., axis=1))

        N_obs_20p = N_obs_all[rmsd_K_daily_all/K < 0.2]
        N_obs_min_20p = N_obs_20p[0] if len(N_obs_20p) > 0 else np.nan
        if len(N_obs_20p) > 0:
            print('{:d} ({:0.1f}s): K = {:0.3f} m/s --- N_obs needed for <20% error: {:d}'.format(j, stop-start, K, N_obs_20p[0]))
        else:
            print('{:d} ({:0.1f}s): K = {:0.3f} m/s --- No N_obs with <20% error; best error =  {:0.3f}'.format(j, stop-start, K, np.min(rmsd_K_daily_all/K)))
        
        outputs.append((K, N_obs_min_20p, np.min(rmsd_K_daily_all)))

    outputs = np.array(outputs, dtype=[('K','f8'), ('N_obs_min_20p','f8'), ('rmsd_best','f8')])
    return outputs

def plot_scatter_K_vs_P_conditional(sssp_per_sys, sssp, conds, log_y=False, fig_size=(8,5), afs=20, tfs=20, lfs=16, save_name='no_name_fig.pdf', save_fig=False):
    
    i_cond = condition_systems_indices(sssp_per_sys, conds)

    Mstar_sample = sssp['Mstar_all'][i_cond]
    P_sample = sssp_per_sys['P_all'][i_cond]
    det_sample = sssp_per_sys['det_all'][i_cond]
    Mp_sample = sssp_per_sys['mass_all'][i_cond]
    Rp_sample = sssp_per_sys['radii_all'][i_cond]
    e_sample = sssp_per_sys['e_all'][i_cond]
    incl_sample = sssp_per_sys['incl_all'][i_cond]
    clusterids_sample = sssp_per_sys['clusterids_all'][i_cond]

    K_sample = []
    max_pl = np.shape(P_sample)[1]
    for i in range(len(P_sample)):
        Mstar_sys = Mstar_sample[i]
        P_sys = P_sample[i]
        Mp_sys = Mp_sample[i]
        Rp_sys = Rp_sample[i]
        e_sys = e_sample[i]
        incl_sys = incl_sample[i]
        
        Mp_sys = Mp_sys[P_sys > 0]
        Rp_sys = Rp_sys[P_sys > 0]
        e_sys = e_sys[P_sys > 0]
        incl_sys = incl_sys[P_sys > 0]
        P_sys = P_sys[P_sys > 0]
        
        K_sys = rv_K(Mp_sys, P_sys, e=e_sys, i=incl_sys, Mstar=Mstar_sys)
        K_sample.append(list(K_sys) + [0]*(max_pl - len(K_sys)))
    K_sample = np.array(K_sample)

    # To make a scatter plot of K/K_max vs P:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1,1,left=0.15,bottom=0.15,right=0.95,top=0.85,wspace=0,hspace=0)
    ax = plt.subplot(plot[:,:])
    plt.title('Systems conditioned on a planet with\n' r'$P = [%s,%s]$d, $R_p = [%s,%s] R_\oplus$' % (conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
    real_det_bools = (P_sample > 0) & (det_sample == 1)
    real_undet_bools = (P_sample > 0) & (det_sample == 0)
    plt.scatter(P_sample[real_det_bools], (K_sample/np.max(K_sample, axis=1)[:,None])[real_det_bools], s=10.*(Rp_sample[real_det_bools])**2., c=K_sample[real_det_bools], vmin=0., vmax=5., label='Detected')
    plt.scatter(P_sample[real_undet_bools], (K_sample/np.max(K_sample, axis=1)[:,None])[real_undet_bools], s=10.*(Rp_sample[real_undet_bools])**2., c=K_sample[real_undet_bools], vmin=0., vmax=5., edgecolors='r', label='Undetected')
    plt.fill_betweenx(y=[0,1], x1=conds['P_lower'], x2=conds['P_upper'], color='g', alpha=0.2)
    plt.gca().set_xscale("log")
    if log_y:
        plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    ax.set_xticks([3,10,30,100,300])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([3., 300.])
    if log_y:
        plt.ylim([np.min(K_sample[K_sample > 0]), np.max(K_sample/np.max(K_sample, axis=1)[:,None])])
    else:
        plt.ylim([0., np.max(K_sample/np.max(K_sample, axis=1)[:,None])])
    plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'$K/K_{\rm max}$', fontsize=tfs)
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=True, fontsize=lfs)
    cbar = plt.colorbar()
    cbar.set_label(r'$K$', fontsize=lfs)
    
    if save_fig:
        plt.savefig(save_name + 'v1.pdf')
        plt.close()

    # To make a scatter plot of K vs P:
    fig = plt.figure(figsize=fig_size)
    plot = GridSpec(1,1,left=0.15,bottom=0.15,right=1,top=0.85,wspace=0,hspace=0)
    ax = plt.subplot(plot[:,:])
    plt.title('Systems conditioned on a planet with\n' r'$P = [%s,%s]$d, $R_p = [%s,%s] R_\oplus$' % (conds['P_lower'], conds['P_upper'], conds['Rp_lower'], conds['Rp_upper']), va='center', ha='center', fontsize=tfs)
    real_det_bools = (P_sample > 0) & (det_sample == 1)
    real_undet_bools = (P_sample > 0) & (det_sample == 0)
    plt.scatter(P_sample[real_det_bools], K_sample[real_det_bools], s=10.*(Rp_sample[real_det_bools])**2., c=(K_sample/np.max(K_sample, axis=1)[:,None])[real_det_bools], label='Detected')
    plt.scatter(P_sample[real_undet_bools], K_sample[real_undet_bools], s=10.*(Rp_sample[real_undet_bools])**2., c=(K_sample/np.max(K_sample, axis=1)[:,None])[real_undet_bools], edgecolors='r', label='Undetected')
    plt.fill_betweenx(y=[0,np.max(K_sample)], x1=conds['P_lower'], x2=conds['P_upper'], color='g', alpha=0.2)
    plt.gca().set_xscale("log")
    if log_y:
        plt.gca().set_yscale("log")
    ax.tick_params(axis='both', labelsize=afs)
    ax.set_xticks([3,10,30,100,300])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.xlim([3., 300.])
    if log_y:
        plt.ylim([np.min(K_sample[K_sample > 0]), np.max(K_sample)])
    else:
        plt.ylim([0., np.max(K_sample)])
    plt.xlabel(r'Orbital period $P$ (days)', fontsize=tfs)
    plt.ylabel(r'RV semi-amplitude $K$ (m/s)', fontsize=tfs)
    plt.legend(loc='upper right', bbox_to_anchor=(1,1), ncol=1, frameon=True, fontsize=lfs)
    cbar = plt.colorbar()
    cbar.set_label(r'$K/K_{\rm max}$', fontsize=lfs)
    
    if save_fig:
        plt.savefig(save_name + 'v2.pdf')
        plt.close()



def generate_latex_table_RVobs_systems_conditional(outputs_RVs, nan_str=r'$>1000$', N_sample=40):
    table_array = []
    for i in range(N_sample):
        osys = outputs_RVs[i]
        N_obs_str = '{:d}'.format(int(osys['N_obs_min_20p'])) if not np.isnan(osys['N_obs_min_20p']) else nan_str
        N_obs_fitall_str = '{:d}'.format(int(osys['N_obs_min_20p_fitall'])) if not np.isnan(osys['N_obs_min_20p_fitall']) else nan_str
        N_obs_ideal_str = '{:d}'.format(int(osys['N_obs_min_20p_ideal'])) if not np.isnan(osys['N_obs_min_20p_ideal']) else nan_str
        row_str = '{:d} & {:0.3f} & {:0.3f} & {:0.3f} & {:0.3f} & {:0.3f} & {:0.3f} & '.format(int(osys['n_pl']), osys['P_cond'], osys['Rp_cond'], osys['Mp_cond'], osys['K_cond'], osys['K_max'], osys['K_sum'])
        row_str += ' & '.join([N_obs_str, N_obs_fitall_str, N_obs_ideal_str]) + r' \\'
        table_array.append(row_str)
    table_array = np.array(table_array)
    return table_array
    



# Functions to fit and use a line for a set of simulations for log(N_obs) vs. log(K_cond) (ideal single-planet case):

def fit_line_loglog_Nobs_K_single_planets(outputs_ideal, σ_1obs, p0):
    # outputs_ideal: dictionary with simulated RV observations ('K' and 'N_obs_min_20p' fields)
    # σ_1obs: single measurement precision, to serve as normalization point K_norm
    # p0: initial guesses for line parameters N_obs(K_norm) and slope
    
    K_norm = σ_1obs # single measurement precision serves as normalization point
    f_linear = lambda p, x: p[0] + p[1]*x - p[1]*np.log10(K_norm)
    f_err = lambda p, x, y: y - f_linear(p,x)
    
    log_K = np.log10(outputs_ideal['K'][outputs_ideal['N_obs_min_20p'] > 5]) # this should remove NaNs and 5s
    log_N_obs = np.log10(outputs_ideal['N_obs_min_20p'][outputs_ideal['N_obs_min_20p'] > 5])
    fit = scipy.optimize.leastsq(f_err, p0, args=(log_K, log_N_obs), full_output=1)
    logN_slope = fit[0]
    logN = logN_slope[0] # normalization parameter; log10(N_obs) at K_norm
    slope = logN_slope[1] # slope parameter
    return logN, slope

def linear_logNobs_logK(K, K_norm, Nobs_norm, slope, Nobs_min=5, round_to_ints=True):
    logK = np.log10(np.array(K))
    logNobs_K = slope*(logK - np.log10(K_norm)) + np.log10(Nobs_norm) # log(N_obs) at K
    Nobs_K = 10.**logNobs_K # N_obs at K
    Nobs_K[Nobs_K < 5] = 5
    if round_to_ints:
        Nobs_K = np.round(Nobs_K).astype(int)
    return Nobs_K
        
