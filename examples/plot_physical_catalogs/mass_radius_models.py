# To import required modules:
import numpy as np
import scipy.interpolate

from syssimpyplots.general import *





##### To load some mass-radius tables:

# NWG-2018 model:
MR_table_file = '../../src/syssimpyplots/data/MRpredict_table_weights3025_R1001_Q1001.txt'
with open(MR_table_file, 'r') as file:
    lines = (line for line in file if not line.startswith('#'))
    MR_table = np.genfromtxt(lines, names=True, delimiter=', ')

# Li Zeng models:
# https://www.cfa.harvard.edu/~lzeng/tables/massradiusEarthlikeRocky.txt
# https://www.cfa.harvard.edu/~lzeng/tables/massradiusFe.txt
MR_earthlike_rocky = np.genfromtxt('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters-select_files/Miscellaneous_data/MR_earthlike_rocky.txt', names=['mass','radius']) # mass and radius are in Earth units
MR_pure_iron = np.genfromtxt('/Users/hematthi/Documents/GradSchool/Research/ExoplanetsSysSim_Clusters-select_files/Miscellaneous_data/MR_pure_iron.txt', names=['mass','radius']) # mass and radius are in Earth units

# To construct an interpolation function for each MR relation:
MR_NWG2018_interp = scipy.interpolate.interp1d(10.**MR_table['log_R'], 10.**MR_table['05'])
MR_earthlike_rocky_interp = scipy.interpolate.interp1d(MR_earthlike_rocky['radius'], MR_earthlike_rocky['mass'])
MR_pure_iron_interp = scipy.interpolate.interp1d(MR_pure_iron['radius'], MR_pure_iron['mass'])

# To find where the Earth-like rocky relation intersects with the NWG2018 mean relation (between 1.4-1.5 R_earth):
def diff_MR(R):
    M_NWG2018 = MR_NWG2018_interp(R)
    M_earthlike_rocky = MR_earthlike_rocky_interp(R)
    return np.abs(M_NWG2018 - M_earthlike_rocky)
# The intersection is approximately 1.472 R_earth
radii_switch = 1.472

radii_min, radii_max = 0.5, 10.

# IDEA 1: Normal distribution for rho centered around Earth-like rocky, with a sigma_rho that grows with radius
# To define sigma_rho such that log10(sigma_rho) is a linear function of radius:
rho_earthlike_rocky = rho_from_M_R(MR_earthlike_rocky['mass'], MR_earthlike_rocky['radius']) # mean density (g/cm^3) for Earth-like rocky as a function of radius
rho_pure_iron = rho_from_M_R(MR_pure_iron['mass'], MR_pure_iron['radius']) # mean density (g/cm^3) for pure iron as a function of radius

sigma_rho_at_radii_switch = 3. # std of mean density (g/cm^3) at radii_switch
sigma_rho_at_radii_min = 1. # std of mean density (g/cm^3) at radii_min
rho_radius_slope = (np.log10(sigma_rho_at_radii_switch)-np.log10(sigma_rho_at_radii_min)) / (radii_switch - radii_min) # dlog(rho)/dR; slope between radii_min and radii_switch in log(rho)
sigma_rho = 10.**( rho_radius_slope*(MR_earthlike_rocky['radius'] - radii_min) + np.log10(sigma_rho_at_radii_min) )

# IDEA 2: Lognormal distribution for mass centered around Earth-like rocky, with a sigma_log_M that grows with radius
# To define sigma_log_M as a linear function of radius:
sigma_log_M_at_radii_switch = 0.3 # std of log_M (Earth masses) at radii_switch
sigma_log_M_at_radii_min = 0.04 # std of log_M (Earth masses) at radii_min
sigma_log_M_radius_slope = (sigma_log_M_at_radii_switch - sigma_log_M_at_radii_min) / (radii_switch - radii_min)
sigma_log_M = sigma_log_M_radius_slope*(MR_earthlike_rocky['radius'] - radii_min) + sigma_log_M_at_radii_min

##### [H20 adopted "IDEA 2" for the modification below 'radii_switch] #####
# H20 model:
end_ELR = 27-1
start_NWG = 284-1 # index closest to log10(R=1.472)
radius_evals_H20 = np.concatenate((MR_earthlike_rocky['radius'][:end_ELR], 10.**MR_table['log_R'][start_NWG:]))
mass_evals_med_H20 = np.concatenate((MR_earthlike_rocky['mass'][:end_ELR], 10.**MR_table['05'][start_NWG:]))
mass_evals_016_H20 = np.concatenate((10.**(np.log10(MR_earthlike_rocky['mass'])-sigma_log_M)[:end_ELR], 10.**MR_table['016'][start_NWG:]))
mass_evals_084_H20 = np.concatenate((10.**(np.log10(MR_earthlike_rocky['mass'])+sigma_log_M)[:end_ELR], 10.**MR_table['084'][start_NWG:]))
