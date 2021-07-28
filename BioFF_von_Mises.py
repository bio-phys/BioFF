"""
Copyright (c) 2021 Juergen Koefinger, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
"""

import scipy.stats
import copy, matplotlib
from bioen import optimize
import bioen 
import polymer_model as pm
from importlib import reload
import binless_wham as WHAM
import numpy as np

def reweighting(u, u_ref, w_ref=1):
    """
    Boltzmann reweigthing of an ensemble. The states in the ensemble are either 
    distributed according to the reference distribution sucht that all reference 
    weights are equal (1/N) or they have been drawn from a different distribution 
    and their weights in the reference ensemble are given (w_ref). The latter 
    is the case if WHAM has been used to combine different simulations and to 
    calculate the corresponding weights in the reference ensemble.    
    """
    du = u - u_ref
    du -= np.max(du)
    weights = w_ref*np.exp(-du)
    weights /= weights.sum()
    return weights

def von_Mises_weights(phi_mat, kappa, u_ref, w_ref=1):
    u = pm.calc_Us(phi_mat, kappa)
    weights = reweighting(u, u_ref, w_ref=1)
    return weights

def SKL(w, w0):
    """
    Kullback-Leibler divergence.
    """
    idx = np.where(w!=0)[0]
    return (w[idx]*np.log(w[idx]/w0[idx])).sum()

def chi2(w, yTilde, YTilde):
    """
    Chi-square for 1d obervable. 
    """
    return (((w*yTilde).sum()-YTilde)**2).sum()

def neg_log_posterior(w, w0, yTilde, YTilde, theta):
    """
    The BioEn negative log-posterior. 
    """
    SKL_val = SKL(w, w0)
    chi2_val = chi2(w, yTilde, YTilde)
    return theta*SKL_val + chi2_val/2., SKL_val, chi2_val

def set_nj(cfx_dict):
    """
    Calculation of the list of structures in the simulation to be combined with binless WHAM. 
    Returns input for the binless WHAM code. 
    """
    nj = []
    for kappa in cfx_dict:
        nj.append(len(cfx_dict[kappa]))
    nj=np.asarray(nj)
    n_struct = nj.sum()
    return nj, n_struct

def get_wkj(nj, kappas_WHAM, phi_mats, kappa_of_interest):
    """
    Calculation of the matrix of biases wkj for binless WHAM. 
    """
    n_struct = nj.sum()
    n_bias = len(kappas_WHAM)
    wkj = np.zeros((n_struct, n_bias))
    for i_win, n in enumerate(nj):
        i1 = nj[:i_win].sum()
        i2 = nj[:i_win+1].sum()
        kappa_win = kappas_WHAM[i_win]
        for i_bias, kappa in enumerate(kappas_WHAM):
            new_energies = pm.von_Mises_energy(phi_mats[kappa_win], kappa_of_interest)
            energies = pm.von_Mises_energy(phi_mats[kappa_win], kappa)
            biases = new_energies-energies
            wkj[i1:i2, i_bias] = -biases 
    return wkj

def effective_sample_size(weights):
    """
    Kish's effective sample size
    """
    return (weights.sum())**2/(weights**2).sum()

def rank_weights(w):
    """
    Ranking of the weights in decending order and calculation of the corresponding 
    cumulative weights (complementary CDF)
    """
    rank=np.arange(1, len(w)+1)
    idx = np.argsort(w)
    return rank, (w[idx][::-1]).cumsum()
  
def calc_n_states_given_weight_fraction(w, weight_mass):    
    """
    Calculate the number of weights for which the complementary cumulative weights reach 
    the value weight_mass. That is, the number of the largest weights that combined have 
    a statistical weight given by weight_mass. weight_mass is just one minus the p-value. 
    """
    rank, w_ranked = rank_weights(w)
    idx = np.searchsorted(w_ranked, weight_mass)
    n = rank[idx]
    return rank, w_ranked, n

def bin_range(x):
    """
    Convenience function to calculate the binning range based on the minumum and maximum value of a list-like input. 
    """
    return (np.min(x), np.max(x))

def histogram(x, n_bins=100, range=None, density=True, weights=None):
    """
    Convenience function, which not only evaluates numpy.histogram but also returns the bin centers. 
    """
    h, e = np.histogram(x, bins=n_bins, range=range, density=density, weights=weights)
    c = (e[1:]+e[:-1])/2.
    return h, c

def energy_distributions(all_energies, kappas, weights, n_bins=1000, range=None):
    """
    Convenience function to calculate histograms the same binning once without and also with weights. 
    """
    E_range = bin_range(all_energies)
    h, c = histogram(all_energies, n_bins=n_bins, range=E_range, density=False)
    q_w={}
    for kappa in kappas:
        q_w[kappa], c_w = histogram(all_energies, weights=weights[kappa], range=E_range, n_bins=n_bins, density=True)
    return c, h, q_w

def convergence(kappa_old, kappa_new, threshold=0.05):
    """
    Checks if absolute of relative change in kappa is smaller than threshold. Used to determine 
    convergence in BioFF-by-reweighting. 
    """
    relative_change = np.abs((kappa_old-kappa_new)/kappa_new)
    if relative_change<threshold:
        return True
    else:
        return False
    
def f_threshold(n, n0, k=10.):
    """
    Enforcing lower limit on effective sample size n. Used to define new objective function for
    optimization is product with the the neg. log-poster. 
    """
    return np.exp(-k*(n-n0)/n0)+1

def objective_function(kappa, w0, all_phi_mats, yTilde, YTilde,  all_u_refs, theta, n_threshold = 300, k_threshold=10.):
    """
    Objective function for BioFF optimization.
    """
    all_u = pm.von_Mises_energy(all_phi_mats, kappa)
    w = reweighting(all_u, all_u_refs, w0)
    L_val, SKL_val, chi2_val = neg_log_posterior(w, w0, yTilde, YTilde, theta)
    n = effective_sample_size(w)
    return L_val*f_threshold(n, n_threshold, k_threshold)