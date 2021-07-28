"""
Copyright (c) 2021 Juergen Koefinger, Max Planck Institute of Biophysics, Frankfurt am Main, Germany
"""

import numpy as np
from numba import jit
import scipy.stats
import copy
import pickle

def hstack_dict(dictionary, ordered_keys):
    """
    Stacks the arrays of a dictionary in a single array according to the order given by the a list of keys. 
    """
    return np.hstack([dictionary[k] for k in ordered_keys])

def get_cdf(x):
    """
    Cumulative distribution function without binning. 
    """
    x=np.sort(x)
    #x.sort()
    cdf  = np.arange(1, x.shape[0]+1)
    cdf = cdf/cdf.shape[0]
    return x, cdf

@jit(nopython=True)
def distances(pos, r):
    """
    Calculate all pair-distances. 
    """
    N = len(pos)-1
    counter=0
    for i in range(N-1):
        for j in range(i+1,N):
            r[counter]=((pos[i]-pos[j])**2).sum()
            counter+=1
    r=np.sqrt(r)
    return r

def get_phi_matrix(configs):
    """
    Generate 2d array out of list of 1d arrays. 
    """
    phi_mat = []
    for config in configs:
        phi_mat.append(config['phis'])
    return np.transpose(np.asarray(phi_mat))

def von_Mises_probs(phis, kappa, mu=0):
    """
    List of von Mises proabilties for list of angles phi. 
    """
    probs = scipy.stats.vonmises.pdf(phis, kappa=kappa, loc=mu)
    return probs

def von_Mises_prob(phis, kappa, mu=0):
    """
    Joint von Mises probability for list of angles phi. 
    """
    probs = scipy.stats.vonmises.pdf(phis, kappa=kappa, loc=mu)
    return probs.prod()

def von_Mises_energy(phis, kappa, mu=0, axis=0):
    """
    Probability of a von Mises polymer. 
    Returns:
    Von Mises energy as negative logarithm of joint probability.
    """
    return -np.log(np.prod(von_Mises_probs(phis, kappa, mu=mu), axis=axis))

# def calc_Us(phi_mat, kappa):
#     """
#     Legacy function. Same as von_Mises_energy().
#     """
#     us = von_Mises_energy(phi_mat, kappa, mu=0, axis=0)
#     return us 

def recalculate_von_Mises_energies(configs, kappa, mu=0):
    """
    Recalculate von Mises energies for dictionary of configurations and return single array. 
    """
    new_energies = []
    for config in configs:
        new_energies.append(von_Mises_energy(config['phis'], kappa, mu=mu))
    new_energies = np.asarray(new_energies)
    return(new_energies) 

# def recalculate_von_Mises_energies_from_phi_mat(phi_mat, kappa, mu=0):
#     """
#     """
#     new_energies = von_Mises_energy(phi_mat, kappa, mu=mu, axis=0)
#     return new_energies

def get_vecs(vecs, phis):
    """
    Bond vectors from angles drawn from von Mises distribution.  
    """
    cum_phis = phis.cumsum()
    vecs[:,0]=np.sin(cum_phis)
    vecs[:,1]=np.cos(cum_phis)
    return vecs

def get_pos_from_vecs(vecs, pos):
    """
    Bead positions from bond vectors. 
    """
    N = len(vecs)
    pos = np.zeros((N+1,2))
    for i in range(N):
        pos[i+1,:]=pos[i,:]+vecs[i,:]
    return pos

def center_config(config):
    """
    Center bead postions. Used for visualization of polymers (2d plots).
    """
    mean = config['pos'].mean(axis=0)
    config['pos'][:,0]-=mean[0]
    config['pos'][:,1]-=mean[1]
    return config


def recalculate_from_phis(config, kappa, mu=0):
    """
    Calculate bond vectors, bead positions, all inter-bead distances, 
    and the energy of a polymer configuration. 
    """
    config['vecs'] = get_vecs(config['vecs'], config['phis'])
    config['pos'] = get_pos_from_vecs(config['vecs'], config['pos'])
    config['r'] = distances(config['pos'], config['r'])
    config['E_vonMises']  = von_Mises_energy(config['phis'], kappa, mu=mu)
    config = center_config(config)
    return config

def random_config(N, kappa=1, mu=0, q_recalc=True):
    """
    Generate a random polymer configuration by drawing angles from von Mises distribution. 
    Calculate bead positions, bond vectors, inter-bead distances, and energy. 
    """
    config = init_config(N)
    config['phis'] = scipy.stats.vonmises.rvs(kappa, size=N, loc=mu)
    config = recalculate_from_phis(config, kappa, mu=mu)
    #config['E_vonMises']  = von_Mises_energy(config['phis'], kappa)
    return config

def generate_vonMises_config(N, kappa, mu):
    """
    Same as random_config(). 
    """
    return random_config(N, kappa=kappa, mu=mu)

def generate_vonMises_configs(N_configs, N, kappa, mu=0):
    """
    Generate N_configs uncorrelated  von Mises polymer configurations by drawing from the corresponding distribution. 
    """
    configs = []
    for i in range(N_configs):
        configs.append(generate_vonMises_config(N, kappa, mu))
    return configs

def init_config(N):
    """
    Initializes dictionary holding polymer configuration and addtional information. 
    r_idx_map maps a pair of bead indices onto the single index indicating 
    the corresponding distance in config['r'].
    
    """
    config={}
    config['N']  = N
    config['vecs'] = np.zeros((N,2))
    config['pos'] = np.zeros((N+1,2))
    config['phis'] = np.zeros(N)
    config['r'] = np.zeros(N*(N-1)//2)
    counter=0
    config['r_idx_map'] = np.zeros((N,N), dtype=np.int)
    counter=0
    for i in range(N-1):
        for j in range(i+1,N):
            config['r_idx_map'][i,j]=counter
            config['r_idx_map'][j,i]=counter
            counter+=1
    return config

def extract_energies(configs, key='E_tot'):
    """
    Generate 1d array out of list of 1d arrays. 
    """
    energies=[]
    for config in configs:
        energies.append(config[key])
    return np.asarray(energies)

def extract_pair_distances(configs, i1=-1, i2=-1):
    """
    Extract either pair distance give by index i1 and i2 or generate single array with all pair-distances from list of configurations.
    """
    pair_distances=[]
    #print("Extracting distance %d" % idx)
    if i1 == -1 or i2 == -1:
        for config in configs:
            pair_distances.append(config['r'])
    else:
        for config in configs:
            idx=config['r_idx_map'][i1,i2]
            pair_distances.append(config['r'][idx])
    return np.asarray(pair_distances)

    
#     #print("Extracting distance %d" % idx)
#     for config in configs:
#         idx=config['r_idx_map'][i1,i2]
#         pair_distances.append(config['r'][idx])
#     return np.asarray(pair_distances)

def load_configs(N, kappa, epsilon, k_rep,  sample_info, ipath="./", pref=""):
    """
    Load configuration using pickle. 
    """
    name=pref+"configs_N%d_kappa%4.3f_eps%4.3f_krep%4.3f_nrun%d_nsample%d" % (N, kappa, epsilon, k_rep,  sample_info['n_run'], sample_info['n_sample'])
    with open(ipath+"/"+name+".pkl", 'br') as fp:
        configs = pickle.load(fp)
    return configs

def dump_configs(configs, N, kappa, epsilon, k_rep,  sample_info, opath="./", pref=""):
    """
    Dump configuration using pickle. 
    """
    name=pref+"configs_N%d_kappa%4.3f_eps%4.3f_krep%4.3f_nrun%d_nsample%d" % (N, kappa, epsilon, k_rep,  sample_info['n_run'], sample_info['n_sample'])
    with open(opath+"/"+name+".pkl", 'bw') as fp:
        pickle.dump(configs, fp)
    print(name)
    
def update_energies(phi_mats, kappas, energies):
    """
    Recalculate energies and update dictionary 'energies' for specified kappa values ('kappas')
    """
    for kappa in kappas:
        #energies[kappa] = recalculate_von_Mises_energies_from_phi_mat(phi_mats[kappa], kappa)   
        energies[kappa] =von_Mises_energy(phi_mats[kappa], kappa)
    return 

def update_dists(cfx_dict, kappas, i1, i2, key, dists):
    """
    Update distances dictionary 'dists' for specified kappa values ('kappas') 
    from list of configurations 'cfx_dict'.
    """
    for kappa in kappas:
        dists[key][kappa] = extract_pair_distances(cfx_dict[kappa], i1, i2)    
    return 

def update_dist_stats(dists, kappas, dists_stats):
    """
    Update distances dictionary 'dists_stats' for specified kappa values ('kappas') 
    from dictionary distances.
    """
    for key  in  dists:
        for kappa in kappas:
            dists_stats[key][kappa] = {}
            dists_stats[key][kappa]['mean'] = dists[key][kappa].mean()
            dists_stats[key][kappa]['std'] = dists[key][kappa].std()
    return 

def init_from_single_sim(kappa, N_beads, epsilon, k_rep, sample_info, idx_labels, ipath, pref="vonMises_", N_configs=None):
    """
    Initialize configurations and 'label' and 'e2e' distance information from file. 
    """
    cfx_dict={}
    cfx_dict[kappa] = load_configs(N_beads, kappa, epsilon, k_rep,  sample_info, ipath=ipath, pref=pref)
    if N_configs!=None:
        cfx_dict[kappa]=cfx_dict[kappa][:N_configs]
    phi_mats={}
    phi_mats[kappa] = get_phi_matrix(cfx_dict[kappa])
    energies = {}
    update_energies(phi_mats, phi_mats.keys(), energies)
    dists = {}
    dists['e2e']={}
    update_dists(cfx_dict, cfx_dict.keys(), 0, N_beads-1, 'e2e', dists)
    dists['label']={}
    update_dists(cfx_dict, cfx_dict.keys(), idx_labels[0], idx_labels[1], 'label', dists)

    dists_stats={}
    dists_stats['e2e']={}
    dists_stats['label']={}
    update_dist_stats(dists, [kappa], dists_stats)
    return cfx_dict, phi_mats, energies, dists, dists_stats

