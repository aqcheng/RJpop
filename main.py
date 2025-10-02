import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from scipy.interpolate import InterpolatedUnivariateSpline as spline

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, DistributionGenerateRJ
from eryn.moves.tempering import make_ladder
from eryn.prior import ProbDistContainer, uniform_dist

import pdfs as pdfs

from astropy.cosmology import Planck15
import astropy.units as u

try:
    import cupy as xp
    xp.cuda.runtime.setDevice(1)
    from cupyx import scatter_add

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    print(f'Using cpu: {e}')
    def scatter_add(a, slices, value):
        res = xp.array(a)
        xp.add.at(res, slices, value)
        return res

parser = argparse.ArgumentParser()
# General
parser.add_argument('-s', '--seed', '-seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--test', '-test', action='store_true', help='Run in test mode (fewer steps)')
parser.add_argument('--prior', '-prior', type=str, help='Prior json file')

#runtime arguments
parser.add_argument('--nwalkers', '-nwalkers', type=int, default=40, help='Number of walkers per temperature')
parser.add_argument('--ntemps', '-ntemps', type=int, default=5, help='Number of temperatures (for parallel tempering)')
parser.add_argument('--nsteps', '-nsteps', type=int, default=5000, help='Number of steps')
parser.add_argument('--burn', '-burn', type=int, default=2000, help='Number of burn-in steps')
parser.add_argument('--thin-by', '-thin_by', type=int, default=1, help='Thinning factor')
parser.add_argument('--outdir', '-outdir', type=str, help='Output directory')

# Hyperparameters
parser.add_argument('-n', '--ncomp-max', '--ncomp_max', '-ncomp_max', dest='ncomp_max', type=int, default=10,
                help='Maximum number of components for the mass model')
parser.add_argument('--use-mchirp', '--use_mchirp', '-use_mchirp', dest='use_mchirp', action='store_true',
                help='Use chirp mass instead of m1')

args = parser.parse_args()
xp.random.seed(args.seed)

# save settings
with open(os.path.join(args.outdir, 'settings.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# list of models and their hyperparameters -- these are manually hard coded in 
MODELS = {
    'mass': {
        'skew-t': {
            'model': pdfs.jf_skew_t(),
            'param_latex': {
                'a': r'\alpha_1',
                'b': r'\alpha_2',
                'loc': r'\mu_m',
                'scale': r'\sigma_m',
                'xmin': r'm_{\min}',
                'xmax': r'm_{\max}',
            },
            'params_fix': {}
        },
        'PL': {
            'model': pdfs.smoothed_powerlaw(),
            'param_latex': {
                'alpha': r'\alpha',
                'p': r'p_m',
                'xmin': r'm_{\min}',
                'xmax': r'm_{\max}',
            },
            'params_fix': {}
        }
    },
    'chieff': {
        'gen_gauss': {
            'model': pdfs.gen_gaussian(),
            'param_latex': {
                'beta_chieff': r'$\beta$',
                'mu_chieff': r'$\mu_{\chi}$',
                'sigma_chieff': r'$\sigma_{\chi}$',
            },
            'params_fix': {
                'xmin': -1.,
                'xmax': 1.
            }
        },
        'gauss': {
            'model': pdfs.gaussian(),
            'param_latex': {
                'loc': r'$\mu_{\chi}$',
                'scale': r'$\sigma_{\chi}$',
            },
            'params_fix': {
                'xmin': -1.,
                'xmax': 1.
            }
        }
    },
    'rate': {
        'MD': {
            'model': pdfs.MD_rate(cosmo=Planck15),
            'param_latex': {
                'gamma': r'$\gamma$',
                'kappa': r'$\kappa$',
                'zp': r'$z_p$',
            }, 
            'params_fix': {}
        },
        'PL': {
            'model': pdfs.PL_rate(cosmo=Planck15),
            'param_latex': {
                'gamma': r'$\gamma$',
                'R0': '$R_0$',
            },
            'params_fix': {}
        }
    }
}
# set up prior
with open(args.prior_json, 'r') as f: 
    # this should be a list of dictionaries, one for each branch, the last of which should be global
    # each dictionary should have keys corresponding to each model (mass, q, chieff, rate), with the value as a subdictionary
    # each dictionary should also have a "__branch__" key corresponding to the branch name
    # each subdictionary should have the prior ranges for each parameter as a list or a float if parameter should be fixed,
    # as well as a '__model__' key corresponding to the model name
    # parameter names must match the names defined in the functions (in pdfs.py)
    branch_priordicts = json.load(f)
    BRANCH_NAMES = [d['__branch__'] for d in branch_priordicts]

priors = {} # {branch_name: ProbDistContainer({i: prior})}

HP_ORDERING = [{} for _ in BRANCH_NAMES]
BRANCH_DIMS = []

# construct dictionary with the primary models belonging to each branch
# note that models whose primary model are local could also have global parameters (hence branch_id list)
# [ {param: {  __model___: name of model (e.g. skew_t, gaussian),
#          model_hp_names: [hp_name1, hp_name2, ...], 
#          latex_hp_names: [hp_latex1, hp_latex2, ...],
#              branch_idx: [branch_idx1, branch_idx2, ...]       i.e. which branch the hyperparameter belongs to (should be either the list index or -1 for global)
#                 col_idx: (param_idx1, param_idx2) }]            i.e. which column in data array in the branch the hyperparameter is found
for branch_idx, input_dict in enumerate(branch_priordicts):
    branch_name = input_dict.pop('__branch__')
    branch_priordict = {}
    hp_idx = 0
    for param, param_priordict in input_dict.items():
        model_name = param_priordict.pop('__model__')
        HP_ORDERING[branch_idx][param] = {
            '__model__': model_name,
            'model_hp_names': [], 
            'latex_hp_names': [], 
            'branch_idx': [], 
            'col_idx': []
        }
        for hp, hp_val in param_priordict.items():
            
            if isinstance(hp_val, list) or isinstance(hp_val, tuple):
                branch_priordict[hp_idx] = uniform_dist(hp_val[0], hp_val[1])
                HP_ORDERING[branch_idx][param]['model_hp_names'].append(hp)
                HP_ORDERING[branch_idx][param]['latex_hp_names'].append(MODEL_METAINFO[param]['param_latex'].get(hp, hp))
                HP_ORDERING[branch_idx][param]['branch_idx'].append(branch_idx)
                HP_ORDERING[branch_idx][param]['col_idx'].append(hp_idx)
                hp_idx += 1
            elif isinstance(hp_val, float) or isinstance(hp_val, int):
                MODELS[param][model_name]['params_fix'][hp] = hp_val
            else:
                raise ValueError(f"Invalid prior range for {param} {hp}: {hp_val}")
            
            if hp == 'R0':
                R0_BRANCH_IDX = branch_idx
                R0_COL_IDX = hp_idx
    priors[branch_name] = ProbDistContainer(dict(branch_priordict))
    BRANCH_DIMS.append(hp_idx)

# save hp_ordering
with open(os.path.join(args.outdir, 'hyperparameter_ordering_metainfo.json'), 'w') as f:
    json.dump(HP_ORDERING, f, indent=4)

def loglike(hyperparams, groups, data, injections):
    # eq 5 of https://arxiv.org/pdf/2305.08909
    
    # hyperparams is a tuple of parameters corresponding to each branch
    # hyperparams of each branch come in a tot_nleaves x ndims, each corresponding to an active leaf of the branch
    # groups labels each leaf with a 'group id' that corresponds to its (temperature, walker) combination

    # tot_nleaves x (ampl, m1 hp, q hp, chieff hp, ...), nevs x nsamples -> ngroups x nevs x nsamples 
    # -> ngroups x nevs (sum) -> ngroups (product)

    ngroups = xp.amax(groups[-1]) + 1 # since global groups is just range(ngroups)
    nevs, nsamples = data['q'].shape
    ninjs = injections['q'].shape[0] 

    dNdtheta_per_group = xp.zeros((ngroups, nevs * nsamples), dtype=xp.float64) # evaluated on PE samples
    dNdtheta_per_group_injs = xp.zeros((ngroups, ninjs), dtype=xp.float64) # evaluated on injections

    # iterate over branches -- different local branches will have different models
    for branch_idx, branch_dict in enumerate(HP_ORDERING[:-1]): # all local branches

        branch_groups = groups[branch_idx]
        branch_nleaves = hyperparams[branch_idx].shape[0]

        dNdtheta_per_leaf = xp.ones((branch_nleaves, 1, 1), dtype=xp.float64) # evaluated on PE samples
        dNdtheta_per_leaf_injs = xp.ones((branch_nleaves, 1), dtype=xp.float64) # evaluated on injections

        # evaluate likelihood for each event-level parameter
        for param_name, param_dict in branch_dict.items(): 

            model_func = MODELS[param_name][param_dict['__model__']]['model']

            model_hp_names = param_dict['model_hp_names']
            model_hp_branch_idxs = param_dict['branch_idx']
            model_hp_col_idxs = param_dict['col_idx']

            # model_hp_vals_dict will be unpacked into the model function
            model_hp_vals_dict = dict(MODEL_METAINFO[param]['params_fix']) # start with fixed hyperparameters
            
            # now get the hyperparameter values from this branch or the global branch
            for hp, hp_branch_idx, hp_col_idx in zip(model_hp_names, model_hp_branch_idxs, model_hp_col_idxs):

                hp_vals = hyperparams[hp_branch_idx][:, hp_col_idx]

                if hp_branch_idx != branch_idx: 
                    # should only happen when this is a global parameter in a local model
                    # data has shape (ngroups,) when it should have shape (tot_nleaves,)
                    hp_vals = hp_vals[branch_groups] # repeats the global parameters correct # of times
                model_hp_vals_dict[hp] = hp_vals[..., None] # reshapes to be broadcast with samples, which will be flattened
            
            # compute dNdtheta per leaf for each of the local models
            dNdtheta_per_leaf = dNdtheta_per_leaf * model_func(data[param].flatten(), **model_hp_vals_dict)
            dNdtheta_per_leaf_injs = dNdtheta_per_leaf_injs * model_func(injections[param], **model_hp_vals_dict)
        
        # combine leaves, add to dNdtheta_per_group
        dNdtheta_per_group += scatter_add(
            xp.zeros((ngroups, nevs, nsamples), dtype=xp.float64),
            branch_groups, # (tot_nleaves,)
            dNdtheta_per_leaf, # (tot_nleaves, nevs x nsamples)
        ) # shape (ngroups, nevs x nsamples)
        dNdtheta_per_group_injs += scatter_add(
            xp.zeros((ngroups, nevs), dtype=xp.float64),
            branch_groups, # (tot_nleaves,)
            dNdtheta_per_leaf_injs, # (tot_nleaves, ninjs)
        ) # shape (ngroups, ninjs)
        
    # multiply this by global dP/dtheta for global theta
    for global_param_name, global_param_dict in HP_ORDERING[-1].items():
        
        model_func = MODELS[global_param_name][global_param_dict['__model__']]['model']

        model_hp_names = global_param_dict['model_hp_names']
        model_hp_branch_idxs = global_param_dict['branch_idx']
        model_hp_col_idxs = global_param_dict['col_idx']

        # model_hp_vals_dict will be unpacked into the model function
        model_hp_vals_dict = dict(MODEL_METAINFO[global_param_name]['params_fix']) # start with fixed hyperparameters
        
        # now get the hyperparameter values from this branch or the global branch
        for hp, hp_branch_idx, hp_col_idx in zip(model_hp_names, model_hp_branch_idxs, model_hp_col_idxs):
            model_hp_vals_dict[hp] = hyperparams[hp_branch_idx][:, hp_col_idx][..., None]
        
        dNdtheta_per_group *= model_func(data[global_param_name].flatten(), **model_hp_vals_dict)
        dNdtheta_per_group_injs *= model_func(injections[global_param_name], **model_hp_vals_dict)

    # divide by prior
    dNdtheta_per_group /= data['priors'] 
    dNdtheta_per_group_injs /= injections['priors']

    # reshape dNdtheta on data back into nevs, nsamples
    dNdtheta_per_group = dNdtheta_per_group.reshape((ngroups, nevs, nsamples))

    # sum the samples, multiply event-level likelihoods together
    logl = xp.sum(xp.log(xp.sum(dNdtheta_per_group, axis=-1)), axis=1) # shape (ngroups,)

    # Poisson detection probability, evaluated on injections
    # this multiplies the likelihood by exp(-xi(Lambda)N(Lambda)), i.e. subtracts Nexp(Lambda) = xi(Lambda)N(Lambda) from log-likelihood
    
    # Importance sampling variance check for selection function - need dPdtheta_per_inj for this
    ampls_per_group = scatter_add(
        xp.zeros((ngroups,), dtype=xp.float64),
        groups[R0_BRANCH_IDX],
        hyperparams[R0_BRANCH_IDX][:, R0_COL_IDX]
    )
    dPdtheta_per_group_injs = dNdtheta_per_group_injs / ampls_per_group[:, None]
    mu_selection = (1. / injections['ninjs']) * xp.sum(dPdtheta_per_group_injs, axis=1) 
    var_selection = (1. / injections['ninjs'] ** 2) * xp.sum(dPdtheta_per_group_injs** 2, axis=1) \
                     - mu_selection ** 2 / injections['ninjs'] # eq A5 of O4 populations paper https://arxiv.org/pdf/2508.18083
    logl_var = nevs**2 * var_selection # eq A3 is in terms of the pdet estimator, not nexp

    selection_func_term = xp.where(
        logl_var > 1,
        -pdfs.INF,
        -mu_selection * ampls_per_group
    )
    logl += selection_func_term

    return logl # shape (ngroups,)

# -----------------------------
# LOAD IN DATA
# -----------------------------
# TODO

# -----------------------------
# SET UP RUN
# -----------------------------

betas = np.linspace(1.0, 0.0, args.ntemps)
        
# Initialize parameter chains (walker positions)
# coords is a dict keyed by branch name each of shape:
#   (ntemps, nwalkers, nleaves_max, ndims) drawn from priors.
coords = {
    branch_name: priors[branch_name].rvs(size=(args.ntemps, args.nwalkers, args.ncomp_max, BRANCH_DIMS[branch_idx]))
    for branch_idx, branch_name in enumerate(BRANCH_NAMES)
}

# Build the initial sampler state. `inds` is optional and not defined here;
# we rely on `provide_groups=True` in EnsembleSampler to manage groups.
state = State(coords)

# Define MCMC moves to use (Gibbs/Stretch/Random-walk)
# For the Gaussian branch, we optionally use per-component Gibbs updates with
# a diagonal Gaussian proposal scaled by `factors` (one scale per parameter):
# factors = np.array([0.005, 0.1, 0.05, 0.05])  # Step sizes for Gaussian moves (len == dict_dims['gauss'])
factor = 0.05
moves = []  # Will add Gaussian moves for each component

for branch_idx, branch_name in enumerate(BRANCH_NAMES[:-1]):
    branch_ndims = BRANCH_DIMS[branch_idx]
    
    for i in range(args.ncomp_max):
        # Change parameters of a single Gaussian at a time (Gibbs)
        # `cov` is diagonal and broadcast-multiplied by `factors` so each param
        # has its own step scale. Requires len(factors) == dict_dims['gauss'].
        cov = {branch_name: np.diag(np.ones(branch_ndims)) * factor}
        gibbs_array_i = np.zeros((args.ncomp_max, branch_ndims), dtype='bool')
        gibbs_array_i[i, :] = True
        gaussian_move_i = GaussianMove(cov, gibbs_sampling_setup=(branch_name, gibbs_array_i))
        moves += [(gaussian_move_i, 0.2)]

# RJMCMC moves to change model dimensionality (add/remove leaves)
rj_moves = []
nleaves_min_dict, nleaves_max_dict = {}, {}
for branch_idx, branch_name in enumerate(BRANCH_NAMES[:-1]):
    nleaves_min_dict[branch_name] = 1 if branch_idx == 0 else 0
    nleaves_max_dict[branch_name] = args.ncomp_max
nleaves_min_dict['global'], nleaves_max_dict['global'] = 1, 1

prior_move_gauss = DistributionGenerateRJ(
    priors, nleaves_min=nleaves_min_dict, nleaves_max=nleaves_max_dict,
    gibbs_sampling_setup=BRANCH_NAMES[0]
)
rj_moves = [(prior_move_gauss, 0.5)]

# -----------------------------
# RUN SAMPLER, SAVE OUTPUT
# -----------------------------
ensemble = EnsembleSampler(
    args.nwalkers,
    BRANCH_DIMS,  # assumes ndim_max
    loglike,
    priors,
    tempering_kwargs=dict(betas=betas),
    nbranches=len(BRANCH_NAMES),
    branch_names=BRANCH_NAMES,
    nleaves_max=args.ncomp_max,
    nleaves_min=0,    # if some branches are 0 and some branches are 1, what do I put???
    provide_groups=True,
    moves=moves,
    vectorize=True,
    rj_moves=rj_moves  # basic generation of new leaves from the prior
)

# Run the MCMC
ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)