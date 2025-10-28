import os
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import GaussianMove, StretchMove, DistributionGenerateRJ
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform

import pdfs as pdfs

from astropy.cosmology import Planck15

try:
    import cupy as xp
    # xp.cuda.runtime.setDevice(0)
    from cupyx import scatter_add

except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    print(f'Using cpu: {e}')
    def scatter_add(a, slices, value):
        xp.add.at(a, slices, value)


# INPUT ARGUMENTS

parser = argparse.ArgumentParser()
# General
parser.add_argument('-s', '--seed', '-seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--test', '-test', action='store_true', help='Run in test mode (fewer steps)')
parser.add_argument('--label', '-label', default='out', type=str, help="Label for output files. Default: 'out'")

# Input files
parser.add_argument('--prior', '-prior', type=str, help='Prior json file')
parser.add_argument('--PE_samples', type=str, help='PE samples npz file')
parser.add_argument('--injs', '-injs', type=str, help='Injections npz file')

#runtime arguments
parser.add_argument('--nwalkers', '-nwalkers', type=int, default=80, help='Number of walkers per temperature')
parser.add_argument('--ntemps', '-ntemps', type=int, default=10, help='Number of temperatures (for parallel tempering)')
parser.add_argument('--Tmax', default=None, type=float, help='Maximum temperature (for parallel tempering)')
parser.add_argument('--nsteps', '-nsteps', type=int, default=2000, help='Number of steps')
parser.add_argument('--burn', '-burn', type=int, default=0, help='Number of burn-in steps. Default: 0 (discard burn-in in post)')
parser.add_argument('--thin_by', '-thin_by', type=int, default=1, help='Thinning factor')
parser.add_argument('--outdir', '-outdir', type=str, help='Output directory')

# Hyperparameters
parser.add_argument('--use_mchirp', action='store_true', help='Use chirp mass instead of m1')

args = parser.parse_args()
xp.random.seed(args.seed)

os.makedirs(args.outdir, exist_ok=True)

# save settings
with open(os.path.join(args.outdir, 'settings.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# dictionary of models and their hyperparameters -- these are manually hard coded in 
MODELS = {
    'mass': {
        'skew-t': {
            'model': pdfs.jf_skew_t(),
            'param_latex': {
                'a': r'$\alpha_1$',
                'b': r'$\alpha_2$',
                'loc': r'$\mu_m$',
                'scale': r'$\sigma_m$',
                'xmin': r'$m_{\min}$',
                'xmax': r'$m_{\max}$',
            },
            'params_fix': {}
        },
        'PLS': {
            'model': pdfs.smoothed_powerlaw(),
            'param_latex': {
                'alpha': r'$\alpha$',
                'p': r'$p_m$',
                'xmin': r'$m_{\min}$',
            },
            'params_fix': {'xmax': 300.0}
        },
        'PLS_LVK': {
            'model': pdfs.LVK_Plancktaper_powerlaw(),
            'param_latex': {
                'alpha': r'$\alpha$',
                'delta': r'$\delta_m$',
                'xmin': r'$m_{\min}$',
            },
            'params_fix': {'xmax': 300.0}
        },
        'gauss': {
            'model': pdfs.gaussian(),
            'param_latex': {
                'loc': r'$\mu_p$',
                'scale': r'$\sigma_p$',
            },
            'params_fix': {'xmax': 300.0}
        }
    },
    'mass_ratio': {
        'PL': {
            'model': pdfs.powerlaw(),
            'param_latex': {
                'beta': r'$\beta$',
            },
            'params_fix': {
                'xmax': 1.0,
            }
        }
    },
    'chi_eff': {
        'gen_gauss': {
            'model': pdfs.gen_gaussian(),
            'param_latex': {
                'beta': r'$\beta_{\chi}$',
                'loc': r'$\mu_{\chi}$',
                'scale': r'$\sigma_{\chi}$',
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
    'redshift': {
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
                'gamma': r'$\gamma$'
            },
            'params_fix': {}
        }
    }
}

# load in and setup prior
with open(args.prior, 'r') as f: 
    # this should be a list of dictionaries, one for each branch, the last of which should be global
    # each dictionary should have keys corresponding to each model (mass, q, chieff, rate), with the value as a subdictionary
    # each dictionary should also have a "__branch__" key corresponding to the branch name
    # each subdictionary should have the prior ranges for each parameter as a list or a float if parameter should be fixed,
    # as well as a '__model__' key corresponding to the model name
    # parameter names must match the names defined in the functions (in pdfs.py)
    branch_priordicts = json.load(f)
    branch_names = [d['__branch__'] for d in branch_priordicts]

priors = {} # {branch_name: ProbDistContainer({i: prior})}

hp_ordering = [{} for _ in branch_names]
ordered_latex_hp_names = []
branch_dims = []
nleaves_min_dict, nleaves_max_dict = {}, {}

factors = [] # covariance matrices for gibbs sampling of the leaf
factor_default = 0.005

# construct list of dictionaries `hp_ordering` with the primary models belonging to each branch - this will be used in logl function
# note that models whose primary model are local could also have global parameters (hence branch_id list)
# [ {param: {  __model___: name of model (e.g. skew_t, gaussian),
#          model_hp_names: [hp_name1, hp_name2, ...], 
#          latex_hp_names: [hp_latex1, hp_latex2, ...],
#              branch_idx: [branch_idx1, branch_idx2, ...]       i.e. which branch the hyperparameter belongs to (should be either the list index or -1 for global)
#                 col_idx: (param_idx1, param_idx2) }]            i.e. which column in data array in the branch the hyperparameter is found
for branch_idx, input_dict in enumerate(branch_priordicts):
    
    branch_name = input_dict.pop('__branch__')
    nleaves_min_dict[branch_name], nleaves_max_dict[branch_name] = input_dict.pop('__ncomp__', [1,1])
    factors.append(np.asarray(input_dict.pop('__factor__', factor_default)))

    branch_priordict = {}
    hp_idx = 0

    ordered_latex_hp_names.append([])

    for param, param_priordict in input_dict.items():
        if param.startswith('__') or param.endswith('__'): # ignore special / metainfo keys
            continue

        model_name = param_priordict.pop('__model__')
        
        hp_ordering[branch_idx][param] = {
            '__model__': model_name,
            'model_hp_names': [], 
            'latex_hp_names': [], 
            'branch_idx': [], 
            'col_idx': []
        }
        
        for hp, hp_range in param_priordict.items():
            
            if isinstance(hp_range, list) or isinstance(hp_range, tuple):
                
                branch_priordict[hp_idx] = uniform_dist(hp_range[0], hp_range[1])
                
                hp_ordering[branch_idx][param]['model_hp_names'].append(hp)
                latex_name = MODELS[param][model_name]['param_latex'].get(hp, hp)
                ordered_latex_hp_names[-1].append(latex_name)
                hp_ordering[branch_idx][param]['latex_hp_names'].append(latex_name)

                hp_ordering[branch_idx][param]['branch_idx'].append(branch_idx)
                hp_ordering[branch_idx][param]['col_idx'].append(hp_idx)

                hp_idx += 1

            elif isinstance(hp_range, float) or isinstance(hp_range, int):
                MODELS[param][model_name]['params_fix'][hp] = hp_range
            else:
                raise ValueError(f"Invalid prior range for {param} {hp}: {hp_range}")
    
    if branch_name == 'global':
        # add global mmin parameter with prior [3, 6]
        branch_priordict[hp_idx] = uniform_dist(3, 6)
        ordered_latex_hp_names[-1].append(r'$m_{\min}^{\text{global}}$')
        hp_idx += 1  
    else:
        # add R0 for each local branch
        branch_priordict[hp_idx] = uniform_dist(5., 1e3) # rate prior, in 1 / (Gpc^3 yr)
        ordered_latex_hp_names[-1].append('$R_0$')
        hp_idx += 1
    
    # each branch will have the number of parameters specified in the prior file + 1
    # this is R0 for local branches and a global mmin
        
    priors[branch_name] = ProbDistContainer(dict(branch_priordict))
    branch_dims.append(hp_idx)

# save hp_ordering, latex parameter names, and nleaves_min_max of each branch
with open(os.path.join(args.outdir, 'hyperparameter_ordering_metainfo.json'), 'w') as f:
    json.dump(hp_ordering, f, indent=4)
with open(os.path.join(args.outdir, 'ordered_hyperparameter_latex.json'), 'w') as f:
    json.dump(ordered_latex_hp_names, f, indent=4)
with open(os.path.join(args.outdir, 'nleaves_min_max.json'), 'w') as f:
    json.dump([
        nleaves_min_dict,
        nleaves_max_dict
    ], f, indent=4)

# ----------------------------------------
# LOAD IN DATA - PE samples and injections
# ----------------------------------------

mass_key = 'mass_1_source'
if args.use_mchirp:
    mass_key = 'chirp_mass_source'
params = list(MODELS.keys())
params.remove('mass')
params.append(mass_key)
params.append('prior')

with np.load(args.PE_samples) as f:
    PE_samples = {k : xp.asarray(v) for k, v in f.items() if k in params} # needed for initialization
    PE_samples['mass'] = PE_samples.pop(mass_key)

    # med_mmin_data = min(
    #     np.amin(np.median(f['mass_1_source'][()], axis=1)),
    #     np.amin(np.median(f['mass_2_source'][()], axis=1))
    # ) # for initializing the sampler - currently not used

with np.load(args.injs) as f:
    injections = {k : xp.asarray(v) for k, v in f.items() if k in params + ['w']}
    injections['total_generated'] = int(f['total_generated'])
    injections['mass'] = injections.pop(mass_key)
    TOBS = float(f['Tobs_yr'])

# ------------------------
# LOG-LIKELIHOOD FUNCTION
# ------------------------

def _eval_logl_on_data(param_name, model_func, model_hp_vals_dict, mmin_glob_grouped, data_flat):
    # this is just to make the code a bit cleaner since we evalute this for both PE samples and injections
    if param_name == 'mass_ratio':
        if args.use_mchirp:
            pass # do later - conditional chirp mass distribution
        else:
            model_hp_vals_dict['xmin'] = mmin_glob_grouped / data_flat['mass']
        
    # compute dNdtheta per leaf for each of the local models
    model_pdf_vals = model_func(data_flat[param_name], **model_hp_vals_dict)
    return model_pdf_vals

ZZ_INT = xp.linspace(0, 3, 1000) 
def loglike(np_hyperparams, groups, data, injections):
    """Compute the per-group log-likelihood for the hierarchical population model.

    Implements the hierarchical likelihood with selection effects (see Eq. 5 of
    https://arxiv.org/pdf/2305.08909), evaluated on PE samples (for events) and
    injections (for the selection function). Supports NumPy or CuPy backends and
    always returns a NumPy array.

    Parameters
    ----------
    np_hyperparams : list[array-like]
        List of hyperparameter arrays for each branch in `BRANCH_NAMES` order.
        For local branches: arrays are shape (tot_nleaves, ndims). For the global
        branch: arrays are shape (ngroups, ndims). 
    groups : list[xp.ndarray]
        For each branch, a 1D integer array of shape (tot_nleaves,) mapping
        each leaf to a group id in [0, ngroups-1]. The global branch entry is
        typically `xp.arange(ngroups)`. Each group id maps to a different 
        (temperature, walker) combination.
    data : dict[str, xp.ndarray]
        PE samples for event-level parameters. Expected keys include
        `'mass'`, `'mass_ratio'`, `'chi_eff'`, `'redshift'`, and `'prior'`.
        Each parameter array has shape (nevs, nsamples). `'prior'` matches the
        same shape and is used to divide out the event-level prior.
    injections : dict
        Injection campaign information used to compute selection effects.
        Contains parameter arrays (shape (ninjs,)), `'prior'` (shape (ninjs,)),
        weights `'w'` (shape (nfound,)), and metadata like `'total_generated'`.
        Also uses module-level `TOBS` for the observing time.

    Returns
    -------
    numpy.ndarray
        Array of shape (ngroups,) with the log-likelihood value for each group.

    Notes
    -----
    - Enforces a global minimum mass `mmin_glob` in mass-related terms and uses
      conditional modeling for mass ratio when applicable.
    - Combines per-leaf contributions within a branch via scatter-add into
      per-group totals and multiplies across parameters, then across events.
    - Selection effects are accounted for using injections to compute the
      detection efficiency and a Poisson term; numerical stability guards are
      applied and the final result is clipped to avoid NaNs/inf.
    """

    # tot_nleaves x (ampl, m1 hp, q hp, chieff hp, ...), nevs x nsamples -> ngroups x nevs x nsamples 
    # -> ngroups x nevs (sum) -> ngroups (product)

    hyperparams = [xp.array(hp, copy=True) for hp in np_hyperparams]

    ngroups = groups[-1].shape[0] # since global groups is just range(ngroups)
    nevs, nsamples = data['mass_ratio'].shape
    ninjs = injections['mass_ratio'].shape[0] 

    data_flat = {k : data[k].flatten() for k in data}

    ampls_per_group = xp.zeros((ngroups,), dtype=xp.float64)
    dNdtheta_per_group = xp.zeros((ngroups, nevs * nsamples), dtype=xp.float64) # evaluated on PE samples
    dNdtheta_per_group_injs = xp.zeros((ngroups, ninjs), dtype=xp.float64) # evaluated on injections

    # get mmin for correct conditional mass ratio distribution
    mmin_glob = hyperparams[-1][:, -1]

    # iterate over branches -- different local branches will have different models
    for branch_idx, branch_dict in enumerate(hp_ordering[:-1]): # all local branches

        # print(f'branch: {branch_names[branch_idx]}')

        branch_groups = groups[branch_idx]
        branch_nleaves = hyperparams[branch_idx].shape[0]

        N0 = TOBS * hyperparams[branch_idx][:, -1] # Tobs * R0

        dNdtheta_per_leaf = N0[:, None] * xp.ones((branch_nleaves, 1), dtype=xp.float64) # evaluated on PE samples
        dNdtheta_per_leaf_injs = N0[:, None] * xp.ones((branch_nleaves, 1), dtype=xp.float64) # evaluated on injections

        mmin_glob_grouped = mmin_glob[branch_groups][..., None]

        # evaluate likelihood for each event-level parameter
        for param_name, param_dict in branch_dict.items(): 

            model_name = param_dict['__model__']
            model_func = MODELS[param_name][model_name]['model'].pdf

            model_hp_names = param_dict['model_hp_names']
            model_hp_branch_idxs = param_dict['branch_idx']
            model_hp_col_idxs = param_dict['col_idx']

            # model_hp_vals_dict will be unpacked into the model function
            model_hp_vals_dict = dict(MODELS[param_name][model_name]['params_fix']) # start with fixed hyperparameters
            
            # now get the hyperparameter values from this branch or the global branch
            for hp, hp_branch_idx, hp_col_idx in zip(model_hp_names, model_hp_branch_idxs, model_hp_col_idxs):

                hp_vals = hyperparams[hp_branch_idx][:, hp_col_idx]

                if hp_branch_idx != branch_idx: 
                    # should only happen when this is a global parameter in a local model
                    # hyperparams then has shape (ngroups,) when it should have shape (tot_nleaves,)
                    hp_vals = hp_vals[branch_groups] # repeats the global parameters correct # of times
                model_hp_vals_dict[hp] = hp_vals[..., None] # reshapes to be broadcast with samples, which will be flattened
            
            if param_name == 'mass':
                # enforce global mmin
                if 'xmin' in model_hp_vals_dict:
                    # dNdtheta_per_leaf = dNdtheta_per_leaf * (model_hp_vals_dict['xmin'] >= mmin_glob_grouped) 
                    pass
                else:
                    model_hp_vals_dict['xmin'] = mmin_glob_grouped 

            # compute dNdtheta per leaf for each of the local models
            dNdtheta_per_leaf = dNdtheta_per_leaf * _eval_logl_on_data(param_name, model_func, model_hp_vals_dict, mmin_glob_grouped, data_flat)
            dNdtheta_per_leaf_injs = dNdtheta_per_leaf_injs * _eval_logl_on_data(param_name, model_func, model_hp_vals_dict, mmin_glob_grouped, injections)

            # print(f'fraction of leaves 0 after {param_name} (data)', xp.sum(dNdtheta_per_leaf == 0) / dNdtheta_per_leaf.size)
            # print(f'fraction of leaves 0 after {param_name} (injs)', xp.sum(dNdtheta_per_leaf_injs == 0) / dNdtheta_per_leaf_injs.size)
        
        # combine leaves, add to dNdtheta_per_group
        scatter_add(
            dNdtheta_per_group,
            branch_groups, # (tot_nleaves,)
            dNdtheta_per_leaf, # (tot_nleaves, nevs x nsamples)
        ) # shape (ngroups, nevs x nsamples)
        scatter_add(
            dNdtheta_per_group_injs,
            branch_groups, # (tot_nleaves,)
            dNdtheta_per_leaf_injs, # (tot_nleaves, ninjs)
        ) # shape (ngroups, ninjs)
        scatter_add(
            ampls_per_group,
            branch_groups, # (tot_nleaves,)
            N0, # (tot_nleaves,)
        ) # shape (ngroups,)

        # print(f'fraction of groups 0 after {branch_names[branch_idx]} (data)', xp.sum(dNdtheta_per_group == 0) / dNdtheta_per_group.size)
        # print(f'fraction of groups 0 after {branch_names[branch_idx]} (injs)', xp.sum(dNdtheta_per_group_injs == 0) / dNdtheta_per_group_injs.size)
        
    # multiply this by global dP/dtheta for global theta
    for global_param_name, global_param_dict in hp_ordering[-1].items():

        model_name = global_param_dict['__model__']
        model_func = MODELS[global_param_name][model_name]['model'].pdf

        model_hp_names = global_param_dict['model_hp_names']
        model_hp_branch_idxs = global_param_dict['branch_idx']
        model_hp_col_idxs = global_param_dict['col_idx']

        # model_hp_vals_dict will be unpacked into the model function
        model_hp_vals_dict = dict(MODELS[global_param_name][model_name]['params_fix']) # start with fixed hyperparameters
        
        # now get the hyperparameter values from this branch or the global branch
        for hp, hp_branch_idx, hp_col_idx in zip(model_hp_names, model_hp_branch_idxs, model_hp_col_idxs):
            model_hp_vals_dict[hp] = hyperparams[hp_branch_idx][:, hp_col_idx][..., None]
        
        dNdtheta_per_group *= model_func(data_flat[global_param_name], **model_hp_vals_dict)
        dNdtheta_per_group_injs *= model_func(injections[global_param_name], **model_hp_vals_dict)

        if global_param_name == 'redshift':
            # need to calculate total four-volume for normalization
            z_norm = xp.trapz(
                model_func(ZZ_INT, **model_hp_vals_dict),
                ZZ_INT, axis=-1
            )
            N_TOT = ampls_per_group * z_norm
            dPdtheta_per_group_injs = dNdtheta_per_group_injs / N_TOT[:, None]

        # this assumes that m1/m2 are not in global branch

        # print(f'fraction of groups 0 after param {global_param_name} (data)', xp.sum(dNdtheta_per_group == 0) / dNdtheta_per_group.size)
        # print(f'fraction of groups 0 after param {global_param_name} (injs)', xp.sum(dNdtheta_per_group_injs == 0) / dNdtheta_per_group_injs.size)

    # divide by prior
    dNdtheta_per_group /= data_flat['prior'] 
    # reshape dNdtheta on data back into nevs, nsamples
    dNdtheta_per_group = dNdtheta_per_group.reshape((ngroups, nevs, nsamples))

    # sum the samples, multiply event-level likelihoods together
    logl = xp.sum(xp.log(xp.sum(dNdtheta_per_group, axis=-1)), axis=1) # shape (ngroups,)

    # Poisson detection probability, evaluated on injections
    # this multiplies the likelihood by exp(-xi(Lambda)N(Lambda)), i.e. subtracts Nexp(Lambda) = xi(Lambda)N(Lambda) from log-likelihood
    
    # Importance sampling variance check for selection function - need dPdtheta_per_inj for this
    ninjs, nfound = injections['total_generated'], injections['w'].shape[0]
    mu_selection = (1. / ninjs) * xp.sum(injections['w'] * dPdtheta_per_group_injs / injections['prior'], axis=-1) 
    var_selection = (1. / nfound ** 2) * xp.sum(
        injections['w'] * (dPdtheta_per_group_injs / injections['prior'] * nfound / ninjs - mu_selection[:, None])** 2, axis=-1
    )  # eq 12 of Essick 2021, https://iopscience.iop.org/article/10.3847/2515-5172/ac2ba7

    Neff = mu_selection**2 / var_selection # threshold on Neff > 4 * Nevs
    selection_func_term = -mu_selection * N_TOT - xp.log1p((Neff / (4 * nevs))**(-50)) # eq B34 of https://arxiv.org/pdf/2302.07289

    # print('Frac NaN pre-selection:', xp.sum(logl <= -pdfs.INF) / logl.size)
    
    logl += selection_func_term

    # print('Frac NaN post-selection:', xp.sum(logl <= -pdfs.INF) / logl.size)
    
    # print('Frac excluded from logl var:', xp.sum(logl_var > 1) / logl_var.size)
    
    # mmin_diff = hyperparams[0][:,2] - mmin_glob # should be a positive number
    # print('Bad mmin diff?', mmin_diff[logl <= -pdfs.INF])

    logl = xp.clip(logl, -pdfs.INF, None) # avoid NaNs

    return logl.get() # output needs to be a numpy array

# -----------------------------
# TEST LIKELIHOOD FUNCTION
# -----------------------------
# maybe test loglike function above and see if it actually reprodjces P(m1) as expected
test_data = {

}

if args.test:
    ngroups_test = 5
    test_hp_vals = [priors[branch_name].rvs(size=(ngroups_test*nleaves_max_dict[branch_name],)) for branch_name in branch_names]

    test_groups = [xp.repeat(xp.arange(ngroups_test), nleaves_max_dict[branch_name]) for branch_name in branch_names]
    PE_samples_downsampled = {k: v[:, :100] for k, v in PE_samples.items()}
    # injs_downsampled = {k: v[:200] if isinstance(v, xp.ndarray) else v for k, v in injections.items()}
    ll = loglike(test_hp_vals, test_groups, PE_samples_downsampled, injections)

    print('test logl', ll)
    if np.any(~np.isfinite(ll)):
        raise ValueError('test logl has non-finite values')

# -----------------------------
# SET UP RUN
# -----------------------------
        
# Initialize parameter chains (walker positions)
# coords is a dict keyed by branch name each of shape:
#   (ntemps, nwalkers, nleaves_max, ndims) drawn from priors.
coords = {
    branch_name: priors[branch_name].rvs(size=(args.ntemps, args.nwalkers, nleaves_max_dict[branch_name],))
    for branch_name in branch_names
}
# # need one component to cover low masses so that initial loglike is not -inf
# coords[branch_names[MMIN_BRANCH]][..., 0, MMIN_COL_IDX] = 0.9*med_mmin_data

# Build the initial sampler state. `inds` is optional and not defined here;
# we rely on `provide_groups=True` in EnsembleSampler to manage groups.
state = State(coords)

# Define MCMC moves to use (Gibbs/Stretch/Random-walk)
# For the Gaussian branch, we optionally use per-component Gibbs updates with
# a diagonal Gaussian proposal scaled by `factors` (one scale per parameter), which is specified in the prior
moves = []  # Will add Gaussian moves for each variable-leaf branch
rj_moves = [] # RJMCMC moves to change model dimensionality (add/remove leaves)
for branch_idx, branch_name in enumerate(branch_names):

    branch_ndims = branch_dims[branch_idx]
    ncomp_max = nleaves_max_dict[branch_name]
    factor = factors[branch_idx]

    if ncomp_max == 1:
        stretch_move_i = StretchMove(gibbs_sampling_setup=branch_name, live_dangerously=True)
        moves.append((stretch_move_i, 0.1))
    
    else:
    
        for i in range(ncomp_max):
            # Change parameters of a single leaf at a time (Gibbs)
            # `cov` is diagonal and broadcast-multiplied by `factors` so each param
            # has its own step scale. Requires len(factors) == dict_dims['gauss'].
            cov = {branch_name: np.diag(np.ones(branch_ndims)) * factor}
            gibbs_array_i = np.zeros((ncomp_max, branch_ndims), dtype='bool')
            gibbs_array_i[i, :] = True # change one leaf at a time
            gaussian_move_i = GaussianMove(cov, gibbs_sampling_setup=(branch_name, gibbs_array_i)) # UPDATE CODE SO THAT IT DOES NOT GAUSSIANMOVE EMPTY LEAVES
            moves.append((gaussian_move_i, 0.2)) # '0.2' is the relative weight of which proposal to do
        
        rj_move = DistributionGenerateRJ(
            priors, nleaves_min=nleaves_min_dict, nleaves_max=nleaves_max_dict,
            gibbs_sampling_setup=branch_name # change one branch at a time
        )
        rj_moves.append((rj_move, 0.5))

if not rj_moves:
    rj_moves = False

# -----------------------------
# RUN SAMPLER, SAVE OUTPUT
# -----------------------------
ensemble = EnsembleSampler(
    args.nwalkers,
    branch_dims, 
    loglike,
    priors,
    args=[PE_samples, injections],
    tempering_kwargs=dict(ntemps=args.ntemps, Tmax=args.Tmax),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max_dict,
    nleaves_min=nleaves_min_dict,
    provide_groups=True,
    moves=moves,
    vectorize=True,
    rj_moves=rj_moves, 
    fill_zero_leaves_val=-pdfs.INF,
    backend=os.path.join(args.outdir, f'backend_{args.label}{"_test" if args.test else ""}.h5') # use backend for testing/tuning/debugging purposes
)

# Run the MCMC
nsteps = 20 if args.test else args.nsteps
burn = 5 if args.test else args.burn

ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=args.thin_by)

# # Extract and reshape samples for each branch

# samples_dict = {}
# for branch_idx, branch_name in enumerate(branch_names):
#     samples_dict[branch_name] = ensemble.get_chain()[branch_name][:,0]
# for branch_idx, branch_name in enumerate(branch_names[:-1]): # all except global
#     samples_dict[branch_name+'_nleaves'] = ensemble.get_nleaves()[branch_name][:,0]
# samples_dict['logl'] = ensemble.get_log_like()[:, 0]
# print(samples_dict['logl'].shape)

# print('maximum logl:', np.amax(samples_dict['logl']))

# # Save samples and log-likelihoods to .npz file
# np.savez(
#     os.path.join(args.outdir, 'samples.npz'),
#     **samples_dict,
# )