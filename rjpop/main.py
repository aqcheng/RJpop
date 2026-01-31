import os
import numpy as np
import argparse
import json
from xp import xp, scatter_add, use_cupy

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.moves import GaussianMove, StretchMove, DistributionGenerateRJ, MTDistGenMoveRJ
from eryn.prior import ProbDistContainer, uniform_dist, log_uniform
from eryn.backends import HDFBackend

from pdfs import INF, MassModel

# packages for post-processing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils
from utils import corner_defaults
plt.style.use(utils.plot_style)

import seaborn as sns
from corner import corner
from glob import glob

from sklearn.cluster import KMeans

# INPUT ARGUMENTS

parser = argparse.ArgumentParser()
# General
parser.add_argument('-s', '--seed', '-seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--test', '-test', action='store_true', help='Run in test mode (fewer steps)')
parser.add_argument('--label', '-label', default='out', type=str, help="Label for output subdirectory. Default: 'out'")

# Input files
parser.add_argument('--prior', '-prior', type=str, help='Prior json file')
parser.add_argument('--PE_samples', type=str, help='PE samples npz file')
parser.add_argument('--injs', '-injs', type=str, help='Injections npz file')

#runtime arguments
parser.add_argument('--nwalkers', '-nwalkers', type=int, default=80, help='Number of walkers per temperature')
parser.add_argument('--ntemps', '-ntemps', type=int, default=10, help='Number of temperatures (for parallel tempering)')
parser.add_argument('--Tmax', default=None, type=float, help='Maximum temperature (for parallel tempering)')
parser.add_argument('--rj_num_try', default=1, type=int, help='Number of tries for MT MCMC')
parser.add_argument('--nsteps', '-nsteps', type=int, default=2000, help='Number of steps')
parser.add_argument('--burn', '-burn', type=int, default=0, help='Number of burn-in steps.')
parser.add_argument('--discard', '-discard', type=int, default=None, help='Number of steps to discard in post. Default: None (determine automatically)')
parser.add_argument('--outdir', '-outdir', type=str, help='Output directory')

# Hyperparameters
parser.add_argument('--use_mchirp', action='store_true', help='Use chirp mass instead of m1 - not currently implemented')
parser.add_argument('--min_sep', '--minsep', type=float, default=3.0, help='Enforce a minimum separation in feature space between components. Default: 3')

# plotting options
parser.add_argument('--LVK_plot', type=str, choices=['default', 'spline', 'none'], default='default', help='Which LVK results to plot as reference, if any')
parser.add_argument('--skip_corner', action='store_true', help='Skip the corner plots (speeds up post-processing for debugging or reruns)')
parser.add_argument('--replot', action='store_true', help='(Re)plot all the ppds, even if the plots already exist')

args = parser.parse_args()
print(args.__dict__)

xp.random.seed(args.seed)

outdir = os.path.join(args.outdir, args.label)
figpath = os.path.join(outdir, 'figures')
datapath = os.path.join(outdir, 'data')
if args.test:
    outdir += '_test'
else:
    os.makedirs(figpath, exist_ok=True)
    os.makedirs(datapath, exist_ok=True)

# -----------------------------
# Prior parsing / sampler setup
# -----------------------------

# load in and setup prior
with open(args.prior, 'r') as f: 
    # this should be a list of dictionaries, one for each branch, the last of which should be global
    # each dictionary should have keys corresponding to each model (mass, q, chieff, rate), with the value as a subdictionary
    # each dictionary should also have a "__branch__" key corresponding to the branch name
    # each subdictionary should have the prior ranges for each parameter as a list or a float if parameter should be fixed,
    # as well as a '__model__' key corresponding to the model name
    # parameter names must match the names defined in the functions (in pdfs.py)
    input_priordicts = json.load(f)
    branch_names = [d['__branch__'] for d in input_priordicts]

    # save prior in output directory
    if not args.test:
        with open(os.path.join(outdir, 'prior.json'), 'w') as f:
            json.dump(input_priordicts, f, indent=4)

## globals, to be updated by input prior file
priors = {} # {branch_name: ProbDistContainer({i: prior})}
hp_ordering = [{} for _ in branch_names]
branch_priordicts = [{} for _ in branch_names] 
params = set()
branch_dims = []
nleaves_min_dict, nleaves_max_dict = {}, {}

factors = [] # covariance matrices for gibbs sampling of the leaf
factor_default = 0.005

# construct list of dictionaries `hp_ordering` with the primary models belonging to each branch - this will be used in logl function
# note that models whose primary model are local could also have global parameters (hence branch_id list)
# [ {param: {  __model___: name of model (e.g. skew_t, gaussian),
#          model_hp_names: [hp_name1, hp_name2, ...], 
#          latex_hp_names: [hp_latex1, hp_latex2, ...],
#                 col_idx: (param_idx1, param_idx2) }]            i.e. which column in data array in the branch the hyperparameter is found

def _process_param_hp_priors(
    param, input_param_priordict, 
    branch_idx, output_branch_infodict, # outputs
    hp_idx_start=0
):

    if param == 'mass_2_source':
        param_get = 'mass_1_source' # same models
    else:
        param_get = param

    output_branch_infodict[param] = {'model_hp_names': [],  'col_idx': []}
    if '__model__' in input_param_priordict:
        model_name = input_param_priordict.pop('__model__')
        model = utils.MODELS[param_get][model_name]['model']
        hps_fix = utils.MODELS[param_get][model_name].get('params_fix', {}).copy()
        
        output_branch_infodict[param].update({
            'model_name': model_name,
            'model': model,
        })

        if hasattr(model, 'params'):
            params.update(model.params)
        
    else:
        # this should only be if the model was specified in a local branch
        assert branch_idx in [-1, len(branch_names)-1], "param must be global if no model specified"
        assert param in params, "param must be evaluated in a local branch if no model specified"
        hps_fix = {}
    
    hp_idx = hp_idx_start
    
    # update infodict + priordict
    for hp, hp_range in input_param_priordict.items():

        if hp in hps_fix: 
            hps_fix.pop(hp) # overwrite
        
        if isinstance(hp_range, list) or isinstance(hp_range, tuple):

            if len(hp_range) > 2 and hp_range[-1] == 'log':
                branch_priordicts[branch_idx][hp_idx] = log_uniform(hp_range[0], hp_range[1])
            else:
                branch_priordicts[branch_idx][hp_idx] = uniform_dist(hp_range[0], hp_range[1])
            
            # update infodict
            output_branch_infodict[param]['model_hp_names'].append(hp)
            output_branch_infodict[param]['col_idx'].append(hp_idx)

            try:
                hp_ordering[branch_idx]['__latex__'].append(utils.MODELS[param_get][model_name]['param_latex'][hp])
            except:
                hp_ordering[branch_idx]['__latex__'].append(utils.MODELS[param_get]['param_latex'].get(hp, hp))
            params.add(param)

            hp_idx += 1

        elif isinstance(hp_range, float) or isinstance(hp_range, int):
            hps_fix[hp] = hp_range
        
        elif isinstance(hp_range, dict): # hp_range is a subdictionary for another parameter hp
            # the use case for this is for joint (m1, m2) or (m1, q) evaluation
            if param != 'mass' or not hp.startswith('mass'):
                raise ValueError(f"Joint evaluation only currently supported for mass")
            
            if 'subparams' in output_branch_infodict[param]:
                output_branch_infodict[param]['subparams'].append(hp)
            else:
                output_branch_infodict[param]['subparams'] = [hp]

            hp_idx = _process_param_hp_priors( # hierarchical branch infodict
                hp, hp_range, branch_idx, output_branch_infodict=output_branch_infodict[param], hp_idx_start=hp_idx
            )

        else:
            raise ValueError(f"Invalid prior range for {param} {hp}: {hp_range}")

    output_branch_infodict[param]['params_fix'] = hps_fix
    if 'subparams' in output_branch_infodict[param]:
        # instantiate joint model
        submodels = {f'{subparam}_model': output_branch_infodict[param][subparam]['model'] for subparam in output_branch_infodict[param]['subparams']}
        output_branch_infodict[param]['model'] = model(**submodels)
    
    return hp_idx

for branch_idx, input_dict in enumerate(input_priordicts):
    
    branch_name = input_dict.pop('__branch__')
    nleaves_min_dict[branch_name], nleaves_max_dict[branch_name] = input_dict.pop('__ncomp__', [1,1])
    factors.append(np.asarray(input_dict.pop('__factor__', factor_default)))

    hp_idx = 0
    hp_ordering[branch_idx]['__latex__'] = []
    
    for param, input_param_priordict in input_dict.items():
        if param.startswith('__') or param.endswith('__'): # ignore special / metainfo keys
            continue

        hp_idx = _process_param_hp_priors(
            param, input_param_priordict, branch_idx,
            output_branch_infodict=hp_ordering[branch_idx], hp_idx_start=hp_idx
        )

    if branch_name != 'global':
        #  add R0 for each local branch
        branch_priordicts[branch_idx][hp_idx] = uniform_dist(0.05, 100) # rate prior, in 1 / (Gpc^3 yr)
        hp_ordering[branch_idx]['__latex__'].append('$R_0$')
        hp_idx += 1
    
    # if branch_name == 'global':
    #     # add global mmin parameter with prior [3, 7]
    #     branch_priordicts[branch_idx][hp_idx] = uniform_dist(3, 7)
    #     hp_ordering[branch_idx]['latex_names'].append(r'$m_{\min}^{\mathrm{global}}$')
    #     hp_idx += 1  
    # else:
    #     # add R0 for each local branch
    #     branch_priordicts[branch_idx][hp_idx] = uniform_dist(0.05, 100) # rate prior, in 1 / (Gpc^3 yr)
    #     hp_ordering[branch_idx]['latex_names'].append('$R_0$')
    #     hp_idx += 1
    
    # each local branch will have the number of parameters specified in the prior file + 1 (R0)
        
    priors[branch_name] = ProbDistContainer(branch_priordicts[branch_idx], use_cupy=use_cupy)
    branch_dims.append(hp_idx)

# save hp_ordering, latex parameter names, and nleaves_min_max of each branch
if not args.test:
    with open(os.path.join(outdir, 'hyperparameter_ordering_metainfo.json'), 'w') as f:
        json.dump(hp_ordering, f, indent=4, default=lambda x: str(x.__class__))
    with open(os.path.join(outdir, 'nleaves_min_max.json'), 'w') as f:
        json.dump([
            nleaves_min_dict,
            nleaves_max_dict
        ], f, indent=4)
    # print settings
    print(json.dumps(args.__dict__, indent=2))
    # save settings
    with open(os.path.join(outdir, 'settings.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)

# ----------------------------------------
# LOAD IN DATA - PE samples and injections
# ----------------------------------------

params = sorted(list(params)) 
with np.load(args.PE_samples) as f:
    PE_samples = {k : xp.asarray(v, dtype=xp.float32) for k, v in f.items() if k in params + ['prior']} # needed for initialization
with np.load(args.injs) as f:
    injections = {k : xp.asarray(v, dtype=xp.float32) for k, v in f.items() if k in params + ['prior', 'w']}
    injections['total_generated'] = int(f['total_generated'])
    TOBS = xp.asarray(float(f['Tobs_yr']), dtype=xp.float32)

if args.test:
    # downsample injections and PE
    PE_samples = {k: v[:, :500] for k, v in PE_samples.items()}
    injections = {k: v[:500] if isinstance(v, xp.ndarray) else v for k, v in injections.items()}

nevs, nsamples = PE_samples[params[0]].shape
ninjs = injections[params[0]].shape[0]

# ---------------------------
# Likelihood helper functions
# ---------------------------

def _get_parent_param(branch_idx, param):
    for superparam in utils.skip_dunder(hp_ordering[branch_idx].keys()):
        if 'subparams' in hp_ordering[branch_idx][superparam]: # joint model
            if param in hp_ordering[branch_idx][superparam]['subparams']:
                return superparam
    if param.startswith('mass'):
        return 'mass'
    return param

def model_func_wrapper(branch_idx, param, model_hp_vals_dict, data_flat, pad=True):
    """
    Wrapper for the model pdf. Picks the right model pdf from `branch_idx` and
    `param`, and also takes the output of `unpack_hp_vals` as input. Evaluates on
    `data_flat` dictionary of samples to evaluate pdf on.

    If `pad=True`, then pads the input hyperparameters before passing to the model
    function.
    """
    if pad:
        input_hp_vals = utils.recursive_pad(model_hp_vals_dict)
    else:
        input_hp_vals = model_hp_vals_dict

    if 'subparams' in hp_ordering[branch_idx][param]: # joint model
        input_data = data_flat
    else:
        input_data = data_flat[param]
    model = hp_ordering[branch_idx][param]['model']
    return model.pdf(input_data, **input_hp_vals)

def model_moments_wrapper(branch_idx, param, model_hp_vals_dict):
    model = hp_ordering[branch_idx][param]['model']
    moments = model.moments(**model_hp_vals_dict)

    if type(moments) is dict:
        return moments
    else:
       return {param: moments}

def unpack_hp_vals(branch_idx, param, hyperparams, branch_groups=None, param_dict=None):
    """Unpack hyperparameters into a model object + kwargs dict.

    Parameters
    ----------
    branch_idx : int
        Branch index into `hp_ordering` / `branch_names` for the *target* model.
        Use `-1` for the global branch.
    param : str
        Name of the parameter being evaluated (key into `utils.MODELS` and
        `hp_ordering[branch_idx]`).
    hyperparams : list
        List of per-branch hyperparameter arrays. Must have length
        `len(branch_names)`. Each `hyperparams[j]` must have last dimension
        indexing hyperparameter columns, i.e. `hyperparams[j][..., hp_col_idx]`
        must be valid.

        Common shapes used in this script:
        - Likelihood (`loglike`): `hyperparams[j]` has shape
          `(tot_nleaves_j, branch_dims[j])`.
        - Postprocessing / moments: `hyperparams[j]` has shape
          `(ndraws, branch_dims[j])` for the global branch and
          `(ndraws, branch_dims[j])` for a single selected leaf.
        - PPD evaluation: `hyperparams[j]` is often `(ndraws, ncomps_j, branch_dims[j])`.

        The implementation is intentionally written using `...` indexing so the
        leading dimensions may vary; only the *last axis* is assumed to be the
        hyperparameter column axis.
    branch_groups : array-like or None
        Optional integer index array used when a local model depends on global
        hyperparameters.

        If provided and a requested hyperparameter lives on a different branch
        (`hp_branch_idx != branch_idx`), we apply `hp_vals = hp_vals[branch_groups]`
        to repeat/group the global values so they align with the local leaves.

        Typical shape is `(tot_nleaves_branch,)`, mapping each local leaf to the
        appropriate global group index.
    pad : bool
        If True, add a trailing singleton axis to each hyperparameter value via
        `hp_vals[..., None]` so it broadcasts cleanly with flattened sample axes
        (e.g. `nevs * nsamples`). If False, return the raw `hp_vals` without the
        extra axis (useful for moment computations).

    Returns
    -------
    model : object
        Distribution object with `.pdf(...)`.
    model_hp_vals_dict : dict
        Dict of model hyperparameter arrays, shaped for broadcasting according
        to `pad`.
    """

    if param_dict is None: 
        param_dict = hp_ordering[branch_idx][param]

    # output dict - model_hp_vals_dict will be unpacked into the model function
    # start with fixed hyperparameters - can be overwritten
    model_hp_vals_dict = param_dict.get('params_fix', {}).copy() 

    model_hp_names = param_dict['model_hp_names']
    model_hp_col_idxs = param_dict['col_idx']

    # get global hps of this param, if needed
    if param in hp_ordering[-1] and branch_names[branch_idx] != 'global': 
        global_hp_vals = unpack_hp_vals(-1, param, hyperparams)
        if branch_groups is None:
            model_hp_vals_dict.update(global_hp_vals)
        else: # apply groups to broadcast with local hp vals
            model_hp_vals_dict.update({
                k: (v[branch_groups] if xp.asarray(v).size > 1 else v)
                for k, v in global_hp_vals.items()
            })
    
    # now get the hyperparameter values from this branch
    for hp, hp_col_idx in zip(model_hp_names, model_hp_col_idxs):
        hp_vals = xp.asarray(hyperparams[branch_idx][..., hp_col_idx])
        if hp == 'xmin' and 'xmin' in model_hp_vals_dict:
            # take the most stringent minimum
            # hp might have nans in it, so we need to handle that
            model_hp_vals_dict[hp] = xp.where(
                (hp_vals > model_hp_vals_dict[hp]) & xp.isfinite(hp_vals),
                hp_vals,
                model_hp_vals_dict[hp]
            )
        else:
            model_hp_vals_dict[hp] = hp_vals
    
    if 'subparams' in param_dict: # joint model - evaluate subdictionaries
        for subparam in param_dict['subparams']:
            model_hp_vals_dict[f'{subparam}_kwargs'] = unpack_hp_vals(
                branch_idx, subparam, hyperparams, branch_groups, param_dict=param_dict[subparam]
            )

    return model_hp_vals_dict

def eval_param_model(data_flat, branch_idx, param, hyperparams, branch_groups=None):
    """Convenience wrapper: unpack hyperparameters then evaluate the model pdf."""
    model_hp_vals_dict = unpack_hp_vals(branch_idx, param, hyperparams, branch_groups)
    return model_func_wrapper(branch_idx, param, model_hp_vals_dict, data_flat)

def eval_param_moments(branch_idx, param, hyperparams, branch_groups=None):
    """Convenience wrapper: unpack hyperparameters then evaluate the model moments."""
    model_hp_vals_dict = unpack_hp_vals(branch_idx, param, hyperparams, branch_groups)
    return model_moments_wrapper(branch_idx, param, model_hp_vals_dict)

def eval_param_marginals(x_arr, branch_idx, param, hyperparams):

    """
    Gets the marginal pdfs for given param. If `param == "redshift"`, then evaluates
    the rate psi(z).
    """

    x = xp.asarray(x_arr) # numpy input ok
    if param in hp_ordering[branch_idx]:
        if param == 'redshift':
            model_hp_vals_dict = utils.recursive_pad(unpack_hp_vals(branch_idx, param, hyperparams))
            return hp_ordering[branch_idx][param]['model'].psi(x, **model_hp_vals_dict)

        return eval_param_model({param: x}, branch_idx, param, hyperparams)
    
    # else: get marginal from a joint parent model
    parent_param = _get_parent_param(branch_idx, param)
    parent_param_dict = hp_ordering[branch_idx][parent_param]
    
    parent_model_hp_vals_dict = unpack_hp_vals(branch_idx, parent_param, hyperparams, param_dict=parent_param_dict)
    parent_model_hp_vals_dict = utils.recursive_pad(parent_model_hp_vals_dict)
    parent_model = hp_ordering[branch_idx][parent_param]['model']
    return parent_model.get_marginal_pdf(x, param, **parent_model_hp_vals_dict)

def _vectorize_moments_dict(moments_dict, rescale=True):
    res = []
    for p, (mu, sig) in moments_dict.items():
        scale = utils.PARAM_SCALES[p] if rescale else 1
        res.extend([mu/scale, sig/scale])
    
    return xp.stack(res, axis=-1).astype(xp.float32) 
    # should be (nsamples, ncomp, dfeat) or (nleaves_tot, dfeat) if not grouped

def compute_branch_moment_features(branch_idx, hps, branch_groups=None):
    """
    Build a moment-based feature vector for a single RJMCMC leaf, scaled by the typical parameter
    scales set in utils.PARAM_SCALES.

    Parameters
    ----------
    branch_idx : int
        Which local RJ branch this leaf belongs to.
    hyperparams : list
        List of length `len(branch_names)` used as a lightweight container for
        hyperparameters required by `eval_param_moments`.

        Convention:
        - `hyperparams[-1]` (global branch) is an array of shape
          (nsamples, 1, branch_dims[-1]).
        - `hyperparams[branch_idx]` is an array of shape (nsamples, ncomps, branch_dims[branch_idx])
        - All other entries may be None (won't be read)

    Notes
    -----
    The feature vector concatenates (mean, std) for each parameter in
    `params_to_use`, each scaled by `utils.PARAM_SCALES[param]` so Euclidean distance is
    comparable across dimensions.
    """

    moments_dict = {}
    for p in utils.skip_dunder(hp_ordering[branch_idx].keys()):
        moments_dict.update(eval_param_moments(branch_idx, p, hps, branch_groups))
    
    feat_params = moments_dict.keys()
    feats = _vectorize_moments_dict(moments_dict, rescale=True)
    
    return feats, feat_params

def _label_samples_loc_global_kmeans_moments(branch_idx, samples_loc, samples_global, ncomp, rng_seed=0):
    """Assign stable component labels via global KMeans on moment features.

    This performs a *global* clustering over all active leaves across samples for
    a given branch. Inactive leaves (NaN or rate<=0) are assigned label -1.

    Returns
    -------
    labels : np.ndarray
        Shape (nsamples, ncomp). Each entry is in [0, ncomp-1] for active
        leaves, or -1 for inactive.
    feats : np.ndarray
        Shape (nsamples, nleaves_max, 2 * nparams).
    feat_params : list
        List of parameter names used to compute the features, in order. There
        are 2 * len(feat_params) features per sample.
    """
    nsamples, nleaves_max, _ = samples_loc.shape 
    active = utils.leaf_active_mask(samples_loc)

    hps = [None] * len(branch_names)
    hps[branch_idx] = xp.asarray(samples_loc)
    hps[-1] = xp.asarray(samples_global)

    feats, feat_params = compute_branch_moment_features(branch_idx, hps)
    feats = utils.to_numpy(feats)
    dfeats = feats.shape[-1]

    X = feats.reshape(-1, dfeats)
    active_flat = active.reshape(-1)
    X_act = X[active_flat]
    if X_act.shape[0] < ncomp:
        raise ValueError('Too few active leaves to fit KMeans')

    km = KMeans(n_clusters=ncomp, random_state=rng_seed, n_init='auto')
    labels_act = km.fit_predict(X_act)

    # order clusters by decreasing cluster size (number of assigned active points)
    counts = np.bincount(labels_act, minlength=ncomp)
    order = np.argsort(-counts, kind='stable')
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(ncomp)
    labels_act = inv_order[labels_act]

    labels_flat = np.full(X.shape[0], -1, dtype=np.int32)
    labels_flat[active_flat] = labels_act.astype(np.int32)
    labels = labels_flat.reshape(nsamples, nleaves_max)

    return labels, feats, list(feat_params)

def _get_bad_leaves(feats, groups, nleaves_max, min_sep=1):

    """
    For the given features of a branch, identify the per-group draws violating the constraint that 
    the (scaled) leaf features (mean, width) must be at least min_sep apart at least one dimension.
    """

    # order = xp.argsort(groups)
    # feats_grouped = feats[order]
    # groups_grouped = groups[order] 
    # not needed - groups should always be sorted

    bad = xp.zeros_like(groups, dtype=bool)
    # min_sep2 = min_sep**2

    for k in range(1, nleaves_max): # sliding group
        same = groups[k:] == groups[:-k]
        if not xp.any(same):
            continue

        diff = feats[k:] - feats[:-k]

        # d2 = xp.sum(diff**2, axis=1)
        # viol = same & (d2 < min_sep2)

        # check that at least one parameter is at least min_sep away
        viol = same & xp.all(xp.abs(diff) < min_sep, axis=-1)

        bad[k:][viol] = True
        bad[:-k][viol] = True
    
    return utils.to_numpy(groups[bad])

def loglike(hyperparams, groups, data, injections):
    """Compute the per-group log-likelihood for the hierarchical population model.

    Implements the hierarchical likelihood with selection effects (see Eq. 5 of
    https://arxiv.org/pdf/2305.08909), evaluated on PE samples (for events) and
    injections (for the selection function). Supports NumPy or CuPy backends and
    always returns a NumPy array.

    Parameters
    ----------
    hyperparams : list[array-like]
        List of hyperparameter arrays for each branch in `BRANCH_NAMES` order.
        For local branches: arrays are shape (tot_nleaves, ndims). For the global
        branch: arrays are shape (ngroups, ndims). 
    groups : list[array-like]
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

    hyperparams = [xp.asarray(v, dtype=xp.float32) for v in hyperparams]
    groups = [xp.asarray(v, dtype=xp.int32) for v in groups]
    
    ngroups = groups[-1].shape[0] # since global groups is just range(ngroups)

    data_flat = {k : data[k].ravel() for k in data}

    dNdtheta_per_group = xp.zeros((ngroups, nevs * nsamples), dtype=xp.float32) # evaluated on PE samples
    dNdtheta_per_group_injs = xp.zeros((ngroups, ninjs), dtype=xp.float32) # evaluated on injections

    # iterate over branches -- different local branches will have different models
    all_branch_moments = [] # moments dict of each branch
    for branch_idx, branch_dict in enumerate(hp_ordering[:-1]): # all local branches

        branch_groups = groups[branch_idx]
        branch_nleaves = hyperparams[branch_idx].shape[0]

        N0 = TOBS * hyperparams[branch_idx][:, -1] # Tobs * R0

        dNdtheta_per_leaf = N0[:, None] * xp.ones((branch_nleaves, nevs * nsamples), dtype=xp.float32) # evaluated on PE samples
        dNdtheta_per_leaf_injs = N0[:, None] * xp.ones((branch_nleaves, ninjs), dtype=xp.float32) # evaluated on injections

        # evaluate likelihood for each event-level parameter
        branch_moments = {}
        for param_name in utils.skip_dunder(branch_dict.keys()): 

            model_hp_vals_dict = unpack_hp_vals(branch_idx, param_name, hyperparams, branch_groups)
            
            # compute dNdtheta per leaf for each of the local models
            dNdtheta_per_leaf *= model_func_wrapper(branch_idx, param_name, model_hp_vals_dict, data_flat)
            dNdtheta_per_leaf_injs *= model_func_wrapper(branch_idx, param_name, model_hp_vals_dict, injections)

            # get moments
            branch_moments.update(model_moments_wrapper(branch_idx, param_name, model_hp_vals_dict))

            if args.test:
                print(f'fraction of nan leaves after {param_name} (data)', xp.sum(xp.isnan(dNdtheta_per_leaf)) / dNdtheta_per_leaf.size)
                print(f'fraction of nan leaves after {param_name} (injs)', xp.sum(xp.isnan(dNdtheta_per_leaf_injs)) / dNdtheta_per_leaf_injs.size)
                print(f'fraction of zero leaves after {param_name} (data)', xp.sum(dNdtheta_per_leaf == 0) / dNdtheta_per_leaf.size)
                print(f'fraction of zero leaves after {param_name} (injs)', xp.sum(dNdtheta_per_leaf_injs == 0) / dNdtheta_per_leaf_injs.size)

                # print(dNdtheta_per_leaf.dtype)
                # print(dNdtheta_per_leaf_injs.dtype)
        
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
        all_branch_moments.append(branch_moments)

        if args.test:
            print(f'fraction of groups nan after {branch_names[branch_idx]} (data)', xp.sum(xp.isnan(dNdtheta_per_group)) / dNdtheta_per_group.size)
            print(f'fraction of groups nan after {branch_names[branch_idx]} (injs)', xp.sum(xp.isnan(dNdtheta_per_group_injs)) / dNdtheta_per_group_injs.size)
            print(f'fraction of zero groups after {branch_names[branch_idx]} (data)', xp.sum(dNdtheta_per_group == 0) / dNdtheta_per_group.size)
            print(f'fraction of zero groups after {branch_names[branch_idx]} (injs)', xp.sum(dNdtheta_per_group_injs == 0) / dNdtheta_per_group_injs.size)

            # print(dNdtheta_per_group.dtype)
            # print(dNdtheta_per_group_injs.dtype)
        
    # multiply this by global dP/dtheta for global theta
    for global_param_name in utils.skip_dunder(hp_ordering[-1].keys()):

        if 'model' in hp_ordering[-1][global_param_name]: 
            # only want purely global parameters
            # some parameters will live in local branches w/ some global parameters -- those already have been evaluated
            model_hp_vals_dict = unpack_hp_vals(-1, global_param_name, hyperparams)
            
            dNdtheta_per_group *= model_func_wrapper(-1, global_param_name, model_hp_vals_dict, data_flat)
            dNdtheta_per_group_injs *= model_func_wrapper(-1, global_param_name, model_hp_vals_dict, injections)

            if args.test:
                print(f'fraction of zero groups after {global_param_name} (data)', xp.sum(dNdtheta_per_group == 0) / dNdtheta_per_group.size)
                print(f'fraction of zero groups after {global_param_name} (injs)', xp.sum(dNdtheta_per_group_injs == 0) / dNdtheta_per_group_injs.size)

                print(dNdtheta_per_group.dtype)
                print(dNdtheta_per_group_injs.dtype)

    # divide by prior
    dNdtheta_per_group = dNdtheta_per_group / data_flat['prior'] 
    dNdtheta_per_group_injs = dNdtheta_per_group_injs / injections['prior']
    # reshape dNdtheta on data back into nevs, nsamples
    dNdtheta_per_group = dNdtheta_per_group.reshape((ngroups, nevs, nsamples))

    # if args.test:
    #     print(dNdtheta_per_group.dtype)
    #     print(dNdtheta_per_group_injs.dtype)

    # Importance sampling check for PE monte carlo integral
    # see Appendix A1 of GWTC4 paper https://arxiv.org/pdf/2508.18083
    mu_ll = xp.mean(dNdtheta_per_group, axis=-1).astype(xp.float64) # (ngroups, nevs)
    var_ll = (xp.sum(dNdtheta_per_group**2, axis=-1)/(nsamples - 1) - mu_ll**2)/nsamples # (ngroups, nevs)
    var_logl_pop = xp.sum(var_ll/(mu_ll**2), axis=1)

    # multiply event-level likelihoods together
    logl = xp.sum(xp.clip(xp.log(mu_ll), -INF, None), axis=1) # shape (ngroups,)

    if args.test:
        print('initial logl frac nans:', xp.count_nonzero(xp.isnan(logl)) / ngroups)
        print('initial logl frac -infs:', xp.count_nonzero(logl < -INF) / ngroups)

    # Poisson detection probability, evaluated on injections
    # this multiplies the likelihood by exp(-xi(Lambda)N(Lambda))
    
    # Importance sampling variance check for selection function 
    ngen = injections['total_generated']
    m = injections['w'].shape[0]
    mu_logl_sel = (1. / ngen) * xp.sum(injections['w'] * dNdtheta_per_group_injs, axis=-1).astype(xp.float64)
    var_logl_sel = 1. / (m * (m - 1)) * xp.sum(
        injections['w'] * (dNdtheta_per_group_injs * (m / ngen)  - mu_logl_sel[:, None])** 2, axis=-1
    )  # eq 12 of Essick 2021, https://iopscience.iop.org/article/10.3847/2515-5172/ac2ba7 

    logl -= mu_logl_sel # subtracts Nexp(Lambda) = xi(Lambda)N(Lambda) from log-likelihood

    # print('after sel logl # nans:', xp.count_nonzero(~xp.isfinite(logl)))
    
    # threshold on total monte-carlo variance
    var_logl_tot = var_logl_pop + var_logl_sel # equation A3 of GTWC4 paper (typo in paper - should be N(Lambda) instead of Ndet)
    logl = xp.where(var_logl_tot < 1, logl, -INF)

    if args.test:
        print('pop logl var:', var_logl_pop)
        print('sel logl var:', var_logl_sel)

    # new - enforce minimum distance between components
    if args.min_sep:
        bad_groups = set()
        for b_idx, branch_moments_dict in enumerate(all_branch_moments):
            nleaves_max = nleaves_max_dict[branch_names[b_idx]]
            if nleaves_max > 1:
                branch_feats = _vectorize_moments_dict(branch_moments_dict, rescale=True)
                # is this the fastest way to do this?
                bad_groups.update(_get_bad_leaves(branch_feats, groups[b_idx], nleaves_max, min_sep=args.min_sep))
        logl[xp.fromiter(bad_groups, dtype='int')] = -INF

        if args.test:
            print('fraction of groups masked out by min sep:', len(bad_groups) / ngroups)

    return utils.to_numpy(xp.clip(logl, -INF, None))
    # output needs to be a numpy array

# -----------------------------
# TEST LIKELIHOOD FUNCTION
# -----------------------------
# maybe test loglike function above and see if it actually reprodjces P(m1) as expected

if args.test:
    ngroups_test = 5

    for _ in range(10):
        print('')
        test_hp_vals = [priors[branch_name].rvs(size=(ngroups_test*nleaves_max_dict[branch_name],)) for branch_name in branch_names]

        # # test true vals
        # test_hp_vals = [
        #     np.stack([
        #         4 * np.ones((ngroups_test,), dtype=np.float64), # alpha
        #         5 * np.ones((ngroups_test,), dtype=np.float64), # delta
        #         18 * ( 1 - 0.04 ) * np.ones((ngroups_test,), dtype=np.float64) # R0
        #     ], axis=-1),
        #     np.stack([
        #         33.5 * np.ones((ngroups_test,), dtype=np.float64), # mu_p
        #         4 * np.ones((ngroups_test,), dtype=np.float64), # sigma_p,
        #         18 * 0.04 * np.ones((ngroups_test,), dtype=np.float64) # R0
        #     ], axis=-1),
        #     np.stack([
        #         3.2 * np.ones((ngroups_test,), dtype=np.float64), # gamma
        #         1.5 * np.ones((ngroups_test,), dtype=np.float64), # beta
        #         0.0 * np.ones((ngroups_test,), dtype=np.float64), # mu chi
        #         10**(-0.85) * np.ones((ngroups_test,), dtype=np.float64), # sigma chi,
        #         5 * np.ones((ngroups_test,), dtype=np.float64), # mmin global
        #     ], axis=-1)
        # ]

        test_groups = [xp.repeat(xp.arange(ngroups_test), nleaves_max_dict[branch_name]) for branch_name in branch_names]
        
        ll = loglike(test_hp_vals, test_groups, PE_samples, injections)

        print('test logl', ll)
        if np.any(~np.isfinite(ll)):
            raise ValueError('test logl has non-finite values')

# =====================
# Sampler initialization
# =====================

backend_path = os.path.join(outdir, 'backend.h5')
backend = HDFBackend(backend_path)
if os.path.exists(backend_path):
    try:
        state = backend.get_last_sample()
    except:
        os.remove(backend_path) # faulty backend / it hasn't been run

if os.path.exists(backend_path):
    print(f'Initializing from previous run at {backend_path}')
else:
    # Initialize parameter chains (walker positions)
    # coords is a dict keyed by branch name each of shape:
    #   (ntemps, nwalkers, nleaves_max, ndims) drawn from priors.
    print('Drawing initial coordinates...')
    coords = {
        branch_name: priors[branch_name].rvs(size=(args.ntemps, args.nwalkers, nleaves_max_dict[branch_name],))
        for branch_name in branch_names
    }

    # check that the first mass branch has one guy that has support at m=10
    mchar = 10
    b_idx_check = [b_idx for b_idx in range(len(branch_names)) if 'mass' in hp_ordering[b_idx]][0]

    for b_idx, bn in enumerate(branch_names[:-1]):

        ncomps = nleaves_max_dict[bn]
        if (ncomps <= 1 or args.min_sep == 0) and b_idx != b_idx_check:
            continue

        bad_init = True
        bad_init_count = 0

        while bad_init:
            if bad_init_count >= 1000:
                raise ValueError(f'Initialization failed after {bad_init_count} attempts. Check the prior ranges for branch {bn}')
            bad_init_count += 1

            bad_groups_mask = xp.zeros((args.ntemps, args.nwalkers), dtype='bool')
            input_hps = [xp.asarray(coords[bn]) for bn in branch_names]
        
            if b_idx == b_idx_check: 
                pdf_vals = eval_param_marginals(mchar, b_idx, 'mass_1_source', input_hps).reshape(
                    (args.ntemps, args.nwalkers, ncomps)
                )
                bad_groups_mask = xp.logical_or(
                    bad_groups_mask, xp.all(pdf_vals < 1e-6, axis=-1)
                ) # no leaves have support at mchar
                print(f'    frac of bad groups from p(m) check ({bn}): {xp.sum(bad_groups_mask)}/{bad_groups_mask.size}')

            if ncomps > 1 and args.min_sep > 0: # make sure leaves are initialized sufficiently far apart
                feats, _ = compute_branch_moment_features(b_idx, input_hps)
                minsep_bad_groups_mask = ~utils.check_min_separation(feats, min_sep=args.min_sep, xp=xp)
                print(f'    frac of bad groups from min sep check ({bn}): {xp.sum(minsep_bad_groups_mask)}/{minsep_bad_groups_mask.size}')
                bad_groups_mask = xp.logical_or(bad_groups_mask, minsep_bad_groups_mask)

            bad_groups = np.nonzero(utils.to_numpy(bad_groups_mask))
            n_bad_groups = len(bad_groups[0])
            if n_bad_groups:
                # redraw the bad walker/temps
                print('    Redrawing', n_bad_groups, f'bad groups for branch {bn}')
                rvs_shape = (n_bad_groups, ncomps)
                coords[bn][bad_groups + (slice(None),)] = priors[bn].rvs(size=rvs_shape)
            else:
                bad_init = False
    
    # sort all the leaves by m1
    input_hps = [xp.asarray(coords[bn]) for bn in branch_names]
    for b_idx, bn in enumerate(branch_names[:-1]):
        if 'mass' in hp_ordering[b_idx] and nleaves_max_dict[bn] > 1:
            moments = eval_param_moments(b_idx, 'mass', input_hps)
            mu = moments['mass_1_source'][0]
            sort_idx = xp.argsort(mu, axis=-1)
            # Broadcast and take
            sorted_coords = xp.take_along_axis(input_hps[b_idx], xp.expand_dims(sort_idx, axis=-1), axis=-2)
            coords[bn] = utils.to_numpy(sorted_coords)

    # Build the initial sampler state. `inds` is optional and not defined here;
    # we rely on `provide_groups=True` in EnsembleSampler to manage groups.
    state = State(coords)


### Define MCMC moves to use (Gibbs/Stretch/Random-walk)

rj_branches = [ bn for bn in branch_names if nleaves_max_dict[bn] > nleaves_min_dict[bn] ]
# use stretch move for all non-RJ branches
nonrj_sampling_setup = [ (bn, None) for bn in branch_names if bn not in rj_branches ]
stretch_move = StretchMove(gibbs_sampling_setup=nonrj_sampling_setup, live_dangerously=True)
moves = [(stretch_move, 0.3)]

# For non-RJ branches, we update each component AND the hyperparameters corresponding to each GW
# parameter independently, with a diagonal Gaussian proposal scaled by `factors` (one scale per 
# parameter), which is specified in the prior.
rj_moves = [] # RJMCMC moves to change model dimensionality (add/remove leaves)
for branch_name in rj_branches:

    branch_idx = branch_names.index(branch_name)
    branch_ndims = branch_dims[branch_idx]
    ncomp_max = nleaves_max_dict[branch_name]

    # `cov` is diagonal and broadcast-multiplied by `factors` so each param
    # has its own step scale. Requires len(factors) == dict_dims['gauss'].
    factor = factors[branch_idx]
    cov = {branch_name: np.diag(np.ones(branch_ndims)) * factor}

    # move each leaf AND parameter separately
    for param_name in utils.skip_dunder(hp_ordering[branch_idx].keys()):
        param_dict = hp_ordering[branch_idx][param_name]
        param_dim_inds = utils.recursive_get(param_dict, 'col_idx')
        for leaf_idx in range(ncomp_max):
            gibbs_array_i = np.zeros((ncomp_max, branch_ndims), dtype='bool')
            gibbs_array_i[leaf_idx, param_dim_inds] = True 

            gaussian_move_i = GaussianMove(cov, gibbs_sampling_setup=(branch_name, gibbs_array_i)) 
            if args.test:
                print(branch_name, gibbs_array_i)
            # can code be updated s.t. it doesn't update empty leaves?
            moves.append((gaussian_move_i, 0.2)) # '0.2' is the relative weight of which proposal to do
    
    # update rates together (can I update them on the simplex? / reparameterize)
    gibbs_rate_arr = np.zeros((ncomp_max, branch_ndims), dtype='bool')
    gibbs_rate_arr[:, -1] = True
    moves.append((GaussianMove(cov, gibbs_sampling_setup=(branch_name, gibbs_rate_arr)), 0.4))
    if args.test:
        print(branch_name, gibbs_rate_arr)
    
    if args.rj_num_try > 1: # use multiple try MCMC
        rj_move = MTDistGenMoveRJ(
            priors, num_try=args.rj_num_try, nleaves_min=nleaves_min_dict, nleaves_max=nleaves_max_dict,
            gibbs_sampling_setup=branch_name # change one branch at a time
        )
    else:
        rj_move = DistributionGenerateRJ(
            priors, nleaves_min=nleaves_min_dict, nleaves_max=nleaves_max_dict,
            gibbs_sampling_setup=branch_name # change one branch at a time
        )
    
    rj_moves.append((rj_move, 1.0))

if not rj_moves:
    rj_moves = False

# -----------------------------
# RUN SAMPLER, SAVE OUTPUT
# -----------------------------

# prepare logpath
logpath = os.path.join(outdir, 'info.txt')
if os.path.exists(logpath):
    utils.print_to(logpath, '\n\n' + '#'*50 + '\n')

# Run the MCMC
nsteps = 20 if args.test else args.nsteps
burn = 5 if args.test else args.burn

if args.test:
    backend=None

if nsteps + burn:
    try:
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
            fill_zero_leaves_val=-INF,
            backend=backend 
        )
    except ValueError as e:
        print(f"Can't use existing backend: {e}")
        if nsteps + burn: # start a new backend if we're running MCMC
            old_backend_path = utils.unique_path(backend_path)
            print(f'Moving old backend to {old_backend_path}')
            os.rename(backend_path, old_backend_path)
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
                fill_zero_leaves_val=-INF,
                backend=backend 
            )
    last_state = ensemble.run_mcmc(state, nsteps, burn=burn, progress=True)

    if args.test:
        exit()

    print(f'Saved backend to {backend_path}')

    # ------------------------------
    # Sampler diagnostics / thinning
    # ------------------------------

    logP = ensemble.get_log_posterior()[:,0]
    print('Maximum log posterior:', np.amax(logP))
    print('Fraction of samples with -inf log posterior:', np.sum(logP < -INF/2) / logP.size)
    print('# of samplers stuck at -inf:', np.sum(np.sum(logP < -INF/2, axis=0) > nsteps/2))
    nsamps, nwalker = logP.shape

    converged = False
    while not converged:
        # compute discard automatically -  fit linear + flat
        discard = utils.get_discard_from_chain(logP) if args.discard is None else args.discard

        if discard > nsamps - 500:
            print(f'Sampler has not converged - running for another 1000 steps')
            last_state = ensemble.run_mcmc(last_state, 1000, burn=burn, progress=True)
            print(f'Saved backend to {backend}')
            print('Maximum log posterior:', np.amax(logP))
            logP = ensemble.get_log_posterior()[:,0]
        else:
            converged = True

        print('\n########## MCMC DONE - PRINTING DIAGNOSTICS ##########\n')

        # print diagnostics
        utils.print_to(logpath, f'Acceptance fraction: {np.mean(ensemble.acceptance_fraction[0]):.2g}')
        if any([nleaves_max_dict[bn] > nleaves_min_dict[bn] for bn in branch_names]):
            utils.print_to(logpath, f'RJ acceptance fraction: {np.mean(ensemble.rj_acceptance_fraction[0]):.2g}')
        if args.ntemps > 1:
            utils.print_to(logpath, f'Swap acceptance fraction: {ensemble.swap_acceptance_fraction}')
        
        del ensemble
else:
    logP = backend.get_log_posterior()[:,0]
    discard = utils.get_discard_from_chain(logP) if args.discard is None else args.discard
    last_state = state

# plot log posterior

for i in range(logP.shape[-1]): # iterate through all walkers
    plt.plot(logP[:,i], lw=0.2, alpha=0.2, color='cornflowerblue')
# plt.axvline(x_break, color='r', lw=1)
plt.axvline(discard, color='r', ls='--', lw=1)
plt.plot(np.mean(logP, axis=-1), color='k', ls='--')
plt.ylabel('log P')
plt.xlabel('step')
plt.ylim(bottom=np.nanpercentile(logP[:,0], 1))
plt.savefig(os.path.join(figpath, 'logP_chains.png'))
print('Saved log posterior chains in logP_chains.png')
plt.close()

# compute thin
chain = backend.get_chain(discard=discard)
chain = {name: value[:, :1] for name, value in chain.items()} # get minimum temp but keep the dims
tau = utils.get_integrated_act_wrap(chain)
utils.print_to(logpath, f'\nIntegrated autocorrelation times: {tau}')

thin = round(max(np.max(arr) for _, arr in tau.items()) * 0.5)

logP = backend.get_log_posterior(discard=discard, thin=thin)
nsamples = logP[:,0].size
utils.print_to(logpath, f'Using thin {thin} and discard {discard} ({nsamples} samples)')

def save_or_not(path):
    if not '/' in path:
        if path.endswith('.png') or path.endswith('.pdf'):
            path = os.path.join(figpath, path)
        elif path.endswith('.npy') or path.endswith('.npz'):
            path = os.path.join(datapath, path)
    if '*' in path:
        matches = glob(path) 
        return len(matches) == 0 or args.replot
    return (not os.path.exists(path)) or args.replot

# plot temperature convergence
temp_plotname = 'log_posterior_temps.png'
logP_low, logP_high = np.nanpercentile(logP[:,0], [0.1, 99.9])
if args.ntemps > 1 and save_or_not(temp_plotname):
    beta_mean = np.mean(backend.get_betas(), axis=0)
    for i in range(logP.shape[1]):
        logP_temp = logP[:, i].ravel()
        logP_temp = logP_temp[logP_temp > logP_low]
        _ = plt.hist(logP_temp, histtype='step', bins=40, label=f'$\\beta \sim {beta_mean[i]:.2f}$', log=True)
    plt.xlim(logP_low, logP_high)
    plt.legend()
    plt.savefig(os.path.join(figpath, temp_plotname))
    print(f'Saved log posterior temperature convergence in {temp_plotname}')
    plt.close()

# prepare for big corner plot + sorting
chain = backend.get_chain(discard=discard, thin=thin, temp_index=0)

# plot nleaves chain if rj
nleaves_plotname = 'nleaves_chains.png'
if rj_branches and save_or_not(nleaves_plotname):
    fig, axes = plt.subplots(len(rj_branches), 1, figsize=(4, 3*len(rj_branches)), sharex=True, sharey=True)
    if len(rj_branches) == 1:
        axes = [axes]
    for ax, bn in zip(axes, rj_branches):
        nleaves_branch = np.count_nonzero(utils.leaf_active_mask(chain[bn]), axis=-1) # (nsteps, nwalkers)
        for walker_idx in np.random.choice(nleaves_branch.shape[1], size=5, replace=False):
            ax.plot(nleaves_branch[:, walker_idx], alpha=0.5, lw=0.5)
        ax.plot(np.mean(nleaves_branch, axis=-1), color='k', ls='--')
        ax.text(
            0.5, 0.9, bn, transform=ax.transAxes, 
            ha='center', va='center', fontsize=12, fontweight='bold',
        )
        ax.set_ylabel('num of leaves')
    
    plt.xlabel('step')
    plt.ylim(bottom=0)

    plt.savefig(os.path.join(figpath, nleaves_plotname))
    print(f'Saved nleaves chains in {nleaves_plotname}')
    plt.close()

# ----------------------------------
# Post-processing: labeling and PPDs
# ----------------------------------

bf_threshold = 0.05

# RJ model selection:
# Define a model as the tuple of per-branch active component counts across the
# local RJ branches. We only visualize models that have a BF > 0.05 compared to
# the preferred model.

# Here, we identify the different models and compute draw indices per model, evidences, 
# and select the top models for plotting.

samples_unsorted = {
    bn: chain[bn].reshape(-1, nleaves_max_dict[bn], branch_dims[b_idx], copy=True) \
        for b_idx, bn in enumerate(branch_names)
}
nsamples = samples_unsorted['global'].shape[0]
del chain

sig_names, model_inds = [], []
n_labelled_comps = np.ones(len(branch_names), dtype=np.int32)
if rj_branches:
    
    model_counts = []
    for bn in rj_branches:
        active_b = utils.leaf_active_mask(samples_unsorted[bn])
        b_nleaves_samples = np.sum(active_b, axis=1).astype(np.int32)
        model_counts.append(b_nleaves_samples)

    if save_or_not('ncomponents.png'):
        fig, axes = plt.subplots(len(model_counts), 1, sharex=True, figsize=(4, len(model_counts)*1.5))
        if len(model_counts) <= 1:
            axes = [axes]
        for ax, bn, b_nleaves_samples in zip(axes, rj_branches, model_counts):
            k_max = nleaves_max_dict[bn]
            counts = np.bincount(b_nleaves_samples, minlength=k_max + 1)
            k = np.arange(k_max + 1)
            ax.bar(k, counts, width=0.85, align="center")
            ax.set_ylabel('Counts')
            ax.text(
                0.5, 0.9, bn, transform=ax.transAxes, 
                ha='center', va='center', fontsize=12, fontweight='bold',
            )

        axes[-1].set_xlabel('Number of Components')
        axes[-1].set_xlim(0, nleaves_max_dict[bn]+1)
        axes[-1].set_xticks(np.arange(nleaves_min_dict[bn], nleaves_max_dict[bn]+1))

        plt.savefig(os.path.join(figpath, 'ncomponents.png'))
        print('Saved posterior on # of components per RJ branch in ncomponents.png')
        plt.close()  

    model_counts = np.stack(model_counts, axis=1) # shape (nsamples, nrjbranches), of # of components of each branch
    sigs_all, inv, counts = np.unique(model_counts, axis=0, return_inverse=True, return_counts=True)
    # sigs are the unique model 'signatures', i.e. tuples with the # of components of each rj branch

    sig_names_all = [', '.join([f'{int(sig_count)} {rj_bn}' for rj_bn, sig_count in zip(rj_branches, sig)]) for sig in sigs_all]

    # save relative evidences
    bayes_factors_all = counts / np.amax(counts)
    utils.print_to(logpath, '\nBAYES FACTORS (relative to preferred model):')
    pad = max([len(str(list(s))) for s in sig_names_all]) + 1
    for sig_name, bf in zip(sig_names_all, bayes_factors_all):
        utils.print_to(logpath, f'{sig_name:<{pad}}: {bf:.5f}')

    sigs, bayes_factors = [], []
    for t, bf in enumerate(bayes_factors_all):

        if bf > bf_threshold: # ignore insignificant models:
            idx = np.nonzero(inv == t)[0]
            model_inds.append(idx.copy())
            sigs.append(sigs_all[t])
            sig_names.append(sig_names_all[t])
            bayes_factors.append(float(bf))

    nmodels_plot = len(model_inds)
    rj_labelled_comps = np.amax(np.stack(sigs, axis=0), axis=0)
    for rj_branch, ncomps in zip(rj_branches, rj_labelled_comps):
        n_labelled_comps[branch_names.index(rj_branch)] = ncomps
    # this is the maximum number of components for each RJ branch across all (significant) models

    best_model_idx = np.argmax(bayes_factors)
else:
    nmodels_plot = 0
    n_labelled_comps = np.array([nleaves_max_dict[bn] for bn in branch_names], dtype=np.int32)
    best_model_idx = 0
sig_names.append(0) # for all non-rj parameters
model_inds.append(np.arange(nsamples))

samples_and_labels_datapath = os.path.join(datapath, 'samples_and_labels.npy')
if os.path.exists(samples_and_labels_datapath) and not args.replot:
    print('Loading samples and labels from samples_and_labels.npy')
    megadict = np.load(samples_and_labels_datapath, allow_pickle=True).item()
    samples_dict = megadict['samples']
    labels_dict = megadict['labels']
    feats_dict = megadict['feats']
else:
    print('\n Sorting samples via k-means...')

    samples_dict = {}
    labels_dict = {}
    feats_dict = {}

    samples_global = samples_unsorted['global']
    nsamps = samples_global.shape[0]
    for branch_idx, branch_name in enumerate(branch_names):

        samples_loc = samples_unsorted[branch_name]
        nleaves_max = nleaves_max_dict[branch_name]

        if nleaves_max == 1:
            samples_dict[branch_name] = samples_unsorted[branch_name]

        elif branch_name in rj_branches:

            ncomp = n_labelled_comps[branch_idx]
            try:
                print(f'   ...sorting branch {branch_name} via k-means with {ncomp} components...')
                labels_loc, feats, feat_params = _label_samples_loc_global_kmeans_moments(
                    branch_idx=branch_idx,
                    samples_loc=samples_loc,
                    samples_global=samples_global,
                    ncomp=ncomp,
                    rng_seed=args.seed + branch_idx + 1,
                )
                samples_loc_sorted = samples_loc

            except Exception as e:
                print(f'   ...labeling failed for branch {branch_name}, falling back to sorting by rate: {e}')

                rate = samples_loc[..., -1]
                rate = np.nan_to_num(rate, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
                rate = np.where(rate > 0, rate, -np.inf)
                order = np.argsort(rate, axis=1)[:, ::-1]
                samples_loc_sorted = np.take_along_axis(
                    samples_loc,
                    order[..., None],
                    axis=1,
                )

                labels_loc = np.broadcast_to(
                    np.arange(nleaves_max, dtype=np.int32)[None, :],
                    (nsamps, nleaves_max),
                ).copy()
                labels_loc[~utils.leaf_active_mask(samples_loc_sorted)] = -1
                feats = None

                n_labelled_comps[branch_idx] = nleaves_max_dict[branch_name]

            labels_dict[branch_name] = labels_loc
            samples_dict[branch_name] = samples_loc_sorted
            if feats is not None:
                feats_dict[branch_name] = feats

        else: # non-rj branch with more than 1 component, sort by m1
            print(f'   ...sorting branch {branch_name} via primary mass (non-rj branch)...')
            if 'mass' in hp_ordering[branch_idx]:
                input_hps = [None] * len(branch_names)
                input_hps[branch_idx] = xp.asarray(samples_loc)
                input_hps[-1] = xp.asarray(samples_global)

                mu, _ = eval_param_moments(branch_idx, 'mass', input_hps)
                sort_idx = xp.argsort(mu, axis=-1)
                # Broadcast and take
                sorted_samples = xp.take_along_axis(input_hps[branch_idx], xp.expand_dims(sort_idx, axis=-1), axis=-2)
                
                samples_dict[branch_name] = utils.to_numpy(sorted_samples)
                # labels_dict[branch_name] = np.broadcast_to(np.arange(ncomps)[None, :], (nsamps, ncomps))

    print('...done.')
    # 'samples_dict' is a dict of arrays of each branch with shape (nsamples, ncomps, nparams)
    del samples_unsorted

    # save samples_dict
    np.save(
        os.path.join(datapath, 'samples_and_labels.npy'),
        {
            'samples': samples_dict,
            'labels': labels_dict,
            'feats': feats_dict
        }, allow_pickle=True
    )
    print('Saved samples_and_labels.npy')

# convert to rate at z=0.2, consistent with LVK figures
# assumes that redshift is a global model

redshift_param_dict = hp_ordering[-1]['redshift']
z_02_factor = redshift_param_dict['model'].psi(
    0.2, 
    **{hp: samples_dict['global'][:, 0, col_idx] for hp, col_idx in zip(redshift_param_dict['model_hp_names'], redshift_param_dict['col_idx'])}
)
z_02_factor = utils.to_numpy(z_02_factor)

# get rates by component (and label components)
def _filter_rates(arr):
    return np.where(~np.isfinite(arr) | (arr < 0), 0.0, arr)

# assign colors to components
n_labelled_comps = [int(samples_dict[bn].shape[1]) for bn in branch_names[:-1]]
component_palette = sns.color_palette('Set2', sum(n_labelled_comps))

# assign names and colors to components
comp_labels = [] # names
comp_label_sigs = [] # (branch_name, label_idx)
comp_palette_dict = {} # dict of list of colors
start_idx = 0
for branch_idx, branch_name in enumerate(branch_names[:-1]):
    labels = labels_dict.get(branch_name, None)
    ncomps = samples_dict[branch_name].shape[1]
    comp_palette_dict[branch_name] = component_palette[start_idx:start_idx+ncomps]
    component_palette.extend(list(component_palette[start_idx:start_idx+ncomps]))

    if ncomps > 1:
        for label_idx in range(ncomps):
            comp_label_sigs.append((branch_name, label_idx))
            label_alpha = utils.ALPHABET[label_idx]
            if len(branch_names) > 2:
                comp_label = f'{branch_name} {label_alpha}'
                comp_labels.append(comp_label)
            else:
                comp_labels.append(label_alpha) # only 1 local branch
    else:
        comp_labels.append(branch_name)
        comp_label_sigs.append((branch_name, 0))
    start_idx += ncomps
# these are lists - careful that we plot things in the same order as we assign names

# get R02s aggregated by label
R02s_by_label = []
for bn in branch_names[:-1]:
    ncomps = samples_dict[bn].shape[1]
    labels = labels_dict.get(bn, None)
    for comp_idx in range(ncomps):
        if labels is None:
            R02s_by_label.append(_filter_rates(samples_dict[bn][:, comp_idx, -1]))
        else:
            R02s_by_label.append(np.sum(
                _filter_rates(samples_dict[bn][..., -1]) * (comp_idx == labels),
                axis=1
            ))
R02s_by_label = np.stack(R02s_by_label, axis=-1) * z_02_factor[:, None] # (nsamples, ncomps_tot)
R02_tot = np.sum(R02s_by_label, axis=-1)
branching_fracs = R02s_by_label / R02_tot[:, None]

# R02_tot and z_02_factor have shape (nsamples,) (no extra dimensions)

ndraws = 500

### CORNER PLOTS

def _corner_plot_wrapper(X, labels_cols, name=None, fig=None, title=None, **corner_kwargs):

    X = np.asarray(X)
    good = np.all(np.isfinite(X), axis=1)
    X = X[good]
    if X.shape[0] < 10:
        print(f'Skipping corner plot for {name}: too few finite samples ({X.shape[0]})')
        return

    input_kwargs = corner_defaults.copy()
    input_kwargs.update(corner_kwargs)
    fig = corner(X, labels=labels_cols, fig=fig, **input_kwargs)

    if title:
        fig.suptitle(title, y=1.02)

    if name is None:
        return fig
    safe_name = utils.get_safe_fn(name) # split by non-alphanumeric then join by '_'
    outpath = os.path.join(figpath, f'corner_{safe_name}.pdf')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'Saved corner_{safe_name}.pdf')

# plot component branching fractions + rates, color coding for different models
# plot the component if in any model it contributes more than 1%
nonsmall_comps = []
for inds in model_inds:
    nonsmall_comps.append(np.nonzero(np.median(branching_fracs[inds], axis=0) > 0.01)[0])
nonsmall_comps = np.unique(np.concatenate(nonsmall_comps))
nonsmall_comp_label_sigs = [comp_label_sigs[int(i)] for i in nonsmall_comps]
nonsmall_comp_labels = [comp_labels[int(i)] for i in nonsmall_comps]

if not args.skip_corner: # skip for debugging

    print('Preparing corner plots')

    # plot global branch
    X = samples_dict['global'][:,0]
    _corner_plot_wrapper(X, hp_ordering[-1]['__latex__'], 'global')

    for data, data_name in zip([branching_fracs, R02s_by_label], ['branching_fracs', 'rates']):
        if save_or_not(f'corner_{data_name}.pdf'):
            ranges = [(0.0, float(np.nanpercentile(data[:, comp], 99.9))) for comp in nonsmall_comps]
            if rj_branches: # plot models in different colors
                model_palette = sns.color_palette('Dark2')
                fig = None
                handles = []
                for model_idx, (sig, inds) in enumerate(zip(sig_names[:-1], model_inds[:-1])):
                    c = matplotlib.colors.to_hex(model_palette[model_idx])
                    fig = _corner_plot_wrapper(
                        data[inds][:, nonsmall_comps], 
                        nonsmall_comp_labels, 
                        name=None, 
                        fig=fig, 
                        color=c,
                        fill_contours=False,
                        # scale_hist=True,
                        range=ranges,
                        show_titles=False,
                        quantiles=None
                    )
                    bf = bayes_factors[model_idx]
                    handles.append(matplotlib.lines.Line2D([], [], color=c, label=sig + f' ($\\mathcal{{B}}={bf:.2f})$'))
                    # plot overall quantiles
                    fig = _corner_plot_wrapper(
                        data[:, nonsmall_comps], nonsmall_comp_labels, 
                        name=None, color='k', range=ranges, fig=fig, **utils.invisible_corner_kwargs
                    )
                    fontsize = 18 if len(nonsmall_comps) > 2 else 12
                    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=fontsize)
                    # fig.suptitle(r'$\mathcal{R}(z=0.2)$ [Gpc${}^{-3}$ yr${}^{-1}$]', y=1.02)
            else:
                fig = _corner_plot_wrapper(
                    data[:, nonsmall_comps], nonsmall_comp_labels, 
                    name=None, color='k', range=ranges, 
                )
            fig.savefig(os.path.join(figpath, f'corner_{data_name}.pdf'))
            print(f'Saved corner_{data_name}.pdf')
        else:
            print(f'Skipping corner_{data_name}.pdf')

    # plot all components
    for i, (comp_label, comp_label_sig) in enumerate(zip(nonsmall_comp_labels, nonsmall_comp_label_sigs)):
        if save_or_not(f'corner_{comp_label}*.pdf'):
            bn, comp_idx = comp_label_sig

            sb = samples_dict[bn]
            names = hp_ordering[branch_names.index(bn)]['__latex__']
            if bn in rj_branches:
                X = sb[ labels_dict[bn] == comp_idx ]
            else:
                X = sb[:, comp_idx]
            _corner_plot_wrapper(X, names, comp_label)

            # also plot a separate corner plot for the most preferred model
            best_model_inds = model_inds[best_model_idx]
            if bn in rj_branches:
                X = sb[best_model_inds][ labels_dict[bn][best_model_inds] == comp_idx ]
            else:
                X = sb[best_model_inds][:, comp_idx]
            _corner_plot_wrapper(X, names, f'{comp_label}_{sig_names[best_model_idx]}')
        else:
            print(f'Skipping corner_{comp_label}*.pdf')

### extract and plot PPDs
print('\nPlotting PPDs')

model_draw_ctx_fn = 'model_draw_ctxs.npy'
if save_or_not(model_draw_ctx_fn):
    # prepare draws for each models
    model_draw_ctxs = {}
    for sig_name, inds in zip(sig_names, model_inds):
        samples_input, labels_input = [], []
        if inds.size > ndraws:
            sel_inds = np.random.choice(inds, size=ndraws, replace=False)
        else:
            sel_inds = inds
        for bn in branch_names:
            samples_input.append(utils.to_numpy(samples_dict[bn][sel_inds]))
            if bn in labels_dict:
                labels_input.append(np.asarray(labels_dict[bn][sel_inds]))
            else:
                labels_input.append(None)
        z_02_factor_draws = utils.to_numpy(z_02_factor[sel_inds])
        R02_tot_draws = utils.to_numpy(R02_tot[sel_inds])
        R02_labelled_draws = utils.to_numpy(R02s_by_label[sel_inds])

        model_draw_ctxs[sig_name] = {
            'sel_inds': sel_inds,
            'samples_input': samples_input,
            'labels_input': labels_input,
            'z_02_factor_draws': z_02_factor_draws,
            'R02_labelled_draws': R02_labelled_draws,
            'R02_tot_draws': R02_tot_draws,
        }
    # save
    np.save(os.path.join(datapath, model_draw_ctx_fn), model_draw_ctxs, allow_pickle=True)
    print(f'Saved {model_draw_ctx_fn}')
else:
    print(f'Loading {model_draw_ctx_fn}')
    model_draw_ctxs = np.load(os.path.join(datapath, 'model_draw_ctxs.npy'), allow_pickle=True).item()

# construct grid for evaluating data
ngrid = 101
data_grid = {
    'mass_1_source': np.linspace(2, 100, ngrid),
    'mass_2_source': np.linspace(2, 100, ngrid),
    'mass_ratio': np.linspace(0, 1, ngrid),
    'chi_eff': np.linspace(-1, 1, ngrid),
    'redshift': np.linspace(0, 1.5, ngrid),
}

def _leaf_rates_z02(samples_leaf, z_02_factor_draws):
    """Convert per-leaf rates to z=0.2 rates (NumPy), masking invalid/negative."""
    r0 = utils.to_numpy(samples_leaf[..., -1])
    r0 = np.nan_to_num(r0, nan=0.0, posinf=0.0, neginf=0.0)
    r0 = np.where(r0 > 0, r0, 0.0)
    return r0 * z_02_factor_draws.reshape((-1,) + (1,) * (r0.ndim - 1)) # nsamples is the 0th dimension
    # nsamples, ncomps

def _branch_label_ppd(x, param_name, branch_idx, draw_ctx, use_labels=True):
    """Compute (label-aggregated) PPD contributions for one branch.

    For each posterior draw and each cluster label, sum contributions from all
    leaves assigned that label (component) from k-means clustering. This allows multiple 
    leaves from the same draw to map to the same label (they add).

    Returns
    -------
    rate_ppd_by_label : np.ndarray
        Shape (nlabels, ndraws, ngrid_x). Each entry is a rate-weighted PPD
        contribution for that label.
    rate_by_label : np.ndarray
        Shape (nlabels, ndraws, 1). The corresponding total rate(z=0.2) per draw.
    x : np.ndarray
        The grid used for this parameter.
    """

    branch_name = branch_names[branch_idx]
    ncomps = n_labelled_comps[branch_idx]
    labels = draw_ctx['labels_input'][branch_idx]

    hyperparams = draw_ctx['samples_input']
    rate02 = _leaf_rates_z02(hyperparams[branch_idx], draw_ctx['z_02_factor_draws']) # (nsamples, ncomp)

    leaf_pdf = utils.to_numpy(eval_param_marginals(x, branch_idx, param_name, hyperparams))
    leaf_pdf = np.nan_to_num(leaf_pdf, nan=0.0, posinf=0.0, neginf=0.0)

    # leaf_pdf has shape (nsamples, ncomps, ngrid)
    if branch_name in rj_branches and use_labels:
        # aggregate by label (sum of rate-weighted leaves with same label, per sample)
        rate_ppd_by_label = np.zeros((ncomps, leaf_pdf.shape[0], leaf_pdf.shape[-1]), dtype=np.float64) # ncomps, nsamples, ngrid
        rate_by_label = np.zeros((ncomps, leaf_pdf.shape[0]), dtype=np.float64)
        for label in range(ncomps): # label assigned by kmeans to each leaf
            w = (rate02 * (labels == label)).astype(np.float64)
            rate_ppd_by_label[label] = np.sum(leaf_pdf * w[..., None], axis=1)
            rate_by_label[label] = np.sum(w, axis=1)
    else:
        rate_by_label = np.swapaxes(rate02, 0, 1)
        rate_ppd_by_label = np.swapaxes(leaf_pdf * rate02[..., None], 0, 1)
    
    if param_name == 'redshift':
        # want rate at z=0.2
        rate_ppd_by_label = rate_ppd_by_label * draw_ctx['z_02_factor_draws'][:, None]

    return rate_ppd_by_label, rate_by_label
    # output shapes: (ncomps, nsamples, ngrid), (ncomps, nsamples), (ngrid)

if args.LVK_plot != 'none':

    lvk_res_path = '/work/aqc/data/GWTC_data/processed/o4a-astro'

    # plot LVK results as reference
    # does not work for chirp mass
    from popsummary.popresult import PopulationResult
    
    if args.LVK_plot == 'default':
        res = PopulationResult(fname = os.path.join(lvk_res_path, 'data_release/BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5'))
    elif args.LVK_plot == 'spline':
        res = PopulationResult(fname = os.path.join(lvk_res_path, 'data_release/BBHMassSpinRedshift_BSplineIID.h5'))
    spin_res = PopulationResult(fname = os.path.join(lvk_res_path, 'data_release/BBHSpin_EpsSkewNormalChiEff.h5'))
else:
    res, spin_res = None, None


def plot_model_ppd(param_name, color_tot='cornflowerblue', use_labels=True):
    """
    Plot PPDs using one subplot per RJ model (top-N by posterior frequency).

    Components are grouped by (branch_name, label_id). For each model subplot, we
    evaluate PPDs on a random subsample of posterior draws belonging to that RJ
    model signature, and plot credible intervals for each component plus a total.

    CI=None plots individual ppds rather than a credible interval.
    """

    param_name_check = 'mass' if param_name.startswith('mass') else param_name
    # assumes joint mass model -- currently hard coded in

    x = data_grid[param_name]
    dict_out = {} # save ppd by component
    if any(param_name_check in hps for hps in hp_ordering[:-1]):

        # local parameter

        if param_name == 'mass_1_source' or param_name == 'mass_2_source':
            figsize = (16, 4 * (nmodels_plot+1))
            fontsize = 20
        else:
            figsize = (12, 4 * (nmodels_plot+1))
            fontsize = 17

        fig, axes = plt.subplots(nmodels_plot+1, 2, figsize=figsize, sharex=True, sharey=True)
        if nmodels_plot == 0:
            axes = [axes]
        # last row will be combined between all models
    
        for model_idx, (row_axes, sig) in enumerate(zip(axes, sig_names)):
            # plot each model
            ax_comp, ax_tot = row_axes

            # plot GWTC-4 on all plots
            utils.setup_and_plot_GWTC4(param_name, ax_comp, res=res, spin_res=spin_res)
            utils.setup_and_plot_GWTC4(param_name, ax_tot, res=res, spin_res=spin_res)

            draw_ctx = model_draw_ctxs[sig]

            ppd_by_comp = []
            rate_by_comp = []

            for branch_idx, branch_name in enumerate(branch_names[:-1]):
                palette = comp_palette_dict[branch_name]
                if param_name_check not in hp_ordering[branch_idx]:
                    continue
                branch_rate_ppd_by_label, branch_rate_by_label = _branch_label_ppd(
                    x, param_name, branch_idx, draw_ctx, use_labels
                )

                ppd_by_comp.append(branch_rate_ppd_by_label)
                rate_by_comp.append(branch_rate_by_label)
                
                if param_name == 'chi_eff':
                    branch_rate_ppd_by_label = branch_rate_ppd_by_label / draw_ctx['R02_tot_draws'][:, None] # want to plot chieff normalized
                
                # each ppd is (nsamples, ngrid)
                for label_idx, ppd in enumerate(branch_rate_ppd_by_label):
                    legend_label = branch_name if (label_idx == 0 and len(branch_names) > 2) else None
                    color = palette[label_idx] if use_labels else color_tot

                    utils.plot_ppds(ax_comp, x, ppd, CI=None, color=color, label=legend_label)
                    # use individual ppds for components

            # plot total
            ppd_by_comp = np.concatenate(ppd_by_comp, axis=0)
            rate_by_comp = np.concatenate(rate_by_comp, axis=0)
            tot_ppd = np.sum(ppd_by_comp, axis=0)
            if param_name == 'chi_eff':
                tot_ppd = tot_ppd / draw_ctx['R02_tot_draws'][:, None]
            utils.plot_ppds(ax_tot, x, tot_ppd, CI=90, color=color_tot, label='Total PPD')

            # formatting
            ax_tot.set_ylabel(None)
            if model_idx < nmodels_plot:
                ax_comp.set_xlabel(None)
                ax_tot.set_xlabel(None)
            
            # label each row with model name and bayes factor (if RJ)
            if sig:
                bf = f'{bayes_factors[model_idx]:.2f}' if bayes_factors[model_idx] > 0.01 else f'{bayes_factors[model_idx]:.2e}'
                plt_label = utils.textsc_ify(f'{sig} ($\\mathcal{{B}}={bf}$)')
            elif rj_branches: # no label needed unless RJ
                plt_label = utils.textsc_ify('Combined')
            ax_tot.text(
                0.96, 0.95, plt_label, transform=ax_tot.transAxes, 
                ha='right', va='top', fontsize=fontsize, 
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2)
            )

            # save to dictionary
            dict_out[sig] = {}
            dict_out[f'{sig} rates'] = {}
            for comp_label, ppd, rate in zip(comp_labels, ppd_by_comp, rate_by_comp):
                dict_out[sig][comp_label] = utils.to_numpy(ppd)
                dict_out[f'{sig} rates'][comp_label] = utils.to_numpy(rate)
        
        # legend
        all_handles, all_labels = [], []
        for ax in axes[-1]: # collect legend labels
            h, l = ax.get_legend_handles_labels()
            for handle, label in zip(h, l):
                if label not in all_labels:  
                    all_handles.append(handle)
                    all_labels.append(label)
        fig.legend(
            handles=all_handles, labels=all_labels, loc='lower center', 
            bbox_to_anchor=(0.5, 0.98), ncol=len(all_handles), fontsize=fontsize+2, frameon=False
        )
        plt.tight_layout()
    
    elif param_name_check in hp_ordering[-1]:

        # global parameter

        fig, ax = plt.subplots()
        draw_ctx = model_draw_ctxs[0]
        R02 = draw_ctx['R02_tot_draws']

        ppd, R02 = _branch_label_ppd(data_grid[param_name], param_name, -1, draw_ctx)
        ppd, R02 = ppd[0], R02[0] # remove ncomps axis
        if param_name == 'chi_eff':
            ppd = ppd / R02[:, None]
        
        utils.setup_and_plot_GWTC4(param_name, ax, res=res)
        utils.plot_ppds(ax, x, ppd, CI=90, color=color_tot)
        ax.legend()

        # save to dictionary
        dict_out['ppd'] = utils.to_numpy(ppd)
        dict_out['rate'] = utils.to_numpy(R02)
    
    else:
        raise ValueError(f'Unrecognized param name {param_name}')
    
    fn = f'{param_name}_labelled.pdf' if use_labels else f'{param_name}.pdf'
    plt.savefig(os.path.join(figpath, fn))
    print(f'Saved {fn}')
    plt.close()

    dict_out['x'] = utils.to_numpy(x)
    return dict_out

ppds_fn = 'ppds.npy'
if save_or_not(ppds_fn):
    ppds_dict = {}
    if rj_branches:
        ppds_dict['rj_branches'] = np.array(rj_branches, dtype=str)
        ppds_dict['model_sigs'] = sig_names
        ppds_dict['bayes_factors'] = bayes_factors
        ppds_dict['comp_labels'] = {
            'comp_names': comp_labels,
            'comp_sigs': comp_label_sigs,
            'colors': component_palette
        }
        
    for param_name in ['mass_1_source', 'mass_2_source', 'chi_eff', 'redshift', 'mass_ratio']:
        ppds_dict[f'{param_name}'] = plot_model_ppd(param_name, use_labels=False)
        for b_idx, bn in enumerate(branch_names[:-1]):
            if nleaves_max_dict[bn] > 1:
                if _get_parent_param(b_idx, param_name) in hp_ordering[b_idx]: # rj
                    ppds_dict[f'{param_name}_labelled'] = plot_model_ppd(param_name, use_labels=True)
                    break

    np.save(os.path.join(datapath, ppds_fn), ppds_dict, allow_pickle=True)
    print(f'Saved {ppds_fn}')
else:
    print(f'Loading in ppds from {ppds_fn}')
    ppds_dict = np.load(os.path.join(datapath, ppds_fn), allow_pickle=True).item()

# finally, plot 2D!!

def _extract_marg_ppd(x_param, model_sig, comp_label=None):

    if 'ppd' in ppds_dict[x_param]: # x is a global parameter
        return ppds_dict[x_param]['ppd']
    else:
        if comp_label is None:
            res = 0.
            for comp_label in ppds_dict[x_param][model_sig].keys():
                res = res + ppds_dict[x_param][model_sig][comp_label]
            return res
        elif comp_label in ppds_dict[x_param][model_sig]:
            return ppds_dict[x_param][model_sig][comp_label]
        else:
            raise ValueError(f'Unrecognized component label {comp_label}')

def _compute_mass_ppds(x_param, y_param, model_sig=0):
    
    xx = ppds_dict[x_param]['x']
    yy = ppds_dict[y_param]['x']

    hyperparams = model_draw_ctxs[model_sig]['samples_input']

    xx_, yy_ = np.meshgrid(xx, yy, indexing='ij')
    mass_data = {x_param: xp.asarray(xx_.ravel()), y_param: xp.asarray(yy_.ravel())}
    if 'mass_2_source' not in mass_data:
        mass_data['mass_2_source'] = mass_data['mass_1_source'] * mass_data['mass_ratio']
    elif 'mass_ratio' not in mass_data:
        mass_data['mass_ratio'] = mass_data['mass_2_source'] / mass_data['mass_1_source']
    else:
        raise ValueError(f'Unrecognized mass parameter combination {x_param} and {y_param}. One parameter must be `mass_1_source` and the other must be `mass_ratio` or `mass_2_source`.')

    mass_ppds = {}
    for branch_idx, bn in enumerate(branch_names[:-1]):
        branch_R02 = hyperparams[branch_idx][..., -1] * \
                     model_draw_ctxs[model_sig]['z_02_factor_draws'][:, None] # (nsamples, ncomps)
        if 'mass' in hp_ordering[branch_idx]:
            if 'model' in hp_ordering[branch_idx]['mass']: # primary model
                branch_xy_ppd = utils.to_numpy(eval_param_model(
                    mass_data, branch_idx, 'mass', hyperparams
                )) * branch_R02[..., None] # this is R(m1, q)
                if 'mass_2_source' in (x_param, y_param): # convert to p(m1, m2)
                    branch_xy_ppd = branch_xy_ppd / utils.to_numpy(mass_data['mass_1_source'])
                branch_xy_ppd = np.nan_to_num(branch_xy_ppd, nan=0.0, posinf=0.0, neginf=0.0)
                mass_ppds[bn] = branch_xy_ppd.reshape(branch_xy_ppd.shape[:-1] + (xx.size, yy.size))
                # should be (nsamples, ncomp, ngrid, ngrid)
    
    return mass_ppds

def _plot_param_2D_contours_and_marginals(
    x_param, y_param, model_sig, color='cornflowerblue', 
    axes=None, comp_label=None, savefig=None, mass_ppds=None, **contour_kwargs
):
    assert model_sig in ppds_dict['model_sigs'], f"model_name must be one of {ppds_dict['model_sigs']}"
    assert x_param in ppds_dict
    assert y_param in ppds_dict

    x_global = 'ppd' in ppds_dict[x_param]
    y_global = 'ppd' in ppds_dict[y_param]

    draw_ctx = model_draw_ctxs[model_sig]
    comp_labels = ppds_dict['comp_labels']['comp_names']
    comp_rates = draw_ctx['R02_labelled_draws'].T # (nsamples, ncomps) -> (ncomps, nsamples)
    model_sig_safe = utils.get_safe_fn(model_sig) # for saving plots
    
    if x_param.startswith('mass') and y_param.startswith('mass') and mass_ppds is None: 
        # pre-compute mass ppds to save time in recursive calls
        mass_ppds = _compute_mass_ppds(x_param, y_param, model_sig)

    if (comp_label == 'all') and not (x_global and y_global): # plot all together, in one plot
        # this is just a recursive call
        ncomps = len(comp_labels)

        # weight each components by alpha ~ sqrt(braching frac.) -- helps with visibility
        branching_fracs = comp_rates / np.nansum(comp_rates, axis=0, keepdims=True)
        comp_alphas = np.sqrt(np.nanmean(branching_fracs, axis=1))
        comp_alphas /= np.amax(comp_alphas)
        comp_alphas *= contour_kwargs.get('alpha', 1.0)

        idx_order = np.argsort(comp_alphas) # plot smallest components first

        if isinstance(color, (list, sns.palettes._ColorPalette)) and len(color) >= ncomps:
            colors = color[:ncomps]
        else:
            print('A color palette is required. Defaulting to `Set2`.')
            colors = sns.color_palette('Set2', ncomps)
        
        for idx in idx_order:
            comp_label, color, alpha = comp_labels[idx], colors[idx], comp_alphas[idx]
            axes = _plot_param_2D_contours_and_marginals(
                x_param, y_param, model_sig, color=color, alpha=alpha, savefig=None,
                axes=axes, comp_label=comp_label, mass_ppds=mass_ppds, **contour_kwargs
            )
        handles = [matplotlib.lines.Line2D([], [], color=colors[i], label=utils.textsc_ify(comp_labels[i].capitalize())) for i in range(ncomps)]
        ax_x = axes[2]
        ax_x.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=ncomps)

        if savefig:
            if not isinstance(savefig, str):
                savefig = f'contours2d_{x_param}_{y_param}_{model_sig_safe}_labelled.pdf'
            axes[0].savefig(os.path.join(figpath, savefig))
            print(f'Saved {savefig}')
        return axes

    # want to plot marginals on the side
    p_x_marg = _extract_marg_ppd(x_param, model_sig, comp_label)
    p_y_marg = _extract_marg_ppd(y_param, model_sig, comp_label)

    # get x/y grid from dict
    xx = ppds_dict[x_param]['x']
    yy = ppds_dict[y_param]['x']

    if x_param.startswith('mass') and y_param.startswith('mass'):
        if comp_label is None: # just want total
            p_xy = [np.nansum(b_ppd, axis=1) for b_ppd in mass_ppds.values()] # add components together
            p_xy = np.sum(np.stack(p_xy), axis=0) # add branches together
        else: # have to extract component from labels
            comp_idx = comp_labels.index(comp_label)
            bn, b_label_idx = ppds_dict['comp_labels']['comp_sigs'][comp_idx]
            labels = draw_ctx['labels_input'][branch_names.index(bn)]
            mask = (labels == b_label_idx)[..., None, None]
            p_xy = np.nansum(mass_ppds[bn]*mask, axis=1)
    else:
        if (comp_label is None) and not (x_global and y_global):
            p_xy = 0.
            for comp_label_, comp_rate in zip(comp_labels, comp_rates):
                comp_x_ppd = _extract_marg_ppd(x_param, model_sig, comp_label_)
                comp_y_ppd = _extract_marg_ppd(y_param, model_sig, comp_label_)
                comp_xy_ppd = np.where(
                    comp_rate[:, None, None] > 0,
                    np.expand_dims(comp_x_ppd, 2) * np.expand_dims(comp_y_ppd, 1) / comp_rate[:, None, None],
                    0.
                )
                p_xy = p_xy + comp_xy_ppd
        else:
            R02_tot = draw_ctx['R02_tot_draws']
            p_xy = np.expand_dims(p_x_marg, 2) * np.expand_dims(p_y_marg, 1) / R02_tot[:, None, None]
        
    p_xy_mean = np.nanmean(p_xy, axis=0)
    if np.all(np.isclose(p_xy_mean, 0.0, atol=1e-10)):
        print(f'WARNING: PDF is zero for {x_param} vs {y_param} for model `{model_sig}` and component `{comp_label}`.')
    else:
        axes = utils.plot_2D_contours_and_marginals(
            xx, yy, p_x_marg, p_y_marg, p_xy_mean, color=color, axes=axes, 
            xlabel = utils.texnames.get(x_param, x_param),
            ylabel = utils.texnames.get(y_param, y_param),
            **contour_kwargs
        )
    if savefig:
        if not isinstance(savefig, str):
            suffix = 'tot' if comp_label is None else utils.get_safe_fn(comp_label)
            savefig = f'contours2d_{x_param}_{y_param}_{model_sig_safe}_{suffix}.pdf'
        axes[0].savefig(os.path.join(figpath, savefig))
        print(f'Saved {savefig}')
    return axes

# plot 2d contours of best model
best_sig_name = sig_names[best_model_idx]
for param_x, param_y in [
    ('mass_ratio', 'chi_eff'),
    ('mass_1_source', 'chi_eff'),
    ('mass_1_source', 'mass_ratio'),
    ('mass_1_source', 'mass_2_source'),
]:
    x_param_ylog = False if param_x == 'chi_eff' else True
    y_param_ylog = False if param_y == 'chi_eff' else True
    if save_or_not(f'contours2d_{param_x}_{param_y}*_tot.pdf'):
        _plot_param_2D_contours_and_marginals( # total
            param_x, param_y, model_sig=best_sig_name, savefig=True,
            x_param_ylog=x_param_ylog, y_param_ylog=y_param_ylog, levels=(0.5, 0.9, 0.99), 
            color='cornflowerblue', comp_label=None, CI=90
        )
    if save_or_not(f'contours2d_{param_x}_{param_y}*_labelled.pdf'):
        _plot_param_2D_contours_and_marginals( # all individually
            param_x, param_y, model_sig=best_sig_name, savefig=True,
            x_param_ylog=x_param_ylog, y_param_ylog=y_param_ylog, levels=(0.5, 0.9), 
            color=component_palette, comp_label='all', CI=None
        )

R02_low, R02_med, R02_high = np.percentile(R02_tot, [5, 50, 95])
utils.print_to(logpath, f'\n\nThe total rate at z=0.2 is {R02_med:.1f} +{R02_high-R02_med:.1f} -{R02_med-R02_low:.1f} Gpc-3 yr-1')
utils.print_to(logpath, rf"\mathcal{{R}}(z=0.2) = "
      rf"{R02_med:.1f}^{{+{R02_high-R02_med:.1f}}}_{{-{R02_med-R02_low:.1f}}}"
      r"\ \mathrm{Gpc}^{-3}\ \mathrm{yr}^{-1}")
