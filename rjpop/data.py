import os
from typing import Literal

import numpy as np
import pdfs
import plot
import seaborn as sns
import utils
from astropy.cosmology import Planck15
from eryn.prior import ProbDistContainer, log_uniform, uniform_dist
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from xp import INF, scatter_add, xp

# ---------------------------
# --------- GLOBALS ---------
# ---------------------------

# should capitalize these
hp_ordering = {}
nleaves_min_dict = {}
nleaves_max_dict = {}

branch_names = []
branch_dims = []
RNG_SEED = 0

params = set()
branch_priordicts = []

# pre-processing


# construct list of dictionaries `hp_ordering` with the primary models belonging to each branch - this will be used in logl function
# note that models whose primary model are local could also have global parameters (hence branch_id list)
# [ {param: {  __model___: name of model (e.g. skew_t, gaussian),
#          model_hp_names: [hp_name1, hp_name2, ...],
#          latex_hp_names: [hp_latex1, hp_latex2, ...],
#                 col_idx: (param_idx1, param_idx2) }]            i.e. which column in data array in the branch the hyperparameter is found
def _process_param_hp_priors(
    param,
    input_param_priordict,
    branch_idx,
    output_branch_infodict,  # outputs
    hp_idx_start=0,
):

    if param == "mass_2_source":
        param_get = "mass_1_source"  # same models
    else:
        param_get = param

    output_branch_infodict[param] = {"model_hp_names": [], "col_idx": []}
    if "__model__" in input_param_priordict:
        model_name = input_param_priordict.pop("__model__")
        model = pdfs.MODELS[param_get][model_name]["model"]
        hps_fix = pdfs.MODELS[param_get][model_name].get("params_fix", {}).copy()

        output_branch_infodict[param].update(
            {
                "model_name": model_name,
                "model": model,
            }
        )

        if hasattr(model, "params"):
            params.update(model.params)

    else:
        # this should only be if the model was specified in a local branch
        assert branch_idx in [-1, len(branch_names) - 1], (
            "param must be global if no model specified"
        )
        assert param in params, "param must be evaluated in a local branch if no model specified"
        hps_fix = {}
        model_name = None

    hp_idx = hp_idx_start
    params.add(param)

    # update infodict + priordict
    for hp, hp_range in input_param_priordict.items():
        if hp in hps_fix:
            hps_fix.pop(hp)  # overwrite

        if isinstance(hp_range, list) or isinstance(hp_range, tuple):
            if len(hp_range) > 2 and hp_range[-1] == "log":
                branch_priordicts[branch_idx][hp_idx] = log_uniform(hp_range[0], hp_range[1])
            else:
                branch_priordicts[branch_idx][hp_idx] = uniform_dist(hp_range[0], hp_range[1])

            # update infodict
            output_branch_infodict[param]["model_hp_names"].append(hp)
            output_branch_infodict[param]["col_idx"].append(hp_idx)

            try:
                hp_ordering[branch_idx]["__latex__"].append(
                    pdfs.MODELS[param_get][model_name]["param_latex"][hp]
                )
            except KeyError:
                hp_ordering[branch_idx]["__latex__"].append(
                    pdfs.MODELS[param_get]["param_latex"].get(hp, hp)
                )

            hp_idx += 1

        elif isinstance(hp_range, float) or isinstance(hp_range, int):
            hps_fix[hp] = hp_range

        elif isinstance(hp_range, dict):  # hp_range is a subdictionary for another parameter hp
            # the use case for this is for joint (m1, m2) or (m1, q) evaluation
            if param != "mass" or not hp.startswith("mass"):
                raise ValueError("Joint evaluation only currently supported for mass")

            if "subparams" in output_branch_infodict[param]:
                output_branch_infodict[param]["subparams"].append(hp)
            else:
                output_branch_infodict[param]["subparams"] = [hp]

            hp_idx = _process_param_hp_priors(  # hierarchical branch infodict
                hp,
                hp_range,
                branch_idx,
                output_branch_infodict=output_branch_infodict[param],
                hp_idx_start=hp_idx,
            )

        else:
            raise ValueError(f"Invalid prior range for {param} {hp}: {hp_range}")

    output_branch_infodict[param]["params_fix"] = hps_fix
    if "subparams" in output_branch_infodict[param]:
        # instantiate joint model
        submodels = {
            f"{subparam}_model": output_branch_infodict[param][subparam]["model"]
            for subparam in output_branch_infodict[param]["subparams"]
        }
        output_branch_infodict[param]["model"] = model(**submodels)

    return hp_idx


def process_input_priordict(input_priordicts, rate_prior=(0.05, 100)):
    priors = {}  # {branch_name: ProbDistContainer({i: prior})}
    covs_all = []  # covariance matrices for gibbs sampling of the leaf

    global hp_ordering
    global branch_names
    global branch_priordicts
    global branch_dims
    global nleaves_min_dict
    global nleaves_max_dict

    branch_names = [d["__branch__"] for d in input_priordicts]
    hp_ordering = [{} for _ in branch_names]
    branch_priordicts = [{} for _ in branch_names]

    for branch_idx, input_dict in enumerate(input_priordicts):
        branch_name = input_dict.pop("__branch__")
        nleaves_min_dict[branch_name], nleaves_max_dict[branch_name] = input_dict.pop(
            "__ncomp__", [1, 1]
        )
        covs_all.append(np.asarray(input_dict.pop("__factor__", 0.005)))

        hp_idx = 0
        hp_ordering[branch_idx]["__latex__"] = []

        for param, input_param_priordict in input_dict.items():
            if param.startswith("__") or param.endswith("__"):  # ignore special / metainfo keys
                continue

            hp_idx = _process_param_hp_priors(
                param,
                input_param_priordict,
                branch_idx,
                output_branch_infodict=hp_ordering[branch_idx],
                hp_idx_start=hp_idx,
            )

        if branch_name != "global":
            #  add R0 for each local branch
            branch_priordicts[branch_idx][hp_idx] = uniform_dist(*rate_prior)
            hp_ordering[branch_idx]["__latex__"].append("$R_0$")
            hp_idx += 1

        # each local branch will have the number of parameters specified in the prior file + 1 (R0)

        priors[branch_name] = ProbDistContainer(branch_priordicts[branch_idx])
        branch_dims.append(hp_idx)

        if covs_all[-1].size < hp_idx - 1:
            print(
                f"WARNING: covariance diagonal size is too small. The last {covs_all[-1].size - hp_idx + 1} parameters will default to Gaussian step size 1."
            )

    return priors, covs_all


# ---------------------------
# Likelihood helper functions
# ---------------------------


def get_parent_param(branch_idx, param):
    for superparam in utils.skip_dunder(hp_ordering[branch_idx].keys()):
        if "subparams" in hp_ordering[branch_idx][superparam]:  # joint model
            if param in hp_ordering[branch_idx][superparam]["subparams"]:
                return superparam
    if param.startswith("mass"):
        return "mass"
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

    if "subparams" in hp_ordering[branch_idx][param]:  # joint model
        input_data = data_flat
    else:
        input_data = data_flat[param]
    model = hp_ordering[branch_idx][param]["model"]
    return model.pdf(input_data, **input_hp_vals)


def model_moments_wrapper(branch_idx, param, model_hp_vals_dict):
    model = hp_ordering[branch_idx][param]["model"]
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
        Name of the parameter being evaluated (key into `pdfs.MODELS` and
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
    model_hp_vals_dict = param_dict.get("params_fix", {}).copy()

    model_hp_names = param_dict["model_hp_names"]
    model_hp_col_idxs = param_dict["col_idx"]

    # get global hps of this param, if needed
    if param in hp_ordering[-1] and branch_names[branch_idx] != "global":
        global_hp_vals = unpack_hp_vals(-1, param, hyperparams)
        if branch_groups is None:
            model_hp_vals_dict.update(global_hp_vals)
        else:  # apply groups to broadcast with local hp vals
            model_hp_vals_dict.update(
                {
                    k: (v[branch_groups] if xp.asarray(v).size > 1 else v)
                    for k, v in global_hp_vals.items()
                }
            )

    # now get the hyperparameter values from this branch
    for hp, hp_col_idx in zip(model_hp_names, model_hp_col_idxs, strict=True):
        hp_vals = xp.asarray(hyperparams[branch_idx][..., hp_col_idx])
        if hp == "xmin" and "xmin" in model_hp_vals_dict:
            # take the most stringent minimum
            # hp might have nans in it, so we need to handle that
            model_hp_vals_dict[hp] = xp.where(
                (hp_vals > model_hp_vals_dict[hp]) & xp.isfinite(hp_vals),
                hp_vals,
                model_hp_vals_dict[hp],
            )
        if hp == "xmax" and "xmax" in model_hp_vals_dict:
            # take the most stringent maximum
            # hp might have nans in it, so we need to handle that
            model_hp_vals_dict[hp] = xp.where(
                (hp_vals < model_hp_vals_dict[hp]) & xp.isfinite(hp_vals),
                hp_vals,
                model_hp_vals_dict[hp],
            )
        else:
            model_hp_vals_dict[hp] = hp_vals

    if "subparams" in param_dict:  # joint model - evaluate subdictionaries
        for subparam in param_dict["subparams"]:
            model_hp_vals_dict[f"{subparam}_kwargs"] = unpack_hp_vals(
                branch_idx, subparam, hyperparams, branch_groups, param_dict=param_dict[subparam]
            )

    return model_hp_vals_dict


def eval_param_model(data_flat, branch_idx, param, hyperparams, branch_groups=None, pad=True):
    """Convenience wrapper: unpack hyperparameters then evaluate the model pdf."""
    model_hp_vals_dict = unpack_hp_vals(branch_idx, param, hyperparams, branch_groups)
    return model_func_wrapper(branch_idx, param, model_hp_vals_dict, data_flat, pad=pad)


def eval_param_moments(branch_idx, param, hyperparams, branch_groups=None):
    """Convenience wrapper: unpack hyperparameters then evaluate the model moments."""
    model_hp_vals_dict = unpack_hp_vals(branch_idx, param, hyperparams, branch_groups)
    return model_moments_wrapper(branch_idx, param, model_hp_vals_dict)


def eval_param_marginals(x_arr, branch_idx, param, hyperparams, psi_z=True):
    """
    Gets the marginal pdfs for given param. If `param == "redshift"`, then evaluates
    the rate psi(z) if psi_z=True and the pdf 1/R_0 dN/dz if psi_z=False.
    """

    x = xp.asarray(x_arr)  # numpy input ok
    pad = True if x.size > 1 else False

    if param in hp_ordering[branch_idx]:
        if param == "redshift":
            model_hp_vals_dict = unpack_hp_vals(branch_idx, param, hyperparams)
            if pad:
                model_hp_vals_dict = utils.recursive_pad(model_hp_vals_dict)
            if psi_z:
                return hp_ordering[branch_idx][param]["model"].psi(x, **model_hp_vals_dict)
            else:
                return hp_ordering[branch_idx][param]["model"].pdf(x, **model_hp_vals_dict)

        return eval_param_model({param: x}, branch_idx, param, hyperparams, pad=pad)

    # else: get marginal from a joint parent model
    parent_param = get_parent_param(branch_idx, param)
    parent_param_dict = hp_ordering[branch_idx][parent_param]

    parent_model_hp_vals_dict = unpack_hp_vals(
        branch_idx, parent_param, hyperparams, param_dict=parent_param_dict
    )
    if pad:
        parent_model_hp_vals_dict = utils.recursive_pad(parent_model_hp_vals_dict)
    parent_model = hp_ordering[branch_idx][parent_param]["model"]
    return parent_model.get_marginal_pdf(x, param, **parent_model_hp_vals_dict)


def get_auto_scale(moments_dict):
    scales = {}
    for p, feats in moments_dict.items():
        muvar = xp.nanvar(feats[0])
        if len(feats) > 1:
            stdvar, stdmed2 = xp.nanvar(feats[1]), xp.nanmedian(feats[1]) ** 2
        else:
            stdvar, stdmed2 = muvar, 0
        scales[p] = xp.sqrt(muvar + stdvar + stdmed2)
    return scales


def vectorize_moments_dict(moments_dict, rescale: Literal["auto", "manual", None] = None):
    """
    Turn the moments dict output by `eval_param_moments` into feature vectors.
    Rescales if needed.
    """
    res = []
    if rescale == "auto":
        scales_dict = get_auto_scale(moments_dict)
    elif rescale == "manual":
        scales_dict = pdfs.PARAM_SCALES
    else:
        scales_dict = {}
    for p, feats in moments_dict.items():
        scale = scales_dict.get(p, 1)
        res.extend([feat / scale for feat in feats])

    return xp.stack(res, axis=-1).astype(xp.float32)
    # should be (nsamples, ncomp, dfeat) or (nleaves_tot, dfeat) if not grouped


def compute_branch_moment_features(
    branch_idx, hps, branch_groups=None, autoscale=True, include_rate=False
):
    """
    Build a moment-based feature vector for a single RJMCMC leaf, scaled by the typical parameter
    scales set in pdfs.PARAM_SCALES or whitened automatically (autoscale=True). Also returns the
    whitening scale as an array with the same length as the # of features.

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
    `params_to_use`, each scaled by `pdfs.PARAM_SCALES[param]` so Euclidean distance is
    comparable across dimensions.
    """

    moments_dict = {}
    for p in utils.skip_dunder(hp_ordering[branch_idx].keys()):
        moments_dict.update(eval_param_moments(branch_idx, p, hps, branch_groups))
    if include_rate:
        z_bidx = branch_idx if "redshift" in hp_ordering[branch_idx] else -1
        z02 = eval_param_marginals(0.2, z_bidx, "redshift", hps)
        moments_dict["rate"] = [xp.asarray(hps[branch_idx][..., -1]) * z02]

    feats = vectorize_moments_dict(moments_dict, rescale="auto" if autoscale else "manual")
    feat_scales = get_auto_scale(moments_dict) if autoscale else pdfs.PARAM_SCALES

    return feats, np.array(
        [float(feat_scales[p]) for p, moments in moments_dict.items() for moment in moments]
    )


def hp_list_from_dict(hp_dict):

    hps = [None] * len(branch_names)
    for b_idx, bn in enumerate(branch_names):
        if bn in hp_dict:
            hps[b_idx] = hp_dict[bn]

    return hps


def hp_dict_from_list(hps):
    return {branch_names[b_idx]: hps[b_idx] for b_idx in range(len(hps))}


def label_kmeans(feats, active_mask, ncomp, rng_seed=0):
    """
    Calls sklearn's KMeans to assign ncomp labels based on features. The input features
    are computed from reversible jump samples (shape (nsamples, nleaves_max, dfeats)),
    and so active_mask masks out the inactive leaves.
    """

    nsamples, nleaves_max, dfeats = feats.shape

    X = feats.reshape(-1, dfeats)
    active_flat = active_mask.reshape(-1)
    X_act = X[active_flat]
    if X_act.shape[0] < ncomp:
        raise ValueError("Too few active leaves to fit KMeans")

    km = KMeans(n_clusters=ncomp, random_state=rng_seed, n_init="auto")
    labels_act = km.fit_predict(X_act)

    # order clusters by the column -1 in feats (should be rate)
    order = np.argsort(-km.cluster_centers_[:, -1], kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(ncomp)
    labels_act = inv_order[labels_act]

    labels_flat = np.full(X.shape[0], -1, dtype=np.int32)
    labels_flat[active_flat] = labels_act.astype(np.int32)
    labels = labels_flat.reshape(nsamples, nleaves_max)

    return labels, km.cluster_centers_[order]


def label_branch_samples_kmeans(branch_idx, samples_dict, ncomp, autoscale=False, rng_seed=0):
    """
    Computes features from samples and assigns ncomp component labels to branch branch_idx via
    KMeans for the given samples. Wraps label_kmeans function above.

    Returns
    -------
    labels : np.ndarray
        Shape (nsamples, ncomp). Each entry is in [0, ncomp-1] for active
        leaves, or -1 for inactive.
    feats : np.ndarray
        Shape (nsamples, nleaves_max, nfeats) where nfeats is usually 2*nparams.
        (Unwhitened) features of branch branch_idx computed from samples.
    scales : np.ndarray
        Shape (nfeats,). List of parameter scales used to whiten the features.
    cluster_centers: np.ndarray
        Shape (ncomp, nfeats). (Unwhitened) cluster centers.
    """

    hps = hp_list_from_dict(samples_dict)
    active = utils.leaf_active_mask(hps[branch_idx])

    feats, scales = compute_branch_moment_features(
        branch_idx, hps, autoscale=autoscale, include_rate=True
    )
    feats = utils.to_numpy(feats)

    labels, cluster_centers = label_kmeans(feats, active, ncomp, rng_seed=rng_seed)

    return labels, feats * scales, cluster_centers * scales, scales


def subpop_kdes_from_samples(samples):
    """
    Generate kdes and weights proportional to leaf occurrence rate.
    Input must be a dictionary of {bn: (-1, nleaves_max_dict[bn], branch_dims[b_idx])}

    Outputs {bn: [kde1, kde2, ...]}
    """

    kde_dict = {}
    weights_dict = {}

    samples_shaped = samples.copy()
    samples_shaped.update(
        {bn: arr.reshape((-1,) + arr.shape[-2:]) for bn, arr in samples.items() if arr.ndim > 3}
    )

    for branch_idx, branch_name in enumerate(branch_names):
        nleaves_max = nleaves_max_dict[branch_name]
        nleaves_min = nleaves_min_dict[branch_name]

        if not nleaves_max > nleaves_min:
            continue  # concerned only with rj moves

        samples_loc = samples_shaped[branch_name]
        leaf_active_mask = utils.leaf_active_mask(samples_loc)

        # rj branch
        try:
            kde_dict[branch_name] = []
            weights_dict[branch_name] = []
            labels_loc = label_branch_samples_kmeans(
                branch_idx=branch_idx,
                samples_dict=samples_shaped,
                ncomp=nleaves_max,
                rng_seed=RNG_SEED + branch_idx + 1,
            )[0]
            n_active_leaves = np.sum(leaf_active_mask)
            for label in range(nleaves_max):
                nleaves_label = np.sum(labels_loc == label)
                if nleaves_label > 100:
                    kde_dict[branch_name].append(gaussian_kde(samples_loc[labels_loc == label].T))
                    # gaussian_kde needs (ndims, nsamples)
                    weights_dict[branch_name].append(float(nleaves_label / n_active_leaves))
        except Exception as e:
            print(
                f"Failed to sort leaves for branch {branch_name}: {e}. Using a single KDE for all leaves."
            )
            kde_dict[branch_name] = [gaussian_kde(samples_loc[leaf_active_mask].T)]
            weights_dict[branch_name] = [1.0]

    return kde_dict, weights_dict


def get_bad_leaves(feats, groups, nleaves_max, min_sep=1):
    """
    For the given features of a branch, identify the per-group draws violating the constraint that
    the (scaled) leaf features (mean, width) must be at least min_sep apart at least one dimension.
    Used within the log-likelihood evaluation to set bad leaves to -inf log-likelihood.
    """

    bad = xp.zeros_like(groups, dtype=bool)
    # min_sep2 = min_sep**2

    for k in range(1, nleaves_max):  # sliding group
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


# ---------------------------
# LIKELIHOOD FUNCTION !!
# ---------------------------


def loglike(hyperparams, groups, data, injections, min_sep=1, debug=False):
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

    ngroups = groups[-1].shape[0]  # since global groups is just range(ngroups)

    data_flat = {k: data[k].ravel() for k in data}
    nevs, nsamples = data["redshift"].shape
    ninjs = injections["redshift"].shape[0]

    dNdtheta_per_group = xp.zeros(
        (ngroups, nevs * nsamples), dtype=xp.float32
    )  # evaluated on PE samples
    dNdtheta_per_group_injs = xp.zeros(
        (ngroups, ninjs), dtype=xp.float32
    )  # evaluated on injections

    # iterate over branches -- different local branches will have different models
    all_branch_moments = []  # moments dict of each branch
    for branch_idx, branch_dict in enumerate(hp_ordering[:-1]):  # all local branches
        branch_groups = groups[branch_idx]
        branch_nleaves = hyperparams[branch_idx].shape[0]

        N0 = injections["TOBS"] * hyperparams[branch_idx][:, -1]  # Tobs * R0

        dNdtheta_per_leaf = N0[:, None] * xp.ones(
            (branch_nleaves, nevs * nsamples), dtype=xp.float32
        )  # evaluated on PE samples
        dNdtheta_per_leaf_injs = N0[:, None] * xp.ones(
            (branch_nleaves, ninjs), dtype=xp.float32
        )  # evaluated on injections

        # evaluate likelihood for each event-level parameter
        branch_moments = {}
        for param_name in utils.skip_dunder(branch_dict.keys()):
            model_hp_vals_dict = unpack_hp_vals(branch_idx, param_name, hyperparams, branch_groups)

            # compute dNdtheta per leaf for each of the local models
            dNdtheta_per_leaf *= model_func_wrapper(
                branch_idx, param_name, model_hp_vals_dict, data_flat
            )
            dNdtheta_per_leaf_injs *= model_func_wrapper(
                branch_idx, param_name, model_hp_vals_dict, injections
            )

            # get moments
            branch_moments.update(
                model_moments_wrapper(branch_idx, param_name, model_hp_vals_dict)
            )

            if debug:
                print(
                    f"fraction of nan leaves after {param_name} (data)",
                    xp.sum(xp.isnan(dNdtheta_per_leaf)) / dNdtheta_per_leaf.size,
                )
                print(
                    f"fraction of nan leaves after {param_name} (injs)",
                    xp.sum(xp.isnan(dNdtheta_per_leaf_injs)) / dNdtheta_per_leaf_injs.size,
                )
                print(
                    f"fraction of zero leaves after {param_name} (data)",
                    xp.sum(dNdtheta_per_leaf == 0) / dNdtheta_per_leaf.size,
                )
                print(
                    f"fraction of zero leaves after {param_name} (injs)",
                    xp.sum(dNdtheta_per_leaf_injs == 0) / dNdtheta_per_leaf_injs.size,
                )

                # print(dNdtheta_per_leaf.dtype)
                # print(dNdtheta_per_leaf_injs.dtype)

        # combine leaves, add to dNdtheta_per_group
        scatter_add(
            dNdtheta_per_group,
            branch_groups,  # (tot_nleaves,)
            dNdtheta_per_leaf,  # (tot_nleaves, nevs x nsamples)
        )  # shape (ngroups, nevs x nsamples)
        scatter_add(
            dNdtheta_per_group_injs,
            branch_groups,  # (tot_nleaves,)
            dNdtheta_per_leaf_injs,  # (tot_nleaves, ninjs)
        )  # shape (ngroups, ninjs)
        all_branch_moments.append(branch_moments)

        if debug:
            print(
                f"fraction of groups nan after {branch_names[branch_idx]} (data)",
                xp.sum(xp.isnan(dNdtheta_per_group)) / dNdtheta_per_group.size,
            )
            print(
                f"fraction of groups nan after {branch_names[branch_idx]} (injs)",
                xp.sum(xp.isnan(dNdtheta_per_group_injs)) / dNdtheta_per_group_injs.size,
            )
            print(
                f"fraction of zero groups after {branch_names[branch_idx]} (data)",
                xp.sum(dNdtheta_per_group == 0) / dNdtheta_per_group.size,
            )
            print(
                f"fraction of zero groups after {branch_names[branch_idx]} (injs)",
                xp.sum(dNdtheta_per_group_injs == 0) / dNdtheta_per_group_injs.size,
            )

            # print(dNdtheta_per_group.dtype)
            # print(dNdtheta_per_group_injs.dtype)

    # multiply this by global dP/dtheta for global theta
    for global_param_name in utils.skip_dunder(hp_ordering[-1].keys()):
        if "model" in hp_ordering[-1][global_param_name]:
            # only want purely global parameters
            # some parameters will live in local branches w/ some global parameters -- those already have been evaluated
            model_hp_vals_dict = unpack_hp_vals(-1, global_param_name, hyperparams)

            dNdtheta_per_group *= model_func_wrapper(
                -1, global_param_name, model_hp_vals_dict, data_flat
            )
            dNdtheta_per_group_injs *= model_func_wrapper(
                -1, global_param_name, model_hp_vals_dict, injections
            )

            if debug:
                print(
                    f"fraction of zero groups after {global_param_name} (data)",
                    xp.sum(dNdtheta_per_group == 0) / dNdtheta_per_group.size,
                )
                print(
                    f"fraction of zero groups after {global_param_name} (injs)",
                    xp.sum(dNdtheta_per_group_injs == 0) / dNdtheta_per_group_injs.size,
                )

                print(dNdtheta_per_group.dtype)
                print(dNdtheta_per_group_injs.dtype)

    # divide by prior
    dNdtheta_per_group = dNdtheta_per_group / data_flat["prior"]
    dNdtheta_per_group_injs = dNdtheta_per_group_injs / injections["prior"]
    # reshape dNdtheta on data back into nevs, nsamples
    dNdtheta_per_group = dNdtheta_per_group.reshape((ngroups, nevs, nsamples))

    # if debug:
    #     print(dNdtheta_per_group.dtype)
    #     print(dNdtheta_per_group_injs.dtype)

    # Importance sampling check for PE monte carlo integral
    # see Appendix A1 of GWTC4 paper https://arxiv.org/pdf/2508.18083
    mu_ll = xp.mean(dNdtheta_per_group, axis=-1).astype(xp.float64)  # (ngroups, nevs)
    var_ll = (
        xp.sum(dNdtheta_per_group**2, axis=-1) / (nsamples - 1) - mu_ll**2
    ) / nsamples  # (ngroups, nevs)
    var_logl_pop = xp.sum(var_ll / (mu_ll**2), axis=1)

    # multiply event-level likelihoods together
    logl = xp.sum(xp.clip(xp.log(mu_ll), -INF, None), axis=1)  # shape (ngroups,)

    if debug:
        print("initial logl frac nans:", xp.count_nonzero(xp.isnan(logl)) / ngroups)
        print("initial logl frac -infs:", xp.count_nonzero(logl < -INF) / ngroups)

    # Poisson detection probability, evaluated on injections
    # this multiplies the likelihood by exp(-xi(Lambda)N(Lambda))

    # Importance sampling variance check for selection function
    ngen = injections["total_generated"]
    m = injections["w"].shape[0]
    mu_logl_sel = (1.0 / ngen) * xp.sum(injections["w"] * dNdtheta_per_group_injs, axis=-1).astype(
        xp.float64
    )
    var_logl_sel = (
        1.0
        / (m * (m - 1))
        * xp.sum(
            injections["w"] * (dNdtheta_per_group_injs * (m / ngen) - mu_logl_sel[:, None]) ** 2,
            axis=-1,
        )
    )  # eq 12 of Essick 2021, https://iopscience.iop.org/article/10.3847/2515-5172/ac2ba7

    logl -= mu_logl_sel  # subtracts Nexp(Lambda) = xi(Lambda)N(Lambda) from log-likelihood

    # print('after sel logl # nans:', xp.count_nonzero(~xp.isfinite(logl)))

    # threshold on total monte-carlo variance
    var_logl_tot = (
        var_logl_pop + var_logl_sel
    )  # equation A3 of GTWC4 paper (typo in paper - should be N(Lambda) instead of Ndet)
    logl = xp.where(var_logl_tot < 1, logl, -INF)

    if debug:
        print("pop logl var:", var_logl_pop)
        print("sel logl var:", var_logl_sel)

    # new - enforce minimum distance between components
    if min_sep:
        bad_groups = set()
        for b_idx, branch_moments_dict in enumerate(all_branch_moments):
            nleaves_max = nleaves_max_dict[branch_names[b_idx]]
            if nleaves_max > 1:
                branch_feats = vectorize_moments_dict(branch_moments_dict, rescale="manual")
                bad_groups.update(
                    get_bad_leaves(branch_feats, groups[b_idx], nleaves_max, min_sep=min_sep)
                )
        logl[xp.fromiter(bad_groups, dtype="int")] = -INF

        if debug:
            print("fraction of groups masked out by min sep:", len(bad_groups) / ngroups)

    return utils.to_numpy(xp.clip(logl, -INF, None))
    # output needs to be a numpy array


# rate post-processing


def param_branch_idx(branch_idx, param="redshift"):

    # returns -1 for global branch if global parameter

    if param not in hp_ordering[branch_idx]:
        assert param in hp_ordering[-1], (
            f"{param} not found in branch {branch_idx} or global branch"
        )
        branch_idx = -1
    return branch_idx


def filter_rates(arr):
    return np.where(~np.isfinite(arr) | (arr < 0), 0.0, arr)


def get_z_factor(samples, branch_idx=-1, z=0.2):
    if z == 0:
        return 1.0
    if type(samples) is list:
        samples = hp_dict_from_list(samples)
    elif type(samples) is not dict:
        raise ValueError(f"samples must be list or dict, not {type(samples)}")

    branch_idx = param_branch_idx(branch_idx, "redshift")
    redshift_param_dict = hp_ordering[branch_idx]["redshift"]
    if isinstance(samples, dict):
        branch_samples = samples[branch_names[branch_idx]]
    else:  # list of hyperparameters
        branch_samples = samples[branch_idx]
    z_02_factor = redshift_param_dict["model"].psi(
        z,
        **{
            hp: branch_samples[..., col_idx]
            for hp, col_idx in zip(
                redshift_param_dict["model_hp_names"],
                redshift_param_dict["col_idx"],
                strict=True,
            )
        },
    )
    return utils.to_numpy(z_02_factor)


def get_rates(branch_idx, samples, z=0.2):
    if isinstance(samples, dict):
        samples = [utils.to_numpy(samples.get(bn, None)) for bn in branch_names]
    if branch_idx == -1 or branch_idx == len(branch_names) - 1:
        # global - add all branch rates together along ncomps axis
        rate = np.nansum(
            np.concatenate([(b_samples[..., -1]) for b_samples in samples[:-1]], axis=-1),
            axis=-1,
            keepdims=True,
        )
    else:
        rate = samples[branch_idx][..., -1]
    return filter_rates(rate * get_z_factor(samples, branch_idx, z=z))


def aggregate_by_label(arr, labels, ncomps=None, comp_axis=1):
    # aggregates array of samples by label, summing along comp_axis

    res = []
    if ncomps is None:
        ncomps = np.nanmax(labels) + 1

    arr = filter_rates(utils.to_numpy(arr))
    for comp_idx in range(ncomps):
        res.append(
            np.sum(
                arr * (comp_idx == labels),
                axis=comp_axis,
            )
        )

    return np.stack(res, axis=comp_axis)


def get_branch_Ntot(branch_idx, hps, labels=None, zmax=1.5):
    if type(hps) is dict:
        hps = hp_list_from_dict(hps)
    branch_idx = param_branch_idx(branch_idx, "redshift")
    rate_model = hp_ordering[branch_idx]["redshift"]["model"]
    rate_hp_dict = unpack_hp_vals(branch_idx, "redshift", hps)
    R0 = xp.asarray(get_rates(branch_idx, hps, z=0))

    Ntot = rate_model.Ntot(R0=R0, zmax=zmax, **rate_hp_dict)
    if labels is None:
        return Ntot

    Ntot = filter_rates(utils.to_numpy(Ntot))
    # aggregate by label
    Ntot = aggregate_by_label(Ntot, labels)

    return Ntot


def get_branch_rates(branch_idx, hps, labels=None, z=0.2):
    rates = get_rates(branch_idx, hps, z=z)

    if labels is None:
        return rates
    # aggregate by label
    rates = aggregate_by_label(rates, labels)

    return rates


# def get_Ntot_by_label(hps, n_labelled_comps):
#     if not branch_names:
#         setup_data_module(model_name)

#     draw_ctx = get_draw_ctx(model_name, model_sig)
#     hps = draw_ctx["samples_input"]
#     labels = draw_ctx["labels_input"]

#     comp_label_info = get_comp_label_info(model_name)

#     # for comp_name, comp_sig, c in zip(
#     #     comp_label_info["comp_names"],
#     #     comp_label_info["comp_sigs"],
#     #     comp_label_info["colors"],
#     #     strict=True
#     # ):


#     comp_Ntot = {}
#     if "redshift" in hp_ordering[-1]:
#         Ntot = get_branch_Ntot(hps, -1)
#         R0s = np.sum(np.stack([np.nansum(hp[..., -1], axis=1) for hp in hps[:-1]], axis=0), axis=0)
#         comp_Ntot = {comp_name: Ntot for comp_name in comp_label_info["comp_names"]}


#     else:
#         for comp_name, comp_sig in zip(
#             comp_label_info["comp_names"],
#             comp_label_info["comp_sigs"],
#             strict=True
#         ):
#             pass


# ---------------------------
# 2d contour plots
# ---------------------------

ngrid = 256
data_grid = {
    "mass_1_source": np.linspace(2, 100, 2 * ngrid),
    "mass_2_source": np.linspace(2, 100, 2 * ngrid),
    "mass_ratio": np.linspace(0, 1, ngrid),
    "chi_eff": np.linspace(-1, 1, ngrid),
    "redshift": np.linspace(0, 1.5, ngrid),
}


def _extract_marg_ppd(
    x_param, model_sig, model_ppds_dict, submodel_ppds_dict=None, comp_label=None
):

    if submodel_ppds_dict:
        ppds_dict = submodel_ppds_dict
    else:
        ppds_dict = model_ppds_dict

    if "ppd" in model_ppds_dict[x_param]:  # x is a global parameter
        res = model_ppds_dict[x_param]["ppd"]
    else:
        if comp_label is None:
            res = np.sum(model_ppds_dict[x_param][model_sig], axis=0)
        elif comp_label in ppds_dict[f"{x_param}_labelled"][model_sig]:
            res = ppds_dict[f"{x_param}_labelled"][model_sig][comp_label]
        else:
            raise ValueError(f"Unrecognized component label {comp_label}")

    if x_param == "redshift":
        # for 2D we want to plot R(z) not psi(z)
        zz = data_grid["redshift"]
        res *= 4 * np.pi * Planck15.differential_comoving_volume(zz).value / 1e9 / (1 + zz)

    return res


def _compute_mass_ppds(
    x_param, y_param, draw_ctxs, model_ppds_dict, submodel_ppds_dict=None, model_sig=0
):

    if submodel_ppds_dict:
        model_draw_ctxs = draw_ctxs["submodel"]
    else:
        model_draw_ctxs = draw_ctxs["model"]

    xx = model_ppds_dict[x_param]["x"]
    yy = model_ppds_dict[y_param]["x"]

    hyperparams = model_draw_ctxs[model_sig]["samples_input"]

    xx_, yy_ = np.meshgrid(xx, yy, indexing="ij")
    mass_data = {x_param: xp.asarray(xx_.ravel()), y_param: xp.asarray(yy_.ravel())}
    if "mass_2_source" not in mass_data:
        mass_data["mass_2_source"] = mass_data["mass_1_source"] * mass_data["mass_ratio"]
    elif "mass_ratio" not in mass_data:
        mass_data["mass_ratio"] = mass_data["mass_2_source"] / mass_data["mass_1_source"]
    else:
        raise ValueError(
            f"Unrecognized mass parameter combination {x_param} and {y_param}. One parameter must be `mass_1_source` and the other must be `mass_ratio` or `mass_2_source`."
        )

    mass_ppds = {}
    for branch_idx, bn in enumerate(branch_names[:-1]):
        branch_R02 = get_rates(branch_idx, hyperparams)  # (nsamples, ncomps)
        if "mass" in hp_ordering[branch_idx]:
            if "model" in hp_ordering[branch_idx]["mass"]:  # primary model
                branch_xy_ppd = (
                    utils.to_numpy(eval_param_model(mass_data, branch_idx, "mass", hyperparams))
                    * branch_R02[..., None]
                )  # this is R(m1, q)
                if "mass_2_source" in (x_param, y_param):  # convert to p(m1, m2)
                    branch_xy_ppd = branch_xy_ppd / utils.to_numpy(mass_data["mass_1_source"])
                branch_xy_ppd = np.nan_to_num(branch_xy_ppd, nan=0.0, posinf=0.0, neginf=0.0)
                mass_ppds[bn] = branch_xy_ppd.reshape(
                    branch_xy_ppd.shape[:-1] + (xx.size, yy.size)
                )
                # should be (nsamples, ncomp, ngrid, ngrid)

    return mass_ppds


def plot_param_2D_contours_and_marginals(
    x_param,
    y_param,
    model_sig,
    draw_ctxs,
    model_ppds_dict,
    submodel_ppds_dict=None,
    color="cornflowerblue",
    axes=None,
    comp_label=None,
    savefig=None,
    figpath=None,
    mass_ppds=None,
    return_pxy_mean=False,
    rasterize=True,
    comp_name_dict=None,
    idx_order=None,
    **plot_kwargs,
):

    if submodel_ppds_dict:
        ppds_dict = submodel_ppds_dict
        model_draw_ctxs = draw_ctxs["submodel"]
    else:
        ppds_dict = model_ppds_dict
        model_draw_ctxs = draw_ctxs["model"]

    rj_branches = any(
        [
            nmax > nmin
            for nmax, nmin in zip(
                nleaves_max_dict.values(), nleaves_min_dict.values(), strict=True
            )
        ]
    )

    if figpath is None:
        figpath = os.getcwd()

    if not submodel_ppds_dict:
        if rj_branches:
            assert model_sig in ppds_dict["model_sigs"], (
                f"model_name must be one of {ppds_dict['model_sigs']}"
            )
        else:
            assert model_sig == 0
    assert x_param in model_ppds_dict
    assert y_param in model_ppds_dict

    x_global = "ppd" in model_ppds_dict[x_param]
    y_global = "ppd" in model_ppds_dict[y_param]

    draw_ctx = model_draw_ctxs[model_sig]
    if rj_branches:
        comp_labels = model_ppds_dict["comp_labels"]["comp_names"]
    else:
        comp_labels = branch_names[:-1]

    comp_legend_labels = (
        [comp_name_dict.get(x, x) for x in comp_labels] if comp_name_dict else comp_labels
    )

    comp_rates = draw_ctx["R02_labelled_draws"].T  # (nsamples, ncomps) -> (ncomps, nsamples)
    model_sig_safe = utils.get_safe_fn(model_sig)  # for saving plots
    if submodel_ppds_dict:
        if model_sig in draw_ctxs["submodel"].keys():
            model_sig_safe += "_submodel"

    if x_param.startswith("mass") and y_param.startswith("mass") and mass_ppds is None:
        # pre-compute mass ppds to save time in recursive calls
        mass_ppds = _compute_mass_ppds(
            x_param,
            y_param,
            model_ppds_dict,
            submodel_ppds_dict=submodel_ppds_dict,
            model_sig=model_sig,
        )

    if (comp_label == "all") and not (x_global and y_global):  # plot all together, in one plot
        # this is just a recursive call
        ncomps = len(comp_labels)

        # weight each components by alpha ~ sqrt(braching frac.) -- helps with visibility
        branching_fracs = comp_rates / np.nansum(comp_rates, axis=0, keepdims=True)
        comp_alphas = np.sqrt(np.nanmean(branching_fracs, axis=1))
        comp_alphas /= np.amax(comp_alphas)
        comp_alphas *= plot_kwargs.get("alpha", 1.0)

        if idx_order is None:
            idx_order = np.argsort(comp_alphas)  # plot smallest components first

        if isinstance(color, (list, sns.palettes._ColorPalette)) and len(color) >= ncomps:
            colors = color[:ncomps]
        else:
            # get color palette from ppds_dict
            colors = model_ppds_dict["comp_labels"]["colors"]

        h, l = [], []
        p_xy_mean_tot = 0.0

        for idx in idx_order:

            comp_label, color, alpha = comp_labels[idx], colors[idx], comp_alphas[idx]
            axes, p_xy_mean = plot_param_2D_contours_and_marginals(
                x_param,
                y_param,
                model_sig,
                draw_ctxs,
                model_ppds_dict,
                submodel_ppds_dict,
                color=color,
                savefig=None,
                axes=axes,
                comp_label=comp_label,
                mass_ppds=mass_ppds,
                return_pxy_mean=True,
                **plot_kwargs | dict(alpha=alpha),
            )
            p_xy_mean_tot = p_xy_mean_tot + p_xy_mean
            if not np.all(np.isclose(p_xy_mean, 0.0, atol=1e-10)):
                h.append(
                    Patch(
                        facecolor=to_rgba(color, alpha=0.7 * alpha),
                        edgecolor=color,
                        label=plot.textsc_ify(comp_legend_labels[idx]),
                    )
                )
                l.append(plot.textsc_ify(comp_legend_labels[idx]))

        l, h = utils.sort_lists(l, h)
        axes[0].legend(
            handles=h, labels=l, loc="lower center", bbox_to_anchor=(0.5, 0.888), ncol=ncomps
        )

        if savefig:
            if not isinstance(savefig, str):
                savefig = f"contours2d_{x_param}_{y_param}_{model_sig_safe}_labelled.pdf"
            axes[0].savefig(os.path.join(figpath, savefig))
            print(f"Saved {savefig}")
        if return_pxy_mean:
            return axes, p_xy_mean
        return axes

    # want to plot marginals on the side
    p_x_marg = _extract_marg_ppd(
        x_param,
        model_sig,
        model_ppds_dict,
        submodel_ppds_dict=submodel_ppds_dict,
        comp_label=comp_label,
    )
    p_y_marg = _extract_marg_ppd(
        y_param,
        model_sig,
        model_ppds_dict,
        submodel_ppds_dict=submodel_ppds_dict,
        comp_label=comp_label,
    )

    # get x/y grid from dict
    xx = model_ppds_dict[x_param]["x"]
    yy = model_ppds_dict[y_param]["x"]

    if x_param.startswith("mass") and y_param.startswith("mass"):
        if comp_label is None:  # just want total
            p_xy = [
                np.nansum(b_ppd, axis=1) for b_ppd in mass_ppds.values()
            ]  # add components together
            p_xy = np.sum(np.stack(p_xy), axis=0)  # add branches together
        else:  # have to extract component from labels
            comp_idx = comp_labels.index(comp_label)
            bn, b_label_idx = model_ppds_dict["comp_labels"]["comp_sigs"][comp_idx]
            labels = draw_ctx["labels_input"][branch_names.index(bn)]
            mask = (labels == b_label_idx)[..., None, None]
            p_xy = np.nansum(mass_ppds[bn] * mask, axis=1)
    else:
        if (comp_label is None) and not (x_global and y_global) and rj_branches:
            p_xy = 0.0
            for comp_label_, comp_rate in zip(comp_labels, comp_rates, strict=True):
                comp_x_ppd = _extract_marg_ppd(
                    x_param,
                    model_sig,
                    model_ppds_dict,
                    submodel_ppds_dict=submodel_ppds_dict,
                    comp_label=comp_label_,
                )
                comp_y_ppd = _extract_marg_ppd(
                    y_param,
                    model_sig,
                    model_ppds_dict,
                    submodel_ppds_dict=submodel_ppds_dict,
                    comp_label=comp_label_,
                )
                comp_xy_ppd = np.where(
                    comp_rate[:, None, None] > 0,
                    np.expand_dims(comp_x_ppd, 2)
                    * np.expand_dims(comp_y_ppd, 1)
                    / comp_rate[:, None, None],
                    0.0,
                )
                p_xy = p_xy + comp_xy_ppd
        else:
            R02_tot = draw_ctx["R02_tot_draws"]
            p_xy = (
                np.expand_dims(p_x_marg, 2) * np.expand_dims(p_y_marg, 1) / R02_tot[:, None, None]
            )

    p_xy_mean = np.nanmean(p_xy, axis=0)
    if np.all(np.isclose(p_xy_mean, 0.0, atol=1e-10)):
        print(
            f"WARNING: PDF is zero for {x_param} vs {y_param} for model `{model_sig}` and component `{comp_label}`."
        )
    else:
        axes = plot.plot_2D_contours_and_marginals(
            xx,
            yy,
            p_x_marg,
            p_y_marg,
            p_xy_mean,
            color=color,
            axes=axes,
            rasterize_ppds=rasterize,
            xlabel=plot.texnames.get(x_param, x_param),
            ylabel=plot.texnames.get(y_param, y_param),
            **plot_kwargs,
        )

        # plot m1m2 triangle if needed
        ax_joint = axes[1]
        if len(ax_joint.patches) == 0:  # no triangle yet
            if x_param == "mass_1_source" and y_param == "mass_2_source":
                ax_joint = plot.shade_triangle(ax_joint)
            elif x_param == "mass_2_source" and y_param == "mass_1_source":
                ax_joint = plot.shade_triangle(ax_joint, plane="lower half")

    if savefig:
        if not isinstance(savefig, str):
            fn_parts = ["contours2d", x_param, y_param]
            if model_sig:
                fn_parts.append(model_sig_safe)
            fn_parts.append("tot" if comp_label is None else utils.get_safe_fn(comp_label))
            savefig = "_".join(fn_parts) + ".pdf"
        axes[0].savefig(os.path.join(figpath, savefig))
        print(f"Saved {savefig}")

    if return_pxy_mean:
        return axes, p_xy_mean

    return axes
