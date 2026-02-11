from typing import Literal

import numpy as np
import utils
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
        else:
            model_hp_vals_dict[hp] = hp_vals

    if "subparams" in param_dict:  # joint model - evaluate subdictionaries
        for subparam in param_dict["subparams"]:
            model_hp_vals_dict[f"{subparam}_kwargs"] = unpack_hp_vals(
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

    x = xp.asarray(x_arr)  # numpy input ok
    if param in hp_ordering[branch_idx]:
        if param == "redshift":
            model_hp_vals_dict = utils.recursive_pad(
                unpack_hp_vals(branch_idx, param, hyperparams)
            )
            return hp_ordering[branch_idx][param]["model"].psi(x, **model_hp_vals_dict)

        return eval_param_model({param: x}, branch_idx, param, hyperparams)

    # else: get marginal from a joint parent model
    parent_param = get_parent_param(branch_idx, param)
    parent_param_dict = hp_ordering[branch_idx][parent_param]

    parent_model_hp_vals_dict = unpack_hp_vals(
        branch_idx, parent_param, hyperparams, param_dict=parent_param_dict
    )
    parent_model_hp_vals_dict = utils.recursive_pad(parent_model_hp_vals_dict)
    parent_model = hp_ordering[branch_idx][parent_param]["model"]
    return parent_model.get_marginal_pdf(x, param, **parent_model_hp_vals_dict)


def vectorize_moments_dict(moments_dict, rescale: Literal["auto", "manual", None] = None):
    """
    Turn the moments dict output by `eval_param_moments` into feature vectors.
    Rescales if needed.
    """
    res = []
    for p, (mu, sig) in moments_dict.items():
        scale = 1
        if rescale == "auto":
            scale = xp.sqrt(xp.nanvar(mu) + xp.nanvar(sig) + xp.nanmean(sig) ** 2)
        elif rescale == "manual":
            scale = utils.PARAM_SCALES[p]
        res.extend([mu / scale, sig / scale])

    return xp.stack(res, axis=-1).astype(xp.float32)
    # should be (nsamples, ncomp, dfeat) or (nleaves_tot, dfeat) if not grouped


def compute_branch_moment_features(branch_idx, hps, branch_groups=None, autoscale=True):
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
    rescale = "auto" if autoscale else "manual"
    feats = vectorize_moments_dict(moments_dict, rescale=rescale)

    return feats, feat_params


def label_samples_kmeans(
    branch_idx, samples_loc, samples_global, ncomp, autoscale=False, rng_seed=0
):
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

    feats, feat_params = compute_branch_moment_features(branch_idx, hps, autoscale=autoscale)
    feats = utils.to_numpy(feats)
    dfeats = feats.shape[-1]

    X = feats.reshape(-1, dfeats)
    active_flat = active.reshape(-1)
    X_act = X[active_flat]
    if X_act.shape[0] < ncomp:
        raise ValueError("Too few active leaves to fit KMeans")

    km = KMeans(n_clusters=ncomp, random_state=rng_seed, n_init="auto")
    labels_act = km.fit_predict(X_act)

    # order clusters by decreasing cluster size (number of assigned active points)
    counts = np.bincount(labels_act, minlength=ncomp)
    order = np.argsort(-counts, kind="stable")
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(ncomp)
    labels_act = inv_order[labels_act]

    labels_flat = np.full(X.shape[0], -1, dtype=np.int32)
    labels_flat[active_flat] = labels_act.astype(np.int32)
    labels = labels_flat.reshape(nsamples, nleaves_max)

    return labels, feats, list(feat_params)


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
            labels_loc = label_samples_kmeans(
                branch_idx=branch_idx,
                samples_loc=samples_loc,
                samples_global=samples_shaped["global"],
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
                branch_feats = vectorize_moments_dict(branch_moments_dict, rescale=True)
                # is this the fastest way to do this?
                bad_groups.update(
                    get_bad_leaves(branch_feats, groups[b_idx], nleaves_max, min_sep=min_sep)
                )
        logl[xp.fromiter(bad_groups, dtype="int")] = -INF

        if debug:
            print("fraction of groups masked out by min sep:", len(bad_groups) / ngroups)

    return utils.to_numpy(xp.clip(logl, -INF, None))
    # output needs to be a numpy array
