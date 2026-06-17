import os
import re
from pathlib import Path

import numpy as np
from eryn.backends import Backend
from eryn.ensemble import EnsembleSampler
from eryn.utils.utility import get_integrated_act
from scipy.optimize import curve_fit

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "rjpop"))
from xp import xp

### MISC UTILS


def check_min_separation(X, min_sep, xp=xp):
    """
    Checks for an input array of shape (..., M, N) that the minimum separation between the M points
    in N-dimensional space is at least min_sep.

    Returns: (...,) boolean array
    """
    X_ = xp.nan_to_num(xp.asarray(X), nan=xp.inf)

    M = X.shape[-2]
    i, j = xp.triu_indices(M, k=1)

    diff = X_[..., i, :] - X_[..., j, :]  # (..., P, N)
    d2 = xp.sum(diff**2, axis=-1)  # (..., P)
    d2[xp.isnan(d2)] = xp.inf

    return xp.all(d2 >= min_sep**2, axis=-1)


### PATH / FILE / DATA STUFF


def sort_lists(*lists, reverse=False, strict=True):
    """Sorts multiple lists by the first list."""
    return [
        list(x) for x in zip(*sorted(zip(*lists, strict=strict), reverse=reverse), strict=True)
    ]


def to_numpy(arr):
    """Convert CuPy->NumPy if needed; otherwise return a NumPy view/copy."""
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def intersection(*lists):
    return list(set(lists[0]).intersection(*lists))


def print_to(fname, text):
    with open(fname, "a") as f:
        f.write(text + "\n")


def unique_path(path: str | Path) -> Path:
    """
    If `path` exists, return `stem_{i}.suffix` for the first i that doesn't exist.
    Example: backend.h5 -> backend_0.h5 -> backend_1.h5 -> ...
    """
    p = Path(path)
    if not p.exists():
        return p

    i = 0
    while True:
        candidate = p.with_name(f"{p.stem}_{i}{p.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def skip_dunder(iter):
    for x in iter:
        if type(x) is str:
            if not (x.startswith("__") or x.endswith("__")):
                yield x

def pad(arr):
    arr_ = xp.asarray(arr)
    if arr_.size <= 1:
        return arr_
    else:
        return arr_[..., None]

def recursive_pad(datadict):
    """
    Adds one dimension to each array of a dictionary of arrays and subdictionaries of arrays,
    to be broadcastable with another 1D array.
    """
    res = {}
    for k, v in datadict.items():
        if isinstance(v, dict):
            res[k] = recursive_pad(v)
        else:
            res[k] = pad(v)

    return res


def recursive_get(datadict, key, return_superkey=False, superkey=None):
    """
    Recursively collects all values associated with ``key`` in ``datadict`` and
    its nested dictionaries. When ``return_superkey`` is True, also returns the
    immediate dictionary key each value came from.
    """
    res = []
    superkeys = []

    for k, v in datadict.items():
        if k == key:
            values = v if isinstance(v, list) else [v]
            res.extend(values)
            superkeys.extend([superkey] * len(values))
        elif isinstance(v, dict):
            child_vals, child_keys = recursive_get(
                v, key, return_superkey=True, superkey=k
            )
            res.extend(child_vals)
            superkeys.extend(child_keys)

    if return_superkey:
        return res, superkeys
    return res


def get_safe_fn(name):
    return "_".join(re.split(r"\W+", str(name)))


ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def EnsembleSamplerWrapper(*args, **kwargs):
    try:
        ensemble = EnsembleSampler(*args, **kwargs)
    except Exception as e:
        print(f"Can't use existing backend: {e}")
        # start a new backend
        backend_path = kwargs["backend"].filename
        old_backend_path = unique_path(backend_path)
        print(f"Moving old backend to {old_backend_path}")
        os.rename(backend_path, old_backend_path)
        ensemble = EnsembleSampler(*args, **kwargs)
    return ensemble


### POST-PROCESSING


def leaf_active_mask(samples_loc):
    """
    Mask of active RJ leaves: finite and strictly positive rate column.
    Assumes that the last column is the rate.
    """
    rate = samples_loc[..., -1]
    return np.isfinite(rate) & (rate > 0)


def get_integrated_act_wrap(samples, average=True, fast=False):
    """
    Compute the integrated auto-correlation time. For RJ moves, passes the number of leaves and total rate to eryn's `get_integrated_act` to compute the autocorrelation time.

    Args:
        samples (dict): The samples to compute the auto-correlation time for.
        average (bool): Whether to average the auto-correlation time across all dimensions. Default is True.
        fast (bool): Whether to use the fast method for computing the auto-correlation time. Default is False.
    Returns:
        tau (array): The integrated auto-correlation time for the samples.
    """
    tau = {}
    for name in samples.keys():
        chain = samples[name]
        nsteps, ntemps, nw, nl, ndims = chain.shape
        if np.any(np.isnan(chain)) and nl > 1:
            # RJ branch - use number of active leaves + total rate to calculate autocorrelation time
            active_mask = leaf_active_mask(chain)
            n_active_leaves = np.count_nonzero(active_mask, axis=-1, keepdims=True)
            tot_rate = np.sum(np.where(active_mask, chain[..., -1], 0.0), axis=-1, keepdims=True)
            tau_rj = get_integrated_act(
                np.concatenate((n_active_leaves, tot_rate), axis=-1), average=average, fast=fast
            )
            tau[name] = tau_rj[np.isfinite(tau_rj)]

        else:
            # non RJ branch
            chain = chain.reshape(nsteps, ntemps, nw, nl * ndims)
            tau[name] = get_integrated_act(chain, average=average, fast=fast)

    return tau


def get_discard_from_chain(logP):
    """
    Compute the discard value for a given log posterior chain by fitting a simple linear + flat
    model of convergence to the chain. Assumes logP has shape (nsamples, nwalkers).
    """

    def simple_logP_model(x, m, x_break, y_break):
        return np.where(x < x_break, y_break + m * (x - x_break), y_break)

    nsamps = logP.shape[0]

    x = np.arange(nsamps, dtype=float)
    y = np.nansum(logP, axis=-1) / np.count_nonzero(
        np.isfinite(logP), axis=-1
    )  # average across walkers

    x_halfway = int(len(y) // 2)
    p0 = (max(2e-3, (y[x_halfway] - y[0]) / x_halfway), x_halfway, y[x_halfway])
    bounds = ([1e-3, 0.0, np.amin(logP)], [np.inf, len(y) - 1, np.amax(logP)])

    m, x_break, y_break = curve_fit(simple_logP_model, x, y, p0=p0, bounds=bounds)[0]
    discard = int(x_break + 0.2 * (nsamps - x_break))

    return discard


def get_thin_from_chain(chain, return_tau=False):
    chain = {
        name: value[:, :1] for name, value in chain.items()
    }  # get minimum temp but keep the dims
    tau = get_integrated_act_wrap(chain)
    tau_values = [np.nanmax(arr) for _, arr in tau.items() if np.isfinite(np.nanmax(arr))]
    thin = int(np.ceil(max(tau_values) * 0.5)) if tau_values else 1

    if return_tau:
        return thin, tau
    return thin


def get_move_acceptance_fraction(moves, temp_index=None):
    """
    Gets the mean acceptance fraction from a list of moves.
    """
    if temp_index is None:
        temp_index = slice(None)
    n_acc, n_prop = [], []
    for move in moves:
        n_acc.append(move.accepted[temp_index])
        n_prop.append(move.num_proposals)
    acc_frac = sum(n_acc) / sum(n_prop)
    return np.mean(acc_frac), np.std(acc_frac)


# These are wrappers to get diagnostics without requiring explicitly referring to a sampler.
# Code is pretty much lifted directly from eryn.ensemble.EnsembleSampler


def acceptance_fraction(obj: EnsembleSampler | Backend, temp_index=None) -> tuple[float, float]:
    assert isinstance(obj, (EnsembleSampler, Backend)), "obj must be an EnsembleSampler or Backend"
    backend = obj.backend if isinstance(obj, EnsembleSampler) else obj
    frac = backend.accepted[temp_index] / float(backend.iteration)
    return np.mean(frac), np.std(frac)


def rj_acceptance_fraction(obj: EnsembleSampler | Backend, temp_index=None) -> tuple[float, float]:
    assert isinstance(obj, (EnsembleSampler, Backend)), "obj must be an EnsembleSampler or Backend"
    backend = obj.backend if isinstance(obj, EnsembleSampler) else obj
    if not backend.rj:
        print("No reversible jump moves in the backend.")
        return None
    frac = backend.rj_accepted / float(backend.iteration)
    return np.mean(frac), np.std(frac)


def swap_acceptance_fraction(
    obj: EnsembleSampler | Backend, temp_index=None
) -> tuple[float, float]:
    assert isinstance(obj, (EnsembleSampler, Backend)), "obj must be an EnsembleSampler or Backend"
    backend = obj.backend if isinstance(obj, EnsembleSampler) else obj
    return backend.swaps_accepted / float(backend.iteration * backend.nwalkers)
