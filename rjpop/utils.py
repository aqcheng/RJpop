import os
import sys
from pathlib import Path

import numpy as np
from astropy.cosmology import Planck15
from scipy.optimize import curve_fit

sys.path.append("/work/aqc/lib/Eryn/src")
import re

import pdfs
from eryn.backends import Backend
from eryn.ensemble import EnsembleSampler
from eryn.utils.utility import get_integrated_act
from xp import xp

### HARD CODED MODEL DEFINITIONS

# dictionary of models and their hyperparameters -- these are manually hard coded in
XMAX_FIX = 300.0
MODELS = {
    "mass": {
        "m1_q": {"model": pdfs.m1_q_model, "param_latex": {}},
        "gaussian_copula": {
            "model": pdfs.gaussian_copula_mass_model,
            "param_latex": {"rho": r"$\rho$"},
        },
        "sym_gaussian_copula": {
            "model": pdfs.sym_gaussian_copula_mass_model,
            "param_latex": {"rho": r"$\rho$"},
        },
    },
    "mass_1_source": {  # for p(m1) or p(m2)
        "param_latex": {  # shared between all models of the parameter
            "xmin": r"$m_{\min}$",
            "xmax": r"$m_{\max}$",
        },
        "skew-t": {
            "model": pdfs.jf_skew_t(),
            "param_latex": {
                "logalpha": r"$\log_{10}\alpha$",
                "logkappa": r"$\log_{10}\kappa$",
                "loc": r"$\mu_m$",
                "scale": r"$\sigma_m$",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
        "PLS": {
            "model": pdfs.smoothed_powerlaw(),
            "param_latex": {
                "alpha": r"$\alpha$",
                "p": r"$p_m$",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
        "PLS_LVK": {
            "model": pdfs.LVK_Plancktaper_powerlaw(),
            "param_latex": {
                "alpha": r"$\alpha$",
                "delta": r"$\delta_m$",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
        "gauss": {
            "model": pdfs.gaussian(),
            "param_latex": {
                "loc": r"$\mu_p$",
                "scale": r"$\sigma_p$",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
    },
    "mass_ratio": {
        "PL": {
            "model": pdfs.powerlaw(),
            "param_latex": {
                "beta": r"$\beta$",
            },
            "params_fix": {
                "xmax": 1.0,
            },
        },
        "gauss": {
            "model": pdfs.gaussian(),
            "param_latex": {
                "loc": r"$\mu_{q}$",
                "scale": r"$\sigma_{q}$",
            },
            "params_fix": {"xmax": 1.0},
        },
    },
    "chi_eff": {
        "gen_gauss": {
            "model": pdfs.gen_gaussian(),
            "param_latex": {
                "beta": r"$\beta_{\chi}$",
                "loc": r"$\mu_{\chi}$",
                "scale": r"$\sigma_{\chi}$",
            },
            "params_fix": {"xmin": -1.0, "xmax": 1.0},
        },
        "gauss": {
            "model": pdfs.gaussian(),
            "param_latex": {
                "loc": r"$\mu_{\chi}$",
                "scale": r"$\sigma_{\chi}$",
            },
            "params_fix": {"xmin": -1.0, "xmax": 1.0},
        },
    },
    "redshift": {
        "MD": {
            "model": pdfs.MD_rate(cosmo=Planck15),
            "param_latex": {
                "gamma": r"$\gamma$",
                "kappa": r"$\kappa$",
                "zp": r"$z_p$",
            },
            "params_fix": {},
        },
        "PL": {
            "model": pdfs.PL_rate(cosmo=Planck15),
            "param_latex": {"gamma": r"$\gamma$"},
            "params_fix": {},
        },
    },
}
PARAM_SCALES = {  # characteristic scales for each parameter
    "mass_1_source": 8.0,
    "mass_2_source": 8.0,
    "mass_ratio": 0.25,
    "chi_eff": 0.04,
}
RATE_FACTOR = 10.0

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
    return [list(x) for x in zip(*sorted(zip(*lists, strict=strict), reverse=reverse), strict=True)]


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
            v_ = xp.asarray(v)
            if v_.size > 1:
                res[k] = v_[..., None]
            else:
                res[k] = v_

    return res


def recursive_get(datadict, key):
    """
    Recursively gets and concatenates all dict[key] for all subdictionaries of datadict, including itself.
    Assumes that dict[key] are lists.
    """
    res = []
    for k, v in datadict.items():
        if k == key:
            res.extend(v)
        elif type(v) is dict:
            res.extend(recursive_get(v, key))
    return res


def get_safe_fn(name):
    return "_".join(re.split(r"\W+", name))


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
    Compute the integrated auto-correlation time for RJ moves.

    Args:
        samples (dict): The samples to compute the auto-correlation time for.
        nleaves (dict): The number of active leaves. This is only used for RJ branches. Default is None.
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
            # RJ branch - use number of active leaves to calculate autocorrelation time
            n_active_leaves = np.count_nonzero(leaf_active_mask(chain), axis=-1, keepdims=True)
            tau_rj = get_integrated_act(n_active_leaves, average=average, fast=fast)
            if np.isfinite(tau_rj):
                tau[name] = tau_rj

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


def acceptance_fraction(obj: EnsembleSampler | Backend, temp_index=None) -> float:
    assert isinstance(obj, (EnsembleSampler, Backend)), "obj must be an EnsembleSampler or Backend"
    backend = obj.backend if isinstance(obj, EnsembleSampler) else obj
    frac = backend.accepted[temp_index] / float(backend.iteration)
    return np.mean(frac), np.std(frac)


def rj_acceptance_fraction(obj: EnsembleSampler | Backend, temp_index=None) -> float:
    assert isinstance(obj, (EnsembleSampler, Backend)), "obj must be an EnsembleSampler or Backend"
    backend = obj.backend if isinstance(obj, EnsembleSampler) else obj
    if not backend.rj:
        print("No reversible jump moves in the backend.")
        return None
    frac = backend.rj_accepted / float(backend.iteration)
    return np.mean(frac), np.std(frac)


def swap_acceptance_fraction(obj: EnsembleSampler | Backend, temp_index=None) -> float:
    assert isinstance(obj, (EnsembleSampler, Backend)), "obj must be an EnsembleSampler or Backend"
    backend = obj.backend if isinstance(obj, EnsembleSampler) else obj
    return backend.swaps_accepted / float(backend.iteration * backend.nwalkers)
