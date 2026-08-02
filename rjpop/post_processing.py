# helper functions for post-processing plots

import data
import matplotlib.pyplot as plt
import utils
from PIL import Image

import json
import os
from glob import glob

import numpy as np
from xp import xp
from scipy.stats import truncnorm as _truncnorm

from load_config import load_config

cfg = load_config()

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = cfg.get("RJpop_out_path")

if DATADIR is None:
    raise ValueError("RJpop_out_path must be specified in config.json")

FIGPATH = f"{DATADIR}/_figures"
# FIGPATH_DARK = f"{DATADIR}/_figures/dark"
FIGPATH_DARK = None  # don't save dark mode images for now

# ── data helpers ─────────────────────────────────────────────────────────────


def load_megadict(model_name):
    path = os.path.join(DATADIR, model_name, "data", "samples_and_labels.npy")
    return np.load(path, allow_pickle=True).item()


def load_hpinfo(model_name):
    path = os.path.join(DATADIR, model_name, "hyperparameter_ordering_metainfo.json")
    with open(path) as f:
        return json.load(f)


def load_PE_samples_and_names(use_xp=True):
    PE_samples_fn = cfg.get("pe_samples_path", None)
    if PE_samples_fn is None:
        raise ValueError("`pe_samples_path` must be specified in config.json")
    elif not os.path.exists(PE_samples_fn):
        raise FileNotFoundError(f"PE samples file not found: {PE_samples_fn}")

    with np.load(PE_samples_fn) as f:
        if use_xp:
            PE_samples = {
                k: xp.asarray(v, dtype=xp.float32)
                for k, v in f.items()
                if k not in ("chirp_mass_source", "event_names")
            }
        else:
            PE_samples = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in f.items()
                if k not in ("chirp_mass_source", "event_names")
            }
        event_names = np.asarray(f["event_names"])

    return PE_samples, event_names


def get_param_col(hpinfo, branch_idx, *path_keys, param_name):
    """Navigate hpinfo[branch_idx] via path_keys and return col_idx of param_name."""
    d = hpinfo[branch_idx]
    for key in path_keys:
        d = d[key]
    return d["col_idx"][d["model_hp_names"].index(param_name)]


def sig_inds(mega, model_sig, submodel=False):
    """Indices into the full sample array for a given model/submodel signature."""
    if submodel:
        names = mega["submodel_sig_names"]
        inds = mega["submodel_inds"]
    else:
        names = mega["sig_names"]
        inds = mega["model_inds"]
    return inds[names.index(model_sig)]


def leaf_samples(mega, branch, label_idx, model_sig, submodel=False, cols=None):
    """
    Flatten posterior samples for leaves in `branch` with `label_idx`,
    restricted to `model_sig`.

    cols : None → all dims (returns 2-D); int → 1-D; list/tuple → 2-D subset.
    """
    idx = sig_inds(mega, model_sig, submodel)
    s = mega["samples"][branch][idx]  # (n, nleaves, ndims)
    lbl = mega["labels"][branch][idx]  # (n, nleaves)
    mask = lbl == label_idx  # bool (n, nleaves)

    if cols is None:
        return s[mask]  # (n_match, ndims)
    if isinstance(cols, (list, tuple)):
        return s[mask][:, list(cols)]  # (n_match, len(cols))
    return s[..., cols][mask]  # (n_match,)


def R0_per_sample(mega, branch, label_idx, model_sig, submodel=False):
    """Sum R0 of matching leaves per posterior sample. Returns (n_sig,) array."""
    idx = sig_inds(mega, model_sig, submodel)
    s = mega["samples"][branch][idx]  # (n, nleaves, ndims)
    lbl = mega["labels"][branch][idx]  # (n, nleaves)
    R0 = np.where(np.isfinite(s[..., -1]) & (s[..., -1] > 0), s[..., -1], 0.0)
    return np.nansum(R0 * (lbl == label_idx), axis=1)  # (n,)


def total_R0_per_sample(mega, local_branches, model_sig, submodel=False):
    """Sum R0 across all leaves in all local branches for each sample. Returns (n_sig,)."""
    idx = sig_inds(mega, model_sig, submodel)
    tot = np.zeros(len(idx))
    for bn in local_branches:
        s = mega["samples"][bn][idx]  # (n, nleaves, ndims)
        R0 = np.where(np.isfinite(s[..., -1]) & (s[..., -1] > 0), s[..., -1], 0.0)
        tot += np.nansum(R0, axis=1)
    return tot


def global_param_samples(mega, model_sig, col, submodel=False):
    """Return 1-D samples from the global branch (single leaf) for a given model signature."""
    idx = sig_inds(mega, model_sig, submodel)
    return mega["samples"]["global"][idx, 0, col]


# ── statistics / formatting helpers ─────────────────────────────────────────


def trunc_gauss_ppf(q, mu, sigma, xmin=-1.0, xmax=1.0):
    """Truncated-Gaussian PPF for array inputs mu, sigma."""
    a = (xmin - mu) / sigma
    b = (xmax - mu) / sigma
    return _truncnorm.ppf(q, a, b, loc=mu, scale=sigma)


def chieff_pos_frac(mu, sigma, xmin=-1.0, xmax=1.0):
    """P(chi_eff > 0) for a truncated Gaussian on [xmin, xmax], array inputs mu, sigma."""
    a = (xmin - mu) / sigma
    b = (xmax - mu) / sigma
    return _truncnorm.sf(0.0, a, b, loc=mu, scale=sigma)


def _nz(val, fmt):
    """Format val; force negative-zero strings like '-0.00' to '0.00'."""
    s = format(val, fmt)
    # check if the value is negative-zero after rounding (e.g. -0.003 → '-0.00')
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]  # strip the minus
    return s


def fmt_ci(samples, lo=5, hi=95, fmt=".1f"):
    """LaTeX string  med^{+hi-med}_{-med-lo}  for 90 % CI."""
    s = np.asarray(samples).ravel()
    s = s[np.isfinite(s)]
    low, med, high = np.percentile(s, [lo, 50, hi])
    m = _nz(med, fmt)
    dh = _nz(high - med, fmt)
    dl = _nz(med - low, fmt)
    return rf"{m}^{{+{dh}}}_{{-{dl}}}"


def fmt_pct(prob, fmt=".1f"):
    """LaTeX string for a probability (0–1) rendered as a percentage."""
    return rf"{prob * 100:{fmt}}\%"


# ── gamma confidence helper ───────────────────────────────────────────────────


def paired_gamma_confidence(
    mega, model_sig, gamma_col, label_a=0, label_b=1, ncomps=None
):
    """
    Posterior probability that gamma[label_a] > gamma[label_b].
    Filters to samples containing exactly one leaf of each required label.
    """
    idx = sig_inds(mega, model_sig, submodel=False)
    gamma = mega["samples"]["local"][idx][..., gamma_col]  # (n, nleaves)
    lbls = mega["labels"]["local"][idx]  # (n, nleaves)

    if ncomps is None:
        ncomps = int(np.max(lbls)) + 1

    # Keep only samples with exactly 1 of each label 0..ncomps-1
    valid = np.ones(len(idx), dtype=bool)
    for li in range(ncomps):
        valid &= np.sum(lbls == li, axis=-1) == 1

    gamma_f = gamma[valid]
    lbls_f = lbls[valid]

    gamma_a = gamma_f[lbls_f == label_a]  # one entry per valid sample
    gamma_b = gamma_f[lbls_f == label_b]

    return np.mean(gamma_a > gamma_b)


# data helper


def setup_data_module(model_name):
    datapath = f"{DATADIR}/{model_name}"
    with open(os.path.join(datapath, "prior.json")) as f:
        input_priordict = json.load(f)
        _ = data.process_input_priordict(input_priordict)


def get_draw_ctx(model_name, model_sig=None, submodel=False):
    datapath = f"{DATADIR}/{model_name}"
    fn = "submodel_draw_ctxs.npy" if submodel else "model_draw_ctxs.npy"
    draw_ctx = np.load(os.path.join(datapath, "data", fn), allow_pickle=True).item()
    if model_sig is None:
        return draw_ctx
    if model_sig not in draw_ctx:
        raise ValueError(
            f"Model signature {model_sig} not found in draw context. Expecting one of {draw_ctx.keys()}"
        )
    return draw_ctx[model_sig]


def get_ppds(model_name, model_sig=None, param=None, submodel=False):
    datapath = f"{DATADIR}/{model_name}/data"
    fn = "ppds_submodels.npy" if submodel else "ppds.npy"
    ppds_dict = np.load(os.path.join(datapath, fn), allow_pickle=True).item()
    if param is None:
        return ppds_dict
    ppds_dict = ppds_dict[f"{param}_labelled"]
    if model_sig in ppds_dict:
        return ppds_dict[model_sig]
    else:
        raise ValueError(
            f"Model signature {model_sig} not found in ppds dict for parameter {param}. Expecting one of {ppds_dict.keys()}"
        )


def get_posterior_prob_and_weights(model_name, model_sig=None):
    datapath = f"{DATADIR}/{model_name}/data"

    if model_sig:
        ev_prob_fn = f"event_probabilities_{utils.get_safe_fn(model_sig)}.npy"
        ev_wts_fn = (
            f"PE_posterior_reweighted_samples_{utils.get_safe_fn(model_sig)}.npy"
        )
    else:
        ev_prob_fn = glob(f"{datapath}/event_probabilities_*.npy")
        if not ev_prob_fn:
            raise FileNotFoundError(
                f"No event probabilities file matching pattern `event_probabilities_*.npy` found in {datapath}"
            )
        ev_prob_fn = ev_prob_fn[0]

        ev_wts_fn = glob(f"{datapath}/PE_posterior_reweighted_samples_*.npy")
        if not ev_wts_fn:
            raise FileNotFoundError(
                f"No posterior weights file matching pattern `PE_posterior_reweighted_samples_*.npy` found in {datapath}"
            )
        ev_wts_fn = ev_wts_fn[0]

    ev_prob = np.load(ev_prob_fn)
    ev_wts = np.load(ev_wts_fn)

    return ev_prob, ev_wts


def get_comp_label_info(model_name):
    return get_ppds(model_name)["comp_labels"]


def save_plot(filename, fig=None, dpi=300, bbox_inches="tight", **savefig_kwargs):
    """
    Save the current figure as PDF (normal) and PNG (transparent background,
    inverted colours for dark-background presentations).

    Parameters
    ----------
    filename : str
        Base filename without extension. Saved to ``FIGPATH``.
    fig : matplotlib.figure.Figure, optional
        Figure to save. Uses ``plt.gcf()`` if None.
    dpi : int
        DPI for the PNG output.
    bbox_inches : str
        Bounding-box mode passed to ``savefig``.
    **savefig_kwargs
        Extra arguments forwarded to ``fig.savefig`` for the PDF.
    """

    if fig is None:
        fig = plt.gcf()

    os.makedirs(FIGPATH, exist_ok=True)
    pdf_path = os.path.join(FIGPATH, f"{filename}.pdf")
    fig.savefig(pdf_path, bbox_inches=bbox_inches, **savefig_kwargs)

    if FIGPATH_DARK:
        os.makedirs(FIGPATH_DARK, exist_ok=True)
        png_path = os.path.join(FIGPATH_DARK, f"{filename}.png")
        fig.savefig(png_path, bbox_inches=bbox_inches, transparent=True, dpi=dpi)

    img = Image.open(png_path).convert("RGBA")
    arr = np.array(img)
    arr[:, :, :3] = 255 - arr[:, :, :3]
    Image.fromarray(arr).save(png_path)
