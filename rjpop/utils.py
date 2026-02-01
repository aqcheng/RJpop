import os
import numpy as np
import pdfs
from astropy.cosmology import Planck15
from pathlib import Path
from scipy.optimize import curve_fit

from eryn.utils.utility import get_integrated_act

from popsummary.popresult import PopulationResult
lvk_res_path = '/work/aqc/data/GWTC_data/processed/o4a-astro'
import sys
sys.path.append(os.path.join(lvk_res_path, 'figure_scripts'))
import plot_funcs_bbh_mass as pf
pf.setup()

from xp import xp

# plotting
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as pltc
from matplotlib.patches import Polygon

import re

### HARD CODED MODEL DEFINITIONS

# dictionary of models and their hyperparameters -- these are manually hard coded in 
XMAX_FIX = 300.0
MODELS = {
    'mass': {
        'm1_q': {
            'model': pdfs.m1_q_model,
            'param_latex': {}
        },
        'gaussian_copula': {
            'model': pdfs.gaussian_copula_mass_model,
            'param_latex': {
                'rho': r'$\rho$'
            }
        },
        'sym_gaussian_copula': {
            'model': pdfs.sym_gaussian_copula_mass_model,
            'param_latex': {
                'rho': r'$\rho$'
            }
        }
    },
    'mass_1_source': { # for p(m1) or p(m2)
        'param_latex': { # shared between all models of the parameter
            'xmin': r'$m_{\min}$',
            'xmax': r'$m_{\max}$'
        },
        'skew-t': {
            'model': pdfs.jf_skew_t(),
            'param_latex': {
                'logalpha': r'$\log_{10}\alpha$',
                'logkappa': r'$\log_{10}\kappa$',
                'loc': r'$\mu_m$',
                'scale': r'$\sigma_m$'
            },
            'params_fix': {'xmax': XMAX_FIX},
        },
        'PLS': {
            'model': pdfs.smoothed_powerlaw(),
            'param_latex': {
                'alpha': r'$\alpha$',
                'p': r'$p_m$',
            },
            'params_fix': {'xmax': XMAX_FIX},
        },
        'PLS_LVK': {
            'model': pdfs.LVK_Plancktaper_powerlaw(),
            'param_latex': {
                'alpha': r'$\alpha$',
                'delta': r'$\delta_m$',
            },
            'params_fix': {'xmax': XMAX_FIX},
        },
        'gauss': {
            'model': pdfs.gaussian(),
            'param_latex': {
                'loc': r'$\mu_p$',
                'scale': r'$\sigma_p$',
            },
            'params_fix': {'xmax': XMAX_FIX},
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
        },
        'gauss': {
            'model': pdfs.gaussian(),
            'param_latex': {
                'loc': r'$\mu_{q}$',
                'scale': r'$\sigma_{q}$',
            },
            'params_fix': {
                'xmax': 1.
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
PARAM_SCALES = { # characteristic scales for each parameter
    'mass_1_source': 8,
    'mass_2_source': 8,
    'mass_ratio': 0.25,
    'chi_eff': 0.05,
}

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

    diff = X_[..., i, :] - X_[..., j, :]   # (..., P, N)
    d2 = xp.sum(diff**2, axis=-1)        # (..., P)
    d2[xp.isnan(d2)] = xp.inf

    return xp.all(d2 >= min_sep**2, axis=-1)

### PATH / FILE / DATA STUFF

def to_numpy(arr):
    """Convert CuPy->NumPy if needed; otherwise return a NumPy view/copy."""
    return arr.get() if hasattr(arr, 'get') else np.asarray(arr)

def intersection(*lists):
    return list(set(lists[0]).intersection(*lists))

def print_to(fname, text):
    with open(fname, 'a') as f:
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
            if not (x.startswith('__') or x.endswith('__')):
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
    return '_'.join(re.split(r'\W+', name))

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

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
            #RJ branch - use number of active leaves to calculate autocorrelation time
            n_active_leaves = np.count_nonzero(leaf_active_mask(chain), axis=-1, keepdims=True)
            tau_rj = get_integrated_act(n_active_leaves, average=average, fast=fast)
            if np.isfinite(tau_rj):
                tau[name] = tau_rj

        else:
            #non RJ branch
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
    y = np.nansum(logP, axis=-1) / np.count_nonzero(np.isfinite(logP), axis=-1) # average across walkers

    x_halfway = int(len(y) // 2)
    p0 = (max(2e-3, (y[x_halfway] - y[0]) / x_halfway), x_halfway, y[x_halfway])
    bounds = ([1e-3, 0.0, np.amin(logP)], [np.inf, len(y) - 1, np.amax(logP)])

    m, x_break, y_break = curve_fit(simple_logP_model, x, y, p0=p0, bounds=bounds)[0]
    discard = int(x_break + 0.2*(nsamps-x_break))

    return discard

### PLOTTING

def textsc_ify(word: str) -> str:
    return f'$\\textsc{{{word}}}$'

def plot_ppds(ax, x, ppds, color='k', CI=90, label=None, fill_alpha = 0.3, lw = 2, ls='-', swap_xy=False):
    if CI is None:
        for ppd in ppds: # plot individual ppds
            if np.any(ppd > 0):
                if swap_xy:
                    ax.plot(ppd, x, color=color, alpha=0.2, lw=0.1)
                else:
                    ax.plot(x, ppd, color=color, alpha=0.2, lw=0.1)
    
    else:
        dist = (100 - CI)/2
        percs = [dist, 50, 100 - dist]
        low, med, high = np.nanpercentile(ppds, percs, axis=0)
        if swap_xy:
            ax.plot(med, x, color = color, lw = lw, ls = ls)
        else:
            ax.plot(x, med, color = color, lw = lw, ls = ls)
        if label is not None:
            label = textsc_ify(label)
        
        if swap_xy:
            ax.fill_betweenx(x, low, high, color = color, alpha = fill_alpha, label=label)
        else:
            ax.fill_between(x, low, high, color = color, alpha = fill_alpha, label=label)

def get_level_values(Z, percentiles):
    """
    Z: 2D array of non-negative weights (already on the plotting grid).
    percentiles: iterable of fractions in (0, 1]; e.g. [0.5, 0.9, 0.99].
 
    Returns contour levels you can pass to contour/contourf so that each
    level encloses the requested fraction of the total probability mass.
    """
    Z = np.nan_to_num(Z, nan=0, posinf=0, neginf=0)
    total = Z.sum()
    if total <= 0:
        raise ValueError("Z must have positive mass.")
    Z /= total
 
    flat = np.sort(Z.ravel())[::-1]    # highest densities first
    cdf = np.cumsum(flat)
    levels = []
    for p in percentiles:
        idx = np.searchsorted(cdf, p)
        levels.append(flat[idx-1])
    levels = levels[::-1] + [flat[0]]
    # include largest positive value so outermost contour closes
    return np.unique(levels) * total # scale back

def initialize_2D_plotting_axes(figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=(4, 1),
        height_ratios=(1, 4),
        wspace=0.05,
        hspace=0.05
    )
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    return fig, ax_joint, ax_x, ax_y

def plot_2D_contours_and_marginals(
    xx, yy, x_marg_ppd, y_marg_ppd, p_xy_mean, color='cornflowerblue', axes=None, 
    contour=True, CI=None, levels=(0.5, 0.9, 0.99), xlabel=None, ylabel=None,
    x_param_ylog=False, y_param_ylog=False, x_param_ylim=None, y_param_ylim=None,
    alpha=1, **plot_kwargs
):

    if axes is None:
        fig, ax_joint, ax_x, ax_y = initialize_2D_plotting_axes()
    else:
        fig, ax_joint, ax_x, ax_y = axes

    plot_ppds(ax_x, xx, x_marg_ppd, color=color, CI=CI)
    plot_ppds(ax_y, yy, y_marg_ppd, color=color, CI=CI, swap_xy=True)
    # swap x and y for y param
    
    # construct colormap
    base = pltc.to_rgba(color, alpha=alpha)
    light = pltc.to_rgba(color, alpha=0.3*alpha)
    cmap = pltc.LinearSegmentedColormap.from_list(
        f'{str(color).capitalize()}s',
        [light, base]
    )
    vmin, vmax = np.nanmin(p_xy_mean[p_xy_mean>0]), np.nanmax(p_xy_mean)
    # extent = (xx[0], xx[-1], yy[0], yy[-1])
    if contour:
        contour_levels = get_level_values(p_xy_mean, levels)
        X, Y = np.meshgrid(xx, yy) 
        ax_joint.contourf(
            X, Y, p_xy_mean.T,
            levels=contour_levels,
            cmap=cmap,
            origin='lower',
            **plot_kwargs
        )
        ax_joint.contour(
            X, Y, p_xy_mean.T,
            levels=contour_levels,
            colors=color,
            origin='lower',
            alpha=alpha,
            **plot_kwargs
        )
    else:
        ax_joint.imshow(
            p_xy_mean.T,
            aspect='auto',
            origin='lower',
            cmap=cmap,
            norm=pltc.LogNorm(vmin=vmin,vmax=vmax),
            **plot_kwargs
        )
    ax_joint.set_xlabel(xlabel)
    ax_joint.set_ylabel(ylabel)
    ax_x.tick_params(axis="x", labelbottom=False)
    ax_y.tick_params(axis="y", labelleft=False)

    ref_x = np.nanpercentile(x_marg_ppd, 99)
    ref_y = np.nanpercentile(y_marg_ppd, 99)
    if x_param_ylog:
        ax_x.set_yscale('log')
        ax_x.set_ylim(1e-4*ref_x, 3*ref_x)
    else:
        ax_x.set_ylim(0, ref_x)
    if y_param_ylog:
        ax_y.set_xscale('log')
        ax_y.set_xlim(1e-4*ref_y, 3*ref_y)
    else:
        ax_y.set_xlim(0, ref_y)
    ax_x.set_ylim(x_param_ylim)
    ax_y.set_xlim(y_param_ylim)

    return fig, ax_joint, ax_x, ax_y
    # fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=len(handles))

def shade_triangle(ax, plane='upper half', color="#e7e7e7"):
    """
    Shade the y>x (or y<x) half plane gray.
    """

    # after your imshow call
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if plane.lower() == 'upper half':
        coords = [(0, ymax), (0, 0), (xmax, ymax)]
    elif plane.lower() == 'lower half':
        coords = [(xmax, 0), (0, 0), (xmax, ymax)]

    triangle = Polygon(
        coords,
        facecolor=color,          # light gray
        edgecolor="none",
        alpha=1.0,
        label="m1m2-mask",
        zorder = 3  # cover grid
    )
    ax.add_patch(triangle)

    # ensure limits stay unchanged
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax

corner_defaults = dict(
    color='darkred',
    plot_points=False,
    levels=[0.5, 0.9],
    show_titles=True, 
    title_kwargs={"fontsize": 15}, 
    label_kwargs={"fontsize": 15},
    density=True,
    # smooth=0.9,
    smooth=None, 
    fill_contours=True,
    bins=20, 
    title_fmt='.2f', 
    hist_bin_factor=1,
    quantiles=[0.05, 0.5, 0.95]
)

invisible_corner_kwargs = dict(
    hist_kwargs=dict(alpha=0),
    fill_contours=False, plot_datapoints=False, plot_density=False, 
    plot_contours=False, no_fill_contours=True
)

plot_style = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "savefig.dpi": 250,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "text.usetex": True
}

texnames = {
    'mass_1_source': r'$m_1$',
    'mass_2_source': r'$m_2$',
    'mass_ratio': r'$q$',
    'redshift': r'$z$',
    'chi_eff': r'$\chi_\mathrm{eff}$'
}

def setup_and_plot_GWTC4(
    param_name, 
    ax, 
    res: PopulationResult | None = None, 
    spin_res: PopulationResult | None = None
):

    x, y = None, None

    if param_name == 'chi_eff':
        if spin_res is not None:
            x, y = spin_res.get_rates_on_grids('Effective inspiral spin')
            x, y = x[0], y.T # bruh
            label = 'LVK skew-normal'
        ax.grid(color='silver', alpha=0.5, ls=':', zorder=0)
        ax.set_xlabel(r'$\chi_\mathrm{eff}$')
        ax.set_ylabel(r'$p(\chi_\mathrm{eff})$')
        ax.set_ylim(0, 5.5)
        ax.set_xlim(-0.35, 0.65)
        ax.axvline(0, ls='--', color='gray')

    elif param_name == 'mass_1_source':
        if res is not None:
            x, y = pf.get_params(res, 'mass_1')
            if 'BSpline' in res.fname:
                label = r'$\textsc{LVK Spline}$'
            else:
                label = r'$\textsc{LVK BP2P}$'
        pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,100), yrange=(1e-3,40))
    
    elif param_name == 'mass_2_source':
        pf.setup_mass_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0), xrange=(2,80), yrange=(1e-3,40))
        ax.set_xlabel(r"$m_2 \left[ \mathrm{M}_\odot \right]$")
        # don't plot LVK results
        return

    elif param_name == 'mass_ratio':
        if res is not None:
            if 'BSpline' in res.fname:
                x, y = pf.get_params(res, 'rate_vs_mass_ratio_at_z0-2', rate = False)
                label = r'$\textsc{LVK Spline}$'
            else:
                x, y = pf.get_params(res, 'mass_ratio')
                label = r'$\textsc{LVK BP2P}$'
        pf.setup_mass_ratio_plot(ax, grid_kwargs=dict(ls='dotted', color = 'k', alpha = 0.3))

    elif param_name == 'redshift':
        if res is not None:
            x, y = res.get_rates_on_grids('redshift')
            x = x[0] # bro why did the LVK code it like this
            label = 'LVK'

        ax.set_yscale('log')
        ax.set_ylim([8, 3e3])
        ax.set_xlim([0,1.5])

        ax.set_xlabel('$z$')
        ax.set_ylabel('$\\mathcal{R}(z)$ [Gpc${}^{-3}$ yr${}^{-1}$]')

        ax.grid(ls = ':', alpha = 0.2, lw = 1, color = 'k')
    
    else:
        print(f'Unrecognized plotting parameter {param_name}')
        return

    if x is not None and y is not None:
        pf.plot_90CI(ax, x, y, color = 'k', median = False, fill = False, secondary_ls ='--', lw = 0.8, label = label, fill_alpha = 0.3)  