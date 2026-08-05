import matplotlib.colors as pltc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec, rcParams
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from popsummary.popresult import PopulationResult

### PLOTTING UTILS

param_latex = {
    "mass_1_source": "$m_1$",
    "mass_2_source": "$m_2$",
    "mass_ratio": "$q$",
    "chi_eff": r"$\chi_{\mathrm{eff}}$",
    "redshift": "$z$",
}

plot_style = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.4,
    "axes.labelsize": 10,
    "axes.titlesize": 12,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 11,
    "figure.titlesize": 12,
    "lines.linewidth": 0.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": ":",
    "figure.figsize": (3.5, 2.75),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.format": "pdf",
    "savefig.bbox": "tight",
    "text.usetex": True,
    "axes.unicode_minus": False,
    "axes.labelpad": 2.5,
}


def setup():
    rcParams.update(plot_style)
    # rcParams["xtick.major.pad"] = "6"
    # rcParams["ytick.major.pad"] = "6"


def textsc_ify(word: str) -> str:
    return f"$\\textsc{{{word}}}$"

def add_superscript(latex: str, superscript: str) -> str:
    res = latex.strip("$")
    exp = rf"\rm{{{superscript}}}"
    res += rf"^{{{exp}}}"
    return f"${res}$"

def plot_ppds(
    ax,
    x,
    ppds,
    color="k",
    CI=90,
    label=None,
    line_alpha=0.15,
    fill_alpha=0.3,
    lw=1,
    secondary_lw=0,
    lws=0.3,
    ls="-",
    secondary_ls="--",
    swap_xy=False,
    rasterize=False,
):
    x = x.ravel()
    if CI is None:
        ppds = ppds[np.any(ppds > 0, axis=-1)]
        segments = np.empty((ppds.shape[0], ppds.shape[1], 2))
        segments[..., 0] = x
        segments[..., 1] = ppds
        if swap_xy:
            segments[..., [0, 1]] = segments[..., [1, 0]]
        lc = LineCollection(segments, linewidths=lws, alpha=line_alpha, colors=color)
        if rasterize:
            lc.set_rasterized(True)
        ax.add_collection(lc)

        if label is not None:
            ax.plot([], [], lw=1, color=color, label=label)

    else:
        dist = (100 - CI) / 2
        percs = [dist, 50, 100 - dist]
        low, med, high = np.nanpercentile(ppds, percs, axis=0)

        if fill_alpha: # plot fill
            if swap_xy:
                ax.fill_betweenx(x, low, high, color=color, alpha=fill_alpha, label=label)
            else:
                ax.fill_between(x, low, high, color=color, alpha=fill_alpha, label=label)
            label = None # don't label lines
        
        if lw: # plot median
            if swap_xy:
                ax.plot(med, x, color=color, lw=lw, ls=ls, label=label)
            else:
                ax.plot(x, med, color=color, lw=lw, ls=ls, label=label)
            label = None
        
        if secondary_lw: # plot percentiles
            if swap_xy:
                ax.plot(low, x, color=color, lw=secondary_lw, ls="--", label=label)
                ax.plot(high, x, color=color, lw=secondary_lw, ls="--")
            else:
                ax.plot(x, low, color=color, lw=secondary_lw, ls=secondary_ls, label=label)
                ax.plot(x, high, color=color, lw=secondary_lw, ls=secondary_ls)


def plot_chains(
    data, ax=None, color="cornflowerblue", xlabel="step", ylabel=None, walkers_plot=None
):
    """
    Plots chains of a 2D data array of shape (nsteps, nwalkers) on a matplotlib axis.
    """
    if ax is None:
        fig, ax = plt.subplots()
    if walkers_plot is None:
        inds = np.arange(data.shape[-1])
    elif isinstance(walkers_plot, int):
        if walkers_plot > data.shape[-1]:
            inds = np.arange(data.shape[-1])
        else:
            inds = np.random.default_rng().choice(data.shape[-1], walkers_plot, replace=False)
    elif hasattr(walkers_plot, "__iter__"):
        inds = walkers_plot
    else:
        raise ValueError("walkers_plot must be None, int, or iterable of int")

    for i in inds:  # iterate through all walkers
        ax.plot(data[:, i], lw=0.2, alpha=0.2, color=color)
    ax.plot(np.mean(data, axis=-1), color="k", ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax


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

    flat = np.sort(Z.ravel())[::-1]  # highest densities first
    cdf = np.cumsum(flat)
    levels = []
    for p in percentiles:
        idx = np.searchsorted(cdf, p)
        levels.append(flat[idx - 1])
    levels = levels[::-1] + [flat[0]]
    # include largest positive value so outermost contour closes
    return np.unique(levels) * total  # scale back


def initialize_2D_plotting_axes(figsize=(4,4)):
    plt.close()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(
        2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05
    )
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_x = fig.add_subplot(gs[0, 0], sharex=ax_joint)
    ax_y = fig.add_subplot(gs[1, 1], sharey=ax_joint)
    return fig, ax_joint, ax_x, ax_y


def plot_2D_contours_and_marginals(
    xx,
    yy,
    x_marg_ppd,
    y_marg_ppd,
    p_xy_mean,
    color="cornflowerblue",
    axes=None,
    contour=True,
    contourf=True,
    figsize=(4,4),
    CI=None,
    rasterize_ppds=True,
    levels=(0.5, 0.9, 0.99),
    xlabel=None,
    ylabel=None,
    x_param_ylim=None,
    y_param_ylim=None,
    alpha=1,
    ppd_kwargs=None,
    contour_kwargs=None,
    contourf_kwargs=None,
):

    if contour_kwargs is None:
        contour_kwargs = {}
    if contourf_kwargs is None:
        contourf_kwargs = {}
    if ppd_kwargs is None:
        ppd_kwargs = {}

    if axes is None:
        fig, ax_joint, ax_x, ax_y = initialize_2D_plotting_axes(figsize)
    else:
        fig, ax_joint, ax_x, ax_y = axes

    ppd_kwargs_ = dict(color=color, CI=CI, rasterize=rasterize_ppds) | ppd_kwargs
    plot_ppds(ax_x, xx, x_marg_ppd, **ppd_kwargs_)
    plot_ppds(ax_y, yy, y_marg_ppd, swap_xy=True, **ppd_kwargs_)
    # swap x and y for y param

    # construct colormap
    base = pltc.to_rgba(color, alpha=alpha)
    light = pltc.to_rgba(color, alpha=0.3 * alpha)
    cmap = pltc.LinearSegmentedColormap.from_list(f"{str(color).capitalize()}s", [light, base])
    vmin, vmax = np.nanmin(p_xy_mean[p_xy_mean > 0]), np.nanmax(p_xy_mean)
    # extent = (xx[0], xx[-1], yy[0], yy[-1])
    if contour or contourf:
        contour_levels = get_level_values(p_xy_mean, levels)
        X, Y = np.meshgrid(xx, yy)
        if contourf:
            ax_joint.contourf(
                X,
                Y,
                p_xy_mean.T,
                levels=contour_levels,
                cmap=cmap,
                origin="lower",
                **contourf_kwargs,
            )
        if contour:
            ax_joint.contour(
                X,
                Y,
                p_xy_mean.T,
                levels=contour_levels,
                colors=color,
                origin="lower",
                alpha=alpha,
                **contour_kwargs,
            )
    else:
        ax_joint.imshow(
            p_xy_mean.T,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=pltc.LogNorm(vmin=vmin, vmax=vmax),
            **contour_kwargs,
        )

    ref_x = np.nanpercentile(x_marg_ppd, 99)
    ref_y = np.nanpercentile(y_marg_ppd, 99)

    # logscale both axes
    if ref_x > 0:
        ax_x.set_yscale("log")
        if x_param_ylim:
            ax_x.set_ylim(x_param_ylim)
        else:
            ax_x.set_ylim(1e-3 * ref_x, 3 * ref_x)

    if ref_y > 0:
        ax_y.set_xscale("log")
        if y_param_ylim:
            ax_y.set_xlim(y_param_ylim)
        else:
            ax_y.set_xlim(1e-3 * ref_y, 3 * ref_y)

    ax_joint.set_xlabel(xlabel, fontsize=10)
    ax_joint.set_ylabel(ylabel, fontsize=10)
    ax_x.tick_params(axis="both", which="both", labelleft=False, labelbottom=False, length=0)
    ax_y.tick_params(axis="both", which="both", labelleft=False, labelbottom=False, length=0)
    
    if xlabel is not None:
        if "redshift" in xlabel or "z" in xlabel:
            p_xlabel = f"$p({xlabel.strip('$')})$"
        else:
            p_xlabel = f"$p({xlabel.strip('$')} | z=0.2)$"
        if "redshift" in ylabel or "z" in ylabel:
            p_ylabel = f"$p({ylabel.strip('$')})$"
        else:
            p_ylabel = f"$p({ylabel.strip('$')} | z=0.2)$"

        ax_x.set_ylabel(p_xlabel, fontsize=8, labelpad=6)
        ax_y.set_xlabel(p_ylabel, fontsize=8, labelpad=6)

    return fig, ax_joint, ax_x, ax_y
    # fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=len(handles))


def shade_triangle(ax, plane="upper half", color="#e7e7e7"):
    """
    Shade the y>x (or y<x) half plane gray.
    """

    # after your imshow call
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if plane.lower() == "upper half":
        coords = [(0, ymax), (0, 0), (xmax, ymax)]
    elif plane.lower() == "lower half":
        coords = [(xmax, 0), (0, 0), (xmax, ymax)]

    triangle = Polygon(
        coords,
        facecolor=color,  # light gray
        edgecolor="none",
        alpha=1.0,
        label="m1m2-mask",
        zorder=3,  # cover grid
    )
    ax.add_patch(triangle)

    # ensure limits stay unchanged
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return ax


corner_defaults = dict(
    color="cornflowerblue",
    levels=[0.5, 0.9],
    show_titles=True,
    title_kwargs={"fontsize": 11},
    label_kwargs={"fontsize": 11},
    labelpad=-0.07,
    hist_kwargs=dict(histtype="stepfilled", alpha=0.7, density=True),
    density=True,
    smooth=1.0,
    plot_datapoints=False,
    plot_density=False,
    fill_contours=True,
    bins=12,
    title_fmt=".2f",
    hist_bin_factor=1,
    max_n_ticks=4,
    use_math_text=True,
    quantiles=[0.05, 0.5, 0.95],
)

invisible_corner_kwargs = dict(
    hist_kwargs=dict(alpha=0),
    fill_contours=False,
    plot_datapoints=False,
    plot_density=False,
    plot_contours=False,
    no_fill_contours=True,
)

texnames = {
    "mass_1_source": r"$m_1$",
    "mass_2_source": r"$m_2$",
    "mass_ratio": r"$q$",
    "redshift": r"$z$",
    "chi_eff": r"$\chi_\mathrm{eff}$",
}

# color palettes - reorder to the colors I like better hehe

Tab10 = sns.color_palette("deep")
tab10 = Tab10[8:6:-1] + Tab10[6:3:-1] + Tab10[8:] + Tab10[:3]

Set2 = sns.color_palette("Set2")
set2 = [Set2[0], Set2[2], Set2[3], Set2[1]] + Set2[4:]

# hard coded in - for consistency with NPLNP colors
skewt_colors = [set2[0], set2[2], set2[1], set2[3]]

#####################################################################
### popsummary helpers - mostly lifted from GWTC-4 plotting code, ###
### https://zenodo.org/records/16911563                           ###
#####################################################################

def get_params(dataset, param, rate = True):
    # lifted from GWTC-4 plotting code, https://zenodo.org/records/16911563
    param = [param] if not isinstance(param, list) else param
    for p in param:
        dat = dataset.get_rates_on_grids(p)
        x = dat[0][0]
        pdf = dat[1]
    if rate:
        lamb = dataset.get_hyperparameter_samples(hyperparameters = 'lamb')
        pdf = pdf * (1 + 0.2)**lamb.reshape(-1,1)
    return x, pdf

def setup_mass_plot(ax, m="m_1", xrange=(2,100), yrange=(1e-3,40), yscale = 'log', xscale = 'linear', label_kwargs = {}, grid_kwargs={}):
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
    ax.set_xlabel(rf"${m} \left[ \mathrm{{M}}_\odot \right]$", **label_kwargs)
    ax.set_ylabel(rf'$\mathrm{{d}}\mathcal{{R}}/\mathrm{{d}}{m} \left[ \mathrm{{Gpc}}^{{-3}} \, \mathrm{{yr}}^{{-1}} \, \mathrm{{M}}_\odot^{{-1}} \right]$', **label_kwargs)
    ax.grid(**grid_kwargs)

def setup_mass_ratio_plot(ax, xrange=(0,1), yrange=(4e-2,3e2), yscale = 'log', xscale = 'linear', label_kwargs = {}, grid_kwargs={}):
    ax.set_yscale(yscale)
    ax.set_xscale(xscale)
    ax.set_ylim(*yrange)
    ax.set_xlim(*xrange)
    ax.set_xlabel(r"$q$", **label_kwargs)
    ax.set_ylabel(r'$\mathrm{d}\mathcal{R}/\mathrm{d}q \left[ \mathrm{Gpc}^{-3} \, \mathrm{yr}^{-1} \right]$', **label_kwargs)
    ax.grid(**grid_kwargs)

#####################################################################

def setup_and_plot_GWTC(
    param_name,
    ax,
    res: PopulationResult | None = None,
    spin_res: PopulationResult | None = None,
    catalog: str | None = None,
    label=True,
    CI_kwargs=None,
):

    x, y = None, None
    label_default = None

    if param_name == "chi_eff":
        if spin_res is not None:
            if catalog == "GWTC4":
                x, y = spin_res.get_rates_on_grids("Effective inspiral spin")
                x, y = x[0], y.T  # bruh
                label_default = '$\\textsc{LVK skew-normal (gwtc-4)}$'
            elif catalog == "GWTC5":
                x, y = spin_res.get_rates_on_grids("chi_eff")
                label_default = '$\\textsc{LVK bivariate skew-normal (gwtc-5)}$'
        ax.grid(color="silver", alpha=0.5, ls=":", zorder=0)
        ax.set_xlabel(r"$\chi_\mathrm{eff}$")
        ax.set_ylabel(r"$p(\chi_\mathrm{eff})$")
        ax.set_ylim(0, 5.5)
        ax.set_xlim(-0.35, 0.65)
    
    else:
        if catalog == "GWTC4":
            label_default = r"$\textsc{LVK BP2P (gwtc-4)}$"
        elif catalog == "GWTC5":
            label_default = r"$\textsc{LVK Default (gwtc-5)}$"

        if param_name == "mass_1_source":
            if res is not None:
                x, y = get_params(res, "mass_1")
                x = x.ravel()
            setup_mass_plot(ax)

        elif param_name == "mass_2_source":
            setup_mass_plot(ax, m="m_2")
            # don't plot LVK results
            return

        elif param_name == "mass_ratio":
            if res is not None:
                x, y = get_params(res, "mass_ratio")
                x = x.ravel()
            setup_mass_ratio_plot(ax)

        elif param_name == "redshift":
            if res is not None:
                x, y = res.get_rates_on_grids("redshift")
                x = x.ravel()

            ax.set_yscale("log")
            ax.set_ylim([8, 3e3])
            ax.set_xlim([0, 1.5])
            ax.set_xlabel("$z$")
            ax.set_ylabel("$\\mathcal{R}(z)$ [Gpc${}^{-3}$ yr${}^{-1}$]")

        else:
            print(f"Unrecognized plotting parameter {param_name}")
            return

    if not isinstance(label, str):
        label = label_default if label else None

    if x is not None and y is not None:
        line_default_kwargs = dict(
            color="k",
            CI=90,
            label=label,
            fill_alpha=0,
            lw=0,
            secondary_lw=0.8
        )
        if CI_kwargs:
            line_default_kwargs.update(CI_kwargs)

        plot_ppds(ax, x, y, **line_default_kwargs)


def plot_all_param_ppds(
    ppds_dict,
    model_sig,
    comp_label_info,
    res=None,
    spin_res=None,
    catalog=None,
    title=None,
    labeldict=None,
    savepath=None,
    rasterize=False,
):
    """Condensed PPD figure for a single (sub)model signature.

    Shows mass_1, mass ratio, and chi_eff, each both broken down by labelled
    component (left axis of each pair) and as a total with 90% CI (right axis),
    overlaid with the LVK reference results.

    Parameters
    ----------
    ppds_dict : dict
        A model/submodel ppds dict (e.g. the contents of ``ppds_submodels.npy``)
        holding, for each parameter, ``ppds_dict[f"{param}_labelled"][model_sig]
        [comp_name]`` arrays of shape (ndraws, ngrid) and an "x" grid.
    model_sig : str
        The (sub)model signature key to plot.
    comp_label_info : dict
        ``{"comp_names": [...], "colors": [...]}`` for the labelled components.
    res, spin_res : PopulationResult or None
        LVK reference results for the GWTC overlays.
    catalog : str or None
        "GWTC4"/"GWTC5", forwarded to :func:`setup_and_plot_GWTC`.
    title : str or None
        Figure suptitle.
    labeldict : dict or None
        Optional mapping ``comp_name -> display label``.
    savepath : str or None
        If given, the figure is saved there (tight bbox) and closed.
    """
    fig = plt.figure(figsize=(7, 3.5))
    gs = gridspec.GridSpec(2, 5, width_ratios=(1, 1, 0.1, 1, 1), hspace=0.35, wspace=0.15)
    axm = fig.add_subplot(gs[0, 0:2])
    axq = fig.add_subplot(gs[1, 0])
    axqtot = fig.add_subplot(gs[1, 1], sharex=axq, sharey=axq)
    axmtot = fig.add_subplot(gs[0, 3:5], sharex=axm, sharey=axm)
    axchi = fig.add_subplot(gs[1, 3])
    axchitot = fig.add_subplot(gs[1, 4], sharex=axchi, sharey=axchi)

    comp_axes = [axm, axq, axchi]
    tot_axes = [axmtot, axqtot, axchitot]
    param_names = ["mass_1_source", "mass_ratio", "chi_eff"]

    for comp_ax, tot_ax, param_name in zip(comp_axes, tot_axes, param_names):
        param_ppds = ppds_dict[f"{param_name}_labelled"]
        x = param_ppds["x"]
        sig_ppds = param_ppds[model_sig]

        # gather the labelled components present in this (sub)model
        comps = []
        for comp_name, color in zip(comp_label_info["comp_names"], comp_label_info["colors"]):
            if comp_name in sig_ppds:
                ppds = sig_ppds[comp_name]
                if np.any(ppds):
                    comps.append((comp_name, color, ppds))
        tot_ppds = sum((ppds for _, _, ppds in comps), start=0)

        # mass and mass ratio are shown rate-weighted, but chi_eff is shown as a
        # normalized density p(chi_eff). The stored chi_eff ppds are rate-weighted
        # (dR/dchi_eff), so divide each draw by its total rate -- equivalently the
        # integral of the rate-weighted total over chi_eff (no-op if already a density).
        norm = 1.0
        if param_name == "chi_eff" and np.any(tot_ppds):
            norm = np.trapezoid(tot_ppds, x, axis=-1)[:, None]
            norm = np.where(norm == 0, 1.0, norm)

        for comp_name, color, ppds in comps:
            label = comp_name if labeldict is None else labeldict.get(comp_name, comp_name)
            plot_ppds(comp_ax, x, ppds / norm, color=color, CI=None,
                      label=textsc_ify(label), rasterize=rasterize, line_alpha=0.1)
        if np.any(tot_ppds):
            plot_ppds(tot_ax, x, tot_ppds / norm, color="cornflowerblue",
                      CI=90, lw=0.7, secondary_lw=0.7, label="Total")
        setup_and_plot_GWTC(param_name, comp_ax, res=res, spin_res=spin_res,
                            catalog=catalog, CI_kwargs=dict(color="dimgray"))
        setup_and_plot_GWTC(param_name, tot_ax, res=res, spin_res=spin_res,
                            catalog=catalog, CI_kwargs=dict(color="dimgray"))

        if param_name == "chi_eff" and np.any(tot_ppds):
            comp_ax.set_ylim(0, 8)

    for ax in tot_axes:
        ax.set_ylabel(None)
        ax.tick_params(axis="y", which="both", labelleft=False)
    for ax in (axqtot, axchitot):
        ax.tick_params(axis="y", which="both", labelleft=False, length=0)
    for ax in comp_axes + tot_axes:
        ax.yaxis.labelpad = 1.75
        ax.xaxis.labelpad = 0
        ax.grid(True)
    for ax in (axq, axqtot):
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])

    all_handles, all_labels = [], []
    for ax in (axm, axmtot):
        h, l = ax.get_legend_handles_labels()
        for h_, l_ in zip(h, l, strict=True):
            if l_ not in all_labels:
                all_handles.append(h_)
                all_labels.append(l_)
    if all_handles:
        all_labels, all_handles = zip(*sorted(zip(all_labels, all_handles)))
        ncol = len(all_handles)
        if ncol > 5:
            ncol = int(np.ceil(len(all_handles) / 2))
        fig.legend(handles=all_handles, labels=all_labels, loc="lower center",
                   bbox_to_anchor=(0.5, 0.88), ncol=ncol, columnspacing=1,
                   frameon=False, fontsize=9)
    if title:
        fig.suptitle(title, y=1.02, fontsize=14)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_ev_contour(
    param_x, param_y, x_samples, y_samples, weights=None, CI=90,
    ax=None, xlim=None, ylim=None, **contour_kwargs,
):
    """Plot a single (population-reweighted) PE posterior contour for one event.

    A weighted Gaussian KDE is fit to the standardized ``(param_x, param_y)``
    samples, and the ``CI``% highest-density contour is drawn.

    Parameters
    ----------
    param_x, param_y : str
        Parameter names (used only for axis labels via ``param_latex``).
    x_samples, y_samples : array-like
        The event's PE posterior samples for each parameter, shape (nsamples,).
    weights : array-like or None
        Per-sample weights (e.g. population-reweighting weights). If None, uniform.
    CI : float
        Credible level (percent) enclosed by the drawn contour.
    ax : matplotlib axis or None
    xlim, ylim : tuple or None
        Axis limits.
    **contour_kwargs
        Forwarded to ``ax.contour`` (e.g. ``colors``, ``zorder``).
    """
    from KDEpy import FFTKDE

    x_samples = np.asarray(x_samples, dtype=float)
    y_samples = np.asarray(y_samples, dtype=float)
    if weights is None:
        weights = np.ones(x_samples.shape[0])
    else:
        weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if weights.sum() <= 0:
        weights = np.ones(x_samples.shape[0])
    weights = weights / weights.sum()

    samples = np.stack([x_samples, y_samples], axis=-1)
    mean = np.average(samples, weights=weights, axis=0)
    std = np.sqrt(np.average((samples - mean) ** 2, weights=weights, axis=0))
    samples_normed = (samples - mean) / std

    Neff = 1.0 / np.sum(weights ** 2)
    scott_bw = Neff ** (-1 / (2 + 4))  # Scott's rule in 2D (standardized data)
    grid_pts = 512
    kde = FFTKDE(kernel="gaussian", bw=scott_bw).fit(samples_normed, weights=weights)
    grid, ZZ_flat = kde.evaluate(grid_pts)
    ZZ = ZZ_flat.reshape(grid_pts, grid_pts).T
    x = np.unique(grid[:, 0]) * std[0] + mean[0]
    y = np.unique(grid[:, 1]) * std[1] + mean[1]

    # highest-density level enclosing CI% of the probability mass
    flat_sorted = np.sort(ZZ.ravel())[::-1]
    cumsum = np.cumsum(flat_sorted) / flat_sorted.sum()
    level = flat_sorted[np.searchsorted(cumsum, CI / 100)]

    if ax is None:
        _, ax = plt.subplots()
    contour_kwargs = dict(linewidths=0.35) | contour_kwargs
    ax.contour(x, y, ZZ, levels=[level], **contour_kwargs)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(param_latex.get(param_x, param_x))
    ax.set_ylabel(param_latex.get(param_y, param_y))
    return ax


def plot_pop_weighted_posteriors(
    param_x, param_y, PE_samples, prob, colors,
    posterior_pop_weights=None, comp_names=None, grids=None,
    ax=None, legend=True, title=None, conf_thresh=0.75, savepath=None,
):
    """Plot one population-reweighted PE contour per event in (param_x, param_y).

    Each event's contour is colored by its dominant labelled component when that
    component's probability exceeds ``conf_thresh``, otherwise light gray.

    Parameters
    ----------
    param_x, param_y : str
        Parameters to plot.
    PE_samples : dict
        Maps parameter name -> array of shape (nevs, nsamples) of PE samples
        (must be numpy arrays).
    prob : np.ndarray
        Per-event labelled-component probabilities, shape (nevs, ncomps).
    colors : sequence
        One color per labelled component (index-aligned with ``prob`` columns).
    posterior_pop_weights : np.ndarray or None
        Per-event, per-sample reweighting weights, shape (nevs, nsamples). If
        None, contours use the unweighted PE posteriors.
    comp_names : sequence of str or None
        Component names for the legend (index-aligned with ``colors``).
    grids : dict or None
        Maps parameter name -> 1D grid; used to set axis limits.
    ax : matplotlib axis or None
    legend, title : see below.
    conf_thresh : float
        Probability above which an event is colored by its dominant component.
    savepath : str or None
        If given, the figure is saved there (tight bbox) and closed.
    """
    if ax is None:
        _, ax = plt.subplots()
    x_all = np.asarray(PE_samples[param_x])
    y_all = np.asarray(PE_samples[param_y])
    nevs = x_all.shape[0]
    if posterior_pop_weights is None:
        posterior_pop_weights = [None] * nevs
    xlim = (grids[param_x][0], grids[param_x][-1]) if grids is not None else None
    ylim = (grids[param_y][0], grids[param_y][-1]) if grids is not None else None

    for ev_idx in range(nevs):
        ev_prob = prob[ev_idx]
        if np.any(ev_prob > conf_thresh):
            color = colors[int(np.argmax(ev_prob))]
            zorder = 2
        else:
            color = "lightgray"
            zorder = 1
        plot_ev_contour(
            param_x, param_y, x_all[ev_idx], y_all[ev_idx],
            weights=posterior_pop_weights[ev_idx], ax=ax,
            colors=[color], zorder=zorder, xlim=xlim, ylim=ylim,
        )

    if legend and comp_names is not None:
        real_comps = np.nonzero(np.any(prob > 0, axis=0))[0]
        real_comps = real_comps[np.argsort([comp_names[int(i)] for i in real_comps])]
        handles = [
            plt.Line2D([], [], color=colors[int(i)], lw=0.8, label=textsc_ify(comp_names[int(i)]))
            for i in real_comps
        ]
        ax.legend(handles=handles, fontsize=9, loc="lower right")
    if title:
        ax.set_title(title, fontsize=11)
    if savepath:
        fig = ax.get_figure()
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    return ax
