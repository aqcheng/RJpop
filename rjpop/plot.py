import matplotlib.colors as pltc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec, rcParams
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from popsummary.popresult import PopulationResult

# Optional LVK plotting helpers (from the data release figure_scripts directory).
# Call setup_lvk_plot_funcs(lvk_res_path) before using setup_and_plot_GWTC4.
pf = None


def setup_lvk_plot_funcs(lvk_res_path):
    """Load plot_funcs_bbh_mass from the LVK data release figure_scripts directory."""
    global pf
    import os
    import sys

    sys.path.append(os.path.join(lvk_res_path, "figure_scripts"))
    try:
        import plot_funcs_bbh_mass as _pf

        pf = _pf
    except ImportError as e:
        print(f"Warning: could not import LVK plot functions from {lvk_res_path}: {e}")


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
    "axes.labelpad": 2.5,
}


def setup():
    rcParams.update(plot_style)
    # rcParams["xtick.major.pad"] = "6"
    # rcParams["ytick.major.pad"] = "6"


def textsc_ify(word: str) -> str:
    return f"$\\textsc{{{word}}}$"


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
    swap_xy=False,
    rasterize=False,
):
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

        if swap_xy:
            ax.fill_betweenx(x, low, high, color=color, alpha=fill_alpha, label=label)
        else:
            ax.fill_between(x, low, high, color=color, alpha=fill_alpha, label=label)
            
        if swap_xy:
            if lw:
                ax.plot(med, x, color=color, lw=lw, ls=ls)
            if secondary_lw:
                ax.plot(low, x, color=color, lw=secondary_lw, ls="--")
                ax.plot(high, x, color=color, lw=secondary_lw, ls="--")
        else:
            if lw:
                ax.plot(x, med, color=color, lw=lw, ls=ls)
            if secondary_lw:
                ax.plot(x, low, color=color, lw=secondary_lw, ls="--")
                ax.plot(x, high, color=color, lw=secondary_lw, ls="--")
        if label is not None:
            label = textsc_ify(label)


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
    ax_x.set_ylabel(
        f"$\\mathrm{{d}}\\mathcal{{R}}/\\mathrm{{d}}{xlabel.strip('$')}$", fontsize=8, labelpad=6
    )
    ax_y.set_xlabel(
        f"$\\mathrm{{d}}\\mathcal{{R}}/\\mathrm{{d}}{ylabel.strip('$')}$", fontsize=8, labelpad=6
    )

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


def setup_and_plot_GWTC4(
    param_name,
    ax,
    res: PopulationResult | None = None,
    spin_res: PopulationResult | None = None,
    label=True,
    pf_kwargs=None,
    CI_kwargs=None,
):

    if pf_kwargs is None:
        pf_kwargs = {}

    x, y = None, None
    label_default = None

    if param_name == "chi_eff":
        if spin_res is not None:
            x, y = spin_res.get_rates_on_grids("Effective inspiral spin")
            x, y = x[0], y.T  # bruh
            label_default = "LVK skew-normal"
        ax.grid(color="silver", alpha=0.5, ls=":", zorder=0)
        ax.set_xlabel(r"$\chi_\mathrm{eff}$")
        ax.set_ylabel(r"$p(\chi_\mathrm{eff})$")
        ax.set_ylim(0, 5.5)
        ax.set_xlim(-0.35, 0.65)

    elif param_name == "mass_1_source":
        if res is not None and pf is not None:
            x, y = pf.get_params(res, "mass_1")
            if "BSpline" in res.fname:
                label_default = r"$\textsc{LVK Spline}$"
            else:
                label_default = r"$\textsc{LVK BP2P}$"
        if pf is not None:
            pf.setup_mass_plot(ax, xrange=(2, 100), yrange=(1e-3, 40), **pf_kwargs)

    elif param_name == "mass_2_source":
        if pf is not None:
            pf.setup_mass_plot(ax, xrange=(2, 100), yrange=(1e-3, 40), **pf_kwargs)
            ylabel = ax.get_ylabel()
            ax.set_ylabel(ylabel.replace("m_1", "m_2"))
        ax.set_xlabel(r"$m_2 \,\left[ \mathrm{M}_\odot \right]$")
        # don't plot LVK results
        return

    elif param_name == "mass_ratio":
        if res is not None and pf is not None:
            if "BSpline" in res.fname:
                x, y = pf.get_params(res, "rate_vs_mass_ratio_at_z0-2", rate=False)
                label_default = r"$\textsc{LVK Spline}$"
            else:
                x, y = pf.get_params(res, "mass_ratio")
                label_default = r"$\textsc{LVK BP2P}$"
        if pf is not None:
            pf.setup_mass_ratio_plot(ax, **pf_kwargs)

    elif param_name == "redshift":
        if res is not None:
            x, y = res.get_rates_on_grids("redshift")
            x = x[0]  # bro why did the LVK code it like this
            label_default = r"$\textsc{LVK}$"

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

    if x is not None and y is not None and pf is not None:
        line_default_kwargs = dict(
            color="k",
            median=False,
            fill=False,
            secondary_ls="--",
            lw=0.8,
            label=label,
            fill_alpha=0.3,
        )
        if CI_kwargs:
            line_default_kwargs.update(CI_kwargs)

        pf.plot_90CI(ax, x, y, **line_default_kwargs)
