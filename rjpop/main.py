import argparse
import builtins
import json
import os
from glob import glob

import sys

if ("--cpu" in sys.argv) or ("-cpu" in sys.argv):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import data

# packages for post-processing
import matplotlib as mpl
import numpy as np
from corner import corner
from eryn.backends import HDFBackend
from eryn.moves import (
    DistributionGenerateRJ,
    GaussianMove,
    MTDistGenMoveRJ,
    StretchMove,
)
from eryn.state import State
from moves import RateMove, UpdateKDEMove
from pdfs import INF, RATE_FACTOR
from xp import xp

from rjpop.load_config import load_config
cfg = load_config()

mpl.use("Agg")
import matplotlib.pyplot as plt
import plot
import utils

plot.setup()


def _safe_print(*args, **kwargs):
    try:
        builtins.print(*args, **kwargs)
    except BrokenPipeError:
        pass


print = _safe_print  # noqa: PLW0603 - intentionally override for safe logging

if not hasattr(np, "in1d"): # eryn is built on an older version of numpy; np.in1d has been phased out
    def _in1d(ar1, ar2, assume_unique=False, invert=False):
        ar1_flat = np.asarray(ar1).ravel()
        ar2_flat = np.asarray(ar2).ravel()
        return np.isin(ar1_flat, ar2_flat, assume_unique=assume_unique, invert=invert)
 
    np.in1d = _in1d

# fmt: off
parser = argparse.ArgumentParser()
# General
parser.add_argument('-s', '--seed', '-seed', type=int, default=1, help='Random seed for reproducibility')
parser.add_argument('--test', '-test', action='store_true', help='Run in test mode (fewer steps)')
parser.add_argument('--label', '-label', default='out', type=str, help="Label for output subdirectory. Default: 'out'")

# Input files
parser.add_argument('--prior', '-prior', type=str, help='Prior json file')
parser.add_argument('--PE_samples', type=str, help='PE samples npz file')
parser.add_argument('--injs', '-injs', type=str, help='Injections npz file')
parser.add_argument('--ignore_events', type=str, nargs='+', help='Which events (if any) to ignore')

#runtime arguments
parser.add_argument('--nwalkers', '-nwalkers', type=int, default=50, help='Number of walkers per temperature')
parser.add_argument('--ntemps', '-ntemps', type=int, default=5, help='Number of temperatures (for parallel tempering)')
parser.add_argument('--Tmax', default=None, type=float, help='Maximum temperature (for parallel tempering)')
parser.add_argument('--rj_num_try', default=1, type=int, help='Number of tries for MT MCMC')
parser.add_argument('--group_chunk_size', type=int, default=None, help='Evaluate the likelihood in batches of at most this many groups (temperature x walker x rj_num_try combinations) to cap peak GPU memory. Default: None (all groups in one pass). Lower it if you hit out-of-memory errors during RJ moves.')
parser.add_argument('--nsteps', '-nsteps', type=int, default=2000, help='Number of steps')
parser.add_argument('--burn', '-burn', type=int, default=0, help='Number of burn-in steps.')
parser.add_argument('--kde_update', type=int, default=0, help='Every kde_update steps, update the KDE from which the reversible jump moves are proposed. Default: 0 (never update, use prior for rj proposals)')
parser.add_argument('--discard', '-discard', type=int, default=None, help='Number of steps to discard in post. Default: None (determine automatically)')
parser.add_argument('--outdir', '-outdir', type=str, default=cfg.get("RJpop_out_path"), help='Output directory. Uses RJpop_out_path from config.json if not specified.')
parser.add_argument('--cpu', action='store_true', help='Force CPU instead of GPU')

# Hyperparameters
parser.add_argument('--use_mchirp', action='store_true', help='Use chirp mass instead of m1 - not currently implemented')
parser.add_argument('--min_sep', '--minsep', type=float, default=3.0, help='Enforce a minimum separation in feature space between components. Default: 3')
parser.add_argument('--rate_prior', type=float, nargs=2, default=(0.05, 100), help="The minimum and maximum rate for the rate prior of each component. Default: (0.05, 100)")

# plotting options
# parser.add_argument('--LVK_plot', type=str, choices=['default', 'spline', 'none'], default='default', help='Which LVK results to plot as reference, if any')
# parser.add_argument('--lvk_res_path', type=str, default=lvk_res_path_default, help='Path to the LVK population data release directory (required when --LVK_plot != none)')
parser.add_argument('--popsummary_path', type=str, default=None, help='Path to the popsummary directory for LVK results. Retrieves from config.json [o4b-astro_path]/popsummary_files, then falls back to [o4a-astro_path]/data_release by default')
parser.add_argument('--skip_corner', action='store_true', help='Skip the corner plots (speeds up post-processing for debugging or reruns)')
parser.add_argument('--plot-only', action='store_true', help='Plot only, assuming there is already a results file; overrides args.nsteps and args.nburn')
parser.add_argument('--ndraws', type=int, default=1000, help='Number of draws for creating posterior probability plots. Default: 1000')
parser.add_argument('--replot', action='store_true', help='(Re)plot all the ppds, even if the plots already exist')
parser.add_argument('--corner_param', type=int, nargs='+', default=None, help='An additional parameter to plot across all components. Should be a list of indices, corresponding to the parameter index of each branch in order. (-1 to skip branch)')
parser.add_argument('--rasterize', type=int, default=1, help='Whether (1) or not (0) to rasterize the ppd plots. Default: 1')
# fmt: on

args = parser.parse_args()
# print settings
print(json.dumps(args.__dict__, indent=2))


data.RNG_SEED = args.seed
xp.random.seed(args.seed)
rng = np.random.default_rng(args.seed)

outdir = os.path.join(args.outdir, args.label)
if args.test:
    outdir += "_test"
figpath = os.path.join(outdir, "figures")
datapath = os.path.join(outdir, "data")
os.makedirs(figpath, exist_ok=True)
os.makedirs(datapath, exist_ok=True)

# -----------------------------
# Prior parsing / sampler setup
# -----------------------------

# load in and setup prior
if os.path.exists(args.prior):
    priorpath = args.prior
else:
    priorpath = os.path.join(cfg.get("RJpop_out_path"), "_priors", args.prior)

if not os.path.exists(priorpath):
    raise FileNotFoundError(f"Prior file {priorpath} not found")

with open(priorpath, "r") as f:
    # this should be a list of dictionaries, one for each branch, the last of which should be global
    # each dictionary should have keys corresponding to each model (mass, q, chieff, rate), with the value as a subdictionary
    # each dictionary should also have a "__branch__" key corresponding to the branch name
    # each subdictionary should have the prior ranges for each parameter as a list or a float if parameter should be fixed,
    # as well as a '__model__' key corresponding to the model name
    # parameter names must match the names defined in the functions (in pdfs.py)
    input_priordicts = json.load(f)

    # save prior in output directory
    if not args.test:
        with open(os.path.join(outdir, "prior.json"), "w") as f:
            json.dump(input_priordicts, f, indent=4)

# process input prior
priors, covs_all = data.process_input_priordict(input_priordicts, rate_prior=args.rate_prior)

PLOT_ONLY = args.plot_only | (args.nsteps + args.burn == 0)

# save hp_ordering, latex parameter names, and nleaves_min_max of each branch
if (not args.test) and (not PLOT_ONLY):  # don't overwrite if just replotting
    with open(os.path.join(outdir, "hyperparameter_ordering_metainfo.json"), "w") as f:
        json.dump(data.hp_ordering, f, indent=4, default=lambda x: str(x.__class__))
    with open(os.path.join(outdir, "nleaves_min_max.json"), "w") as f:
        json.dump([data.nleaves_min_dict, data.nleaves_max_dict], f, indent=4)
    # save settings
    with open(os.path.join(outdir, "settings.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)

# ----------------------------------------
# LOAD IN DATA - PE samples and injections
# ----------------------------------------

for attr in ["PE_samples", "injs"]:
    path = getattr(args, attr)
    if not os.path.exists(path):
        setattr(args, attr, os.path.join(cfg.get("lvk_out_path"), path))

params = sorted(list(data.params))
with np.load(args.PE_samples) as f:
    PE_samples = {
        k: xp.asarray(v, dtype=xp.float32) for k, v in f.items() if k in params + ["prior"]
    }  # needed for initialization
    event_names = np.asarray(f["event_names"], dtype=str) if "event_names" in f.files else None

with np.load(args.injs) as f:
    injections = {
        k: xp.asarray(v, dtype=xp.float32) for k, v in f.items() if k in params + ["prior", "w"]
    }
    injections["total_generated"] = int(f["total_generated"])
    injections["TOBS"] = xp.asarray(float(f["Tobs_yr"]), dtype=xp.float32)

if args.ignore_events:
    event_names_list = np.loadtxt(
        args.PE_samples.replace("PE_samples.npz", "waveforms.txt"), dtype=str
    )[:, 0]
    inds_delete = xp.asarray(
        [int(np.where(event_names_list == name)[0][0]) for name in args.ignore_events]
    )

    # remove events from PE_samples
    for k, v in PE_samples.items():
        if v.size > 1:
            PE_samples[k] = xp.delete(v, inds_delete, axis=0)
    if event_names is not None:
        event_names = np.delete(event_names, utils.to_numpy(inds_delete))

    print(
        f"Removed {len(inds_delete)} event(s) from PE samples ({len(event_names_list)} -> {PE_samples['chi_eff'].shape[0]} events)"
    )

if args.test:
    # downsample injections and PE
    PE_samples = {k: v[:, :500] for k, v in PE_samples.items()}
    injections = {k: v[:500] if xp.asarray(v).size > 1 else v for k, v in injections.items()}

# -----------------------------
# TEST LIKELIHOOD FUNCTION
# -----------------------------
# maybe test loglike function above and see if it actually reprodjces P(m1) as expected

loglike_kwargs = {"min_sep": args.min_sep, "group_chunk_size": args.group_chunk_size}

if args.test:
    ngroups_test = 5

    for _ in range(10):
        print("")
        test_hp_vals = [
            priors[branch_name].rvs(size=(ngroups_test * data.nleaves_max_dict[branch_name],))
            for branch_name in data.branch_names
        ]

        test_groups = [
            xp.repeat(xp.arange(ngroups_test), data.nleaves_max_dict[branch_name])
            for branch_name in data.branch_names
        ]

        ll = data.loglike(
            test_hp_vals, test_groups, PE_samples, injections, debug=True, **loglike_kwargs
        )

        print("test logl", ll)
        if np.any(~np.isfinite(ll)):
            raise ValueError("test logl has non-finite values")

# =====================
# Sampler initialization
# =====================

backend_path = os.path.join(outdir, "backend.h5")
backend = HDFBackend(backend_path)
if os.path.exists(backend_path):
    if backend.initialized and backend.iteration > 0:
        state = backend.get_last_sample()
        print(f"Initializing from previous run at {backend_path} with iter={backend.iteration}")
    else:
        os.remove(backend_path)  # faulty backend / it hasn't been run

if not os.path.exists(backend_path):
    # Initialize parameter chains (walker positions)
    # coords is a dict keyed by branch name each of shape:
    #   (ntemps, nwalkers, nleaves_max, ndims) drawn from priors.
    print("Drawing initial coordinates...")
    coords = {
        branch_name: priors[branch_name].rvs(
            size=(
                args.ntemps,
                args.nwalkers,
                data.nleaves_max_dict[branch_name],
            )
        )
        for branch_name in data.branch_names
    }

    # check that the first mass branch has one guy that has support at m=10
    mchar = 10
    mock_data = {"mass_1_source": mchar, "mass_ratio": 0.9, "mass_2_source": mchar * 0.9}
    b_idx_check = [
        b_idx for b_idx in range(len(data.branch_names)) if "mass" in data.hp_ordering[b_idx]
    ][0]

    for b_idx, bn in enumerate(data.branch_names[:-1]):
        ncomps = data.nleaves_max_dict[bn]
        if (ncomps <= 1 or args.min_sep == 0) and b_idx != b_idx_check:
            continue

        bad_init = True
        bad_init_count = 0

        while bad_init:
            if bad_init_count >= 1000:
                raise ValueError(
                    f"Initialization failed after {bad_init_count} attempts. Check the prior ranges for branch {bn}"
                )
            bad_init_count += 1

            bad_groups_mask = xp.zeros((args.ntemps, args.nwalkers), dtype="bool")
            input_hps = [xp.asarray(coords[bn]) for bn in data.branch_names]

            if b_idx == b_idx_check:
                pdf_vals = data.eval_param_model(mock_data, b_idx, "mass", input_hps).reshape(
                    (args.ntemps, args.nwalkers, ncomps)
                )
                bad_groups_mask = xp.logical_or(
                    bad_groups_mask, xp.all(pdf_vals < 1e-12, axis=-1)
                )  # no leaves have support at mchar
                print(
                    f"    frac of bad groups from p(m) check ({bn}): {xp.sum(bad_groups_mask)}/{bad_groups_mask.size}"
                )

            if (
                ncomps > 1 and args.min_sep > 0
            ):  # make sure leaves are initialized sufficiently far apart
                feats, _ = data.compute_branch_moment_features(b_idx, input_hps)
                minsep_bad_groups_mask = ~utils.check_min_separation(
                    feats, min_sep=args.min_sep, xp=xp
                )
                print(
                    f"    frac of bad groups from min sep check ({bn}): {xp.sum(minsep_bad_groups_mask)}/{minsep_bad_groups_mask.size}"
                )
                bad_groups_mask = xp.logical_or(bad_groups_mask, minsep_bad_groups_mask)

            bad_groups = np.nonzero(utils.to_numpy(bad_groups_mask))
            n_bad_groups = len(bad_groups[0])
            if n_bad_groups:
                # redraw the bad walker/temps
                print("    Redrawing", n_bad_groups, f"bad groups for branch {bn}")
                rvs_shape = (n_bad_groups, ncomps)
                coords[bn][bad_groups + (slice(None),)] = priors[bn].rvs(size=rvs_shape)
            else:
                bad_init = False

    # sort all the leaves by m1
    input_hps = [xp.asarray(coords[bn]) for bn in data.branch_names]
    for b_idx, bn in enumerate(data.branch_names[:-1]):
        if "mass" in data.hp_ordering[b_idx] and data.nleaves_max_dict[bn] > 1:
            moments = data.eval_param_moments(b_idx, "mass", input_hps)
            mu = moments["mass_1_source"][0]
            sort_idx = xp.argsort(mu, axis=-1)
            # Broadcast and take
            sorted_coords = xp.take_along_axis(
                input_hps[b_idx], xp.expand_dims(sort_idx, axis=-1), axis=-2
            )
            coords[bn] = utils.to_numpy(sorted_coords)

    # Build the initial sampler state. `inds` is optional and not defined here;
    # we rely on `provide_groups=True` in EnsembleSampler to manage groups.
    state = State(coords)


### Define MCMC moves to use (Gibbs/Stretch/Random-walk)

rj_branches = [
    bn for bn in data.branch_names if data.nleaves_max_dict[bn] > data.nleaves_min_dict[bn]
]
# use stretch move for all non-RJ branches
nonrj_sampling_setup = [(bn, None) for bn in data.branch_names if bn not in rj_branches]
stretch_move = StretchMove(gibbs_sampling_setup=nonrj_sampling_setup, live_dangerously=True)
moves = [(stretch_move, 0.3)]

# For non-RJ branches, we update each component AND the hyperparameters corresponding to each GW
# parameter independently, with a diagonal Gaussian proposal given by the covariance matrix (one scale per
# parameter), which is specified in the prior.
rj_moves = []  # RJMCMC moves to change model dimensionality (add/remove leaves)
for branch_name in rj_branches:
    branch_idx = data.branch_names.index(branch_name)
    branch_ndims = data.branch_dims[branch_idx]
    ncomp_max = data.nleaves_max_dict[branch_name]

    # `cov` is diagonal. Requires len(covs_all) == len(data.branch_names).
    cov_branch = np.ones(branch_ndims)
    cov_branch[: covs_all[branch_idx].size] = covs_all[branch_idx]
    cov_branch = np.diag(cov_branch)
    cov = {branch_name: cov_branch}

    # move each leaf AND parameter separately
    for param_name in utils.skip_dunder(data.hp_ordering[branch_idx].keys()):
        param_dict = data.hp_ordering[branch_idx][param_name]
        param_dim_inds = utils.recursive_get(param_dict, "col_idx")
        for leaf_idx in range(ncomp_max):
            gibbs_array_i = np.zeros((ncomp_max, branch_ndims), dtype="bool")
            gibbs_array_i[leaf_idx, param_dim_inds] = True

            gaussian_move_i = GaussianMove(cov, gibbs_sampling_setup=(branch_name, gibbs_array_i))
            if args.test:
                print(branch_name, gibbs_array_i)
            moves.append(
                (gaussian_move_i, 0.2)
            )  # '0.2' is the relative weight of which proposal to do

    # update rates together
    gibbs_rate_arr = np.zeros((ncomp_max, branch_ndims), dtype="bool")
    gibbs_rate_arr[:, -1] = True

    simplex_move = RateMove(
        RATE_FACTOR,
        branch_name,
        type="orthogonal",
        gibbs_sampling_setup=(branch_name, gibbs_rate_arr),
    )
    tot_move = RateMove(
        RATE_FACTOR,
        branch_name,
        type="radial",
        gibbs_sampling_setup=(branch_name, gibbs_rate_arr),
    )
    moves.append((simplex_move, 0.4))
    moves.append((tot_move, 0.4))

    # moves.append((GaussianMove(cov, gibbs_sampling_setup=(branch_name, gibbs_rate_arr)), 0.4))
    if args.test:
        print(branch_name, gibbs_rate_arr)

    if branch_name in rj_branches:
        if args.rj_num_try > 1:  # use multiple try MCMC
            rj_move = MTDistGenMoveRJ(
                priors,
                num_try=args.rj_num_try,
                nleaves_min=data.nleaves_min_dict,
                nleaves_max=data.nleaves_max_dict,
                gibbs_sampling_setup=branch_name,  # change one branch at a time
            )
        else:
            rj_move = DistributionGenerateRJ(
                priors,
                nleaves_min=data.nleaves_min_dict,
                nleaves_max=data.nleaves_max_dict,
                gibbs_sampling_setup=branch_name,  # change one branch at a time
            )

    rj_moves.append((rj_move, 1.0))

if not rj_moves:
    rj_moves = False

# -----------------------------
# RUN SAMPLER, SAVE OUTPUT
# -----------------------------

# prepare logpath
logpath = os.path.join(outdir, "info.txt")
if os.path.exists(logpath):
    utils.print_to(logpath, "\n\n" + "#" * 50 + "\n")

# Run the MCMC
nsteps = 50 if args.test else args.nsteps
if args.kde_update > 0:
    burn = 100 if args.test else args.burn
else:
    burn = 5 if args.test else args.burn
    if args.test:
        backend = None

ensemble_kwargs = dict(
    nwalkers=args.nwalkers,
    ndims=data.branch_dims,
    log_like_fn=data.loglike,
    priors=priors,
    args=[PE_samples, injections],
    kwargs=loglike_kwargs,
    tempering_kwargs=dict(ntemps=args.ntemps, Tmax=args.Tmax),
    nbranches=len(data.branch_names),
    branch_names=data.branch_names,
    nleaves_max=data.nleaves_max_dict,
    nleaves_min=data.nleaves_min_dict,
    provide_groups=True,
    moves=moves,
    vectorize=True,
    rj_moves=rj_moves,
    fill_zero_leaves_val=-INF,
    backend=backend,
    track_moves=not (args.kde_update > 0),
)

if not PLOT_ONLY:
    # this wrapper is just to handle errors in case of incompatible existing backend
    ensemble = utils.EnsembleSamplerWrapper(**ensemble_kwargs)
    if args.kde_update > 0 and rj_moves:
        kde_update_fn = UpdateKDEMove(log=logpath)
        # we need to store chain during burn in, so reset backend manually
        if burn:
            print(f"\nBurning in manually for {burn} steps...")
            state = ensemble.run_mcmc(state, nsteps=burn, progress=True)
            kde_update_fn(burn, state, ensemble)  # final update of kde rj moves

            rj_move_name = ensemble.rj_moves[0].__class__.__name__
            tot_rj_leaves = sum([data.nleaves_max_dict[bn] for bn in rj_branches])
            ensemble.reset(
                nleaves_max=data.nleaves_max_dict,
                ntemps=args.ntemps,
                branch_names=data.branch_names,
                rj=True,
                key_order=ensemble.key_order,
            )  # reset backend
            print("...done.")
        else:
            assert backend.iteration > 0, "Backend needs to have ran if burn=0 and kde_update>0"
            kde_update_fn(backend.iteration, state, ensemble)

        # already done with burn
        burn = 0

        # want to update during sampling as well
        ensemble.update_fn = kde_update_fn
        ensemble.update_iterations = 30 if args.test else args.kde_update

    print(f"Sampling for {nsteps} steps{f' with {burn} burn-in steps' if burn else ''}...")
    last_state = ensemble.run_mcmc(state, nsteps, burn=burn, progress=True)

    if args.test:
        from shutil import rmtree

        rmtree(outdir)  # clear junk if testing
        exit()

    print(f"Saved backend to {backend_path}")

    # ------------------------------
    # Sampler diagnostics / thinning
    # ------------------------------

    logP = ensemble.get_log_posterior()[:, 0]
    print("Maximum log posterior:", np.amax(logP))
    print(
        "Fraction of samples with -inf log posterior:",
        np.sum(logP < -(INF**0.5)) / logP.size,
    )
    print(
        "# of samplers stuck at -inf:",
        np.sum(np.sum(logP < -(INF**0.5), axis=0) > nsteps / 2),
    )
    nsamps, nwalker = logP.shape

    converged = False
    while not converged:
        # compute discard automatically -  fit linear + flat
        discard = utils.get_discard_from_chain(logP) if args.discard is None else args.discard

        if discard > nsamps - 500:
            print("Sampler has not converged - running for another 1000 steps")
            last_state = ensemble.run_mcmc(last_state, 1000, burn=burn, progress=True)
            print(f"Saved backend to {backend}")
            print("Maximum log posterior:", np.amax(logP))
            logP = ensemble.get_log_posterior()[:, 0]
        else:
            converged = True

        # print diagnostics

        del ensemble
else:
    logP = backend.get_log_posterior()[:, 0]
    discard = utils.get_discard_from_chain(logP) if args.discard is None else args.discard
    last_state = state

# print diagnostics
utils.print_to(
    logpath, f"\n########## PRINTING DIAGNOSTICS -- iter = {backend.iteration} ##########\n"
)
acc_frac, acc_frac_std = utils.acceptance_fraction(backend)
utils.print_to(logpath, f"Total acceptance fraction: {acc_frac:.2g} +/- {acc_frac_std:.2g}")
if backend.rj:
    rj_acc_frac, rj_acc_frac_std = utils.rj_acceptance_fraction(backend)
    utils.print_to(
        logpath, f"Total RJ acceptance fraction: {rj_acc_frac:.2g} +/- {rj_acc_frac_std:.2g}"
    )
if args.ntemps > 1:
    utils.print_to(logpath, f"Swap acceptance fraction: {utils.swap_acceptance_fraction(backend)}")

####### PLOT ALL CHAINS ######
chain = backend.get_chain(temp_index=0)
nsteps, nwalkers = logP.shape
n_glob_params = len(data.hp_ordering[-1]["__latex__"])
n_chain_plots = (
    2 + len(rj_branches) + n_glob_params
)  # logP, total rate, nleaves, global parameters
fig, axes = plt.subplots(n_chain_plots, 1, figsize=(5, 2 * n_chain_plots), sharex=True)

# logP
plot.plot_chains(logP, ax=axes[0], xlabel=None, ylabel="log P")
axes[0].set_ylim(np.nanpercentile(logP[discard:].ravel(), [1, 99]))
# total rate
R0_tot = np.nansum(
    np.concatenate([chain[bn][..., -1] for bn in data.branch_names[:-1]], axis=-1), axis=-1
)  # (nsteps, nwalkers)
plot.plot_chains(R0_tot, ax=axes[1], xlabel=None, ylabel=r"$\mathcal{R}(0)$")
axes[1].set_ylim(np.nanpercentile(R0_tot[discard:].ravel(), [1, 99]))
# n leaves per branch
for i, bn in enumerate(rj_branches):
    ax_idx = i + 2
    nleaves_branch = np.count_nonzero(
        utils.leaf_active_mask(chain[bn]), axis=-1
    )  # (nsteps, nwalkers)
    plot.plot_chains(
        nleaves_branch, ax=axes[ax_idx], xlabel=None, ylabel=f"{bn} " + r"$n_{\mathrm{leaves}}$"
    )
    axes[ax_idx].set_ylim(0, data.nleaves_max_dict[bn])

# global parameters
for i, param_latex in enumerate(data.hp_ordering[-1]["__latex__"]):
    ax_idx = -n_glob_params + i
    dat = chain["global"][..., 0, i]
    plot.plot_chains(dat, ax=axes[ax_idx], xlabel=None, ylabel=param_latex)

axes[-1].set_xlabel("step")
axes[-1].set_xlim(0, nsteps)
for ax in axes:
    # plot discard line
    ax.axvline(discard, color="r", ls="--", lw=1)

plt.tight_layout(h_pad=0.1)
plt.savefig(os.path.join(figpath, "chains.png"))
print("Saved chains in chains.png")
plt.close()

#################################

# compute thin
thin, tau = utils.get_thin_from_chain(backend.get_chain(discard=discard), return_tau=True)
utils.print_to(logpath, f"\nIntegrated autocorrelation times: {tau}")

chain = backend.get_chain(discard=discard, thin=thin, temp_index=0)
logP = backend.get_log_posterior(discard=discard, thin=thin)
nsamples = logP[:, 0].size
utils.print_to(logpath, f"Using thin {thin} and discard {discard} ({nsamples} samples)")


REPLOT = args.replot or not PLOT_ONLY

def save_or_not(path):  # determine if it needs to be replotted
    if "/" not in path:
        if path.endswith(("png", "pdf")):
            path = os.path.join(figpath, path)
        elif path.endswith(("npy", "npz")):
            path = os.path.join(datapath, path)
    if "*" in path:
        matches = glob(path)
        res = len(matches) == 0 or REPLOT
    else:
        res = (not os.path.exists(path)) or REPLOT
    if not res:
        print(f"Skipping {path}")
    return res


# plot temperature convergence
temp_plotname = "log_posterior_temps.png"
logP_low, logP_high = np.nanpercentile(logP[:, 0], [0.1, 99.9])
if args.ntemps > 1 and save_or_not(temp_plotname):
    beta_mean = np.mean(backend.get_betas(), axis=0)
    for i in range(logP.shape[1]):
        logP_temp = logP[:, i].ravel()
        logP_temp = logP_temp[logP_temp > logP_low]
        _ = plt.hist(
            logP_temp,
            histtype="step",
            bins=40,
            label=rf"$\beta \sim {beta_mean[i]:.2f}$",
            log=True,
        )
    plt.xlim(logP_low, logP_high)
    plt.legend()
    plt.savefig(os.path.join(figpath, temp_plotname))
    print(f"Saved log posterior temperature convergence in {temp_plotname}")
    plt.close()

# ----------------------------------
# Post-processing: labeling and PPDs
# ----------------------------------

bf_threshold = 0.1

# RJ model selection:
# Define a model as the tuple of per-branch active component counts across the
# local RJ branches. We only visualize models that have a BF > 0.05 compared to
# the preferred model.

# Here, we identify the different models and compute draw indices per model, evidences,
# and select the top models for plotting.

samples_dict = {
    bn: chain[bn].reshape(-1, data.nleaves_max_dict[bn], data.branch_dims[b_idx], copy=True)
    for b_idx, bn in enumerate(data.branch_names)
}
nsamples = samples_dict["global"].shape[0]
del chain

sigs, bayes_factors = [], []
sig_names, model_inds = [], []
n_labelled_comps = np.array(
    [data.nleaves_max_dict[bn] for bn in data.branch_names[:-1]], dtype=np.int32
)
if rj_branches:
    model_counts = []
    for bn in rj_branches:
        active_b = utils.leaf_active_mask(samples_dict[bn])
        b_nleaves_samples = np.sum(active_b, axis=1).astype(np.int32)
        model_counts.append(b_nleaves_samples)

    if save_or_not("ncomponents.png"):
        fig, axes = plt.subplots(
            len(model_counts), 1, sharex=True, figsize=(4, len(model_counts) * 1.5)
        )
        if len(model_counts) <= 1:
            axes = [axes]
        for ax, bn, b_nleaves_samples in zip(axes, rj_branches, model_counts, strict=True):
            k_max = data.nleaves_max_dict[bn]
            counts = np.bincount(b_nleaves_samples, minlength=k_max + 1)
            k = np.arange(k_max + 1)
            ax.bar(k, counts, width=0.85, align="center")
            ax.set_ylabel("Counts")
            ax.text(
                0.5,
                0.9,
                bn,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontweight="bold",
            )

        axes[-1].set_xlabel("Number of Components")
        axes[-1].set_xlim(0, data.nleaves_max_dict[bn] + 1)
        axes[-1].set_xticks(np.arange(data.nleaves_min_dict[bn], data.nleaves_max_dict[bn] + 1))

        plt.savefig(os.path.join(figpath, "ncomponents.png"))
        print("Saved posterior on # of components per RJ branch in ncomponents.png")
        plt.close()

    model_counts = np.stack(
        model_counts, axis=1
    )  # shape (nsamples, nrjbranches), of # of components of each branch
    sigs_all, inv, counts = np.unique(
        model_counts, axis=0, return_inverse=True, return_counts=True
    )
    sigs_all = [tuple([int(i) for i in sig]) for sig in sigs_all]
    # sigs are the unique model 'signatures', i.e. tuples with the # of components of each rj branch

    sig_names_all = [
        ", ".join(
            [
                f"{int(sig_count)} {rj_bn}"
                for rj_bn, sig_count in zip(rj_branches, sig, strict=True)
            ]
        )
        for sig in sigs_all
    ]

    # save relative evidences
    bayes_factors_all = counts / np.amax(counts)
    utils.print_to(logpath, "\nBAYES FACTORS (relative to preferred model):")
    pad = max([len(str(list(s))) for s in sig_names_all]) + 1
    for sig_name, bf in zip(sig_names_all, bayes_factors_all, strict=True):
        utils.print_to(logpath, f"{sig_name:<{pad}}: {bf:.5f}")

    for t, bf in enumerate(bayes_factors_all):
        if bf > bf_threshold:  # ignore insignificant models:
            idx = np.nonzero(inv == t)[0]
            model_inds.append(idx.copy())
            sigs.append(sigs_all[t])
            sig_names.append(sig_names_all[t])
            bayes_factors.append(float(bf))

    rj_labelled_comps = np.amax(np.stack(sigs, axis=0), axis=0)
    for rj_branch, ncomps in zip(rj_branches, rj_labelled_comps, strict=True):
        n_labelled_comps[data.branch_names.index(rj_branch)] = ncomps
    # this is the maximum number of components for each RJ branch across all (significant) models

    best_model_idx = np.argmax(bayes_factors)
else:  
    best_model_idx = 0
sigs.append(0)
sig_names.append(0)  # for all non-rj parameters
model_inds.append(np.arange(nsamples))
bayes_factors.append(-1)

samples_and_labels_datapath = os.path.join(datapath, "samples_and_labels.npy")
if os.path.exists(samples_and_labels_datapath) and not REPLOT:
    print("Loading samples and labels from samples_and_labels.npy")
    megadict = np.load(samples_and_labels_datapath, allow_pickle=True).item()
    samples_dict = megadict["samples"]
    labels_dict = megadict["labels"]
    feats_dict = megadict["feats"]
    cluster_centers_dict = megadict["cluster_centers"]
else:
    if rj_branches:
        print("\n Sorting samples via k-means...")

    labels_dict = {}
    feats_dict = {}
    feat_scales_dict = {}
    cluster_centers_dict = {}

    for branch_idx, branch_name in enumerate(data.branch_names[:-1]):
        samples_loc = samples_dict[branch_name]
        nleaves_max = data.nleaves_max_dict[branch_name]

        if branch_name in rj_branches:
            ncomp = n_labelled_comps[branch_idx]

            print(f"   ...sorting branch {branch_name} via k-means with {ncomp} components...")
            labels, feats, cluster_centers, feat_scales = data.label_branch_samples_kmeans(
                branch_idx=branch_idx,
                samples_dict=samples_dict,
                ncomp=ncomp,
                autoscale=True,  # scale params automatically instead of manually
                rng_seed=args.seed + branch_idx + 1,
            )
            samples_loc_sorted = samples_loc
            feat_scales_dict[branch_name] = feat_scales

            labels_dict[branch_name] = labels
            feats_dict[branch_name] = feats
            cluster_centers_dict[branch_name] = cluster_centers

        else:
            hps = data.hp_list_from_dict(samples_dict)
            feats, scales = data.compute_branch_moment_features(
                branch_idx, hps, autoscale=True, include_rate=True
            )
            feats = utils.to_numpy(feats) * scales

            if (
                nleaves_max > 1
            ):  # non-rj branch with more than 1 component, sort by last feat (rate)
                print(f"   ...sorting branch {branch_name} via p(m1) mean...")
                labels = utils.to_numpy(xp.argsort(xp.argsort(-feats[..., -1], axis=-1), axis=-1))
                # two argsorts to get rank
            else:
                labels = np.zeros_like(feats[..., 0], dtype="int")

            # get cluster centers from median
            branch_cluster_centers = np.empty(nleaves_max, feats.shape[-1])  # ncomp, nfeats
            for i in range(nleaves_max):
                branch_cluster_centers[i] = np.median(feats[labels == i], axis=0)

            feats_dict[branch_name] = feats
            labels_dict[branch_name] = labels
            cluster_centers_dict[branch_name] = branch_cluster_centers

    print("...done.")
    # 'samples_dict' is a dict of arrays of each branch with shape (nsamples, ncomps, nparams)

    # save samples_dict
    megadict = {
        "samples": samples_dict,
        "labels": labels_dict,
        "feats": feats_dict,
        "feat_scales": feat_scales_dict,
        "cluster_centers": cluster_centers_dict,
        "sigs": sigs,
        "sig_names": sig_names,
        "model_inds": model_inds,
        "bayes_factors": bayes_factors
    }
    np.save(
        os.path.join(datapath, "samples_and_labels.npy"),
        megadict,
        allow_pickle=True,
    )
    print("Saved samples_and_labels.npy")

# calculate ideal gaussian proposal covariance according to rule of thumb sigma = ~ 2.4 * sigma_data / sqrt(nd)

utils.print_to(logpath, "\n\nIdeal gaussian proposal covariances:")
utils.print_to(logpath, "")
for branch_name in rj_branches:
    branch_idx = data.branch_names.index(branch_name)
    branch_labels = labels_dict[branch_name]
    for param_name in utils.skip_dunder(data.hp_ordering[branch_idx].keys()):
        param_dict = data.hp_ordering[branch_idx][param_name]
        param_dim_inds = utils.recursive_get(param_dict, "col_idx")
        param_samples = samples_dict[branch_name][..., param_dim_inds]
        comp_proposal_vars = []
        for label_idx in range(n_labelled_comps[branch_idx]):
            comp_proposal_vars.append(
                4 * np.var(param_samples[branch_labels == label_idx], axis=0) / len(param_dim_inds)
            )
        proposal_vars = np.amin(
            np.stack(comp_proposal_vars), axis=0
        )  # take the smallest step per component
        proposal_vars_str = ", ".join([f"{v:.2g}" for v in proposal_vars])
        utils.print_to(logpath, f"{branch_name} {param_name}: {proposal_vars_str}")

# convert to rate at z=0.2, consistent with LVK figures

# assign colors to components
component_palette_unsorted = plot.skewt_colors if "skewt" in args.label else plot.set2

# assign names and colors to components - sort by cluster center of last column (rate)
comp_labels = []  # names
comp_label_sigs = []  # (branch_name, label_idx)
rate_cluster_centers = []  # cluster center of last feat - rate
for b_idx, branch_name in enumerate(data.branch_names[:-1]):
    labels = labels_dict.get(branch_name, None)
    ncomps = n_labelled_comps[b_idx]
    if ncomps > 1:
        for label_idx in range(ncomps):
            comp_label_sigs.append((branch_name, label_idx))
            label_alpha = utils.ALPHABET[label_idx]
            if len(data.branch_names) > 2:
                comp_label = f"{branch_name} {label_alpha}"
                comp_labels.append(comp_label)
            else:
                comp_labels.append(label_alpha)  # only 1 local branch
            rate_cluster_centers.append(cluster_centers_dict[branch_name][label_idx, -1])
    else:
        comp_labels.append(branch_name)
        comp_label_sigs.append((branch_name, 0))
        rate_cluster_centers.append(cluster_centers_dict[branch_name][0, -1])

# assign colors based on m1_cluster_centers
rate_cluster_centers = np.array(rate_cluster_centers)
component_palette = [
    component_palette_unsorted[int(i)] for i in np.argsort(np.argsort(-rate_cluster_centers))
]
# component palette is now sorted by the order of the branches and labels
comp_palette_dict = {}  # dict of list of colors for plotting convenience
start_idx = 0
for b_idx, branch_name in enumerate(data.branch_names[:-1]):
    ncomps = n_labelled_comps[b_idx]
    comp_palette_dict[branch_name] = component_palette[start_idx : start_idx + ncomps]
    start_idx += ncomps

# define and count submodels
tot_n_labelled_comps = len(comp_label_sigs)
submodel_sig_names_all, submodel_model_signames_all = [], []
if rj_branches:
    submodel_counts = np.zeros((nsamples, tot_n_labelled_comps), dtype=np.int32)
    for col_idx, (rj_bn, comp_idx) in enumerate(comp_label_sigs):
        submodel_counts[:, col_idx] += (labels_dict[rj_bn] == comp_idx).sum(axis=1)
    submodel_sigs_all, inv, counts_ = np.unique(
        submodel_counts, axis=0, return_inverse=True, return_counts=True
    )
    for sig in submodel_sigs_all:
        submodel_sig_names_all.append(
            ", ".join(
                [
                    comp_name
                    for comp_name, sig_count in zip(comp_labels, sig, strict=True)
                    for _ in range(sig_count)
                ]
            )
        )
        submodel_model_sig = []
        col_idx_start = 0
        for rj_bn in rj_branches:
            col_idx_end = col_idx_start + n_labelled_comps[data.branch_names.index(rj_bn)]
            submodel_model_sig.append(int(sum(sig[col_idx_start:col_idx_end])))
            col_idx_start = col_idx_end
        submodel_model_sig_name = sig_names_all[sigs_all.index(tuple(submodel_model_sig))]
        submodel_model_signames_all.append(submodel_model_sig_name)

    submodel_model_signames_all = {
        name: model_sig
        for name, model_sig in zip(
            submodel_sig_names_all, submodel_model_signames_all, strict=True
        )
    }
    # dictionary that maps submodel name -> which model it belongs to (e.g. 'PL A, gauss B' -> (1 PL, 1 gauss))

# submodel signatures are a tuple of counts corresponding to each component in comp_label_sigs
submodel_names, submodel_sigs = [], []
submodel_inds, submodel_bfs = [], []
if rj_branches:
    # same as above - relative evidences
    submodel_bf_all = counts_ / np.amax(counts)  # use same relative factor as above
    utils.print_to(logpath, "\n\nSUBMODEL BAYES FACTORS (relative to preferred model):")
    for sig_name, bf in zip(submodel_sig_names_all, submodel_bf_all, strict=True):
        utils.print_to(logpath, f"{sig_name:<{pad}}: {bf:.5f}")

    for t, bf in enumerate(submodel_bf_all):
        if bf > bf_threshold:  # ignore insignificant models:
            idx = np.nonzero(inv == t)[0]
            submodel_inds.append(idx.copy())
            submodel_sigs.append(submodel_sigs_all[t])
            submodel_names.append(submodel_sig_names_all[t])
            submodel_bfs.append(float(bf))

# add to samples_and_labels if not already
if "submodel_sigs" not in megadict:
    megadict.update({
        "submodel_sigs": submodel_sigs,
        "submodel_sig_names": submodel_names,
        "submodel_inds": submodel_inds,
        "submodel_bayes_factors": submodel_bfs,
    })
    np.save(
        os.path.join(datapath, "samples_and_labels.npy"),
        megadict,
        allow_pickle=True,
    )
    print("Saved submodel information to samples_and_labels.npy")


# get R02s aggregated by label
R0s_by_label, R02s_by_label, Ntots_by_label = [], [], []
for b_idx, bn in enumerate(data.branch_names[:-1]):
    labels = labels_dict.get(bn, None)
    R0s_by_label.append(data.get_branch_rates(b_idx, samples_dict, labels=labels, z=0))
    R02s_by_label.append(data.get_branch_rates(b_idx, samples_dict, labels=labels, z=0.2))
    Ntots_by_label.append(data.get_branch_Ntot(b_idx, samples_dict, labels=labels))
R0s_by_label = np.concatenate(R0s_by_label, axis=-1)  # (nsamples, ncomps_tot)
R02s_by_label = np.concatenate(R02s_by_label, axis=-1)  # (nsamples, ncomps_tot)
Ntots_by_label = np.concatenate(Ntots_by_label, axis=-1)

R02_tot = np.sum(R02s_by_label, axis=-1)
branching_fracs = Ntots_by_label / np.sum(Ntots_by_label, axis=-1, keepdims=True)

# R02_tot has shape (nsamples,) (no extra dimensions)

### CORNER PLOTS

def _corner_plot_wrapper(X, labels_cols, name=None, fig=None, title=None, share_axes=False, **corner_kwargs):

    X = np.asarray(X)

    if X.shape[1] <= 1 or X.ndim < 2:
        # plot histogram instead
        hist_kwargs = {
            "bins": plot.corner_defaults["bins"],
            "density": plot.corner_defaults["density"],
            "color": plot.corner_defaults["color"],
        }
        fig, ax = plt.subplots()
        ax.hist(X, histtype="step", **hist_kwargs)
        ax.set_xlabel(labels_cols[0])
        ax.set_ylabel(rf"$p({labels_cols[0].strip('$')})$")
        # quantiles
        qs = np.percentile(X.ravel(), 100 * np.asarray(plot.corner_defaults["quantiles"]))
        for q in qs:
            ax.axvline(q, color="k", linestyle="--")
        ax.set_title(rf"${qs[1]:.2f}^{{+{qs[2] - qs[1]:.2f}}}_{{-{qs[1] - qs[0]:.2f}}}$")

    else:
        good = np.all(np.isfinite(X), axis=1)
        X = X[good]
        if X.shape[0] < 10:
            print(f"Skipping corner plot for {name}: too few finite samples ({X.shape[0]})")
            return

        input_kwargs = plot.corner_defaults.copy()
        input_kwargs.update(corner_kwargs)

        if "range" not in input_kwargs:
            ranges = []
            for col in X.T:
                ranges.append(
                    (float(np.nanpercentile(col, 0.1)), float(np.nanpercentile(col, 99.9)))
                )
            if share_axes:
                ranges_ = [(min([x[0] for x in ranges]), max([x[1] for x in ranges]))] * len(ranges)
                ranges = ranges_
            input_kwargs["range"] = ranges

        if fig is None:
            figlen = 1.8 * X.shape[1]
            fig = plt.figure(figsize=(figlen, figlen))
        fig = corner(X, labels=labels_cols, fig=fig, **input_kwargs)

        if share_axes: # plot diagonal line
            axes = fig.axes
            n = X.shape[1]
            inds = np.tril_indices(n, k=1)
            inds_flat = n*inds[0] + inds[1]
            for i in inds_flat:
                axes[i].plot([ranges[0][0], ranges[0][0]], [ranges[0][1], ranges[0][1]], color="darkgray", ls="--", lw=0.7)

    if title:
        fig.suptitle(title, y=1.02)

    if name is None:
        return fig
    safe_name = utils.get_safe_fn(name)  # split by non-alphanumeric then join by '_'
    outpath = os.path.join(figpath, f"corner_{safe_name}.pdf")
    fig.savefig(outpath)
    plt.close(fig)
    print(f"Saved corner_{safe_name}.pdf")


# plot component branching fractions + rates, color coding for different models
# plot the component if in any model it contributes more than 1%
nonsmall_comps = []
for inds in model_inds:
    nonsmall_comps.append(np.nonzero(np.median(branching_fracs[inds], axis=0) > 0.001)[0])
nonsmall_comps = np.unique(np.concatenate(nonsmall_comps))
nonsmall_comp_label_sigs = [comp_label_sigs[int(i)] for i in nonsmall_comps]
nonsmall_comp_labels = [comp_labels[int(i)] for i in nonsmall_comps]

if not args.skip_corner:  # skip for debugging
    print("Preparing corner plots")

    # plot global branch
    if save_or_not("corner_global.pdf"):
        X = samples_dict["global"][:, 0]
        _corner_plot_wrapper(X, data.hp_ordering[-1]["__latex__"], "global")

    for data_arr, data_name in zip(
        [branching_fracs, R02s_by_label], ["branching_fracs", "rates"], strict=True
    ):
        if save_or_not(f"corner_{data_name}.pdf"):
            ranges = [
                (0.0, float(np.nanpercentile(data_arr[:, comp], 99.9))) for comp in nonsmall_comps
            ]
            if rj_branches:  # plot models in different colors
                fig = None
                handles = []
                for model_idx, (sig, inds) in enumerate(
                    zip(sig_names[:-1], model_inds[:-1], strict=True)
                ):
                    c = mpl.colors.to_hex(plot.tab10[model_idx])
                    fig = _corner_plot_wrapper(
                        data_arr[inds][:, nonsmall_comps],
                        nonsmall_comp_labels,
                        name=None,
                        fig=fig,
                        color=c,
                        fill_contours=False,
                        no_fill_contours=True,
                        contour_kwargs=dict(linewidths=2),
                        smooth=1.5,
                        bins=25,
                        hist_kwargs=dict(histtype="stepfilled", color=c, alpha=0.7),
                        range=ranges,
                        show_titles=False,
                        quantiles=None,
                    )
                    bf = bayes_factors[model_idx]
                    handles.append(
                        mpl.lines.Line2D(
                            [], [], color=c, label=sig + f" ($\\mathcal{{B}}={bf:.2f})$"
                        )
                    )
                    # plot overall quantiles
                    fig = _corner_plot_wrapper(
                        data_arr[:, nonsmall_comps],
                        nonsmall_comp_labels,
                        name=None,
                        color="k",
                        range=ranges,
                        fig=fig,
                        **plot.invisible_corner_kwargs,
                    )
                    fontsize = 11 if len(nonsmall_comps) > 2 else 9
                    fig.legend(
                        handles=handles,
                        loc="upper right",
                        frameon=False,
                        fontsize=fontsize,
                    )
                    # fig.suptitle(r'$\mathcal{R}(z=0.2)$ [Gpc${}^{-3}$ yr${}^{-1}$]', y=1.02)
            else:
                fig = _corner_plot_wrapper(
                    data_arr[:, nonsmall_comps],
                    nonsmall_comp_labels,
                    name=None,
                    color="k",
                    range=ranges,
                )
            fig.savefig(os.path.join(figpath, f"corner_{data_name}.pdf"))
            print(f"Saved corner_{data_name}.pdf")

    # plot all components
    for comp_label, comp_label_sig in zip(
        nonsmall_comp_labels, nonsmall_comp_label_sigs, strict=True
    ):
        if save_or_not(f"corner_{utils.get_safe_fn(comp_label)}*pdf"):
            bn, comp_idx = comp_label_sig

            sb = samples_dict[bn]
            names = data.hp_ordering[data.branch_names.index(bn)]["__latex__"]
            if bn in rj_branches:
                X = sb[labels_dict[bn] == comp_idx]
            else:
                X = sb[:, comp_idx]
            _corner_plot_wrapper(X, names, comp_label)

            # also plot a separate corner plot for the most preferred model
            if rj_branches:
                best_model_inds = model_inds[best_model_idx]
                if bn in rj_branches:
                    X = sb[best_model_inds][labels_dict[bn][best_model_inds] == comp_idx]
                else:
                    X = sb[best_model_inds][:, comp_idx]
                _corner_plot_wrapper(X, names, f"{comp_label}_{sig_names[best_model_idx]}")

    if args.corner_param is not None and rj_branches:
        n_submodels_plot = min(2, len(submodel_bfs))
        # plot best 2 submodels
        for submodel_idx in np.argsort(submodel_bfs)[::-1][:n_submodels_plot]:
            X = []
            corner_param_labels = []
            # corner_param_ranges = []  # use prior ranges
            submodel_sig = submodel_sigs[submodel_idx]
            submodel_sample_inds = submodel_inds[submodel_idx]
            comp_sigs_ordered = [
                (rj_b, label_idx)
                for rj_b in rj_branches
                for label_idx in range(n_labelled_comps[data.branch_names.index(rj_b)])
            ]
            start_col_idx = 0
            # plot each branch
            for b_idx, (bn, p_idx) in enumerate(
                zip(data.branch_names, args.corner_param, strict=False)
            ):
                if p_idx == -1:
                    continue
                p_idx = p_idx % len(data.hp_ordering[b_idx]["__latex__"])
                p_latex = data.hp_ordering[b_idx]["__latex__"][p_idx]
                param_range = (
                    data.branch_priordicts[b_idx][p_idx].min_val,
                    data.branch_priordicts[b_idx][p_idx].max_val,
                )
                if bn in rj_branches:
                    # plot component parameter of branch
                    for comp_count, (rj_b, label_idx) in zip(
                        submodel_sig, comp_sigs_ordered, strict=True
                    ):
                        if comp_count <= 0 or rj_b != bn:
                            continue
                        comp_name = plot.textsc_ify(
                            f"{rj_b} {utils.ALPHABET[label_idx]}"
                            if len(rj_branches) > 1
                            else utils.ALPHABET[label_idx]
                        )
                        label_filter = labels_dict[rj_b][submodel_sample_inds] == label_idx
                        samples_filtered = samples_dict[rj_b][submodel_sample_inds][label_filter]
                        if comp_count == 1:
                            X.append(samples_filtered[..., p_idx])
                            corner_param_labels.append(f"{p_latex} ({comp_name})")
                            # corner_param_ranges.append(param_range)
                        else:  # more than one of the same label which is not great, sort by the value
                            samples_filtered = samples_filtered.reshape(
                                -1, comp_count, samples_filtered.shape[-1]
                            )
                            samples_filtered = np.sort(samples_filtered, axis=1)
                            for i in range(comp_count):
                                X.append(samples_filtered[:, i, p_idx])
                                corner_param_labels.append(f"{p_latex} ({comp_name}{i + 1})")
                                # corner_param_ranges.append(param_range)

                else:  # just 1 component in that branch
                    X.append(samples_dict[bn][submodel_sample_inds][..., p_idx].ravel())
                    corner_param_labels.append(f"{p_latex} ({utils.textsc_ify(bn)})")
                    # corner_param_ranges.append(param_range)

            X = np.stack(X, axis=-1)
            submodel_name = submodel_names[submodel_idx]
            _corner_plot_wrapper(X, corner_param_labels, name=f"custom_param_{submodel_name}", share_axes=True)

### extract and plot PPDs
print("\nPlotting PPDs")

draw_ctxs = {}
for model_or_submodel, sig_names_, model_inds_ in zip(
    ("model", "submodel"),
    (sig_names, submodel_names),
    (model_inds, submodel_inds),
    strict=True,
):
    if sig_names_:
        draw_ctx_fn = f"{model_or_submodel}_draw_ctxs.npy"
        if save_or_not(draw_ctx_fn):
            # prepare draws for each models
            draw_ctx = {}
            for sig_name, inds in zip(sig_names_, model_inds_, strict=True):
                samples_input, labels_input = [], []
                if inds.size > args.ndraws:
                    sel_inds_of_model_inds = np.arange(args.ndraws) * int(inds.size // args.ndraws)
                    sel_inds = inds[sel_inds_of_model_inds] # thin out
                else:
                    sel_inds_of_model_inds = np.arange(len(inds))
                    sel_inds = inds
                for bn in data.branch_names:
                    samples_input.append(utils.to_numpy(samples_dict[bn][sel_inds]))
                    if bn in labels_dict:
                        labels_input.append(np.asarray(labels_dict[bn][sel_inds]))
                    else:
                        labels_input.append(None)
                R02_tot_draws = R02_tot[sel_inds]
                R02_labelled_draws = R02s_by_label[sel_inds]

                R0_labelled_draws = R0s_by_label[sel_inds]
                Ntot_labelled_draws = Ntots_by_label[sel_inds]

                draw_ctx[sig_name] = {
                    "sel_inds": sel_inds,
                    "sel_inds_of_model_inds": sel_inds_of_model_inds,
                    "samples_input": samples_input,
                    "labels_input": labels_input,
                    "R02_labelled_draws": R02_labelled_draws,
                    "R0_labelled_draws": R0_labelled_draws,
                    "Ntot_labelled_draws": Ntot_labelled_draws,
                    "R02_tot_draws": R02_tot_draws,
                }
            # save
            np.save(os.path.join(datapath, draw_ctx_fn), draw_ctx, allow_pickle=True)
            print(f"Saved {draw_ctx_fn}")
            draw_ctxs[model_or_submodel] = draw_ctx
        else:
            print(f"Loading {draw_ctx_fn}")
            draw_ctxs[model_or_submodel] = np.load(
                os.path.join(datapath, draw_ctx_fn), allow_pickle=True
            ).item()

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

    branch_name = data.branch_names[branch_idx]
    ncomps = n_labelled_comps[branch_idx]
    labels = draw_ctx["labels_input"][branch_idx]

    hyperparams = draw_ctx["samples_input"]
    if param_name == "redshift":
        rate = data.get_rates(branch_idx, hyperparams, z=0)  # rate at z=0
    else:  # want to plot rate at z=0.2, consistent with LVK
        rate = data.get_rates(branch_idx, hyperparams, z=0.2)  # (nsamples, ncomp)

    leaf_pdf = utils.to_numpy(data.eval_param_marginals(x, branch_idx, param_name, hyperparams))
    leaf_pdf = np.nan_to_num(leaf_pdf, nan=0.0, posinf=0.0, neginf=0.0)

    # leaf_pdf has shape (nsamples, ncomps, ngrid)
    if branch_name in rj_branches and use_labels:
        # aggregate by label (sum of rate-weighted leaves with same label, per sample)
        rate_ppd_by_label = np.zeros(
            (ncomps, leaf_pdf.shape[0], leaf_pdf.shape[-1]), dtype=np.float64
        )  # ncomps, nsamples, ngrid
        rate_by_label = np.zeros((ncomps, leaf_pdf.shape[0]), dtype=np.float64)
        for label in range(ncomps):  # label assigned by kmeans to each leaf
            w = (rate * (labels == label)).astype(np.float64)
            rate_ppd_by_label[label] = np.sum(leaf_pdf * w[..., None], axis=1)
            rate_by_label[label] = np.sum(w, axis=1)
    else:
        rate_by_label = np.swapaxes(rate, 0, 1)
        rate_ppd_by_label = np.swapaxes(leaf_pdf * rate[..., None], 0, 1)

    return rate_ppd_by_label, rate_by_label
    # output shapes: (ncomps, nsamples, ngrid), (ncomps, nsamples), (ngrid)

popsummary_path = args.popsummary_path 
if popsummary_path is None:
    if cfg.get("o4b-astro_path", None) is not None:
        popsummary_path = os.path.join(cfg.get("o4b-astro_path"), "popsummary_files")
        plot_cat = "GWTC5"
    elif cfg.get("o4a-astro_path", None) is not None:
        popsummary_path = os.path.join(cfg.get("o4a-astro_path"), "data_release")
        plot_cat = "GWTC4"
    else:
        plot_cat = None 
else:
    plot_cat = "GWTC5" if "o4b-astro" in popsummary_path else (
        "GWTC4" if "o4a-astro" in popsummary_path else None
    )

res, spin_res = None, None 
if popsummary_path is not None: 
    from popsummary.popresult import PopulationResult
    if plot_cat == "GWTC5":
        res = PopulationResult(
            fname=os.path.join(
                popsummary_path,
                "gwtc5_updated_default_mmax_mass_TwoPeakBrokenPowerLawSmoothedMassDistribution_redshift_PowerLawRedshift_magnitude_iid_spin_magnitude_gaussian_tilt_iid_spin_orientation_popsummary_result.h5",
            )
        )
        spin_res = PopulationResult(
            fname=os.path.join(
                popsummary_path,
                "260512_skewNormalChiEffChiP_varcut1/popsummary_var_cut_1.0_with2Drates.h5",
            )
        )
    elif plot_cat == "GWTC4":
        res = PopulationResult(
            fname=os.path.join(
                popsummary_path,
                "BBHMassSpinRedshift_BrokenPowerLawTwoPeaks_GaussianComponentSpins_PowerLawRedshift.h5",
            )
        )
        spin_res = PopulationResult(
            fname=os.path.join(
                popsummary_path,
                "BBHSpin_EpsSkewNormalChiEff.h5",
            )
        )

def plot_model_ppd(
    param_name, color_tot="cornflowerblue", use_labels=True, plot_submodels=False, max_plot=4,
):
    """
    Plot PPDs using one subplot per RJ model (top-N by posterior frequency).

    Components are grouped by (branch_name, label_id). For each model subplot, we
    evaluate PPDs on a random subsample of posterior draws belonging to that RJ
    model signature, and plot ppd draws as well as the 90 CI for the total ppd.
    """

    # get right draw ctx
    if plot_submodels:
        model_draw_ctxs = draw_ctxs["submodel"]
        bfs = submodel_bfs
    else:
        model_draw_ctxs = draw_ctxs["model"]
        bfs = bayes_factors

    # sort by bf
    sig_names = list(model_draw_ctxs.keys())
    bfs, sig_names = utils.sort_lists(bfs, sig_names, strict=False, reverse=True)

    if sig_names[-1] == 0: # exclude total
        sig_names = sig_names[:-1]
        bfs = bfs[:-1]

    nplot = min(len(sig_names), max_plot)  # at most max_plot panels
    bfs = bfs[:nplot]
    sig_names = sig_names[:nplot]

    param_name_check = "mass" if param_name.startswith("mass") else param_name
    # assumes joint mass model -- currently hard coded in

    x = data.data_grid[param_name]
    dict_out = {}  # save ppd by component
    if any(param_name_check in hps for hps in data.hp_ordering[:-1]):
        # local parameter

        if param_name == "mass_1_source" or param_name == "mass_2_source":
            figsize = (8, 2 * nplot)
        else:
            figsize = (6, 2 * nplot)
        fontsize = 11

        fig, axes = plt.subplots(nplot, 2, figsize=figsize, sharex=True, sharey=True)
        if nplot == 1:
            axes = [axes]

        for model_idx, (row_axes, sig) in enumerate(zip(axes, sig_names, strict=True)):
            # plot each model
            ax_comp, ax_tot = row_axes

            # plot GWTC-4 on all plots
            plot.setup_and_plot_GWTC(param_name, ax_comp, res=res, spin_res=spin_res, label=None, catalog=plot_cat)
            plot.setup_and_plot_GWTC(param_name, ax_tot, res=res, spin_res=spin_res, catalog=plot_cat)

            draw_ctx = model_draw_ctxs[sig]

            ppd_by_comp = []
            rate_by_comp = []

            for branch_idx, branch_name in enumerate(data.branch_names[:-1]):
                palette = comp_palette_dict[branch_name]
                if param_name_check not in data.hp_ordering[branch_idx]:
                    continue
                branch_rate_ppd_by_label, branch_rate_by_label = _branch_label_ppd(
                    x, param_name, branch_idx, draw_ctx, use_labels
                )
                
                ppd_by_comp.append(branch_rate_ppd_by_label)
                rate_by_comp.append(branch_rate_by_label)

                # each ppd is (nsamples, ngrid)
                for label_idx, ppd in enumerate(branch_rate_ppd_by_label):
                    if use_labels:
                        comp_label = comp_labels[comp_label_sigs.index((branch_name, label_idx))]
                        legend_label = plot.textsc_ify(comp_label)
                    else:
                        legend_label = None
                    color = palette[label_idx] if use_labels else color_tot

                    # want to plot chieff normalized
                    norm = 1 if param_name != "chi_eff" else draw_ctx["R02_tot_draws"][:, None]

                    plot.plot_ppds(
                        ax_comp,
                        x,
                        ppd / norm,
                        CI=None,
                        color=color,
                        label=legend_label,
                        rasterize=args.rasterize,
                    )
                    # use individual ppds for components

            # plot total
            ppd_by_comp = np.concatenate(ppd_by_comp, axis=0)
            rate_by_comp = np.concatenate(rate_by_comp, axis=0)
            tot_ppd = np.sum(ppd_by_comp, axis=0)
            # if param_name == "chi_eff":
            #     tot_ppd = tot_ppd / draw_ctx["R02_tot_draws"][:, None]
            plot.plot_ppds(ax_tot, x, tot_ppd, CI=90, color=color_tot, label="Total PPD")

            # formatting
            ax_tot.set_ylabel(None)
            if model_idx < nplot - 1:
                ax_comp.set_xlabel(None)
                ax_tot.set_xlabel(None)

            # label each row with model name and bayes factor (if RJ)
            plt_label = None
            if sig:
                bf = f"{bfs[model_idx]:.2f}" if bfs[model_idx] > 0.01 else f"{bfs[model_idx]:.2e}"
                plt_label = plot.textsc_ify(f"{sig} ($\\mathcal{{B}}={bf}$)")
            elif rj_branches:  # no label needed unless RJ
                plt_label = plot.textsc_ify("Combined")
            ax_tot.text(
                0.97,
                0.95,
                plt_label,
                transform=ax_tot.transAxes,
                ha="right",
                va="top",
                fontsize=fontsize,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2),
            )

            # save to dictionary
            dict_out[sig] = {}
            dict_out[f"{sig} rates"] = {}
            if use_labels:
                for comp_label, ppd, rate in zip(
                    comp_labels, ppd_by_comp, rate_by_comp, strict=True
                ):
                    dict_out[sig][comp_label] = utils.to_numpy(ppd)
                    dict_out[f"{sig} rates"][comp_label] = utils.to_numpy(rate)
            else:  # save all components by their original RJ leaves
                dict_out[sig] = utils.to_numpy(ppd_by_comp)
                dict_out[f"{sig} rates"] = utils.to_numpy(rate_by_comp)

        # legend
        all_handles, all_labels = [], []

        def _collect_handles_labels(axes):
            for ax in axes:
                h, l = ax.get_legend_handles_labels()
                if len(h):
                    l, h = utils.sort_lists(l, h)
                    for h_, l_ in zip(h, l, strict=True):
                        if l_ not in all_labels:
                            all_handles.append(h_)
                            all_labels.append(l_)

        if 0 not in sig_names and len(axes) > 1:
            _collect_handles_labels(axes[:-1, 0])  # first column
        _collect_handles_labels(axes[-1])  # last row

        ncol = len(all_handles)
        if ncol > 5:
            ncol = int(np.ceil(len(all_handles) / 2))
        fig.legend(
            handles=all_handles,
            labels=all_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=ncol,
            frameon=False,
            fontsize=fontsize,
        )
        plt.tight_layout()

    elif param_name_check in data.hp_ordering[-1]:
        # global parameter

        fig, ax = plt.subplots()
        draw_ctx = model_draw_ctxs[0]
        R02 = draw_ctx["R02_tot_draws"]

        ppd, R02 = _branch_label_ppd(data.data_grid[param_name], param_name, -1, draw_ctx)
        ppd, R02 = ppd[0], R02[0]  # remove ncomps axis
        if param_name == "chi_eff":
            ppd = ppd / R02[:, None]

        plot.setup_and_plot_GWTC(param_name, ax, res=res, catalog=plot_cat)
        plot.plot_ppds(ax, x, ppd, CI=90, color=color_tot)
        ax.legend()

        # save to dictionary
        dict_out["ppd"] = utils.to_numpy(ppd)
        dict_out["rate"] = utils.to_numpy(R02)

    else:
        raise ValueError(f"Unrecognized param name {param_name}")

    fn_parts = [param_name]
    if use_labels:
        fn_parts.append("labelled")
    if plot_submodels:
        fn_parts.append("submodels")
    fn = f"{'_'.join(fn_parts)}.pdf"
    plt.savefig(os.path.join(figpath, fn))
    print(f"Saved {fn}")
    plt.close()

    dict_out["x"] = utils.to_numpy(x)
    return dict_out


ppds_fn = "ppds.npy"
submodel_ppds_fn = "ppds_submodels.npy"
if save_or_not(ppds_fn) or save_or_not(submodel_ppds_fn):
    model_ppds_dict = {}
    submodel_ppds_dict = {}
    if rj_branches:
        model_ppds_dict["rj_branches"] = np.array(rj_branches, dtype=str)
        model_ppds_dict["model_sigs"] = sig_names
        model_ppds_dict["bayes_factors"] = bayes_factors
        model_ppds_dict["comp_labels"] = {
            "comp_names": comp_labels,
            "comp_sigs": comp_label_sigs,
            "colors": component_palette,
        }

    for param_name in [
        "mass_1_source",
        "chi_eff",
        "redshift",
        "mass_ratio",
        "mass_2_source"
    ]:
        model_ppds_dict[f"{param_name}"] = plot_model_ppd(param_name, use_labels=False)
        # don't need to plot unlabelled twice
        for b_idx, bn in enumerate(data.branch_names[:-1]):
            if data.nleaves_max_dict[bn] > 1:
                if data.get_parent_param(b_idx, param_name) in data.hp_ordering[b_idx]:  # local
                    model_ppds_dict[f"{param_name}_labelled"] = plot_model_ppd(
                        param_name, use_labels=True
                    )
                    if bn in rj_branches:
                        submodel_ppds_dict[f"{param_name}_labelled"] = plot_model_ppd(
                            param_name, use_labels=True, plot_submodels=True
                        )
                    break

    np.save(os.path.join(datapath, ppds_fn), model_ppds_dict, allow_pickle=True)
    print(f"Saved {ppds_fn}")
    if submodel_ppds_dict:
        np.save(os.path.join(datapath, submodel_ppds_fn), submodel_ppds_dict, allow_pickle=True)
        print(f"Saved {submodel_ppds_fn}")
        plot_submodels = True
else:
    print(f"Loading in ppds from {ppds_fn}")
    model_ppds_dict = np.load(os.path.join(datapath, ppds_fn), allow_pickle=True).item()
    if os.path.exists(os.path.join(datapath, submodel_ppds_fn)) and "submodel" in draw_ctxs:
        plot_submodels = True
        submodel_ppds_dict = np.load(
            os.path.join(datapath, submodel_ppds_fn), allow_pickle=True
        ).item()
    else:
        plot_submodels = False

# ----------------------------------------------------------------------
# Leading submodel (highest Bayes factor): condensed PPD plot, event
# likelihoods (N*L), and population-reweighted posteriors.
# ----------------------------------------------------------------------
if plot_submodels and submodel_bfs:
    lead_idx = int(np.argmax(submodel_bfs))
    lead_sig = submodel_names[lead_idx]
    lead_safe = utils.get_safe_fn(lead_sig)
    lead_draw_ctx = draw_ctxs["submodel"][lead_sig]

    # title indicating which (sub)model this is
    if len(data.branch_names) <= 2:  # single local branch -> "x components"
        ncomp = int(sum(submodel_sigs[lead_idx]))
        submodel_title = f"\\texttt{{{args.label}}} - {ncomp} components"
    else:  # multiple local branches -> "(x1 branch1, x2 branch2)"
        submodel_title = (
            f"\\texttt{{{args.label}}} - ({submodel_model_signames_all[lead_sig]})"
        )

    # condensed PPD plot (m1, q, chi_eff; components + totals)
    all_param_ppds_fn = f"all_param_ppds_{lead_safe}.pdf"
    if save_or_not(all_param_ppds_fn):
        plot.plot_all_param_ppds(
            submodel_ppds_dict,
            lead_sig,
            comp_label_info={
                "comp_names": comp_labels,
                "comp_sigs": comp_label_sigs,
                "colors": component_palette,
            },
            res=res,
            spin_res=spin_res,
            catalog=plot_cat,
            title=submodel_title,
            savepath=os.path.join(figpath, all_param_ppds_fn),
            rasterize=args.rasterize,
        )
        print(f"Saved {all_param_ppds_fn}")

    # event likelihoods (N*L) and population-reweighted posteriors
    NL_fn = f"NL_{lead_safe}.npy"
    ev_prob_fn = f"event_probabilities_{lead_safe}.npy"
    reweighted_fn = f"PE_posterior_reweighted_samples_{lead_safe}.npy"
    if save_or_not(NL_fn) or save_or_not(ev_prob_fn) or save_or_not(reweighted_fn):
        ncomps_per_branch = [
            int(n_labelled_comps[b_idx]) for b_idx in range(len(data.branch_names) - 1)
        ]

        print("Computing event likelihoods (N*L) for leading submodel...")
        NL = data.compute_NL_event_likelihoods(lead_draw_ctx, PE_samples)
        np.save(os.path.join(datapath, NL_fn), NL)
        print(f"Saved {NL_fn}")

        print("Computing population-reweighted posteriors for leading submodel...")
        event_probabilities = data.get_pop_weighted_posterior(
            lead_draw_ctx, PE_samples, ncomps_per_branch, NL=NL, average_over_PE=True
        )
        np.save(os.path.join(datapath, ev_prob_fn), event_probabilities)
        print(f"Saved {ev_prob_fn}")

        reweighted_samples = data.get_pop_weighted_posterior(
            lead_draw_ctx, PE_samples, ncomps_per_branch, NL=NL, average_over_PE=False
        )
        np.save(os.path.join(datapath, reweighted_fn), reweighted_samples)
        print(f"Saved {reweighted_fn}")
    else:
        event_probabilities = np.load(os.path.join(datapath, ev_prob_fn))
        reweighted_samples = np.load(os.path.join(datapath, reweighted_fn))

    # population-weighted PE contour plots: one reweighted KDE contour per event,
    # colored by its dominant labelled component
    PE_np = {
        p: utils.to_numpy(PE_samples[p])
        for p in ("mass_1_source", "chi_eff", "redshift", "mass_ratio")
    }
    for param_x, param_y in [
        ("mass_1_source", "chi_eff"),
        ("mass_1_source", "redshift"),
        ("mass_1_source", "mass_ratio"),
    ]:
        pwp_fn = f"{param_x}_{param_y}_pop_weighted_posteriors_{lead_safe}.pdf"
        if save_or_not(pwp_fn):
            plot.plot_pop_weighted_posteriors(
                param_x,
                param_y,
                PE_np,
                event_probabilities,
                component_palette,
                posterior_pop_weights=reweighted_samples,
                comp_names=comp_labels,
                grids=data.data_grid,
                title=submodel_title,
                savepath=os.path.join(figpath, pwp_fn),
            )
            print(f"Saved {pwp_fn}")

    # human-readable event probabilities table (event name + per-component prob)
    ev_prob_txt_fn = f"event_probabilities_{lead_safe}.txt"
    if save_or_not(ev_prob_txt_fn):
        names = (
            [str(n) for n in event_names]
            if event_names is not None
            else [f"event_{i}" for i in range(len(event_probabilities))]
        )
        name_w = max(len("event"), *(len(n) for n in names))
        col_w = [max(len(str(c)), 8) for c in comp_labels]
        lines = [
            f"{'event':<{name_w}}"
            + "".join(f"  {str(c):>{w}}" for c, w in zip(comp_labels, col_w, strict=True))
        ]
        for name, probs in zip(names, event_probabilities, strict=True):
            lines.append(
                f"{name:<{name_w}}"
                + "".join(f"  {p:>{w}.4f}" for p, w in zip(probs, col_w, strict=True))
            )
        with open(os.path.join(datapath, ev_prob_txt_fn), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print(f"Saved {ev_prob_txt_fn}")

    # popsummary export: hyperparameter samples, 1D
    # marginal rates, and population-reweighted single-event posterior
    popsummary_fn = data.get_popsummary_fn(lead_sig, outdir)
    if save_or_not(popsummary_fn):
        if os.path.exists(popsummary_fn):
            os.remove(popsummary_fn)  # PopulationResult creates the file fresh
        print(f"Writing popsummary file for submodel '{lead_sig}'...")
        ps_res = data.initialize_popsummary(
            megadict,
            model_ppds_dict,
            submodel_ppds_dict,
            draw_ctxs["submodel"],
            lead_sig,
            args,
            outdir,
            popmodel_label=args.label,
        )
        event_params = [p for p in data.data_grid if p in model_ppds_dict]
        reweighted_event_samples = data.compute_reweighted_event_samples(
            lead_draw_ctx, PE_samples, event_params
        )
        ps_res.set_reweighted_event_samples(
            reweighted_event_samples,
            hyperparameter_sample_idx_map=lead_draw_ctx["sel_inds_of_model_inds"],
        )
        print(f"Saved {os.path.basename(popsummary_fn)}")
    else:
        ps_res = PopulationResult(popsummary_fn)

# finally, plot 2D!!

# plot 2d contours of best model
for param_x, param_y in [
    ("mass_ratio", "chi_eff"),
    ("mass_1_source", "chi_eff"),
    ("mass_1_source", "mass_ratio"),
    ("mass_2_source", "mass_1_source"),
    ("redshift", "mass_1_source"),
    ("redshift", "chi_eff"),
]:
    if "ppd" in model_ppds_dict[param_x] or "ppd" in model_ppds_dict[param_y]:
        # global parameter - no need to plot contours
        continue
    if save_or_not(f"contours2d_{param_x}_{param_y}*_tot.pdf"):
        for sig_name, bf in zip(sig_names, bayes_factors, strict=False):
            if bf > 0.5:
                data.plot_param_2D_contours_and_marginals(  # total
                    param_x,
                    param_y,
                    model_sig=sig_name,
                    draw_ctxs=draw_ctxs,
                    model_ppds_dict=model_ppds_dict,
                    savefig=True,
                    figpath=figpath,
                    levels=(0.5, 0.9, 0.99),
                    color="cornflowerblue",
                    comp_label=None,
                    CI=90,
                    rasterize=args.rasterize
                )
    if save_or_not(f"contours2d_{param_x}_{param_y}*_labelled.pdf"):
        for sig_name, bf in zip(sig_names, bayes_factors, strict=False):
            if bf > 0.5:
                res = ps_res if bf == np.amax(bayes_factors) else None
                data.plot_param_2D_contours_and_marginals(  # all individually
                    param_x,
                    param_y,
                    model_sig=sig_name,
                    draw_ctxs=draw_ctxs,
                    model_ppds_dict=model_ppds_dict,
                    savefig=True,
                    figpath=figpath,
                    levels=(0.5, 0.9),
                    color=component_palette,
                    comp_label="all",
                    CI=None,
                    rasterize=args.rasterize,
                    popsummary_res=res
                )
    if plot_submodels:
        if save_or_not(f"contours2d_{param_x}_{param_y}*_submodel_labelled.pdf"):
            for sig_name, bf in zip(submodel_names, submodel_bfs, strict=True):
                if bf > 0.5:
                    data.plot_param_2D_contours_and_marginals(  # all individually
                        param_x,
                        param_y,
                        model_sig=sig_name,
                        draw_ctxs=draw_ctxs,
                        model_ppds_dict=model_ppds_dict,
                        submodel_ppds_dict=submodel_ppds_dict,
                        savefig=True,
                        figpath=figpath,
                        levels=(0.5, 0.9),
                        color=component_palette,
                        comp_label="all",
                        CI=None,
                        rasterize=args.rasterize
                    )

R02_low, R02_med, R02_high = np.percentile(R02_tot, [5, 50, 95])
utils.print_to(
    logpath,
    f"\n\nThe total rate at z=0.2 is {R02_med:.1f} +{R02_high - R02_med:.1f} -{R02_med - R02_low:.1f} Gpc-3 yr-1",
)
utils.print_to(
    logpath,
    rf"\mathcal{{R}}(z=0.2) = "
    rf"{R02_med:.1f}^{{+{R02_high - R02_med:.1f}}}_{{-{R02_med - R02_low:.1f}}}"
    r"\ \mathrm{Gpc}^{-3}\ \mathrm{yr}^{-1}",
)
