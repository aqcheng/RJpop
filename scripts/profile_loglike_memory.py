#!/usr/bin/env python
"""Profile the GPU memory footprint of ``data.loglike``.

This is a diagnostic / benchmarking tool (not part of the MCMC run). It loads the
real PE-sample and injection datasets, builds ``loglike`` inputs at a controllable
number of groups, and measures peak reserved GPU memory with CuPy's memory pool.

Background
----------
``loglike`` is evaluated by Eryn once per move on a batch of ``ngroups`` states:

    in-model moves :  ngroups = ntemps * nwalkers
    RJ moves       :  ngroups = ntemps * nwalkers * rj_num_try   <- the memory peak

Every working array in ``loglike`` scales linearly with ``ngroups`` (and with
``nleaves_max``), so peak GPU memory grows ~linearly with the batch size. The
``group_chunk_size`` argument of ``loglike`` caps that peak by evaluating the
batch in sub-batches of groups.

Usage
-----
Run from the repository root (so that ``rjpop`` is importable):

    # 1) Scaling sweep + per-line attribution at a fixed ngroups (find the bottleneck):
    python scripts/profile_loglike_memory.py --mode sweep

    # 2) Largest ngroups that fits in a memory budget (binary search; OOM-safe):
    python scripts/profile_loglike_memory.py --mode maxgroups --budget-gb 40

    # 3) Effect of chunking: peak vs group_chunk_size at a fixed total ngroups:
    python scripts/profile_loglike_memory.py --mode chunk --ngroups 200

    # NPLNP label (prior_NPLNP.json, GWTC2p1_3_4 PE, o1234ab injections):
    #   ntemps=7, nwalkers=30  ->  in-model ngroups=210, RJ ngroups=420
    python scripts/profile_loglike_memory.py \\
        --label NPLNP --mode sweep --sweep-ngroups 25,50,100,200,300,420
    python scripts/profile_loglike_memory.py \\
        --label NPLNP --mode maxgroups --budget-gb 40
    python scripts/profile_loglike_memory.py \\
        --label NPLNP --mode maxgroups --budget-gb 80

Override dataset paths with --pe / --injs and the prior with --prior.
Requires a CUDA GPU (the model evaluation must run on the GPU).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

# make ``rjpop`` (data, xp, pdfs, ...) importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rjpop"))
warnings.filterwarnings("ignore")

import data  # noqa: E402
from xp import xp  # noqa: E402

GB = 1e9

DEFAULT_PE = "/scratch/gpfs/ANDREASB/aqc/data/LVK_processed/GWTC345_PE_samples.npz"
DEFAULT_INJ = "/scratch/gpfs/ANDREASB/aqc/data/LVK_processed/o1234ab_injs.npz"
DEFAULT_PRIOR = "examples/prior_skewt.json"

# Per-label dataset/prior presets.  Override any individual field with --prior/--pe/--injs.
LABEL_PRESETS = {
    "NPLNP": {
        "prior": "examples/prior_NPLNP.json",
        "pe": "/scratch/gpfs/ANDREASB/aqc/data/LVK_processed/GWTC2p1_3_4_PE_samples.npz",
        "injs": "/scratch/gpfs/ANDREASB/aqc/data/LVK_processed/o1234ab_injs.npz",
    },
}


def load(prior_path, pe_path, inj_path):
    """Set up the ``data`` module state and load the real datasets onto the GPU."""
    with open(prior_path) as f:
        priors, _ = data.process_input_priordict(json.load(f), rate_prior=(0.05, 100))
    params = sorted(list(data.params))
    with np.load(pe_path) as f:
        PE = {k: xp.asarray(v, dtype=xp.float32) for k, v in f.items() if k in params + ["prior"]}
    with np.load(inj_path) as f:
        INJ = {
            k: xp.asarray(v, dtype=xp.float32)
            for k, v in f.items()
            if k in params + ["prior", "w"]
        }
        INJ["total_generated"] = int(f["total_generated"])
        INJ["TOBS"] = xp.asarray(float(f["Tobs_yr"]), dtype=xp.float32)
    nevs, nsamp = PE["redshift"].shape
    print(f"PE: {nevs} events x {nsamp} samples ({nevs * nsamp} pts) | injections: {INJ['redshift'].shape[0]}")
    print(f"nleaves_max={data.nleaves_max_dict}")
    return priors, PE, INJ


def make_inputs(priors, ngroups):
    """Build (hyperparams, groups) lists for ``ngroups`` groups, mirroring main.py test mode."""
    hp = [priors[b].rvs(size=(ngroups * data.nleaves_max_dict[b],)) for b in data.branch_names]
    grp = [xp.repeat(xp.arange(ngroups), data.nleaves_max_dict[b]) for b in data.branch_names]
    return hp, grp


def _sync_and_pool():
    import cupy

    cupy.cuda.Stream.null.synchronize()
    return cupy.get_default_memory_pool()


def measure_peak(priors, PE, INJ, ngroups, **loglike_kwargs):
    """Return (peak_reserved_GB, work_GB) for one ``loglike`` call at ``ngroups`` groups."""
    pool = _sync_and_pool()
    hp, grp = make_inputs(priors, ngroups)
    pool.free_all_blocks()
    _sync_and_pool()
    base = pool.total_bytes()
    ll = data.loglike(hp, grp, PE, INJ, min_sep=1, **loglike_kwargs)
    _sync_and_pool()
    peak = pool.total_bytes()
    del hp, grp, ll
    pool.free_all_blocks()
    return peak / GB, (peak - base) / GB


def mode_sweep(priors, PE, INJ, args):
    """Scaling sweep + per-line memory attribution via CuPy's LineProfileHook."""
    sweep_values = (
        [int(x) for x in args.sweep_ngroups.split(",")]
        if args.sweep_ngroups
        else [10, 25, 50, 100]
    )
    print("\n=== peak reserved GPU memory vs ngroups (full data) ===")
    print(f"{'ngroups':>8} {'peak_GB':>9} {'work_GB':>9} {'GB/group':>9}")
    for ng in sweep_values:
        try:
            peak, work = measure_peak(priors, PE, INJ, ng)
            print(f"{ng:>8} {peak:>9.2f} {work:>9.2f} {work / ng:>9.3f}")
        except Exception as e:
            print(f"{ng:>8} {'--':>9} {'--':>9} {'OOM':>9}  ({str(e).splitlines()[0]})")
            break

    from cupy.cuda.memory_hooks import LineProfileHook

    ng = args.ngroups or 50
    print(f"\n=== per-line memory attribution (LineProfileHook, ngroups={ng}) ===")
    pool = _sync_and_pool()
    hp, grp = make_inputs(priors, ng)
    pool.free_all_blocks()
    hook = LineProfileHook()
    with hook:
        data.loglike(hp, grp, PE, INJ, min_sep=1)
        _sync_and_pool()
    hook.print_report()


def mode_chunk(priors, PE, INJ, args):
    """Peak memory vs group_chunk_size at a fixed total ngroups (shows the chunking win)."""
    ng = args.ngroups
    print(f"\n=== peak reserved GPU memory vs group_chunk_size (ngroups={ng}) ===")
    print(f"{'chunk':>8} {'peak_GB':>9} {'status':>10}")
    for chunk in [None, ng, ng // 2, ng // 4, 25, 10]:
        try:
            peak, _ = measure_peak(priors, PE, INJ, ng, group_chunk_size=chunk)
            print(f"{str(chunk):>8} {peak:>9.2f} {'ok':>10}")
        except Exception as e:  # cupy OutOfMemoryError etc.
            print(f"{str(chunk):>8} {'--':>9} {'OOM':>10}  ({str(e).splitlines()[0]})")


def mode_maxgroups(priors, PE, INJ, args):
    """Binary-search the largest ngroups whose peak stays under --budget-gb (OOM-safe)."""
    import cupy

    budget = args.budget_gb
    chunk = args.chunk  # None -> no chunking; else fixed chunk size
    print(f"\n=== largest ngroups under {budget} GB budget (group_chunk_size={chunk}) ===")

    def fits(ng):
        try:
            peak, _ = measure_peak(priors, PE, INJ, ng, group_chunk_size=chunk)
            ok = peak <= budget
            print(f"  ngroups={ng:5d}: peak={peak:6.2f} GB -> {'fits' if ok else 'over budget'}")
            return ok
        except cupy.cuda.memory.OutOfMemoryError:
            print(f"  ngroups={ng:5d}: OutOfMemoryError")
            return False

    lo, hi = 1, args.hi
    # expand hi until it fails (capped), then binary search
    while fits(hi) and hi < args.cap:
        lo, hi = hi, min(hi * 2, args.cap)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    print(f"\n==> max ngroups that fits in {budget} GB: {lo}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["sweep", "chunk", "maxgroups"], default="sweep")
    p.add_argument("--label", default=None, help="named config preset (e.g. NPLNP); sets --prior/--pe/--injs defaults")
    p.add_argument("--prior", default=None)
    p.add_argument("--pe", default=None)
    p.add_argument("--injs", default=None)
    p.add_argument("--ngroups", type=int, default=None, help="total ngroups for chunk/sweep modes")
    p.add_argument("--sweep-ngroups", default=None, help="comma-separated ngroups list for sweep mode")
    p.add_argument("--budget-gb", type=float, default=40.0, help="memory budget for maxgroups mode")
    p.add_argument("--chunk", type=int, default=None, help="group_chunk_size for maxgroups mode")
    p.add_argument("--hi", type=int, default=64, help="initial upper bound for maxgroups search")
    p.add_argument("--cap", type=int, default=4096, help="hard cap on ngroups search")
    args = p.parse_args()

    # Apply label preset, then let explicit --prior/--pe/--injs override.
    preset = LABEL_PRESETS.get(args.label, {}) if args.label else {}
    prior_path = args.prior or preset.get("prior", DEFAULT_PRIOR)
    pe_path = args.pe or preset.get("pe", DEFAULT_PE)
    inj_path = args.injs or preset.get("injs", DEFAULT_INJ)
    if args.label:
        print(f"Label preset '{args.label}': prior={prior_path}, pe={pe_path}, injs={inj_path}")

    priors, PE, INJ = load(prior_path, pe_path, inj_path)
    {"sweep": mode_sweep, "chunk": mode_chunk, "maxgroups": mode_maxgroups}[args.mode](
        priors, PE, INJ, args
    )


if __name__ == "__main__":
    main()
