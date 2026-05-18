#!/usr/bin/env python3
r"""
Generate LaTeX macros from RJMCMC population analysis results.

Writes scripts/macros.tex with \newcommand definitions quoting
90% credible intervals as  med^{+(hi-med)}_{-(med-lo)}.
Requires \usepackage{xspace} in LaTeX preamble.

Models used
-----------
skewt     : "3 local" model signature  (submodel=False)
NPLNP     : "PL A, gauss A, gauss B"   (submodel=True)
skewt_z   : "2 local" and "3 local"    (submodel=False) for per-comp gamma
skewt_ID  : "3 local"                  (submodel=False) for per-comp rho

Component name mapping (ten=10 Msun peak, thirty=30 Msun peak, tail=high mass tail)
skewt / skewt_z / skewt_ID :  A=label 0=ten,  B=label 1=thirty,  C=label 2=tail
NPLNP                       :  gauss A=label 0 (gauss)=ten, gauss B=label 1=thirty,
                               PL A=label 0 (PL)=tail
"""

import json
import os

import numpy as np
from scipy.stats import truncnorm as _truncnorm

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
O4_PATH    = "/data/aqc/rjmcmc_pop/results/O4"
OUT_PATH   = os.path.join(SCRIPT_DIR, "macros.tex")

# ── data helpers ─────────────────────────────────────────────────────────────

def load_megadict(model_name):
    path = os.path.join(O4_PATH, model_name, "data", "samples_and_labels.npy")
    return np.load(path, allow_pickle=True).item()


def load_hpinfo(model_name):
    path = os.path.join(O4_PATH, model_name, "hyperparameter_ordering_metainfo.json")
    with open(path) as f:
        return json.load(f)


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
        inds  = mega["submodel_inds"]
    else:
        names = mega["sig_names"]
        inds  = mega["model_inds"]
    return inds[names.index(model_sig)]


def leaf_samples(mega, branch, label_idx, model_sig, submodel=False, cols=None):
    """
    Flatten posterior samples for leaves in `branch` with `label_idx`,
    restricted to `model_sig`.

    cols : None → all dims (returns 2-D); int → 1-D; list/tuple → 2-D subset.
    """
    idx = sig_inds(mega, model_sig, submodel)
    s   = mega["samples"][branch][idx]   # (n, nleaves, ndims)
    lbl = mega["labels"][branch][idx]    # (n, nleaves)
    mask = lbl == label_idx              # bool (n, nleaves)

    if cols is None:
        return s[mask]                          # (n_match, ndims)
    if isinstance(cols, (list, tuple)):
        return s[mask][:, list(cols)]           # (n_match, len(cols))
    return s[..., cols][mask]                   # (n_match,)


def R0_per_sample(mega, branch, label_idx, model_sig, submodel=False):
    """Sum R0 of matching leaves per posterior sample. Returns (n_sig,) array."""
    idx  = sig_inds(mega, model_sig, submodel)
    s    = mega["samples"][branch][idx]          # (n, nleaves, ndims)
    lbl  = mega["labels"][branch][idx]           # (n, nleaves)
    R0   = np.where(np.isfinite(s[..., -1]) & (s[..., -1] > 0), s[..., -1], 0.0)
    return np.nansum(R0 * (lbl == label_idx), axis=1)  # (n,)


def total_R0_per_sample(mega, local_branches, model_sig, submodel=False):
    """Sum R0 across all leaves in all local branches for each sample. Returns (n_sig,)."""
    idx = sig_inds(mega, model_sig, submodel)
    tot = np.zeros(len(idx))
    for bn in local_branches:
        s = mega["samples"][bn][idx]             # (n, nleaves, ndims)
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


def _nz(val, fmt):
    """Format val; force negative-zero strings like '-0.00' to '0.00'."""
    s = format(val, fmt)
    # check if the value is negative-zero after rounding (e.g. -0.003 → '-0.00')
    if s.startswith("-") and float(s) == 0.0:
        s = s[1:]   # strip the minus
    return s


def fmt_ci(samples, lo=5, hi=95, fmt=".1f"):
    """LaTeX string  med^{+hi-med}_{-med-lo}  for 90 % CI."""
    s = np.asarray(samples).ravel()
    s = s[np.isfinite(s)]
    low, med, high = np.percentile(s, [lo, 50, hi])
    m  = _nz(med,          fmt)
    dh = _nz(high - med,   fmt)
    dl = _nz(med  - low,   fmt)
    return rf"{m}^{{+{dh}}}_{{-{dl}}}"


def fmt_pct(prob, fmt=".1f"):
    """LaTeX string for a probability (0–1) rendered as a percentage."""
    return rf"{prob * 100:{fmt}}\%"


# ── macro registry ───────────────────────────────────────────────────────────

macros = {}   # name → raw LaTeX body (without \newcommand wrapper)


def add_ci(name, samples, lo=5, hi=95, fmt=".1f"):
    if name in macros:
        print(f"  [WARNING] duplicate macro \\{name} – overwriting")
    macros[name] = r"\ensuremath{" + fmt_ci(samples, lo, hi, fmt) + r"}\xspace"
    print(f"  \\{name}  =  {fmt_ci(samples, lo, hi, fmt)}")


def add_pct(name, prob, fmt=".1f"):
    if name in macros:
        print(f"  [WARNING] duplicate macro \\{name} – overwriting")
    macros[name] = r"\ensuremath{" + fmt_pct(prob, fmt) + r"}\xspace"
    print(f"  \\{name}  =  {fmt_pct(prob, fmt)}")


def add_val(name, val, fmt=".2f"):
    if name in macros:
        print(f"  [WARNING] duplicate macro \\{name} – overwriting")
    s = _nz(val, fmt)
    macros[name] = r"\ensuremath{" + s + r"}\xspace"
    print(f"  \\{name}  =  {s}")


# ── gamma confidence helper ───────────────────────────────────────────────────

def paired_gamma_confidence(mega, model_sig, gamma_col, label_a=0, label_b=1, ncomps=None):
    """
    Posterior probability that gamma[label_a] > gamma[label_b].
    Filters to samples containing exactly one leaf of each required label.
    """
    idx   = sig_inds(mega, model_sig, submodel=False)
    gamma = mega["samples"]["local"][idx][..., gamma_col]   # (n, nleaves)
    lbls  = mega["labels"]["local"][idx]                    # (n, nleaves)

    if ncomps is None:
        ncomps = int(np.max(lbls)) + 1

    # Keep only samples with exactly 1 of each label 0..ncomps-1
    valid = np.ones(len(idx), dtype=bool)
    for li in range(ncomps):
        valid &= (np.sum(lbls == li, axis=-1) == 1)

    gamma_f = gamma[valid]
    lbls_f  = lbls[valid]

    gamma_a = gamma_f[lbls_f == label_a]   # one entry per valid sample
    gamma_b = gamma_f[lbls_f == label_b]

    return np.mean(gamma_a > gamma_b)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1.  SKEWT  –  "3 local" ───────────────────────────────────────────────
    print("\n── skewt (3 local) ──────────────────────────────────────")
    mega_sk  = load_megadict("skewt")
    hp_sk    = load_hpinfo("skewt")
    SIG_SK   = "3 local"
    BNS_SK   = ["local"]

    # Column indices (verified from hyperparameter_ordering_metainfo.json)
    SK_MU_M       = get_param_col(hp_sk, 0, "mass", "mass_1_source", param_name="loc")    # 2
    SK_MU_Q       = get_param_col(hp_sk, 0, "mass", "mass_ratio",    param_name="loc")    # 4
    SK_MU_CHI     = get_param_col(hp_sk, 0, "chi_eff",               param_name="loc")    # 6
    SK_SIGMA_CHI  = get_param_col(hp_sk, 0, "chi_eff",               param_name="scale")  # 7

    # Component labels:  A=0=ten,  B=1=thirty,  C=2=tail
    for comp_tag, lbl in [("ten", 0), ("thirty", 1), ("tail", 2)]:

        # chi_eff distribution percentiles (1st, 50th, 95th)
        chi = leaf_samples(mega_sk, "local", lbl, SIG_SK, cols=[SK_MU_CHI, SK_SIGMA_CHI])
        mu_c, sig_c = chi[:, 0], chi[:, 1]

        ppf_05 = trunc_gauss_ppf(0.05, mu_c, sig_c)
        ppf_50 = trunc_gauss_ppf(0.50, mu_c, sig_c)
        ppf_95 = trunc_gauss_ppf(0.95, mu_c, sig_c)

        add_ci(f"chiefflow{comp_tag}skewt",  ppf_05, fmt=".2f")
        add_ci(f"chieffmed{comp_tag}skewt",  ppf_50, fmt=".2f")
        add_ci(f"chieffhigh{comp_tag}skewt", ppf_95, fmt=".2f")

        # mu_chi > 0 confidence (only for ten and tail)
        if comp_tag in ("ten", "tail"):
            add_pct(f"muchiposconfidence{comp_tag}skewt", np.mean(mu_c > 0))

        # mu_m
        mu_m = leaf_samples(mega_sk, "local", lbl, SIG_SK, cols=SK_MU_M)
        add_ci(f"mum{comp_tag}skewt", mu_m, fmt=".1f")

        # mu_q
        mu_q = leaf_samples(mega_sk, "local", lbl, SIG_SK, cols=SK_MU_Q)
        add_ci(f"muq{comp_tag}skewt", mu_q, fmt=".2f")
        add_val(f"muq{comp_tag}skewttenperc", np.percentile(mu_q[np.isfinite(mu_q)], 10), fmt=".2f")

        # rate at z = 0 (= R0 parameter)
        R0 = R0_per_sample(mega_sk, "local", lbl, SIG_SK)
        add_ci(f"rate{comp_tag}skewt", R0, fmt=".1f")

    # Branching fractions
    R0_tot_sk = total_R0_per_sample(mega_sk, BNS_SK, SIG_SK)
    for comp_tag, lbl in [("ten", 0), ("thirty", 1), ("tail", 2)]:
        R0c = R0_per_sample(mega_sk, "local", lbl, SIG_SK)
        add_ci(f"bfrac{comp_tag}skewt", R0c / R0_tot_sk, fmt=".2f")

    # Global gamma (redshift evolution)
    SK_GAMMA = get_param_col(hp_sk, 1, "redshift", param_name="gamma")  # 0
    gamma_sk = global_param_samples(mega_sk, SIG_SK, SK_GAMMA)
    add_ci("gammaskewt", gamma_sk, fmt=".1f")

    # ── 2.  NPLNP  –  "PL A, gauss A, gauss B"  (submodel=True) ──────────────
    print("\n── NPLNP (PL A, gauss A, gauss B) ──────────────────────")
    mega_np  = load_megadict("NPLNP")
    hp_np    = load_hpinfo("NPLNP")
    SIG_NP   = "PL A, gauss A, gauss B"
    BNS_NP   = ["PL", "gauss"]
    SUB_NP   = True

    # PL branch col indices  (branch_idx=0)
    PL_ALPHA      = get_param_col(hp_np, 0, "mass", "mass_1_source", param_name="alpha")  # 0
    PL_P          = get_param_col(hp_np, 0, "mass", "mass_1_source", param_name="p")      # 1
    PL_XMIN       = get_param_col(hp_np, 0, "mass", "mass_1_source", param_name="xmin")   # 2
    PL_MU_Q       = get_param_col(hp_np, 0, "mass", "mass_ratio",    param_name="loc")
    PL_MU_CHI     = get_param_col(hp_np, 0, "chi_eff",               param_name="loc")    # 6
    PL_SIGMA_CHI  = get_param_col(hp_np, 0, "chi_eff",               param_name="scale")  # 7

    # Gauss branch col indices  (branch_idx=1)
    G_MU_M        = get_param_col(hp_np, 1, "mass", "mass_1_source", param_name="loc")    # 0
    G_SIGMA_M     = get_param_col(hp_np, 1, "mass", "mass_1_source", param_name="scale")  # 1
    G_MU_Q        = get_param_col(hp_np, 1, "mass", "mass_ratio",    param_name="loc")
    G_MU_CHI      = get_param_col(hp_np, 1, "chi_eff",               param_name="loc")    # 4
    G_SIGMA_CHI   = get_param_col(hp_np, 1, "chi_eff",               param_name="scale")  # 5

    # tail (PL A = PL branch label 0)
    chi_pla = leaf_samples(mega_np, "PL", 0, SIG_NP, submodel=SUB_NP, cols=[PL_MU_CHI, PL_SIGMA_CHI])
    mu_c_tail, sig_c_tail = chi_pla[:, 0], chi_pla[:, 1]

    ppf_05 = trunc_gauss_ppf(0.05, mu_c_tail, sig_c_tail)
    ppf_50 = trunc_gauss_ppf(0.50, mu_c_tail, sig_c_tail)
    ppf_95 = trunc_gauss_ppf(0.95, mu_c_tail, sig_c_tail)
    add_ci("chiefflowtailNPLNP",  ppf_05, fmt=".2f")
    add_ci("chieffmedtailNPLNP",  ppf_50, fmt=".2f")
    add_ci("chieffhightailNPLNP", ppf_95, fmt=".2f")
    add_pct("muchiposconfidencetailNPLNP", np.mean(mu_c_tail > 0))

    # mu_q for tail
    mu_q_tail = leaf_samples(mega_np, "PL", 0, SIG_NP, submodel=SUB_NP, cols=PL_MU_Q)
    add_ci("muqtailNPLNP", mu_q_tail, fmt=".2f")
    add_val("muqtailNPLNPtenperc", np.percentile(mu_q_tail[np.isfinite(mu_q_tail)], 10), fmt=".2f")

    # mu_m for tail: mode of smoothed power law = xmin * (1 + p/alpha)
    pla_params = leaf_samples(mega_np, "PL", 0, SIG_NP, submodel=SUB_NP,
                               cols=[PL_ALPHA, PL_P, PL_XMIN])
    mode_tail  = pla_params[:, 2] * (1 + pla_params[:, 1] / pla_params[:, 0])
    add_ci("mumtailNPLNP", mode_tail, fmt=".1f")

    # ten (gauss A = gauss branch label 0) and thirty (gauss B = gauss branch label 1)
    for comp_tag, lbl in [("ten", 0), ("thirty", 1)]:
        chi_g = leaf_samples(mega_np, "gauss", lbl, SIG_NP, submodel=SUB_NP,
                             cols=[G_MU_CHI, G_SIGMA_CHI])
        mu_c, sig_c = chi_g[:, 0], chi_g[:, 1]

        ppf_05 = trunc_gauss_ppf(0.05, mu_c, sig_c)
        ppf_50 = trunc_gauss_ppf(0.50, mu_c, sig_c)
        ppf_95 = trunc_gauss_ppf(0.95, mu_c, sig_c)
        add_ci(f"chiefflow{comp_tag}NPLNP",  ppf_05, fmt=".2f")
        add_ci(f"chieffmed{comp_tag}NPLNP",  ppf_50, fmt=".2f")
        add_ci(f"chieffhigh{comp_tag}NPLNP", ppf_95, fmt=".2f")

        if comp_tag == "ten":  # only requested for ten
            add_pct(f"muchiposconfidence{comp_tag}NPLNP", np.mean(mu_c > 0))

        g_params = leaf_samples(mega_np, "gauss", lbl, SIG_NP, submodel=SUB_NP,
                                cols=[G_MU_M, G_SIGMA_M])
        add_ci(f"mum{comp_tag}NPLNP",    g_params[:, 0], fmt=".1f")
        add_ci(f"sigmam{comp_tag}NPLNP", g_params[:, 1], fmt=".1f")

        mu_q = leaf_samples(mega_np, "gauss", lbl, SIG_NP, submodel=SUB_NP, cols=G_MU_Q)
        add_ci(f"muq{comp_tag}NPLNP", mu_q, fmt=".2f")
        add_val(f"muq{comp_tag}NPLNPtenperc", np.percentile(mu_q[np.isfinite(mu_q)], 10), fmt=".2f")

    # Rates and branching fractions
    R0_PLA    = R0_per_sample(mega_np, "PL",    0, SIG_NP, submodel=SUB_NP)
    R0_gaussA = R0_per_sample(mega_np, "gauss", 0, SIG_NP, submodel=SUB_NP)
    R0_gaussB = R0_per_sample(mega_np, "gauss", 1, SIG_NP, submodel=SUB_NP)
    R0_tot_np = R0_PLA + R0_gaussA + R0_gaussB

    add_ci("ratetailNPLNP",   R0_PLA,    fmt=".1f")
    add_ci("ratetenNPLNP",    R0_gaussA, fmt=".1f")
    add_ci("ratethirtyNPLNP", R0_gaussB, fmt=".1f")

    add_ci("bfractailNPLNP",   R0_PLA    / R0_tot_np, fmt=".2f")
    add_ci("bfractenNPLNP",    R0_gaussA / R0_tot_np, fmt=".2f")
    add_ci("bfracthirtyNPLNP", R0_gaussB / R0_tot_np, fmt=".2f")

    # Global gamma (redshift evolution)
    NP_GAMMA = get_param_col(hp_np, 2, "redshift", param_name="gamma")  # 0
    gamma_np = global_param_samples(mega_np, SIG_NP, NP_GAMMA, submodel=SUB_NP)
    add_ci("gammaNPLNP", gamma_np, fmt=".1f")

    # ── 3.  SKEWT_Z  –  per-component gamma,  "2 local" and "3 local" ─────────
    print("\n── skewt_z (gamma per component) ───────────────────────")
    mega_sz  = load_megadict("skewt_z")
    hp_sz    = load_hpinfo("skewt_z")

    # skewt_z local branch: logalpha(0),logkappa(1),mu_m(2),sigma_m(3),
    #                        mu_q(4),sigma_q(5),mu_chi(6),sigma_chi(7),gamma(8),R0(9)
    SZ_GAMMA = get_param_col(hp_sz, 0, "redshift", param_name="gamma")  # 8

    for model_sig, comp_suffix in [("2 local", "twocomp"), ("3 local", "threecomp")]:
        for comp_tag, lbl in [("ten", 0), ("thirty", 1)]:
            g = leaf_samples(mega_sz, "local", lbl, model_sig, cols=SZ_GAMMA)
            add_ci(f"gamma{comp_tag}{comp_suffix}", g, fmt=".1f")

    g_tail_3 = leaf_samples(mega_sz, "local", 2, "3 local", cols=SZ_GAMMA)
    add_ci("gammatailthreecomp", g_tail_3, fmt=".1f")

    prob_AB  = paired_gamma_confidence(mega_sz, "2 local", SZ_GAMMA, ncomps=2)
    prob_ABC = paired_gamma_confidence(mega_sz, "3 local", SZ_GAMMA, ncomps=3)

    add_pct("gammatenbiggerconfidenceAB",  prob_AB)
    add_pct("gammatenbiggerconfidenceABC", prob_ABC)

    # ── 4.  SKEWT_ID  –  per-component rho,  "3 local" ───────────────────────
    print("\n── skewt_ID (rho per component, 3 local) ───────────────")
    mega_id  = load_megadict("skewt_ID")
    hp_id    = load_hpinfo("skewt_ID")
    SIG_ID   = "3 local"

    # skewt_ID local branch: rho(0),logalpha(1),logkappa(2),mu_m(3),
    #                         sigma_m(4),mu_chi(5),sigma_chi(6),R0(7)
    ID_RHO = get_param_col(hp_id, 0, "mass", param_name="rho")  # 0

    for comp_tag, lbl in [("ten", 0), ("thirty", 1), ("tail", 2)]:
        rho = leaf_samples(mega_id, "local", lbl, SIG_ID, cols=ID_RHO)
        add_ci(f"rho{comp_tag}", rho, fmt=".2f")

    # ── 5.  SKEWT_ID  –  q at per-component PPD peak ────────────────────────
    print("\n── skewt_ID (q at per-component PPD peak, 3 local) ─────")
    ppds_id = np.load(os.path.join(O4_PATH, "skewt_ID", "data", "ppds.npy"),
                      allow_pickle=True).item()
    q_grid  = np.array(ppds_id["mass_ratio"]["x"])                    # (256,)
    mrl     = ppds_id["mass_ratio_labelled"]["3 local"]
    for comp_tag, comp_key in [("ten", "A"), ("thirty", "B"), ("tail", "C")]:
        ppd  = np.array(mrl[comp_key])                                # (500, 256)
        qmax = q_grid[np.argmax(ppd, axis=1)]                        # (500,)
        add_ci(f"qmaxskewtID{comp_tag}", qmax, fmt=".2f")

    # ── 6.  Write macros.tex ───────────────────────────────────────────────────
    print(f"\n── writing {len(macros)} macros to {OUT_PATH} ──────────")
    with open(OUT_PATH, "w") as f:
        f.write("% Auto-generated population-analysis macros\n")
        f.write("% Source: scripts/macros.py\n")
        f.write("% Requires \\usepackage{xspace} in LaTeX preamble\n\n")

        sections = [
            ("chi_eff distribution percentiles – skewt",
             [n for n in macros if "chieff" in n and "skewt" in n.lower()]),
            ("chi_eff distribution percentiles – NPLNP",
             [n for n in macros if "chieff" in n and "NPLNP" in n]),
            ("mu_chi > 0 confidence",
             [n for n in macros if "muchipos" in n]),
            ("mu_m – skewt",
             [n for n in macros if n.startswith("mum") and "skewt" in n.lower()]),
            ("mu_m – NPLNP",
             [n for n in macros if n.startswith("mum") and "NPLNP" in n]),
            ("mu_q – skewt",
             [n for n in macros if n.startswith("muq") and "skewt" in n.lower() and "tenperc" not in n]),
            ("mu_q – NPLNP",
             [n for n in macros if n.startswith("muq") and "NPLNP" in n and "tenperc" not in n]),
            ("mu_q 10th percentile – skewt",
             [n for n in macros if n.startswith("muq") and "skewt" in n.lower() and "tenperc" in n]),
            ("mu_q 10th percentile – NPLNP",
             [n for n in macros if n.startswith("muq") and "NPLNP" in n and "tenperc" in n]),
            ("sigma_m – NPLNP",
             [n for n in macros if n.startswith("sigmam")]),
            ("branching fractions – skewt",
             [n for n in macros if n.startswith("bfrac") and "skewt" in n.lower()]),
            ("branching fractions – NPLNP",
             [n for n in macros if n.startswith("bfrac") and "NPLNP" in n]),
            ("rates at z=0 – skewt",
             [n for n in macros if n.startswith("rate") and "skewt" in n.lower()]),
            ("rates at z=0 – NPLNP",
             [n for n in macros if n.startswith("rate") and "NPLNP" in n]),
            ("gamma – global redshift evolution (skewt, NPLNP)",
             [n for n in macros if n in ("gammaskewt", "gammaNPLNP")]),
            ("gamma (skewt_z, per component)",
             [n for n in macros if n.startswith("gamma") and "comp" in n.lower()]),
            ("gamma_ten > gamma_thirty confidence (skewt_z)",
             [n for n in macros if "tenbigger" in n]),
            ("rho (skewt_ID, 3 local)",
             [n for n in macros if n.startswith("rho")]),
            ("q at per-component PPD peak (skewt_ID, 3 local)",
             [n for n in macros if n.startswith("qmax")]),
        ]

        written = set()
        for title, names in sections:
            if not names:
                continue
            f.write(f"% {title}\n")
            for name in sorted(names):
                f.write(rf"\newcommand\{name}{{{macros[name]}}}" + "\n")
                written.add(name)
            f.write("\n")

        # catch any stragglers not covered by sections above
        remaining = [n for n in sorted(macros) if n not in written]
        if remaining:
            f.write("% miscellaneous\n")
            for name in remaining:
                f.write(rf"\newcommand\{name}{{{macros[name]}}}" + "\n")

    print("Done.")
