# rjpop

**rjpop** is a gravitational wave population inference framework that uses reversible-jump (RJ) MCMC to search for subpopulations within the binary black hole (BBH) population. The core idea is to model the BBH population as a flexible number $N$ of smooth "blobs" in the 4D parameter space $(m_1, q, \chi_\mathrm{eff}, z)$, where each blob represents a subpopulation. 

More precisely, the BBH merger rate is modeled as:

$$\frac{dN}{d\theta\,dz} = \sum_{k=1}^N  \frac{\mathcal{R}_0^k \psi(z|\lambda^k)}{1+z}\frac{dV_c}{dz} p^k(\theta|\lambda^k)$$

where:
- $N$ is the (inferred) number of subpopulations
- $\mathcal{R}_0^k$ is the local merger rate density of the $k$th subpopulation
- $p^k(\theta|\lambda^k)$ is its probability distribution over masses, mass ratio, and effective spin
- $\psi(z|\lambda^k)$ is its redshift evolution function
- $\lambda^k$ are the hyperparameters for the *k*th subpopulation

Reversible jump allows us to infer $\lambda^k$ for $k=1, 2, \ldots, N$ for varying $N$ by effectively simultaneously doing model comparison during the inference, thus optimizing the model complexity. We use the reversible-jump framework implemented in [Eryn](https://github.com/mikekatz04/Eryn), which also includes affine-invariant sampling and parallel tempering.

This code was used to [analyze the GWTC-4 dataset](https://inspirehep.net/literature/3159273). The version of this code used to produce the results in the paper is archived at [`v1.0.0`](https://github.com/aqcheng/RJpop/releases/tag/v1.0.0). 
The `main` branch reflects ongoing development and may differ from the paper version.

## Repository structure

`rjpop` contains the code for the inference framework, `scripts` contains scripts for processing GWTC data into the products used by `rjpop`, and `examples` contains prior configurations and argument files for several runs used in the associated paper.

```
rjpop/
├── main.py       # Main MCMC script (argument parsing, sampler setup, post-processing)
├── data.py       # Prior parsing, likelihood, and post-processing utilities
├── pdfs.py       # Population model distributions (mass, spin, redshift, rates)
├── moves.py      # Custom MCMC moves (RateMove, UpdateKDEMove, GaussianLeafMove)
├── plot.py       # Plotting utilities
├── utils.py      # General utilities (convergence diagnostics, I/O helpers)
└── xp.py         # NumPy/CuPy abstraction for optional GPU acceleration

scripts/
├── process_GWTC_PE.py    # Extract and preprocess LVK PE samples from GWTC catalogs
├── process_GWTC_injs.py  # Preprocess O1–O4a injection campaign
├── process_O3_injs.py    # Preprocess O3-only injection campaign
└── process_MDC_PE.py     # Preprocess MDC (mock data challenge) PE samples

examples/
├── prior_skewt.json      # Single-branch skew-t mass model (1–4 components)
├── prior_skewt_ID.json   # Skew-t with independent-diagonal (symmetric copula) mass model
├── prior_skewt_z.json    # Skew-t with per-component redshift evolution
├── prior_NPLNP.json      # Two-branch model: power-law + Gaussian mass components
├── args_skewt.txt        # Sampler arguments for prior_skewt
├── args_skewt_ID.txt     # Sampler arguments for prior_skewt_ID
├── args_skewt_z.txt      # Sampler arguments for prior_skewt_z
└── args_NPLNP.txt        # Sampler arguments for prior_NPLNP
```

## Installation

**Dependencies:**
- [Eryn](https://github.com/mikekatz04/Eryn) — the ensemble sampler backend
- Standard scientific Python stack: `numpy`, `scipy`, `astropy`, `h5py`, `matplotlib`, `seaborn`, `corner`, `sklearn`
- Optional: `cupy` for GPU acceleration
- Optional: `popsummary` for overlaying LVK reference results on plots

Install Eryn from source:
```bash
git clone https://github.com/mikekatz04/Eryn
pip install -e Eryn/
```

## Usage

### 1. Prepare data

Process LVK parameter estimation samples and injections into the NPZ format expected by the sampler:

```bash
python scripts/process_GWTC_PE.py --in_path /path/to/lvk/samples --out_path ./data
python scripts/process_GWTC_injs.py  # edit inpath/outpath at top of script
```

The PE samples NPZ should contain arrays keyed by parameter name (`mass_1_source`, `mass_2_source`, `mass_ratio`, `chi_eff`, `redshift`) plus `prior` (the PE sampling prior evaluated at each sample). The injections NPZ additionally needs `w` (mixture weights), `total_generated`, and `Tobs_yr`.

### 2. Write a prior configuration

The prior is specified as a JSON file — a list of branch dictionaries. Each branch (except the last `"global"` branch) defines a set of RJ-enabled subpopulation components. The global branch holds parameters shared across all components (e.g., redshift evolution).

See the [examples/](examples/) directory for complete prior files used in the paper. For instance, [examples/prior_skewt.json](examples/prior_skewt.json) defines a single branch of 1–4 components, each with a Jones-Faddy skew-t mass distribution and a shared power-law redshift evolution:

```json
[
  {
    "__branch__": "local",
    "__ncomp__": [1, 4],
    "mass": {
      "__model__": "m1_q",
      "mass_1_source": {
        "__model__": "skew-t",
        "logalpha": [-1.0, 2.0],
        "logkappa": [-2.0, 2.0],
        "loc": [6.0, 60.0],
        "scale": [1.0, 15.0]
      },
      "mass_ratio": {
        "__model__": "gauss",
        "loc": [0.1, 1.0],
        "scale": [0.03, 1.0]
      }
    },
    "chi_eff": {
      "__model__": "gauss",
      "loc": [-1.0, 1.0],
      "scale": [0.03, 1.0]
    },
    "__factor__": [0.05, 0.01, 0.5, 0.1, 0.03, 0.03, 0.003, 0.003, 0.3]
  },
  {
    "__branch__": "global",
    "redshift": {
      "__model__": "PL",
      "gamma": [-6.0, 6.0]
    },
    "mass_1_source": {
      "xmin": [3, 7]
    }
  }
]
```

**Key conventions:**
- `"__ncomp__": [min, max]` — range of allowed component counts for this branch
- Parameter ranges `[min, max]` are sampled uniformly; a scalar fixes the parameter
- `"__factor__"` sets the Gaussian proposal scale for each hyperparameter (in the same order they appear in the prior)
- Available models are defined in `pdfs.py` and listed in `pdfs.MODELS`

### 3. Run the sampler

The args files in [examples/](examples/) can be passed directly to the submission script:

```bash
python rjpop/main.py $(cat examples/args_skewt.txt)
```

Or equivalently:

```bash
python rjpop/main.py \
    --prior examples/prior_skewt.json \
    --PE_samples /path/to/PE_samples.npz \
    --injs /path/to/injections.npz \
    --outdir results/ \
    --label skewt \
    --nwalkers 20 \
    --ntemps 5 \
    --Tmax 5 \
    --burn 20000 \
    --nsteps 30000 \
    --min_sep 1 \
    --rj_num_try 2
```

Key arguments:

| Argument | Description |
|---|---|
| `--prior` | Path to prior JSON file |
| `--PE_samples` | Path to PE samples NPZ |
| `--injs` | Path to injections NPZ |
| `--nwalkers` | Walkers per temperature (default: 50) |
| `--ntemps` | Number of temperatures for parallel tempering (default: 5) |
| `--nsteps` | Number of MCMC steps (default: 2000) |
| `--burn` | Burn-in steps with KDE updates (default: 0) |
| `--kde_update` | Update RJ KDE proposals every N steps; 0 = use prior (default: 0) |
| `--min_sep` | Minimum separation between components in whitened feature space (default: 3.0) |
| `--rate_prior` | `[min max]` uniform rate prior per component in Gpc⁻³ yr⁻¹ (default: 0.05 100) |
| `--LVK_plot` | Overlay LVK reference results: `default`, `spline`, or `none` (default: `default`) |
| `--lvk_res_path` | Path to LVK population data release directory (required if `--LVK_plot != none`) |

### 4. Outputs

Results are written to `<outdir>/<label>/`:

```
backend.h5                              # Eryn HDF5 chain backend
data/ppds.npy                           # Marginal PPDs (pickled dict)
data/ppds_submodels.npy                 # Per-component PPDs
figures/                                # Auto-generated plots
prior.json                              # Copy of prior config
hyperparameter_ordering_metainfo.json   # Hyperparameter metadata
settings.json                           # Run settings
```

## Population models

The following models are available (see `pdfs.py` for details):

**Mass models** (`mass_1_source`, `mass_2_source`): `skew_t`, `gaussian`, `gen_gaussian`, `SGED`, `smoothed_powerlaw`, `LVK_Plancktaper_powerlaw`

**Joint mass models** (`mass`): `m1_q` (independent $m_1$ and $q|m_1$), `m1_q_m2max` (with $m_2 \leq m_\mathrm{max}$ cutoff), `gaussian_copula`, `sym_gaussian_copula`

**Spin models** (`chi_eff`): `gaussian`, `SGED`, `gen_gaussian`

**Redshift/rate models** (`redshift`): `PL_rate` (power-law $\psi(z) \propto (1+z)^\gamma$), `MD_rate` (Madau–Dickinson)

## GPU acceleration

If [CuPy](https://cupy.dev/) is installed, the likelihood and model evaluations automatically run on GPU. No code changes are needed — the `xp.py` module handles the NumPy/CuPy dispatch transparently. Note that this
code has not been tested on CPUs.

## Citation

If you use this code, please cite the associated paper (in prep.) and [Eryn](https://github.com/mikekatz04/Eryn).
