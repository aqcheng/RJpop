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

To install this code, simply `git clone` this repository
```bash
git clone https://github.com/aqcheng/RJpop.git
```
or simply download it.

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
or alternatively, `uv pip install -e Eryn/`.

## Usage

### 1. Prepare data

The scripts in the `scripts` directory are used to process the LVK parameter estimation samples and injections into the format expected by the sampler. The path to the data files, as well as the target output paths, can be specified as arguments to the script, e.g.

```bash
python scripts/process_GWTC_PE.py --in_paths /path/to/lvk/PE/samples/1 /path/to/lvk/PE/samples/2 --out_path /path/to/output
python scripts/process_GWTC_injs.py --in_path /path/to/lvk/injections --out_path /path/to/output
```

`process_GWTC_PE.py` takes for `--in_paths` the directories to the parameter estimation data releases for each observing run. For an analysis on GWTC-4, one would input the directories downloaded from Zenodo records [8177023](https://zenodo.org/records/8177023) and [17602505](https://zenodo.org/records/17602505) for O1-3 and O4a, respectively. 

`process_GWTC_injs.py` expects for `--in_path` the path to the injections of the cumulative search sensitivity estimates, with polar spins. For GWTC-4, this is `mixture-semi_o1_o2-real_o3_o4a-polar_spins_20250503134659UTC.hdf` of [16740128](https://zenodo.org/records/16740128).

The PE samples NPZ should contain arrays keyed by parameter name (`mass_1_source`, `mass_2_source`, `mass_ratio`, `chi_eff`, `redshift`) plus `prior` (the PE sampling prior evaluated at each sample). The injections NPZ additionally needs `w` (mixture weights), `total_generated`, and `Tobs_yr`.

#### the `config.json` file

Alternatively, the paths to the data files can be specified in a `config.json` file in the root `RJpop` directory, which is used as the default if no arguments are provided. This is provided for the user's convenience to avoid having to specify data paths for each script. An example can be found in `examples/example_config.json`. 

The allowed config settings are as follows:
- `lvk_out_path`: The path to the output directory for the processed injection and PE `.npz` files. Used in `process_GWTC_PE.py` and `process_GWTC_injs.py` as the  output directory if `--out_path` is not specified. `rjpop/main.py` (the main MCMC script) will also load the PE and injection data from here if they are not supplied as arguments
- `gwtc_injs_in_path`: The path to the mixture injections file, used in `process_GWTC_injs.py` if `--in_path` is not specified.
- `gwtc_pe_in_paths`: A list of paths to the GWTC PE samples files. Used in `process_GWTC_PE.py` if `--in_paths` is not specified.

A number of other settings can also be specified to be used in `main.py`:
- `RJpop_out_path`: The directory for the output data products of the inference
- `o4a-astro_path` and/or `o4b-astro_path`: Paths to the LVK population analysis results for [GWTC-4](https://zenodo.org/records/16911563) and/or [GWTC-5](https://zenodo.org/records/20292639), respectively. This is just used to automatically plot the inferered posterior distributions against the LVK analyses. If `o4b-astro_path` is given, then the `Default BBH` GWTC-5 analysis is plotted; otherwise if `o4a-astro_path` is given, then the `Default BBH` GWTC-4 analysis is plotted.

### 2. Write a prior configuration

The prior is specified as a JSON file — a list of branch dictionaries. Each branch (except the last `"global"` branch) defines a set of RJ-enabled subpopulation components. The global branch holds parameters shared across all components (e.g., redshift evolution).

See the [examples/](examples/) directory for complete prior files used in the paper. For instance, [examples/prior_skewt.json](examples/prior_skewt.json) defines a single branch of 1–4 components, each with a Jones-Faddy skew-t primary mass distribution, Gaussian conditional mass ratio distribution, Gaussian effective spin distribution, and a shared global power-law redshift evolution. Note the hierarchical specification of `"mass"` because $m_1, q$ is a joint distribution.

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
- `"__ncomp__": [nmin, nmax]` — range of allowed component counts for this branch
- Parameter ranges `[min, max]` indicate a uniform prior, whereas specifying a scalar `x` fixes it at that value. 
- `"__factor__"` sets the Gaussian proposal scale for each hyperparameter (in the same order they appear in the prior), necessary for reversible-jump branches (i.e. when $n_{\min} < n_{\max}$) where the Gibbs sampling move is Gaussian. For non-reversible-jump branches, the affine-invariant "stretch-move" is used and no specification of `"__factor__"` is needed.
- Available models for `"__model__"` are defined in `rjpop/pdfs.py` and listed in the `pdfs.MODELS` dictionary

### 3. Run the sampler

Run the main script `rjpop/main.py`:
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

See the argument files in [examples/](examples/) for examples.

Key arguments:

| Argument | Description |
|---|---|
| `--prior` | Path to prior JSON file |
| `--PE_samples` | Path to PE samples NPZ |
| `--injs` | Path to injections NPZ |
| `--seed` | Random seed for reproducibility (default: 1) |
| `--label` | Label for output subdirectory (default: `out`) |
| `--outdir` | Output directory. Defaults to `RJpop_out_path` from `config.json` |
| `--nwalkers` | Walkers per temperature (default: 50) |
| `--ntemps` | Number of temperatures for parallel tempering (default: 5) |
| `--Tmax` | Maximum temperature for parallel tempering (default: None) |
| `--min_sep` | Minimum separation in feature space between components along any dimension (default: 1) |
| `--nsteps` | Number of MCMC steps (default: 20000) |
| `--burn` | Burn-in steps (default: 20000) |
| `--rj_num_try` | Number of tries for MT reversible-jump MCMC (default: 1) |
| `--group_chunk_size` | Evaluate the likelihood in batches of at most this many groups to cap peak GPU memory. Default: None (all groups in one pass) |

A `--test` mode is also available, which is a helpful mode for debugging and verifying the installation with verbose outputs. It downsamples inputs to 500 samples, runs 10 likelihood tests, 5 burn steps and 50 short MCMC steps. To run it, add the `--test` flag to the run that you would like to test.

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

The following models are available (see `pdfs.py` for details, as well as Section 6.2 of the [paper](https://arxiv.org/pdf/2605.25980)):

**(Primary) mass models** (can be used for $p(m_1)$ or $p(m_2)$): `skew_t`, `gaussian`, `gen_gaussian`, `SGED`, `smoothed_powerlaw`, `LVK_Plancktaper_powerlaw`

**Joint mass models** (`mass`): `m1_q` ($p(m_1, q) = p(m_1) p(q | m_1)$), `m1_q_m2max` (with $m_2 \leq m_\mathrm{max}$ cutoff), `gaussian_copula`, `sym_gaussian_copula`

**Spin models** (`chi_eff`): `gaussian`, `SGED`, `gen_gaussian`

**Redshift/rate models** (`redshift`): `PL_rate` (power-law $\psi(z) \propto (1+z)^\gamma$), `MD_rate` (Madau–Dickinson)

## GPU acceleration

If [CuPy](https://cupy.dev/) is installed, the likelihood and model evaluations automatically run on GPU; otherwise they run on CPU with `numpy`. CPU can also be forced with the `--cpu` flag.

## Citation

If you use this code, please cite [our paper](https://arxiv.org/abs/2605.25980) and [Eryn](https://github.com/mikekatz04/Eryn).
