"""
main.py

Population inference for gravitational-wave (GW) events using reversible-jump MCMC (RJ-MCMC).

This script uses the eryn library to perform hierarchical Bayesian inference on GW event data, modeling the population with a mixture of power-law and Gaussian components, and accounting for selection effects.

Inputs: LVC event samples, injection samples, and prior information.
Outputs: Posterior samples for population parameters, including mixture weights, power-law slopes, Gaussian means/widths, spin parameters, etc.

Sections:
1. Imports and configuration
2. Argument parsing
3. Likelihood definition
4. Data loading and preprocessing
5. Model/parameter setup
6. Sampler initialization and run
7. Output saving
"""

import os

# Limit the number of threads for reproducibility and efficiency
os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline as spline

from eryn.ensemble import EnsembleSampler
from eryn.state import State
# from eryn.prior import PriorContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, DistributionGenerateRJ
from eryn.moves.tempering import make_ladder
from scipy.interpolate import RegularGridInterpolator as splinend
from eryn.prior import ProbDistContainer, uniform_dist

import pdfs as pdfs_func

from astropy.cosmology import Planck15
import astropy.units as u


# import corner as triangle

import time
import argparse

# Use GPU arrays if cupy is available, otherwise fall back to numpy
try:
    import cupy as xp
    xp.cuda.runtime.setDevice(1)
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    print('use cpu')


def parser():
    parser.add_argument('-seed', '--seed', type=int, default=1,
                        help='Random seed for reproducibility')
    parser.add_argument('-mass_model', '--mass_model', type=str, default='skew-t', choices=['skew-t', 'PL'],
                        help='Main mass model to use. Options: skew-t, PL')
    parser.add_argument('-include_gauss', '--include_gauss', action='store_true',
                        help='Include Gaussians as a secondary mixture model')
    parser.add_argument('-ncomp_max', '--ncomp_max', type=int, default=10,
                        help='Maximum number of components for the mass model')
    parser.add_argument('-ncomp_min', '--ncomp_min', type=int, default=1,
                        help='Minimum number of components for the mass model')
    parser.add_argument('-test', '--test', action='store_true',
                        help='Run in test mode (fewer steps)')
    parser.add_argument('-zmax', '--zmax', type=float, default=2.3,
                        help='Maximum redshift for population modeling')
    parser.add_argument('-use_mchirp', '-use_mchirp', action='store_true',
                        help='Use chirp mass instead of m1')
    return parser.parse_args()

args=parser()
xp.random.seed(args.seed)


def loglike(xs, groups):
    """
    Log-likelihood function for hierarchical population inference.
    Args:
        xs: Tuple of parameter arrays for each branch (gauss, pl, global, spins, q)
        groups: Tuple of group indices for each branch
    Returns:
        logl: Log-likelihood array for each walker
    """
    # Unpack parameters and group indices
    xgauss0, xpl, xglobal, xspins, xqpl = xs
    group_main, group_gauss, group_global, group_spins, group_qpl = groups

    xgauss = xp.copy(xp.asarray(xgauss0))

    # Unpack varying parameters for each branch
    values_pl = {par: xpl[:, kpar] for kpar, par in enumerate(params_vary['pl'])}
    values_global = {par: xglobal[:, kpar] for kpar, par in enumerate(params_vary['global'])}
    values_spins = {par: xspins[:, kpar] for kpar, par in enumerate(params_vary['spins'])}

    # Compute beta distribution parameters for spin magnitude
    values_spins['abeta'] = (values_spins['meanchi'] - values_spins['meanchi'] ** 2 - values_spins['varchi']) * (values_spins['meanchi'] / values_spins['varchi'])
    values_spins['bbeta'] = (values_spins['meanchi'] - values_spins['meanchi'] ** 2 - values_spins['varchi']) * ((1 - values_spins['meanchi']) / values_spins['varchi'])

    betaq = np.copy(xqpl[:, 0])

    # Choose mass variable (m1 or mchirp)
    if args.use_mchirp == 0:
        data_m = data['m1']
        data_m1 = data['m1']
    else:
        data_m = data['mchirp']
        data_m1 = data['m1']

    # Compute population PDFs for each event sample
    pdf_m1_z = pdfs_func.m1_z_pdf_gauss_pl(data_m, data['z'], xgauss, group_gauss, values_global, group_global, dVc_spl, return_norm=False, to_norm=False, zmax=zmax)
    pdf_q = pdfs_func.q_pdf_sharp(data['q'], data_m1, betaq, values_global)
    pdf_tilt = pdfs_func.spin_tilt_pdf(data['tilt1'], data['tilt2'], values_spins)
    pdf_chi1 = pdfs_func.spin_mag_pdf(data['chi1'], values_spins)
    pdf_chi2 = pdfs_func.spin_mag_pdf(data['chi2'], values_spins)

    # Compute total event PDF and likelihood integrand
    pdf = pdf_m1_z * pdf_q * pdf_tilt * pdf_chi1 * pdf_chi2
    integrand = (pdf / (data['priors'][None, :])).reshape(-1, nevents, nsamp)

    bfs = xp.sum(integrand, axis=2)

    # Effective sample size check for importance sampling
    neffs = bfs ** 2 / xp.sum(integrand ** 2, axis=2)
    bfs[neffs < nevents] = 0

    logl = xp.sum(xp.log((1. / nsamp) * bfs), axis=1)
    first_logl = logl.copy()

    # --- Selection function term (for normalization) ---
    if args.use_mchirp == 0:
        injections_m = injections['m1']
        injections_m1 = injections['m1']
    else:
        injections_m = injections['mchirp']
        injections_m1 = injections['m1']

    pdf_m1_z_selection = pdfs_func.m1_z_pdf_gauss_pl(injections_m, injections['z'], xgauss, group_gauss, values_global, group_global, dVc_spl, return_norm=False, to_norm=False, zmax=zmax)
    pdf_q_selection = pdfs_func.q_pdf_sharp(injections['q'], injections_m1, betaq, values_global)
    pdf_tilt_selection = pdfs_func.spin_tilt_pdf(injections['tilt1'], injections['tilt2'], values_spins)
    pdf_chi1_selection = pdfs_func.spin_mag_pdf(injections['chi1'], values_spins)
    pdf_chi2_selection = pdfs_func.spin_mag_pdf(injections['chi2'], values_spins)

    pdf_selection = pdf_m1_z_selection * pdf_q_selection * pdf_tilt_selection * pdf_chi1_selection * pdf_chi2_selection

    # Importance sampling variance check for selection function
    mu_selection = (1. / injections['ninjs']) * xp.sum(pdf_selection / injections['priors'][None, :], axis=1)
    var_selection = (1. / injections['ninjs'] ** 2) * xp.sum((pdf_selection / injections['priors'][None, :]) ** 2, axis=1) - mu_selection ** 2 / injections['ninjs']
    Neff = mu_selection ** 2 / var_selection

    selection_function_term = mu_selection / np.exp((3. + nevents) / (2. * Neff))
    selection_function_term[Neff < 4 * nevents] = xp.inf

    # Sanity checks: set selection function to zero for invalid samples
    selection_function_term[xp.sum(pdf_m1_z_selection < 0, axis=1) > 0] = 0
    selection_function_term[xp.sum(xp.isinf(pdf_m1_z_selection), axis=1) > 0] = 0
    selection_function_term[xp.sum(xp.isinf(pdf_q_selection), axis=1) > 0] = 0
    selection_function_term[xp.sum(pdf_q_selection < 0, axis=1) > 0] = 0

    logl += -selection_function_term

    # More sanity checks: set logl to large negative value for invalid cases
    logl[xp.isinf(first_logl)] = -1E300
    logl[xp.isinf(logl)] = -1E300
    logl[selection_function_term == 0] = -1E300

    # Require at least one active group per walker/temperature
    groups_to_concatenate = [group_gauss]
    if args.only_gauss == 0:
        groups_to_concatenate += [group_pl]
    allgroups = np.unique(np.concatenate(groups_to_concatenate))
    indsabsent = np.setdiff1d(np.arange(len(xglobal)), allgroups)
    logl[indsabsent] = -1E300

    # Hard prior boundaries for beta distribution
    logl[values_spins['abeta'] < 0] = -1E300
    logl[values_spins['bbeta'] < 0] = -1E300

    # Debugging: catch NaNs
    if xp.sum(xp.isnan(logl)):
        breakpoint()
    try:
        logl = logl.get()  # For cupy arrays
    except AttributeError:
        pass
    return logl


xp.random.seed(args.seed)



# -----------------------------
# 4. Data loading and preprocessing
# -----------------------------

# Load injection samples (for selection function normalization)

file_injection=np.load('/data/atoubiana/ligo_pop_obs/selection_function_elements.npz')

injections={}
injections['m1']=xp.asarray(file_injection['m1s'])
injections['q']=xp.asarray(file_injection['qs'])
injections['z']=xp.asarray(file_injection['zs'])
injections['chi1']=xp.asarray(file_injection['chi1s'])
injections['chi2']=xp.asarray(file_injection['chi2s'])
injections['tilt1']=xp.asarray(file_injection['tilt1s'])
injections['tilt2']=xp.asarray(file_injection['tilt2s'])
injections['priors']=xp.asarray(file_injection['inj_priors'])
injections['ninjs']=xp.asarray(file_injection['ninjs'][0])


# Load LVC event samples (data to fit)
data_lvc = np.load('/data/atoubiana/ligo_pop_obs/lvc_data_lvc_samples_full.npz')

nevents = data_lvc['nevents'][0]  # Number of GW events
nsamp = data_lvc['nsamples'][0]   # Number of samples per event

# Convert event sample arrays to xp (numpy/cupy) and flatten
# Each key is an array of shape (nevents * nsamp,)
data = {}
data['m1'] = xp.asarray(data_lvc['m1s'].reshape(-1))
data['q'] = xp.asarray(data_lvc['qs'].reshape(-1))
data['z'] = xp.asarray(data_lvc['zs'].reshape(-1))
data['chi1'] = xp.asarray(data_lvc['chi1s'].reshape(-1))
data['chi2'] = xp.asarray(data_lvc['chi2s'].reshape(-1))
data['tilt1'] = xp.asarray(data_lvc['tilt1s'].reshape(-1))
data['tilt2'] = xp.asarray(data_lvc['tilt2s'].reshape(-1))
data['priors'] = xp.asarray(data_lvc['priors'].reshape(-1))


# Convert to chirp mass and adjust priors if requested

if args.use_mchirp==1:
    data['priors']=data['priors']*(1+data['q'])**(1./5)/(data['q']**(3./5.))
    data['mchirp']=data['m1']*data['q']**(3./5.)/((1+data['q'])**(1./5.))

    injections['priors']=injections['priors']*(1+injections['q'])**(1./5)/(injections['q']**(3./5.))
    injections['mchirp']=injections['m1']*injections['q']**(3./5.)/((1+injections['q'])**(1./5.))


# -----------------------------
# 5. Model/parameter setup
# -----------------------------
# Spline comoving volume for redshift PDF

if args.zmax is None:
    zmin,zmax=0,5
else:
    zmin,zmax=0,args.zmax

zs=xp.linspace(zmin,zmax,1000)

try:
    dVc=xp.copy(Planck15.differential_comoving_volume(zs.get()))*4.*np.pi
    dVc_spl=spline(zs.get(),dVc)
except AttributeError:
    dVc=xp.copy(Planck15.differential_comoving_volume(zs))*4.*np.pi
    dVc_spl=spline(zs,dVc)



    
# Define branches and parameters for population model

branch_names=['gauss','pl','global','spins','q']

params_vary={}
params_vary['pl']=['log10_lambda_pl','alpha','kappa_z']
params_vary['gauss']=['log10_lambda_peak','mum','sigmam','kappa_z']
params_vary['global']=['mmin','mmax','deltam']
params_vary['spins']=['zeta','sigmat','meanchi','varchi']
params_vary['q']=['betaq']

if args.only_gauss==1:
    params_vary['global']=['mmin']


dict_dims={}
dict_nleaves_min={}
dict_nleaves_max={}

# Define number of leaves (mixture components) for each branch

dict_nleaves_min['pl'],dict_nleaves_max['pl']=1,1
if args.zero_pl==1:
    dict_nleaves_min['pl'],dict_nleaves_max['pl']=0,1
dict_nleaves_min['gauss'],dict_nleaves_max['gauss']=args.ngauss_min,args.ngauss_max
dict_nleaves_min['global'],dict_nleaves_max['global']=1,1
dict_nleaves_min['spins'],dict_nleaves_max['spins']=1,1
dict_nleaves_min['q'],dict_nleaves_max['q']=1,1

if args.only_gauss==1:
    dict_nleaves_min['gauss']=max(1,args.ngauss_min)
    dict_nleaves_min['pl'],dict_nleaves_max['pl']=0,0


for branch in branch_names:
    dict_dims[branch]=len(params_vary[branch])

ndims=[]
nleaves_min=[]
nleaves_max=[]
for branch in branch_names:
    ndims+=[dict_dims[branch]]
    nleaves_min+=[dict_nleaves_min[branch]]
    nleaves_max+=[dict_nleaves_max[branch]]

ngauss_max=args.ngauss_max
ngauss_min=args.ngauss_min


# Define prior ranges for each parameter

mins={}
maxs={}

mins['mum'],maxs['mum']=2,100
mins['sigmam'],maxs['sigmam']=1,10
    
mins['log10_lambda_peak'],maxs['log10_lambda_peak']=2.,10.


mins['alpha'],maxs['alpha']=0.,4.
mins['betaq'],maxs['betaq']=-1,7
mins['mmin'],maxs['mmin']=2,10
mins['mmax'],maxs['mmax']=30,100

mins['deltam'],maxs['deltam']=0.5,10

mins['log10_lambda_pl'],maxs['log10_lambda_pl']=2.,10.


mins['zeta'],maxs['zeta']=0,1
mins['meanchi'],maxs['meanchi']=0,0.95
mins['varchi'],maxs['varchi']=0.005,0.25
mins['sigmat'],maxs['sigmat']=0.1,4


mins['alpha_z'],maxs['alpha_z']=1.,10.
mins['theta_z'],maxs['theta_z']=0.,1.

mins['kappa_z'],maxs['kappa_z']=-3,9


priors={}
for branch in branch_names:
    dict_prior_branch={}
    for kpar,par in enumerate(params_vary[branch]):
        dict_prior_branch[kpar]=uniform_dist(mins[par],maxs[par])
    priors[branch]=ProbDistContainer(dict_prior_branch)


# -----------------------------
# 6. Sampler initialization and run
# -----------------------------
# Sampler settings
nwalkers = 40  # Number of walkers per temperature
ntemps = 5     # Number of temperatures (for parallel tempering)
tempering_kwargs = dict(ntemps=ntemps)
betas = np.linspace(1.0, 0.0, ntemps)  # Inverse temperatures

if args.test == 1:
    nsteps = 20   # Fewer steps for testing
    burn = 5
else:
    nsteps = 5000
    burn = 2000

thin_by = 1

# Output path setup
path = 'run_lvk_gwtc4'
if not os.path.exists(path):
    os.mkdir(path)
path += '/run_%d' % (args.seed)
print(path)
if not os.path.exists(path):
    os.mkdir(path)

# Initialize parameter chains (walker positions)
# coords is a dict keyed by branch name (e.g., 'gauss','pl',...), each of shape:
#   (ntemps, nwalkers, nleaves_max[branch], dict_dims[branch]) drawn from priors.
coords = {
    name: priors[name].rvs(size=(ntemps, nwalkers, nleaf,))
    for nleaf, name in zip(nleaves_max, branch_names)
}

# Build the initial sampler state. `inds` is optional and not defined here;
# we rely on `provide_groups=True` in EnsembleSampler to manage groups.
state = State(coords)

# Define MCMC moves to use (Gibbs/Stretch/Random-walk)
# For the Gaussian branch, we optionally use per-component Gibbs updates with
# a diagonal Gaussian proposal scaled by `factors` (one scale per parameter):
factors = np.array([0.005, 0.1, 0.05, 0.05])  # Step sizes for Gaussian moves (len == dict_dims['gauss'])

if args.ngauss_max > 1:
    moves = []  # Will add Gaussian moves for each component
else:
    # If only one Gaussian, use StretchMove for all
    stretch_moves = StretchMove(live_dangerously=True)
    moves = [(stretch_moves, 0.5)]

for branch in branch_names:
    if args.ngauss_max > 1 and branch == 'gauss':
        for i in range(args.ngauss_max):
            # Change parameters of a single Gaussian at a time (Gibbs)
            # `cov` is diagonal and broadcast-multiplied by `factors` so each param
            # has its own step scale. Requires len(factors) == dict_dims['gauss'].
            cov = {"gauss": np.diag(np.ones(dict_dims['gauss'])) * factors}
            gibbs_array_i = np.zeros((dict_nleaves_max['gauss'], dict_dims['gauss']), dtype='bool')
            gibbs_array_i[i, :] = True
            gaussian_move_i = GaussianMove(cov, gibbs_sampling_setup=("gauss", gibbs_array_i))
            moves += [(gaussian_move_i, 0.2)]
    else:
        # Use StretchMove for other branches
        stretch_move_i = StretchMove(gibbs_sampling_setup=branch, live_dangerously=True)
        moves += [(stretch_move_i, 0.1)]

data = {key: xp.asarray(val) for key, val in data.items()}

# RJMCMC moves to change model dimensionality (add/remove leaves)
rj_moves = []
if ngauss_max > ngauss_min:
    prior_move_gauss = DistributionGenerateRJ(
        priors, nleaves_min=dict_nleaves_min, nleaves_max=dict_nleaves_max,
        gibbs_sampling_setup="gauss"
    )
    rj_moves = [(prior_move_gauss, 0.5)]

if args.zero_pl == 1:
    # Note: as written, this overrides any Gaussian RJ move above.
    # If you want both RJ moves active, append instead of overwrite.
    prior_move_pl = DistributionGenerateRJ(
        priors, nleaves_min=dict_nleaves_min, nleaves_max=dict_nleaves_max,
        gibbs_sampling_setup="pl"
    )
    rj_moves = [(prior_move_pl, 0.5)]

if len(rj_moves) == 0:
    rj_moves = False

# -----------------------------
# 7. Run sampler and save output
# -----------------------------
ensemble = EnsembleSampler(
    nwalkers,
    ndims,  # assumes ndim_max
    loglike,
    priors,
    tempering_kwargs=dict(betas=betas),
    nbranches=len(branch_names),
    branch_names=branch_names,
    nleaves_max=nleaves_max,
    nleaves_min=nleaves_min,
    provide_groups=True,
    moves=moves,
    vectorize=True,
    rj_moves=rj_moves  # basic generation of new leaves from the prior
)

# Run the MCMC
ensemble.run_mcmc(state, nsteps, burn=burn, progress=True, thin_by=thin_by)

# Extract and reshape samples for each branch
samples_gauss = ensemble.get_chain()['gauss'][:, 0].reshape(nsteps, nwalkers, dict_nleaves_max['gauss'], dict_dims['gauss'])
samples_pl = ensemble.get_chain()['pl'][:, 0].reshape(nsteps, nwalkers, dict_nleaves_max['pl'], dict_dims['pl'])
samples_global = ensemble.get_chain()['global'][:, 0].reshape(nsteps, nwalkers, dict_nleaves_max['global'], dict_dims['global'])
samples_spins = ensemble.get_chain()['spins'][:, 0].reshape(nsteps, nwalkers, dict_nleaves_max['spins'], dict_dims['spins'])
samples_q = ensemble.get_chain()['q'][:, 0].reshape(nsteps, nwalkers, dict_nleaves_max['q'], dict_dims['q'])
# samples_z = ensemble.get_chain()['z'][:,0].reshape(nsteps, nwalkers, dict_nleaves_max['z'],dict_dims['z'])

samples_leaves_gauss = ensemble.get_nleaves()['gauss'][:, 0].reshape(nsteps, nwalkers, 1)
samples_leaves_pl = ensemble.get_nleaves()['pl'][:, 0].reshape(nsteps, nwalkers, 1)

# Thinning for output
npts = nsteps
thin = 10
# if args.samples2==1:
#     thin=5
inds_keep = np.arange(npts)[::thin]

samples_gauss = samples_gauss[inds_keep]
samples_pl = samples_pl[inds_keep]
samples_global = samples_global[inds_keep]
samples_spins = samples_spins[inds_keep]
samples_q = samples_q[inds_keep]
# samples_z = samples_z[inds_keep]
samples_leaves_gauss = samples_leaves_gauss[inds_keep]
samples_leaves_pl = samples_leaves_pl[inds_keep]

loglikes = ensemble.get_log_like()[:, 0].reshape(nsteps, nwalkers, 1)
loglikes = loglikes[::thin]

print(np.amax(loglikes))

# Save samples and log-likelihoods to .npz file
np.savez(
    path + '/samples.npz',
    samples_gauss=samples_gauss,
    samples_leaves_gauss=samples_leaves_gauss,
    samples_pl=samples_pl,
    samples_leaves_pl=samples_leaves_pl,
    samples_global=samples_global,
    samples_spins=samples_spins,
    samples_q=samples_q,
    loglikes=loglikes
)
