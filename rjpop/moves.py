from copy import deepcopy

import data
import numpy as np
import utils

from eryn.moves import GaussianMove, MHMove
from eryn.prior import ProbDistContainer
from eryn.utils import Update


class GaussianLeafMove(MHMove):
    """Gaussian Metropolis–Hastings move that perturbs one occupied leaf.

    This variant mirrors Eryn's standard Gaussian move but, instead of
    proposing updates for every leaf, it selects **one** non-empty leaf per
    walker and temperature (based on ``branches_inds[branch_name]``) and draws
    a Gaussian displacement only there. This keeps the move usable even when
    reversible-jump steps create empty leaves that would otherwise waste
    proposals.

    Parameters
    ----------
    sigmas : array_like
        Per-dimension standard deviations for the Gaussian move added to the
        chosen leaf.
    branch_name : str, optional
        Branch to update. If ``gibbs_sampling_setup`` is provided as a tuple,
        it overrides this.
    gibbs_sampling_setup : tuple(str, ndarray), optional
        ``(branch_name, mask)`` where ``mask`` has shape
        ``(nleaves_max, ndim)`` indicating which parameters to update for each
        leaf (used to mirror Eryn's ``GaussianMove`` Gibbs behavior).
    *args, **kwargs : Any
        Forwarded to :class:`eryn.moves.MHMove`.
    """

    def __init__(self, sigmas, branch_name, *args, **kwargs):

        self.sigmas = sigmas
        self.branch_name = branch_name

        super().__init__(*args, **kwargs)

    def get_proposal(self, branches_coords, random, branches_inds, **kwargs):

        coords = branches_coords[self.branch_name]
        inds = branches_inds[self.branch_name]

        ntemps, nwalkers, nleaves_max, ndim = coords.shape

        # choose one occupied leaf per (temp, walker) using provided RNG
        random_choice = np.where(inds, random.rand(ntemps, nwalkers, nleaves_max), -np.inf)
        result_indices = np.argmax(random_choice, axis=2)
        has_candidate = inds.any(axis=2)

        coords_leaf = np.take_along_axis(coords, result_indices[..., None, None], axis=2)[:, :, 0]

        new_points = np.copy(coords_leaf)
        if np.any(has_candidate):
            noise = random.randn(*coords_leaf[has_candidate].shape) * self.sigmas
            new_points[has_candidate] = coords_leaf[has_candidate] + noise

        proposal = {self.branch_name: np.copy(coords)}
        row_indices, col_indices = np.meshgrid(
            np.arange(ntemps), np.arange(nwalkers), indexing="ij"
        )
        proposal[self.branch_name][
            row_indices[has_candidate],
            col_indices[has_candidate],
            result_indices[has_candidate],
            :,
        ] = new_points[has_candidate]

        factors = np.zeros((ntemps, nwalkers))

        return proposal, factors


class KDE_prop(ProbDistContainer):
    """Proposal wrapper that samples directly from an N-D KDE.

    The distribution relies solely on the provided KDE: sampling uses
    ``kde.resample`` and densities delegate to ``kde.pdf`` / ``kde.logpdf``.
    Works for any dimensionality supported by the KDE (points shaped
    ``(..., ndim)``).

    Parameters
    ----------
    kde : object
        KDE-like object with ``resample(n)``, ``pdf(points)`` and
        ``logpdf(points)`` methods (e.g., ``scipy.stats.gaussian_kde``).
    """

    def __init__(self, kde):

        self.kde = kde

        # infer dimensionality from a single resample
        sample = np.asarray(self.kde.resample(1))
        if sample.ndim == 1:
            self.ndim = sample.shape[0]
        else:
            self.ndim = sample.shape[0]

    def rvs(self, size=1):
        """Draw samples from the KDE.

        Parameters
        ----------
        size : int or tuple, optional
            Number of samples to draw. If tuple, the result is reshaped to
            ``size + (ndim,)``. Default: 1.

        Returns
        -------
        np.ndarray
            Sample(s) shaped ``(ndim,)`` for a single draw or ``size + (ndim,)``
            for multiple draws.
        """

        if not isinstance(size, (int, tuple)):
            raise ValueError("size must be an integer or tuple of ints.")

        size_tuple = size if isinstance(size, tuple) else (size,)
        total = int(np.prod(size_tuple))

        kde_samples = np.asarray(self.kde.resample(total)).T  # shape (total, ndim)

        if total == 1 and size_tuple == (1,):
            return kde_samples[0]

        return kde_samples.reshape(size_tuple + (self.ndim,))

    def pdf(self, points):
        """Evaluate the KDE PDF at ``points`` (supports broadcasted shapes)."""

        pts = np.asarray(points)
        pts = pts.reshape((-1, self.ndim))
        pdf_vals = self.kde.pdf(pts.T)
        return np.asarray(pdf_vals).reshape(points.shape[:-1])

    def logpdf(self, points):
        """Evaluate the KDE log-PDF at ``points`` (supports broadcasted shapes)."""

        pts = np.asarray(points)
        pts = pts.reshape((-1, self.ndim))
        log_pdf_vals = self.kde.logpdf(pts.T)
        return np.asarray(log_pdf_vals).reshape(points.shape[:-1])

    def copy(self):
        return deepcopy(self)


def kde_dist(kde):
    """Factory for an N-D KDE-backed proposal distribution."""
    return KDE_prop(kde)


class _isotropic_proposal(object):
    """
    Lifted directly from eryn.
    """

    def __init__(self, scale, factor, mode):
        self.index = 0
        self.scale = scale

        if isinstance(scale, float):
            self.invscale = 1.0 / scale
        else:
            self.invscale = np.linalg.inv(np.linalg.cholesky(scale))

        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError("'factor' must be >= 1.0")
            self._log_factor = np.log(factor)

        if mode not in self.allowed_modes:
            raise ValueError(
                ("'{0}' is not a recognized mode. Please select from: {1}").format(
                    mode, self.allowed_modes
                )
            )
        self.mode = mode

    def get_factor(self, rng):
        if self._log_factor is None:
            return 1.0
        return np.exp(rng.uniform(-self._log_factor, self._log_factor))

    def get_updated_vector(self, rng, x0):
        return x0 + self.get_factor(rng) * self.scale * rng.randn(*(x0.shape))

    def __call__(self, x0, rng):
        nw, nd = x0.shape
        xnew = self.get_updated_vector(rng, x0)
        if self.mode == "random":
            m = (range(nw), rng.randint(x0.shape[-1], size=nw))
        elif self.mode == "sequential":
            m = (range(nw), self.index % nd + np.zeros(nw, dtype=int))
            self.index = (self.index + 1) % nd
        else:
            return xnew, np.zeros(nw)
        x = np.array(x0)
        x[m] = xnew[m]
        return x, np.zeros(nw)


class _orthogonal_proposal(_isotropic_proposal):
    """
    Proposes a sum-preserving move.
    """

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        z = rng.randn(*x0.shape)
        mean = np.mean(z, axis=-1, keepdims=True)
        delta = z - mean  # zero sum move
        return x0 + self.scale * self.get_factor(rng) * delta


class _radial_proposal(_isotropic_proposal):
    """Proposes a move that moves all leaves equally"""

    allowed_modes = ["vector"]

    def get_updated_vector(self, rng, x0):
        return x0 + self.scale * self.get_factor(rng) * rng.randn(x0.shape[0], 1)


class RateMove(GaussianMove):
    """
    Proposes a move that either preserves the sum or moves all components equally.
    """

    allowed_types = ["orthogonal", "radial"]

    def __init__(self, sigma, branch_name, type, factor=None, **kwargs):

        if type == "orthogonal":
            proposal = _orthogonal_proposal(scale=sigma, factor=factor, mode="vector")
        elif type == "radial":
            proposal = _radial_proposal(scale=sigma, factor=factor, mode="vector")
        else:
            raise ValueError(f"Unknown type {type}. Type must be one of {self.allowed_types}")

        self.all_proposal = {branch_name: proposal}
        super(GaussianMove, self).__init__(**kwargs)


class UpdateKDEMove(Update):
    """
    Replace/update the reversible jump move with KDEs with the more recent samples.
    """

    def __init__(self, nsamples_use=1000, log=None):
        self.nsamples_use = nsamples_use
        self.log = log
    
    def _print(self, msg):
        if self.log is None:
            print(msg)
        else:
            utils.print_to(self.log, msg)

    def __call__(self, iter, last_sample, sampler):
        """Call update function.

        Args:
            iter (int): Iteration of the sampler.
            last_sample (obj): Last state of sampler (:class:`eryn.state.State`).
            sampler (obj): Full sampler oject (:class:`eryn.ensemble.EnsembleSampler`).
        """

        # print previous acceptance fraction
        acc_frac, acc_frac_std = utils.get_move_acceptance_fraction(sampler.rj_moves, temp_index=0)
        self._print(
            f"Updating RJ KDE (iter={iter}). Acceptance fraction: {acc_frac:.2g} +/- {acc_frac_std:.2g}"
        )

        thin = utils.get_thin_from_chain(sampler.get_chain())
        discard = int(max(iter - self.nsamples_use / sampler.nwalkers * thin, 0))
        chain = sampler.get_chain(discard=discard, thin=thin, temp_index=0)

        kdes_dict, weights_dict = data.subpop_kdes_from_samples(chain)

        rj_move_class = sampler.rj_moves[0].__class__
        num_try = sampler.rj_moves[0].num_try if hasattr(sampler.rj_moves[0], "num_try") else None

        rj_moves, rj_weights = [], []
        for bn, kdes in kdes_dict.items():
            for kde, wt in zip(kdes, weights_dict[bn], strict=False):
                # annoying but basically have to redo ensemble.__init__ actions
                rj_move = rj_move_class(
                    {bn: kde_dist(kde)},
                    num_try=num_try,
                    nleaves_min=data.nleaves_min_dict,
                    nleaves_max=data.nleaves_max_dict,
                    gibbs_sampling_setup=bn,
                    temperature_control=sampler.temperature_control,
                )
                rj_move.accepted = np.zeros((sampler.ntemps, sampler.nwalkers), dtype=np.int32)
                rj_moves.append(rj_move)
                rj_weights.append(wt)
        # renormalize weights
        rj_weights = np.asarray(rj_weights)
        rj_weights /= rj_weights.sum()

        # remove old rj moves from 'all_moves'
        for old_rj_move in sampler.rj_moves:
            for old_move_key in list(sampler.all_moves.keys()):
                if old_move_key.startswith(old_rj_move.__class__.__name__):
                    sampler.all_moves.pop(old_move_key)

        sampler.rj_moves = rj_moves
        sampler.rj_weights = rj_weights
        # update 'all_moves' - we are updating all of the RJ moves
        for i, move in enumerate(rj_moves):
            sampler.all_moves[f"{move.__class__.__name__}_{i}"] = move
        sampler.move_keys = list(sampler.all_moves.keys())

        # note that we don't update the `move_info` stored in the backend, so that index
        # labels across updates will have shared acceptance fractions. The true acceptance
        # fractions per updated KDE move will be stored in the move itself (and printed to log)

        return list(zip(rj_moves, rj_weights, strict=True))
