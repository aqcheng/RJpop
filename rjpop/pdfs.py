from abc import ABC, abstractmethod

import numpy as np
import utils
from astropy.cosmology import Planck15
from xp import EPS, INF, special, xp

ZMAX = 2


def trapz(y, x=None, dx=1.0, axis=-1):
    """
    Lifted from `numpy <https://github.com/numpy/numpy/blob/v1.15.1/numpy/lib/function_base.py#L3804-L3891>`_.

    Integrate along the given axis using the composite trapezoidal rule.
    Integrate `y` (`x`) along given axis.

    Parameters
    ==========
    y : array_like
        Input array to integrate.
    x : array_like, optional
        The sample points corresponding to the `y` values. If `x` is None,
        the sample points are assumed to be evenly spaced `dx` apart. The
        default is None.
    dx : scalar, optional
        The spacing between sample points when `x` is None. The default is 1.
    axis : int, optional
        The axis along which to integrate.

    Returns
    =======
    trapz : float
        Definite integral as approximated by trapezoidal rule.


    References
    ==========
    .. [1] Wikipedia page: http://en.wikipedia.org/wiki/Trapezoidal_rule

    Examples
    ========
    >>> trapz([1,2,3])
    4.0
    >>> trapz([1,2,3], x=[4,6,8])
    8.0
    >>> trapz([1,2,3], dx=2)
    8.0
    >>> a = xp.arange(6).reshape(2, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> trapz(a, axis=0)
    array([ 1.5,  2.5,  3.5])
    >>> trapz(a, axis=1)
    array([ 2.,  8.])
    """
    y = xp.asanyarray(y)
    if x is None:
        d = dx
    else:
        x = xp.asanyarray(x)
        if x.ndim == 1:
            d = xp.diff(x)
            # reshape to correct shape
            shape = [1] * y.ndim
            shape[axis] = d.shape[0]
            d = d.reshape(shape)
        else:
            d = xp.diff(x, axis=axis)
    ndim = y.ndim
    slice1 = [slice(None)] * ndim
    slice2 = [slice(None)] * ndim
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)
    product = d * (y[tuple(slice1)] + y[tuple(slice2)]) / 2.0
    try:
        ret = product.sum(axis)
    except ValueError:
        ret = xp.add.reduce(product, axis)
    return ret


def rescale(x, loc, scale):
    return (x - loc) / scale


class dist(ABC):
    """
    Base class for distributions. Strictly speaking, only the _pdf method is necessary to implement
    during inference.
    """

    @staticmethod
    @abstractmethod
    def _pdf(x, *args, **kwargs):
        # The core PDF logic for the standard distribution (loc=0, scale=1)
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _logpdf(x, *args, **kwargs):
        # The core logpdf logic for the standard distribution
        raise NotImplementedError

    # these are optional
    @staticmethod
    def _cdf(x, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _logcdf(x, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _ppf(x, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def pdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._pdf((x - loc) / scale, *args, **kwargs) / scale

    @classmethod
    def logpdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._logpdf((x - loc) / scale, *args, **kwargs) - xp.log(scale)

    @classmethod
    def cdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._cdf((x - loc) / scale, *args, **kwargs)

    @classmethod
    def logcdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._logcdf((x - loc) / scale, *args, **kwargs)

    @classmethod
    def ppf(cls, q, *args, loc=0, scale=1, **kwargs):
        return cls._ppf(q, *args, **kwargs) * scale + loc

    @staticmethod
    def _moments(*args, **kwargs):
        return 0.0, 0.0

    @classmethod
    def moments(cls, *args, loc=0, scale=1, **kwargs):
        mean, std = cls._moments(*args, **kwargs)
        return mean * scale + loc, std * scale


class trunc_dist(dist):
    @staticmethod
    @abstractmethod
    def _cdf(x, *args, **kwargs):
        # The core cdf logic for the standard distribution w/o limits. Required for trunc_dist
        raise NotImplementedError

    @classmethod
    def pdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):

        # Build the unnormalized truncated pdf in place to avoid full-size copies
        # (`_pdf` returns a fresh owned buffer). Multiplying by all-True masks when a
        # bound is infinite is a no-op, so those branches are simply skipped.
        pdf_unnorm = cls._pdf(rescale(x, loc, scale), *args, **kwargs)
        pdf_unnorm /= scale
        if xmin is not -INF:
            pdf_unnorm *= x >= xmin
        if xmax is not INF:
            pdf_unnorm *= x <= xmax

        # normalize
        cdf_high = cls._cdf(rescale(xmax, loc, scale), *args, **kwargs) if xmax is not INF else 1
        cdf_low = cls._cdf(rescale(xmin, loc, scale), *args, **kwargs) if xmin is not -INF else 0
        norm = cdf_high - cdf_low
        cdf_high, cdf_low = None, None  # for memory

        norm[~(norm > 0)] = xp.inf  # Guard against division by zero (degenerate truncation)

        pdf_unnorm /= norm
        return pdf_unnorm

    @classmethod
    def logpdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        x_, xmin_, xmax_ = (
            rescale(x, loc, scale),
            rescale(xmin, loc, scale),
            rescale(xmax, loc, scale),
        )

        logpdf_unnorm = xp.where(
            (x < xmin) | (x > xmax), -INF, cls._logpdf(x_, *args, **kwargs) - xp.log(scale)
        )

        # normalize
        logcdf_high = xp.where(xmax >= INF, 0.0, cls._logcdf(xmax_, *args, **kwargs))
        logcdf_low = xp.where(xmin <= -INF, -INF, cls._logcdf(xmin_, *args, **kwargs))
        log_norm = xp.logaddexp(logcdf_high, logcdf_low + xp.log(-1))
        # Guard against log(0) and -INF - (-INF) (degenerate truncation)
        return xp.where(log_norm > 0, logpdf_unnorm - log_norm, -INF)

    @classmethod
    def _get_minmax_quantiles(cls, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        """
        Returns the quantiles of xmin and xmax given loc and scale.
        """
        q_low = (
            0.0
            if xmin is -INF
            else xp.clip(cls._cdf(rescale(xmin, loc, scale), *args, **kwargs), 0.0, 1 - EPS)
        )
        q_high = (
            1.0
            if xmax is INF
            else xp.clip(cls._cdf(rescale(xmax, loc, scale), *args, **kwargs), EPS, 1.0)
        )
        return q_low, q_high

    @classmethod
    def cdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        _q_low, _q_high = cls._get_minmax_quantiles(
            *args, loc=loc, scale=scale, xmin=xmin, xmax=xmax, **kwargs
        )

        _q = cls._cdf(rescale(x, loc, scale), *args, **kwargs)
        return xp.clip((_q - _q_low) / (_q_high - _q_low), 0.0, 1.0)

    @classmethod
    def ppf(cls, q, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        _q_low, _q_high = cls._get_minmax_quantiles(
            *args, loc=loc, scale=scale, xmin=xmin, xmax=xmax, **kwargs
        )

        _q = xp.clip(q * (_q_high - _q_low) + _q_low, EPS, 1 - EPS)
        return cls._ppf(_q, *args, **kwargs) * scale + loc


def lnbetainc(a, b, x):
    x = xp.asarray(x, dtype=xp.float64)
    a = xp.asarray(x, dtype=xp.float64)
    b = xp.asarray(x, dtype=xp.float64)

    template = xp.empty(xp.broadcast_shapes(a.shape, b.shape, x.shape), dtype=xp.float64)

    small = x <= 10 * EPS
    big = x >= 1 - 10 * EPS

    # assuming x is always the last axis
    template[..., small] = a * xp.log(x[small]) - xp.log(a) - special.betaln(a, b)
    template[..., big] = xp.log1p(-special.betainc(b, a, 1 - x[big]))

    good = ~small & ~big
    template[..., good] = xp.log(special.betainc(a, b, x[good]))
    return template


class jf_skew_t(trunc_dist):
    r"""
    Jones and Faddy skew-t distribution with an alternative parameterization. See
    [1] and docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jf_skew_t.html
    for further details on the canonical parameterization.

    The probability density function in the canonical parameterization is given by

    .. math::
        f(x; a, b) = C_{a,b}^{-1}
                    \left(1+\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{a+1/2}
                    \left(1-\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{b+1/2}

    for real numbers :math:`a>0` and :math:`b>0`, where
    :math:`C_{a,b} = 2^{a+b-1}B(a,b)(a+b)^{1/2}`, and :math:`B` denotes the
    beta function (`scipy.special.beta`).

    We reparameterize this with an overall tail weight parameter :math:`\alpha`
    and skew parameter :math:`\log\kappa` such that

    .. math::
        \alpha =  a + b   \qquad   \kappa = \frac{a}{b}

    Thus, the distribution is positively skewed when :math:`\log\kappa > 0`
    and negatively skewed when :math:`\log\kappa < 0`.

    Then, we shift the distribution such that the mode is always at :math:`0`.
    The mode of the distribution is given by

    .. math::
        m = \frac{(a - b) \sqrt{a + b}}{\sqrt{(2a + 1) (2b + 1)}}

    Therefore, the pdf we use is given by

    .. math::
        p(x; \alpha, \log\kappa) = f(x + m; a(\alpha, \kappa), b(\alpha, \kappa))

    References
    ----------
    .. [1] M.C. Jones and M.J. Faddy. "A skew extension of the t distribution,
           with applications" *Journal of the Royal Statistical Society*.
           Series B (Statistical Methodology) 65, no. 1 (2003): 159-174.
           :doi:`10.1111/1467-9868.00378`

    %(example)s
    """

    @staticmethod
    def _reparam(logalpha, logkappa):
        kappa = xp.power(10, logkappa)
        alpha = xp.power(10, logalpha)
        a = alpha * kappa / (1.0 + kappa)
        b = alpha / (1.0 + kappa)
        return a, b

    @staticmethod
    def _reparam_and_shift(logalpha, logkappa):
        a, b = jf_skew_t._reparam(logalpha, logkappa)
        mode = (a - b) * xp.sqrt(a + b) / xp.sqrt((2 * a + 1) * (2 * b + 1))

        return a, b, mode

    @staticmethod
    def _pdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)

        c = 2 ** (a + b - 1) * special.beta(a, b) * xp.sqrt(a + b)

        # u = (x + shift) / sqrt(a + b + (x + shift)^2), and then
        # result = (1 + u)^(a + 0.5) * (1 - u)^(b + 0.5) / c.
        # Computed with in-place ops on locally-owned full-size buffers (the input
        # `x` is never mutated) to keep only ~2 full-size arrays alive at once --
        # this is the dominant memory term when evaluated on PE samples / injections.
        u = x + shift
        denom = u * u
        denom += a + b
        xp.sqrt(denom, out=denom)
        u /= denom  # u = (x + shift) / sqrt(...)
        del denom

        right = 1.0 - u
        right **= b + 0.5
        u += 1.0
        u **= a + 0.5
        u *= right
        u /= c

        # Guard against NaN from 0/0 or inf/inf when c is degenerate
        return xp.nan_to_num(u, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _logpdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)

        # Calculate the log of the normalization constant C
        # Use betaln for log(B(a,b)) to avoid underflow/overflow
        log_c = (a + b - 1) * xp.log(2) + special.betaln(a, b) + 0.5 * xp.log(a + b)

        # Pre-compute common terms
        # Clamp u to (-1+eps, 1-eps) to avoid log1p(-1)=-inf or log1p(1) issues at extreme x
        u_raw = (x + shift) / xp.sqrt(a + b + (x + shift) ** 2)
        u = xp.clip(u_raw, -1 + EPS, 1 - EPS)
        log_kernel = (a + 0.5) * special.log1p(u) + (b + 0.5) * special.log1p(-u)

        # Combine terms and adjust for the scale parameter
        return log_kernel - log_c

    @staticmethod
    def _moments(logalpha, logkappa, **kwargs):
        r"""
        Returns the mode and a characteristic width via the Gaussian curvature variance

        .. math::
            \sigma_z = \sqrt{ \frac{A(A+1)^4}{(4 a b+2 A+1)^3}
                              \left[ \frac{a+\frac{1}{2}}{(2 a+1)^2} +
                                     \frac{b+\frac{1}{2}}{(2 b+1)^2} \right]^{-1} }
        """
        a, b = jf_skew_t._reparam(logalpha, logkappa)

        # gaussian curvature variance
        A = a + b
        u_star = (a - b) / (A + 1.0)

        denom = (
            (1.0 - u_star**2) ** 3
            * (A + 1.0) ** 2
            * ((a + 0.5) / (2.0 * a + 1.0) ** 2 + (b + 0.5) / (2.0 * b + 1.0) ** 2)
        )
        var = A / denom

        return 0.0, xp.sqrt(var)  # mode at 0 by definition

    @staticmethod
    def _cdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)
        # In-place ops to keep only 2 full-size arrays live at once
        # (same pattern as the optimised _pdf above).
        y = x + shift          # owned buffer
        denom = y * y
        denom += a + b
        xp.sqrt(denom, out=denom)
        y /= denom             # y = (x+shift) / sqrt(a+b+(x+shift)^2)
        del denom
        y += 1.0
        y *= 0.5               # y = (1 + …) / 2
        return special.betainc(a, b, y)

    @staticmethod
    def _logcdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)
        y = x + shift
        denom = y * y
        denom += a + b
        xp.sqrt(denom, out=denom)
        y /= denom
        del denom
        y += 1.0
        y *= 0.5
        return lnbetainc(a, b, y)


class gaussian(trunc_dist):
    """
    Implements the Gaussian (Normal) distribution.
    """

    @staticmethod
    def _logpdf(x):
        return -0.5 * x**2 - 0.5 * xp.log(2 * xp.pi)

    @classmethod
    def _pdf(cls, x):
        return xp.exp(cls._logpdf(x))

    @staticmethod
    def _cdf(x):
        return special.ndtr(x)

    @staticmethod
    def _logcdf(x):
        return special.log_ndtr(x)

    @staticmethod
    def _ppf(q):
        return special.ndtri(q)
    
    @staticmethod
    def _moments(**kwargs):
        return 0, 1


class gen_gaussian(trunc_dist):
    """
    Implements the generalized Gaussian (Normal) distribution.
    Lifted from https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_continuous_distns.py#L11395-L11488
    """

    @staticmethod
    def _pdf(x, beta):
        res = beta / (2 * special.gamma(1.0 / beta)) * xp.exp(-(abs(x) ** beta))
        return xp.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _logpdf(x, beta):
        return xp.log(0.5 * beta) - special.gammaln(1.0 / beta) - abs(x) ** beta

    @staticmethod
    def _cdf(x, beta):
        c = 0.5 * xp.sign(x)
        # evaluating (.5 + c) first prevents numerical cancellation
        return (0.5 + c) - c * special.gammaincc(1.0 / beta, abs(x) ** beta)

    @staticmethod
    def _logcdf(x, beta):
        out = xp.full_like(x, -INF, dtype=xp.float64)
        log1p_term = xp.clip(special.gammainc(1.0 / beta, abs(x) ** beta), 0.0, 1.0 - EPS)
        out = xp.log(0.5) + xp.log1p(xp.sign(x) * log1p_term)
        return out

    @staticmethod
    def _moments(beta, **kwargs):
        return 0.0, xp.sqrt(special.gamma(3 / beta) / special.gamma(1 / beta))


class SGED(trunc_dist):
    r"""
    Standard form of the skewed Generalized Error Distribution (SGED).
    The probability density function is

    .. math::

        p(x \mid n, \kappa) =
        \frac{n}{(\lambda_L + \lambda_R)\,\Gamma(1/n)}
        \begin{cases}
        \exp\left[-\left(\frac{-x}{\lambda_L}\right)^n\right],
        & x < 0, \\
        \exp\left[-\left(\frac{x}{\lambda_R}\right)^n\right],
        & x \ge 0,
        \end{cases}

    where

    .. math::

        \lambda_L = \kappa, \qquad \lambda_R = \kappa^{-1}.

    Parameters
    ----------
    n : float
        Shape parameter (``n > 0``). Controls peakedness and tail thickness:
        ``n = 2`` gives a Gaussian core, ``n < 2`` produces sharper peaks and
        heavier tails, and ``n > 2`` yields flatter peaks with lighter tails.

    kappa : float
        Skew parameter (``kappa > 0``). Controls asymmetry:
        ``kappa = 1`` corresponds to the symmetric generalized Gaussian;
        ``kappa > 1`` gives a longer left tail; ``kappa < 1`` gives a longer
        right tail.
    """

    @staticmethod
    def _lam_from_logkappa(logkappa):
        lam_L = 10**logkappa
        lam_R = 10 ** (-logkappa)
        return lam_L, lam_R

    @staticmethod
    def _pdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        norm = n / ((lam_L + lam_R) * special.gamma(1.0 / n))

        zL = (-x / lam_L) ** n
        zR = (x / lam_R) ** n

        return norm * xp.where(x < 0, xp.exp(-zL), xp.exp(-zR))

    @staticmethod
    def _logpdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        lognorm = xp.log(n) - xp.log(lam_L + lam_R) - special.gammaln(1.0 / n)

        zL = (-x / lam_L) ** n
        zR = (x / lam_R) ** n

        return lognorm - xp.where(x < 0, zL, zR)

    @staticmethod
    def _cdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        wL = lam_L / (lam_L + lam_R)
        wR = lam_R / (lam_L + lam_R)

        tL = (-x / lam_L) ** n
        tR = (x / lam_R) ** n

        return xp.where(
            x < 0,
            wL * special.gammaincc(1.0 / n, tL),
            wL + wR * special.gammainc(1.0 / n, tR),
        )

    @staticmethod
    def _logcdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        wL = lam_L / (lam_L + lam_R)
        wR = lam_R / (lam_L + lam_R)

        tL = (-x / lam_L) ** n
        tR = (x / lam_R) ** n

        cdf = xp.where(
            x < 0,
            wL * special.gammaincc(1.0 / n, tL),
            wL + wR * special.gammainc(1.0 / n, tR),
        )

        return xp.log(xp.clip(cdf, EPS, 1.0))

    @staticmethod
    def _moments(n, logkappa):
        var = (
            (special.gamma(3.0 / n) / special.gamma(1.0 / n))
            * (10 ** (3 * logkappa) + 10 ** (-3 * logkappa))
            / (10**logkappa + 10 ** (-logkappa))
        )
        return 0.0, xp.sqrt(var)


class skew_gaussian(trunc_dist):
    r"""
    Skew-normal effective spin distribution: a skewed, ``[xmin, xmax]``-truncated
    Gaussian. The (un-truncated) density is

    .. math::

        \pi(x \mid \mu, \sigma, \epsilon) \propto
        \begin{cases}
        (1 + \epsilon)\, \mathcal{N}(x \mid \mu, \sigma(1 + \epsilon)) & x \le 0, \\
        (1 - \epsilon)\, \mathcal{N}(x \mid \mu, \sigma(1 - \epsilon)) & x \ge 0,
        \end{cases}

    where :math:`\mu` (``loc``) and :math:`\sigma` (``scale``) set the location
    and scale and :math:`\epsilon` (``epsilon``) is the skew parameter:
    :math:`\epsilon > 0` gives more support at :math:`x < \mu`, while
    :math:`\epsilon < 0` gives more support at :math:`x > \mu`. The distribution
    reduces to a standard, symmetric Gaussian when :math:`\epsilon = 0`.

    Because the piecewise split is at :math:`x = 0` (not at :math:`\mu`), this is
    not a location-scale family, so the ``loc`` / ``scale`` / truncation handling
    of ``trunc_dist`` is reimplemented here directly. Note that the :math:`1 \pm
    \epsilon` prefactor cancels the :math:`1 / (1 \pm \epsilon)` from the per-side
    scale, leaving an overall :math:`1 / \sigma`.

    See Banagiri et al. 2025 (Eq. B37).
    """

    @staticmethod
    def _side_scales(scale, epsilon):
        # left scale (x <= 0) and right scale (x >= 0)
        return scale * (1 + epsilon), scale * (1 - epsilon)

    @staticmethod
    def _base_logpdf(x, loc, scale, epsilon):
        # log of the un-truncated skew density g(x), with the (1 +/- eps)
        # prefactor folded in (it cancels the 1/(1 +/- eps) of the per-side scale).
        s_left, s_right = skew_gaussian._side_scales(scale, epsilon)
        s = xp.where(x <= 0, s_left, s_right)
        z = (x - loc) / s
        return -0.5 * z**2 - 0.5 * xp.log(2 * xp.pi) - xp.log(scale)

    @staticmethod
    def _base_cdf(x, loc, scale, epsilon):
        # G(x) = integral_{-inf}^{x} g(x') dx', the un-normalized base CDF.
        x = xp.asarray(x)
        s_left, s_right = skew_gaussian._side_scales(scale, epsilon)
        g0 = (1 + epsilon) * special.ndtr(-loc / s_left)  # base mass below x = 0
        left = (1 + epsilon) * special.ndtr((x - loc) / s_left)
        right = g0 + (1 - epsilon) * (
            special.ndtr((x - loc) / s_right) - special.ndtr(-loc / s_right)
        )
        return xp.where(x <= 0, left, right)

    # standard (loc=0, scale=1) form, kept to satisfy the dist/trunc_dist interface
    @staticmethod
    def _pdf(x, epsilon=0.0):
        return xp.exp(skew_gaussian._base_logpdf(x, 0.0, 1.0, epsilon))

    @staticmethod
    def _logpdf(x, epsilon=0.0):
        return skew_gaussian._base_logpdf(x, 0.0, 1.0, epsilon)

    @staticmethod
    def _cdf(x, epsilon=0.0):
        return skew_gaussian._base_cdf(x, 0.0, 1.0, epsilon)

    @classmethod
    def pdf(cls, x, epsilon=0.0, loc=0, scale=1, xmin=-INF, xmax=INF):
        norm = cls._base_cdf(xmax, loc, scale, epsilon) - cls._base_cdf(
            xmin, loc, scale, epsilon
        )
        pdf = xp.exp(cls._base_logpdf(x, loc, scale, epsilon))
        pdf *= (x >= xmin) & (x <= xmax)
        norm = xp.where(norm > 0, norm, xp.inf)  # guard degenerate truncation
        return pdf / norm

    @classmethod
    def logpdf(cls, x, epsilon=0.0, loc=0, scale=1, xmin=-INF, xmax=INF):
        norm = cls._base_cdf(xmax, loc, scale, epsilon) - cls._base_cdf(
            xmin, loc, scale, epsilon
        )
        logpdf = cls._base_logpdf(x, loc, scale, epsilon) - xp.log(norm)
        logpdf = xp.where((x < xmin) | (x > xmax), -INF, logpdf)
        return xp.where(norm > 0, logpdf, -INF)

    @classmethod
    def cdf(cls, x, epsilon=0.0, loc=0, scale=1, xmin=-INF, xmax=INF):
        G_lo = cls._base_cdf(xmin, loc, scale, epsilon)
        G_hi = cls._base_cdf(xmax, loc, scale, epsilon)
        norm = xp.where(G_hi - G_lo > 0, G_hi - G_lo, xp.inf)
        return xp.clip((cls._base_cdf(x, loc, scale, epsilon) - G_lo) / norm, 0.0, 1.0)

    @classmethod
    def ppf(cls, q, epsilon=0.0, loc=0, scale=1, xmin=-INF, xmax=INF):
        s_left, s_right = cls._side_scales(scale, epsilon)
        G_lo = cls._base_cdf(xmin, loc, scale, epsilon)
        G_hi = cls._base_cdf(xmax, loc, scale, epsilon)
        # invert the un-normalized base CDF at the rescaled quantile target
        t = G_lo + xp.clip(xp.asarray(q), EPS, 1 - EPS) * (G_hi - G_lo)
        t = xp.asarray(t)
        g0 = (1 + epsilon) * special.ndtr(-loc / s_left)
        left = loc + s_left * special.ndtri(xp.clip(t / (1 + epsilon), EPS, 1 - EPS))
        right_q = special.ndtr(-loc / s_right) + (t - g0) / (1 - epsilon)
        right = loc + s_right * special.ndtri(xp.clip(right_q, EPS, 1 - EPS))
        return xp.where(t <= g0, left, right)

    @staticmethod
    def _moments(*args, **kwargs):
        return 0, 1


class powerlaw:

    @staticmethod
    def pdf(x, beta, xmin, xmax):
        r"""
        This is lifted directly from https://github.com/ColmTalbot/gwpopulation/blob/main/gwpopulation/utils.py

        Power-law probability

        .. math::
            p(x) = \frac{1 + \beta}{x_\max^{1 + \beta} - x_\min^{1 + \beta}} x^\beta

        Note the sign change from the smoothed power law, which is defined as x^{-\alpha}.

        Parameters
        ----------
        x: float, array-like
            The abscissa values (:math:`x`)
        beta: float, array-like
            The spectral index of the distribution (:math:`\beta`)
        xmin: float, array-like
            The minimum of the distribution (:math:`x_\min`)
        xmax: float, array-like
            The maximum of the distribution (:math:`x_\max`)

        Returns
        -------
        prob: float, array-like
            The distribution evaluated at `xx`

        """
        log_ratio = xp.log(xmax / xmin)
        beta = xp.atleast_1d(beta)

        # Compute normalization without materializing beta_, xmin_, xmax_
        norm_beta_ne1 = (1.0 + beta) / (xmax ** (1.0 + beta) - xmin ** (1.0 + beta))
        norm = xp.where(beta == -1.0, 1.0 / log_ratio, norm_beta_ne1)

        prob = xp.power(x, beta)
        prob *= norm
        prob *= (x <= xmax) & (x >= xmin)

        return prob

    @staticmethod
    def cdf(x, beta, xmin, xmax):
        log_ratio = xp.log(xmax / xmin)
        beta = xp.atleast_1d(beta)
        a = 1.0 + beta
        x_clipped = xp.clip(x, xmin, xmax)
        cdf_ne1 = (xp.power(x_clipped, a) - xp.power(xmin, a)) / (
            xp.power(xmax, a) - xp.power(xmin, a)
        )
        cdf_beta_m1 = xp.log(x_clipped / xmin) / log_ratio
        cdf_inner = xp.where(beta == -1.0, cdf_beta_m1, cdf_ne1)
        return xp.where(x <= xmin, 0.0, xp.where(x >= xmax, 1.0, cdf_inner))

    @staticmethod
    def ppf(q, beta, xmin, xmax):
        a = 1.0 + beta
        return xp.where(
            beta == -1.0,
            xmin * (xmax / xmin) ** q,
            xp.power((xmax**a - xmin**a) * q + xmin**a, 1 / a),
        )

    @classmethod
    def moments(cls, beta, xmin, xmax):
        kwargs = dict(beta=beta, xmin=xmin, xmax=xmax)
        med = cls.ppf(0.5, **kwargs)
        width = med - cls.ppf(0.16, **kwargs)
        return med, width


class smoothed_powerlaw:
    r"""
    Implements the power-law distribution with minimum, maximum, and lower-end smoothing.
    This is a different smoothing function from LVK s.t. the smoothing function is integrable.

    .. math::

        f(x; \alpha, x_{\min}, x_{\max}, p) = (1/C) x^{-\alpha} \left(1 - \frac{x_{\min}}{x} \right)^p

        C = x_{\min}^{1-\alpha} I_{1-x_{\min}/x_{\max}}(p+1, \alpha-1)

    where I_u(a, b) is the incomplete beta function. Note that :math:`\alpha > 1`, since
    computing the normalization for :math:`\alpha <= 1` involves hypergeometric functions that are
    not currently implemented by cupy.
    """

    @staticmethod
    def pdf(x, alpha, xmin, xmax, p):

        x = xp.asarray(x)

        # Incomplete beta function
        u = 1 - xmin / xmax
        a = p + 1
        b = alpha - 1

        C = xp.power(xmin, 1 - alpha) * special.betainc(a, b, u) * special.beta(a, b)
        # need to multiply regularized incomplete beta by beta(a,b) to get the incomplete beta function

        return xp.where(
            (x >= xmin) & (x <= xmax) & (C > 0.0) & (alpha >= 1.0),
            (1 / C) * xp.power(x, -alpha) * xp.power(1.0 - xmin / x, p),
            0.0,
        )

    @staticmethod
    def cdf(x, alpha, xmin, xmax, p):

        x = xp.asarray(x)

        a = p + 1
        b = alpha - 1
        u_max = 1 - xmin / xmax

        # regularized incomplete beta at xmax
        I_max = special.betainc(a, b, u_max)

        # argument of incomplete beta
        u = 1 - xmin / x

        F = special.betainc(a, b, u) / I_max

        return xp.where(x < xmin, 0.0, xp.where(x > xmax, 1.0, F))

    @staticmethod
    def ppf(q, alpha, xmin, xmax, p):

        q = xp.asarray(q)

        a = p + 1
        b = alpha - 1
        u_max = 1 - xmin / xmax

        # regularized incomplete beta at xmax
        I_max = special.betainc(a, b, u_max)

        # inverse regularized incomplete beta
        u = special.betaincinv(a, b, q * I_max)

        x = xmin / (1.0 - u)

        return xp.clip(x, xmin, xmax)

    @staticmethod
    def logpdf(x, alpha, xmin, xmax, p):

        x = xp.asarray(x)

        u = 1 - xmin / xmax
        a = p + 1
        b = alpha - 1

        lnIu = lnbetainc(a, b, u) + special.betaln(a, b)  # stable for large/small params
        lnC = (1 - alpha) * xp.log(xmin) + lnIu

        # Use safe ratio to avoid log(0)
        ratio = xp.clip(xmin / x, 0, 1 - EPS)

        # Main terms
        return xp.where(
            (x >= xmin) & (x <= xmax), -alpha * xp.log(x) + p * xp.log1p(-ratio) - lnC, -INF
        )

    @classmethod
    def moments(cls, alpha, xmin, xmax, p):
        # use mode for first moment, quantiles for width (approximated w/o low end smoothing)
        mode = xmin * (1 + p / alpha)
        width = (powerlaw.ppf(0.5, -alpha, xmin, xmax) - mode) / 2
        return mode, width


def sigmoid_smooth(x, xmin, delta):
    eps = delta / xp.log(INF)  # for numerical stability
    x_safe = xp.where(
        x < xmin + eps, xmin + eps, xp.where(x > xmin + delta - eps, xmin + delta - eps, x)
    )
    return 1.0 / (1.0 + xp.exp(delta / (x_safe - xmin) + delta / (x_safe - xmin - delta)))


def log_sigmoid_smooth(x, xmin, delta):
    """
    Sigmoid low-mass tapering function. See
    https://arxiv.org/pdf/2111.03634 Eqs. (B5), (B6)
    """
    eps = delta / xp.log(INF)  # for numerical stability
    x_safe = xp.where(
        x < xmin + eps, xmin + eps, xp.where(x > xmin + delta - eps, xmin + delta - eps, x)
    )
    return -xp.log1p(xp.exp(delta / (x_safe - xmin) + delta / (x_safe - xmin - delta)))


class LVK_Plancktaper_powerlaw(dist):
    """
    Implements the power-law distribution with Planck-tapered lower end, as used by the LVK.
    """

    var_names = ["alpha", "xmin", "xmax", "delta"]

    @staticmethod
    def _unnorm_pdf(x, alpha, xmin, xmax, delta):
        return xp.where(
            (x > xmin) & (x < xmax), xp.power(x, -alpha) * sigmoid_smooth(x, xmin, delta), 0
        )

    @staticmethod
    def _unnorm_logpdf(x, alpha, xmin, xmax, delta):
        return xp.where(
            (x > xmin) & (x < xmax), -alpha * xp.log(x) + log_sigmoid_smooth(x, xmin, delta), -INF
        )

    @classmethod
    def _pdf(cls, x, alpha, xmin, xmax, delta, xx_int=None):
        pdf_unnorm = cls._unnorm_pdf(x, alpha, xmin, xmax, delta)

        # compute normalization - assumes x is the last dimension
        if xx_int is None:
            xx_int = xp.linspace(xp.amin(xp.asarray(xmin)), xp.amax(xp.asarray(xmax)), 256)

        norm = xp.trapezoid(cls._unnorm_pdf(xx_int, alpha, xmin, xmax, delta), xx_int, axis=-1)

        if xp.asarray(norm).ndim > 0:
            norm = norm[:, None]

        return pdf_unnorm / norm

    @classmethod
    def _logpdf(cls, x, alpha, xmin, xmax, delta, xx_int=None):
        logpdf_unnorm = cls._unnorm_logpdf(x, alpha, xmin, xmax, delta)

        # compute normalization - assumes x is the last dimension
        if xx_int is None:
            xx_int = xp.linspace(xp.amin(xp.asarray(xmin)), xp.amax(xp.asarray(xmax)), 256)

        norm = xp.trapezoid(cls._unnorm_pdf(xx_int, alpha, xmin, xmax, delta), xx_int, axis=-1)[
            :, None
        ]
        return logpdf_unnorm - xp.log(norm)


_mm = xp.linspace(2, 300, 1024)
_dm = _mm[1] - _mm[0]


class MassModel:
    @abstractmethod
    def pdf(self, data, **kwargs):
        """
        For consistency (especially with PE priors), always return p(m1, q) regardless of model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_marginal_pdf(self, param_vals, param, *args, **kwargs):
        """
        Return marginalized p(param). Allowed values of param: 'mass_1_source', 'mass_ratio', 'mass_2_source'
        """
        raise NotImplementedError

    @abstractmethod
    def moments(self, *args, **kwargs) -> dict:
        """
        Return moments of the secondary distributions, in dictionary format.
        """
        raise NotImplementedError


class m1_q_model(MassModel):
    def __init__(self, mass_1_source_model: dist, mass_ratio_model: dist):
        self.m1_model = mass_1_source_model
        self.q_model = mass_ratio_model

    params = ("mass_1_source", "mass_ratio")

    def _p_q_given_m1(self, data, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs):
        mass_ratio_kwargs = mass_ratio_kwargs.copy()
        if "xmax" not in mass_ratio_kwargs:
            mass_ratio_kwargs["xmax"] = 1.0
        return self.q_model.pdf(
            data["mass_ratio"],
            xmin=mass_1_source_kwargs["xmin"] / data["mass_1_source"],
            **mass_ratio_kwargs,
        )

    def _p_q(self, q, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs):
        qq, mm1 = xp.meshgrid(q, _mm, indexing="ij", copy=False)
        data_flat = {"mass_ratio": qq.ravel(), "mass_1_source": mm1.ravel()}

        p_m_q = self.pdf(
            data_flat,
            mass_1_source_kwargs=mass_1_source_kwargs,
            mass_ratio_kwargs=mass_ratio_kwargs,
            **kwargs,
        )
        p_m_q = p_m_q.reshape(p_m_q.shape[:-1] + (len(q), len(_mm)))

        return xp.sum(p_m_q, axis=-1) * _dm

    def _p_m2(self, m2, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs):
        mm2, mm1 = xp.meshgrid(m2, _mm, indexing="ij")
        qq = mm2 / mm1
        data_flat = {"mass_ratio": qq.ravel(), "mass_1_source": mm1.ravel()}

        p_m_q = self.pdf(
            data_flat,
            mass_1_source_kwargs=mass_1_source_kwargs,
            mass_ratio_kwargs=mass_ratio_kwargs,
            **kwargs,
        )
        p_m1_m2 = p_m_q.reshape(p_m_q.shape[:-1] + (len(m2), len(_mm))) / _mm
        return xp.sum(p_m1_m2, axis=-1) * _dm

    def get_marginal_pdf(
        self, param_vals, param, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs
    ):
        if param == "mass_1_source":
            return self.m1_model.pdf(param_vals, **mass_1_source_kwargs)
        elif param == "mass_ratio":
            return self._p_q(param_vals, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs)
        elif param == "mass_2_source":
            return self._p_m2(param_vals, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs)
        else:
            raise ValueError(f"Unknown parameter {param}")

    def pdf(self, data, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs):
        # p(m1) * p(q | m1), multiplied in place to avoid an extra full-size copy
        p = self.m1_model.pdf(data["mass_1_source"], **mass_1_source_kwargs)
        p *= self._p_q_given_m1(data, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs)
        return p

    def moments(self, mass_1_source_kwargs, mass_ratio_kwargs, **kwargs):
        mass_ratio_kwargs = mass_ratio_kwargs.copy()
        mass_ratio_kwargs["xmin"] = 0.0
        mass_ratio_kwargs["xmax"] = 1.0
        m1_mu, m1_std = self.m1_model.moments(**mass_1_source_kwargs)
        q_mu, q_std = self.q_model.moments(**mass_ratio_kwargs)
        return {"mass_1_source": (m1_mu, m1_std), "mass_ratio": (q_mu, q_std)}


class m1_q_m2max_model(m1_q_model):
    def _p_q_given_m1(self, data, mass_1_source_kwargs, mass_ratio_kwargs, m2max):
        mass_ratio_kwargs = mass_ratio_kwargs.copy()
        mass_ratio_kwargs["xmax"] = xp.minimum(
            mass_ratio_kwargs.get("xmax", 1.0), m2max / data["mass_1_source"]
        )
        return self.q_model.pdf(
            data["mass_ratio"],
            xmin=mass_1_source_kwargs["xmin"] / data["mass_1_source"],
            **mass_ratio_kwargs,
        )


# class Mc_q_model(MassModel):

#     def __init__(self, Mc_model: dist, q_model: dist):
#         self.Mc_model = Mc_model
#         self.q_model = q_model


def gaussian_copula(u, v, rho):
    r"""
    Implements the Gaussian copula

    .. math::
        c_\rho(u, v) = \frac{1}{\sqrt{1-\rho^2}}
                    \exp \left(\frac{2 \rho z_1 z_2-\rho^2 (z_1^2+z_2^2)}{2(1-\rho^2)}\right)

    where :math: `z_i=\Phi^{-1}(u_i)`.

    Parameters
    ----------
    rho : float
        Correlation parameter.
    u, v : array-like
        The CDF values of the marginals.

    Returns
    -------
    array-like
        Copula values.
    """

    z1 = xp.clip(special.ndtri(u), -INF, INF)
    z2 = xp.clip(special.ndtri(v), -INF, INF)
    return xp.exp((2 * rho * z1 * z2 - rho**2 * (z1**2 + z2**2)) / (2 * (1.0 - rho**2))) / xp.sqrt(
        1.0 - rho**2
    )


def _check_gaussian_copula_params(m1, m2, rho):
    return (m1 >= m2) * (rho > -1) * (rho < 1)


class gaussian_copula_mass_model(MassModel):
    def __init__(self, mass_1_source_model: dist, mass_2_source_model: dist):
        self.m1_model = mass_1_source_model
        self.m2_model = mass_2_source_model

    params = ("mass_1_source", "mass_2_source")

    _zz = xp.linspace(-5, 5, 512)
    _dz = _zz[1] - _zz[0]

    def _compute_norm(self, rho, mass_1_source_kwargs, mass_2_source_kwargs):
        r"""
        Computes P(m2 <= m1), which is the normalization of the joint pdf.
        For the Gaussian copula, this is the integral

        .. math::
            P (m_2 < m_1) = \int_{-\infty}^{\infty} \phi\left(z_1\right)
            \Phi\left(\frac{\Phi^{-1}\left(F_2\left(F_1^{-1}\left(\Phi\left(z_1\right)\right)\right)\right)-\rho z_1}{\sqrt{1-\rho^2}}\right) \, dz_1
        """

        # m1 = F1^{-1}(Phi(z1))
        m1 = self.m1_model.ppf(special.ndtr(self._zz), **mass_1_source_kwargs)

        # threshold
        m2_q = self.m2_model.cdf(m1, **mass_2_source_kwargs)
        z2_thr = special.ndtri(m2_q)

        # conditional CDF
        arg = (z2_thr - rho * self._zz) / xp.sqrt(1 - rho**2)
        integrand = gaussian._pdf(self._zz) * xp.where(
            xp.isfinite(arg),
            special.ndtr(arg),
            0.0,  # outside region of integration
        )

        return xp.sum(integrand) * self._dz

    def pdf(self, data, rho, mass_1_source_kwargs, mass_2_source_kwargs):

        m1, m2 = data["mass_1_source"], data["mass_2_source"]

        u = self.m1_model.cdf(m1, **mass_1_source_kwargs)
        v = self.m2_model.cdf(m2, **mass_2_source_kwargs)
        norm = self._compute_norm(rho, mass_1_source_kwargs, mass_2_source_kwargs)

        jacobian = m1  # P(m1, m2) -> P(m1, q)
        m1_pdf = self.m1_model.pdf(m1, **mass_1_source_kwargs)
        m2_pdf = self.m2_model.pdf(m2, **mass_2_source_kwargs)
        copula_vals = gaussian_copula(u, v, rho)

        res = (
            m1_pdf
            * m2_pdf
            * copula_vals
            / norm
            * jacobian
            * _check_gaussian_copula_params(m1, m2, rho)
        )

        return xp.nan_to_num(res, copy=False, nan=0, posinf=0, neginf=0)

    def _p_q(self, q, mass_1_source_kwargs, mass_2_source_kwargs, rho):
        qq, mm1 = xp.meshgrid(q, _mm, indexing="ij", copy=False)
        mm2 = mm1 * qq
        data_flat = {"mass_1_source": mm1.ravel(), "mass_2_source": mm2.ravel()}

        p_m1_q = self.pdf(data_flat, rho, mass_1_source_kwargs, mass_2_source_kwargs)
        p_m1_q = p_m1_q.reshape(p_m1_q.shape[:-1] + (len(q), len(_mm)))
        return xp.nansum(p_m1_q, axis=-1) * _dm

    def get_marginal_pdf(self, param_vals, param, rho, mass_1_source_kwargs, mass_2_source_kwargs):
        if param == "mass_1_source":
            return self.m1_model.pdf(param_vals, **mass_1_source_kwargs)
        elif param == "mass_ratio":
            return self._p_q(param_vals, mass_1_source_kwargs, mass_2_source_kwargs, rho)
        elif param == "mass_2_source":
            return self.m2_model.pdf(param_vals, **mass_2_source_kwargs)
        else:
            raise ValueError(f"Unknown parameter {param}")

    def moments(self, mass_1_source_kwargs, mass_2_source_kwargs, rho):
        m1_mu, m1_std = self.m1_model.moments(**mass_1_source_kwargs)
        m2_mu, m2_std = self.m2_model.moments(**mass_2_source_kwargs)
        return {"mass_1_source": (m1_mu, m1_std), "mass_2_source": (m2_mu, m2_std)}


class sym_gaussian_copula_mass_model(MassModel):
    r"""
    Gaussian copula mass model, but assumes that all masses are drawn from the same
    distribution X ~ f(m) with some pairing function (copula). From here we derive
    the expressions for the distributions of :math:`m_1 = \max(X_1, X_2)` and
    :math:`m_2 = \min(X_1, X_2)`.

    It can be shown that

    .. math::
        p(m_1 \mid \rho) &= 2 f(m_1) \Phi\left(\sqrt{\frac{1-\rho}{1+\rho}}\,\Phi^{-1}(F(m_1))\right) \\
        p(m_2 \mid \rho) &= 2 f(m_2) \Phi\left(-\sqrt{\frac{1-\rho}{1+\rho}}\,\Phi^{-1}(F(m_2))\right)

    In these functions `mass_1_source_model` and `mass_1_source_kwargs` refers to the
    model and kwargs of the shared mass model X.
    """

    def __init__(self, mass_1_source_model: dist):
        self.m_model = mass_1_source_model

    params = ("mass_1_source", "mass_2_source")

    def pdf(self, data, rho, mass_1_source_kwargs):
        r"""
        Evaluates the joint density

        .. math::
            p(m_1, m_2 \mid \rho) = 2 p(m_1) p(m_2) c_\rho(F(m_1), F(m_2)) \qquad m_2 < m_1,

        where :math:`c_\rho(u,v)` is the Gaussian copula density.
        """

        m1, m2 = data["mass_1_source"], data["mass_2_source"]
        factor = xp.sqrt((1.0 - rho) / (1.0 + rho))  # per-leaf scalar broadcast

        # --- z1 = Phi^{-1}(F(m1)),  z2 = Phi^{-1}(F(m2)) ---
        u = self.m_model.cdf(m1, **mass_1_source_kwargs)
        z1 = special.ndtri(u)
        del u
        v = self.m_model.cdf(m2, **mass_1_source_kwargs)
        z2 = special.ndtri(v)
        del v

        # --- result = p_m1 built in-place, then *= p_m2 ---
        result = self.m_model.pdf(m1, **mass_1_source_kwargs)  # owned buffer
        ndtr_fz1 = special.ndtr(factor * z1)
        result *= 2.0 * ndtr_fz1
        del ndtr_fz1

        pdf_m2 = self.m_model.pdf(m2, **mass_1_source_kwargs)
        ndtr_fz2 = special.ndtr(-(factor * z2))
        result *= 2.0 * pdf_m2 * ndtr_fz2
        del pdf_m2, ndtr_fz2

        # --- copula = exp(num/denom) / sqrt(1-rho^2) built in-place ---
        # num = 2*rho*z1*z2 - rho^2*(z1^2+z2^2),  denom = 2*(1-rho^2)
        # After this block z1 and z2 are no longer needed.
        rho2 = rho ** 2
        copula = z1 * z2                   # new buffer
        copula *= 2.0 * rho                # 2*rho*z1*z2  (in-place)
        sq_sum = z1 * z1
        sq_sum += z2 * z2                  # z1^2 + z2^2  (one temp for z2*z2)
        del z1, z2
        sq_sum *= rho2
        copula -= sq_sum                   # numerator   (in-place)
        del sq_sum
        copula /= 2.0 * (1.0 - rho2)      # divide by denominator (in-place)
        xp.exp(copula, out=copula)         # exp in-place
        copula /= xp.sqrt(1.0 - rho2)     # normalise copula (in-place, rho2 is tiny)

        result *= copula
        del copula

        # jacobian P(m1,m2)->P(m1,q) = m1; validity mask zeros out m2>m1 or |rho|>=1
        result *= 2.0 * m1 * _check_gaussian_copula_params(m1, m2, rho)
        return xp.nan_to_num(result, copy=False, nan=0, posinf=0, neginf=0)

    def _p_m1(self, m1, rho, mass_1_source_kwargs):
        r"""
        Evaluates the marginal distribution

        .. math::
            p(m_1) = 2 f(m_1) \Phi\ \left(\sqrt{\frac{1-\rho}{1+\rho}}\,\Phi^{-1}(F(m_1))\right)
        """
        arg = xp.sqrt((1.0 - rho) / (1.0 + rho)) * special.ndtri(
            self.m_model.cdf(m1, **mass_1_source_kwargs)
        )
        return 2 * self.m_model.pdf(m1, **mass_1_source_kwargs) * special.ndtr(arg)

    def _p_m2(self, m2, rho, mass_1_source_kwargs):
        r"""
        Evaluates the marginal distribution

        .. math::
            p(m_2) = 2 f(m_2) \Phi\ \left(-\sqrt{\frac{1-\rho}{1+\rho}}\,\Phi^{-1}(F(m_2))\right)
        """
        arg = xp.sqrt((1.0 - rho) / (1.0 + rho)) * special.ndtri(
            self.m_model.cdf(m2, **mass_1_source_kwargs)
        )
        return 2 * self.m_model.pdf(m2, **mass_1_source_kwargs) * special.ndtr(-arg)

    def _p_q(self, q, rho, mass_1_source_kwargs):
        qq, mm1 = xp.meshgrid(q, _mm, indexing="ij", copy=False)
        mm2 = mm1 * qq
        data_flat = {"mass_1_source": mm1.ravel(), "mass_2_source": mm2.ravel()}

        p_m1_q = self.pdf(data_flat, rho, mass_1_source_kwargs)
        p_m1_q = p_m1_q.reshape(p_m1_q.shape[:-1] + (len(q), len(_mm)))
        return xp.nansum(p_m1_q, axis=-1) * _dm

    def get_marginal_pdf(self, param_vals, param, rho, mass_1_source_kwargs):
        if param == "mass_1_source":
            return self._p_m1(param_vals, rho, mass_1_source_kwargs)
        elif param == "mass_ratio":
            return self._p_q(param_vals, rho, mass_1_source_kwargs)
        elif param == "mass_2_source":
            return self._p_m2(param_vals, rho, mass_1_source_kwargs)
        else:
            raise ValueError(f"Unknown parameter {param}")

    def moments(self, mass_1_source_kwargs, rho):
        m_mu, m_std = self.m_model.moments(**mass_1_source_kwargs)
        return {"mass_1_source": (m_mu, m_std)}


###################
# Rate models
###################


class BaseRate:
    def __init__(self, cosmo=Planck15):
        self.cosmo = cosmo

        z_arr = np.concatenate([np.linspace(EPS, 3, 256, endpoint=False), np.linspace(3, 10, 50)])
        dV_dz_arr = (
            4 * np.pi * self.cosmo.differential_comoving_volume(z_arr).value / 1e9
        )  # in Gpc^3
        self._z_arr = xp.asarray(z_arr)
        self._dV_dz_arr = xp.asarray(dV_dz_arr)
        self._log_dV_dz_arr = xp.log(self._dV_dz_arr)

    def dV_dz(self, z):
        return xp.interp(z, self._z_arr, self._dV_dz_arr, left=0, right=0)

    def log_dV_dz(self, z):
        return xp.interp(z, self._z_arr, self._log_dV_dz_arr, left=-INF, right=-INF)

    @abstractmethod
    def psi(self, z, *args, **kwargs):
        """
        Rate model, normalized such that psi(0)=1.
        """
        raise NotImplementedError

    @abstractmethod
    def log_psi(self, z, *args, **kwargs):
        """
        Log rate model, normalized such that log_psi(0)=0.
        """
        raise NotImplementedError

    def pdf(self, z, *args, zmax=ZMAX, **kwargs):
        return self.psi(z, *args, **kwargs) / (1.0 + z) * self.dV_dz(z) * (z < zmax)

    def logpdf(self, z, *args, zmax=ZMAX, **kwargs):
        return xp.where(
            z < zmax,
            self.log_psi(z, *args, **kwargs) - xp.log1p(z) + self.log_dV_dz(z),
            -INF,
        )

    def Ntot(self, *args, R0=1, zmax=ZMAX, **kwargs):
        """
        Calculates the total number of expected mergers in the spacetime volume (up
        to zmax).
        """
        # need to pad to broadcast with z arr integral
        if args:
            args = [utils.pad(x) for x in args]
        if kwargs:
            kwargs = utils.recursive_pad(kwargs)

        z_arr_filter = self._z_arr < zmax
        zz = self._z_arr[z_arr_filter]
        Vzz = self._dV_dz_arr[z_arr_filter]

        psizz = xp.nan_to_num(self.psi(zz, *args, **kwargs), nan=0, posinf=0, neginf=0)

        return xp.trapezoid(Vzz * psizz / (1.0 + zz), zz) * R0


class PL_rate(BaseRate):

    @staticmethod
    def psi(z, gamma):
        """
        Power-law rate model.
        """
        return (1.0 + z) ** gamma

    @staticmethod
    def log_psi(z, gamma):
        return gamma * xp.log1p(z)

    @staticmethod
    def moments(gamma):
        # Just use the power law exponent to differentiate models
        return (gamma,)


class MD_rate(BaseRate):
    @staticmethod
    def psi(z, gamma, kappa, zp):
        """
        Madau-Dickinson rate model. See e.g.
        https://arxiv.org/abs/2003.12152 Eq. (2).
        """
        C = 1 + (1 + zp) ** (-(gamma + kappa))
        return C * (1.0 + z) ** gamma / (1.0 + ((1.0 + z) / (1.0 + zp)) ** (gamma + kappa))

    @staticmethod
    def log_psi_MD(z, gamma, kappa, zp):
        logC = xp.log1p(xp.power(1 + zp, -(gamma + kappa)))
        return gamma * xp.log1p(z) - xp.log1p(((1.0 + z) / (1.0 + zp)) ** (gamma + kappa)) + logC


### HARD CODED MODEL DEFINITIONS

# dictionary of models and their hyperparameters -- these are manually hard coded in
XMAX_FIX = 300.0
MODELS = {
    "mass": {
        "param_latex": {"m2max": r"$m_{2,\max}$"},
        "m1_q": {"model": m1_q_model, "param_latex": {}},
        "m1_q_m2max": {"model": m1_q_m2max_model, "param_latex": {"m2max": r"$m_{2,\max}$"}},
        "gaussian_copula": {
            "model": gaussian_copula_mass_model,
            "param_latex": {"rho": r"$\rho$"},
        },
        "sym_gaussian_copula": {
            "model": sym_gaussian_copula_mass_model,
            "param_latex": {"rho": r"$\rho$"},
            "param_desc": {"rho": "correlation coefficient between m1 and m2"},
        },
    },
    "mass_1_source": {  # for p(m1) or p(m2)
        "param_latex": {  # shared between all models of the parameter
            "xmin": r"$m_{\min}$",
            "xmax": r"$m_{\max}$",
        },
        "param_desc": {
            "xmin": "primary mass minimum",
            "xmax": "primary mass maximum"
        },
        "param_unit": {
            "loc": "solMass",
            "scale": "solMass",
            "delta": "solMass",
            "xmin": "solMass",
            "xmax": "solMass"
        },
        "skew-t": {
            "model": jf_skew_t(),
            "param_latex": {
                "logalpha": r"$\log\alpha$",
                "logkappa": r"$\log\kappa$",
                "loc": r"$\mu_m$",
                "scale": r"$\sigma_m$",
            },
            "param_desc": {
                "logalpha": "m1 skew-t tail weight parameter",
                "logkappa": "m1 skew-t skew parameter",
                "loc": "m1 skew-t location parameter, determines the mode of the skew-t distribution",
                "scale": "m1 skew-t scale parameter",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
        "PLS": {
            "model": smoothed_powerlaw(),
            "param_latex": {
                "alpha": r"$\alpha$",
                "p": r"$p_m$",
            },
            "param_desc": {
                "alpha": "m1 power-law slope (must exceed 1 for normalization)",
                "p": "smoothing exponent governing the low-mass taper of p(m1)",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
        "PLS_LVK": {
            "model": LVK_Plancktaper_powerlaw(),
            "param_latex": {
                "alpha": r"$\alpha$",
                "delta": r"$\delta_m$",
            },
            "param_desc": {
                "alpha": "primary mass power-law exponent",
                "delta": "p(m1) Planck taper width controlling the low-end smoothing",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
        "gauss": {
            "model": gaussian(),
            "param_latex": {
                "loc": r"$\mu_p$",
                "scale": r"$\sigma_p$",
            },
            "param_desc": {
                "loc": "m1 Gaussian mean",
                "scale": "m1 Gaussian standard deviation",
            },
            "params_fix": {"xmax": XMAX_FIX},
        },
    },
    "mass_ratio": {
        "param_desc": {
            "xmin": "minimum mass ratio",
            "xmax": "maximum mass ratio",
        },
        "PL": {
            "model": powerlaw(),
            "param_latex": {
                "beta": r"$\beta$",
            },
            "param_desc": {
                "beta": "mass ratio power-law index",
            },
            "params_fix": {
                "xmax": 1.0,
            },
        },
        "gauss": {
            "model": gaussian(),
            "param_latex": {
                "loc": r"$\mu_{q}$",
                "scale": r"$\sigma_{q}$",
            },
            "param_desc": {
                "loc": "mass ratio Gaussian mean",
                "scale": "mass ratio Gaussian standard deviation",
            },
            "params_fix": {"xmax": 1.0},
        },
    },
    "chi_eff": {
        "param_latex": {
            "loc": r"$\mu_{\chi}$",
            "scale": r"$\sigma_{\chi}$",
        },
        "param_desc": {
            "xmin": "minimum chi_eff",
            "xmax": "maximum chi_eff",
            "loc": "chi_eff Gaussian mean",
            "scale": "chi_eff Gaussian standard deviation"
        },
        "gen_gauss": {
            "model": gen_gaussian(),
            "param_latex": {
                "beta": r"$\beta_{\chi}$"
            },
            "param_desc": {
                "beta": "chi_eff generalized Gaussian shape parameter"
            },
            "params_fix": {"xmin": -1.0, "xmax": 1.0},
        },
        "gauss": {
            "model": gaussian(),
            "params_fix": {"xmin": -1.0, "xmax": 1.0},
        },
        "skew_gauss": {
            "model": skew_gaussian(),
            "param_latex": {
                "epsilon": r"$\epsilon_{\chi}$",
            },
            "param_desc": {
                "epsilon": "chi_eff skew parameter (>0 favors chi_eff < mean)",
            },
            "params_fix": {"xmin": -1.0, "xmax": 1.0},
        },
    },
    "redshift": {
        "MD": {
            "model": MD_rate(cosmo=Planck15),
            "param_latex": {
                "gamma": r"$\gamma$",
                "kappa": r"$\kappa$",
                "zp": r"$z_p$",
            },
            "param_desc": {
                "gamma": "Madau-Dickinson low-z power-law exponent",
                "kappa": "Madau-Dickinson shape parameter",
                "zp": "Madau-Dickinson pivot redshift",
            },
            "params_fix": {},
        },
        "PL": {
            "model": PL_rate(cosmo=Planck15),
            "param_latex": {"gamma": r"$\gamma$"},
            "param_desc": {"gamma": "redshift rate power-law exponent"},
            "params_fix": {},
        },
    },
}
PARAM_SCALES = {  # characteristic scales for each parameter
    "mass_1_source": 8.0,
    "mass_2_source": 8.0,
    "mass_ratio": 0.3,
    "chi_eff": 0.1,
    "redshift": 1.0,
    "rate": 10.0,
}
RATE_FACTOR = 1.0
