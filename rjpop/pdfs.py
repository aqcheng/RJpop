import numpy as np
from abc import ABC, abstractmethod

from astropy.cosmology import Planck15

try:
    from cupyx.scipy import special
    import cupy as xp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    import scipy.special as special

EPS = xp.finfo(xp.float64).eps * 2
INF = xp.finfo(xp.float64).max / 1e6


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
    
    @classmethod
    def pdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._pdf((x - loc) / scale, *args, **kwargs) / scale
    
    @classmethod
    def logpdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._logpdf((x - loc) / scale, *args, **kwargs) - xp.log(scale)
    
    @staticmethod
    def _moments(*args, **kwargs):
        return 0.0, 0.0
    
    @classmethod
    def moments(cls, *args, loc=0, scale=1, **kwargs):
         mean, std = cls._moments(*args, **kwargs)
         return mean * scale + loc, std * scale

def rescale(x, loc, scale):
    return (x - loc) / scale

class trunc_dist(dist):

    @staticmethod
    @abstractmethod
    def _cdf(x, *args, **kwargs):
        # The core cdf logic for the standard distribution. Only required for trunc_dist
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _logcdf(x, *args, **kwargs):
        # The core logcdf logic for the standard distribution. Only required for trunc_dist
        raise NotImplementedError
    
    @staticmethod
    def _ppf(x, *args, **kwargs):
        # The core ppf logic for the standard distribution
        raise NotImplementedError
    
    @classmethod
    def pdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):

        pdf_unnorm = cls._pdf(rescale(x, loc, scale), *args, **kwargs) / scale * ((x >= xmin) & (x <= xmax))

        # normalize
        cdf_high = cls._cdf(rescale(xmax, loc, scale), *args, **kwargs) if xmax is not INF else 1
        cdf_low = cls._cdf(rescale(xmin, loc, scale), *args, **kwargs) if xmin is not -INF else 0
        norm = cdf_high - cdf_low
        cdf_high, cdf_low = None, None # for memory

        norm[~(norm > 0)] = xp.inf # Guard against division by zero (degenerate truncation)
        
        return pdf_unnorm / norm
    
    @classmethod 
    def logpdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        x_, xmin_, xmax_ = rescale(x, loc, scale), rescale(xmin, loc, scale), rescale(xmax, loc, scale)

        logpdf_unnorm = xp.where(
            (x < xmin) | (x > xmax),
            -INF,
            cls._logpdf(x_, *args, **kwargs) - xp.log(scale)
        )
        
        # normalize
        logcdf_high = xp.where(xmax >= INF, 0.0, cls._logcdf(xmax_, *args, **kwargs))
        logcdf_low = xp.where(xmin <= -INF, -INF, cls._logcdf(xmin_, *args, **kwargs))
        log_norm = xp.logaddexp(
            logcdf_high,
            logcdf_low + xp.log(-1)
        )
        # Guard against log(0) and -INF - (-INF) (degenerate truncation)
        return xp.where(log_norm > 0, logpdf_unnorm - log_norm, -INF)
    
    @classmethod
    def ppf(cls, q, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        _q_low = xp.clip(cls._cdf(rescale(xmin, loc, scale), *args, **kwargs), 0.0, 1 - EPS)
        _q_high = xp.clip(cls._cdf(rescale(xmax, loc, scale), *args, **kwargs), EPS, 1.0)
        
        _q = xp.clip(q * (_q_high - _q_low) + _q_low, EPS, 1 - EPS)
        return cls._ppf(_q, *args, **kwargs) * scale + loc

def lnbetainc(a, b, x):
    x = xp.asarray(x, dtype=xp.float64)
    a = xp.asarray(x, dtype=xp.float64)
    b = xp.asarray(x, dtype=xp.float64)
    
    template = xp.empty(
        xp.broadcast_shapes(a.shape, b.shape, x.shape),
        dtype=xp.float64
    )

    small = x <= 10*EPS
    big = x >= 1-10*EPS

    # assuming x is always the last axis
    template[..., small] = a * xp.log(x[small]) - xp.log(a) - special.betaln(a,b)
    template[..., big] = xp.log1p(-special.betainc(b, a, 1-x[big]))

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
        mode = (a - b) * xp.sqrt(a + b) / xp.sqrt( (2*a + 1) * (2*b + 1) )
        # x_ = x + mode # shift by mode

        return a, b, mode

    @staticmethod
    def _pdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)

        c = 2 ** (a + b - 1) * special.beta(a, b) * xp.sqrt(a + b)
        u = (x + shift) / xp.sqrt(a + b + (x + shift)**2)
        result = (1 + u) ** (a + 0.5) * (1 - u) ** (b + 0.5) / c
        # Guard against NaN from 0/0 or inf/inf when c is degenerate
        return xp.nan_to_num(result, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _logpdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)
        
        # Calculate the log of the normalization constant C
        # Use betaln for log(B(a,b)) to avoid underflow/overflow
        log_c = (a + b - 1) * xp.log(2) + special.betaln(a, b) + 0.5 * xp.log(a + b)

        # Pre-compute common terms
        # Clamp u to (-1+eps, 1-eps) to avoid log1p(-1)=-inf or log1p(1) issues at extreme x
        u_raw = (x + shift) / xp.sqrt(a + b + (x + shift)**2)
        u = xp.clip(u_raw, -1 + EPS, 1 - EPS)
        log_kernel = (a + 0.5) * special.log1p(u) + (b + 0.5) * special.log1p(-u)

        # Combine terms and adjust for the scale parameter
        return log_kernel - log_c
    
    @staticmethod
    def _moments(logalpha, logkappa, **kwargs):
        """
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
            (1.0 - u_star**2)**3 *
            (A + 1.0)**2 *
            (
                (a + 0.5) / (2.0 * a + 1.0)**2 +
                (b + 0.5) / (2.0 * b + 1.0)**2
            )
        )
        var = A / denom
        
        return 0.0, xp.sqrt(var) # mode at 0 by definition
    
    @staticmethod
    def _cdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)
        y = (1 + (x + shift) / xp.sqrt(a + b + (x + shift)**2)) * 0.5
        return special.betainc(a, b, y)
    
    @staticmethod
    def _logcdf(x, logalpha, logkappa):
        a, b, shift = jf_skew_t._reparam_and_shift(logalpha, logkappa)
        y = (1 + (x + shift) / xp.sqrt(a + b + (x + shift)**2)) * 0.5
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
        return -xp.sqrt(2) * special.erfcinv(2*q)
    
    @classmethod
    def moments(cls, loc=0, scale=1, xmin=-INF, xmax=INF):
        kwargs = dict(loc=loc, scale=scale, xmin=xmin, xmax=xmax)
        return loc, (cls.ppf(0.84, **kwargs) - cls.ppf(0.16, **kwargs))/2
        # ppf is already scaled

class gen_gaussian(trunc_dist):
    """
    Implements the generalized Gaussian (Normal) distribution.
    Lifted from https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_continuous_distns.py#L11395-L11488
    """

    @staticmethod
    def _pdf(x, beta):
        res = beta / (2 * special.gamma(1./beta)) * xp.exp(-abs(x)**beta)
        return xp.nan_to_num(res, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _logpdf(x, beta):
        return xp.log(0.5*beta) - special.gammaln(1.0/beta) - abs(x)**beta

    @staticmethod
    def _cdf(x, beta):
        c = 0.5 * xp.sign(x)
        # evaluating (.5 + c) first prevents numerical cancellation
        return (0.5 + c) - c * special.gammaincc(1.0/beta, abs(x)**beta)
    
    @staticmethod
    def _logcdf(x, beta):
        out = xp.full_like(x, -INF, dtype=xp.float64)
        log1p_term = xp.clip(special.gammainc(1./beta, abs(x)**beta), 0., 1.-EPS)
        out = xp.log(0.5) + xp.log1p(xp.sign(x) * log1p_term)
        return out
    
    @staticmethod
    def _moments(beta, **kwargs):
        return 0.0, xp.sqrt(special.gamma(3/beta)/special.gamma(1/beta))

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
        lam_R = 10**(-logkappa)
        return lam_L, lam_R

    @staticmethod
    def _pdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        norm = n / ((lam_L + lam_R) * special.gamma(1.0 / n))

        zL = (-x / lam_L) ** n
        zR = ( x / lam_R) ** n

        return norm * xp.where(x < 0, xp.exp(-zL), xp.exp(-zR))

    @staticmethod
    def _logpdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        lognorm = (
            xp.log(n)
            - xp.log(lam_L + lam_R)
            - special.gammaln(1.0 / n)
        )

        zL = (-x / lam_L) ** n
        zR = ( x / lam_R) ** n

        return lognorm - xp.where(x < 0, zL, zR)
    
    @staticmethod
    def _cdf(x, n, logkappa):
        lam_L, lam_R = SGED._lam_from_logkappa(logkappa)
        wL = lam_L / (lam_L + lam_R)
        wR = lam_R / (lam_L + lam_R)

        tL = (-x / lam_L) ** n
        tR = ( x / lam_R) ** n

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
        tR = ( x / lam_R) ** n

        cdf = xp.where(
            x < 0,
            wL * special.gammaincc(1.0 / n, tL),
            wL + wR * special.gammainc(1.0 / n, tR),
        )

        return xp.log(xp.clip(cdf, EPS, 1.0))
    
    @staticmethod
    def _moments(n, logkappa):
        var = (special.gamma(3.0 / n) / special.gamma(1.0 / n)) * \
              (10**(3*logkappa) + 10**(-3*logkappa)) / (10**logkappa + 10**(-logkappa))
        return 0.0, xp.sqrt(var)

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
        norm_beta_ne1 = (1.0 + beta) / (xmax**(1.0 + beta) - xmin**(1.0 + beta))
        norm = xp.where(beta == -1.0, 1.0 / log_ratio, norm_beta_ne1)

        prob = xp.power(x, beta)
        prob *= norm                   
        prob *= (x <= xmax) & (x >= xmin)  

        return prob
    
    @staticmethod
    def ppf(q, beta, xmin, xmax):
        a = 1.0 + beta
        return xp.where(
            beta == -1.0,
            xmin * (xmax / xmin)**q,
            xp.power((xmax**a - xmin**a)*q + xmin**a, 1/a)
        )
    
    @classmethod
    def moments(cls, beta, xmin, xmax):
        kwargs = dict(beta=beta, xmin=xmin, xmax=xmax)
        med = cls.ppf(0.5, **kwargs)
        width = med - cls.ppf(0.16, **kwargs)
        return med, width


class smoothed_powerlaw(dist):

    """
    Implements the power-law distribution with minimum, maximum, and lower-end smoothing.
    This is a different smoothing function from LVK s.t. the smoothing function is integrable.

    .. math::

        f(x; \alpha, x_{\min}, x_{\max}, p) = (1/C) x^{-\alpha} \left(1 - \frac{x_{\min}}{x} \right)^p
        
        C = x_{\min}^{1-\alpha} I_{1-x_{\min}/x_{\max}}(p+1, \alpha-1)
                    
    where I_u(a, b) is the incomplete beta function. Note that :math:`\alpha > 1`, since
    computing the normalization for :math:`\alpha < 1` involves hypergeometric functions that are
    not currently implemented by cupy.
    """

    @staticmethod
    def _pdf(x, alpha, xmin, xmax, p):

        x = xp.asarray(x)

        # Incomplete beta function
        u = 1 - xmin / xmax
        a = p + 1
        b = alpha - 1

        C = xp.power(xmin, 1 - alpha) * special.betainc(a, b, u) * special.beta(a,b)  
        # need to multiply regularized incomplete beta by beta(a,b) to get the incomplete beta function

        return xp.where(
            (x >= xmin) & (x <= xmax) & (C > 0),
            (1/C) * xp.power(x, -alpha) * xp.power(1.0 - xmin / x, p),
            0.0
        )
    
    @staticmethod
    def _logpdf(x, alpha, xmin, xmax, p):

        x = xp.asarray(x)
        
        u = 1 - xmin / xmax
        a = p + 1
        b = alpha - 1

        lnIu = lnbetainc(a, b, u) + special.betaln(a,b)  # stable for large/small params
        lnC = (1-alpha) * xp.log(xmin) + lnIu

        # Use safe ratio to avoid log(0)
        ratio = xp.clip(xmin / x, 0, 1 - EPS)
        
        # Main terms
        return xp.where(
            (x >= xmin) & (x <= xmax),
            -alpha * xp.log(x) + p * xp.log1p(-ratio) - lnC,
            -INF
        )
    
    @classmethod
    def moments(cls, alpha, xmin, xmax, p):
        # use mode for first moment, quantiles for width (approximated w/o low end smoothing)
        mode = xmin * (1 + p/alpha)
        width = (powerlaw.ppf(0.84, -alpha, xmin, xmax) - powerlaw.ppf(0.16, -alpha, xmin, xmax)) / 2 
        return mode, width

def sigmoid_smooth(x, xmin, delta):
    eps = delta/xp.log(INF) # for numerical stability
    x_safe = xp.where(
        x < xmin + eps, 
        xmin + eps, 
        xp.where(
            x > xmin + delta - eps,
            xmin + delta - eps,
            x
        )
    )
    return 1. / (1. + xp.exp(delta/(x_safe - xmin) + delta/(x_safe - xmin - delta)))

def log_sigmoid_smooth(x, xmin, delta):
    """
    Sigmoid low-mass tapering function. See 
    https://arxiv.org/pdf/2111.03634 Eqs. (B5), (B6)
    """
    eps = delta/xp.log(INF) # for numerical stability
    x_safe = xp.where(
        x < xmin + eps, 
        xmin + eps, 
        xp.where(
            x > xmin + delta - eps,
            xmin + delta - eps,
            x
        )
    )
    return -xp.log1p(xp.exp(delta/(x_safe - xmin) + delta/(x_safe - xmin - delta)))

class LVK_Plancktaper_powerlaw(dist):
    """
    Implements the power-law distribution with Planck-tapered lower end, as used by the LVK.
    """

    var_names = ['alpha', 'xmin', 'xmax', 'delta']

    @staticmethod
    def _unnorm_pdf(x, alpha, xmin, xmax, delta):
        return xp.where(
            (x > xmin) & (x < xmax),
            xp.power(x, -alpha) * sigmoid_smooth(x, xmin, delta),
            0
        )
    
    @staticmethod
    def _unnorm_logpdf(x, alpha, xmin, xmax, delta):
        return xp.where(
            (x > xmin) & (x < xmax),
            -alpha * xp.log(x) + log_sigmoid_smooth(x, xmin, delta),
            -INF
        )

    @classmethod
    def _pdf(cls, x, alpha, xmin, xmax, delta, xx_int=None):
        pdf_unnorm = cls._unnorm_pdf(x, alpha, xmin, xmax, delta)

        # compute normalization - assumes x is the last dimension 
        if xx_int is None:
            xx_int = xp.linspace(xp.amin(xp.asarray(xmin)), xp.amax(xp.asarray(xmax)), 256) 
        
        norm = xp.trapz(
            cls._unnorm_pdf(xx_int, alpha, xmin, xmax, delta),
            xx_int,
            axis=-1
        )

        if xp.asarray(norm).ndim > 0:
            norm = norm[:,None]

        return pdf_unnorm / norm
    
    @classmethod
    def _logpdf(cls, x, alpha, xmin, xmax, delta, xx_int=None):
        logpdf_unnorm = cls._unnorm_logpdf(x, alpha, xmin, xmax, delta)

        # compute normalization - assumes x is the last dimension 
        if xx_int is None:
            xx_int = xp.linspace(xp.amin(xp.asarray(xmin)), xp.amax(xp.asarray(xmax)), 256) 
        
        norm = xp.trapezoid(
            cls._unnorm_pdf(xx_int, alpha, xmin, xmax, delta),
            xx_int,
            axis=-1
        )[:,None]
        return logpdf_unnorm - xp.log(norm)

# class primary_mass_q_distribution:

#     def __init__(self, primary_mass_model, q_model):
#         self.primary_mass_model = primary_mass_model
#         self.q_model = q_model

#     def __call__(self, m1, q, m1_kwargs, q_kwargs):
#         return self.primary_mass_model(m1, **m1_kwargs) * self.q_model(q, xmin=m1_kwargs['xmin']/m1, xmax=1., **q_kwargs)

# class chirp_mass_q_distribution:

#     def __init__(self, chirp_mass_model, q_model):
#         self.chirp_mass_model = chirp_mass_model
#         self.q_model = q_model

#     def __call__(self, mchirp, q, mchirp_kwargs, q_kwargs):
#         return self.chirp_mass_model(mchirp, **mchirp_kwargs) * self.q_model(q, xmin=0., xmax=1., **q_kwargs)


###################
# Rate models
###################

class BaseRate:

    def __init__(self, cosmo=Planck15):
        self.cosmo = cosmo
        
        z_arr = np.concatenate([np.linspace(EPS, 3, 256, endpoint=False), np.linspace(3, 10, 50)])
        dV_dz_arr = 4*np.pi * self.cosmo.differential_comoving_volume(z_arr).value / 1e9 # in Gpc^3
        self._z_arr = xp.asarray(z_arr)
        self._dV_dz_arr = xp.asarray(dV_dz_arr)
        self._log_dV_dz_arr = xp.log(self._dV_dz_arr)
    
    def dV_dz(self, z):
        return xp.interp(z, self._z_arr, self._dV_dz_arr, left=0, right=0)
    
    def log_dV_dz(self, z):
        return xp.interp(z, self._z_arr, self._log_dV_dz_arr, left=-INF, right=-INF)
    
    @abstractmethod 
    def psi(self,z, *args, **kwargs):
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

    def pdf(self, z, *args, **kwargs):
        return self.psi(z, *args, **kwargs) / (1.+ z) * self.dV_dz(z)

    def logpdf(self, z, *args, **kwargs):
        return self.log_psi(z, *args, **kwargs) - xp.log1p(z) + self.log_dV_dz(z)

class PL_rate(BaseRate):
    
    @staticmethod
    def psi(z, gamma):
        """
        Power-law rate model.
        """
        return (1.0+z)**gamma
    
    @staticmethod
    def log_psi(z, gamma):
        return gamma*xp.log1p(z)

class MD_rate(BaseRate):
    
    @staticmethod
    def psi(z, gamma, kappa, zp):
        """
        Madau-Dickinson rate model. See e.g.
        https://arxiv.org/abs/2003.12152 Eq. (2).
        """
        C = 1 + (1 + zp)**(-(gamma + kappa))
        return C * (1.+ z)**gamma / (1.+ ((1.+ z) / (1.+ zp))**(gamma + kappa))

    @staticmethod
    def log_psi_MD(z, gamma, kappa, zp):
        logC = xp.log1p(xp.power(1+zp, -(gamma+kappa)))
        return gamma*xp.log1p(z) - xp.log1p( ((1.+z)/(1.+zp))**(gamma+kappa) ) + logC