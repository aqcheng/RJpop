import numpy as np
import scipy
from abc import ABC, abstractmethod

from astropy.cosmology import Planck15

try:
    from cupyx.scipy import special
    import cupy as xp
except (ModuleNotFoundError, ImportError) as e:
    import numpy as xp
    import scipy.special as special

EPS = xp.finfo(xp.float64).eps * 2
INF = xp.finfo(xp.float64).max / 2


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
    @staticmethod
    @abstractmethod
    def _pdf(x, *args, **kwargs):
        # The core PDF logic for the standard distribution (loc=0, scale=1)
        pass
    
    @staticmethod
    @abstractmethod
    def _logpdf(x, *args, **kwargs):
        # The core logpdf logic for the standard distribution
        pass
    
    @classmethod
    def pdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._pdf((x - loc) / scale, *args, **kwargs) / scale
    
    @classmethod
    def logpdf(cls, x, *args, loc=0, scale=1, **kwargs):
        return cls._logpdf((x - loc) / scale, *args, **kwargs) - xp.log(scale)

def rescale(x, loc, scale):
    return (x - loc) / scale

class trunc_dist(dist):

    @staticmethod
    @abstractmethod
    def _cdf(x, *args, **kwargs):
        # The core cdf logic for the standard distribution. Only required for trunc_dist
        pass

    @staticmethod
    @abstractmethod
    def _logcdf(x, *args, **kwargs):
        # The core logcdf logic for the standard distribution. Only required for trunc_dist
        pass
    
    @classmethod
    def pdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        x_, xmin_, xmax_ = rescale(x, loc, scale), rescale(xmin, loc, scale), rescale(xmax, loc, scale)

        pdf_unnorm = xp.where(
            (x < xmin) | (x > xmax),
            xp.zeros_like(x_, dtype=xp.float64),
            cls._pdf(x_, *args, **kwargs) / scale
        )

        # normalize
        cdf_high = cls._cdf(xmax_, *args, **kwargs) if xmax is not INF else 1
        cdf_low = cls._cdf(xmin_, *args, **kwargs) if xmin is not -INF else 0
        return pdf_unnorm / (cdf_high - cdf_low)
    
    @classmethod 
    def logpdf(cls, x, *args, loc=0, scale=1, xmin=-INF, xmax=INF, **kwargs):
        x_, xmin_, xmax_ = rescale(x, loc, scale), rescale(xmin, loc, scale), rescale(xmax, loc, scale)

        logpdf_unnorm = xp.where(
            (x < xmin) | (x > xmax),
            xp.full_like(x_, -INF, dtype=xp.float64),
            cls._logpdf(x_, *args, **kwargs) - xp.log(scale)
        )
        
        # normalize
        cdf_high = cls._cdf(xmax_, *args, **kwargs) if xmax is not INF else 1
        cdf_low = cls._cdf(xmin_, *args, **kwargs) if xmin is not -INF else 0
        return logpdf_unnorm - xp.log(cdf_high - cdf_low)

def lnbetainc(a, b, x):
    x = xp.asarray(x, dtype=xp.float64)
    template = xp.empty_like(x, dtype=xp.float64)

    small = x <= 10*EPS
    big = x >= 1-10*EPS

    template[small] = a * xp.log(x[small]) - xp.log(a) - special.betaln(a,b)
    template[big] = xp.log1p(-special.betainc(b, a, 1-x[big]))

    good = ~small & ~big
    template[good] = xp.log(special.betainc(a, b, x[good]))
    return template

class jf_skew_t(trunc_dist):
    r"""Jones and Faddy skew-t distribution. (Notes and pdf lifted from scipy.stats.jf_skew_t.
    See docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jf_skew_t.html)

    Notes
    -----
    The probability density function for `jf_skew_t` is:

    .. math::

        f(x; a, b) = C_{a,b}^{-1}
                    \left(1+\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{a+1/2}
                    \left(1-\frac{x}{\left(a+b+x^2\right)^{1/2}}\right)^{b+1/2}

    for real numbers :math:`a>0` and :math:`b>0`, where
    :math:`C_{a,b} = 2^{a+b-1}B(a,b)(a+b)^{1/2}`, and :math:`B` denotes the
    beta function (`scipy.special.beta`).

    When :math:`a<b`, the distribution is negatively skewed, and when
    :math:`a>b`, the distribution is positively skewed. If :math:`a=b`, then
    we recover the `t` distribution with :math:`2a` degrees of freedom.

    `jf_skew_t` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    References
    ----------
    .. [1] M.C. Jones and M.J. Faddy. "A skew extension of the t distribution,
           with applications" *Journal of the Royal Statistical Society*.
           Series B (Statistical Methodology) 65, no. 1 (2003): 159-174.
           :doi:`10.1111/1467-9868.00378`

    %(example)s
    """

    @staticmethod
    def _pdf(x, a, b):
        c = 2 ** (a + b - 1) * special.beta(a, b) * xp.sqrt(a + b)
        d1 = (1 + x / xp.sqrt(a + b + x ** 2)) ** (a + 0.5)
        d2 = (1 - x / xp.sqrt(a + b + x ** 2)) ** (b + 0.5)
        return d1 * d2 / c

    @staticmethod
    def _logpdf(x, a, b):
        # Calculate the log of the normalization constant C
        # Use betaln for log(B(a,b)) to avoid underflow/overflow
        log_c = (a + b - 1) * xp.log(2) + special.betaln(a, b) + 0.5 * xp.log(a + b)

        # Pre-compute common terms
        u = x / xp.sqrt(a + b + x**2)  # This term approaches +/- 1 for large |x|
        log_kernel = (a + 0.5) * special.log1p(u) + (b + 0.5) * special.log1p(-u)

        # Combine terms and adjust for the scale parameter
        return log_kernel - log_c
    
    @staticmethod
    def _cdf(x, a, b):
        y = (1 + x / xp.sqrt(a + b + x ** 2)) * 0.5
        return special.betainc(a, b, y)
    
    @staticmethod
    def _logcdf(x, a, b):
        """
        Compact, stable log-CDF for Jones–Faddy skew-t:
        I_y(a,b) with y = (1 + x / sqrt(a + b + x**2)) / 2
        """
        y = (1 + x / xp.sqrt(a + b + x**2)) * 0.5
        return lnbetainc(y, a, b)

class gaussian(trunc_dist):
    """
    Implements the Gaussian (Normal) distribution.
    """
    @staticmethod
    def _pdf(x):
        return xp.exp(-0.5 * x**2) / xp.sqrt(2 * xp.pi)

    @staticmethod
    def _logpdf(x):
        return -0.5 * x**2 - 0.5 * xp.log(2 * xp.pi)
    
    @staticmethod
    def _cdf(x):
        return special.ndtr(x)
    
    @staticmethod
    def _logcdf(x):
        return special.log_ndtr(x)

class gen_gaussian(trunc_dist):
    """
    Implements the generalized Gaussian (Normal) distribution.
    Lifted from https://github.com/scipy/scipy/blob/v1.16.2/scipy/stats/_continuous_distns.py#L11395-L11488
    """

    @staticmethod
    def _pdf(x, beta):
        return beta / (2 * special.gamma(1./beta)) * xp.exp(-abs(x)**beta)

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
        beta_, xmin_, xmax_ = xp.broadcast_arrays(beta, xmin, xmax)
        norm = xp.where(
            beta_ == -1.0,
            1 / xp.log(xmax_ / xmin_),
            (1 + beta_) / (xmax_**(1.0 + beta_) - xmin_**(1.0 + beta_)),
        )

        prob = xp.power(x, beta_)
        prob *= norm
        prob *= (x <= xmax_) & (x >= xmin_)
        return prob


class smoothed_powerlaw(dist):

    """
    Implements the power-law distribution with minimum, maximum, and lower-end smoothing.
    This is a different smoothing function from LVK s.t. the smoothing function is integrable.

    .. math::

        f(x; \alpha, x_{\min}, x_{\max}, p) = (1/C) x^{-\alpha} \left(1 - \frac{x_{\min}}{x} \right)^p
        
        C = x_{\min}^{1-\alpha} I_{1-x_{\min}/x_{\max}}(p+1, \alpha-1)
                    
    where I_u(a, b) is the incomplete beta function.
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
            (x >= xmin) & (x <= xmax),
            (1/C) * xp.power(x, -alpha) * xp.power(1.0 - xmin / x, p),
            0
        )
    
    @staticmethod
    def _logpdf(x, alpha, xmin, xmax, p):

        x = xp.asarray(x)
        
        u = 1 - xmin / xmax
        a = p + 1
        b = alpha - 1

        lnIu = lnbetainc(a, b, u) + special.betaln(a,b)  # stable for large/small params
        lnC = (1-alpha) * xp.log(xmin) + lnIu
        
        # Main terms
        return xp.where(
            (x >= xmin) & (x <= xmax),
            -alpha * xp.log(x) + p * xp.log1p(-xmin / x) - lnC,
            -INF
        )


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
            xx_int = xp.linspace(xp.amin(xp.asarray(xmin)), xp.amax(xp.asarray(xmax)), 1000) 
        
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
            xx_int = xp.linspace(xp.amin(xp.asarray(xmin)), xp.amax(xp.asarray(xmax)), 1000) 
        
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
        
        z_arr = np.concatenate([np.linspace(EPS, 5, 1000, endpoint=False), np.linspace(5, 10, 100)])
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


def power_law(x,ampl_arr,index_arr,up_cut_arr, low_cut_arr):

    norm=(1/(1-index_arr))*(up_cut_arr**(1-index_arr)-low_cut_arr**(1-index_arr))

    if index_arr.ndim==1:
        ans=x[None,:]**(-index_arr[:,None])*(ampl_arr[:,None]/norm[:,None])
        ans[x[None,:]<low_cut_arr[:,None]]=0
        ans[x[None,:]>up_cut_arr[:,None]]=0

    elif index_arr.ndim==2:
        ans=x[None,:]**(-index_arr)*(ampl_arr[:,None]/norm)
        ans[x[None,:]<low_cut_arr]=0
        ans[x[None,:]>up_cut_arr]=0

    return ans




def smooth_exp(m,smooth_scale):

    return xp.exp((smooth_scale/m)+(smooth_scale/(m-smooth_scale)))

def smooth_function(x,params_global,nsmooth=0):

    smooth_scale_arr=xp.asarray(params_global['deltam'])
    low_cut_arr=xp.asarray(params_global['mmin'])

    if nsmooth==0:

        if x.ndim==1:
            ans=(1+smooth_exp(x[None,:]-low_cut_arr[:,None],smooth_scale_arr[:,None]))**(-1)
        elif x.ndim==2:
            ans=(1+smooth_exp(x[None,:,:]-low_cut_arr[:,None,None],smooth_scale_arr[:,None,None]))**(-1)
        ans[x[None,:]*xp.ones_like(low_cut_arr)[:,None]<low_cut_arr[:,None]*xp.ones_like(x)[None,:]]=0
        ans[x[None,:]*xp.ones_like(low_cut_arr)[:,None]>(low_cut_arr+smooth_scale_arr)[:,None]*xp.ones_like(x)[None,:]]=1

    else:

        if low_cut_arr.ndim==1 and smooth_scale_arr.ndim==1:
            ans=(x[None,:]-low_cut_arr[:,None])**nsmooth*(2**(nsmooth-1.)/smooth_scale_arr[:,None]**nsmooth)

            fix = xp.zeros((len(smooth_scale_arr), len(x)), dtype=bool)
            inds_fix=xp.asarray(xp.argwhere((x[None,:]>low_cut_arr[:,None]+smooth_scale_arr[:,None]/2.) & (x[None,:]<low_cut_arr[:,None]+smooth_scale_arr[:,None])))
            fix[(inds_fix[:, 0], inds_fix[:, 1])] = True

            ans[fix]=(1.-(smooth_scale_arr[:,None]+low_cut_arr[:,None]-x[None,:])**nsmooth*(2**(nsmooth-1.)/smooth_scale_arr[:,None]**nsmooth))[fix]

            ans[x[None,:]<low_cut_arr[:,None]]=0
            ans[x[None,:]>(low_cut_arr+smooth_scale_arr)[:,None]]=1

        else:


            if low_cut_arr.ndim==1:
                low_cut_arr=low_cut_arr[:,None]

            if smooth_scale_arr.ndim==1:
                smooth_scale_arr=smooth_scale_arr[:,None]

            ans=(x[None,:]-low_cut_arr)**nsmooth*(2**(nsmooth-1.)/smooth_scale_arr**nsmooth)

            fix = xp.zeros((len(smooth_scale_arr), len(x)), dtype=bool)
            inds_fix=xp.asarray(xp.argwhere((x[None,:]>low_cut_arr+smooth_scale_arr/2.) & (x[None,:]<low_cut_arr+smooth_scale_arr)))
            fix[(inds_fix[:, 0], inds_fix[:, 1])] = True

            ans[fix]=(1.-(smooth_scale_arr+low_cut_arr-x[None,:])**nsmooth*(2**(nsmooth-1.)/smooth_scale_arr**nsmooth))[fix]

            ans[x[None,:]<low_cut_arr]=0
            ans[x[None,:]>low_cut_arr+smooth_scale_arr]=1

    return ans


def combine_gaussians_gamma_z(tm,tz,x1, params_global, group_gauss, group_global, mlow=None, mhigh=None,zmax=2.3,to_renorm_z=True):
    ampl = xp.asarray(10**(x1[:, 0]))
    mean = xp.asarray(x1[:, 1])
    std =  xp.asarray(x1[:, 2])
    if mlow is None:
        mlow = xp.asarray(params_global['mmin'])[group_gauss]
    if mhigh is None:
        mhigh = xp.ones_like(x1[:,0])*100

    gauss_out = gaussian_truncated(tm,ampl,mean,std,mlow,mhigh)
    pdfs_z_out=z_pdf_gamma(tz,x1[:,3:],zmax=zmax,to_renorm=to_renorm_z)

    gauss_out*=pdfs_z_out
    
    num_groups = np.concatenate([group_gauss,group_global]).max() + 1
    group_gauss = xp.asarray(group_gauss)
    unique, unique_index, unique_inverse, unique_count = xp.unique(group_gauss, return_index=True, return_counts=True, return_inverse=True)

    if unique_count.size==0:
        out=0

    else:
        max_per_group = unique_count.max().item()

        diff_temp = xp.ones_like(unique_inverse)
        diff_temp[1:] = (~xp.diff(unique_inverse).astype(bool)).astype(int)
        inds_per_group_gauss = (xp.cumsum(diff_temp) - 1)
        inds_group_subtract = inds_per_group_gauss[unique_index][unique_inverse]
        inds_per_group = inds_per_group_gauss - inds_group_subtract

        template = xp.zeros((num_groups, max_per_group, len(tm)))
        template[(group_gauss, inds_per_group)] = gauss_out
        # sum over middle axis
        out = template.sum(axis=1)


    return out



def combine_gaussians_pl_z(tm,tz,x1, params_global, group_gauss, group_global, dVc_spl, mlow=None, mhigh=None,zmax=2.3):
    ampl = xp.asarray(10**(x1[:, 0]))
    mean = xp.asarray(x1[:, 1])
    std =  xp.asarray(x1[:, 2])
    if mlow is None:
        mlow = xp.asarray(params_global['mmin'])[group_gauss]
    if mhigh is None:
        mhigh = xp.ones_like(x1[:,0])*100

    gauss_out = gaussian_truncated(tm,ampl,mean,std,mlow,mhigh)
    pdfs_z_out=z_pdf(tz,x1[:,3],dVc_spl,zmax=zmax)

    gauss_out*=pdfs_z_out
    
    num_groups = np.concatenate([group_gauss,group_global]).max() + 1
    group_gauss = xp.asarray(group_gauss)
    unique, unique_index, unique_inverse, unique_count = xp.unique(group_gauss, return_index=True, return_counts=True, return_inverse=True)

    if unique_count.size==0:
        out=0

    else:
        max_per_group = unique_count.max().item()

        diff_temp = xp.ones_like(unique_inverse)
        diff_temp[1:] = (~xp.diff(unique_inverse).astype(bool)).astype(int)
        inds_per_group_gauss = (xp.cumsum(diff_temp) - 1)
        inds_group_subtract = inds_per_group_gauss[unique_index][unique_inverse]
        inds_per_group = inds_per_group_gauss - inds_group_subtract

        template = xp.zeros((num_groups, max_per_group, len(tm)))
        template[(group_gauss, inds_per_group)] = gauss_out
        # sum over middle axis
        out = template.sum(axis=1)


    return out




def combine_gaussians(t,x1, params_global, group_gauss, group_global, mlow=None, mhigh=None):
    ampl = xp.asarray(10**(x1[:, 0]))
    mean = xp.asarray(x1[:, 1])
    std =  xp.asarray(x1[:, 2])
    if mlow is None:
        mlow = xp.asarray(params_global['mmin'])[group_gauss]
    if mhigh is None:
        mhigh = xp.ones_like(x1[:,0])*100

    gauss_out = gaussian_truncated(t,ampl,mean,std,mlow,mhigh)
    

    num_groups = np.concatenate([group_gauss,group_global]).max() + 1
    group_gauss = xp.asarray(group_gauss)
    unique, unique_index, unique_inverse, unique_count = xp.unique(group_gauss, return_index=True, return_counts=True, return_inverse=True)

    if unique_count.size==0:
        out=0

    else:
        max_per_group = unique_count.max().item()

        diff_temp = xp.ones_like(unique_inverse)
        diff_temp[1:] = (~xp.diff(unique_inverse).astype(bool)).astype(int)
        inds_per_group_gauss = (xp.cumsum(diff_temp) - 1)
        inds_group_subtract = inds_per_group_gauss[unique_index][unique_inverse]
        inds_per_group = inds_per_group_gauss - inds_group_subtract

        template = xp.zeros((num_groups, max_per_group, len(t)))
        template[(group_gauss, inds_per_group)] = gauss_out
        # sum over middle axis
        out = template.sum(axis=1)

    return out


def combine_ampls(log10_ampls,group_gauss,group_global):
    ampls = xp.asarray(10**log10_ampls)

    num_groups = np.concatenate([group_gauss, group_global]).max() + 1
    group_gauss = xp.asarray(group_gauss)
    unique, unique_index, unique_inverse, unique_count = xp.unique(group_gauss, return_index=True, return_counts=True, return_inverse=True)

    if unique_count.size==0:

        sum_ampls,prod_ampls,nlambdas=xp.zeros(int(num_groups)),xp.ones(int(num_groups)),xp.zeros(int(num_groups))


    else:
        max_per_group = unique_count.max().item()


        diff_temp = xp.ones_like(unique_inverse)
        diff_temp[1:] = (~xp.diff(unique_inverse).astype(bool)).astype(int)
        inds_per_group_gauss = (xp.cumsum(diff_temp) - 1)
        inds_group_subtract = inds_per_group_gauss[unique_index][unique_inverse]
        inds_per_group = inds_per_group_gauss - inds_group_subtract

        template_sum = xp.zeros((num_groups, max_per_group))
        # template_prod = xp.ones((num_groups, max_per_group))
        template_sum[(group_gauss, inds_per_group)] = ampls
        # template_prod[(group_gauss, inds_per_group)] = ampls
        # sum over middle axis
        sum_ampls = template_sum.sum(axis=1)
        # prod_ampls = template_prod.prod(axis=1)

        # nlambdas=xp.zeros((num_groups, max_per_group))
        # nlambdas[(group_gauss, inds_per_group)]=xp.ones(len(ampls))
        # nlambdas=nlambdas.sum(axis=1)


    return sum_ampls


def m1_z_pdf_gauss_gamma(m1,z,xgauss,group_gauss,params_global,group_global,to_norm=True,return_norm=False,zmax=2.3,to_renorm_z=True):

    #to_renorm normalises the z_pdf to the desired range

    gauss_gamma_sum=combine_gaussians_gamma_z(m1,z,xgauss,params_global,group_gauss,group_global,zmax=zmax,to_renorm_z=to_renorm_z)
    ans=gauss_gamma_sum

    if to_norm or return_norm:
        sum_ampl_gauss_gamma=combine_ampls(xgauss[:,0],group_gauss,group_global)
        norm=sum_ampl_gauss_gamma

    if to_norm:
        ans/=norm[:,None]

    if return_norm:
        return ans,norm

    else:
        return ans
    

def m1_z_pdf_gauss_pl(m1,z,xgauss,group_gauss,params_global,group_global,dVc_spl,to_norm=True,return_norm=False,zmax=2.3):

    #to_renorm normalises the z_pdf to the desired range

    gauss_gamma_sum=combine_gaussians_pl_z(m1,z,xgauss,params_global,group_gauss,group_global,dVc_spl,zmax=zmax)
    ans=gauss_gamma_sum

    
    if to_norm or return_norm:
        sum_ampl_gauss_gamma=combine_ampls(xgauss[:,0],group_gauss,group_global)
        norm=sum_ampl_gauss_gamma

    if to_norm:
        ans/=norm[:,None]

    if return_norm:
        return ans,norm

    else:
        return ans




def m1_pdf_gauss(m1,xgauss,group_gauss,params_global,group_global,to_norm=True,return_norm=False):

    gauss_sum=combine_gaussians(m1,xgauss,params_global,group_gauss,group_global)
    ans=gauss_sum

    if to_norm or return_norm:
        sum_ampl_gauss=combine_ampls(xgauss[:,0],group_gauss,group_global)
        norm=sum_ampl_gauss

    if to_norm:
        ans/=norm[:,None]

    if return_norm:
        return ans,norm

    else:
        return ans
    



def spin_tilt_pdf(tilts1,tilts2,params_spins):

    zeta=xp.asarray(params_spins['zeta'])
    sigmat=xp.asarray(params_spins['sigmat'])

    tilts1_ext=xp.repeat(tilts1,len(zeta)).reshape((len(tilts1),len(zeta))).T
    tilts2_ext=xp.repeat(tilts2,len(zeta)).reshape((len(tilts2),len(zeta))).T

    try:
        norm_gauss=2./special.erf(xp.sqrt(2)/sigmat)
    except (TypeError, NameError) as e:
        norm_gauss=2./scipy.special.erf(xp.sqrt(2)/sigmat)

    # breakpoint()
    # ans=zeta[:,None]*gaussian(tilt,xp.ones(len(sigmat)),xp.zeros(len(sigmat)),sigmat)*norm_gauss[:,None]+(1-zeta[:,None])*(1/2)
    ans=(1-zeta[:,None])*(1/4)+zeta[:,None]*gaussian(tilts1,xp.ones(len(sigmat)),xp.ones(len(sigmat)),sigmat)*gaussian(tilts2,xp.ones(len(sigmat)),xp.ones(len(sigmat)),sigmat)*norm_gauss[:,None]**2

    ans[tilts1_ext<-1]=0
    ans[tilts1_ext>1]=0

    ans[tilts2_ext<-1]=0
    ans[tilts2_ext>1]=0

    return ans


def spin_tilt_pdf_marg(tilts,params_spins):

    zeta=xp.asarray(params_spins['zeta'])
    sigmat=xp.asarray(params_spins['sigmat'])

    tilts_ext=xp.repeat(tilts,len(zeta)).reshape((len(tilts),len(zeta))).T


    try:
        norm_gauss=2./special.erf(xp.sqrt(2)/sigmat)
    except (TypeError, NameError) as e:
        norm_gauss=2./scipy.special.erf(xp.sqrt(2)/sigmat)

    # breakpoint()
    # ans=zeta[:,None]*gaussian(tilt,xp.ones(len(sigmat)),xp.zeros(len(sigmat)),sigmat)*norm_gauss[:,None]+(1-zeta[:,None])*(1/2)
    ans=(1-zeta[:,None])*(1/2)+zeta[:,None]*gaussian(tilts,xp.ones(len(sigmat)),xp.ones(len(sigmat)),sigmat)*norm_gauss[:,None]

    ans[tilts_ext<-1]=0
    ans[tilts_ext>1]=0

    return ans

def beta_pdf(x, a, b):
    # unnormalized
    log_pdf = (a-1)*xp.log(x)+(b-1)*xp.log(1-x) - (special.gammaln(a) + special.gammaln(b) - special.gammaln(a + b))


    return xp.exp(log_pdf)

def spin_mag_pdf(chi,params_spins):

    abetas=xp.asarray(params_spins['abeta'])
    bbetas=xp.asarray(params_spins['bbeta'])
    ans=beta_pdf(chi[None,:],abetas[:,None],bbetas[:,None])

    # breakpoint()

    ans[abetas<=0,:]=0
    ans[bbetas<=0,:]=0

    return ans

def beta_pdf_not_norm(x, a, b):
    # unnormalized
    log_pdf = (a-1)*xp.log(x)+(b-1)*xp.log(1-x)

    return xp.exp(log_pdf)

def spin_mag_pdf_not_norm(chi,params_spins):

    abetas=xp.asarray(params_spins['abeta'])
    bbetas=xp.asarray(params_spins['bbeta'])
    ans=beta_pdf_not_norm(chi[None,:],abetas[:,None],bbetas[:,None])

    ans[abetas<=0,:]=0
    ans[bbetas<=0,:]=0

    return ans

def q_pdf_sharp(q,m1,betaq,params_global):

    betaq = xp.asarray(betaq)
    # q = xp.asarray(q)

    mmin=xp.asarray(params_global['mmin'])

    # breakpoint()

    ans=q[None,:]**(betaq[:,None])
    
    ans=ans*(betaq[:,None]+1.)/(1.-(mmin[:,None]/m1[None,:])**(betaq[:,None]+1.))

    ans[q[None,:]*m1[None,:]<mmin[:,None]]=0.

    return ans


def psi_of_z(zs,kappas):


    return (1+zs[None,:])**(kappas[:,None])


def z_pdf_not_norm(zs,kappas,dVc_spl,zmax):

    try:
        ans=(1+zs[None,:])**(kappas[:,None]-1)*xp.asarray(dVc_spl(zs.get()))[None,:]
    except AttributeError:
        ans=(1+zs[None,:])**(kappas[:,None]-1)*xp.asarray(dVc_spl(zs))[None,:]

    z_ext=xp.repeat(zs,len(kappas)).reshape((len(zs),len(kappas))).T


    ans[z_ext>zmax]=0

    return ans


def z_pdf(zs,kappas,dVc_spl,zmax=2.3):

    z_grid=xp.linspace(0,zmax,1000)
    pdf_grid=z_pdf_not_norm(z_grid,kappas,dVc_spl,zmax)
    norm=trapz(pdf_grid,x=z_grid,axis=1)

    ans=z_pdf_not_norm(zs,kappas,dVc_spl,zmax)/norm[:,None]

    return ans


def z_pdf_gamma(zs,xz,zmax=2.3,to_renorm=False):

    #to_renorm normalises the z_pdf to the desired range

    alpha=xz[:,0]
    theta=xz[:,1]

    ans0=zs[None,:]**(alpha[:,None]-1.)*xp.exp(-zs[None,:]/theta[:,None])

    
    if to_renorm:
        zs_grid=xp.linspace(0,zmax,1000)
        ans_grid=zs_grid[None,:]**(alpha[:,None]-1.)*xp.exp(-zs_grid[None,:]/theta[:,None])

        norm_grid=trapz(ans_grid,x=zs_grid,axis=1)

        ans=ans0/norm_grid[:,None]

        ans[ans0==0.]=0.

    else:
        ans=ans0

    return ans

def z_pdf_madau_dickinson_not_norm(zs,dVc_spl,zmax):

    try:
        ans=((1+zs)**(2.7-1.)/(1.+((1.+zs)/2.9)**5.6))*xp.asarray(dVc_spl(zs.get()))
    except AttributeError:
        ans=((1+zs)**(2.7-1.)/(1.+((1.+zs)/2.9)**5.6))*xp.asarray(dVc_spl(zs))


    ans[zs>zmax]=0

    return ans


def compute_vt(kappas,dVc_spl,zmax,Tobs):

    z_grid=xp.linspace(0,zmax,1000)
    pdf_grid=z_pdf_not_norm(z_grid,kappas,dVc_spl,zmax)
    norm=trapz(pdf_grid,x=z_grid,axis=1)

    vt=norm*Tobs/(365.25*24*60*60)

    return vt



def draw_z(kappa,dVc_spl,zmax):

    zs=np.linspace(0,zmax,1000)
    pdfs=z_pdf_not_norm(zs,kappa,dVc_spl,zmax)
    pdf_max=np.amax(pdfs)

    draw=True

    while draw:
        ztry=np.random.uniform(0,zmax)
        pdf_try=z_pdf_not_norm(np.array([ztry]),kappa,dVc_spl,zmax)
        pkeep=np.random.rand()
        if pkeep*pdf_max<pdf_try:
            draw=False

    return ztry


def draw_z_madau_dickinson(dVc_spl,zmax):

    zs=np.linspace(0,zmax,1000)
    pdfs=z_pdf_madau_dickinson_not_norm(zs,dVc_spl,zmax)
    pdf_max=np.amax(pdfs)

    # breakpoint()

    draw=True

    while draw:
        ztry=np.random.uniform(0,zmax)
        pdf_try=z_pdf_madau_dickinson_not_norm(np.array([ztry]),dVc_spl,zmax)
        pkeep=np.random.rand()
        if pkeep*pdf_max<pdf_try:
            draw=False

    return ztry


def draw_m1_z_gamma(npts,xgauss,group_gauss,params_global,group_global,xneg_gauss=None,group_neg_gauss=None,xnar=None,group_nar_gauss=None,xbpl=None,group_bpl=None,neg_gauss=False,broken_pl=False,nsmooth=0,zmax=2.3):

    ms=np.linspace(2,100,1000)
    zs=np.linspace(0,zmax,1000)
    pdfs=np.zeros((1000,1000))

   
    
    for i in range(1000):
        pdfs[i]=m1_z_pdf_gauss_gamma(np.ones(1000)*ms[i],zs,xgauss,group_gauss,params_global,group_global,zmax=zmax)
    
    pdf_max=np.amax(pdfs)


    samples=np.zeros((npts,2))

    for i in range(npts):
        draw=True
        while draw:
            m1try=np.random.uniform(2,100)
            ztry=np.random.uniform(0,zmax)
            pdf_try=m1_z_pdf_gauss_gamma(np.array([m1try]),np.array([ztry]),xgauss,group_gauss,params_global,group_global,zmax=zmax)
            pkeep=np.random.rand()
            if pkeep*pdf_max<pdf_try:
                draw=False

        samples[i,0]=m1try
        samples[i,1]=ztry

    return samples


def draw_m1_z_pl(npts,xgauss,group_gauss,params_global,group_global,dVc_spl,zmax=2.3):

    ms=np.linspace(2,100,1000)
    zs=np.linspace(0,zmax,1000)
    pdfs=np.zeros((1000,1000))

   
    
    for i in range(1000):
        pdfs[i]=m1_z_pdf_gauss_pl(np.ones(1000)*ms[i],zs,xgauss,group_gauss,params_global,group_global,dVc_spl,zmax=zmax)
    
    pdf_max=np.amax(pdfs)


    samples=np.zeros((npts,2))

    for i in range(npts):
        draw=True
        while draw:
            m1try=np.random.uniform(2,100)
            ztry=np.random.uniform(0,zmax)
            pdf_try=m1_z_pdf_gauss_pl(np.array([m1try]),np.array([ztry]),xgauss,group_gauss,params_global,group_global,dVc_spl,zmax=zmax)
            pkeep=np.random.rand()
            if pkeep*pdf_max<pdf_try:
                draw=False

        samples[i,0]=m1try
        samples[i,1]=ztry

    return samples


def draw_q_m1_sharp(m1,betaq,values_global):

    qs=np.linspace(0.01,1,1000)
    pdfs=q_pdf_sharp(qs,np.ones(len(qs))*m1,betaq,values_global)
    pdf_max=np.amax(pdfs)

    draw=True

    while draw:
        qtry=np.random.uniform(0.01,1)
        pdf_try=q_pdf_sharp(np.array([qtry]),np.ones(1)*m1,betaq,values_global)
        pkeep=np.random.rand()
        if pkeep*pdf_max<pdf_try:
            draw=False

    return qtry


def draw_spin_mag(nsamples,params_spins):

    chis=np.linspace(0.0001,1,1000)
    pdfs=spin_mag_pdf(chis,params_spins)
    pdf_max=np.amax(pdfs)

    

    samples=np.zeros(nsamples)

    for i in range(nsamples):

        # print(i)

        draw=True

        while draw:
            chitry=np.random.uniform(0.0001,1)
            pdf_try=spin_mag_pdf(xp.array([chitry]),params_spins)[0]
            pkeep=np.random.rand()
            if pkeep*pdf_max<pdf_try:
                draw=False

        samples[i]=chitry
        
    return samples

def draw_spin_tilts(nsamples,params_spins):

    zeta=params_spins['zeta']
    sigmat=params_spins['sigmat']

    samples=np.zeros((nsamples,2))

    for i in range(nsamples):

        pdecide=np.random.uniform(0,1)
        
        if pdecide>zeta:
            tilt1=np.random.uniform(-1,1)
            tilt2=np.random.uniform(-1,1)
        
        else:
            to_try=True
            while to_try:
                tilt1=np.random.normal(1,sigmat)
                if tilt1>-1 and tilt1<1:
                    to_try=False

            to_try=True
            while to_try:
                tilt2=np.random.normal(1,sigmat)
                if tilt2>-1 and tilt2<1:
                    to_try=False

        samples[i,0]=tilt1
        samples[i,1]=tilt2

    return samples
    


