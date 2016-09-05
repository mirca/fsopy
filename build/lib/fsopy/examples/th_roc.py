import numpy as np
import math
import scipy.special as sps
from scipy.integrate import quad
from ..utils.marcumq import marcumQ

FADINGS = ['exp_weibull']

__all__ = ['th_roc_glq', 'th_roc_num']

def th_roc_glq(mod_order, snr_db, n_samples, n_thresh, n_terms, fading, *args):
    """
    Computes the theorectical CROC using the Gauss-Laguerre quadrature.

    Parameters
    ----------
    mod_order : int
        Modulation order.
    snr_db : float
        Signal-to-noise ratio in dB.
    n_samples : int
        Number of transmitted symbols.
    n_thresh : int
        Number of thresholds to be evaluated.
    n_terms : int
        Number of terms for the Gauss-Laguerre quadrature.
    fading : str
        Name of the fading.
    args : array-like
        Fading parameters.
    """

    if fading not in FADINGS:
        raise NotImplementedError('the formulations for this fading is not'
                                  ' implemented yet.')

    thresholds = np.linspace(.0, 100.0, n_thresh)
    
    # symbol energy
    Es = 1./mod_order
    # noise variance
    var_w = Es*sps.exp10(-snr_db/10.)

    Pf = 1 - sps.gammainc(n_samples/2., thresholds/(2*var_w))
    Pm = np.zeros(n_thresh)

    # Gauss-Laguerre quadrature
    glq = 0.0

    if fading == 'exp_weibull':
        beta, alpha, eta = args[0:3]
        roots, weights = sps.orthogonal.la_roots(n_terms, 0.0)

        for k in range(n_terms):
            glq = (glq + weights[k]*(1 - math.exp(-roots[k])**(alpha-1))*
                   (1 - marcumQ(math.sqrt(2*n_samples*Es*(eta*roots[k]**(1./beta))**2/var_w),
                                np.sqrt(2*thresholds/var_w),
                                n_samples)))
        Pm *= alpha

    return Pf, Pm

def th_roc_num(mod_order, snr_db, n_samples, n_thresh, fading, *args):
    """
    Computes the theorectical CROC using the Gauss-Laguerre quadrature.

    Parameters
    ----------
    mod_order : int
        Modulation order.
    snr_db : float
        Signal-to-noise ratio in dB.
    n_samples : int
        Number of transmitted symbols.
    n_thresh : int
        Number of thresholds to be evaluated.
    n_terms : int
        Number of terms for the Gauss-Laguerre quadrature.
    fading : str
        Name of the fading.
    args : array-like
        Fading parameters.
    """

    if fading not in FADINGS:
        raise NotImplementedError('the formulations for this fading is not'
                                  ' implemented yet.')

    thresholds = np.linspace(.0, 100.0, n_thresh)
    
    # symbol energy
    Es = 1./mod_order
    # noise variance
    var_w = Es*sps.exp10(-snr_db/10.)

    Pf = 1 - sps.gammainc(n_samples/2., thresholds/(2*var_w))
    Pm = np.zeros(n_thresh)

    if fading == 'exp_weibull':
        beta, alpha, eta = args[0:3]
        for k in range(n_thresh):
            integrand = lambda u: (alpha*math.exp(-u)*(1 - math.exp(-u)**(alpha-1))*
                                   (1 - marcumQ(math.sqrt(2*n_samples*Es*(eta*u**(1./beta))**2/var_w),
                                                math.sqrt(2*thresholds[k]/var_w),
                                                n_samples)))
            Pm[k] = quad(integrand, 0.0, np.inf, epsrel=1e-9, epsabs=0)[0]

    return Pf, Pm

