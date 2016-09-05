import numpy as np
import math
import scipy.special as sps
from scipy.integrate import quad
from scipy.special import gamma, kv
from ..utils.marcumq import marcumQ
from ..utils.progressbar import printProgress

FADINGS = ['exp_weibull', 'gamma_gamma']

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
    Pm = 0.0

    printProgress(0, n_terms, prefix='Progress', suffix='Complete', barLength=50)

    if fading == 'exp_weibull':
        beta, alpha, eta = args[0:3]        
        roots, weights = sps.orthogonal.la_roots(n_terms, 0.0)
 
        for k in range(n_terms):
            Pm = Pm + (weights[k] * (1 - math.exp(-roots[k]))**(alpha - 1))*(1 - marcumQ(math.sqrt(n_samples * Es * (eta * roots[k]**(1./beta))**2 / var_w), np.sqrt(thresholds / var_w), n_samples / 2.0))
            printProgress(k, n_terms-1, prefix='Progress', suffix='Complete', barLength=50)

        Pm = alpha*Pm

    elif fading == 'gamma_gamma':
        beta, alpha = args[0:2]
        roots, weights = sps.orthogonal.la_roots(n_terms, 0.5*(alpha + beta))

        for k in range(n_terms):
            Pm = Pm + weights[k] * math.exp(roots[k]) * kv(alpha - beta, 2 * math.sqrt(alpha * beta * roots[k])) * (1 - marcumQ(roots[k] * math.sqrt(n_samples * Es /var_w), np.sqrt(thresholds / var_w), n_samples / 2.0)) 
            printProgress(k, n_terms-1, prefix='Progress', suffix='Complete', barLength=50)

        Pm = Pm * 2 * (alpha * beta)**(0.5 * (alpha + beta)) / (gamma(alpha) * gamma(beta))

    return Pf, Pm

def th_roc_num(mod_order, snr_db, n_samples, n_thresh, fading, *args):
    """
    Computes the theorectical CROC using the scipy numerical integration
    library.

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

    printProgress(0, n_thresh, prefix='Progress', suffix='Complete', barLength=50)        
    if fading == 'exp_weibull':
        beta, alpha, eta = args[0:3]

        for k in range(n_thresh):
            integrand = lambda u: (alpha*math.exp(-u)*(1 - math.exp(-u))**(alpha-1)) * (1 - marcumQ(math.sqrt(n_samples*Es*(eta*u**(1./beta))**2/var_w), math.sqrt(thresholds[k]/var_w), n_samples/2.0))
            Pm[k] = quad(integrand, 0.0, np.inf, epsrel=1e-9, epsabs=0)[0]
            printProgress(k, n_thresh-1, prefix='Progress', suffix='Complete', barLength=50)

    elif fading == 'gamma_gamma':
        beta, alpha = args[0:2]

        for k in range(n_thresh):
            integrand = lambda r: r**(0.5 * (alpha + beta)) * kv(alpha - beta, 2 * math.sqrt(alpha * beta * r)) * (1 - marcumQ(r * math.sqrt(n_samples * Es / var_w), np.sqrt(thresholds[k] / var_w), n_samples / 2.0))
            Pm[k] = quad(integrand, 0.0, np.inf, epsrel=1e-9, epsabs=0)[0] * 2 * (alpha * beta)**(0.5 * (alpha + beta)) / (gamma(alpha) * gamma(beta))
            printProgress(k, n_thresh-1, prefix='Progress', suffix='Complete', barLength=50)

    return Pf, Pm
