import numpy as np
import scipy.special as sps
from scipy.integrate import quad
from ..utils.marcumq import marcumQ

FADINGS = ['exp_weibull']

def th_roc_glq(mod_order, snr_db, n_samples, n_thresh, n_terms, fading, *args):
    """
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
    Pd = np.zeros(L)

    # Gauss-Laguerre quadrature
    glq = 0.0

    if fading == 'exp_weibull':
        beta, alpha, eta = args[0:2]
        roots, weights = sps.orthogonal.la_roots(n_terms, 0.0)
        cond_cdf = 1 - marcumQ()

        for n in range(n_terms):
            glq = glq + weights[n]

