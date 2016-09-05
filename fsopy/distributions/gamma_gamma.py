import numpy as np
from scipy.special import kv, gamma
from ..simulation.sampling import rejection_sampling


__all__ = ['pdf', 'rvs']


def pdf(r, beta, alpha):
    """ Computes the probability density function (pdf) of a random variable
    with Gamma Gamma distribution.

    Parameters
    ----------
    r : numpy.ndarray
        Support of the random variable. Must be [a,b), a > 0, b > a.
    beta : float
        Shape parameter related to the small-scale scintilation.
    alpha : float
        Shape parameter related to the large-scale scintilation.

    Return
    ------
    pdf : numpy.ndarray
        The expression of the pdf.
    """

    return 2 * (alpha * beta)**((alpha + beta) / 2) * np.power(r, (alpha + beta)/2) * kv(alpha - beta, 2 * np.sqrt(alpha * beta * r)) / (gamma(alpha) * gamma(beta))

def rvs(K, beta, alpha, inter=None):
    """ Generates ``K`` i.i.d. samples according to the Gamma Gamma
    (GG) distribution using the acceptance-rejection method.

    Parameters
    ----------
    K : integer
        Number of i.i.d samples.
    beta : float
        Shape parameter related to the small-scale scintilation.
    alpha : float
        Shape parameter related to the large-scale scintilation.
    inter : float (optional)
        Interval on which the samples will be. Default values are ``a=1e-6``
        and ``b=10.0``.
    
    Return
    ------
    rvs : numpy.ndarray
        1-D array of with ``K`` i.i.d samples from the Gamma Gamma distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from fsopy.distributions import gamma_gamma
    >>> samples = gamma_gamma.rvs(int(1e6), 1, 1, inter=(1e-6, 4.0))
    >>> plt.hist(samples, bins=100, normed=True)
    >>> r = np.linspace(1e-6, 4., int(1e4))
    >>> pdf = gamma_gamma.pdf(r, 1, 1)
    >>> plt.plot(r, pdf)
    >>> plt.show()
    """

    if inter is None:
        inter = (1e-6, 10.0)

    return rejection_sampling(pdf, inter, K, beta, alpha)
