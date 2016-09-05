import numpy as np
from ..simulation.sampling import rejection_sampling


__all__ = ['pdf', 'rvs']


def pdf(r, beta, alpha, eta):
    """ Computes the probability density function (pdf) of a random variable
    with Exponentiated Weibull distribution.

    Parameters
    ----------
    r : numpy.ndarray
        Support of the random variable. Must be [a,b), a > 0, b > a.
    beta : float
        Shape parameter related to the scintillation index.
    alpha : float
        Shape parameter related to the receiver aperture size. It is also the
        number of multipath scatter components at the receiver.
    eta : float
        Scale parameter that depdens on ``beta``.

    Return
    ------
    pdf : numpy.ndarray
        The expression of the pdf.
    """

    return ((alpha * beta / eta) * np.power(r / eta, beta - 1.0) *
            np.exp(- np.power(r / eta, beta)) *
            np.power(1.0 - np.exp(- np.power(r / eta, beta)), alpha - 1.0))


def rvs(K, beta, alpha, eta, inter=None):
    """ Generates ``K`` i.i.d. samples according to the Exponentiadted Weibull
    (EW) distribution using the acceptance-rejection method.

    Parameters
    ----------
    K : integer
        Number of i.i.d samples.
    beta : float
        Shape parameter related to the scintillation index.
    alpha : float
        Shape parameter related to the receiver aperture size. It is also the
        number of multipath scatter components at the receiver.
    eta : float
        Scale parameter that depdens on ``beta``.
    inter : float (optional)
        Interval on which the samples will be. Default values are ``a=1e-6``
        and ``b=10.0``.
    
    Return
    ------
    rvs : numpy.ndarray
        1-D array of with ``K`` i.i.d samples from the EW distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> from fsopy import exp_weibull
    >>> samples = exp_weibull.rvs(int(1e6), 1, 1, 1, inter=(1e-6, 4.0))
    >>> plt.hist(samples, bins=100, normed=True)
    >>> r = np.linspace(1e-6, 4., int(1e4))
    >>> pdf = exp_weibull.pdf(r, 1, 1, 1)
    >>> plt.plot(r, pdf)
    >>> plt.show()
    """

    if inter is None:
        inter = (1e-6, 10.0)

    return rejection_sampling(pdf, inter, K, beta, alpha, eta)
