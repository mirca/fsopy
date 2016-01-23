import numpy as np
#TODO from ..simulation.rejection_sampling import rejection_sampling


__future__ = ['rvs']
__all__ = ['pdf']


def pdf(r, beta, alpha, eta):
    """ Computes the probability density function (pdf) of a random variable
    with Exponentiated Weibull distribution.

    Parameters
    ----------
    r : numpy.ndarray
        Support of the random variable. Must be [a,b), a > 0.
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

    Examples
    --------
    TODO
    """
    return (alpha * beta / eta) * np.power(r / eta, beta - 1.0) * np.exp(- np.power(r / eta, beta)) * np.power(1.0 - np.exp(- np.power(r / eta, beta)), alpha - 1.0)


def rvs(K, beta, alpha, eta, a=1e-6, b=10.0):
    """ Generates ``K`` i.i.d. samples according to the Exponentiadted Weibull
    (EW) distribution using the acceptance-rejection method.

    Parameters
    ----------
    beta : float
        Shape parameter related to the scintillation index.
    alpha : float
        Shape parameter related to the receiver aperture size. It is also the
        number of multipath scatter components at the receiver.
    eta : float
        Scale parameter that depdens on ``beta``.
    K : integer
        Number of i.i.d samples.
    a, b : float, float, optional, optional
        Interval on which the samples will be. Default values are ``a=1e-6``
        and ``b=10.0``.
    
    Return
    ------
    rvs : numpy.ndarray
        1-D array of with ``K`` i.i.d samples from the EW distribution.

    Examples
    --------
    TODO
    """
    return rejection_sampling(pdf, a, b, K, beta, alpha, eta)
