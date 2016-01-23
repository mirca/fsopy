import numpy as np

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
    The expression of the pdf.

    Examples
    --------
    TODO
    """
    return (alpha * beta / eta) * np.power(r / eta, beta - 1.0)*
           np.exp(- np.power(r / eta, beta)) *
           np.power(1.0 - np.exp(- np.power(r / eta, beta)), alpha - 1.0)


def rvs(beta, alpha, eta, K):
    # TODO
