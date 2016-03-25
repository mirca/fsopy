import numpy as np
import scipy.stats as stats


__all__ = ['rejection_sampling']


def rejection_sampling(density, a, b, K, *args):
    """ Generates ``K`` samples from ``density`` on the interval ``(a,b)``
    using a (not-so-optimal) implementation of the acceptance-rejection method.

    Parameters
    ----------
    density : function
        The expression for the probability density function (pdf).
    a, b : float, float
        Interval on which the samples will be generated.
    K : integer
        Number of desired samples
    args : tuple
        Any parameters, if needed, of the pdf.

    Return
    ------
    rvs : numpy.ndarray
        1-D array containing ``K`` samples from the distribution ``density``. 
    """
    x = np.linspace(a, b, K)	
    pdf = density(x, *args)
    pdfmax = np.max(pdf)

    if np.isnan(pdfmax) or np.isinf(pdfmax):
    	return "error(rejection_sampling): pdfmax is NaN."
    
    K_min = 0
    rvs = np.zeros(1)
    while K_min < K+1:
    	    u1 = stats.uniform.rvs(loc=a,scale=b-a,size=K)
    	    u2 = stats.uniform.rvs(size=K)
    	    idx = np.where(u2<=density(u1,*args)/pdfmax)[0]
    	    rvs = np.concatenate([rvs,u1[idx]])
    	    K_min = np.size(rvs)
	
    rvs = np.reshape(rvs,[K_min,1])
    return rvs[1:K+1]
