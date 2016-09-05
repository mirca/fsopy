import scipy.stats

__all__ = ['marcumQ']

def marcumQ(a, b, M):
    """
    See Wikipedia's page
    https://en.wikipedia.org/wiki/Marcum_Q-function
    """
    
    return scipy.stats.ncx2.sf(b*b, 2*M, a*a)
