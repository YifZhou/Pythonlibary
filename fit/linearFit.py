import numpy as np


def linearFit(x, y, dy=None):
    if dy is None:
        dy = np.ones(len(y))
    """
    my own linear fit routine, since there is no good scipy or numpy linearFit routine written up
    """
    Y = np.mat(y).T
    A = np.mat([np.ones(len(x)), x]).T
    C = np.mat(np.diagflat(dy**2))
    mat1 = (A.T * C**(-1) * A)**(-1)
    mat2 = A.T * C**(-1) * Y
    b, m = mat1 * mat2
    b = b.flat[0]
    m = m.flat[0]
    sigb, sigm = np.sqrt(np.diag(mat1))
    chisq = 1. / (len(x) - 2) * np.sum((y - m * x - b)**2 / dy**2)
    return b, m, sigb, sigm, chisq
