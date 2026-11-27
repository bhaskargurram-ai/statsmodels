"""
This file contains analytic implementations of rotation methods.
"""

import numpy as np
import scipy as sp


def target_rotation(A, H, full_rank=False):
    r"""
    Analytically performs orthogonal rotations towards a target matrix,
    i.e., we minimize:

    .. math::
        \phi(L) =\frac{1}{2}\|AT-H\|^2.

    where :math:`T` is an orthogonal matrix. This problem is also known as
    an orthogonal Procrustes problem.

    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:

    .. math::
        T = (A^*HH^*A)^{-\frac{1}{2}}A^*H,

    see Green (1952). In other cases the solution is given by :math:`T = UV`,
    where :math:`U` and :math:`V` result from the singular value decomposition
    of :math:`A^*H`:

    .. math::
        A^*H = U\Sigma V,

    see Schonemann (1966).

    Parameters
    ----------
    A : ndarray
        Non-rotated factors.
    H : ndarray
        Target matrix.
    full_rank : bool, optional
        If set to True full rank is assumed. Default is False.

    Returns
    -------
    ndarray
        The rotation matrix :math:`T`.

    References
    ----------
    [1] Green (1952, Psychometrika) - The orthogonal approximation of an
    oblique structure in factor analysis

    [2] Schonemann (1966) - A generalized solution of the orthogonal
    procrustes problem

    [3] Gower, Dijksterhuis (2004) - Procrustes problems
    """
    ATH = A.T.dot(H)
    if full_rank or np.linalg.matrix_rank(ATH) == A.shape[1]:
        T = sp.linalg.fractional_matrix_power(ATH.dot(ATH.T), -1/2).dot(ATH)
    else:
        U, D, V = np.linalg.svd(ATH, full_matrices=False)
        T = U.dot(V)
    return T


def procrustes(A, H):
    r"""
    Analytically solves the following Procrustes problem:

    .. math::
        \phi(L) =\frac{1}{2}\|AT-H\|^2.

    (With no further conditions on :math:`H`)

    Under the assumption that :math:`A^*H` has full rank, the analytical
    solution :math:`T` is given by:

    .. math::
        T = (A^*HH^*A)^{-\frac{1}{2}}A^*H,

    see Navarra, Simoncini (2010).

    Parameters
    ----------
    A : ndarray
        Non-rotated factors.
    H : ndarray
        Target matrix.

    Returns
    -------
    ndarray
        The rotation matrix :math:`T`.

    References
    ----------
    [1] Navarra, Simoncini (2010) - A guide to empirical orthogonal functions
    for climate data analysis
    """
    return np.linalg.inv(A.T.dot(A)).dot(A.T).dot(H)


def promax(A, k=4, normalize=True):
    r"""
    Performs promax rotation of the matrix :math:`A`.

    Promax rotation (Hendrickson and White, 1964) is performed in the
    following steps:

    * Determine varimax rotated loadings :math:`V`, with Kaiser normalization
      if ``normalize`` is True.

    * Construct the target matrix :math:`H` with elements
      :math:`V_{ij} |V_{ij}|^{k-1}`.

    * Regress the target on :math:`V` to obtain :math:`U` (least squares
      Procrustes) and scale the columns of :math:`U` so that the factors
      have unit variance.

    * The rotated loadings are the pattern matrix :math:`L = VU`.

    This is the algorithm of ``promax`` in R's stats package, which
    ``factanal`` uses, with the same default power ``k=4``.

    Parameters
    ----------
    A : ndarray
        Non-rotated factors.
    k : float, optional
        Power of the target, should be larger than 1. The default is 4.
    normalize : bool, optional
        If True (the default), the varimax rotation uses Kaiser
        normalization, the rows of :math:`A` are scaled to unit length
        before the rotation and scaled back afterwards.

    Returns
    -------
    ndarray
        The rotated factors, the pattern matrix :math:`L`.
    ndarray
        The rotation matrix :math:`T`, such that :math:`L = A(T^*)^{-1}` and
        the correlation of the rotated factors is :math:`T^*T`, as for the
        oblique rotations of ``rotate_factors``.

    References
    ----------
    [1] Hendrickson, White (1964) - Promax: A quick method for rotation to
    oblique simple structure

    [2] Browne (2001) - An overview of analytic rotation in exploratory
    factor analysis
    """
    if k <= 1:
        raise ValueError("k must be larger than 1")
    A = np.asarray(A, dtype=np.float64)
    # Varimax rotation, V = A R with R orthogonal
    from ._wrappers import rotate_factors

    if normalize:
        h = np.sqrt((A**2).sum(1))[:, None]
        h[h == 0] = 1
        V, R = rotate_factors(A / h, "varimax")
        V = V * h
    else:
        V, R = rotate_factors(A, "varimax")
    # Target and least squares rotation towards it
    H = V * np.abs(V) ** (k - 1)
    U = procrustes(V, H)
    # Scale the columns so that the factors have unit variance
    d = np.sqrt(np.diag(np.linalg.inv(U.T.dot(U))))
    U = U * d
    L = V.dot(U)
    # L = A R U = A (T^*)^{-1}
    T = np.linalg.inv(R.dot(U)).T
    return L, T
