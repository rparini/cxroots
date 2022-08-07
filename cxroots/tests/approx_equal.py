import numpy as np


def sets_approx_equal(a, b, decimal=10):
    """
    Test if iterables a and b are approximately the same, up to reordering.

    Works by greedily reordering a and b so that they are the closest to
    each other element-wise and then using np.testing.assert_almost_equal

    .. np.testing.assert_almost_equal:
       https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html
    """

    a, b = np.array(a), np.array(b)

    # greedily reorder b so that it is closest to a element-wise
    for ai, a_element in np.ndenumerate(a):
        # find the element of b closest to a_element
        bi = np.argmin(np.abs(b - a_element))
        # put the element of b closest to a_element in the same place as a_element
        b[ai], b[bi] = b[bi], b[ai]

    # use numpy to compare the two arrays element-wise
    return np.testing.assert_almost_equal(a, b, decimal)


def roots_approx_equal(a, b, decimal=10):
    a_roots, a_multiplicities = a
    b_roots, b_multiplicities = b

    a_roots, b_roots = np.array(a_roots), np.array(b_roots)
    a_multiplicities, b_multiplicities = (
        np.array(a_multiplicities),
        np.array(b_multiplicities),
    )

    for ai, a_element in np.ndenumerate(a_roots):
        bi = np.argmin(np.abs(b_roots - a_element))
        b_roots[ai], b_roots[bi] = b_roots[bi], b_roots[ai]
        b_multiplicities[ai], b_multiplicities[bi] = (
            b_multiplicities[bi],
            b_multiplicities[ai],
        )

    np.testing.assert_almost_equal(a_roots, b_roots, decimal)
    np.testing.assert_almost_equal(a_multiplicities, b_multiplicities, decimal)
