import numpy as np
import scipy.integrate
from numpydoc.docscrape import FunctionDoc


def remove_para(*paras):
    def wrapper(func):
        doc = FunctionDoc(func)
        for i, p in enumerate(doc["Parameters"][:]):
            if p.name.split(":")[0].rstrip() in paras:
                del doc["Parameters"][i]
        func.__doc__ = doc
        return func

    return wrapper


def update_docstring(**dic):
    def wrapper(func):
        doc = FunctionDoc(func)
        for k, v in dic.items():
            doc[k] = v
        func.__doc__ = doc
        return func

    return wrapper


class NumberOfRootsChanged(Exception):
    pass


def integrate_quad_complex(func, *args, **kwargs):
    """
    A thin wrapper around scipy.integrate.quad that copes
    with the integrand returning complex values
    """

    def integrand_real(t):
        return np.real(func(t))

    def integrand_imag(t):
        return np.imag(func(t))

    # full_output=0 ensures only 2 values returned
    integral_real, abserr_real = scipy.integrate.quad(
        integrand_real, full_output=0, *args, **kwargs
    )
    integral_imag, abserr_imag = scipy.integrate.quad(
        integrand_imag, full_output=0, *args, **kwargs
    )
    integral = integral_real + 1j * integral_imag
    err = abserr_real + 1j * abserr_imag
    return integral, err
