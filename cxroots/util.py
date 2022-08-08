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
    # full_output=0 ensures only 2 values returned
    integral_re, abserr_re = scipy.integrate.quad(
        func=lambda t: np.real(func(t)), full_output=0, *args, **kwargs
    )
    integral_im, abserr_im = scipy.integrate.quad(
        func=lambda t: np.imag(func(t)), full_output=0, *args, **kwargs
    )
    integral = integral_re + 1j * integral_im
    err = abserr_re + 1j * abserr_im
    return integral, err
