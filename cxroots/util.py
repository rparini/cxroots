from typing import Callable

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


def integrate_quad_complex(
    func: Callable[[float], complex], *args, **kwargs
) -> complex:
    """
    A thin wrapper around scipy.integrate.quad that copes
    with the integrand returning complex values
    """
    # full_output=0 ensures only 2 values returned
    integral_real, _ = scipy.integrate.quad(
        lambda t: np.real(func(t)), *args, full_output=0, **kwargs
    )
    integral_imag, _ = scipy.integrate.quad(
        lambda t: np.imag(func(t)), *args, full_output=0, **kwargs
    )
    integral = integral_real + 1j * integral_imag
    return integral
