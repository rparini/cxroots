from math import factorial, pi
from typing import Optional, Union, overload

import numpy as np
import numpy.typing as npt

from .types import AnalyticFunc, ComplexScalarOrArray, ScalarOrArray


def central_diff(
    f: AnalyticFunc,
) -> AnalyticFunc:
    h = 5e-6

    @overload
    def df(
        z: Union[npt.NDArray[np.complex_], npt.NDArray[np.float_]]
    ) -> ComplexScalarOrArray:
        ...

    @overload
    def df(z: Union[complex, float]) -> complex:
        ...

    def df(z: ScalarOrArray) -> ComplexScalarOrArray:
        return (f(z + h) - f(z - h)) / (2 * h)

    return df
