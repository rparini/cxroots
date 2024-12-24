from typing import overload

import numpy as np
import numpy.typing as npt

from .types import AnalyticFunc, ComplexScalarOrArray, ScalarOrArray


def approx_deriv(
    f: AnalyticFunc,
) -> AnalyticFunc:
    h = 1e-5

    @overload
    def df(
        z: npt.NDArray[np.complexfloating] | npt.NDArray[np.floating],
    ) -> ComplexScalarOrArray: ...

    @overload
    def df(z: complex | float) -> complex: ...

    def df(z: ScalarOrArray) -> ComplexScalarOrArray:
        return (-f(z + 2 * h) + 8 * f(z + h) - 8 * f(z - h) + f(z - 2 * h)) / (12 * h)

    return df
