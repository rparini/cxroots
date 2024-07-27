from typing import overload

import numpy as np
import numpy.typing as npt

from .types import AnalyticFunc, ComplexScalarOrArray, ScalarOrArray


def central_diff(
    f: AnalyticFunc,
) -> AnalyticFunc:
    h = 1e-6

    @overload
    def df(
        z: npt.NDArray[np.complex128] | npt.NDArray[np.float64],
    ) -> ComplexScalarOrArray: ...

    @overload
    def df(z: complex | float) -> complex: ...

    def df(z: ScalarOrArray) -> ComplexScalarOrArray:
        return (f(z + h) - f(z - h)) / (2 * h)

    return df
