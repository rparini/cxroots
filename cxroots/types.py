from typing import Literal, Protocol, Tuple, Union, overload

import numpy as np
import numpy.typing as npt

IntegrationMethod = Literal["quad", "romb"]
Color = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
ScalarOrArray = Union[complex, float, npt.NDArray[np.complex_], npt.NDArray[np.float_]]
ComplexScalarOrArray = Union[complex, npt.NDArray[np.complex_]]


class AnalyticFunc(Protocol):
    @overload
    def __call__(self, z: Union[complex, float]) -> complex:
        ...

    @overload
    def __call__(
        self, z: Union[npt.NDArray[np.complex_], npt.NDArray[np.float_]]
    ) -> ComplexScalarOrArray:
        # Note that the function may return a scalar in this case if, for example,
        # it's a constant function
        ...

    def __call__(self, z: ScalarOrArray) -> ComplexScalarOrArray:
        ...
