from typing import Literal, Protocol, Union, overload

import numpy as np
import numpy.typing as npt

IntegrationMethod = Literal["quad", "romb"]
Color = Union[str, tuple[float, float, float], tuple[float, float, float, float]]  # noqa: UP007
ScalarOrArray = Union[  # noqa: UP007
    complex, float, npt.NDArray[np.complex128], npt.NDArray[np.float64]
]
ComplexScalarOrArray = Union[complex, npt.NDArray[np.complex128]]  # noqa: UP007


class AnalyticFunc(Protocol):
    @overload
    def __call__(self, z: complex | float) -> complex: ...

    @overload
    def __call__(
        self, z: npt.NDArray[np.complex128] | npt.NDArray[np.float64]
    ) -> ComplexScalarOrArray:
        # Note that the function may return a scalar in this case if, for example,
        # it's a constant function
        ...

    def __call__(self, z: ScalarOrArray) -> ComplexScalarOrArray: ...
