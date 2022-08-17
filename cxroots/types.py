from typing import Literal, Protocol, Union, overload

import numpy as np
import numpy.typing as npt

IntegrationMethod = Literal["quad", "romb"]


class AnalyticFunc(Protocol):
    @overload
    def __call__(self, z: Union[complex, float]) -> complex:
        ...

    @overload
    def __call__(
        self, z: Union[npt.NDArray[np.complex_], npt.NDArray[np.float_]]
    ) -> npt.NDArray[np.complex_]:
        ...

    def __call__(
        self, z: Union[complex, float, npt.NDArray[np.complex_], npt.NDArray[np.float_]]
    ) -> Union[complex, npt.NDArray[np.complex_]]:
        ...
