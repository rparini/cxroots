from typing import Callable, Literal, Union

import numpy as np
import numpy.typing

AnalyticFunc = Callable[
    [Union[complex, numpy.typing.NDArray[np.complex128]]],
    Union[complex, numpy.typing.NDArray[np.complex128]],
]
IntegrationMethod = Literal["quad", "romb"]
