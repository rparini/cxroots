from .contours.circle import Circle
from .contours.annulus import Annulus
from .contours.annulus_sector import AnnulusSector
from .contours.rectangle import Rectangle

from .root_finding import find_roots
from .root_finding_demo import demo_find_roots
from .root_counting import count_roots
from .derivative import cx_derivative
from .derivative import find_multiplicity

# Define public interface
__all__ = [
    "Circle",
    "Annulus",
    "AnnulusSector",
    "Rectangle",
    "count_roots",
    "find_roots",
    "demo_find_roots",
    "find_multiplicity",
    "cx_derivative",
]
