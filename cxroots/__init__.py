from .contours.annulus import Annulus
from .contours.annulus_sector import AnnulusSector
from .contours.circle import Circle
from .contours.rectangle import Rectangle
from .derivative import cx_derivative, find_multiplicity
from .root_counting import count_roots
from .root_finding import find_roots
from .root_finding_demo import demo_find_roots

# Define public interface
__all__ = [
    "Annulus",
    "AnnulusSector",
    "Circle",
    "Rectangle",
    "cx_derivative",
    "find_multiplicity",
    "count_roots",
    "find_roots",
    "demo_find_roots",
]
