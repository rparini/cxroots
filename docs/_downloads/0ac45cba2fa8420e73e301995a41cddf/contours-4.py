from numpy import pi
from cxroots import AnnulusSector
annulusSector = AnnulusSector(
    center=0.2, radii=(0.5, 1.25), phi_range=(-pi/4, pi/4)
)
annulusSector.show()