from numpy import pi
from cxroots import AnnulusSector
center = 0.2
r = [0.5, 1.25]
phi = [pi/4, -pi/4]
annulusSector = AnnulusSector(center, r, phi)
annulusSector.show()