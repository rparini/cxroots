from numpy import pi

from cxroots import Circle, Annulus, AnnulusSector, Rectangle

Circle(0, 2).show('circle.png')
Rectangle([-2,2],[-1,1]).show('rectangle.png')
Annulus(0, [1,2]).show('annulus.png')
AnnulusSector(0, [1,2], [0,pi]).show('annulussefig.png')