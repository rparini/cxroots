Contours
========
cxroots finds all the roots an analytic function which lie in the interior of a user specified contour.  
The posibilities for choosing this contour are:

* :ref:`Circle`
* :ref:`Rectangle`
* :ref:`Annulus`
* :ref:`Annulus Sector`

Contours can be plotted using:

.. automethod:: cxroots.Contours.Contour.show

Circle
------
.. autoclass:: cxroots.Circle

.. plot:: docex_contour_circle.py
   :include-source:

Rectangle
---------
.. autoclass:: cxroots.Rectangle

.. plot:: docex_contour_rectangle.py
	:include-source:

Annulus
-------
.. autoclass:: cxroots.Annulus

.. plot:: docex_contour_annulus.py
   :include-source:

Annulus Sector
--------------
.. autoclass:: cxroots.AnnulusSector

.. plot:: docex_contour_annulussector_a.py
   :include-source:

.. plot:: docex_contour_annulussector_b.py
   :include-source:
