Contours
========

The cxroots module allows the user to specify four different types of contours which are all subclasses of :class:`~cxroots.contour.Contour`:

* :ref:`contours:Circle`
* :ref:`contours:Rectangle`
* :ref:`contours:Annulus`
* :ref:`contours:Annulus Sector`

.. autoclass:: cxroots.contour.Contour
	:members: __call__, plot, show, subdivisions, distance, integrate, count_roots, roots, demo_roots, contains

Circle
------
.. autoclass:: cxroots.Circle

Rectangle
---------
.. autoclass:: cxroots.Rectangle

Annulus
-------
.. autoclass:: cxroots.Annulus

Annulus Sector
--------------
.. autoclass:: cxroots.AnnulusSector
