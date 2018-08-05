Contours
========

The cxroots module allows the user to specify four different types of contours which are all subclasses of :class:`~cxroots.Contour.Contour`:

* :ref:`Circle`
* :ref:`Rectangle`
* :ref:`Annulus`
* :ref:`Annulus Sector`

.. autoclass:: cxroots.Contour.Contour
	:members: __call__, plot, show, subdivisions, distance, integrate, count_roots, approximate_roots, roots, demo_roots, contains

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
