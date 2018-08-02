Symmetric Bilinear Form
=======================

To find the distinct roots :math:`\{z_i\}_{i=1}^n` and corresponding multiplicities :math:`\{m_i\}` of the given function :math:`f(z)` within the given contour :math:`C` use is made of the fact that

.. math::
	\frac{1}{2\pi i} \int_C \phi(z)\psi(z) \frac{f'(z)}{f(z)} dz
	= \sum_{i=1}^n m_i \phi(z_i) \psi(z_i)

In `cxroots` the left hand side is computed using the function :func:`~cxroots.CountRoots.prod`.

.. autofunction:: cxroots.CountRoots.prod