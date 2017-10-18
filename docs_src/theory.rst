Theory
======

The goal of cxroots is to compute the distinct roots :math:`\{z_i\}_{i=1}^n` and corresponding multiplicities :math:`\{m_i\}` of the given function :math:`f(z)` within the given contour :math:`C`.
cxroots starts by computing the total number of roots (counting multiplicities) :math:`N` within :math:`C`.
If :math:`N>M`, where :math:`M` is supplied by the user, then the contour is subdivided until a set of contours is obtained where the number of roots within each one is :math:`\leq M`.

The roots within each of these contours are approximated using the method of [1] which involves calculating integrals of the form

.. math::
	\frac{1}{2\pi i} \int_C \phi(z)\psi(z) \frac{f'(z)}{f(z)} dz
	= \sum_{i=1}^n m_i \phi(z_i) \psi(z_i)

where :math:`\phi`, :math:`\psi` are formal orthogonal polynomials.  
If :math:`f'(z)` is not provided by the user then it is instead approximated using a finite difference method.
These roots are then refined using the Newton-Raphson method if :math:`f'(z)` is provided and Muller's method otherwise.


References
----------
[1] Kravanja, Peter, and Marc Van Barel. "Zeros of analytic functions." Computing the Zeros of Analytic Functions. Springer Berlin Heidelberg, 2000. 1-59.