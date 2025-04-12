## Changelog

### 3.2.0 - 12/Apr/2025

- Support scipy 1.15
- Removes romberg integration option from contour `integrate` method

### 3.1.0 - 2/Feb/2025

- Changes derivative approximation to use 1D five-point sencil
- Pin scipy<1.15 due to upcoming `scipy.integrate.romberg` removal
- Change project to use pyproject.toml
- Set __version__ using setuptools-scm

### 3.0.2 - 27/Jul/2024

- Adds compatibility with numpy v2 by replacing deprecated types (thanks [@MartinCapraro](https://github.com/MartinCapraro))

### 3.0.1 - 23/Jul/2024

- Temporarily require numpy<2

### 3.0.0 - 23/Jul/2024

- Requires Python >= 3.10
- Removed `df_approx_order` parameter
- Removed `cx_derivative` function
- Removed `find_multiplicity` function
- Removed dependency on numdifftools
- Removed upper bound on required numpy version
- Removed `util.integrate_quad_complex` and use `scipy.integrate.quad`'s  `complex_func` argument instead
- Renamed `NumberOfRootsChanged` to `NumberOfRootsChangedError`
- Switch to [ruff](https://github.com/astral-sh/ruff) formatting/linting

### 2.1.0 - 27/Dec/2023

- Improved reliability by subdividing contours that produce invalid approximations for multiplicites, rather than ignoring the individual invalid approximations.

### 2.0.1 - 29/Aug/2023

- Record in setup.py that only numpy < 1.25 will work with cxroots currently (https://github.com/rparini/cxroots/issues/285) 
- Fix type annotation for `util.integrate_quad_complex` function
- Add more debug logging for root counting
- Adjust root counting error advice to include chaning integration method
- Fix type issue in contour.__call__ by hard coding array dtype

### 2.0.0 - 20/Aug/2022

#### Breaking changes

- Drop support for Python 2. Cxroots now requires python 3.8 or later
- The `verbose` argument has been removed from some functions. It still exists for the `Contour.roots` method and `find_roots` function but it will now create a progress bar, using [rich](https://github.com/Textualize/rich), rather than printing debugging information to the console.
- Remove unused `Contour.randomPoint` method
- All `camelCase` functions and arguments changed to be `snake_case`
- `m` argument renamed to `df_approx_order`
- Renames `attempt_best` argument to `refine_roots_beyond_tol`
- Removed `return_animation` argument from `demo_find_roots` function and `Contour.demo_roots` method. Instead, the `demo_roots_animation` function or `Contour.demo_roots_animation` method can be used to get a `animation.FuncAnimation` object that would animate the rootfinding process without displaying it.
- Renames `RootResult.original_contour` attribute to `RootResult.contour`
- The `Contour._size_plot` method was renamed to `Contour.size_plot` and given a docstring
- Removed `Contour.approximate_roots` method as it is intended for users to call `Contour.roots` instead
- The `count_roots`, `find_roots` and `demo_find_roots` are no longer exposed directly by importing cxroots. Instead, it is intended that the contour method is used instead. For example, use `Circle(0,3).find_roots(f)` instead of `find_roots(Circle(0,3),f)`

#### New features

- cxroots now logs rootfinding progress which can be accessed using the standard library's [logging module](https://docs.python.org/3/library/logging.html). See the [documentation](https://rparini.github.io/cxroots/logging.html) for examples.
- Add `cxroots[plot]` install option that will install dependencies for plotting contours and roots
- Added type annotations for all public interfaces

#### Minor changes and fixes

- Use [Black](https://github.com/psf/black) formatting and added pre-commit hook
- Contour arrows to are now scale-independent ([#153](https://github.com/rparini/cxroots/issues/153), thanks [@llohse](https://github.com/llohse))
- Renamed internal files to camel_case.py
- Warnings from `scipy.integrate.quad` are no longer suppressed by cxroots while calculating the bilinear product
- Changed default absolute and relative integration tolernaces to 1.49e-08 to match scipy's defaults for `scipy.integrate.quad` and `scipy.integrate.romberg`
- Fixed issue with `newton` iteration method when `refine_roots_beyond_tol` was True and the routine would not exit if the error of the previous and current iterations were only equal
- The `callback` for the `muller` iteration method will now correctly be passed the value of the evaluated function for the iteration, rather than the error.
- Fixed description of `root_tol` and `refine_roots_beyond_tol` in `iterate_to_root` docstring
- Changes default `root_tol` to 0 for `secant`, `newton` and `muller` functions
- Changes starting points for muller's method used when root refining to be complex, to guard against the iterations keeping to the real line.

### 1.1.11 - 22/Dec/2021

- Fixed error when using `romb` integration method when supplied with a derivative function that returns a constant value

### 1.1.10 - 22/Jul/2020

- Replaced docrep with numpydoc

### 1.1.9 - 1/Mar/2020

- Fix depreciation warning from scipy

### 1.1.8 - 20/Apr/2019

- Fix NotImplementedError in Contour base ([#31](https://github.com/rparini/cxroots/issues/31), thanks [@fchapoton](https://github.com/fchapoton))
- Added `__call__` placeholder to ComplexPath

### 1.1.7 - 10/Feb/2019

- Fix `setup.py test`.
- Fix invalid escape characters

### 1.1.6 - 19/Sep/2018

- Added example to index page of documentation.
- Removed unnecessary imports.
- Enable `setup.py test`.

### 1.1.5 - 5/Aug/2018

- Got project description working on PyPi.

### 1.1.0 - 5/Aug/2018

- New tests and converted old tests to pytest.
- Documentation overhauled to include separate tutorial, user guide and reference sections. In the process added/improved many docstrings.
- Function capitalisation now consistent.
- Changed AnnulusSector's rRange -> radii to match Annulus.
- Contours which have some known roots and unknown roots in them will be subdivided to avoid calculating already known roots.
- If the number of roots (counting multiplicities) in a contour does not match the sum of the number of roots in its subdivided contours then the number of roots is recalculated in both the parent and child contours with NIntAbsTol set to half the provided value.
- Romberg integration now requires the last two consecutive iterations (instead of just the last) to satisfy the prescribed error tolerance before the last iteration is accepted as the result.
- If Romberg integration is used then the function evaluations are cached and the number of roots within a contour checked while the roots within the contour are being calculated.
- Introduced parameter m which is only used if df=None and method='quad'. The argument order=m is the order of the error term in the approximation for the derivative used during contour integration.
- Implemented methods to determine the distance between a point and the closest point on a line or circular arc in the complex plane.
- If a proposed subdivision introduces a contour with a minimum distance of < 0.01 from any known root then it is discarded and a new subdivision chosen.
- Exposes the parameter errStop to the user. The number of distinct roots within a contour, n, is determined by checking if all the elements of a list of contour integrals involving formal orthogonal polynomials are sufficently close to zero, ie. that the absolute value of each element is < errStop. If errStop is too large/small then n may be smaller/larger than it actually is.
- Computing multiplicities now reuses integrals used to calculate distinct roots.
- If the computed multiplicity for a root is not sufficiently close to an integer (as determined by integerTol) then the multiplicity will be determined again by computing the roots within a small circle centred on the root.
- Final multiplicities are now returned as integers, rather than rounded floats.
- Muller’s method rather than Newton-Raphson iteration if df is not provided.
- Added attemptIterBest parameter. If True then the iterative method used to refine the roots will exit when error of the previous iteration, x0, was at least as good as the current iteration, x, in the sense that abs(f(x)) >= abs(f(x0)) and the previous iteration satisfied abs(dx0) < newtonStepTol. In this case the previous iteration is returned as the approximation of the root. The idea is that it will attempt to get the best accuracy it can at the expense of more function evaluations.
- RootResult now a subclass of namedtuple()

### 1.0.0 - 12/Oct/2017

Initial release version.
