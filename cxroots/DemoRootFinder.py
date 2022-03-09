from __future__ import division

import numpy as np

from .RootFinder import find_roots_gen

def demo_find_roots(originalContour, f, df=None, saveFile=None, automaticAnim=False,
	returnAnim=False, writer=None, **rootsKwargs):
	"""
	An animated demonstration of the root finding process using matplotlib.

	Parameters
	----------
	saveFile : str, optional
		If given then the animation will be saved to disk with filename
		equal to saveFile instead of being shown.
	automaticAnim : bool, optional
		If False (default) then press SPACE to step the animation forward
		If True then the animation will play automatically until all the
		roots have been found.
	returnAnim : bool, optional
		If True then the matplotlib animation object will be returned
		instead of being shown.  Defaults to False.
	writer : str, optional
		Passed to :meth:`matplotlib.animation.FuncAnimation.save`.
	**rootsKwargs
		Additional key word arguments passed to :meth:`~cxroots.Contour.Contour.roots`.
	"""
	import matplotlib.pyplot as plt
	from matplotlib import animation
	fig = plt.gcf()
	ax = plt.gca()

	rootFinder = find_roots_gen(originalContour, f, df, **rootsKwargs)

	def init():
		originalContour.plot(linecolor='k', linestyle='--')
		originalContour._sizePlot()

	def update_frame(args):
		roots, multiplicities, boxes, numberOfRemainingRoots = args

		plt.cla() # clear axis
		originalContour.plot(linecolor='k', linestyle='--')
		for box in boxes:
			if not hasattr(box, '_color'):
				cmap = plt.get_cmap('jet')
				box._color = cmap(np.random.random())

			plt.text(box.centralPoint.real, box.centralPoint.imag, box._numberOfRoots)
			box.plot(linecolor=box._color)

		plt.scatter(np.real(roots), np.imag(roots), color='k', marker='x')
		ax.text(0.02, 0.95, 'Zeros remaining: %i'%numberOfRemainingRoots, transform=ax.transAxes)
		originalContour._sizePlot()
		fig.canvas.draw()

	if saveFile:
		automaticAnim = True

	if automaticAnim or returnAnim:
		anim = animation.FuncAnimation(fig, update_frame, init_func=init, frames=rootFinder)
		if returnAnim: return anim

	else:
		def draw_next(event):
			if event.key == ' ':
				update_frame(next(rootFinder))
		fig.canvas.mpl_connect('key_press_event', draw_next)

	if saveFile:
		anim.save(filename=saveFile, fps=1, dpi=200, writer=writer)
		plt.close()
	else:
		plt.show()
