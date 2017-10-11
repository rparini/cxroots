from __future__ import division
import numpy as np

from .RootFinder import findRootsGen

def demo_findRoots(originalContour, f, df=None, automaticAnim=False, saveFile=None, returnAnim=False, writer=None, **kwargs):
	"""
	An animated demonstration of the root finding process using matplotlib.
	Takes all the parameters of :func:`cxroots.RootFinder.findRoots` as well as:

	Parameters
	----------
	automaticAnim : bool, optional
		If False (default) then press SPACE to step the animation forward
		If True then the animation will play automatically until all the 
		roots have been found.
	saveFile : str, optional
		If given then the animation will be saved to disk with filename 
		equal to saveFile instead of being shown.
	returnAnim : bool, optional
		If True then the matplotlib animation object will be returned 
		instead of being shown.  Defaults to False.
	"""
	import matplotlib.pyplot as plt
	from matplotlib import animation
	fig = plt.gcf()
	ax = plt.gca()

	rootFinder = findRootsGen(originalContour, f, df, **kwargs)
	originalContour.plot(linecolor='k', linestyle='--')
	originalContour.sizePlot()

	def update_frame(args):
		roots, multiplicities, boxes, numberOfRemainingRoots = args
		# print(args)

		plt.cla() # clear axis
		originalContour.plot(linecolor='k', linestyle='--')
		originalContour.sizePlot()
		for box, numberOfEnclosedRoots in boxes:
			if not hasattr(box, '_color'):
				cmap = plt.get_cmap('jet')
				box._color = cmap(np.random.random())
			
			plt.text(box.centralPoint.real, box.centralPoint.imag, numberOfEnclosedRoots)
			box.plot(linecolor=box._color)

		plt.scatter(np.real(roots), np.imag(roots))

		rootsLabel = ax.text(0.02, 0.95, 'Zeros remaining: %i'%numberOfRemainingRoots, transform=ax.transAxes)

		fig.canvas.draw()

	if saveFile:
		automaticAnim = True

	if automaticAnim or returnAnim:
		anim = animation.FuncAnimation(fig, update_frame, frames=rootFinder, interval=500)

		if returnAnim:
			return anim

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
