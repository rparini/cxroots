from __future__ import division
from .RootFinder import findRootsGen

def demo_findRoots(originalContour, f, df=None, automaticAnimation=False, saveFile=None, **kwargs):
	"""
	An interactive demonstration of the processess used to find all the roots
	of a given function f within a given originalContour.
	Shares key word arguments with :func:`cxroots.RootFinder.findRootsGen`. 

	If automaticAnimation is False (default) then press the SPACE key 
	to step the animation forward.

	If automaticAnimation is True then the animation will play automatically
	until all the roots have been found.

	If saveFile is given as a string then the anmation will be saved
	"""
	import matplotlib.pyplot as plt
	from matplotlib import animation
	fig = plt.gcf()
	ax = plt.gca()

	rootFinder = findRootsGen(originalContour, f, df, **kwargs)
	originalContour.plot(linecolor='k', linestyle='--')

	### XXX: show total number of roots to be found at the start
	# ax.text(0.02, 0.95, 'Zeros remaining: %i'%originalContour.count_roots(f,df,**kwargs), transform=ax.transAxes)

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
		automaticAnimation = True

	if automaticAnimation:
		anim = animation.FuncAnimation(fig, update_frame, frames=rootFinder, interval=500)

	else:
		def draw_next(event):
			if event.key == ' ':
				update_frame(next(rootFinder)) 
		fig.canvas.mpl_connect('key_press_event', draw_next)

	if saveFile:
		anim.save(filename=saveFile, fps=1, dpi=200)
		plt.close()
	else:
		plt.show()
