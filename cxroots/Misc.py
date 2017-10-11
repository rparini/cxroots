import docrep
docstrings = docrep.DocstringProcessor()

def doc_tab_to_space(func):
	"""
	docrep doesn't like tabs
	"""
	func.__doc__ = func.__doc__.replace('\t', '    ')
	return func
