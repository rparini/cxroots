import docrep
docstrings = docrep.DocstringProcessor()

def doc_tab_to_space(func):
	"""
	docrep doesn't like tabs
	"""
	func.__doc__ = func.__doc__.replace('\t', '    ')
	return func

def remove_para(*paras):
	def wrapper(func):
		func = doc_tab_to_space(func)
		func = docstrings.dedent(func)
		func.__doc__ = docstrings.delete_params_s(func.__doc__, paras)
		return func
	return wrapper

class NumberOfRootsChanged(Exception):
    pass
