import docrep
docstrings = docrep.DocstringProcessor()

def remove_para(*paras):
    def wrapper(func):
        func = docstrings.dedent(func)
        func.__doc__ = docstrings.delete_params_s(func.__doc__, paras)
        return func
    return wrapper

class NumberOfRootsChanged(Exception):
    pass
