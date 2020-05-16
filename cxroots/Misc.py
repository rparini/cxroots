from numpydoc.docscrape import FunctionDoc

import docrep
docstrings = docrep.DocstringProcessor()

def remove_para(*paras):
    def wrapper(func):
        doc = FunctionDoc(func)
        for p in doc['Parameters'].copy():
            if p.name.split(':')[0].rstrip() in paras:
                doc['Parameters'].remove(p)
        func.__doc__ = doc
        return func
    return wrapper

class NumberOfRootsChanged(Exception):
    pass
