from numpydoc.docscrape import FunctionDoc


def remove_para(*paras):
    def wrapper(func):
        doc = FunctionDoc(func)
        for i, p in enumerate(doc["Parameters"][:]):
            if p.name.split(":")[0].rstrip() in paras:
                del doc["Parameters"][i]
        func.__doc__ = doc
        return func

    return wrapper


def update_docstring(**dic):
    def wrapper(func):
        doc = FunctionDoc(func)
        for k, v in dic.items():
            doc[k] = v
        func.__doc__ = doc
        return func

    return wrapper


class NumberOfRootsChangedError(Exception):
    pass
