# codes below are taken from Guido van van Rossum's blog
# https://www.artima.com/weblogs/viewpost.jsp?thread=101605

registry = {}


class MultiMethod(object):
    def __init__(self, name):
        self.name = name
        self.typemap = {}

    def __call__(self, *args):
        # exact match
        types = tuple(arg.__class__ for arg in args)
        function = self.typemap.get(types)
        if function is None:
            raise TypeError("no matching type")
        return function(*args)

    def register(self, types, function):
        if types in self.typemap:
            raise TypeError("duplicate registration")
        self.typemap[types] = function


def multimethod(*types):
    def register(function):
        function = getattr(function, "__lastreg__", function)
        name = function.__name__
        mm = registry.get(name)
        if mm is None:
            mm = registry[name] = MultiMethod(name)
        mm.register(types, function)
        mm.__lastreg__ = function
        return mm
    return register
