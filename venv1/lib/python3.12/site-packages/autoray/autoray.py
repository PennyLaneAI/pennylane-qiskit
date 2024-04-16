"""
AUTORAY - backend agnostic array operations.


Copyright 2019-2023 Johnnie Gray

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import math
import importlib
import functools
import itertools
import threading
import contextlib
from inspect import signature
from collections import OrderedDict, defaultdict


def do(fn, *args, like=None, **kwargs):
    """Do function named ``fn`` on ``(*args, **kwargs)``, peforming single
    dispatch to retrieve ``fn`` based on whichever library defines the class of
    the ``args[0]``, or the ``like`` keyword argument if specified.

    Examples
    --------

    Works on numpy arrays:

        >>> import numpy as np
        >>> x_np = np.random.uniform(size=[5])
        >>> y_np = do('sqrt', x_np)
        >>> y_np
        array([0.32464973, 0.90379787, 0.85037325, 0.88729814, 0.46768083])

        >>> type(y_np)
        numpy.ndarray

    Works on cupy arrays:

        >>> import cupy as cp
        >>> x_cp = cp.random.uniform(size=[5])
        >>> y_cp = do('sqrt', x_cp)
        >>> y_cp
        array([0.44541656, 0.88713113, 0.92626237, 0.64080557, 0.69620767])

        >>> type(y_cp)
        cupy.core.core.ndarray

    Works on tensorflow arrays:

        >>> import tensorflow as tf
        >>> x_tf = tf.random.uniform(shape=[5])
        >>> y_tf = do('sqrt', x_tf)
        >>> y_tf
        <tf.Tensor 'Sqrt_1:0' shape=(5,) dtype=float32>

        >>> type(y_tf)
        tensorflow.python.framework.ops.Tensor

    You get the idea.

    For functions that don't dispatch on the first argument you can use the
    ``like`` keyword:

        >>> do('eye', 3, like=x_tf)
        <tf.Tensor: id=91, shape=(3, 3), dtype=float32>
    """
    backend = choose_backend(fn, *args, like=like, **kwargs)
    return get_lib_fn(backend, fn)(*args, **kwargs)


# ------------------------- efficiently dispatching ------------------------- #


def _default_infer_from_sig(fn, *args, **kwargs):
    """This is the default backend dispatcher, used if no global backend has
    been set. Hot swapping this function out as below avoids having to check
    manually for a global backend or worse, a thread aware global backend, on
    every call to ``do``.
    """
    return _DISPATCHERS[fn](*args, **kwargs)


_global_backend = None
_inferrer_global = _default_infer_from_sig

# this is the function that autoray uses when `do` is called without an
# explicit like/backend argument. It is set to `_default_infer_from_sig` by
# default, but can be set to `_always_the_same` if a global backend is set e.g.
_infer_auto = _inferrer_global

# if a thread that isn't the 'importing' thread tries to set a backend, this
# by default turns on thread aware dispatching, but once such custom sub
# backends have been reset, the global values above are used again.
_global_backends_threadaware = {}
_inferrers_threadaware = {}
_importing_thrid = threading.get_ident()
_backend_lock = threading.Lock()


def _default_infer_from_sig_threadaware(fn, *args, **kwargs):
    # check for a thread aware inferrer, default to the global inferrer
    thrid = threading.get_ident()
    return _inferrers_threadaware.get(thrid, _inferrer_global)(
        fn, *args, **kwargs
    )


def _always_the_same(*args, x, **kwargs):
    return x


def get_backend(get_globally="auto"):
    """Return the universally set backend, if any.

    Parameters
    ----------
    get_globally : {"auto", False, True}, optional
        Which backend to return:

        - True: return the globally set backend, if any.
        - False: return the backend set for the current thread, if any.
        - "auto": return the globally set backend, if this thread is the thread
          that imported autoray. Otherwise return the backend set for the
          current thread, if any.

    Returns
    -------
    backend : str or None
        The name of the backend, or None if no backend is set.
    """
    if get_globally == "auto":
        get_globally = threading.get_ident() == _importing_thrid

    if get_globally:
        backend = _global_backend
    else:
        thrid = threading.get_ident()
        backend = _global_backends_threadaware.get(thrid, None)

    return backend


def set_backend(like, set_globally="auto"):
    """Set a default global backend. The argument ``like`` can be an explicit
    backend name or an ``array``.

    Parameters
    ----------
    like : str or array
        The backend to set. If an array, the backend of the array's class will
        be set.
    set_globally : {"auto", False, True}, optional
        Whether to set the backend globally or for the current thread:

        - True: set the backend globally.
        - False: set the backend for the current thread.
        - "auto": set the backend globally if this thread is the thread that
          imported autoray. Otherwise set the backend for the current thread.

        Only one thread should ever call this function with
        ``set_globally=True``, (by default this is importing thread).
    """
    global _global_backend
    global _infer_auto
    global _inferrer_global

    if like is None:
        backend = None
        inferrer = _default_infer_from_sig
    elif isinstance(like, str):
        backend = like
        inferrer = functools.partial(_always_the_same, x=backend)
    else:
        backend = infer_backend(like)
        inferrer = functools.partial(_always_the_same, x=backend)

    if set_globally == "auto":
        set_globally = threading.get_ident() == _importing_thrid

    if set_globally:
        _global_backend = backend
        _inferrer_global = inferrer
        if not _inferrers_threadaware:
            # only revert the actual function if no subthread backends set
            _infer_auto = inferrer
    else:
        thrid = threading.get_ident()
        _backend_lock.acquire()
        if backend is None:
            _global_backends_threadaware.pop(thrid)
            _inferrers_threadaware.pop(thrid)
        else:
            _global_backends_threadaware[thrid] = backend
            _inferrers_threadaware[thrid] = inferrer

        if _inferrers_threadaware:
            # a subthread backend has been set, so we need to be thread aware
            _infer_auto = _default_infer_from_sig_threadaware
        else:
            # no subthread backend has been set anymore, so we can ignore
            # threads and just use the global inferrer
            _infer_auto = _inferrer_global
        _backend_lock.release()


@contextlib.contextmanager
def backend_like(like, set_globally="auto"):
    """Context manager for setting a default backend. The argument ``like`` can
    be an explicit backend name or an ``array`` to infer it from.

    Parameters
    ----------
    like : str or array
        The backend to set. If an array, the backend of the array's class will
        be set.
    set_globally : {"auto", False, True}, optional
        Whether to set the backend globally or for the current thread:

        - True: set the backend globally.
        - False: set the backend for the current thread.
        - "auto": set the backend globally if this thread is the thread that
          imported autoray. Otherwise set the backend for the current thread.

        Only one thread should ever call this function with
        ``set_globally=True``, (by default this is importing thread).
    """
    if set_globally == "auto":
        set_globally = threading.get_ident() == _importing_thrid

    old_backend = get_backend(get_globally=set_globally)
    try:
        set_backend(like, set_globally)
        yield
    finally:
        set_backend(old_backend, set_globally)


_CUSTOM_BACKENDS = {}


def register_backend(cls, name):
    """Register the name (and by default the module or submodule) of a custom
    array class.

    Parameters
    ----------
    cls : type
        The array class itself.
    name : str
        The name of the backend that should be used for this class. By default
        this wil be assumed to be the location of the relevant functions for
        this class, but this can be overridden.
    """
    if not isinstance(cls, type):
        raise TypeError("The array class itself should be supplied.")

    global _CUSTOM_BACKENDS
    _CUSTOM_BACKENDS[cls] = name


@functools.lru_cache(None)
def _infer_class_backend_cached(cls):
    try:
        import numpy as _numpy

        if issubclass(cls, _numpy.ndarray):
            return "numpy"
    except ImportError:
        # numpy not installed
        pass

    if cls in _CUSTOM_BACKENDS:
        return _CUSTOM_BACKENDS[cls]

    lib = cls.__module__.split(".")[0]

    # check if lib should mapped entirely to another lib
    backend = _BACKEND_ALIASES.get(lib, lib)

    return backend


def infer_backend(array):
    """Get the name of the library that defined the class of ``array`` - unless
    ``array`` is directly a subclass of ``numpy.ndarray``, in which case assume
    ``numpy`` is the desired backend.
    """
    return _infer_class_backend_cached(array.__class__)


multi_class_priorities = {
    "builtins": -2,
    "numpy": -1,
    "autoray.lazy": 1,
}


@functools.lru_cache(None)
def _infer_class_backend_multi_cached(classes):
    return max(
        map(_infer_class_backend_cached, classes),
        key=lambda n: multi_class_priorities.get(n, 0),
    )


def infer_backend_multi(*arrays):
    """Infer which backend should be used for a function that takes multiple
    arguments. This assigns a priority to each backend, and returns the backend
    with the highest priority. By default, the priority is:

    - ``builtins``: -2
    - ``numpy``: -1
    - other backends: 0
    - ``autoray.lazy``: 1

    I.e. when mixing with ``numpy``, other array libraries are preferred, when
    mixing with ``autoray.lazy``, ``autoray.lazy`` is preferred. This has quite
    low overhead due to caching.
    """
    return _infer_class_backend_multi_cached(
        tuple(array.__class__ for array in arrays)
    )


def choose_backend(fn, *args, like=None, **kwargs):
    """Choose a backend based on function name, arguments, and the ``like``
    keyword argument. The default, if ``like`` is not specified, is to infer
    the backend from the function call, the default of which is simply to use
    the first argument, if no custom dispatcher is found. Otherwise the
    backend is chosen based on the ``like`` argument - which can be an explicit
    backend name or an arbitrary object.
    """
    if like is None:
        return _infer_auto(fn, *args, **kwargs)
    elif isinstance(like, str):
        return like
    else:
        return infer_backend(like)


# ------------------- importing and caching the function -------------------- #

# lookup for mapping entire lib to another
_BACKEND_ALIASES = {}

# global (non function specific) aliases
_MODULE_ALIASES = {}

# lookup for when functions are elsewhere than the expected location
_SUBMODULE_ALIASES = {}

# lookup for when functions are simply called something else
_FUNC_ALIASES = {}

# custom wrappers for when functions don't just have different location or
#     name. For example, when kwargs need to be translated or results modified
_CUSTOM_WRAPPERS = {}

# actual cache of funtions to use - this is populated lazily and can be used
#     to directly set an implementation of a function for a specific backend
_FUNCS = {}

# these are functions where a default implementation can be constructed
#     (composed of other functions), but this is only done lazily
_COMPOSED_FUNCTION_GENERATORS = {}


def import_lib_fn(backend, fn):
    # first check explicitly composed functions -> if the function hasn't been
    # called directly yet, it won't have been loaded into the cache, and needs
    # generating before e.g. the ``do`` verrsion will work
    if fn in _COMPOSED_FUNCTION_GENERATORS:
        return _COMPOSED_FUNCTION_GENERATORS[fn](backend)

    try:
        # submodule where function is found for backend,
        #     e.g. ['tensorflow', trace'] -> 'tensorflow.linalg'
        try:
            full_location = _SUBMODULE_ALIASES[backend, fn]

            # if explicit submodule alias given, don't use prepended location
            #     for example, ('torch', 'linalg.svd') -> torch.svd
            only_fn = fn.split(".")[-1]

        except KeyError:
            full_location = backend

            # move any prepended location into the full module path
            #     e.g. 'fn=linalg.eigh' -> ['linalg', 'eigh']
            split_fn = fn.split(".")
            full_location = ".".join([full_location] + split_fn[:-1])
            only_fn = split_fn[-1]

            # try aliases for global (not function specific) modules and
            # submodules:
            #     e.g. 'decimal' -> 'math'
            #     e.g. 'cupy.scipy' -> 'cupyx.scipy'
            # we don't do this if the function location has been explicitly
            # give in _SUBMODULE_ALIASES, as that is already a full path
            for k, v in _MODULE_ALIASES.items():
                if full_location[: len(k)] == k:
                    full_location = full_location.replace(k, v, 1)
                    break

        # cached lookup of custom name function might take
        #     e.g. ['tensorflow', 'sum'] -> 'reduce_sum'
        fn_name = _FUNC_ALIASES.get((backend, fn), only_fn)

        # import the function into the cache
        try:
            lib = importlib.import_module(full_location)
        except ImportError:
            if "." in full_location:
                # sometimes libraries hack an attribute to look like submodule
                mod, *submods = full_location.split(".")
                lib = importlib.import_module(mod)
                # also need to handle nested submodules
                for submod in submods:
                    lib = getattr(lib, submod)
            else:
                # failed to import library at all -> catch + raise ImportError
                raise AttributeError

        # check for a custom wrapper but default to identity
        wrapper = _CUSTOM_WRAPPERS.get((backend, fn), lambda fn: fn)

        # store the function!
        lib_fn = _FUNCS[backend, fn] = wrapper(getattr(lib, fn_name))

    except AttributeError:
        # check if there is a backup function (e.g. for older library version)
        backend_alt = backend + "[alt]"
        if backend_alt in _MODULE_ALIASES:
            return import_lib_fn(backend_alt, fn)

        raise ImportError(
            f"autoray couldn't find function '{fn}' for "
            f"backend '{backend.replace('[alt]', '')}'."
        )

    return lib_fn


def get_lib_fn(backend, fn):
    """Cached retrieval of correct function for backend, all the logic for
    finding the correct funtion only runs the first time.

    Parameters
    ----------
    backend : str
        The module defining the array class to dispatch on.
    fn : str
        The function to retrieve.

    Returns
    -------
    callable
    """

    try:
        lib_fn = _FUNCS[backend, fn]
    except KeyError:
        lib_fn = import_lib_fn(backend, fn)
    return lib_fn


# --------------------------- register your own! ---------------------------- #


def register_function(backend, name, fn, wrap=False):
    """Directly provide your own function.

    Parameters
    ----------
    backend : str
        The name of the backend to register the function for.
    name : str
        Name of the function, e.g. `'sum'` or `'linalg.svd'`.
    fn : callable
        The function to register.
    wrap : bool, optional
        Whether to wrap the old function like ``fn(old_fn)`` rather than
        directly supply the entire new function.
    """
    if wrap:
        old = get_lib_fn(backend, name)
        _FUNCS[backend, name] = fn(old)
    else:
        _FUNCS[backend, name] = fn


# ------------------------------- tree utils -------------------------------- #

TREE_MAP_REGISTRY = {}
TREE_APPLY_REGISTRY = {}
TREE_ITER_REGISTRY = {}


def tree_register_container(cls, mapper, iterator, applier):
    """Register a new container type for use with ``tree_map`` and
    ``tree_apply``.

    Parameters
    ----------
    cls : type
        The container type to register.
    mapper : callable
        A function that takes ``f``, ``tree`` and ``is_leaf`` and returns a new
        tree of type ``cls`` with ``f`` applied to all leaves.
    applier : callable
        A function that takes ``f``, ``tree`` and ``is_leaf`` and applies ``f``
        to all leaves in ``tree``.
    """
    TREE_MAP_REGISTRY[cls] = mapper
    TREE_ITER_REGISTRY[cls] = iterator
    TREE_APPLY_REGISTRY[cls] = applier


IS_CONTAINER_CACHE = {}


def is_not_container(x):
    """The default function to determine if an object is a leaf. This simply
    checks if the object is an instance of any of the registered container
    types.
    """
    try:
        return IS_CONTAINER_CACHE[x.__class__]
    except KeyError:
        isleaf = not any(isinstance(x, cls) for cls in TREE_MAP_REGISTRY)
        IS_CONTAINER_CACHE[x.__class__] = isleaf
        return isleaf


def is_array(x):
    """An alternative leaf tester for addressing only arrays within trees."""
    return hasattr(x, "shape")


def identity(f, tree, is_leaf):
    return tree


TREE_MAPPER_CACHE = {}


def tree_map(f, tree, is_leaf=is_not_container):
    """Map ``f`` over all leaves in ``tree``, returning a new pytree.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.

    Returns
    -------
    pytree
    """
    if is_leaf(tree):
        return f(tree)

    try:
        return TREE_MAPPER_CACHE[tree.__class__](f, tree, is_leaf)
    except KeyError:
        # reverse so later registered classes take precedence
        for cls, mapper in reversed(TREE_MAP_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            # neither leaf nor container -> simply return it
            mapper = identity
        TREE_MAPPER_CACHE[tree.__class__] = mapper
        return mapper(f, tree, is_leaf)


def empty(tree, is_leaf):
    return iter(())


TREE_ITER_CACHE = {}


def tree_iter(tree, is_leaf=is_not_container):
    """Iterate over all leaves in ``tree``.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.
    """
    if is_leaf(tree):
        yield tree
        return

    try:
        yield from TREE_ITER_CACHE[tree.__class__](tree, is_leaf)
    except KeyError:
        # reverse so later registered classes take precedence
        for cls, iterator in reversed(TREE_ITER_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            # neither leaf nor container -> simply ignore it
            iterator = empty
        TREE_ITER_CACHE[tree.__class__] = iterator
        yield from iterator(tree, is_leaf)


def nothing(f, tree, is_leaf):
    pass


TREE_APPLIER_CACHE = {}


def tree_apply(f, tree, is_leaf=is_not_container):
    """Apply ``f`` to all leaves in ``tree``, no new pytree is built.

    Parameters
    ----------
    f : callable
        A function to apply to all leaves in ``tree``.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, ``f`` is only applied
        to objects for which ``is_leaf(x)`` returns ``True``.
    """
    if is_leaf(tree):
        f(tree)
        return

    try:
        TREE_APPLIER_CACHE[tree.__class__](f, tree, is_leaf)
    except KeyError:
        # reverse so later registered classes take precedence
        for cls, applier in reversed(TREE_APPLY_REGISTRY.items()):
            if isinstance(tree, cls):
                break
        else:
            # neither leaf nor container -> simply ignore it
            applier = nothing
        TREE_APPLIER_CACHE[tree.__class__] = applier
        applier(f, tree, is_leaf)


class Leaf:
    """A singleton object to use as a placeholder in a pytree, for
    unflattening.
    """

    __slots__ = ()

    def __repr__(self):
        return "Leaf"


LEAF = Leaf()


def is_leaf_placeholder(x):
    # don't do `x is LEAF` to allow pickling / unpickling
    return x.__class__ is Leaf


def tree_flatten(tree, is_leaf=is_not_container, get_ref=False):
    """Flatten ``tree`` into a list of leaves.

    Parameters
    ----------
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects.
    is_leaf : callable
        A function to determine if an object is a leaf, only objects for which
        ``is_leaf(x)`` returns ``True`` are returned in the flattened list.
    get_ref : bool
        If ``True``, a reference tree is also returned which can be used to
        reconstruct the original tree from a flattened list.

    Returns
    -------
    objs : list
        The flattened list of leaf objects.
    (ref_tree) : pytree
        If ``get_ref`` is ``True``, a reference tree, with leaves of ``Leaf``,
        is returned which can be used to reconstruct the original tree.
    """
    objs = []
    if get_ref:
        # return a new tree with Leaf leaves, as well as the flattened list

        def f(x):
            objs.append(x)
            return LEAF

        ref_tree = tree_map(f, tree, is_leaf)
        return objs, ref_tree
    else:
        tree_apply(objs.append, tree, is_leaf)
        return objs


def tree_unflatten(objs, tree, is_leaf=is_leaf_placeholder):
    """Unflatten ``objs`` into a pytree of the same structure as ``tree``.

    Parameters
    ----------
    objs : sequence
        A sequence of objects to be unflattened into a pytree.
    tree : pytree
        A nested sequence of tuples, lists, dicts and other objects, the objs
        will be inserted into a new pytree of the same structure.
    is_leaf : callable
        A function to determine if an object is a leaf, only objects for which
        ``is_leaf(x)`` returns ``True`` will have the next item from ``objs``
        inserted. By default checks for the ``Leaf`` object inserted by
        ``tree_flatten(..., get_ref=True)``.

    Returns
    -------
    pytree
    """
    objs = iter(objs)
    return tree_map(lambda _: next(objs), tree, is_leaf)


def tree_map_tuple(f, tree, is_leaf):
    return tuple(tree_map(f, x, is_leaf) for x in tree)


def tree_iter_tuple(tree, is_leaf):
    for x in tree:
        yield from tree_iter(x, is_leaf)


def tree_apply_tuple(f, tree, is_leaf):
    for x in tree:
        tree_apply(f, x, is_leaf)


tree_register_container(
    tuple, tree_map_tuple, tree_iter_tuple, tree_apply_tuple
)


def tree_map_list(f, tree, is_leaf):
    return [tree_map(f, x, is_leaf) for x in tree]


def tree_iter_list(tree, is_leaf):
    for x in tree:
        yield from tree_iter(x, is_leaf)


def tree_apply_list(f, tree, is_leaf):
    for x in tree:
        tree_apply(f, x, is_leaf)


tree_register_container(list, tree_map_list, tree_iter_list, tree_apply_list)


def tree_map_dict(f, tree, is_leaf):
    return {k: tree_map(f, v, is_leaf) for k, v in tree.items()}


def tree_iter_dict(tree, is_leaf):
    for v in tree.values():
        yield from tree_iter(v, is_leaf)


def tree_apply_dict(f, tree, is_leaf):
    for v in tree.values():
        tree_apply(f, v, is_leaf)


tree_register_container(dict, tree_map_dict, tree_iter_dict, tree_apply_dict)


# --------------------------- composed functions ---------------------------- #


class Composed:
    """Compose an ``autoray.do`` using function. See the main wrapper
    ``compose``.
    """

    def __init__(self, fn, name=None):
        self._default_fn = fn
        if name is None:
            name = fn.__name__
        self._name = name
        self._supply_backend = "backend" in signature(fn).parameters

        # this registers the fact that when `get_lib_fn` is called, the
        # function can be created even if it doesn't exist for a specific
        # backend yet.
        _COMPOSED_FUNCTION_GENERATORS[self._name] = self.make_function

    def register(self, backend, fn=None):
        """Register a different implementation for ``backend``."""
        if fn is not None:
            register_function(backend, self._name, fn)
        else:
            # wrapper form
            def wrapper(fn):
                register_function(backend, self._name, fn)
                return fn

            return wrapper

    def make_function(self, backend):
        """Make a new function for the specific ``backend``."""
        if self._supply_backend:
            # make sure it inherits __name__ etc
            fn = functools.wraps(self._default_fn)(
                functools.partial(self._default_fn, backend=backend)
            )
        else:
            fn = self._default_fn
        self.register(backend, fn)
        return fn

    def __call__(self, *args, like=None, **kwargs):
        backend = choose_backend(self._name, *args, like=like, **kwargs)
        # `get_lib_fn` will call `make_function` if the function doesn't exist
        fn = get_lib_fn(backend, self._name)
        return fn(*args, **kwargs)

    def __repr__(self):
        return f"Composed('{self._name}')"


def compose(fn, *, name=None):
    """Take a function consisting of multiple ``autoray.do`` calls and compose
    it into a new, single, named function, registered with ``autoray.do``.

    This creates a default implementation of this function for each new backend
    encountered without explicitly having to write each out, but also allows
    for specific implementations to be overridden for specific backends.

    If the function takes a ``backend`` argument, it will be supplied with the
    backend name, to save having to re-choose the backend.

    Specific implementations can be provided by calling the ``register`` method
    of the composed function, or it can itself be used like a decorator::

        @compose
        def foo(x):
            ...

        @foo.register("numpy")
        @numba.njit
        def foo_numba(x):
            ...

    Parameters
    ----------
    fn : callable
        The funtion to compose, and its default implementation.
    name : str, optional
        The name of the composed function. If not provided, the name of the
        function will be used.
    """
    if fn is None:
        return functools.partial(compose, name=name)
    return functools.wraps(fn)(Composed(fn, name))


# ---------------------- special top level functions ------------------------ #


@compose
def shape(x):
    """Get the shape of an array as a tuple of int. This should be preferred
    to calling `x.shape` directly, as it:

        1. Allows customization (e.g. for torch and aesara which return
           different types for shape - use `@shape.register(backend)` to
           customize the behavior from this default implementation).
        2. Can be used on nested lists and tuples, without calling numpy.

    Parameters
    ----------
    x : array_like
        The array to get the shape of. It can be an arbitrary nested list or
        tuple of arrays and scalars, but is assumed not to be ragged.

    Returns
    -------
    shape : tuple of int
        The size of each dimension of the array.
    """
    try:
        return x.shape
    except AttributeError:
        # want to handle builtins / nested stuff
        if isinstance(x, (list, tuple)):
            d = len(x)
            if d != 0:
                # NB: slightly different from np.shape, as we do not check for
                # ragged arrays, but that behavior is seemingly deprecated
                return (d,) + shape(x[0])
            return (d,)
        return ()


@compose
def ndim(x):
    """Get the number of dimensions of an array. This should be preferred to
    calling `x.ndim`, since not all backends implement that, and it can also be
    called on nested lists and tuples.

    Parameters
    ----------
    x : array_like
        The array to get the number of dimensions of. It can be an arbitrary
        nested list or tuple of arrays and scalars.

    Returns
    -------
    ndim : int
    """
    try:
        return x.ndim
    except AttributeError:
        return len(shape(x))


@compose
def size(x):
    """Get the size, or number of elements, of an array. This should be
    preferred to calling `x.size`, since not all backends implement that, and
    it can also be called on nested lists and tuples.

    Parameters
    ----------
    x : array_like
        The array to get the size of. It can be an arbitrary nested list or
        tuple of arrays and scalars.

    Returns
    -------
    size : int
    """
    try:
        return x.size
    except AttributeError:
        return math.prod(shape(x))


def conj(x):
    """Array conjugate."""
    return do("conj", x)


def transpose(x, *args):
    """Array transpose."""
    return do("transpose", x, *args)


def dag(x):
    """Array Hermitian transpose."""
    try:
        return x.H
    except AttributeError:
        return do("conj", do("transpose", x))


def real(x):
    """Array real part."""
    return do("real", x)


def imag(x):
    """Array imaginary part."""
    return do("imag", x)


def reshape(x, shape):
    """Array reshaped."""
    try:
        return x.reshape(shape)
    except AttributeError:
        return do("reshape", x, shape)


def to_backend_dtype(dtype_name, like):
    """Turn string specifier ``dtype_name`` into dtype of backend ``like``."""
    if not isinstance(like, str):
        like = infer_backend(like)

    try:
        return get_lib_fn(like, dtype_name)
    except ImportError:
        return dtype_name


def get_dtype_name(x):
    """Find string specifier ``dtype_name`` of array ``x``."""
    try:
        return x.dtype.name
    except AttributeError:
        # let modules provide their own
        return do("get_dtype_name", x, like=x)


_COMPLEX_DTYPES = {"complex64", "complex128"}
_DOUBLE_DTYPES = {"float64", "complex128"}
_DTYPE_MAP = {
    (False, False): "float32",
    (False, True): "float64",
    (True, False): "complex64",
    (True, True): "complex128",
}


def get_common_dtype(*arrays):
    """Compute the minimal dtype sufficient for ``arrays``."""
    dtypes = set(map(get_dtype_name, arrays))
    has_complex = not _COMPLEX_DTYPES.isdisjoint(dtypes)
    has_double = not _DOUBLE_DTYPES.isdisjoint(dtypes)
    return _DTYPE_MAP[has_complex, has_double]


def astype(x, dtype_name, **kwargs):
    """Cast array as type ``dtype_name`` - tries ``x.astype`` first."""
    dtype = to_backend_dtype(dtype_name, like=x)
    try:
        return x.astype(dtype, **kwargs)
    except AttributeError:
        return do("astype", x, dtype, **kwargs)


def to_numpy(x):
    """Get a numpy version of array ``x``."""
    return do("to_numpy", x)


# -------------------------- some common wrappers --------------------------- #


def svd_not_full_matrices_wrapper(fn):
    @functools.wraps(fn)
    def default_not_full_matrices(*args, **kwargs):
        kwargs.setdefault("full_matrices", False)
        return fn(*args, **kwargs)

    return default_not_full_matrices


def svd_sUV_to_UsVH_wrapper(fn):
    @functools.wraps(fn)
    def numpy_like(*args, **kwargs):
        s, U, V = fn(*args, **kwargs)
        return U, s, dag(V)

    return numpy_like


def svd_UsV_to_UsVH_wrapper(fn):
    @functools.wraps(fn)
    def numpy_like(*args, **kwargs):
        U, s, V = fn(*args, **kwargs)
        return U, s, dag(V)

    return numpy_like


def svd_manual_full_matrices_kwarg(fn):
    @functools.wraps(fn)
    def numpy_like(*args, full_matrices=False, **kwargs):
        U, s, VH = fn(*args, **kwargs)

        if not full_matrices:
            U, VH = U[:, : s.size], VH[: s.size, :]

        return U, s, VH

    return numpy_like


def qr_allow_fat(fn):
    @functools.wraps(fn)
    def numpy_like(a, **kwargs):
        m, n = shape(a)

        if m >= n:
            # square or thin
            return fn(a, **kwargs)

        Q, R_sq = fn(a[:, :m])
        R_r = dag(Q) @ a[:, m:]
        R = do("concatenate", (R_sq, R_r), axis=1, like=a)

        return Q, R

    return numpy_like


def tril_to_band_part(fn):
    @functools.wraps(fn)
    def numpy_like(x, k=0):
        if k < 0:
            raise ValueError(
                "'k' must be positive to recreate 'numpy.tril' "
                "behaviour with 'tensorflow.matrix_band_part'."
            )

        return fn(x, -1, k)

    return numpy_like


def triu_to_band_part(fn):
    @functools.wraps(fn)
    def numpy_like(x, k=0):
        if k > 0:
            raise ValueError(
                "'k' must be negative to recreate 'numpy.triu' "
                "behaviour with 'tensorflow.matrix_band_part'."
            )

        return fn(x, -k, -1)

    return numpy_like


def cholesky_lower(fn):
    @functools.wraps(fn)
    def cholesky_numpy_like(a):
        return fn(a, lower=True)

    return cholesky_numpy_like


def binary_allow_1d_rhs_wrap(fn):
    @functools.wraps(fn)
    def allow_1d_rhs(a, b):
        need_to_convert = ndim(a) != ndim(b)
        if need_to_convert:
            b = reshape(b, (*shape(b), 1))
        x = fn(a, b)
        if need_to_convert:
            x = reshape(x, shape(x)[:-1])
        return x

    return allow_1d_rhs


def scale_random_uniform_manually(fn):
    @functools.wraps(fn)
    def numpy_like(low=0.0, high=1.0, size=None, dtype=None, **kwargs):
        if size is None:
            size = ()

        x = fn(size, **kwargs)

        if (low != 0.0) or (high != 1.0):
            x = (high - low) * x + low

        if (dtype is not None) and get_dtype_name(x) != dtype:
            x = astype(x, dtype)
        return x

    return numpy_like


def scale_random_normal_manually(fn):
    @functools.wraps(fn)
    def numpy_like(loc=0.0, scale=1.0, size=None, dtype=None, **kwargs):
        if size is None:
            size = ()

        x = fn(size=size, **kwargs)

        if (loc != 0.0) or (scale != 1.0):
            x = scale * x + loc

        if (dtype is not None) and get_dtype_name(x) != dtype:
            x = astype(x, dtype)
        return x

    return numpy_like


def with_dtype_wrapper(fn):
    """Add ability to handle `dtype` keyword.
    If not None, `dtype` should be specified as a string, otherwise conversion
    will happen regardless.
    """

    @functools.wraps(fn)
    def with_dtype(*args, dtype=None, **kwargs):
        A = fn(*args, **kwargs)
        if (dtype is not None) and (dtype != get_dtype_name(A)):
            A = astype(A, dtype)
        return A

    return with_dtype


def translate_wrapper(fn, translator):
    """Wrap a function to match the api of another according to a translation.
    The ``translator`` entries in the form of an ordered dict should have
    entries like:

        (desired_kwarg: (backend_kwarg, default_value))

    with the order defining the args of the function.
    """

    @functools.wraps(fn)
    def translated_function(*args, **kwargs):
        new_kwargs = {}
        translation = translator.copy()

        # convert args
        for arg_value in args:
            new_arg_name = translation.popitem(last=False)[1][0]
            new_kwargs[new_arg_name] = arg_value

        # convert kwargs -  but only those in the translation
        for key, value in kwargs.items():
            try:
                new_kwargs[translation.pop(key)[0]] = value
            except KeyError:
                new_kwargs[key] = value

        # set remaining default kwargs
        for key, value in translation.items():
            new_kwargs[value[0]] = value[1]

        return fn(**new_kwargs)

    return translated_function


def make_translator(t):
    return functools.partial(translate_wrapper, translator=OrderedDict(t))


def complex_add_re_im(re, im):
    return re + 1j * im


def allclose(x, y, rtol=1e-05, atol=1e-08):
    return do("all", do("abs", x - y) <= atol + rtol * do("abs", y))


# ----------------------------- Custom dispatchers -------------------------- #


def register_dispatch(fun, dispatcher):
    """Register a new dispatcher.

    This is useful in case the backend to be used by a function cannot be
    inferred from the first argument.
    """
    _DISPATCHERS[fun] = dispatcher


def default_dispatcher(*args, **kwargs):
    """Try to infer backend from first argument passed to function."""
    return infer_backend(args[0])


# lookup of custom dispatcher methods, for cases when backend cannot be
#     inferred accurately from first argument.
_DISPATCHERS = defaultdict(lambda: default_dispatcher)


def join_array_dispatcher(*args, **kwargs):
    """Dispatcher for functions where first argument is a sequence."""
    try:
        return infer_backend(args[0][0])
    except (TypeError, ValueError):
        # user passed an empty sequence, or something non-iterable
        # try to infer backend from first argument as fallback
        return infer_backend(args[0])


# List of functions listed in numpy API as array joining operations
register_dispatch("concatenate", join_array_dispatcher)
register_dispatch("stack", join_array_dispatcher)
register_dispatch("block", join_array_dispatcher)
register_dispatch("vstack", join_array_dispatcher)
register_dispatch("hstack", join_array_dispatcher)
register_dispatch("dstack", join_array_dispatcher)
register_dispatch("column_stack", join_array_dispatcher)
register_dispatch("row_stack", join_array_dispatcher)


def einsum_dispatcher(*args, **_):
    """Dispatcher for handling einsum.

    einsum can be called with a str equation as the first argument, or with
    'interleaved' inputs. This dispatcher handles both cases and also takes
    into account all arrays.
    """
    return infer_backend_multi(*args)


register_dispatch("einsum", einsum_dispatcher)


def binary_dispatcher(*args, **_):
    """There are cases when we want to take into account both backends of two
    arguments, e.g. a lazy variable and a constant array.
    """
    return infer_backend_multi(*args[:2])


register_dispatch("tensordot", binary_dispatcher)
register_dispatch("matmul", binary_dispatcher)
register_dispatch("multiply", binary_dispatcher)
register_dispatch("divide", binary_dispatcher)
register_dispatch("true_divide", binary_dispatcher)
register_dispatch("add", binary_dispatcher)
register_dispatch("subtract", binary_dispatcher)

# TODO: register other binary functions?

# --------------- object to act as drop-in replace for numpy ---------------- #

_partial_functions = {}


class NumpyMimic:
    """A class to mimic the syntax of using `numpy` directly."""

    def __init__(self, submodule=None):
        self.submodule = submodule

    def __getattribute__(self, fn):
        # look out for certain submodules which are not functions
        if fn == "linalg":
            return numpy_linalg
        if fn == "random":
            return numpy_random

        # if this is the e.g. linalg mimic, preprend 'linalg.'
        submod = object.__getattribute__(self, "submodule")
        if submod is not None:
            fn = ".".join((submod, fn))

        # cache the correct partial function
        try:
            pfn = _partial_functions[fn]
        except KeyError:
            pfn = _partial_functions[fn] = functools.partial(do, fn)

        return pfn

    @staticmethod
    def __repr__():
        return "<autoray.numpy>"


numpy = NumpyMimic()
numpy_linalg = NumpyMimic("linalg")
numpy_random = NumpyMimic("random")


# --------------------------------------------------------------------------- #
#                             specific functions                              #
# --------------------------------------------------------------------------- #

# ------------------------------ standard-lib ------------------------------- #

_MODULE_ALIASES["decimal"] = "math"
_MODULE_ALIASES["builtins"] = "numpy"


_builtin_dtype_lookup = {
    int: "int64",
    float: "float64",
    complex: "complex128",
}


def builtins_get_dtype_name(x):
    return _builtin_dtype_lookup[x.__class__]


_FUNCS["builtins", "get_dtype_name"] = builtins_get_dtype_name
_FUNCS["builtins", "complex"] = complex

# ---------------------------------- numpy ---------------------------------- #


def numpy_to_numpy(x):
    return do("asarray", x, like="numpy")


_MODULE_ALIASES["numpy.scipy"] = "scipy"
_FUNCS["numpy", "to_numpy"] = numpy_to_numpy
_FUNCS["numpy", "complex"] = complex_add_re_im
_FUNCS["builtins", "to_numpy"] = numpy_to_numpy
_SUBMODULE_ALIASES["numpy", "linalg.lu"] = "scipy.linalg"
_SUBMODULE_ALIASES["numpy", "linalg.expm"] = "scipy.linalg"
_CUSTOM_WRAPPERS["numpy", "linalg.svd"] = svd_not_full_matrices_wrapper
_CUSTOM_WRAPPERS["numpy", "random.normal"] = with_dtype_wrapper
_CUSTOM_WRAPPERS["numpy", "random.uniform"] = with_dtype_wrapper

# ---------------------------------- cupy ----------------------------------- #


def cupy_to_numpy(x):  # pragma: no cover
    return x.get()


_MODULE_ALIASES["cupy.scipy"] = "cupyx.scipy"
_FUNCS["cupy", "to_numpy"] = cupy_to_numpy
_FUNCS["cupy", "complex"] = complex_add_re_im
_CUSTOM_WRAPPERS["cupy", "linalg.svd"] = svd_not_full_matrices_wrapper


# ----------------------------------- jax ----------------------------------- #


_JAX_RANDOM_KEY = None


def jax_random_seed(seed=None):
    from jax.random import PRNGKey

    global _JAX_RANDOM_KEY
    if seed is None:
        from random import SystemRandom

        seed = SystemRandom().randint(
            -(2**63), 2**63 - 1
        )  # inclusive high
    _JAX_RANDOM_KEY = PRNGKey(seed)


def jax_random_get_key():
    from jax.random import split

    global _JAX_RANDOM_KEY
    if _JAX_RANDOM_KEY is None:
        jax_random_seed()
    _JAX_RANDOM_KEY, subkey = split(_JAX_RANDOM_KEY)
    return subkey


def jax_random_uniform(low=0.0, high=1.0, size=None, **kwargs):
    from jax.random import uniform

    if size is None:
        size = ()
    return uniform(
        jax_random_get_key(), shape=size, minval=low, maxval=high, **kwargs
    )


def jax_random_normal(loc=0.0, scale=1.0, size=None, **kwargs):
    from jax.random import normal

    if size is None:
        size = ()
    x = normal(jax_random_get_key(), shape=size, **kwargs)
    if scale != 1.0:
        x *= scale
    if loc != 0.0:
        x += loc
    return x


def jax_to_numpy(x):
    return do("asarray", x, like="numpy")


_BACKEND_ALIASES["jaxlib"] = "jax"
_MODULE_ALIASES["jax.scipy"] = "jax.scipy"
_MODULE_ALIASES["jax"] = "jax.numpy"
_SUBMODULE_ALIASES["jax", "complex"] = "jax.lax"
_SUBMODULE_ALIASES["jax", "linalg.expm"] = "jax.scipy.linalg"
_SUBMODULE_ALIASES["jax", "linalg.householder_product"] = "jax.lax.linalg"
_CUSTOM_WRAPPERS["jax", "linalg.qr"] = qr_allow_fat
_CUSTOM_WRAPPERS["jax", "linalg.svd"] = svd_not_full_matrices_wrapper
_FUNCS["jax", "to_numpy"] = jax_to_numpy
_FUNCS["jax", "random.seed"] = jax_random_seed
_FUNCS["jax", "random.uniform"] = jax_random_uniform
_FUNCS["jax", "random.normal"] = jax_random_normal


# --------------------------------- aesara ---------------------------------- #


@shape.register("aesara")
def aesara_shape(x):
    return x.type.shape


_MODULE_ALIASES["aesara"] = "aesara.tensor"
_FUNCS["aesara", "shape"] = aesara_shape


# -------------------------------- autograd --------------------------------- #

_MODULE_ALIASES["autograd"] = "autograd.numpy"
_CUSTOM_WRAPPERS["autograd", "linalg.svd"] = svd_not_full_matrices_wrapper
_FUNCS["autograd", "complex"] = complex_add_re_im


# ---------------------------------- dask ----------------------------------- #


def dask_to_numpy(x):
    return x.compute()


def dask_eye_wrapper(fn):
    # Make M work as positional argument
    @functools.wraps(fn)
    def numpy_like(N, M=None, **kwargs):
        return fn(N, M=M, **kwargs)

    return numpy_like


_FUNCS["dask", "to_numpy"] = dask_to_numpy
_FUNCS["dask", "complex"] = complex_add_re_im
_FUNC_ALIASES["dask", "abs"] = "absolute"
_MODULE_ALIASES["dask"] = "dask.array"
_CUSTOM_WRAPPERS["dask", "linalg.svd"] = svd_manual_full_matrices_kwarg
_CUSTOM_WRAPPERS["dask", "linalg.cholesky"] = cholesky_lower
_CUSTOM_WRAPPERS["dask", "random.normal"] = with_dtype_wrapper
_CUSTOM_WRAPPERS["dask", "random.uniform"] = with_dtype_wrapper
_CUSTOM_WRAPPERS["dask", "eye"] = dask_eye_wrapper

# ---------------------------------- mars ----------------------------------- #


def mars_to_numpy(x):
    return x.to_numpy()


_FUNCS["mars", "to_numpy"] = mars_to_numpy
_FUNCS["mars", "complex"] = complex_add_re_im
_MODULE_ALIASES["mars"] = "mars.tensor"
_CUSTOM_WRAPPERS["mars", "linalg.cholesky"] = cholesky_lower


# ----------------------------------- ctf ----------------------------------- #


def ctf_array(x):
    return do("astensor", x, like="ctf")


def ctf_to_numpy(x):
    return x.to_nparray()


def ctf_count_nonzero(x):
    return (x != 0).astype(int).sum()


def ctf_get_dtype_name(x):
    return x.dtype.__name__


_FUNCS["ctf", "array"] = ctf_array
_FUNCS["ctf", "complex"] = complex_add_re_im
_FUNCS["ctf", "allclose"] = allclose
_FUNCS["ctf", "to_numpy"] = ctf_to_numpy
_FUNCS["ctf", "count_nonzero"] = ctf_count_nonzero
_FUNCS["ctf", "get_dtype_name"] = ctf_get_dtype_name

_SUBMODULE_ALIASES["ctf", "float32"] = "numpy"
_SUBMODULE_ALIASES["ctf", "float64"] = "numpy"
_SUBMODULE_ALIASES["ctf", "complex64"] = "numpy"
_SUBMODULE_ALIASES["ctf", "complex128"] = "numpy"
_SUBMODULE_ALIASES["ctf", "linalg.svd"] = "ctf"
_SUBMODULE_ALIASES["ctf", "linalg.eigh"] = "ctf"
_SUBMODULE_ALIASES["ctf", "linalg.qr"] = "ctf"
_SUBMODULE_ALIASES["ctf", "linalg.norm"] = "ctf"

_FUNC_ALIASES["ctf", "random.uniform"] = "random"

_CUSTOM_WRAPPERS["ctf", "random.uniform"] = scale_random_uniform_manually


# ------------------------------- sparse------------------------------------- #


def sparse_array(x):
    return do("COO.from_numpy", x, like="sparse")


def sparse_to_numpy(x):
    return x.todense()


def sparse_transpose(x, axes=None):
    return x.transpose(axes)


def sparse_reshape(x, shape):
    return x.reshape(shape)


def sparse_sum(x, axis=None, keepdims=False, dtype=None, out=None):
    return x.sum(axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def sparse_prod(x, axis=None, keepdims=False, dtype=None, out=None):
    return x.prod(axis=axis, keepdims=keepdims, dtype=dtype, out=out)


def sparse_conj(x):
    return x.conj()


def sparse_real(x):
    return x.real


def sparse_imag(x):
    return x.imag


def sparse_count_nonzero(x):
    return x.nnz


def sparse_random_uniform(low=0.0, high=1.0, size=None, dtype=None, **kwargs):
    def rvs(nnz):
        return do(
            "random.uniform", low, high, (nnz,), dtype=dtype, like="numpy"
        )

    return do("random", size, data_rvs=rvs, **kwargs, like="sparse")


def sparse_random_normal(loc=0.0, scale=1.0, size=None, dtype=None, **kwargs):
    def rvs(nnz):
        return do(
            "random.normal", loc, scale, (nnz,), dtype=dtype, like="numpy"
        )

    return do("random", size, data_rvs=rvs, **kwargs, like="sparse")


_FUNCS["sparse", "array"] = sparse_array
_FUNCS["sparse", "to_numpy"] = sparse_to_numpy
_FUNCS["sparse", "transpose"] = sparse_transpose
_FUNCS["sparse", "reshape"] = sparse_reshape
_FUNCS["sparse", "sum"] = sparse_sum
_FUNCS["sparse", "prod"] = sparse_prod
_FUNCS["sparse", "conj"] = sparse_conj
_FUNCS["sparse", "real"] = sparse_real
_FUNCS["sparse", "real"] = sparse_real
_FUNCS["sparse", "imag"] = sparse_imag
_FUNCS["sparse", "complex"] = complex_add_re_im
_FUNCS["sparse", "count_nonzero"] = sparse_count_nonzero
_FUNCS["sparse", "random.uniform"] = sparse_random_uniform
_FUNCS["sparse", "random.normal"] = sparse_random_normal

# sparse uses numpys __array_func__ interface
for f in (
    "log",
    "log2",
    "log10",
    "exp",
    "sqrt",
    "sign",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
    "tensordot",
    # NB put tensordot here, as sparse.tensordot can produce dense (numpy)
    # arrays but errors when both inputs are dense - we want nested calls to
    # tensordot to handle this
):
    _SUBMODULE_ALIASES["sparse", f] = "numpy"


# ------------------------------- tensorflow -------------------------------- #


def tensorflow_to_numpy(x):
    return x.numpy()


def tensorflow_pad_wrap(tf_pad):
    def numpy_like(array, pad_width, mode="constant", constant_values=0):
        if mode != "constant":
            raise NotImplementedError

        try:
            if len(pad_width) == 1:
                pad_width = pad_width * ndim(array)
        except TypeError:
            pad_width = ((pad_width, pad_width),) * ndim(array)

        return tf_pad(
            array, pad_width, mode="CONSTANT", constant_values=constant_values
        )

    return numpy_like


def tensorflow_where_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(condition, x=None, y=None, **kwargs):
        return tuple(transpose(fn(condition, x, y, **kwargs)))

    return numpy_like


def tensorflow_split_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(ary, indices_or_sections, axis=0, **kwargs):
        if isinstance(indices_or_sections, int):
            return fn(ary, indices_or_sections, axis=axis, **kwargs)
        else:
            diff = do(
                "diff",
                indices_or_sections,
                prepend=0,
                append=shape(ary)[axis],
                like="numpy",
            )
            diff = list(diff)
            return fn(ary, diff, axis=axis)

    return numpy_like


_FUNCS["tensorflow", "to_numpy"] = tensorflow_to_numpy

_SUBMODULE_ALIASES["tensorflow", "log"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "conj"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "real"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "imag"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "power"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "count_nonzero"] = "tensorflow.math"
_SUBMODULE_ALIASES["tensorflow", "diag"] = "tensorflow.linalg"
_SUBMODULE_ALIASES["tensorflow", "trace"] = "tensorflow.linalg"
_SUBMODULE_ALIASES["tensorflow", "tril"] = "tensorflow.linalg"
_SUBMODULE_ALIASES["tensorflow", "triu"] = "tensorflow.linalg"

_FUNC_ALIASES["tensorflow", "sum"] = "reduce_sum"
_FUNC_ALIASES["tensorflow", "min"] = "reduce_min"
_FUNC_ALIASES["tensorflow", "max"] = "reduce_max"
_FUNC_ALIASES["tensorflow", "mean"] = "reduce_mean"
_FUNC_ALIASES["tensorflow", "prod"] = "reduce_prod"
_FUNC_ALIASES["tensorflow", "concatenate"] = "concat"
_FUNC_ALIASES["tensorflow", "clip"] = "clip_by_value"
_FUNC_ALIASES["tensorflow", "arange"] = "range"
_FUNC_ALIASES["tensorflow", "tril"] = "band_part"
_FUNC_ALIASES["tensorflow", "triu"] = "band_part"
_FUNC_ALIASES["tensorflow", "diag"] = "tensor_diag"
_FUNC_ALIASES["tensorflow", "array"] = "convert_to_tensor"
_FUNC_ALIASES["tensorflow", "astype"] = "cast"
_FUNC_ALIASES["tensorflow", "power"] = "pow"
_FUNC_ALIASES["tensorflow", "take"] = "gather"

_CUSTOM_WRAPPERS["tensorflow", "linalg.svd"] = svd_sUV_to_UsVH_wrapper
_CUSTOM_WRAPPERS["tensorflow", "linalg.qr"] = qr_allow_fat
_CUSTOM_WRAPPERS["tensorflow", "linalg.solve"] = binary_allow_1d_rhs_wrap
_CUSTOM_WRAPPERS["tensorflow", "matmul"] = binary_allow_1d_rhs_wrap
_CUSTOM_WRAPPERS["tensorflow", "tril"] = tril_to_band_part
_CUSTOM_WRAPPERS["tensorflow", "triu"] = triu_to_band_part
_CUSTOM_WRAPPERS["tensorflow", "pad"] = tensorflow_pad_wrap
_CUSTOM_WRAPPERS["tensorflow", "where"] = tensorflow_where_wrap
_CUSTOM_WRAPPERS["tensorflow", "split"] = tensorflow_split_wrap
_CUSTOM_WRAPPERS["tensorflow", "random.uniform"] = make_translator(
    [
        ("low", ("minval", 0.0)),
        ("high", ("maxval", 1.0)),
        ("size", ("shape", ())),
    ]
)
_CUSTOM_WRAPPERS["tensorflow", "random.normal"] = make_translator(
    [
        ("loc", ("mean", 0.0)),
        ("scale", ("stddev", 1.0)),
        ("size", ("shape", ())),
    ]
)
_CUSTOM_WRAPPERS["tensorflow", "clip"] = make_translator(
    [
        ("a", ("t", 0.0)),
        ("a_min", ("clip_value_min",)),
        ("a_max", ("clip_value_max",)),
    ]
)


# ---------------------------------- torch ---------------------------------- #


@shape.register("torch")
def torch_shape(x):
    # torch returns a Size object, we want tuple[int]
    return tuple(x.shape)


@size.register("torch")
def torch_size(x):
    return x.numel()


def torch_to_numpy(x):
    return x.detach().cpu().numpy()


def torch_transpose(x, axes=None):
    if axes is None:
        axes = reversed(range(0, x.ndimension()))
    return x.permute(*axes)


def torch_count_nonzero(x):
    return do("sum", x != 0, like="torch")


def torch_astype(x, dtype):
    return x.to(dtype=to_backend_dtype(dtype, like=x))


@functools.lru_cache(None)
def _torch_get_dtype_name(dtype):
    return str(dtype).split(".")[-1]


def torch_get_dtype_name(x):
    return _torch_get_dtype_name(x.dtype)


def torch_real(x):
    # torch doesn't support calling real on real arrays
    try:
        if x.is_complex():
            return x.real
    except AttributeError:
        pass
    return x


def torch_imag(x):
    # torch doesn't support calling imag on real arrays
    try:
        if x.is_complex():
            return x.imag
    except AttributeError:
        pass
    return do("zeros_like", x, like="torch")


def torch_linalg_solve_wrap(fn):
    @binary_allow_1d_rhs_wrap
    def numpy_like(a, b):
        return fn(b, a)[0]

    return numpy_like


def torch_linalg_eigh(x):
    return tuple(do("symeig", x, eigenvectors=True, like="torch"))


def torch_linalg_eigvalsh(x):
    return do("symeig", x, eigenvectors=False, like="torch")[0]


def torch_tensordot_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(a, b, axes=2):
        return fn(a, b, dims=axes)

    return numpy_like


def torch_pad(array, pad_width, mode="constant", constant_values=0):
    if mode != "constant":
        raise NotImplementedError

    try:
        # numpy takes pads like ((0, 0), (1, 1), ... (n-1, n-1))
        # torch takes pads like (n-1, n-1, n-2, n-2, n-3, n-3, ...)
        pad = tuple(itertools.chain.from_iterable(pad_width))[::-1]

        # a single tuple was specified ((a, b),) - use for all axes
        if len(pad) == 2:
            pad = pad * array.ndimension()

    except TypeError:
        # assume int
        pad = (pad_width,) * 2 * array.ndimension()

    return do(
        "nn.functional.pad",
        array,
        pad=pad,
        mode=mode,
        value=constant_values,
        like="torch",
    )


def torch_split_wrap(fn):
    # for torch >=1.8 we can use tensor_split instead, but in current stable
    # release this function has not been added
    @functools.wraps(fn)
    def numpy_like(ary, indices_or_sections, axis=0, **kwargs):
        if isinstance(indices_or_sections, int):
            split_size = shape(ary)[axis] // indices_or_sections
            return fn(ary, split_size, dim=axis, **kwargs)
        else:
            # torch.split doesn't support empty splits
            if len(indices_or_sections) == 0:
                return (ary,)

            diff = do(
                "diff",
                indices_or_sections,
                prepend=0,
                append=shape(ary)[axis],
                like="numpy",
            )
            diff = list(diff)
            return fn(ary, diff, dim=axis)

    return numpy_like


def torch_zeros_ones_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(shape, dtype=None, **kwargs):
        if dtype is not None:
            dtype = to_backend_dtype(dtype, like="torch")
        return fn(shape, dtype=dtype)

    return numpy_like


def torch_eye_wrap(fn):
    @functools.wraps(fn)
    def numpy_like(N, M=None, dtype=None, **kwargs):
        if dtype is not None:
            dtype = to_backend_dtype(dtype, like="torch")
        if M is not None:
            return fn(N, m=M, dtype=dtype, **kwargs)
        else:
            return fn(N, dtype=dtype, **kwargs)

    return numpy_like


_FUNCS["torch", "pad"] = torch_pad
_FUNCS["torch", "real"] = torch_real
_FUNCS["torch", "imag"] = torch_imag
_FUNCS["torch", "astype"] = torch_astype
_FUNCS["torch", "to_numpy"] = torch_to_numpy
_FUNCS["torch", "complex"] = complex_add_re_im
_FUNCS["torch", "transpose"] = torch_transpose
_FUNCS["torch", "count_nonzero"] = torch_count_nonzero
_FUNCS["torch", "get_dtype_name"] = torch_get_dtype_name

_FUNC_ALIASES["torch", "array"] = "tensor"
_FUNC_ALIASES["torch", "clip"] = "clamp"
_FUNC_ALIASES["torch", "concatenate"] = "cat"
_FUNC_ALIASES["torch", "conjugate"] = "conj"
_FUNC_ALIASES["torch", "expand_dims"] = "unsqueeze"
_FUNC_ALIASES["torch", "linalg.expm"] = "matrix_exp"
_FUNC_ALIASES["torch", "max"] = "amax"
_FUNC_ALIASES["torch", "min"] = "amin"
_FUNC_ALIASES["torch", "power"] = "pow"
_FUNC_ALIASES["torch", "random.normal"] = "randn"
_FUNC_ALIASES["torch", "random.uniform"] = "rand"
_FUNC_ALIASES["torch", "split"] = "tensor_split"
_FUNC_ALIASES["torch", "take"] = "index_select"

_SUBMODULE_ALIASES["torch", "linalg.expm"] = "torch"
_SUBMODULE_ALIASES["torch", "random.normal"] = "torch"
_SUBMODULE_ALIASES["torch", "random.uniform"] = "torch"

_CUSTOM_WRAPPERS["torch", "linalg.svd"] = svd_not_full_matrices_wrapper
_CUSTOM_WRAPPERS["torch", "random.normal"] = scale_random_normal_manually
_CUSTOM_WRAPPERS["torch", "random.uniform"] = scale_random_uniform_manually
_CUSTOM_WRAPPERS["torch", "tensordot"] = torch_tensordot_wrap
_CUSTOM_WRAPPERS["torch", "stack"] = make_translator(
    [("arrays", ("tensors",)), ("axis", ("dim", 0))]
)
_CUSTOM_WRAPPERS["torch", "concatenate"] = make_translator(
    [("arrays", ("tensors",)), ("axis", ("dim", 0))]
)
_CUSTOM_WRAPPERS["torch", "tril"] = make_translator(
    [("m", ("input",)), ("k", ("diagonal", 0))]
)
_CUSTOM_WRAPPERS["torch", "triu"] = make_translator(
    [("m", ("input",)), ("k", ("diagonal", 0))]
)
_CUSTOM_WRAPPERS["torch", "clip"] = make_translator(
    [("a", ("input",)), ("a_min", ("min",)), ("a_max", ("max",))]
)
_CUSTOM_WRAPPERS["torch", "ones"] = torch_zeros_ones_wrap
_CUSTOM_WRAPPERS["torch", "zeros"] = torch_zeros_ones_wrap
_CUSTOM_WRAPPERS["torch", "eye"] = torch_eye_wrap
_CUSTOM_WRAPPERS["torch", "empty"] = make_translator([("shape", ("size",))])
_CUSTOM_WRAPPERS["torch", "take"] = make_translator(
    [("a", ("input",)), ("indices", ("index",)), ("axis", ("dim",))]
)
_CUSTOM_WRAPPERS["torch", "expand_dims"] = make_translator(
    [("a", ("input",)), ("axis", ("dim",))]
)

# for older versions of torch, can provide some alternative implementations
_MODULE_ALIASES["torch[alt]"] = "torch"

_FUNCS["torch[alt]", "linalg.eigh"] = torch_linalg_eigh
_FUNCS["torch[alt]", "linalg.eigvalsh"] = torch_linalg_eigvalsh

_SUBMODULE_ALIASES["torch[alt]", "linalg.qr"] = "torch"
_SUBMODULE_ALIASES["torch[alt]", "linalg.svd"] = "torch"
_SUBMODULE_ALIASES["torch[alt]", "linalg.norm"] = "torch"
_SUBMODULE_ALIASES["torch[alt]", "linalg.solve"] = "torch"

_CUSTOM_WRAPPERS["torch[alt]", "split"] = torch_split_wrap
_CUSTOM_WRAPPERS["torch[alt]", "linalg.svd"] = svd_UsV_to_UsVH_wrapper
_CUSTOM_WRAPPERS["torch[alt]", "linalg.qr"] = qr_allow_fat
_CUSTOM_WRAPPERS["torch[alt]", "linalg.solve"] = torch_linalg_solve_wrap


# ---------------------------------- mxnet ---------------------------------- #


def mxnet_to_numpy(x):
    return x.asnumpy()


_MODULE_ALIASES["mxnet"] = "mxnet.numpy"
_FUNCS["mxnet", "to_numpy"] = mxnet_to_numpy
