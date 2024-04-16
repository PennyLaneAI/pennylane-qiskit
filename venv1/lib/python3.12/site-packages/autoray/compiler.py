import functools

from .autoray import (
    do,
    infer_backend,
    backend_like,
    tree_map,
    tree_iter,
    tree_flatten,
    tree_unflatten,
    is_array,
)
from . import lazy


class CompilePython:
    """A simple compiler that unravels all autoray calls, optionally sharing
    intermediates and folding constants, converts this to a code object using
    ``compile``, then executes this using ``exec``.

    Parameters
    ----------
    fn : callable
        Function to compile - should have signature
        ``fn(*args, **kwargs) -> array``, with ``args`` and ``kwargs`` any
        nested combination of ``tuple``, ``list`` and ``dict`` objects
        containing arrays (or other constant arguments), and perform array
        operations on these using ``autoray.do``.
    fold_constants : bool, optional
        Whether to fold all constant array operations into the graph, which
        might increase memory usage.
    share_intermediates : bool, optional
        Whether to cache all computational nodes during the trace, so that any
        shared intermediate results can be identified.
    """

    def __init__(self, fn, fold_constants=True, share_intermediates=True):
        self._fn = fn
        self._fold_constants = fold_constants
        self._share_intermediates = share_intermediates
        self._jit_fn = None

    def setup(self, args, kwargs):
        """Convert the example arrays to lazy variables and trace them through
        the function.
        """
        variables = tree_map(lazy.array, (args, kwargs))

        if self._share_intermediates:
            with backend_like("autoray.lazy"), lazy.shared_intermediates():
                outs = self._fn(*variables[0], **variables[1])
        else:
            with backend_like("autoray.lazy"):
                outs = self._fn(*variables[0], **variables[1])

        return lazy.Function(
            variables, outs, fold_constants=self._fold_constants
        )

    def __call__(self, *args, array_backend=None, **kwargs):
        """If necessary, build, then call the compiled function."""
        if self._jit_fn is None:
            self._jit_fn = self.setup(args, kwargs)

        return self._jit_fn(args, kwargs)


class CompileJax:
    """ """

    def __init__(self, fn, enable_x64=None, platform_name=None, **kwargs):
        self._fn = fn
        self._enable_x64 = enable_x64
        self._platform_name = platform_name
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import jax

        if self._enable_x64 is not None:
            import jax

            jax.config.update("jax_enable_x64", self._enable_x64)

        if self._platform_name is not None:
            import jax

            jax.config.update("jax_platform_name", self._platform_name)

        self._jit_fn = jax.jit(self._fn, **self._jit_kwargs)
        self._fn = None

    def __call__(self, *args, array_backend=None, **kwargs):
        if self._jit_fn is None:
            self.setup()
        out = self._jit_fn(*args, **kwargs)
        if array_backend != "jax":
            out = do("asarray", out, like=array_backend)
        return out


class CompileTensorFlow:
    """ """

    def __init__(self, fn, **kwargs):
        self._fn = fn
        kwargs.setdefault("autograph", False)
        self._jit_fn = None
        self._jit_kwargs = kwargs

    def setup(self):
        import tensorflow as tf

        self._jit_fn = tf.function(**self._jit_kwargs)(self._fn)
        self._fn = None

    def __call__(self, *args, array_backend=None, **kwargs):
        if self._jit_fn is None:
            self.setup()
        out = self._jit_fn(*args, **kwargs)
        if array_backend != "tensorflow":
            out = do("asarray", out, like=array_backend)
        return out


class CompileTorch:
    """ """

    def __init__(self, fn, **kwargs):
        import torch

        self.torch = torch

        if not hasattr(fn, "__name__") and isinstance(fn, functools.partial):
            # torch jit.trace requires fn.__name__ and others
            functools.update_wrapper(fn, fn.func)

        self._fn = fn
        self._jit_fn = None
        kwargs.setdefault("check_trace", False)
        self._jit_kwargs = kwargs

    def setup(self, *args, **kwargs):
        flat_tensors, ref_tree = tree_flatten((args, kwargs), get_ref=True)

        def flat_fn(flat_tensors):
            args, kwargs = tree_unflatten(flat_tensors, ref_tree)
            return self._fn(*args, **kwargs)

        self._jit_fn = self.torch.jit.trace(
            flat_fn, [flat_tensors], **self._jit_kwargs
        )

    def __call__(self, *args, array_backend=None, **kwargs):
        if array_backend != "torch":
            # torch doesn't handle numpy arrays itself
            args = tree_map(self.torch.as_tensor, args, is_array)
        if self._jit_fn is None:
            self.setup(*args, **kwargs)
        out = self._jit_fn(tree_flatten((args, kwargs)))
        if array_backend != "torch":
            out = do("asarray", out, like=array_backend)
        return out


_backend_lookup = {}

_compiler_lookup = {
    "jax": CompileJax,
    "tensorflow": CompileTensorFlow,
    "torch": CompileTorch,
}


class AutoCompiled:
    """Just in time compile a ``autoray.do`` using function. See the main
    wrapper ``autojit``.
    """

    def __init__(self, fn, backend=None, compiler_opts=None):
        self._fn = fn
        self._backend = backend
        self._compiled_fns = {}
        if compiler_opts is None:
            self._compiler_kwargs = {}
        else:
            self._compiler_kwargs = compiler_opts

    def __call__(self, *args, backend=None, **kwargs):
        array_backend = infer_backend(
            next(tree_iter((args, kwargs), is_array))
        )

        if backend is None:
            if self._backend is None:
                # no backend specified anywhere, use the array backend
                backend = array_backend
            else:
                # use the backend specified at init
                backend = self._backend

        # work out which compiler to use for combo of backend and array backend
        try:
            key = _backend_lookup[backend, array_backend]
        except KeyError:
            if backend in _compiler_lookup:
                key = backend
            else:
                key = f"python-{array_backend}"
            _backend_lookup[backend, array_backend] = key

        try:
            fn_compiled = self._compiled_fns[key]
        except KeyError:
            if "python" in key:
                backend = "python"
            backend_compiler = _compiler_lookup.get(backend, CompilePython)
            compiler_kwargs = self._compiler_kwargs.get(backend, {})
            fn_compiled = backend_compiler(self._fn, **compiler_kwargs)
            self._compiled_fns[key] = fn_compiled

        return fn_compiled(*args, array_backend=array_backend, **kwargs)


def autojit(fn=None, *, backend=None, compiler_opts=None):
    """Just-in-time compile an ``autoray`` function, automatically choosing
    the backend based on the input arrays, or via keyword argument.

    The backend used to do the compilation can be set in three ways:

        1. Automatically based on the arrays the function is called with,
           i.e. ``cfn(*torch_arrays)`` will use ``torch.jit.trace``.
        2. In this wrapper, ``@autojit(backend='jax')``, to provide a
           specific default instead.
        3. When you call the function ``cfn(*arrays, backend='torch')`` to
           override on a per-call basis.

    If the arrays supplied are of a different backend type to the compiler,
    then the returned array will also be converted back, i.e.
    ``cfn(*numpy_arrays, backend='tensorflow')`` will return a ``numpy`` array.

    The ``'python'`` backend simply extracts and unravels all the ``do`` calls
    into a code object using ``compile`` which is then run with ``exec``.
    This makes use of shared intermediates and constant folding, strips
    away any python scaffoliding, and is compatible with any library, but the
    resulting function is not 'low-level' in the same way as the other
    backends.

    Parameters
    ----------
    fn : callable
        The autoray function to compile.
    backend : {None, 'python', 'jax', 'torch', 'tensorflow'}, optional
        If set, use this as the default backend.
    compiler_opts : dict[dict], optional
        Dict of dicts when you can supply options for each compiler backend
        separately, e.g.:
        ``@autojit(compiler_opts={'tensorflow': {'jit_compile': True}})``.

    Returns
    -------
    cfn : callable
        The function with auto compilation.
    """
    kws = dict(backend=backend, compiler_opts=compiler_opts)
    if fn is None:
        return functools.partial(autojit, **kws)
    return functools.wraps(fn)(AutoCompiled(fn, **kws))
