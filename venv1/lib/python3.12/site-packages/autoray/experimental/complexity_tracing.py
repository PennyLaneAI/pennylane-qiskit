"""
Functionality for tracing through an autoray.lazy computation and estimating
the cost and scaling.

In the following there are ``cost_*`` functions that estimate the total cost
of a given operation, including sub-leading factors. There are also
`cost_scaling_*` functions that only consider the leading factor of the cost,
so that we can prime number decompose it and extract the scaling.
"""
import math


def cost_tensordot(x):
    x1, x2, axes = x.args
    shape1, shape2 = x1.shape, x2.shape
    cost = math.prod(shape1) * math.prod(shape2)
    for d in axes[0]:
        cost //= shape1[d]
    return cost


cost_scaling_tensordot = cost_tensordot


def cost_qr(x):
    (A,) = x.deps
    shape = A.shape
    m = max(shape)
    n = min(shape)
    return 2 * m * n**2 - (2 / 3) * n**3


def cost_svd(x):
    (A,) = x.deps
    shape = A.shape
    m = max(shape)
    n = min(shape)
    return 4 * m * n**2 - (4 / 3) * n**3


def cost_eigh(x):
    (A,) = x.deps
    m = A.shape[0]
    return 8 / 3 * m**3


def cost_scaling_linalg(x):
    """Here we only care about the leading factor of the cost, which we need to
    preserve so that we can prime number decompose it.
    """
    (A,) = x.deps
    shape = A.shape
    m = max(shape)
    n = min(shape)
    return m * n**2


cost_scaling_qr = cost_scaling_svd = cost_scaling_linalg


def cost_matmul(x):
    A, B = x.deps
    return A.shape[0] * A.shape[1] * B.shape[1]


cost_scaling_matmul = cost_matmul


def cost_einsum(x):
    eq, *operands = x.args
    lhs = eq.split('->')[0]
    terms = lhs.split(',')
    size_dict = {
        ix: d
        for term, x in zip(terms, operands)
        for ix, d in zip(term, x.shape)
    }
    return math.prod(size_dict.values())


cost_scaling_einsum = cost_einsum


def cost_linear(x):
    return math.prod(x.shape)


def cost_nothing(x):
    return 0


COSTS = {
    "qr": cost_qr,
    "qr_stabilized": cost_qr,
    "qr_stabilized_numba": cost_qr,
    "svd": cost_svd,
    "svd_truncated": cost_svd,
    "svd_truncated_numba": cost_svd,
    "eigh": cost_eigh,
    "linalg_eigh": cost_eigh,
    "tensordot": cost_tensordot,
    "matmul": cost_matmul,
    "einsum": cost_einsum,
    # other cheap ops
    "mul": cost_linear,
    "add": cost_linear,
    "neg": cost_linear,
    "sqrt": cost_linear,
    "cupy_sqrt": cost_linear,
    "pow": cost_linear,
    "truediv": cost_linear,
    "log10": cost_linear,
    "cupy_log10": cost_linear,
    "norm": cost_linear,
    "linalg_norm": cost_linear,
    "reshape": cost_linear,
    "conj": cost_linear,
    "conjugate": cost_linear,
    "cupy_conjugate": cost_linear,
    "clip": cost_linear,
    "transpose": cost_linear,
    "torch_transpose": cost_linear,
    "clamp": cost_linear,
    "getitem": cost_nothing,
    "None": cost_nothing,
}

def cost_node(x, allow_missed=True):
    f = x.fn_name
    if f in COSTS:
        return COSTS[f](x)
    elif allow_missed:
        return 0
    else:
        raise ValueError(f"Cost for {f} not implemented.")


def compute_cost(z, print_missed=True):
    C = 0
    missed = {}
    for node in z.descend():
        f = node.fn_name
        if f in COSTS:
            C += COSTS[f](node)
        else:
            missed[f] = missed.get(f, 0) + 1

    if missed and print_missed:
        import warnings
        warnings.warn(f"Missed {missed} in cost computation.")

    return C


COST_SCALINGS = {
    "qr": cost_scaling_qr,
    "qr_stabilized": cost_scaling_qr,
    "qr_stabilized_numba": cost_scaling_qr,
    "svd": cost_scaling_svd,
    "svd_truncated": cost_scaling_svd,
    "svd_truncated_numba": cost_scaling_svd,
    "eigh": cost_scaling_linalg,
    "tensordot": cost_scaling_tensordot,
    "matmul": cost_scaling_matmul,
    "einsum": cost_scaling_einsum,
    # other cheap ops
    "mul": cost_linear,
    "add": cost_linear,
    "neg": cost_linear,
    "sqrt": cost_linear,
    "pow": cost_linear,
    "truediv": cost_linear,
    "log10": cost_linear,
    "norm": cost_linear,
    "reshape": cost_linear,
    "conj": cost_linear,
    "conjugate": cost_linear,
    "clip": cost_linear,
    "transpose": cost_linear,
    "getitem": cost_nothing,
    "None": cost_nothing,
}


def prime_factors(n) -> list[int]:
    fs = []
    if n <= 1:
        return fs

    while n % 2 == 0:
        fs.append(2)
        n = n // 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            fs.append(i)
            n = n / i
    if n > 2:
        fs.append(n)
    return fs


def is_prime(n: int) -> bool:
    for i in range(int(n**0.5), 1, -2 if int(n**0.5) % 2 == 0 else -1):
        if n % i == 0:
            return False
    return False if n in (0, 1) else True


def closest_prime(nt: int) -> int:
    if is_prime(nt):
        return nt
    lower = None
    higher = None
    for i in range(nt if nt % 2 != 0 else nt - 1, 1, -2):
        if is_prime(i):
            lower = i
            break
    c = nt + 1
    while higher is None:
        if is_prime(c):
            higher = c
        else:
            c += 2 if c % 2 != 0 else 1
    return higher if lower is None or higher - nt < nt - lower else lower


def frequencies(it):
    c = {}
    for i in it:
        c[i] = c.get(i, 0) + 1
    return c


def compute_cost_scalings(z, factor_map, print_missed=True):

    counts = {}
    missed = {}

    for node in z.descend():
        f = node.fn_name

        if f in COST_SCALINGS:
            CS = COST_SCALINGS[f](node)
        else:
            missed[f] = missed.get(f, 0) + 1
            continue

        # group operations
        key = (CS, f)
        counts[key] = counts.get(key, 0) + 1

    if missed and print_missed:
        import warnings
        warnings.warn(f"Missed {missed} in cost scaling computation.")

    scalings = []

    for key, freq in counts.items():
        op = {
            "cost": key[0],
            "name": key[1],
            "freq": freq,
        }
        pf = frequencies(prime_factors(op["cost"]))
        for name, factor in factor_map.items():
            op[name] = pf.pop(factor, 0)

        if pf and print_missed:
            import warnings
            warnings.warn(
                f"Missed prime factor(s) {pf} in cost scaling computation, "
                f" for operation {op}."
            )

        scalings.append(op)

    scalings.sort(key=lambda x: x["cost"], reverse=True)
    return scalings
