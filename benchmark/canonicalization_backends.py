"""
Canonicalization backend benchmarks.

These benchmarks compare CVXPY canonicalization backends (SCIPY, COO, CPP, RUST)
on affine atom families and expression-tree shapes that exercise the
matrix-stuffing backends directly.

Structure
---------
- ``BackendCompileCanonicalization`` times ``Problem.get_problem_data`` end to
  end for every (case, backend) pair.
- ``BackendBuildMatrixCanonicalization`` captures the LinOp trees once, then
  times only the backend's ``build_matrix`` call, isolating backend cost from
  the rest of the compilation chain.
- ``DeepExpressionTreeScaling`` / ``WideExpressionTreeScaling`` time
  ``get_problem_data`` as expression-tree depth (``-(-(...-(x)))``) and sum
  width (``A_1 @ x + ... + A_n @ x``) grow.

Backends that are unavailable in the installed CVXPY (e.g. RUST without the
``cvxpy_rust`` extension, COO before CVXPY 1.7) raise ``NotImplementedError``
in ``setup`` so asv reports them as n/a instead of failing.

Expensive problem construction happens once in ``setup``; the timed functions
wrap the prebuilt expressions in a fresh ``Problem`` so the solving-chain cache
cannot leak one backend's canonicalization into another backend's timing while
keeping construction cost out of the timed region.

See ``LINOP_COVERAGE`` below for a checklist mapping every LinOp node type the
backends process to the cases that exercise it.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import reduce
from typing import Callable

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

BACKENDS = ["SCIPY", "COO", "CPP", "RUST"]


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    factory: Callable[[], tuple[cp.Problem, str]]
    supports_cpp: bool = True


def _reshape(expr, shape):
    try:
        return cp.reshape(expr, shape, order="F")
    except TypeError:
        return cp.reshape(expr, shape)


def _vec(expr):
    try:
        return cp.vec(expr, order="F")
    except TypeError:
        return cp.vec(expr)


def _flatten(expr):
    size = int(np.prod(expr.shape)) if expr.shape else 1
    return _reshape(expr, (size,))


def _scalar_to_vector(expr):
    return _reshape(expr, (1,))


def _available_backend(backend: str) -> bool:
    if backend == "SCIPY":
        return True
    if backend == "COO":
        return getattr(cp, "COO_CANON_BACKEND", None) == "COO"
    if backend == "CPP":
        try:
            from cvxpy.cvxcore.python.cppbackend import build_matrix  # noqa: F401
        except ImportError:
            return False
        return True
    if backend == "RUST":
        if getattr(cp, "RUST_CANON_BACKEND", None) != "RUST":
            return False
        try:
            import cvxpy_rust  # noqa: F401
        except ImportError:
            return False
        return True
    return False


def _require_backend(backend: str, supports_cpp: bool = True, case_name: str = "") -> None:
    if not _available_backend(backend):
        raise NotImplementedError(f"{backend} canonicalization backend is not available")
    if backend == "CPP" and not supports_cpp:
        raise NotImplementedError(f"{case_name or 'case'} uses expressions unsupported by CPP")


def _require_cpp_contract(problem: cp.Problem, backend: str, case_name: str) -> None:
    # CVXPY refuses an explicit CPP backend for problems whose expressions the
    # C++ core does not support; report those cells as n/a rather than errors.
    if backend != "CPP":
        return
    checker = getattr(problem, "_supports_cpp", None)
    if checker is not None and not checker():
        raise NotImplementedError(f"{case_name} uses expressions unsupported by CPP")


def _get_backend_instance(
    backend: str,
    id_to_col: dict[int, int],
    param_to_size: dict[int, int],
    param_to_col: dict[int, int],
    param_size_plus_one: int,
    var_length: int,
):
    try:
        from cvxpy.lin_ops.backends import get_backend

        return get_backend(
            backend,
            id_to_col,
            param_to_size,
            param_to_col,
            param_size_plus_one,
            var_length,
        )
    except ImportError:
        from cvxpy.lin_ops.canon_backend import CanonBackend

        return CanonBackend.get_backend(
            backend,
            id_to_col,
            param_to_size,
            param_to_col,
            param_size_plus_one,
            var_length,
        )


def _build_matrix(captured_call: dict, backend: str):
    c = captured_call
    if backend == "CPP":
        from cvxpy.cvxcore.python.cppbackend import build_matrix

        return build_matrix(
            dict(c["id_to_col"]),
            dict(c["param_to_size"]),
            dict(c["param_to_col"]),
            c["var_length"],
            c["constr_length"],
            c["lin_ops"],
        )

    backend_obj = _get_backend_instance(
        backend,
        dict(c["id_to_col"]),
        dict(c["param_to_size"]),
        dict(c["param_to_col"]),
        c["param_size_plus_one"],
        c["var_length"],
    )
    return backend_obj.build_matrix(c["lin_ops"])


def _capture_build_matrix_call(problem: cp.Problem, solver: str) -> dict:
    from cvxpy.cvxcore.python import canonInterface

    captured = []
    original_get_problem_matrix = canonInterface.get_problem_matrix

    def capturing_get_problem_matrix(
        lin_ops,
        var_length,
        id_to_col,
        param_to_size,
        param_to_col,
        constr_length,
        canon_backend=None,
    ):
        captured.append(
            {
                "lin_ops": lin_ops,
                "var_length": var_length,
                "id_to_col": dict(id_to_col),
                "param_to_size": dict(param_to_size),
                "param_to_col": dict(param_to_col),
                "param_size_plus_one": sum(param_to_size.values()),
                "constr_length": constr_length,
            }
        )
        return original_get_problem_matrix(
            lin_ops,
            var_length,
            id_to_col,
            param_to_size,
            param_to_col,
            constr_length,
            canon_backend="SCIPY",
        )

    canonInterface.get_problem_matrix = capturing_get_problem_matrix
    try:
        problem.get_problem_data(solver=solver, canon_backend="SCIPY")
    finally:
        canonInterface.get_problem_matrix = original_get_problem_matrix

    if not captured:
        raise NotImplementedError("problem did not invoke get_problem_matrix")
    return max(captured, key=lambda call: len(call["lin_ops"]))


def _make_core_affine_atoms() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(1)
    x = cp.Variable((8, 6))
    expr = _reshape(cp.transpose(x), (48,))
    odd_even_diff = expr[::2] - expr[1::2]
    scaled = odd_even_diff / 2.5
    objective = cp.Minimize(cp.sum_squares(scaled + rng.standard_normal(24)))
    return cp.Problem(objective), cp.CLARABEL


def _make_matmul_multiply_divide() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(2)
    x = cp.Variable(64)
    dense = rng.standard_normal((96, 64))
    sparse = sp.random(96, 64, density=0.04, format="csc", random_state=2)
    weights = np.linspace(0.5, 1.5, 96)
    expr = cp.multiply(weights, dense @ x) + sparse @ x
    expr = expr / 1.25
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_rmul_promote() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(12)
    x = cp.Variable((12, 16))
    t = cp.Variable()
    right = rng.standard_normal((16, 8))
    target = rng.standard_normal((12, 8))
    # x @ right lowers to an rmul LinOp; adding the scalar variable t promotes it
    expr = x @ right + t
    return cp.Problem(cp.Minimize(cp.sum_squares(expr - target))), cp.CLARABEL


def _make_hstack_vstack() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(3)
    x = cp.Variable(16)
    a = _reshape(rng.standard_normal((16, 16)) @ x, (4, 4))
    b = _reshape(rng.standard_normal((16, 16)) @ x, (4, 4))
    v = cp.vstack([a, b])
    h = cp.hstack([a, cp.transpose(b)])
    expr = cp.vstack([v, cp.transpose(h)])
    return cp.Problem(cp.Minimize(cp.sum_squares(_vec(expr)))), cp.CLARABEL


def _make_concatenate() -> tuple[cp.Problem, str]:
    # cp.concatenate is not supported by the CPP backend
    rng = np.random.default_rng(14)
    x = cp.Variable(16)
    a = _reshape(rng.standard_normal((16, 16)) @ x, (4, 4))
    b = _reshape(rng.standard_normal((16, 16)) @ x, (4, 4))
    expr = cp.concatenate([a, cp.transpose(b)], axis=1)
    expr = cp.concatenate([expr, cp.hstack([a, b])], axis=0)
    return cp.Problem(cp.Minimize(cp.sum_squares(_vec(expr)))), cp.CLARABEL


def _make_diag_trace_kron() -> tuple[cp.Problem, str]:
    x = cp.Variable(8)
    diag_matrix = cp.diag(x)
    kron_right = cp.kron(np.eye(3), diag_matrix) @ np.ones(24)  # kron_r: constant left
    kron_left = cp.kron(diag_matrix, np.eye(2)) @ np.ones(16)  # kron_l: expression left
    pieces = [
        _flatten(kron_right),
        _flatten(kron_left),
        _flatten(cp.upper_tri(diag_matrix)),
        _flatten(cp.diag(diag_matrix)),
        _scalar_to_vector(cp.trace(diag_matrix)),
    ]
    expr = cp.hstack(pieces)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_convolve() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(4)
    x = cp.Variable(80)
    kernel = rng.standard_normal(17)
    target = rng.standard_normal(96)
    conv = cp.convolve if hasattr(cp, "convolve") else cp.conv
    expr = conv(kernel, x) - target
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_cone_atoms_composite() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(13)
    n, m = 60, 90
    x = cp.Variable(n)
    a = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    p_half = rng.standard_normal((n, n)) / n
    p = p_half.T @ p_half + 0.1 * np.eye(n)
    # nonlinear atoms decompose into affine LinOps + cones before reaching the
    # backend; this case represents that whole-chain canonicalization pressure
    objective = cp.Minimize(
        cp.norm1(x) + cp.sum(cp.huber(a @ x - b, M=1.0)) + cp.quad_form(x, p)
    )
    return cp.Problem(objective, [cp.norm(x, 2) <= 10.0]), cp.CLARABEL


def _make_nd_array_ops() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(5)
    x = cp.Variable((2, 3, 4))
    p = cp.Parameter((3, 4), value=rng.standard_normal((3, 4)))
    broadcast = cp.broadcast_to(p, (2, 3, 4))
    expr = cp.sum(cp.transpose(x + broadcast, axes=(2, 1, 0)), axis=(0, 1))
    expr = expr + cp.sum(x[:, 1, :], axis=1)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_nd_matmul() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(6)
    x = cp.Variable((2, 3, 4))
    left = rng.standard_normal((2, 5, 3))
    try:
        expr = left @ x
    except Exception as exc:  # noqa: BLE001
        raise NotImplementedError("CVXPY checkout does not support ND matmul") from exc
    return cp.Problem(cp.Minimize(cp.sum_squares(_vec(expr)))), cp.CLARABEL


def _make_einsum() -> tuple[cp.Problem, str]:
    if not hasattr(cp, "einsum"):
        raise NotImplementedError("CVXPY checkout does not expose cp.einsum")
    rng = np.random.default_rng(7)
    x = cp.Variable((4, 5))
    left = rng.standard_normal((3, 4))
    right = rng.standard_normal((5, 2))
    expr = cp.einsum("ij,jk,kl->il", left, x, right)
    return cp.Problem(cp.Minimize(cp.sum_squares(_vec(expr)))), cp.CLARABEL


def _make_deep_neg_tree() -> tuple[cp.Problem, str]:
    x = cp.Variable(96)
    expr = x
    for _ in range(64):
        expr = -expr
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_wide_sum_tree() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(8)
    x = cp.Variable(48)
    terms = [rng.standard_normal((48, 48)) @ x for _ in range(64)]
    expr = reduce(lambda lhs, rhs: lhs + rhs, terms)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_parameterized_lp() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(9)
    n_vars = 80
    n_constraints = 160
    x = cp.Variable(n_vars)
    a_param = cp.Parameter((n_constraints, n_vars), value=rng.standard_normal(
        (n_constraints, n_vars)
    ))
    b_param = cp.Parameter(n_constraints, value=rng.standard_normal(n_constraints) + 10)
    c = rng.standard_normal(n_vars)
    problem = cp.Problem(cp.Minimize(c @ x), [a_param @ x <= b_param, x >= 0])
    return problem, cp.CLARABEL


def _gini_portfolio_problem(dense_pairs: np.ndarray, returns: np.ndarray) -> cp.Problem:
    n_times, n_assets = returns.shape
    pair_count = dense_pairs.shape[0]
    d = cp.Variable((pair_count, 1))
    w = cp.Variable((n_assets, 1))
    ret_w = cp.Variable((n_times, 1))
    all_pairs_ret_diff = dense_pairs @ ret_w
    constraints = [
        ret_w == returns @ w,
        d >= all_pairs_ret_diff,
        d >= -all_pairs_ret_diff,
        w >= 0,
        cp.sum(w) == 1,
    ]
    objective = cp.Minimize(cp.sum(d) / (n_times * (n_times - 1)))
    return cp.Problem(objective, constraints)


def _make_murray_dense_constant() -> tuple[cp.Problem, str]:
    # Scaled-down gini_portfolio.Murray: a dense ndarray constant that is
    # ~98.9% zeros (density 2/180 ~ 0.011, below cvxpy's
    # SPARSE_DENSITY_THRESHOLD of 0.05) multiplies an affine expression.
    rng = np.random.default_rng(10)
    n_assets = 20
    n_times = 180
    pair_count = n_times * (n_times - 1) // 2
    returns = rng.standard_normal((n_times, n_assets)) / 1000

    j_idx, i_idx = np.triu_indices(n_times, k=1)
    dense_pairs = np.zeros((pair_count, n_times))
    rows = np.arange(pair_count)
    dense_pairs[rows, i_idx] = 1.0
    dense_pairs[rows, j_idx] = -1.0

    return _gini_portfolio_problem(dense_pairs, returns), cp.CLARABEL


def _make_murray_dense_above_threshold() -> tuple[cp.Problem, str]:
    # Same problem shape, but the constant has ~15% nonzeros — above
    # SPARSE_DENSITY_THRESHOLD — so sparsification heuristics must not fire.
    rng = np.random.default_rng(10)
    n_assets = 20
    n_times = 180
    pair_count = n_times * (n_times - 1) // 2
    returns = rng.standard_normal((n_times, n_assets)) / 1000

    mask = rng.random((pair_count, n_times)) < 0.15
    signs = np.where(rng.random((pair_count, n_times)) < 0.5, -1.0, 1.0)
    dense_pairs = np.where(mask, signs, 0.0)

    return _gini_portfolio_problem(dense_pairs, returns), cp.CLARABEL


def _make_kron_diag_dense_affine() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(11)
    n = 8
    reps = 4
    g = cp.Variable((n, n), symmetric=True)
    v = rng.standard_normal((6, n))
    diag_expr = _reshape(cp.diag(v @ g @ v.T), (6, 1))
    expr = cp.kron(np.ones((reps, 1)), diag_expr)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr)), [g >> 0]), cp.CLARABEL


# LinOp node-type coverage checklist. Keys are the LinOp types defined in
# cvxpy/lin_ops/lin_op.py — i.e. everything a canonicalization backend's
# build_matrix dispatch can see. Values name the cases/classes here that
# exercise the type. "no_op" is not reachable from user expressions and is
# intentionally uncovered.
LINOP_COVERAGE = {
    "variable": ["all cases"],
    "scalar_const": ["core_affine_atoms", "murray_dense_constant"],
    "dense_const": ["matmul_multiply_divide", "murray_dense_constant", "wide_sum_tree"],
    "sparse_const": ["matmul_multiply_divide"],
    "param": ["parameterized_lp", "nd_array_ops"],
    "sum": ["matmul_multiply_divide", "wide_sum_tree", "WideExpressionTreeScaling"],
    "neg": ["core_affine_atoms", "deep_neg_tree", "DeepExpressionTreeScaling"],
    "promote": ["rmul_promote"],
    "mul": ["matmul_multiply_divide", "murray_dense_constant"],
    "rmul": ["rmul_promote", "kron_diag_dense_affine"],
    "mul_elem": ["matmul_multiply_divide", "einsum"],
    "div": ["core_affine_atoms", "matmul_multiply_divide"],
    "index": ["core_affine_atoms", "nd_array_ops"],
    "transpose": ["core_affine_atoms", "hstack_vstack", "nd_array_ops"],
    "reshape": ["core_affine_atoms", "hstack_vstack", "diag_trace_kron"],
    "broadcast_to": ["nd_array_ops"],
    "hstack": ["hstack_vstack", "diag_trace_kron"],
    "vstack": ["hstack_vstack"],
    "concatenate": ["concatenate"],
    "sum_entries": ["murray_dense_constant", "nd_array_ops", "einsum"],
    "trace": ["diag_trace_kron"],
    "diag_vec": ["diag_trace_kron"],
    "diag_mat": ["diag_trace_kron", "kron_diag_dense_affine"],
    "upper_tri": ["diag_trace_kron"],
    "conv": ["convolve"],
    "kron_r": ["diag_trace_kron", "kron_diag_dense_affine"],
    "kron_l": ["diag_trace_kron"],
    "no_op": [],
}


CASES = [
    BenchmarkCase("core_affine_atoms", _make_core_affine_atoms),
    BenchmarkCase("matmul_multiply_divide", _make_matmul_multiply_divide),
    BenchmarkCase("rmul_promote", _make_rmul_promote),
    BenchmarkCase("hstack_vstack", _make_hstack_vstack),
    BenchmarkCase("concatenate", _make_concatenate, supports_cpp=False),
    BenchmarkCase("diag_trace_kron", _make_diag_trace_kron),
    BenchmarkCase("convolve", _make_convolve),
    BenchmarkCase("cone_atoms_composite", _make_cone_atoms_composite),
    BenchmarkCase("nd_array_ops", _make_nd_array_ops, supports_cpp=False),
    BenchmarkCase("nd_matmul", _make_nd_matmul, supports_cpp=False),
    BenchmarkCase("einsum", _make_einsum, supports_cpp=False),
    BenchmarkCase("deep_neg_tree", _make_deep_neg_tree),
    BenchmarkCase("wide_sum_tree", _make_wide_sum_tree),
    BenchmarkCase("parameterized_lp", _make_parameterized_lp),
    BenchmarkCase("murray_dense_constant", _make_murray_dense_constant),
    BenchmarkCase("murray_dense_above_threshold", _make_murray_dense_above_threshold),
    BenchmarkCase("kron_diag_dense_affine", _make_kron_diag_dense_affine),
]
CASE_BY_NAME = {case.name: case for case in CASES}
CASE_NAMES = list(CASE_BY_NAME)


class BackendCompileCanonicalization:
    """End-to-end compile benchmark for canonicalization backends."""

    timeout = 300
    param_names = ["case_name", "backend"]
    params = [CASE_NAMES, BACKENDS]

    def setup(self, case_name: str, backend: str) -> None:
        case = CASE_BY_NAME[case_name]
        _require_backend(backend, case.supports_cpp, case.name)
        self.problem, self.solver = case.factory()
        _require_cpp_contract(self.problem, backend, case.name)

    def time_get_problem_data(self, case_name: str, backend: str) -> None:
        # fresh Problem: defeats the solving-chain cache without timing
        # expression/data construction (done once in setup)
        fresh = cp.Problem(self.problem.objective, self.problem.constraints)
        fresh.get_problem_data(solver=self.solver, canon_backend=backend)


class BackendBuildMatrixCanonicalization:
    """Backend-isolated build_matrix benchmark using captured LinOp trees."""

    timeout = 300
    param_names = ["case_name", "backend"]
    params = [CASE_NAMES, BACKENDS]

    def setup(self, case_name: str, backend: str) -> None:
        case = CASE_BY_NAME[case_name]
        _require_backend(backend, case.supports_cpp, case.name)
        problem, solver = case.factory()
        _require_cpp_contract(problem, backend, case.name)
        self.captured_call = _capture_build_matrix_call(problem, solver)

    def time_build_matrix(self, case_name: str, backend: str) -> None:
        _build_matrix(self.captured_call, backend)


class DeepExpressionTreeScaling:
    """Compile time as nesting depth grows: -(-(...-(x))) with `depth` negations."""

    timeout = 300
    param_names = ["depth", "backend"]
    params = [[4, 32, 256], BACKENDS]

    def setup(self, depth: int, backend: str) -> None:
        _require_backend(backend)
        # every hot tree walk (canonicalize_tree, backend serialization)
        # recurses once or more per nesting level; keep headroom above the
        # default 1000-frame limit
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 20 * depth + 1000))
        x = cp.Variable(96)
        expr = x
        for _ in range(depth):
            expr = -expr
        self.objective = cp.Minimize(cp.sum_squares(expr))

    def time_get_problem_data(self, depth: int, backend: str) -> None:
        fresh = cp.Problem(self.objective)
        fresh.get_problem_data(solver=cp.CLARABEL, canon_backend=backend)


class WideExpressionTreeScaling:
    """Compile time as sum width grows: A_1 @ x + ... + A_width @ x."""

    timeout = 300
    param_names = ["width", "backend"]
    params = [[8, 64, 256], BACKENDS]

    def setup(self, width: int, backend: str) -> None:
        _require_backend(backend)
        rng = np.random.default_rng(21)
        x = cp.Variable(32)
        terms = [rng.standard_normal((32, 32)) @ x for _ in range(width)]
        expr = reduce(lambda lhs, rhs: lhs + rhs, terms)
        self.objective = cp.Minimize(cp.sum_squares(expr))

    def time_get_problem_data(self, width: int, backend: str) -> None:
        fresh = cp.Problem(self.objective)
        fresh.get_problem_data(solver=cp.CLARABEL, canon_backend=backend)
