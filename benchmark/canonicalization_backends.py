"""
Canonicalization backend benchmarks.

These benchmarks compare CVXPY canonicalization backends (SCIPY, COO, CPP, RUST)
on affine atom families and expression-tree shapes that exercise the
matrix-stuffing backends directly.

Structure
---------
- ``BackendCompileCanonicalization`` times ``Problem.get_problem_data`` end to
  end for every (case, backend) pair.
- ``BackendBuildMatrixCanonicalization`` captures the LinOp DAG once (cvxpy
  shares subexpression nodes; cvxpy/cvxpy#3423 exploits this), then times only
  the backend's ``build_matrix`` call, isolating backend cost from the rest of
  the compilation chain.
- ``DeepExpressionTreeScaling`` / ``WideExpressionTreeScaling`` time
  ``get_problem_data`` as expression-tree depth (``-(-(...-(x)))``) and sum
  width (``A_1 @ x + ... + A_n @ x``) grow.
- ``WideConstraintScaling`` grows the number of independent constraints —
  each constraint is its own LinOp root, the axis along which backends can
  parallelize.

Backends that are unavailable in the installed CVXPY (RUST without the
``cvxpy_rust`` extension, CPP without the compiled cvxcore extension) raise
``NotImplementedError`` in ``setup`` so asv reports them as n/a instead of
failing.

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


def _available_backend(backend: str) -> bool:
    # SCIPY and COO ship with cvxpy; CPP needs the compiled cvxcore extension
    # and RUST the cvxpy_rust extension, neither of which is guaranteed.
    if backend == "CPP":
        try:
            from cvxpy.cvxcore.python.cppbackend import build_matrix  # noqa: F401
        except ImportError:
            return False
    if backend == "RUST":
        try:
            import cvxpy_rust  # noqa: F401
        except ImportError:
            return False
    return True


def _require_backend(backend: str, supports_cpp: bool = True, case_name: str = "") -> None:
    if not _available_backend(backend):
        raise NotImplementedError(f"{backend} canonicalization backend is not available")
    if backend == "CPP" and not supports_cpp:
        raise NotImplementedError(f"{case_name or 'case'} uses expressions unsupported by CPP")


def _skip_if_cpp_unsupported(problem: cp.Problem, backend: str, case_name: str) -> None:
    # get_problem_data refuses an explicit CPP backend for expressions the C++
    # core does not support; report those cells as n/a rather than errors.
    if backend == "CPP" and not problem._supports_cpp():
        raise NotImplementedError(f"{case_name} uses expressions unsupported by CPP")


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

    from cvxpy.lin_ops.backends import get_backend

    backend_obj = get_backend(
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
    # Compilation invokes get_problem_matrix twice: once for the (scalar)
    # objective and once for the constraints. Benchmark the constraints call —
    # selecting by len(lin_ops) alone ties on single-constraint problems and
    # would pick the trivial objective call.
    return max(captured, key=lambda call: (call["constr_length"], len(call["lin_ops"])))


def _make_core_affine_atoms() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(1)
    x = cp.Variable((8, 6))
    expr = cp.reshape(cp.transpose(x), (48,), order="F")
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
    a = cp.reshape(rng.standard_normal((16, 16)) @ x, (4, 4), order="F")
    b = cp.reshape(rng.standard_normal((16, 16)) @ x, (4, 4), order="F")
    v = cp.vstack([a, b])
    h = cp.hstack([a, cp.transpose(b)])
    expr = cp.vstack([v, cp.transpose(h)])
    return cp.Problem(cp.Minimize(cp.sum_squares(expr.flatten(order="F")))), cp.CLARABEL


def _make_concatenate() -> tuple[cp.Problem, str]:
    # cp.concatenate is not supported by the CPP backend
    rng = np.random.default_rng(14)
    x = cp.Variable(16)
    a = cp.reshape(rng.standard_normal((16, 16)) @ x, (4, 4), order="F")
    b = cp.reshape(rng.standard_normal((16, 16)) @ x, (4, 4), order="F")
    expr = cp.concatenate([a, cp.transpose(b)], axis=1)
    expr = cp.concatenate([expr, cp.hstack([a, b])], axis=0)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr.flatten(order="F")))), cp.CLARABEL


def _make_diag_trace_kron() -> tuple[cp.Problem, str]:
    x = cp.Variable(8)
    diag_matrix = cp.diag(x)
    kron_right = cp.kron(np.eye(3), diag_matrix) @ np.ones(24)  # kron_r: constant left
    kron_left = cp.kron(diag_matrix, np.eye(2)) @ np.ones(16)  # kron_l: expression left
    pieces = [
        kron_right.flatten(order="F"),
        kron_left.flatten(order="F"),
        cp.upper_tri(diag_matrix).flatten(order="F"),
        cp.diag(diag_matrix).flatten(order="F"),
        cp.reshape(cp.trace(diag_matrix), (1,), order="F"),
    ]
    expr = cp.hstack(pieces)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr))), cp.CLARABEL


def _make_convolve() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(4)
    x = cp.Variable(80)
    kernel = rng.standard_normal(17)
    target = rng.standard_normal(96)
    expr = cp.convolve(kernel, x) - target
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
    expr = left @ x
    return cp.Problem(cp.Minimize(cp.sum_squares(expr.flatten(order="F")))), cp.CLARABEL


def _make_einsum() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(7)
    x = cp.Variable((4, 5))
    left = rng.standard_normal((3, 4))
    right = rng.standard_normal((5, 2))
    expr = cp.einsum("ij,jk,kl->il", left, x, right)
    return cp.Problem(cp.Minimize(cp.sum_squares(expr.flatten(order="F")))), cp.CLARABEL


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


def _sparse_density_threshold() -> float:
    # SPARSE_DENSITY_THRESHOLD first shipped after the 1.9.2 release; fall
    # back to its master default on releases that predate it.
    return getattr(cp.settings, "SPARSE_DENSITY_THRESHOLD", 0.05)


# Scaled-down restatement of the Gini portfolio problem in
# benchmark/gini_portfolio.py (its Murray case). Restated rather than imported
# so the pairs-matrix density is a parameter: the two cases below pin one
# density on each side of SPARSE_DENSITY_THRESHOLD, keeping sparsification
# heuristics observable separately from the genuinely-dense path.
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
    # A dense ndarray constant with 2 nonzeros per row (density 2/180 ~ 0.011,
    # below the sparsity threshold) multiplies an affine expression. Guarded so
    # the case reports n/a instead of silently measuring the wrong regime if
    # the threshold ever drops beneath the natural pairs density.
    rng = np.random.default_rng(10)
    n_assets = 20
    n_times = 180
    if 2.0 / n_times >= _sparse_density_threshold():
        raise NotImplementedError(
            "the pairs-matrix density is not below SPARSE_DENSITY_THRESHOLD; "
            "the below-threshold regime does not apply"
        )
    pair_count = n_times * (n_times - 1) // 2
    returns = rng.standard_normal((n_times, n_assets)) / 1000

    j_idx, i_idx = np.triu_indices(n_times, k=1)
    dense_pairs = np.zeros((pair_count, n_times))
    rows = np.arange(pair_count)
    dense_pairs[rows, i_idx] = 1.0
    dense_pairs[rows, j_idx] = -1.0

    return _gini_portfolio_problem(dense_pairs, returns), cp.CLARABEL


def _make_murray_dense_above_threshold() -> tuple[cp.Problem, str]:
    # Same problem shape, but the constant's density sits well above the
    # sparsity threshold (tracking the runtime value, so the case stays
    # meaningful if the threshold changes) — sparsification must not fire.
    rng = np.random.default_rng(10)
    n_assets = 20
    n_times = 180
    pair_count = n_times * (n_times - 1) // 2
    returns = rng.standard_normal((n_times, n_assets)) / 1000

    density = min(3.0 * _sparse_density_threshold(), 0.5)
    mask = rng.random((pair_count, n_times)) < density
    signs = np.where(rng.random((pair_count, n_times)) < 0.5, -1.0, 1.0)
    dense_pairs = np.where(mask, signs, 0.0)

    return _gini_portfolio_problem(dense_pairs, returns), cp.CLARABEL


def _make_shared_subexpressions() -> tuple[cp.Problem, str]:
    # One affine subexpression reused across many terms: the canonicalized
    # LinOp structure is a DAG with shared nodes — the workload targeted by
    # the memoization in cvxpy/cvxpy#3423.
    rng = np.random.default_rng(15)
    x = cp.Variable(64)
    common = rng.standard_normal((64, 64)) @ x
    terms = [cp.sum_squares(common[i:i + 24]) for i in range(0, 40, 2)]
    return cp.Problem(cp.Minimize(sum(terms))), cp.CLARABEL


def _make_kron_diag_dense_affine() -> tuple[cp.Problem, str]:
    rng = np.random.default_rng(11)
    n = 8
    reps = 4
    g = cp.Variable((n, n), symmetric=True)
    v = rng.standard_normal((6, n))
    diag_expr = cp.reshape(cp.diag(v @ g @ v.T), (6, 1), order="F")
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
    "mul": ["matmul_multiply_divide", "murray_dense_constant", "shared_subexpressions"],
    "rmul": ["rmul_promote", "kron_diag_dense_affine"],
    "mul_elem": ["matmul_multiply_divide", "einsum"],
    "div": ["core_affine_atoms", "matmul_multiply_divide"],
    "index": ["core_affine_atoms", "nd_array_ops", "shared_subexpressions"],
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
    BenchmarkCase("shared_subexpressions", _make_shared_subexpressions),
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
        _skip_if_cpp_unsupported(self.problem, backend, case.name)

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
        _skip_if_cpp_unsupported(problem, backend, case.name)
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
    """Compile time as sum width grows: A_1 @ x + ... + A_width @ x.

    Repeated ``+`` flattens into a single AddExpression, so the LinOp sum node
    has ``width`` children — one wide node, not a chain of binary adds.
    """

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


class WideConstraintScaling:
    """Compile time as the number of independent constraints grows.

    Each constraint is its own LinOp root — the axis along which backends can
    parallelize canonicalization across constraints.
    """

    timeout = 300
    param_names = ["n_constraints", "backend"]
    params = [[8, 64, 256], BACKENDS]

    def setup(self, n_constraints: int, backend: str) -> None:
        _require_backend(backend)
        rng = np.random.default_rng(22)
        x = cp.Variable(32)
        self.objective = cp.Minimize(cp.sum(x))
        self.constraints = [
            rng.standard_normal((12, 32)) @ x <= rng.standard_normal(12) + 10.0
            for _ in range(n_constraints)
        ]

    def time_get_problem_data(self, n_constraints: int, backend: str) -> None:
        fresh = cp.Problem(self.objective, self.constraints)
        fresh.get_problem_data(solver=cp.CLARABEL, canon_backend=backend)
