"""
Power cone iteration count experiments.

Compares solve performance between native power cones (approx=False)
and SOC approximation (approx=True) for various atoms.

Usage:
    uv run python power_cone_iterations.py CLARABEL
    uv run python power_cone_iterations.py SCS
    uv run python power_cone_iterations.py MOSEK
"""

import argparse
import cvxpy as cp
import numpy as np


def get_problem_stats(prob, solver):
    """Extract problem statistics after solving."""
    data = prob.get_problem_data(solver)
    cones = data[0]['dims']
    cone_info = []

    # ConeDims attributes: zero, nonneg, exp, soc, psd, p3d, pnd
    # SCS also has 'p' as an alias for p3d
    if hasattr(cones, 'zero') and cones.zero > 0:
        cone_info.append(f"Zero: {cones.zero}")
    if hasattr(cones, 'nonneg') and cones.nonneg > 0:
        cone_info.append(f"Nonneg: {cones.nonneg}")
    if hasattr(cones, 'exp') and cones.exp > 0:
        cone_info.append(f"Exp: {cones.exp}")
    if hasattr(cones, 'soc') and cones.soc:
        cone_info.append(f"SOC: {len(cones.soc)} cones")
    if hasattr(cones, 'psd') and cones.psd:
        cone_info.append(f"PSD: {len(cones.psd)} cones")
    if hasattr(cones, 'p3d') and cones.p3d:
        cone_info.append(f"Pow3D: {len(cones.p3d)} cones")
    if hasattr(cones, 'pnd') and cones.pnd:
        cone_info.append(f"GenPow: {len(cones.pnd)} cones")
    # SCS uses 'p' for power cones (list of alpha values)
    if hasattr(cones, 'p') and cones.p:
        cone_info.append(f"Pow3D: {len(cones.p)} cones")

    n_vars = data[0]['A'].shape[1]
    n_constrs = data[0]['A'].shape[0]

    return n_vars, n_constrs, ", ".join(cone_info)


def print_results(label, prob, solver, solver_name):
    """Print formatted results after solving."""
    n_vars, n_constrs, cone_info = get_problem_stats(prob, solver)

    # Get solver stats - all solvers have num_iters and solve_time
    iterations = prob.solver_stats.num_iters
    solve_time = prob.solver_stats.solve_time

    # Handle None iterations (some solvers don't report this)
    if iterations is None:
        iterations = 0
        time_per_iter = 0
    else:
        time_per_iter = (solve_time / iterations * 1000) if iterations > 0 else 0

    print(f"\n{label}:")
    print(f"  Variables:      {n_vars}")
    print(f"  Constraints:    {n_constrs}")
    print(f"  Cones:          {cone_info}")
    print(f"  Status:         {prob.status}")
    print(f"  Iterations:     {iterations}")
    print(f"  Solve time:     {solve_time*1000:.2f} ms")
    print(f"  Time/iteration: {time_per_iter:.2f} ms")
    print(f"  Optimal value:  {prob.value:.6f}")


def run_geo_mean_experiment(solver, solver_name, n=200):
    """Compare geo_mean with power cones vs SOC approximation."""
    print(f"\n{'='*60}")
    print(f"geo_mean n={n} with {solver_name}")
    print('='*60)

    np.random.seed(42)

    # Power cones (approx=False)
    x = cp.Variable(n, pos=True)
    obj = cp.sum(x)
    constraint = [cp.geo_mean(x, approx=False) >= 0.1, x <= 10]
    prob = cp.Problem(cp.Minimize(obj), constraint)
    prob.solve(solver=solver)
    print_results("approx=False (Power Cones)", prob, solver, solver_name)

    # SOC approximation (approx=True)
    x2 = cp.Variable(n, pos=True)
    obj2 = cp.sum(x2)
    constraint2 = [cp.geo_mean(x2, approx=True) >= 0.1, x2 <= 10]
    prob2 = cp.Problem(cp.Minimize(obj2), constraint2)
    prob2.solve(solver=solver)
    print_results("approx=True (SOC)", prob2, solver, solver_name)


def run_pnorm_experiment(solver, solver_name, n=2000, p=1.5):
    """Compare pnorm with power cones vs SOC approximation."""
    print(f"\n{'='*60}")
    print(f"pnorm(x, {p}) n={n} with {solver_name}")
    print('='*60)

    np.random.seed(42)
    c = np.random.randn(n)

    # Power cones (approx=False)
    x = cp.Variable(n)
    obj = c @ x
    constraint = [cp.pnorm(x, p, approx=False) <= 1]
    prob = cp.Problem(cp.Minimize(obj), constraint)
    prob.solve(solver=solver)
    print_results("approx=False (Power Cones)", prob, solver, solver_name)

    # SOC approximation (approx=True)
    x2 = cp.Variable(n)
    obj2 = c @ x2
    constraint2 = [cp.pnorm(x2, p, approx=True) <= 1]
    prob2 = cp.Problem(cp.Minimize(obj2), constraint2)
    prob2.solve(solver=solver)
    print_results("approx=True (SOC)", prob2, solver, solver_name)


def run_power_experiment(solver, solver_name, n=1000, p=0.5):
    """Compare power with power cones vs SOC approximation."""
    print(f"\n{'='*60}")
    print(f"power(x, {p}) n={n} with {solver_name}")
    print('='*60)

    np.random.seed(42)
    c = np.random.randn(n)

    # Power cones (approx=False)
    x = cp.Variable(n, pos=True)
    obj = c @ x
    constraint = [cp.sum(cp.power(x, p, approx=False)) >= n, x <= 10]
    prob = cp.Problem(cp.Minimize(obj), constraint)
    prob.solve(solver=solver)
    print_results("approx=False (Power Cones)", prob, solver, solver_name)

    # SOC approximation (approx=True)
    x2 = cp.Variable(n, pos=True)
    obj2 = c @ x2
    constraint2 = [cp.sum(cp.power(x2, p, approx=True)) >= n, x2 <= 10]
    prob2 = cp.Problem(cp.Minimize(obj2), constraint2)
    prob2.solve(solver=solver)
    print_results("approx=True (SOC)", prob2, solver, solver_name)


def run_inv_prod_experiment(solver, solver_name, n=50):
    """Compare inv_prod with power cones vs SOC approximation."""
    print(f"\n{'='*60}")
    print(f"inv_prod n={n} with {solver_name}")
    print('='*60)

    np.random.seed(42)
    c = np.random.randn(n)

    # Power cones (approx=False)
    x = cp.Variable(n, pos=True)
    obj = c @ x
    constraint = [cp.inv_prod(x, approx=False) <= 1, x >= 0.1, x <= 10]
    prob = cp.Problem(cp.Minimize(obj), constraint)
    prob.solve(solver=solver)
    print_results("approx=False (Power Cones)", prob, solver, solver_name)

    # SOC approximation (approx=True)
    x2 = cp.Variable(n, pos=True)
    obj2 = c @ x2
    constraint2 = [cp.inv_prod(x2, approx=True) <= 1, x2 >= 0.1, x2 <= 10]
    prob2 = cp.Problem(cp.Minimize(obj2), constraint2)
    prob2.solve(solver=solver)
    print_results("approx=True (SOC)", prob2, solver, solver_name)


def main():
    parser = argparse.ArgumentParser(description="Power cone iteration experiments")
    parser.add_argument("solver", choices=["CLARABEL", "SCS", "MOSEK"],
                        help="Solver to use")
    parser.add_argument("--experiment", "-e",
                        choices=["geo_mean", "pnorm", "power", "inv_prod", "all"],
                        default="all", help="Which experiment to run")
    parser.add_argument("--n", type=int, default=None,
                        help="Problem size (default varies by experiment)")
    args = parser.parse_args()

    solver = getattr(cp, args.solver)
    solver_name = args.solver

    experiments = {
        "geo_mean": lambda: run_geo_mean_experiment(solver, solver_name, n=args.n or 200),
        "pnorm": lambda: run_pnorm_experiment(solver, solver_name, n=args.n or 2000),
        "power": lambda: run_power_experiment(solver, solver_name, n=args.n or 1000),
        "inv_prod": lambda: run_inv_prod_experiment(solver, solver_name, n=args.n or 50),
    }

    if args.experiment == "all":
        for exp in experiments.values():
            exp()
    else:
        experiments[args.experiment]()


if __name__ == "__main__":
    main()
