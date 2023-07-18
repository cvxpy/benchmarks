import cvxpy as cp
from numpy.random import default_rng
from scipy import sparse


class QuantumHilbertMatrix:
    """
    Composition of two quantum channels taken from equation #11 
    of https://arxiv.org/pdf/0904.4483.pdf
    """

    def setup(self):
        d = 2  # local hilbert space dimension
        N = 3  # number of qubits
        dim = d**N  # hilbert space dimension of full system

        I = sparse.identity(dim)  # sparse identity matrix  # noqa: E741

        rng = default_rng(0)
        # sparse matrices with the same density as the ones in my problem
        rhs = sparse.random(
            m=dim**2, n=dim**2, density=0.05810546875, random_state=rng
        )
        A = sparse.random(
            m=dim**2, n=dim**2, density=0.29058837890625, random_state=rng
        )

        AxI = sparse.kron(
            A, I
        )  # doing this beforehand to decrease number of expressions in cvxpy cost function

        X = cp.Variable((dim**2, dim**2), PSD=True)  # PSD variable
        lhs = cp.partial_trace(
            cp.kron(I, cp.partial_transpose(X, dims=[dim, dim], axis=0)) @ AxI,
            dims=[dim, dim, dim],
            axis=1,
        )
        cost = cp.sum_squares(lhs - rhs)
        problem = cp.Problem(cp.Minimize(cost))
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == "__main__":
    sdp_problem = QuantumHilbertMatrix()
    sdp_problem.setup()
    sdp_problem.time_compile_problem()

    print(f"compilation time: {sdp_problem.problem._compilation_time:.3f}")
