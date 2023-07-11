import cvxpy as cp
import numpy as np


class QuantumHilbertMatrix:
    """
    Taken from [...]
    """

    def setup(self):
        # Import packages.
        import cvxpy as cp
        import numpy as np
        from scipy import sparse

        d = 2 # local hilbert space dimension
        N = 3 # number of qubits
        dim = d**N # hilbert space dimension of full system

        I = sparse.identity(dim) # sparse identity matrix
        # sparse matrices with the same density as the ones in my problem
        rhs = sparse.random(m=dim**2,n=dim**2, density=0.05810546875)
        A = sparse.random(m=dim**2,n=dim**2, density=0.29058837890625)

        IxA = sparse.kron(I,A) # doing this beforehand to decrease number of expressions in cvxpy cost function

        X = cp.Variable((dim**2,dim**2), PSD=True) # PSD variable
        lhs =  cp.partial_trace(cp.kron(I, cp.partial_transpose(X, dims=[dim, dim], axis=0)) @ IxA, dims=[dim, dim, dim], axis=1)
        cost = cp.sum_squares(lhs - rhs)
        problem = cp.Problem(cp.Minimize(cost))
        self.problem = problem

    def time_compile_problem(self):
        self.problem.get_problem_data(solver=cp.SCS)


if __name__ == '__main__':
    sdp_problem = QuantumHilbertMatrix()
    sdp_problem.setup()
    sdp_problem.time_compile_problem()
