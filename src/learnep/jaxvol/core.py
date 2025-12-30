"""
核心 MaxVol 算法实现 (JAX Accelerated)。
包含基于 JIT 编译的 MaxVol 迭代优化算法，用于在满秩矩阵中寻找最大体积子矩阵。
"""

import jax.numpy as jnp
from jax.scipy.linalg import lu, solve_triangular
from jax import jit, lax


@jit
def maxvol(A, tol=1.05, max_iters=1000):
    """
    JAX implementation of the MaxVol algorithm.
    """
    A = jnp.array(A)  # Ensure input is JAX array
    N, r = A.shape

    # LU Decomposition
    # A = P @ L @ U
    P, L, U = lu(A)

    # Initial indices I
    # P is an (N, N) permutation matrix.
    # We select the indices corresponding to the first r rows of the permuted matrix.
    I = jnp.argmax(P[:, :r], axis=0)  # noqa: E741

    # Compute initial coefficient matrix B
    # B = A @ (A[I])^-1
    # We solve this using the LU factors:
    # A^T = U^T @ L^T @ P^T
    # We want Q such that U^T @ Q = A^T
    Q = solve_triangular(U, A.T, trans=1, lower=False)

    # Then solve L[:r, :]^T @ B^T = Q
    # B^T = solve(L[:r, :]^T, Q)
    # B = (B^T)^T
    B = solve_triangular(L[:r, :], Q, trans=1, unit_diagonal=True, lower=True).T

    # Iterative refinement loop
    def cond_fun(state):
        iter_num, _, _, max_val = state
        return (iter_num < max_iters) & (max_val > tol)

    def body_fun(state):
        iter_num, B, I, _ = state  # noqa: E741

        # Find element with maximum absolute value
        flat_idx = jnp.argmax(jnp.abs(B))
        i, j = jnp.unravel_index(flat_idx, B.shape)

        max_val = jnp.abs(B[i, j])

        # Update indices I
        I = I.at[j].set(i)  # noqa: E741

        # Sherman-Morrison update for B
        bj = B[:, j]
        bi = B[i, :]

        # bi[j] needs to be decremented by 1.0 for the update formula
        bi_mod = bi.at[j].add(-1.0)

        # Update B
        # B -= outer(bj, bi_mod / B[i, j])
        update_term = jnp.outer(bj, bi_mod / B[i, j])
        B = B - update_term

        return (iter_num + 1, B, I, max_val)

    # Initial check for loop condition
    max_val = jnp.max(jnp.abs(B))

    # State: (iteration_count, Matrix B, Indices I, current_max_value)
    init_state = (0, B, I, max_val)

    # Run the loop
    final_state = lax.while_loop(cond_fun, body_fun, init_state)

    # Extract final indices
    _, _, final_I, _ = final_state

    return final_I
