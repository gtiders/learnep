"""
Pivoted QR (Greedy Gram-Schmidt) 算法实现 (JAX Accelerated)。
用于在亏秩状态下，贪婪地选择与当前子空间距离最远（正交性最好）的向量。
"""

import jax.numpy as jnp
from jax import jit, lax


@jit
def pivoted_qr_jax(A, max_rank, tol=1e-6):
    """
    Perform QR decomposition with column pivoting (Greedy Gram-Schmidt) on A.T
    (i.e. selecting rows of A) to find the most orthogonal subset.

    Args:
        A: (N, m) matrix
        max_rank: Target rank (typically m)
        tol: Tolerance for residual norm

    Returns:
        indices: Selected row indices
        R_diag: Diagonal of R matrix (norms of residuals)
    """
    # We want to select ROWS of A. Standard QR with pivoting selects COLS.
    # So we work on A.T: (m, N).
    # But explicitly transposing huge A is bad.
    # We can implement the greedy selection directly on rows of A.
    # This is effectively "Greedy Gram-Schmidt" on the rows.

    N, m = A.shape

    # State:
    # current_indices (k,)
    # Q: (N, m) - but we don't need full Q, just the basis vectors of the selected subspace?
    # No, actually for row selection we usually maintain the norms of residuals of all remaining rows.

    # Let's implement the iterative residual update approach which is memory efficient.
    # Residuals R = A initially.
    # In each step, pick row with max norm.
    # Orthogonalize all other rows against this picked row.

    # However, for JAX/JIT, in-place updates on huge residual matrix (N, m) are expensive if N is large.
    # But A is immutable in JAX sense.

    # Let's use a slightly different approach: MGS (Modified Gram-Schmidt).
    # We can't easily JIT a loop over N if greedy selection is needed at each step over N.
    # But usually m is small (~100-1000). The loop is over rank k (up to m).

    # Residual norms squared
    # norms = sum(R**2, axis=1)

    # Since we can't update non-selected rows in place efficiently in a pure functional way without copying big arrays,
    # maybe we keep track of the basis Q (k, m) found so far.
    # For each candidate i, residual_norm = ||a_i - Q.T @ (Q @ a_i)||
    # But Q @ a_i is projection coeffs.
    # This requires (N, k) matrix products.

    # Let's implement the standard logic:

    def body_fn(step, state):
        indices, Q, current_norms, key = state

        # Select index with max norm
        # Mask already selected? Max norm calculation handles it if we zero out selected?
        # Or Just use argmax.

        pivot = jnp.argmax(current_norms)

        # Check tolerance (if max residual is small, we are done/rank deficient)
        # We can't break easily in scan/fori_loop, but we can set a flag or just pick dummy.
        # Ideally we use while_loop.

        # New basis vector (unnormalized residual)
        # We need to re-compute the residual perpendicular to current Q
        # v = A[pivot] - sum( (A[pivot].q_j) * q_j )
        # Optimized: We can update the norms incrementally, but here let's compute explicitly for stability?
        # No, recomputing full residual for pivot is O(m*k).

        # Update:
        # We need the full residual of the chosen pivot to normalize it as new basis vector q.
        # But we don't have the full residual matrix. The 'current_norms' is just a tracker.

        # To get the actual residual of the pivot row:
        # r_p = A[pivot] - Q[:step].T @ (Q[:step] @ A[pivot])
        # This is correct.

        # Note: Q is stored as (m, rank) usually? Or (rank, m)?
        # Let's say Q is (max_rank, m).

        # Recover residual
        v = A[pivot]
        projections = jnp.dot(Q[:step], v)  # (step,)
        # sum(proj * q)
        proj_vec = jnp.dot(projections, Q[:step])
        r_vec = v - proj_vec

        norm_r = jnp.linalg.norm(r_vec)

        # Normalize
        q_new = r_vec / (norm_r + 1e-15)  # Avoid div by zero

        # Update Q
        Q = Q.at[step].set(q_new)
        indices = indices.at[step].set(pivot)

        # Update observable norms for next step
        # norm_new^2 = norm_old^2 - |projection_on_q_new|^2
        # proj_new = A @ q_new
        projections_new = jnp.dot(A, q_new)
        new_norms = current_norms - projections_new**2

        # Zero out the selected pivot to avoid re-selecting
        new_norms = new_norms.at[pivot].set(-1.0)

        return indices, Q, new_norms, key

    # Init
    indices = jnp.zeros(max_rank, dtype=jnp.int32)
    Q = jnp.zeros((max_rank, m))  # Basis vectors
    current_norms = jnp.sum(A**2, axis=1)

    # Loop
    # We use lax.fori_loop for fixed iterations up to max_rank
    final_indices, final_Q, _, _ = lax.fori_loop(
        0, max_rank, body_fn, (indices, Q, current_norms, None)
    )

    return final_indices, final_Q


def select_pivoted_qr(A, target_rank=None, tol=1e-5):
    """
    Select up to target_rank rows from A using Pivoted QR / Greedy Gram-Schmidt.
    """
    A = jnp.array(A)
    N, m = A.shape
    if target_rank is None:
        target_rank = m

    target_rank = min(target_rank, N, m)

    indices, Q = pivoted_qr_jax(A, target_rank)

    return indices, Q
