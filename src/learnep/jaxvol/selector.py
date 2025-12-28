"""
MaxVol 高层选择器接口。
封装了核心 MaxVol 算法，实现了分批处理 (Batching) 和迭代细化 (Refinement) 逻辑，
用于处理大规模数据集。
"""
import numpy as np
import jax.numpy as jnp
from .core import maxvol

def find_inverse(m):
    """
    Compute Moore-Penrose pseudo-inverse using JAX.
    """
    return jnp.linalg.pinv(m, rcond=1e-8)

def calculate_maxvol(
    A,
    struct_index,
    gamma_tol=1.001,
    maxvol_iter=1000,
    batch_size=None,
    n_refinement=10,
    mode="GPU" # Kept for compatibility, but implementation is always JAX (GPU/TPU/CPU)
):
    """
    Calculate MaxVol selection with support for large datasets via batching
    and iterative refinement.
    
    Args:
        A: (N, r) matrix of descriptors/features
        struct_index: (N,) array of structure indices corresponding to rows of A
        gamma_tol: Tolerance for maxvol and refinement
        maxvol_iter: Max iterations for maxvol algorithm
        batch_size: Size of batches for processing large A. If None, process all at once.
        n_refinement: Number of refinement iterations to ensure global coverage.
        mode: Ignored (always uses JAX).
    
    Returns:
        A_selected: The selected submatrix rows.
        struct_index_selected: The indices of the selected structures.
    """
    
    # Ensure inputs are numpy arrays for easy slicing/management
    A = np.array(A)
    struct_index = np.array(struct_index)
    N = len(A)
    
    # --- Case 1: Single Batch (Small enough dataset) ---
    if batch_size is None or batch_size >= N:
        A_jax = jnp.array(A)
        selected_indices = maxvol(A_jax, gamma_tol, maxvol_iter)
        
        # Convert JAX array result back to numpy for indexing
        selected_indices = np.array(selected_indices)
        
        return A[selected_indices], struct_index[selected_indices]

    # --- Case 2: Large Dataset (Batch Processing) ---
    print(f"Processing in batches of size {batch_size}...")
    
    batch_num = int(np.ceil(N / batch_size))
    batch_splits = np.array_split(np.arange(N), batch_num)
    
    A_selected = None
    struct_index_selected = None
    
    # Stage 1: Cumulative MaxVol
    print("Stage 1: Cumulative Selection")
    for i, idxs in enumerate(batch_splits):
        if A_selected is None:
            A_joint = A[idxs]
            idx_joint = struct_index[idxs]
        else:
            A_joint = np.vstack([A_selected, A[idxs]])
            idx_joint = np.hstack([struct_index_selected, struct_index[idxs]])
            
        # Run MaxVol on the joint set
        A_joint_jax = jnp.array(A_joint)
        selected = maxvol(A_joint_jax, gamma_tol, maxvol_iter)
        selected = np.array(selected)
        
        prev_len = len(A_selected) if A_selected is not None else 0
        
        A_selected = A_joint[selected]
        struct_index_selected = idx_joint[selected]
        
        n_added = np.sum(selected >= prev_len) if i > 0 else len(selected)
        print(f"Batch {i+1}/{batch_num}: added/kept {len(selected)} envs.")

    # Stage 2: Refinement
    # Check if any structures in the full pool have high extrapolation grade (gamma)
    # relative to the currently selected set, and add them if so.
    print("Stage 2: Refinement")
    
    for ii in range(n_refinement):
        # Calculate inverse of current active set
        inv = find_inverse(jnp.array(A_selected))
        # Convert inv back to numpy for chunk processing if needed, or keep in JAX
        # inv is typically small (r, r), so we can keep it anywhere.
        inv_np = np.array(inv)
        
        # Scan full A for high gamma
        # We process in chunks to avoid OOM if A is huge
        violator_indices = []
        max_gamma_found = 0.0
        
        chunk_size = batch_size if batch_size else 10000
        
        for b_start in range(0, N, chunk_size):
            b_end = min(b_start + chunk_size, N)
            A_chunk = A[b_start:b_end]
            
            # gamma = max(|A_chunk @ inv|, axis=1)
            # A_chunk: (B, r), inv: (r, r) -> resulting (B, r)
            gamma_chunk = np.abs(A_chunk @ inv_np)
            max_gamma_chunk = np.max(gamma_chunk, axis=1)
            
            batch_max = np.max(max_gamma_chunk)
            if batch_max > max_gamma_found:
                max_gamma_found = batch_max
                
            # Find indices where gamma > tol
            # Note: tolerances can be tricky, using strict >
            bad_locs = np.where(max_gamma_chunk > gamma_tol)[0]
            if len(bad_locs) > 0:
                violator_indices.append(bad_locs + b_start)
        
        print(f"Refinement round {ii+1}: Max gamma = {max_gamma_found:.4f}")
        
        if max_gamma_found <= gamma_tol:
            print("Refinement converged.")
            break
            
        if not violator_indices:
            break
            
        all_violators = np.concatenate(violator_indices)
        print(f"  Found {len(all_violators)} structures with gamma > {gamma_tol}. Re-optimizing...")
        
        # Add violators to the pool and re-run MaxVol
        A_violators = A[all_violators]
        idx_violators = struct_index[all_violators]
        
        A_joint = np.vstack([A_selected, A_violators])
        idx_joint = np.hstack([struct_index_selected, idx_violators])
        
        A_joint_jax = jnp.array(A_joint)
        selected = maxvol(A_joint_jax, gamma_tol, maxvol_iter)
        selected = np.array(selected)
        
        A_selected = A_joint[selected]
        struct_index_selected = idx_joint[selected]
        
    return A_selected, struct_index_selected
