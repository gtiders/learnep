
"""
NEP 势函数专用工具集。
包含特征提取 (Get B Projections)、Active Set 计算以及外推等级 (Gamma) 计算等
与计算材料学具体应用场景相关的逻辑。
"""
import numpy as np
from tqdm import tqdm
from .selector import calculate_maxvol, find_inverse
from .asi import save_asi, load_asi
from .adaptive import AdaptiveMaxVolSampler

try:
    from pynep.calculate import NEP
    from ase import Atoms
except ImportError:
    pass

def scan_trajectory_gamma(
    traj, 
    nep_file, 
    asi_file, 
    gamma_min=None, 
    gamma_max=None,
    auto_stop_qr=False,
    std_tol=1e-4
):
    """
    Sequentially scan trajectory to calculate gamma, filtering candidates and handling early stopping.
    
    Args:
        traj: List of Atoms objects
        nep_file: Path to NEP potential
        asi_file: Path to Active Set Inverse
        gamma_min: Lower bound for selecting candidates (inclusive)
        gamma_max: Upper bound for safety cut-off. If exceeded, SCANNIG STOPS.
        auto_stop_qr: If True, stop scanning when standard deviation of gamma scores stabilizes (QR mode).
        std_tol: Tolerance for std dev change to trigger stop.
        
    Returns:
        selected_atoms: List of atoms fitting criteria found before stop.
    """
    calc = NEP(nep_file)
    active_set_inverse = load_asi(asi_file)
    
    # Check Matrix Type
    matrix_types = {}
    is_qr_mode = False
    for e, M in active_set_inverse.items():
        k = M.shape[1]
        m = M.shape[0]
        # Heuristic for Basis (QR) vs Inverse (MaxVol)
        # If matrix is orthonormal columns, it's a basis.
        if k < m:
            gram = M.T @ M
            if np.allclose(gram, np.eye(k), atol=1e-4):
                matrix_types[e] = "basis"
                is_qr_mode = True
        else:
            matrix_types[e] = "inverse"
            
    print(f"Scanning Gamma (Mode: {'QR/Basis' if is_qr_mode else 'MaxVol/Inverse'})...")
    
    if auto_stop_qr and not is_qr_mode:
        print("Warning: --auto-stop (Statistical Termination) is only applicable in QR/Rank-Deficient mode. It will be ignored.")
    
    selected_atoms = []
    gamma_history = []
    running_std = 0.0
    
    # Defaults
    g_min = gamma_min if gamma_min is not None else -1.0
    g_max_cutoff = gamma_max if gamma_max is not None else float('inf')
    
    for i, atoms in enumerate(tqdm(traj, desc="Scanning")):
        calc.calculate(atoms, ["B_projection"])
        B_projection = calc.results["B_projection"]
        
        gamma_values = np.zeros(len(atoms))
        symbols = atoms.get_chemical_symbols()
        
        # Calculate Gamma for this frame
        frame_max_gamma = 0.0
        
        for e, M in active_set_inverse.items():
            indices = [k for k, s in enumerate(symbols) if s == e]
            if not indices:
                continue
            
            b_sub = B_projection[indices]
            
            if matrix_types.get(e) == "basis":
                # Distance metric
                coeffs = b_sub @ M
                proj = coeffs @ M.T
                resid = b_sub - proj
                g_val = np.linalg.norm(resid, axis=1)
            else:
                # MaxVol metric
                coeffs = b_sub @ M
                g_val = np.max(np.abs(coeffs), axis=1)
            
            gamma_values[indices] = g_val
            frame_max_gamma = max(frame_max_gamma, np.max(g_val))
            
        atoms.arrays["gamma"] = gamma_values
        
        # 1. Safety Cut-off (Explosion Detection)
        if frame_max_gamma > g_max_cutoff:
            print(f"[STOP] Frame {i} Gamma ({frame_max_gamma:.4f}) exceeded Max Limit ({g_max_cutoff}). Stopping scan.")
            break
            
        # 2. Candidate Selection
        # If frame max gamma is within range (and not exploded), is it a candidate?
        # Typically we want structures with high gamma.
        if frame_max_gamma >= g_min:
             selected_atoms.append(atoms)
             
        # 3. QR Auto Termination (Statistical Stability)
        if is_qr_mode and auto_stop_qr:
            # Track the characteristic gamma of the frame (max? mean?)
            # User said "new structure score". Usually max is the "score" of the structure.
            gamma_history.append(frame_max_gamma)
            
            if len(gamma_history) > 20: # Warmup
                current_std = np.std(gamma_history)
                prev_std = running_std
                
                # Update running
                running_std = current_std
                
                delta = abs(current_std - prev_std)
                # If std is stable? Or if std assumes a "noise pattern"?
                # "std dev change ... stop"
                if i > 50 and delta < std_tol:
                    print(f"[AUTO-STOP] Gamma distribution stabilized (Std Dev Change {delta:.2e} < {std_tol}). Stopping.")
                    break
        
    return selected_atoms



# ... (Previous get_B_projections and get_active_set) ...
def get_B_projections(traj, nep_file):
    calc = NEP(nep_file)
    with open(nep_file, 'r') as f:
        first_line = f.readline()
        elements = first_line.split(" ")[2:-1]
    
    B_projections = {e: [] for e in elements}
    B_projections_struct_index = {e: [] for e in elements}
    
    print("Calculating B projections...")
    for idx, atoms in enumerate(tqdm(traj, desc="Processing")):
        calc.calculate(atoms, ["B_projection"])
        B_res = calc.results["B_projection"]
        syms = atoms.get_chemical_symbols()
        
        for e in elements:
            mask = [s == e for s in syms]
            if any(mask):
                B_e = B_res[mask]
                B_projections[e].append(B_e)
                B_projections_struct_index[e].extend([idx] * len(B_e))

    final_B = {}
    final_indices = {}
    for e in elements:
        if B_projections[e]:
            final_B[e] = np.vstack(B_projections[e])
            final_indices[e] = np.array(B_projections_struct_index[e])
        else:
            final_B[e] = np.empty((0, 0))
            final_indices[e] = np.array([])
    return final_B, final_indices

def get_active_set(
    B_projections,
    B_projections_struct_index,
    write_asi=True,
    batch_size=10000,
    asi_filename="active_set.asi",
    mode="adaptive" 
):
    print(f"Performing Selection (Mode: {mode})...")
    active_set_matrices = {} 
    active_set_struct_indices = []
    
    for e, matrix in B_projections.items():
        if matrix.size == 0:
            continue
            
        if mode == "adaptive":
            sampler = AdaptiveMaxVolSampler(
                nep_path=None, 
                m_dim=matrix.shape[1]
            )
            N = matrix.shape[0]
            indices_of_rows = np.arange(N)
            selected_row_indices = sampler.process_batch(matrix, indices_of_rows)
            
            A_sel = matrix[selected_row_indices]
            struct_idxs = B_projections_struct_index[e][selected_row_indices]
            
            active_set_struct_indices.extend(struct_idxs)
            
            matrix_to_save = sampler.get_asi_matrix()
            if matrix_to_save is not None:
                active_set_matrices[e] = matrix_to_save
            else:
                active_set_matrices[e] = np.zeros((matrix.shape[1], 0))
            
            print(f"Adaptive Select {e}: {len(selected_row_indices)} features ({sampler.state})")
            
        else:
            A_sel, idx_sel_structs = calculate_maxvol(
                matrix, 
                B_projections_struct_index[e], 
                batch_size=batch_size
            )
            active_set_struct_indices.extend(idx_sel_structs)
            active_set_matrices[e] = find_inverse(A_sel)
            print(f"MaxVol Select {e}: {A_sel.shape}")
        
    active_structs = sorted(list(set(active_set_struct_indices)))
        
    if write_asi:
        print(f"Saving active set inverse/basis to {asi_filename}...")
        save_asi(active_set_matrices, filename=asi_filename)
        
    return active_set_matrices, active_structs
