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
except ImportError:
    pass


def scan_trajectory_gamma(
    traj,
    nep_file,
    asi_file,
    gamma_min=None,
    gamma_max=None,
    min_dist=None,
):
    """
    Sequentially scan trajectory to calculate gamma (MaxVol), filtering candidates.
    Supports Dynamic PCA-MaxVol: Automatically projects descriptors if active set was PCA-reduced.

    Args:
        traj: List of Atoms objects
        nep_file: Path to NEP potential
        asi_file: Path to Active Set Inverse (with optional PCA params)
        gamma_min: Lower bound for selection (default 1.05)
        gamma_max: Safety cut-off (default 10.0)
        min_dist: Minimum allowed interatomic distance (Angstrom).

    Returns:
        selected_atoms: List of atoms fitting criteria.
    """
    calc = NEP(nep_file)
    active_set_data = load_asi(asi_file)

    print("Scanning Trajectory (MaxVol Gamma)...")

    # Default thresholds for MaxVol
    cut_min = gamma_min if gamma_min is not None else 1.05
    cut_safety = gamma_max if gamma_max is not None else 10.0
    print(f"  [Config] Threshold >= {cut_min}, Safety Cutoff > {cut_safety}")

    if min_dist is not None:
        print(f"  [Config] Physical Check: Min Dist >= {min_dist} Å")

    selected_atoms = []

    # Use tqdm context manager for postfix updates
    with tqdm(traj, desc="Scanning") as pbar:
        for i, atoms in enumerate(pbar):
            # Safe calculation
            try:
                calc.calculate(atoms, ["B_projection"])
                B_projection = calc.results["B_projection"]
            except Exception as e:
                tqdm.write(f"  [Error] Failed to calc B_projection for frame {i}: {e}")
                continue

            gamma_values = np.zeros(len(atoms))
            symbols = atoms.get_chemical_symbols()

            frame_max_gamma = 0.0

            for e_sym in set(symbols):
                # ASI Data Keys
                key_asi = e_sym
                key_std = f"{e_sym}_std"  # Legacy support if any
                key_pca_comp = f"{e_sym}_pca_comp"
                key_pca_mean = f"{e_sym}_pca_mean"

                if key_asi not in active_set_data:
                    continue

                M = active_set_data[key_asi]
                indices = [k for k, s in enumerate(symbols) if s == e_sym]
                if not indices:
                    continue

                b_sub = B_projection[indices]

                # --- Apply Dynamic PCA Projection if available ---
                if key_pca_comp in active_set_data and key_pca_mean in active_set_data:
                    pca_comp = active_set_data[key_pca_comp]  # (k, D)
                    pca_mean = active_set_data[key_pca_mean].flatten()  # (D,)

                    # Transform: (X - mean) @ Components.T
                    # Note: sklearn components are (n_comp, n_features).
                    # b_sub is (n_atoms, n_features).
                    # Result = (n_atoms, n_comp). matches M is (n_comp, n_comp).
                    b_sub = np.dot(b_sub - pca_mean, pca_comp.T)

                # --- Calculate MaxVol Gamma ---
                # gamma = norm(b_sub @ M, infinity)
                coeffs = b_sub @ M
                g_val = np.max(np.abs(coeffs), axis=1)

                gamma_values[indices] = g_val
                frame_max_gamma = max(frame_max_gamma, np.max(g_val))

            atoms.arrays["gamma"] = gamma_values

            # Show real-time max value in progress bar to help user estimation
            pbar.set_postfix({"max_val": f"{frame_max_gamma:.2f}"})

            # 1. Safety Cut-off (Explosion Detection)
            if frame_max_gamma > cut_safety:
                tqdm.write(
                    f"[STOP] Frame {i} Gamma ({frame_max_gamma:.3f}) exceeded Safety Limit ({cut_safety}). Stopping scan."
                )
                break

            # 2. Candidate Selection
            if frame_max_gamma >= cut_min:
                # 3. Physical Validity Check (Optional)
                if min_dist is not None:
                    # Calculate atomic distances (mic=True handles PBC)
                    all_dists = atoms.get_all_distances(mic=True)
                    # Mask strict self-distance (diagonal is 0)
                    np.fill_diagonal(all_dists, np.inf)
                    actual_min = np.min(all_dists)

                    if actual_min < min_dist:
                        # Reject this candidate
                        tqdm.write(
                            f"  [Skip] Frame {i} Gamma={frame_max_gamma:.3f} but Min Dist={actual_min:.3f} < {min_dist} Å"
                        )
                        continue

                selected_atoms.append(atoms)
                tqdm.write(
                    f"  [Select] Frame {i}: Max Gamma = {frame_max_gamma:.3f} (>= {cut_min})"
                )

    return selected_atoms


def get_B_projections(traj, nep_file):
    calc = NEP(nep_file)
    with open(nep_file, "r") as f:
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
    mode="adaptive",
):
    print(f"Performing Selection (Mode: {mode})...")

    # Store ASI matrices and potentially PCA params
    active_set_data = {}
    active_set_struct_indices = []

    from sklearn.decomposition import PCA

    for e, matrix in B_projections.items():
        if matrix.size == 0:
            continue

        N, D = matrix.shape

        # --- Dynamic PCA Logic ---
        # Phase 1 & 2: N < D -> Reduce dim to N
        # Phase 3: N >= D -> Full dim

        use_pca = False
        matrix_processed = matrix
        pca_model = None

        if N < D:
            # We want to select "all independent" samples if N < D.
            # MaxVol on N*N matrix will ideally select everything if full rank.
            target_dim = N
            print(
                f"  [Active Learning] Dynamic PCA Triggered for {e}: Samples({N}) < Features({D}). Reducing to dim={target_dim}."
            )

            # Require n_components <= min(N, D) = N
            pca_model = PCA(n_components=target_dim)
            matrix_processed = pca_model.fit_transform(matrix)  # Shape (N, N)

            variance_ratio = np.sum(pca_model.explained_variance_ratio_)
            print(f"  [Active Learning] PCA Explained Variance: {variance_ratio:.4%}")

            use_pca = True
        else:
            print(
                f"  [Active Learning] Full Dimension Mode for {e}: Samples({N}) >= Features({D})."
            )

        # --- Sub-selection ---
        # If N < D and we reduced to N, usually we keep ALL samples.
        # But we still run calculate_maxvol/Adaptive to confirm or just invert directly if N is small?
        # Using the standard flow is safer.

        if mode == "adaptive":
            sampler = AdaptiveMaxVolSampler(
                nep_path=None, m_dim=matrix_processed.shape[1]
            )
            N_sub = matrix_processed.shape[0]
            indices_of_rows = np.arange(N_sub)
            selected_row_indices = sampler.process_batch(
                matrix_processed, indices_of_rows
            )

            A_sel = matrix_processed[selected_row_indices]
            struct_idxs = B_projections_struct_index[e][selected_row_indices]

            active_set_struct_indices.extend(struct_idxs)

            # Get Inverse
            asi_matrix = find_inverse(A_sel)

            active_set_data[e] = asi_matrix
            if use_pca:
                active_set_data[f"{e}_pca_comp"] = pca_model.components_
                active_set_data[f"{e}_pca_mean"] = pca_model.mean_

            print(f"    Adaptive Select {e}: {len(selected_row_indices)} features.")

        else:
            A_sel, idx_sel_structs = calculate_maxvol(
                matrix_processed, B_projections_struct_index[e], batch_size=batch_size
            )
            active_set_struct_indices.extend(idx_sel_structs)

            asi_matrix = find_inverse(A_sel)
            active_set_data[e] = asi_matrix
            if use_pca:
                active_set_data[f"{e}_pca_comp"] = pca_model.components_
                active_set_data[f"{e}_pca_mean"] = pca_model.mean_

            print(f"    MaxVol Select {e}: {A_sel.shape}")

    active_structs = sorted(list(set(active_set_struct_indices)))

    if write_asi:
        print(f"Saving active set (with auto-PCA state) to {asi_filename}...")
        save_asi(active_set_data, filename=asi_filename)

    return active_set_data, active_structs
