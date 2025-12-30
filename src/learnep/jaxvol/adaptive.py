"""
自适应 MaxVol 采样器 (Adaptive Sampler)。
实现了从亏秩 (Rank-Deficient) 到满秩 (Full-Rank) 的自动过渡策略：
1. 亏秩阶段：使用 Pivoted QR 算法，以外推距离为指标构建基底。
2. 满秩阶段：切换至标准 MaxVol 算法，优化和扩展基底。
"""

import numpy as np


class AdaptiveMaxVolSampler:
    def __init__(self, nep_path, m_dim=None, output_asi="active_set.asi"):
        self.nep_path = nep_path
        self.output_asi = output_asi

        self.m = m_dim
        self.state = "RANK_DEFICIENT"

        self.selected_features = []
        self.selected_indices = []

        self.Q_basis = None
        self.A_inv = None

        # Internal stability thresholds
        self.qr_threshold = 1e-3
        self.maxvol_threshold = 1.05

    def _update_basis(self, new_vectors):
        """
        Update the orthonormal basis Q with new vectors using Gram-Schmidt.
        """
        for v in new_vectors:
            if len(self.selected_features) == 0:
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    q = v / norm
                    self.Q_basis = q.reshape(1, -1)
                    self.selected_features.append(v)
            else:
                coeffs = self.Q_basis @ v
                proj = coeffs @ self.Q_basis
                r = v - proj
                dist = np.linalg.norm(r)

                if dist > 1e-10:
                    q = r / dist
                    self.Q_basis = np.vstack([self.Q_basis, q])
                    self.selected_features.append(v)

    def process_batch(self, candidate_features, candidate_global_indices):
        """
        Process a batch of candidates and select those that improve the basis.
        """
        if self.m is None:
            self.m = candidate_features.shape[1]

        selected_local_indices = []

        # Phase 1: Rank Deficient
        if self.state == "RANK_DEFICIENT":
            if self.Q_basis is None:
                norms = np.linalg.norm(candidate_features, axis=1)
                idx = np.argmax(norms)
                self._update_basis([candidate_features[idx]])
                selected_local_indices.append(idx)

            coeffs = candidate_features @ self.Q_basis.T
            projections = coeffs @ self.Q_basis
            residuals = candidate_features - projections
            dists = np.linalg.norm(residuals, axis=1)

            current_dists = dists.copy()
            while True:
                best_idx = np.argmax(current_dists)
                max_dist = current_dists[best_idx]

                if max_dist < self.qr_threshold:
                    break

                selected_local_indices.append(best_idx)
                v_best = candidate_features[best_idx]

                self._update_basis([v_best])

                if len(self.selected_features) >= self.m:
                    self.state = "FULL_RANK"
                    A_mat = np.array(self.selected_features)
                    self.A_inv = np.linalg.pinv(A_mat)
                    break

                q_new = self.Q_basis[-1]
                new_proj_coeffs = candidate_features @ q_new
                d_sq = current_dists**2 - new_proj_coeffs**2
                d_sq[d_sq < 0] = 0
                current_dists = np.sqrt(d_sq)
                current_dists[best_idx] = 0.0

        # Phase 2: Full Rank
        if self.state == "FULL_RANK":
            if self.A_inv is None:
                A_mat = np.array(self.selected_features[: self.m])
                self.A_inv = np.linalg.inv(A_mat)

            coeffs = candidate_features @ self.A_inv
            gammas = np.max(np.abs(coeffs), axis=1)

            high_gamma_indices = np.where(gammas > self.maxvol_threshold)[0]

            for idx in high_gamma_indices:
                if idx not in selected_local_indices:
                    selected_local_indices.append(idx)
                    self.selected_features.append(candidate_features[idx])

        return [candidate_global_indices[i] for i in selected_local_indices]

    def get_asi_matrix(self):
        """
        Return the matrix to be saved in ASI file.
        - If Rank Deficient: Return Q.T (Basis Transpose).
          (b @ Q.T) gives projection coefficients. Distance can be derived.
        - If Full Rank: Return A_inv.
        """
        if self.state == "FULL_RANK" and self.A_inv is not None:
            return self.A_inv

        if self.state == "RANK_DEFICIENT" and self.Q_basis is not None:
            # Q_basis is (k, m). We want (m, k) so b(1,m) @ M works.
            return self.Q_basis.T

        return None
