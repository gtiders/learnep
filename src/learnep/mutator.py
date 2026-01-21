"""结构扰动与生成模块。

提供原子位置随机扰动 (Rattle) 和晶格形变 (Strain) 功能，
并支持将多种操作串联使用，自动进行物理合理性校验和重试。
"""

from typing import List, Optional
import numpy as np
from ase import Atoms
from learnep.validator import is_valid_structure


class StructureMutator:
    """结构突变器。

    支持原子位置扰动 (Rattle) 和晶格应变 (Strain) 的组合操作。
    内置重试机制和自适应幅度衰减，以保证生成结构的物理合理性。
    """

    def __init__(
        self,
        rattle_std: float = 0.0,
        strain_std: float = 0.0,
        rattle_prob: float = 1.0,
        attempt_number: int = 50,
        validator_kwargs: Optional[dict] = None,
    ):
        """初始化突变器。

        Args:
            rattle_std: 原子位置扰动的高斯标准差 (Angstrom)。
                设为 0.0 则不启用扰动。
            strain_std: 晶格应变的高斯标准差 (无量纲)。
                设为 0.0 则不启用形变。
            rattle_prob: 每个原子被扰动的概率 (0.0 - 1.0)。
            attempt_number: 生成失败时的最大重试次数。
            validator_kwargs: 传递给 is_valid_structure 的额外参数，
                例如 {'ratio': 0.7, 'default_radius': 1.0}。
        """
        self.rattle_std = rattle_std
        self.strain_std = strain_std
        self.rattle_prob = rattle_prob
        self.attempt_number = attempt_number
        self.validator_kwargs = validator_kwargs or {}

    def generate(self, atoms: Atoms, num_structures: int = 1) -> List[Atoms]:
        """生成指定数量的扰动结构。

        Args:
            atoms: 原始结构 (seed structure)。
            num_structures: 需要生成的结构数量。

        Returns:
            生成的合法结构列表。如果尝试多次仍无法生成足够数量，
            返回的列表长度可能小于 num_structures。
        """
        generated_frames = []

        # 为了避免死循环，设置一个总的尝试上限
        total_attempts = 0
        max_total_attempts = num_structures * self.attempt_number * 2

        while (
            len(generated_frames) < num_structures
            and total_attempts < max_total_attempts
        ):
            new_atoms = self._mutate_single(atoms)
            if new_atoms is not None:
                generated_frames.append(new_atoms)
            total_attempts += 1

        return generated_frames

    def _mutate_single(self, atoms: Atoms) -> Optional[Atoms]:
        """尝试生成单个结构。包含重试和幅度衰减逻辑。"""
        # 深拷贝以避免修改原始结构
        atoms_copy = atoms.copy()

        # 尝试循环
        for i in range(self.attempt_number):
            # 计算衰减因子 (scale)：随尝试次数增加，扰动幅度逐渐减小
            # 第1次 scale=1.0, 最后一次接近 0.2
            if self.attempt_number > 1:
                scale = 1.0 - 0.8 * (i / (self.attempt_number - 1))
            else:
                scale = 1.0

            # 1. 晶格应变 (Strain)
            # 如果启用了 strain (std > 0)，则应用
            if self.strain_std > 0:
                current_strain_std = self.strain_std * scale
                atoms_candidate = self._apply_strain(atoms_copy, current_strain_std)
            else:
                atoms_candidate = atoms_copy.copy()

            # 2. 原子扰动 (Rattle)
            # 如果启用了 rattle (std > 0)，就在应变后的基础上继续扰动
            if self.rattle_std > 0:
                current_rattle_std = self.rattle_std * scale
                atoms_candidate = self._apply_rattle(
                    atoms_candidate, current_rattle_std
                )

            # 3. 校验
            if is_valid_structure(atoms_candidate, **self.validator_kwargs):
                return atoms_candidate

        # 如果所有尝试都失败
        return None

    def _apply_strain(self, atoms: Atoms, std: float) -> Atoms:
        """应用晶格应变"""
        if std <= 1e-6:
            return atoms

        new_atoms = atoms.copy()

        eps = np.random.normal(0, std, 6)
        eps = np.clip(eps, -2 * std, 2 * std)

        M = np.array(
            [
                [1.0 + eps[0], 0.5 * eps[5], 0.5 * eps[4]],
                [0.5 * eps[5], 1.0 + eps[1], 0.5 * eps[3]],
                [0.5 * eps[4], 0.5 * eps[3], 1.0 + eps[2]],
            ]
        )

        new_cell = atoms.cell[:] @ M
        new_atoms.set_cell(new_cell, scale_atoms=True)

        return new_atoms

    def _apply_rattle(self, atoms: Atoms, std: float) -> Atoms:
        """应用原子随机扰动"""
        if std <= 1e-6:
            return atoms

        new_atoms = atoms.copy()
        new_atoms.rattle(stdev=std)

        return new_atoms
