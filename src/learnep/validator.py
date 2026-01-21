"""结构合理性验证模块。

主要用于通过原子间距离检查结构的物理合理性，过滤掉原子重叠或距离过近的异常结构。
"""

from typing import List
import numpy as np
from ase import Atoms
from ase.data import covalent_radii


def get_valid_indices(
    frames: List[Atoms], ratio: float = 0.7, default_radius: float = 1.0
) -> List[int]:
    """通过原子间距离过滤不合理的物理结构。

    原理：
    利用 ASE 提供的共价半径数据，计算每对原子之间的理论最小键长（共价半径之和）。
    如果结构中任意两个原子的实际距离小于 `理论距离 * ratio`，则认为该结构不合理（发生了严重的原子重叠）。

    Args:
        frames: 待检查的 ASE Atoms 对象列表。
        ratio: 距离缩放因子。
            阈值 = (r_i + r_j) * ratio。
            默认 0.7 是一个经验值，用于容忍热振动或轻微压缩，同时排除非物理的重叠。
        default_radius: 默认原子半径。
            当 ASE 共价半径数据缺失（为 NaN）时使用的默认值。默认为 1.0 埃。

    Returns:
        合理结构的索引列表。
    """
    valid_indices = []

    for i, atoms in enumerate(frames):
        if is_valid_structure(atoms, ratio, default_radius):
            valid_indices.append(i)

    return valid_indices


def is_valid_structure(
    atoms: Atoms, ratio: float = 0.7, default_radius: float = 1.0
) -> bool:
    """检查单个结构是否合理。

    Args:
        atoms: ASE Atoms 对象。
        ratio: 距离缩放因子。
        default_radius: 默认原子半径（用于处理未知的共价半径）。

    Returns:
        True 如果结构合理（无过近原子），否则 False。
    """
    if len(atoms) <= 1:
        return True

    numbers = atoms.get_atomic_numbers()
    radii = covalent_radii[numbers]

    if np.any(np.isnan(radii)):
        radii = np.nan_to_num(radii, nan=default_radius)

    limit_dist = np.add.outer(radii, radii) * ratio
    real_dist = atoms.get_all_distances(mic=True)

    np.fill_diagonal(real_dist, np.inf)

    return not np.any(real_dist < limit_dist)
