"""
MaxVol 算法模块

本模块提供基于 MaxVol (最大体积) 算法的主动学习结构选择功能。
核心功能包括：
- 描述符投影计算 (B_projection)
- MaxVol 算法选择活跃集
- Gamma 值计算（外推等级评估）
- Active Set Inverse (ASI) 文件读写

参考文献:
    Goreinov S., Oseledets, I., Savostyanov, D., Tyrtyshnikov, E., Zamarashkin, N.
    "How to find a good submatrix". Matrix Methods: Theory, Algorithms And
    Applications (2010): 247-256.

代码来源:
    部分基于 https://github.com/AndreiChertkov/teneva
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu, solve_triangular
from typing import Literal
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# 尝试导入 ASE 和 PyNEP
try:
    from ase import Atoms
    from ase.io import read as ase_read, write as ase_write
except ImportError:
    Atoms = None

try:
    from pynep.calculate import NEP
    from pynep.io import load_nep, dump_nep
    from pynep.select import FarthestPointSample
except ImportError:
    NEP = None
    FarthestPointSample = None


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ActiveSetResult:
    """MaxVol 算法的输出结果"""

    inverse_dict: dict[str, NDArray[np.float64]]
    """按元素类型分类的活跃集逆矩阵 {元素符号: 逆矩阵}"""

    structure_indices: list[int]
    """被选中结构在原始轨迹中的索引"""

    active_set_dict: dict[str, NDArray[np.float64]]
    """按元素类型分类的活跃集矩阵 {元素符号: 描述符矩阵}"""


@dataclass
class DescriptorProjectionResult:
    """描述符投影计算的输出结果"""

    projection_dict: dict[str, NDArray[np.float64]]
    """按元素类型分类的 B 投影矩阵 {元素符号: (N_atoms, D) 矩阵}"""

    structure_index_dict: dict[str, NDArray[np.int64]]
    """按元素类型分类的结构索引 {元素符号: 原子所属结构索引数组}"""


# =============================================================================
# 冷启动模式辅助函数
# =============================================================================


def check_data_sufficient(
    trajectory: list[Atoms],
    nep_file: str | Path,
) -> tuple[bool, dict[str, tuple[int, int]]]:
    """
    检查训练数据是否足够运行 MaxVol 算法。

    对于每种元素，检查原子环境数量是否 >= 描述符维度。

    参数:
        trajectory: 训练集轨迹
        nep_file: NEP 势函数文件路径

    返回:
        (是否足够, 各元素统计 {元素: (原子数, 描述符维度)})
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    if len(trajectory) == 0:
        return False, {}

    nep_file = Path(nep_file)
    calc = NEP(str(nep_file))

    # 解析元素列表
    with open(nep_file) as f:
        first_line = f.readline()
        parts = first_line.split()
        n_types = int(parts[1])
        elements = parts[2 : 2 + n_types]

    # 统计每种元素的原子数
    element_counts: dict[str, int] = {elem: 0 for elem in elements}
    for atoms in trajectory:
        for symbol in atoms.get_chemical_symbols():
            if symbol in element_counts:
                element_counts[symbol] += 1

    # 获取 B_projection 的维度（这是 MaxVol 使用的描述符维度）
    # 注意：B_projection 的维度（~960）远大于 descriptor（~30）
    calc.calculate(trajectory[0], ["B_projection"])
    B_proj = calc.results["B_projection"]
    descriptor_dim = B_proj.shape[1]

    # 检查每种元素
    stats: dict[str, tuple[int, int]] = {}
    all_sufficient = True
    for elem in elements:
        count = element_counts[elem]
        stats[elem] = (count, descriptor_dim)
        if count < descriptor_dim:
            all_sufficient = False

    return all_sufficient, stats


def filter_reasonable_structures(
    structures: list[Atoms],
    nep_file: str | Path,
    min_distance: float = 1.0,
    max_force: float = 50.0,
    max_energy_deviation: float = 5.0,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    过滤不合理的结构。

    在进行 FPS 选择前，先过滤掉物理上不合理的结构。

    过滤条件：
    1. 原子重叠：任意两原子距离 < min_distance
    2. 力过大：任意原子受力 > max_force
    3. 能量异常：每原子能量偏离中位数 > max_energy_deviation

    参数:
        structures: 待过滤的结构列表
        nep_file: NEP 势函数文件路径
        min_distance: 最小原子间距（Å），小于此值视为原子重叠
        max_force: 最大原子力（eV/Å），大于此值视为不合理
        max_energy_deviation: 最大每原子能量偏差（eV），相对于中位数
        show_progress: 是否显示进度条

    返回:
        过滤后的合理结构列表
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    if len(structures) == 0:
        return []

    calc = NEP(str(nep_file))

    # 第一遍：计算所有结构的能量
    print("计算结构能量...")
    energies_per_atom = []
    iterator = tqdm(structures, desc="计算能量") if show_progress else structures
    for atoms in iterator:
        try:
            atoms.calc = calc
            e = atoms.get_potential_energy() / len(atoms)
            energies_per_atom.append(e)
        except Exception:
            energies_per_atom.append(float("inf"))

    # 计算中位数能量（更鲁棒）
    valid_energies = [e for e in energies_per_atom if e != float("inf")]
    if valid_energies:
        median_energy = float(np.median(valid_energies))
    else:
        median_energy = 0.0

    # 第二遍：过滤
    print(f"过滤不合理结构（中位能量: {median_energy:.3f} eV/atom）...")
    reasonable = []
    rejected = {"overlap": 0, "force": 0, "energy": 0, "error": 0}

    iterator2 = (
        tqdm(enumerate(structures), total=len(structures), desc="过滤")
        if show_progress
        else enumerate(structures)
    )
    for i, atoms in iterator2:
        # 检查1：原子重叠
        try:
            distances = atoms.get_all_distances(mic=True)
            np.fill_diagonal(distances, float("inf"))
            if distances.min() < min_distance:
                rejected["overlap"] += 1
                continue
        except Exception:
            rejected["error"] += 1
            continue

        # 检查2：力过大
        try:
            atoms.calc = calc
            forces = atoms.get_forces()
            max_f = np.linalg.norm(forces, axis=1).max()
            if max_f > max_force:
                rejected["force"] += 1
                continue
        except Exception:
            rejected["error"] += 1
            continue

        # 检查3：能量异常
        e_per_atom = energies_per_atom[i]
        if abs(e_per_atom - median_energy) > max_energy_deviation:
            rejected["energy"] += 1
            continue

        reasonable.append(atoms)

    print(f"过滤结果: {len(reasonable)}/{len(structures)} 个结构通过")
    print(
        f"  拒绝原因: 原子重叠={rejected['overlap']}, 力过大={rejected['force']}, "
        f"能量异常={rejected['energy']}, 计算错误={rejected['error']}"
    )

    return reasonable


# =============================================================================
# Core MaxVol Algorithm (CPU Version)
# =============================================================================


def _maxvol_core(
    A: NDArray[np.float64],
    gamma_tol: float = 1.001,
    max_iter: int = 1000,
) -> NDArray[np.int64]:
    """
    MaxVol 核心算法：从高矩阵中选择最大体积子矩阵。

    算法通过迭代交换行，使得选中的子矩阵具有最大的行列式（体积）。
    这保证了选中的行能够最大程度地张成原始矩阵的列空间。

    参数:
        A: 输入的高矩阵，形状为 (n, r)，要求 n > r
        gamma_tol: 收敛精度参数，应 >= 1.0
            - 等于 1.0 时会迭代直到完全收敛
            - 大于 1.0 时算法更快但精度略低（推荐 1.01 - 1.1）
        max_iter: 允许的最大迭代次数

    返回:
        被选中行的索引数组，长度为 r

    异常:
        ValueError: 当输入矩阵不是高矩阵时抛出
    """
    n, r = A.shape

    if n < r:
        raise ValueError(f"输入矩阵必须是高矩阵 (n >= r)，当前: n={n}, r={r}")

    # LU decomposition for initialization
    P, L, U = lu(A, check_finite=False)
    selected_indices = P[:, :r].argmax(axis=0)

    # Compute coefficient matrix B = A @ A[I]^(-1)
    Q = solve_triangular(U, A.T, trans=1, check_finite=False)
    B = solve_triangular(
        L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True
    ).T

    # Iterative optimization
    for _ in range(max_iter):
        # Find the element with maximum absolute value
        max_pos = np.abs(B).argmax()
        i, j = divmod(max_pos, r)
        current_gamma = np.abs(B[i, j])

        # Check convergence
        if current_gamma <= gamma_tol:
            break

        # Swap row
        selected_indices[j] = i

        # Update coefficient matrix (Sherman-Morrison formula)
        bj = B[:, j]
        bi = B[i, :].copy()
        bi[j] -= 1.0
        B -= np.outer(bj, bi / B[i, j])

    return selected_indices


def _compute_pinv(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    计算矩阵的伪逆。

    使用较大的条件数阈值 (1e-8) 以避免 GPUMD 使用 float 类型时的数值问题。

    参数:
        matrix: 输入矩阵

    返回:
        伪逆矩阵
    """
    return np.linalg.pinv(matrix, rcond=1e-8)


# =============================================================================
# Batch MaxVol Algorithm
# =============================================================================


def compute_maxvol(
    A: NDArray[np.float64],
    struct_index: NDArray[np.int64],
    gamma_tol: float = 1.001,
    max_iter: int = 1000,
    batch_size: int | None = None,
    n_refinement: int = 10,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    执行 MaxVol 算法，支持批量处理和迭代细化。

    对于大规模数据，使用批量处理策略：
    1. 将数据分成多个批次
    2. 每个批次与之前的结果合并后执行 MaxVol
    3. 最后进行多轮细化确保收敛

    参数:
        A: 描述符矩阵，形状为 (N, D)
        struct_index: 每个环境对应的结构索引
        gamma_tol: MaxVol 收敛阈值
        max_iter: 单次 MaxVol 的最大迭代次数
        batch_size: 批处理大小,None 表示一次性处理
        n_refinement: 批处理后的细化迭代次数

    返回:
        (选中的描述符矩阵, 选中的结构索引)
    """
    # Single batch mode
    if batch_size is None:
        selected = _maxvol_core(A, gamma_tol, max_iter)
        return A[selected], struct_index[selected]

    # Multi-batch mode
    n_batches = int(np.ceil(len(A) / batch_size))
    batch_splits = np.array_split(np.arange(len(A)), n_batches)

    # Stage 1: Cumulative MaxVol
    A_selected: NDArray[np.float64] | None = None
    index_selected: NDArray[np.int64] | None = None

    for i, batch_indices in enumerate(batch_splits):
        if A_selected is None:
            # First batch
            A_joint = A[batch_indices]
            index_joint = struct_index[batch_indices]
        else:
            # Subsequent batches: merge with selected
            A_joint = np.vstack([A_selected, A[batch_indices]])
            index_joint = np.hstack([index_selected, struct_index[batch_indices]])

        selected = _maxvol_core(A_joint, gamma_tol, max_iter)
        prev_len = 0 if A_selected is None else len(A_selected)
        A_selected = A_joint[selected]
        index_selected = index_joint[selected]
        n_added = (selected >= prev_len).sum()
        print(f"Batch {i + 1}/{n_batches}: added {n_added} environments")

    # Stage 2: Refinement
    assert A_selected is not None and index_selected is not None

    for ii in range(n_refinement):
        inv_matrix = _compute_pinv(A_selected)
        gamma_matrix = np.abs(A_selected @ inv_matrix)
        large_gamma_mask = gamma_matrix > gamma_tol
        max_gamma = np.max(gamma_matrix)

        print(
            f"Refinement {ii + 1}: {large_gamma_mask.sum()} envs exceed threshold, max gamma = {max_gamma:.4f}"
        )

        if max_gamma < gamma_tol:
            print("Refinement done")
            break

        # Add environments exceeding threshold to candidates
        A_joint = np.vstack([A_selected, A[large_gamma_mask.any(axis=1)]])
        index_joint = np.hstack(
            [index_selected, struct_index[large_gamma_mask.any(axis=1)]]
        )
        selected = _maxvol_core(A_joint, gamma_tol, max_iter)
        A_selected = A_joint[selected]
        index_selected = index_joint[selected]

    return A_selected, index_selected


# =============================================================================
# Descriptor Computation
# =============================================================================


def compute_descriptor_projection(
    trajectory: list[Atoms],
    nep_file: str | Path,
    show_progress: bool = True,
) -> DescriptorProjectionResult:
    """
    计算轨迹中所有原子的 NEP 描述符投影 (B_projection)。

    描述符投影是 NEP 势函数中每个原子局部环境的低维表示，
    用于评估模型的外推程度。

    参数:
        trajectory: ASE Atoms 对象列表
        nep_file: NEP 势函数文件路径 (nep.txt)
        show_progress: 是否显示进度条

    返回:
        描述符投影结果，包含按元素分类的投影矩阵和结构索引

    异常:
        ImportError: 当 PyNEP 未安装时抛出
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    nep_file = Path(nep_file)
    calc = NEP(str(nep_file))

    # Parse element list from NEP file
    with open(nep_file) as f:
        first_line = f.readline()
        parts = first_line.split()
        n_types = int(parts[1])
        elements = parts[2 : 2 + n_types]  # Format: nep4 N_types elem1 elem2 ... elemN
    print(f"Elements in NEP potential: {elements}")

    # Initialize storage
    projection_dict: dict[str, list[NDArray]] = {elem: [] for elem in elements}
    struct_index_dict: dict[str, list[int]] = {elem: [] for elem in elements}

    # Iterate over trajectory
    iterator = (
        tqdm(
            enumerate(trajectory), total=len(trajectory), desc="Computing B_projection"
        )
        if show_progress
        else enumerate(trajectory)
    )

    for struct_idx, atoms in iterator:
        calc.calculate(atoms, ["B_projection"])
        B_proj = calc.results["B_projection"]

        for atom_proj, symbol in zip(B_proj, atoms.get_chemical_symbols()):
            projection_dict[symbol].append(atom_proj)
            struct_index_dict[symbol].append(struct_idx)

    # Convert to NumPy arrays
    projection_dict_arr = {}
    struct_index_dict_arr = {}
    print("Descriptor matrix shapes:")
    for elem in elements:
        if len(projection_dict[elem]) > 0:
            projection_dict_arr[elem] = np.vstack(projection_dict[elem])
            struct_index_dict_arr[elem] = np.array(
                struct_index_dict[elem], dtype=np.int64
            )
            print(f"  {elem}: {projection_dict_arr[elem].shape}")

            # Verify the matrix is tall (overdetermined system)
            # MaxVol 算法是按元素类型分别进行的，所以每个元素都需要满足超定条件
            n, d = projection_dict_arr[elem].shape
            if n < d:
                error_msg = (
                    f"\n{'=' * 80}\n"
                    f"错误：元素 '{elem}' 的训练数据不足以运行 MaxVol 算法\n"
                    f"{'=' * 80}\n"
                    f"当前状态:\n"
                    f"  元素类型: {elem}\n"
                    f"  该元素的原子环境数量: {n}\n"
                    f"  NEP 描述符维度: {d}\n"
                    f"  需要满足: 原子数 >= 描述符维度\n\n"
                    f"MaxVol 算法要求 (按元素类型):\n"
                    f"  - 每种元素的原子数量必须大于等于描述符维度\n"
                    f"  - 当原子数等于描述符维度时，所有原子都会被选入活跃集\n"
                    f"  - 不同元素类型分别计算活跃集\n\n"
                    f"建议解决方案:\n"
                    f"  1. 增加训练集中包含元素 '{elem}' 的结构数量\n"
                    f"  2. 至少需要 {d} 个 '{elem}' 原子环境\n"
                    f"  3. 建议提供至少 {d * 2} 个 '{elem}' 原子以获得更好的效果\n"
                    f"  4. 检查所有元素类型 ({', '.join(elements)}) 是否都有足够的原子\n"
                    f"{'=' * 80}\n"
                )
                raise ValueError(error_msg)

    return DescriptorProjectionResult(
        projection_dict=projection_dict_arr,
        structure_index_dict=struct_index_dict_arr,
    )


def compute_gamma(
    trajectory: list[Atoms],
    nep_file: str | Path,
    asi_file: str | Path,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    计算轨迹中每个原子的 Gamma 值（外推等级）。

    Gamma 值表示原子局部环境相对于训练集的外推程度：
    - Gamma ≈ 1: 环境在训练集覆盖范围内
    - Gamma > 1: 环境超出训练集，存在外推风险
    - Gamma >> 1: 严重外推，模型预测不可靠

    计算结果存储在每个 Atoms 对象的 arrays["gamma"] 中。

    参数:
        trajectory: ASE Atoms 对象列表
        nep_file: NEP 势函数文件路径
        asi_file: Active Set Inverse 文件路径
        show_progress: 是否显示进度条

    返回:
        更新后的轨迹（原地修改，同时返回引用）
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    calc = NEP(str(nep_file))
    active_set_inv = read_asi_file(asi_file)

    iterator = tqdm(trajectory, desc="Computing gamma") if show_progress else trajectory

    for atoms in iterator:
        # Initialize gamma array
        atoms.arrays["gamma"] = np.zeros(len(atoms))

        # Compute descriptor projection
        calc.calculate(atoms, ["B_projection"])
        B_proj = calc.results["B_projection"]

        # Compute gamma by element
        symbols = atoms.get_chemical_symbols()
        for elem, inv_matrix in active_set_inv.items():
            atom_indices = [i for i, sym in enumerate(symbols) if sym == elem]
            if len(atom_indices) == 0:
                continue

            # gamma = |B @ A^(-1)|_max (max per atom)
            g = B_proj[atom_indices] @ inv_matrix
            g = np.max(np.abs(g), axis=1)
            atoms.arrays["gamma"][atom_indices] = g

    return trajectory


# =============================================================================
# Active Set Generation
# =============================================================================


def generate_active_set(
    descriptor_result: DescriptorProjectionResult,
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
    write_asi: bool = True,
    asi_output_path: str | Path = "active_set.asi",
) -> ActiveSetResult:
    """
    使用 MaxVol 算法从描述符投影中生成活跃集。

    活跃集是训练数据中最具代表性的原子环境子集，
    其张成的空间能够覆盖所有训练数据的描述符。

    参数:
        descriptor_result: 描述符投影计算结果
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小
        write_asi: 是否将结果写入 ASI 文件
        asi_output_path: ASI 文件输出路径

    返回:
        活跃集结果
    """
    print("Running MaxVol algorithm...")
    active_set_dict: dict[str, NDArray] = {}
    all_struct_indices: list[int] = []

    for elem, B_proj in descriptor_result.projection_dict.items():
        print(f"\nProcessing element: {elem}")
        A_selected, index_selected = compute_maxvol(
            B_proj,
            descriptor_result.structure_index_dict[elem],
            gamma_tol=gamma_tol,
            batch_size=batch_size,
        )
        active_set_dict[elem] = A_selected
        all_struct_indices.extend(index_selected.tolist())
        print(f"Active set shape: {A_selected.shape}")

    # Deduplicate and sort structure indices
    structure_indices = sorted(set(all_struct_indices))

    # Compute inverse matrices
    print("\nComputing active set inverse...")
    inverse_dict = {elem: _compute_pinv(A) for elem, A in active_set_dict.items()}

    # Save ASI file
    if write_asi:
        print(f"Saving active set inverse to: {asi_output_path}")
        write_asi_file(inverse_dict, asi_output_path)

    return ActiveSetResult(
        inverse_dict=inverse_dict,
        structure_indices=structure_indices,
        active_set_dict=active_set_dict,
    )


# =============================================================================
# ASI File I/O
# =============================================================================


def write_asi_file(
    active_set_inv: dict[str, NDArray[np.float64]],
    file_path: str | Path = "active_set.asi",
) -> None:
    """
    将活跃集逆矩阵保存到 ASI 文件。

    ASI (Active Set Inverse) 文件格式:
    ```
    元素符号 行数 列数
    矩阵元素1
    矩阵元素2
    ...
    ```

    参数:
        active_set_inv: 按元素分类的逆矩阵字典
        file_path: 输出文件路径
    """
    with open(file_path, "w") as f:
        for elem, matrix in active_set_inv.items():
            f.write(f"{elem} {matrix.shape[0]} {matrix.shape[1]}\n")
            for val in matrix.flatten():
                f.write(f"{val}\n")


def read_asi_file(file_path: str | Path) -> dict[str, NDArray[np.float64]]:
    """
    从 ASI 文件读取活跃集逆矩阵。

    参数:
        file_path: ASI 文件路径

    返回:
        按元素分类的逆矩阵字典
    """
    result: dict[str, NDArray] = {}

    with open(file_path, "r") as f:
        while True:
            line1 = f.readline()
            if not line1:
                break

            parts = line1.split()
            elem, rows, cols = parts[0], int(parts[1]), int(parts[2])

            data = [float(f.readline()) for _ in range(rows * cols)]
            result[elem] = np.array(data).reshape((rows, cols))

    return result


# =============================================================================
# High-Level Selection Functions
# =============================================================================


def select_active_set(
    trajectory: list[Atoms],
    nep_file: str | Path,
    asi_output_path: str | Path = "active_set.asi",
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
) -> tuple[ActiveSetResult, list[Atoms]]:
    """
    从训练轨迹中选择活跃集。

    这是 select_active.py 的函数化版本。

    参数:
        trajectory: 训练集轨迹
        nep_file: NEP 势函数文件路径
        asi_output_path: ASI 文件输出路径
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小

    返回:
        (活跃集结果, 被选中的结构列表)
    """
    # Compute descriptor projection
    descriptor_result = compute_descriptor_projection(trajectory, nep_file)

    # Generate active set
    active_set = generate_active_set(
        descriptor_result,
        gamma_tol=gamma_tol,
        batch_size=batch_size,
        write_asi=True,
        asi_output_path=asi_output_path,
    )

    # Extract selected structures
    selected_structures = [trajectory[i] for i in active_set.structure_indices]

    return active_set, selected_structures


def select_structures_maxvol(
    train_structures: list[Atoms],
    candidate_structures: list[Atoms],
    nep_file: str | Path,
    max_structures: int,
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
) -> list[Atoms]:
    """
    使用 MaxVol 算法从候选结构中选择指定数量的新结构。

    与 select_extension_structures 不同，此函数直接限制输出数量，
    而不是返回所有被 MaxVol 选中的结构。

    算法流程：
    1. 合并训练集和候选集
    2. 计算合并后所有结构的描述符投影
    3. 执行 MaxVol 选择，获取所有被选中的结构索引
    4. 筛选出仅来自候选集的结构
    5. 如果超过 max_structures，使用增量 MaxVol 选择最重要的

    参数:
        train_structures: 当前训练集
        candidate_structures: 候选结构（来自 MD 探索）
        nep_file: NEP 势函数文件路径
        max_structures: 最大选择数量
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小

    返回:
        被选中的结构列表（数量 <= max_structures）
    """
    if len(candidate_structures) == 0:
        print("没有候选结构")
        return []

    if len(candidate_structures) <= max_structures:
        print(
            f"候选结构数 ({len(candidate_structures)}) <= 限制 ({max_structures})，返回全部"
        )
        return candidate_structures

    train_size = len(train_structures)
    merged_trajectory = train_structures + candidate_structures

    # 计算描述符投影
    print("计算描述符投影...")
    descriptor_result = compute_descriptor_projection(merged_trajectory, nep_file)

    # 执行 MaxVol 获取初步选择
    print("执行 MaxVol 选择...")
    active_set = generate_active_set(
        descriptor_result,
        gamma_tol=gamma_tol,
        batch_size=batch_size,
        write_asi=False,
    )

    # 筛选出仅来自候选集的结构索引
    candidate_indices = [i for i in active_set.structure_indices if i >= train_size]
    print(f"MaxVol 从候选集中选出 {len(candidate_indices)} 个结构")

    if len(candidate_indices) <= max_structures:
        # 不需要进一步筛选
        return [merged_trajectory[i] for i in candidate_indices]

    # 需要进一步筛选：使用 MaxVol 选择最具代表性的 max_structures 个
    print(f"需要从 {len(candidate_indices)} 个结构中筛选 {max_structures} 个...")

    # 提取候选结构的描述符
    candidate_atoms = [merged_trajectory[i] for i in candidate_indices]

    # 计算仅候选结构的描述符
    calc = NEP(str(nep_file))

    # 按元素收集描述符和结构映射
    from pynep.calculate import NEP as NEP_Calc

    calc = NEP_Calc(str(nep_file))

    # 计算每个候选结构的平均描述符
    structure_descriptors = []
    for atoms in candidate_atoms:
        desc = calc.get_property("descriptor", atoms)
        # 使用结构的平均描述符
        avg_desc = np.mean(desc, axis=0)
        structure_descriptors.append(avg_desc)

    A = np.vstack(structure_descriptors)
    print(f"结构描述符矩阵形状: {A.shape}")

    n, d = A.shape
    if n <= max_structures:
        # 如果结构数不超过限制，返回全部
        return candidate_atoms

    # 使用贪婪 MaxVol 选择 max_structures 个最具代表性的结构
    selected_indices = _greedy_maxvol_select(A, max_structures)

    selected = [candidate_atoms[i] for i in selected_indices]
    print(f"最终选中 {len(selected)} 个结构")

    return selected


def _greedy_maxvol_select(
    A: NDArray[np.float64],
    k: int,
) -> list[int]:
    """
    贪婪 MaxVol 选择：从 n 个向量中选择 k 个使体积最大化。

    使用贪婪策略：每次选择能最大化当前子空间体积的向量。

    参数:
        A: 描述符矩阵，形状 (n, d)
        k: 要选择的数量

    返回:
        被选中的索引列表
    """
    n, d = A.shape
    k = min(k, n, d)  # 确保 k 不超过 n 和 d

    # 使用 QR 分解进行贪婪选择
    remaining = set(range(n))
    selected = []

    # 初始化：选择范数最大的向量
    norms = np.linalg.norm(A, axis=1)
    first = np.argmax(norms)
    selected.append(first)
    remaining.remove(first)

    # 当前子空间的基
    Q = A[first : first + 1].T.copy()
    Q = Q / np.linalg.norm(Q)

    for _ in range(k - 1):
        if not remaining:
            break

        # 计算每个剩余向量到当前子空间的距离
        max_dist = -1
        best_idx = -1

        for idx in remaining:
            v = A[idx]
            # 投影到当前子空间
            proj = Q @ (Q.T @ v)
            # 计算垂直分量的范数
            dist = np.linalg.norm(v - proj)

            if dist > max_dist:
                max_dist = dist
                best_idx = idx

        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)

            # 更新子空间基（Gram-Schmidt）
            v = A[best_idx]
            proj = Q @ (Q.T @ v)
            new_basis = v - proj
            norm = np.linalg.norm(new_basis)
            if norm > 1e-10:
                new_basis = new_basis / norm
                Q = np.column_stack([Q, new_basis])

    return selected


def select_extension_structures(
    train_trajectory: list[Atoms],
    candidate_trajectory: list[Atoms],
    nep_file: str | Path,
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
) -> list[Atoms]:
    """
    从候选结构中选择需要标注的新结构。

    这是 select_extend.py 的函数化版本。
    算法合并训练集和候选集，执行 MaxVol，然后只返回来自候选集的结构。

    参数:
        train_trajectory: 当前训练集
        candidate_trajectory: 高 Gamma 候选结构
        nep_file: NEP 势函数文件路径
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小

    返回:
        被选中的新结构列表（仅来自候选集）
    """
    train_size = len(train_trajectory)
    merged_trajectory = train_trajectory + candidate_trajectory

    # Compute descriptor projection
    descriptor_result = compute_descriptor_projection(merged_trajectory, nep_file)

    # Generate active set (without writing ASI file)
    active_set = generate_active_set(
        descriptor_result,
        gamma_tol=gamma_tol,
        batch_size=batch_size,
        write_asi=False,
    )

    # Keep only structures from candidate set
    new_structures = [
        merged_trajectory[i] for i in active_set.structure_indices if i >= train_size
    ]

    print(
        f"\nSelected {len(new_structures)} new structures from {len(candidate_trajectory)} candidates"
    )
    return new_structures


def filter_high_gamma_structures(
    trajectory: list[Atoms],
    nep_file: str | Path,
    asi_file: str | Path,
    gamma_min: float = 1.0,
    gamma_max: float = float("inf"),
) -> list[Atoms]:
    """
    根据 Gamma 值筛选结构。

    这是 select_gamma.py 的函数化版本。

    参数:
        trajectory: 待筛选的轨迹
        nep_file: NEP 势函数文件路径
        asi_file: Active Set Inverse 文件路径
        gamma_min: Gamma 下限阈值
        gamma_max: Gamma 上限阈值

    返回:
        满足 gamma_min < max_gamma < gamma_max 的结构列表
    """
    # Compute gamma values
    compute_gamma(trajectory, nep_file, asi_file)

    # Filter
    filtered = []
    for atoms in trajectory:
        max_gamma = atoms.arrays["gamma"].max()
        if gamma_min < max_gamma < gamma_max:
            filtered.append(atoms)

    print(
        f"Filtered {len(filtered)} high-gamma structures from {len(trajectory)} total"
    )
    return filtered


# =============================================================================
# Convenience I/O Functions
# =============================================================================


def read_trajectory(
    file_path: str | Path,
    format: Literal["nep", "xyz", "auto"] = "auto",
) -> list[Atoms]:
    """
    读取轨迹文件。

    参数:
        file_path: 轨迹文件路径
        format: 文件格式 ("nep", "xyz", "auto")

    返回:
        Atoms 对象列表
    """
    file_path = Path(file_path)

    if format == "auto":
        format = "nep" if file_path.suffix == ".xyz" else "xyz"

    if format == "nep" and load_nep is not None:
        try:
            return load_nep(str(file_path))
        except Exception:
            pass

    return ase_read(str(file_path), index=":")


def write_trajectory(
    trajectory: list[Atoms],
    file_path: str | Path,
    format: Literal["nep", "xyz", "auto"] = "auto",
) -> None:
    """
    写入轨迹文件。

    参数:
        trajectory: Atoms 对象列表
        file_path: 输出文件路径
        format: 文件格式 ("nep", "xyz", "auto")
    """
    file_path = Path(file_path)

    if format == "auto":
        format = "nep"

    if format == "nep" and dump_nep is not None:
        try:
            dump_nep(str(file_path), trajectory)
            return
        except Exception:
            pass

    ase_write(str(file_path), trajectory)


# =============================================================================
# FPS (最远点采样) 筛选
# =============================================================================


def apply_fps_filter(
    structures: list[Atoms],
    nep_file: str | Path,
    max_count: int,
    initial_min_distance: float = 0.01,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    使用 FPS (最远点采样) 对结构进行二次筛选。

    该函数用于在 MaxVol 选择后进一步确保结构的多样性。
    如果 FPS 选出的结构数量不足 max_count，会自动降低 min_distance。
    如果超过 max_count，会随机丢弃多余的结构。

    参数:
        structures: 待筛选的结构列表（通常是 MaxVol 选出的结构）
        nep_file: NEP 势函数文件路径
        max_count: 目标结构数量（max_structures_per_iteration）
        initial_min_distance: 初始最小距离阈值
        show_progress: 是否显示进度

    返回:
        筛选后的结构列表（数量 <= max_count）
    """
    if FarthestPointSample is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    if len(structures) == 0:
        return structures

    if len(structures) <= max_count:
        print(f"结构数量 ({len(structures)}) <= 上限 ({max_count})，无需 FPS 筛选")
        return structures

    # 计算描述符（结构级别平均）
    print(f"\n执行 FPS 二次筛选: {len(structures)} → 目标 {max_count}")
    calc = NEP(str(nep_file))

    descriptors = []
    iterator = tqdm(structures, desc="计算描述符") if show_progress else structures

    for structure in iterator:
        desc = calc.get_property("descriptor", structure)
        # 对每个结构求平均描述符
        descriptors.append(np.mean(desc, axis=0))

    descriptors_array = np.array(descriptors)
    print(f"描述符形状: {descriptors_array.shape}")

    # 自动调整 min_distance 以满足 max_count 约束
    min_distance = initial_min_distance
    max_iterations = 10

    for attempt in range(max_iterations):
        sampler = FarthestPointSample(min_distance=min_distance)
        selected_indices = sampler.select(descriptors_array, [])
        n_selected = len(selected_indices)

        print(
            f"  尝试 {attempt + 1}: min_distance={min_distance:.6f}, "
            f"选中 {n_selected} 个结构"
        )

        if n_selected >= max_count:
            # 选出的数量足够或过多
            if n_selected > max_count:
                # 随机丢弃多余的结构
                print(
                    f"  FPS 选出 {n_selected} > {max_count}，随机丢弃 "
                    f"{n_selected - max_count} 个"
                )
                np.random.shuffle(selected_indices)
                selected_indices = selected_indices[:max_count]

            selected_structures = [structures[i] for i in selected_indices]
            print(f"✓ FPS 筛选完成: {len(structures)} → {len(selected_structures)}\n")
            return selected_structures

        # 数量不足，降低 min_distance
        min_distance *= 0.7  # 每次减少30%

        if min_distance < 1e-6:
            # min_distance 太小了，直接返回所有选中的
            print(
                f"  警告: min_distance 已降至 {min_distance:.2e}，"
                f"仅选出 {n_selected} 个结构"
            )
            selected_structures = [structures[i] for i in selected_indices]
            return selected_structures

    # 达到最大尝试次数，返回当前结果
    print(f"  警告: 达到最大尝试次数，FPS 仅选出 {len(selected_indices)} 个结构")
    selected_structures = [structures[i] for i in selected_indices]
    return selected_structures


def prune_training_set_maxvol(
    structures: list[Atoms],
    nep_file: str | Path,
    max_structures: int,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    使用 MaxVol 算法修剪训练集。

    通过计算结构级别的平均描述符，使用 MaxVol 选择最有代表性的结构，
    确保训练集大小不超过描述符维度，提高训练效率。

    参数:
        structures: 原始训练集结构列表
        nep_file: NEP 势函数文件路径
        max_structures: 最大保留结构数
        show_progress: 是否显示进度

    返回:
        修剪后的结构列表（数量 <= max_structures）
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    if len(structures) <= max_structures:
        print(f"训练集大小 ({len(structures)}) <= 上限 ({max_structures})，无需修剪")
        return structures

    print(f"\n执行训练集修剪 (MaxVol): {len(structures)} → {max_structures}")

    # 计算结构级别的平均描述符
    calc = NEP(str(nep_file))
    descriptors = []

    iterator = tqdm(structures, desc="计算描述符") if show_progress else structures

    for structure in iterator:
        desc = calc.get_property("descriptor", structure)
        # 对每个结构求平均描述符
        descriptors.append(np.mean(desc, axis=0))

    descriptors_array = np.array(descriptors)  # shape: (n_structures, descriptor_dim)
    n, d = descriptors_array.shape

    print(f"描述符矩阵形状: {descriptors_array.shape}")
    print(f"  结构数: {n}")
    print(f"  描述符维度: {d}")

    # 检查是否满足 MaxVol 要求
    if n < d:
        print(f"警告: 结构数 ({n}) <= 描述符维度 ({d})，无法使用 MaxVol")
        print("      返回所有结构")
        return structures

    # 使用 MaxVol 算法选择最有代表性的结构
    # 需要选择 max_structures 个结构
    target_count = min(max_structures, d)  # 不能超过描述符维度

    print(f"\n使用 MaxVol 选择 {target_count} 个最有代表性的结构...")

    try:
        # 使用内部的 MaxVol 核心算法
        selected_indices = _maxvol_core(
            descriptors_array, gamma_tol=1.001, max_iter=1000
        )

        # selected_indices 长度为 d，我们需要从中选择 target_count 个
        if len(selected_indices) > target_count:
            # 随机选择 target_count 个
            np.random.shuffle(selected_indices)
            selected_indices = selected_indices[:target_count]

        selected_structures = [structures[i] for i in selected_indices]

        print(f"✓ 修剪完成: {len(structures)} → {len(selected_structures)} 个结构\n")
        return selected_structures

    except Exception as e:
        print(f"警告: MaxVol 修剪失败: {e}")
        print("      回退到随机采样")
        # 如果 MaxVol 失败，回退到随机采样
        indices = list(range(len(structures)))
        np.random.shuffle(indices)
        selected_indices = indices[:target_count]
        return [structures[i] for i in selected_indices]


# =============================================================================
# 候选结构选择函数 (主动学习迭代使用)
# =============================================================================


def select_candidates_maxvol(
    train_trajectory: list[Atoms],
    candidate_trajectory: list[Atoms],
    nep_file: str | Path,
    max_count: int,
    gamma_tol: float = 1.001,
    batch_size: int = 10000,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    使用 MaxVol 算法从候选结构中选择指定数量的新结构。

    该函数首先合并训练集和候选集，计算描述符投影，然后使用 MaxVol
    算法评估每个候选结构对描述符空间覆盖的贡献。只返回来自候选集
    的结构，并按照对空间覆盖贡献从大到小排序。

    算法流程:
    1. 合并训练集和候选集的描述符
    2. 计算训练集的活跃集逆矩阵
    3. 计算每个候选结构的 gamma 值（空间贡献度）
    4. 按 gamma 值从大到小排序，选择最多 max_count 个结构

    参数:
        train_trajectory: 当前训练集结构列表
        candidate_trajectory: 候选结构列表（来自 MD 探索）
        nep_file: NEP 势函数文件路径
        max_count: 最多选择的结构数量
        gamma_tol: MaxVol 收敛阈值
        batch_size: 批处理大小
        show_progress: 是否显示进度条

    返回:
        选中的候选结构列表（仅来自候选集，按 gamma 贡献排序）
    """
    if NEP is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    if len(candidate_trajectory) == 0:
        return []

    if len(candidate_trajectory) <= max_count:
        print(
            f"候选结构数 ({len(candidate_trajectory)}) <= 目标数 ({max_count})，返回所有候选"
        )
        return candidate_trajectory

    print(f"\n使用 MaxVol 选择候选结构: {len(candidate_trajectory)} → 目标 {max_count}")
    print(f"  训练集大小: {len(train_trajectory)}")
    print(f"  候选集大小: {len(candidate_trajectory)}")

    # 计算训练集的活跃集
    print("\n1. 计算训练集的活跃集...")
    train_desc_result = compute_descriptor_projection(
        train_trajectory, nep_file, show_progress=show_progress
    )
    train_active_set = generate_active_set(
        train_desc_result,
        gamma_tol=gamma_tol,
        batch_size=batch_size,
        write_asi=False,
    )

    # 计算候选结构的 gamma 值
    print("\n2. 计算候选结构的 gamma 值...")
    calc = NEP(str(nep_file))

    # 对每个候选结构计算其最大 gamma 值
    candidate_gammas = []
    iterator = (
        tqdm(candidate_trajectory, desc="计算 gamma")
        if show_progress
        else candidate_trajectory
    )

    for atoms in iterator:
        # 计算描述符投影
        calc.calculate(atoms, ["B_projection"])
        B_proj = calc.results["B_projection"]
        symbols = atoms.get_chemical_symbols()

        # 计算每个原子的 gamma
        max_gamma = 1.0
        for elem, inv_matrix in train_active_set.inverse_dict.items():
            atom_indices = [i for i, sym in enumerate(symbols) if sym == elem]
            if len(atom_indices) == 0:
                continue

            g = B_proj[atom_indices] @ inv_matrix
            g = np.max(np.abs(g), axis=1)
            max_gamma = max(max_gamma, float(np.max(g)))

        candidate_gammas.append(max_gamma)

    # 按 gamma 值从大到小排序
    print("\n3. 按 gamma 值排序选择...")
    sorted_indices = np.argsort(candidate_gammas)[::-1]  # 降序

    # 选择 gamma 值最大的 max_count 个结构
    selected_indices = sorted_indices[:max_count]
    selected_gammas = [candidate_gammas[i] for i in selected_indices]

    print(f"  选中 {len(selected_indices)} 个结构")
    print(f"  Gamma 范围: {min(selected_gammas):.4f} ~ {max(selected_gammas):.4f}")

    selected_structures = [candidate_trajectory[i] for i in selected_indices]
    print(
        f"✓ MaxVol 选择完成: {len(candidate_trajectory)} → {len(selected_structures)}\n"
    )

    return selected_structures


def select_candidates_fps(
    candidate_trajectory: list[Atoms],
    nep_file: str | Path,
    max_count: int,
    initial_min_distance: float = 0.01,
    show_progress: bool = True,
) -> list[Atoms]:
    """
    使用 FPS (最远点采样) 算法从候选结构中选择指定数量的结构。

    该函数直接对候选结构应用 FPS 算法，自动调整距离阈值以确保
    选中的结构数量达到目标。FPS 算法保证选中的结构在描述符空间
    中具有最大的多样性。

    参数:
        candidate_trajectory: 候选结构列表
        nep_file: NEP 势函数文件路径
        max_count: 目标结构数量
        initial_min_distance: 初始最小距离阈值
        show_progress: 是否显示进度

    返回:
        筛选后的结构列表（数量 <= max_count）
    """
    if FarthestPointSample is None:
        raise ImportError("请先安装 PyNEP: pip install pynep")

    if len(candidate_trajectory) == 0:
        return []

    if len(candidate_trajectory) <= max_count:
        print(
            f"候选结构数 ({len(candidate_trajectory)}) <= 目标数 ({max_count})，返回所有候选"
        )
        return candidate_trajectory

    print(f"\n使用 FPS 选择候选结构: {len(candidate_trajectory)} → 目标 {max_count}")

    # 计算描述符（结构级别平均）
    calc = NEP(str(nep_file))

    descriptors = []
    iterator = (
        tqdm(candidate_trajectory, desc="计算描述符")
        if show_progress
        else candidate_trajectory
    )

    for structure in iterator:
        desc = calc.get_property("descriptor", structure)
        # 对每个结构求平均描述符
        descriptors.append(np.mean(desc, axis=0))

    descriptors_array = np.array(descriptors)
    print(f"描述符形状: {descriptors_array.shape}")

    # 自动调整 min_distance 以满足 max_count 约束
    min_distance = initial_min_distance
    max_iterations = 15

    for attempt in range(max_iterations):
        sampler = FarthestPointSample(min_distance=min_distance)
        selected_indices = sampler.select(descriptors_array, [])
        n_selected = len(selected_indices)

        print(
            f"  尝试 {attempt + 1}: min_distance={min_distance:.6f}, "
            f"选中 {n_selected} 个结构"
        )

        if n_selected >= max_count:
            # 选出的数量足够或过多
            if n_selected > max_count:
                # 保留前 max_count 个（FPS 已经按距离排序）
                selected_indices = selected_indices[:max_count]

            selected_structures = [candidate_trajectory[i] for i in selected_indices]
            print(
                f"✓ FPS 选择完成: {len(candidate_trajectory)} → "
                f"{len(selected_structures)}\n"
            )
            return selected_structures

        # 数量不足，降低 min_distance
        min_distance *= 0.6  # 每次减少 40%

        if min_distance < 1e-8:
            # min_distance 太小了，直接返回所有选中的
            print(
                f"  警告: min_distance 已降至 {min_distance:.2e}，"
                f"仅选出 {n_selected} 个结构"
            )
            selected_structures = [candidate_trajectory[i] for i in selected_indices]
            return selected_structures

    # 达到最大尝试次数，返回当前结果
    print(f"  警告: 达到最大尝试次数，FPS 仅选出 {len(selected_indices)} 个结构")
    selected_structures = [candidate_trajectory[i] for i in selected_indices]
    return selected_structures
