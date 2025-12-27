"""
配置加载模块

从 YAML 配置文件中加载主动学习的所有参数，并提供路径解析、验证等功能。
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass
class GlobalConfig:
    """全局配置"""

    work_dir: Path
    max_iterations: int
    max_structures_per_iteration: int
    log_file: Path
    initial_nep_model: Path
    initial_nep_restart: Path
    initial_train_data: Path
    submit_command: str
    check_interval: int
    scheduler_type: str  # 调度系统类型: "pbs", "slurm", "lsf", "auto"
    job_status_command: str  # 作业状态查询命令，{job_id} 会被替换


@dataclass
class VaspConfig:
    """VASP 配置"""

    incar_file: Path
    potcar_file: Path
    kpoints_file: Path
    job_script: str
    timeout: int


@dataclass
class NepConfig:
    """NEP 配置"""

    input_content: str
    first_input_content: str
    job_script: str
    timeout: int
    prune_train_set: bool
    max_structures_factor: float


@dataclass
class GpumdCondition:
    """GPUMD 单个探索条件"""

    id: str
    structure_file: Path
    run_in_content: str


@dataclass
class GpumdConfig:
    """GPUMD 配置"""

    conditions: List[GpumdCondition]
    job_script: str
    timeout: int


@dataclass
class SelectionConfig:
    """结构选择配置

    method: 选择方法，可选值:
        - "maxvol": 使用 MaxVol 算法选择能最大化描述符空间覆盖的结构
        - "fps": 使用 FPS (最远点采样) 算法选择多样性最大的结构
    """

    method: str  # 选择方法: "maxvol" 或 "fps"
    gamma_tol: float  # MaxVol 收敛阈值
    batch_size: int  # 批处理大小
    fps_min_distance: float  # FPS 初始最小距离


@dataclass
class BootstrapCondition:
    """冷启动条件配置"""

    id: str
    structure_file: Path
    run_in_content: str


@dataclass
class BootstrapFilterConfig:
    """冷启动结构过滤配置"""

    min_distance: float  # 最小原子间距（Å）
    max_force: float  # 最大原子力（eV/Å）
    max_energy_per_atom: float  # 最大每原子能量偏差（eV）


@dataclass
class BootstrapConfig:
    """冷启动模式配置"""

    enabled: bool
    conditions: list[BootstrapCondition]
    job_script: str
    filter: BootstrapFilterConfig
    timeout: int


@dataclass
class Config:
    """完整配置"""

    global_config: GlobalConfig
    vasp: VaspConfig
    nep: NepConfig
    gpumd: GpumdConfig
    selection: SelectionConfig
    bootstrap: BootstrapConfig


def _resolve_path(path_str: str, work_dir: Path) -> Path:
    """
    解析路径：绝对路径直接使用，相对路径以 work_dir 为基准

    参数:
        path_str: 路径字符串
        work_dir: 工作目录

    返回:
        解析后的绝对路径
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    else:
        return work_dir / path


def _validate_gpumd_run_in(run_in_content: str, condition_id: str) -> None:
    """
    验证 GPUMD run.in 内容是否包含 compute_extrapolation 指令

    参数:
        run_in_content: run.in 文件内容
        condition_id: 条件 ID（用于错误提示）

    抛出:
        ValueError: 如果缺少 compute_extrapolation 指令
    """
    if not re.search(r"^\s*compute_extrapolation\s+", run_in_content, re.MULTILINE):
        raise ValueError(
            f"GPUMD condition '{condition_id}' 的 run_in_content 中缺少 "
            f"'compute_extrapolation' 指令。这是全自动化流程必需的！"
        )


def load_config(config_file: str) -> Config:
    """
    从 YAML 文件加载配置

    参数:
        config_file: 配置文件路径

    返回:
        完整的配置对象

    抛出:
        FileNotFoundError: 配置文件不存在
        ValueError: 配置验证失败
        yaml.YAMLError: YAML 解析失败
    """
    config_path = Path(config_file)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_file}")

    # 读取 YAML
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raise ValueError(f"配置文件为空: {config_file}")

    # 解析全局配置
    global_raw = raw_config.get("global", {})
    work_dir = Path(global_raw.get("work_dir", "./work")).resolve()

    # 确保工作目录存在
    work_dir.mkdir(parents=True, exist_ok=True)

    # 解析路径（相对于 work_dir）
    log_file = _resolve_path(
        global_raw.get("log_file", "active_learning.log"), work_dir
    )
    initial_nep_model = _resolve_path(
        global_raw.get("initial_nep_model", "nep.txt"), work_dir
    )
    initial_nep_restart = _resolve_path(
        global_raw.get("initial_nep_restart", "nep.restart"), work_dir
    )
    initial_train_data = _resolve_path(
        global_raw.get("initial_train_data", "train.xyz"), work_dir
    )

    # 验证初始文件是否存在
    if not initial_nep_model.exists():
        raise FileNotFoundError(f"初始 NEP 模型文件不存在: {initial_nep_model}")
    if not initial_nep_restart.exists():
        raise FileNotFoundError(f"初始 NEP restart 文件不存在: {initial_nep_restart}")
    if not initial_train_data.exists():
        raise FileNotFoundError(f"初始训练数据文件不存在: {initial_train_data}")

    global_config = GlobalConfig(
        work_dir=work_dir,
        max_iterations=global_raw.get("max_iterations", 100),
        max_structures_per_iteration=global_raw.get("max_structures_per_iteration", 50),
        log_file=log_file,
        initial_nep_model=initial_nep_model,
        initial_nep_restart=initial_nep_restart,
        initial_train_data=initial_train_data,
        submit_command=global_raw.get("submit_command", "qsub job.sh"),
        check_interval=global_raw.get("check_interval", 30),
        scheduler_type=global_raw.get("scheduler_type", "pbs"),
        job_status_command=global_raw.get("job_status_command", "qstat {job_id}"),
    )

    # 解析 VASP 配置
    vasp_raw = raw_config.get("vasp", {})
    vasp_config = VaspConfig(
        incar_file=_resolve_path(vasp_raw.get("incar_file", "input/INCAR"), work_dir),
        potcar_file=_resolve_path(
            vasp_raw.get("potcar_file", "input/POTCAR"), work_dir
        ),
        kpoints_file=_resolve_path(
            vasp_raw.get("kpoints_file", "input/KPOINTS"), work_dir
        ),
        job_script=vasp_raw.get("job_script", ""),
        timeout=vasp_raw.get("timeout", 172800),
    )

    # 验证 VASP 输入文件是否存在
    if not vasp_config.incar_file.exists():
        raise FileNotFoundError(f"VASP INCAR 文件不存在: {vasp_config.incar_file}")
    if not vasp_config.potcar_file.exists():
        raise FileNotFoundError(f"VASP POTCAR 文件不存在: {vasp_config.potcar_file}")
    if not vasp_config.kpoints_file.exists():
        raise FileNotFoundError(f"VASP KPOINTS 文件不存在: {vasp_config.kpoints_file}")

    # 解析 NEP 配置
    nep_raw = raw_config.get("nep", {})
    nep_config = NepConfig(
        input_content=nep_raw.get("input_content", ""),
        first_input_content=nep_raw.get("first_input_content", ""),
        job_script=nep_raw.get("job_script", ""),
        timeout=nep_raw.get("timeout", 259200),
        prune_train_set=nep_raw.get("prune_train_set", True),
        max_structures_factor=nep_raw.get("max_structures_factor", 1.0),
    )

    # 解析 GPUMD 配置
    gpumd_raw = raw_config.get("gpumd", {})
    conditions_raw = gpumd_raw.get("conditions", [])

    if not conditions_raw:
        raise ValueError("GPUMD 配置中必须至少包含一个探索条件")

    conditions = []
    for cond_raw in conditions_raw:
        cond_id = cond_raw.get("id", "")
        if not cond_id:
            raise ValueError("GPUMD condition 必须包含 'id' 字段")

        structure_file = _resolve_path(cond_raw.get("structure_file", ""), work_dir)

        if not structure_file.exists():
            raise FileNotFoundError(
                f"GPUMD condition '{cond_id}' 的结构文件不存在: {structure_file}"
            )

        run_in_content = cond_raw.get("run_in_content", "")

        # 验证是否包含 compute_extrapolation
        _validate_gpumd_run_in(run_in_content, cond_id)

        conditions.append(
            GpumdCondition(
                id=cond_id,
                structure_file=structure_file,
                run_in_content=run_in_content,
            )
        )

    gpumd_config = GpumdConfig(
        conditions=conditions,
        job_script=gpumd_raw.get("job_script", ""),
        timeout=gpumd_raw.get("timeout", 86400),
    )

    # 解析选择配置
    selection_raw = raw_config.get("selection", {})

    # 处理向后兼容：如果使用旧的 fps_enabled 参数，转换为新的 method 参数
    if "method" in selection_raw:
        method = selection_raw.get("method", "maxvol").lower()
        if method not in ("maxvol", "fps"):
            raise ValueError(
                f"selection.method 必须是 'maxvol' 或 'fps'，当前值: {method}"
            )
    else:
        # 向后兼容：fps_enabled=True 对应 method="fps"，否则 method="maxvol"
        fps_enabled = selection_raw.get("fps_enabled", False)
        method = "fps" if fps_enabled else "maxvol"

    selection_config = SelectionConfig(
        method=method,
        gamma_tol=selection_raw.get("gamma_tol", 1.001),
        batch_size=selection_raw.get("batch_size", 10000),
        fps_min_distance=selection_raw.get("fps_min_distance", 0.01),
    )

    # 解析冷启动配置
    bootstrap_raw = raw_config.get("bootstrap", {})
    bootstrap_conditions = []
    for cond in bootstrap_raw.get("conditions", []):
        bootstrap_conditions.append(
            BootstrapCondition(
                id=cond.get("id", "bootstrap"),
                structure_file=_resolve_path(cond.get("structure_file", ""), work_dir),
                run_in_content=cond.get("run_in", ""),
            )
        )

    bootstrap_filter_raw = bootstrap_raw.get("filter", {})
    bootstrap_filter = BootstrapFilterConfig(
        min_distance=bootstrap_filter_raw.get("min_distance", 1.0),
        max_force=bootstrap_filter_raw.get("max_force", 50.0),
        max_energy_per_atom=bootstrap_filter_raw.get("max_energy_per_atom", 5.0),
    )

    bootstrap_config = BootstrapConfig(
        enabled=bootstrap_raw.get("enabled", True),
        conditions=bootstrap_conditions,
        job_script=bootstrap_raw.get("job_script", "#!/bin/bash\ngpumd\n"),
        filter=bootstrap_filter,
        timeout=bootstrap_raw.get("timeout", 86400),
    )

    return Config(
        global_config=global_config,
        vasp=vasp_config,
        nep=nep_config,
        gpumd=gpumd_config,
        selection=selection_config,
        bootstrap=bootstrap_config,
    )


def print_config_summary(config: Config) -> None:
    """
    打印配置摘要（用于调试和确认）

    参数:
        config: 配置对象
    """
    print("=" * 80)
    print("配置摘要")
    print("=" * 80)

    print("\n[全局配置]")
    print(f"  工作目录: {config.global_config.work_dir}")
    print(f"  最大迭代次数: {config.global_config.max_iterations}")
    print(f"  每轮最大结构数: {config.global_config.max_structures_per_iteration}")
    print(f"  日志文件: {config.global_config.log_file}")
    print(f"  初始 NEP 模型: {config.global_config.initial_nep_model}")
    print(f"  初始 NEP restart: {config.global_config.initial_nep_restart}")
    print(f"  初始训练数据: {config.global_config.initial_train_data}")
    print(f"  任务提交命令: {config.global_config.submit_command}")

    print("\n[VASP 配置]")
    print(f"  INCAR: {config.vasp.incar_file}")
    print(f"  POTCAR: {config.vasp.potcar_file}")
    print(f"  KPOINTS: {config.vasp.kpoints_file}")
    print(f"  超时时间: {config.vasp.timeout} 秒")

    print("\n[NEP 配置]")
    print(f"  超时时间: {config.nep.timeout} 秒")
    print(f"  输入内容行数: {len(config.nep.input_content.splitlines())}")

    print("\n[GPUMD 配置]")
    print(f"  探索条件数量: {len(config.gpumd.conditions)}")
    for cond in config.gpumd.conditions:
        print(f"    - {cond.id}: {cond.structure_file}")
    print(f"  超时时间: {config.gpumd.timeout} 秒")

    print("\n[结构选择配置]")
    print(f"  选择方法: {config.selection.method}")
    print(f"  Gamma 阈值: {config.selection.gamma_tol}")
    print(f"  批处理大小: {config.selection.batch_size}")
    if config.selection.method == "fps":
        print(f"  FPS 初始最小距离: {config.selection.fps_min_distance}")

    print("=" * 80)


def main():
    """主函数：验证配置文件"""
    import sys

    if len(sys.argv) < 2:
        print("用法: learnep-config <config_file.yaml>")
        sys.exit(1)

    try:
        config = load_config(sys.argv[1])
        print_config_summary(config)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 测试配置加载
    import sys

    if len(sys.argv) < 2:
        print("用法: python config.py <config_file.yaml>")
        sys.exit(1)

    try:
        config = load_config(sys.argv[1])
        print_config_summary(config)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
