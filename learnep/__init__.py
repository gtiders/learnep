"""
NEP Auto - NEP 主动学习自动化框架

基于 MaxVol 算法的 NEP 势函数主动学习框架
"""

__version__ = "0.1.0"

# 导出主要接口
from .config import (
    Config,
    GlobalConfig,
    VaspConfig,
    NepConfig,
    GpumdConfig,
    GpumdCondition,
    SelectionConfig,
    load_config,
    print_config_summary,
)

from .maxvol import (
    ActiveSetResult,
    DescriptorProjectionResult,
    compute_maxvol,
    compute_descriptor_projection,
    compute_gamma,
    generate_active_set,
    write_asi_file,
    read_asi_file,
    select_active_set,
    select_extension_structures,
    filter_high_gamma_structures,
    read_trajectory,
    write_trajectory,
)

from .initialize import initialize_workspace, setup_logger

from .iteration import IterationManager, TaskManager

from .main import main

__all__ = [
    # 版本信息
    "__version__",
    # 配置
    "Config",
    "GlobalConfig",
    "VaspConfig",
    "NepConfig",
    "GpumdConfig",
    "GpumdCondition",
    "SelectionConfig",
    "load_config",
    "print_config_summary",
    # MaxVol 算法
    "ActiveSetResult",
    "DescriptorProjectionResult",
    "compute_maxvol",
    "compute_descriptor_projection",
    "compute_gamma",
    "generate_active_set",
    "write_asi_file",
    "read_asi_file",
    "select_active_set",
    "select_extension_structures",
    "filter_high_gamma_structures",
    "read_trajectory",
    "write_trajectory",
    # 初始化
    "initialize_workspace",
    "setup_logger",
    # 迭代管理
    "IterationManager",
    "TaskManager",
    # 主程序
    "main",
]
