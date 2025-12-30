"""
JAXVol 包初始化文件。
暴露核心算法接口。
默认启用 JAX 64位精度以确保数值稳定性。
"""

import jax

# Enable 64-bit precision globally
jax.config.update("jax_enable_x64", True)

from .selector import calculate_maxvol, find_inverse  # noqa: E402
from .core import maxvol  # noqa: E402
from .cli import active, extend, gamma  # noqa: E402

__all__ = [
    "calculate_maxvol",
    "find_inverse",
    "maxvol",
    "active",
    "extend",
    "gamma",
]
