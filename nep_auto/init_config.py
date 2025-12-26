"""
配置文件初始化工具

将模板配置文件复制到当前目录
"""

import shutil
import sys
from pathlib import Path


def main() -> None:
    """
    将模板配置文件复制到当前目录
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="NEP Auto 配置文件初始化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 在当前目录生成 config.yaml
  nep-auto-init-config

  # 指定输出文件名
  nep-auto-init-config -o my_config.yaml
        """,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="config.yaml",
        help="输出配置文件名 (默认: config.yaml)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="强制覆盖已存在的配置文件",
    )
    args = parser.parse_args()

    # 获取模板配置文件路径
    template_file = Path(__file__).parent / "config_example.yaml"

    if not template_file.exists():
        print(f"错误: 模板配置文件不存在: {template_file}")
        sys.exit(1)

    # 输出文件路径
    output_file = Path(args.output)

    # 检查输出文件是否已存在
    if output_file.exists() and not args.force:
        print(f"错误: 配置文件已存在: {output_file}")
        print("使用 -f/--force 参数强制覆盖")
        sys.exit(1)

    # 复制模板文件
    try:
        shutil.copy2(template_file, output_file)
        print(f"✅ 配置文件已生成: {output_file}")
        print(f"\n请编辑 {output_file} 设置您的参数")
        print("\n主要需要配置的内容:")
        print("  - global.work_dir: 工作目录")
        print("  - global.initial_nep_model: 初始 NEP 模型文件")
        print("  - global.initial_nep_restart: 初始 NEP restart 文件")
        print("  - global.initial_train_data: 初始训练数据文件")
        print("  - vasp.incar_file, potcar_file, kpoints_file: VASP 输入文件")
        print("  - gpumd.conditions: GPUMD 探索条件")
        print("  - nep.input_content: NEP 训练参数")
        print("  - nep.prune_train_set: 是否启用训练集修剪")
        print("  - nep.max_structures_factor: 训练集最大规模系数")
        print("\n配置完成后运行:")
        print(f"  nep-auto-main {output_file}")
    except Exception as e:
        print(f"错误: 复制配置文件失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
