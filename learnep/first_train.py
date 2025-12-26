"""
从头训练第一个 NEP 模型

本模块提供从头训练 NEP 模型的功能，用于生成 initial_nep_model 和 initial_nep_restart。
使用场景：
- 用户有训练数据 train.xyz
- 需要训练出第一个 nep.txt 和 nep.restart
- 然后用这些文件开始主动学习流程
"""

import sys
import argparse
import yaml
from pathlib import Path
import shutil
import logging

from .config import load_config, Config


def setup_logger(log_file: Path) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.Logger("learnep-first-train", level=logging.INFO)

    # 文件处理器
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def first_train(config: Config, logger: logging.Logger) -> None:
    """
    从头训练第一个 NEP 模型

    参数:
        config: 配置对象
        logger: 日志记录器
    """
    work_dir = config.global_config.work_dir

    logger.info("=" * 80)
    logger.info("从头训练第一个 NEP 模型")
    logger.info("=" * 80)

    # 创建训练目录
    train_dir = work_dir / "first_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"创建训练目录: {train_dir}")

    # 复制训练数据
    train_src = config.global_config.initial_train_data
    if not train_src.exists():
        logger.error(f"训练数据不存在: {train_src}")
        raise FileNotFoundError(f"训练数据不存在: {train_src}")

    train_dst = train_dir / "train.xyz"
    shutil.copy2(train_src, train_dst)
    logger.info(f"  复制训练数据: {train_src} -> {train_dst}")

    # 统计训练数据
    from .iteration import read_trajectory

    train_structures = read_trajectory(str(train_dst))
    logger.info(f"  训练集包含 {len(train_structures)} 个结构")

    # 写入 nep.in (使用 first_input_content)
    nep_in_file = train_dir / "nep.in"
    with open(nep_in_file, "w") as f:
        f.write(config.nep.first_input_content)
    logger.info("  创建 nep.in (使用 first_input_content)")

    # 写入作业脚本
    from .iteration import _ensure_done_marker

    job_script_file = train_dir / "job.sh"
    with open(job_script_file, "w") as f:
        f.write(_ensure_done_marker(config.nep.job_script))
    logger.info("  创建作业脚本（已自动添加 DONE 标记）")

    logger.info("")
    logger.info("=" * 80)
    logger.info("准备完成！")
    logger.info("=" * 80)
    logger.info("")
    logger.info("下一步：提交训练任务")
    logger.info("  1. cd {train_dir}")
    logger.info("  2. 提交作业: qsub job.sh (或其他调度命令)")
    logger.info("")
    logger.info("训练完成后：")
    logger.info(f"  - nep.txt 将生成在 {train_dir}/nep.txt")
    logger.info(f"  - nep.restart 将生成在 {train_dir}/nep.restart")
    logger.info("")
    logger.info("然后可以在配置文件中设置：")
    logger.info(f'  initial_nep_model: "{train_dir}/nep.txt"')
    logger.info(f'  initial_nep_restart: "{train_dir}/nep.restart"')
    logger.info("")
    logger.info("之后运行主动学习：")
    logger.info("  learnep config.yaml")
    logger.info("=" * 80)


def main():
    """主入口函数"""
    parser = argparse.ArgumentParser(
        description="从头训练第一个 NEP 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  learnep-first-train config.yaml

说明:
  此命令用于从零开始训练第一个 NEP 模型。
  只需要提供训练数据 (initial_train_data)，不需要 initial_nep_model 和 initial_nep_restart。
  
  训练完成后，生成的 nep.txt 和 nep.restart 可用于开始主动学习流程。
        """,
    )
    parser.add_argument("config", type=str, help="配置文件路径 (YAML)")

    args = parser.parse_args()
    config_file = Path(args.config)

    if not config_file.exists():
        print(f"错误: 配置文件不存在: {config_file}")
        sys.exit(1)

    # 加载配置（不检查 nep.txt 和 nep.restart）
    try:
        with open(config_file) as f:
            raw_config = yaml.safe_load(f)

        # 临时设置 initial_nep_model 和 initial_nep_restart 为 train.xyz 的路径
        # 这样可以绕过检查
        if "global" not in raw_config:
            raw_config["global"] = {}

        work_dir = Path(raw_config["global"].get("work_dir", "./work"))
        raw_config["global"]["initial_nep_model"] = str(work_dir / "dummy_nep.txt")
        raw_config["global"]["initial_nep_restart"] = str(work_dir / "dummy_restart")

        # 创建临时文件以通过验证
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "dummy_nep.txt").touch()
        (work_dir / "dummy_restart").touch()

        config = load_config(raw_config, config_file.parent)

        # 删除临时文件
        (work_dir / "dummy_nep.txt").unlink()
        (work_dir / "dummy_restart").unlink()

    except Exception as e:
        print(f"错误: 配置加载失败: {e}")
        sys.exit(1)

    # 设置日志
    log_file = config.global_config.work_dir / "first_train.log"
    logger = setup_logger(log_file)

    try:
        first_train(config, logger)
    except Exception as e:
        logger.error(f"训练准备失败: {e}")
        logger.exception("详细错误信息:")
        sys.exit(1)


if __name__ == "__main__":
    main()
