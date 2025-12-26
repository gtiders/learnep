"""
初始化脚本

准备第一轮迭代(iter_1)：
1. 从用户提供的初始文件创建 iter_1 目录
2. 生成活跃集（active_set.asi）
3. 准备 GPUMD 探索任务
"""

import shutil
import logging
from pathlib import Path

from .config import Config, load_config
from .maxvol import select_active_set, read_trajectory, write_trajectory, write_asi_file


def _ensure_done_marker(job_script: str) -> str:
    """
    确保作业脚本末尾有 touch DONE 命令

    参数:
        job_script: 原始作业脚本内容

    返回:
        添加了 DONE 标记的脚本
    """
    script = job_script.rstrip()

    # 检查是否已经有 touch DONE
    if "touch DONE" not in script and "touch ./DONE" not in script:
        script += "\n\n# 自动添加：标记任务完成\ntouch DONE\n"

    return script


def setup_logger(log_file: Path) -> logging.Logger:
    """
    设置日志记录器

    参数:
        log_file: 日志文件路径

    返回:
        配置好的 logger
    """
    logger = logging.getLogger("nep_auto")
    logger.setLevel(logging.INFO)

    # 文件处理器
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)

    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def initialize_workspace(config: Config, logger: logging.Logger) -> None:
    """
    初始化工作空间

    第 0 步 (Iteration 1)：
    1. 复制初始 nep.txt 和 train.xyz 到工作目录
    2. 使用 select_active_set() 生成活跃集
    3. 创建 GPUMD 探索任务目录

    参数:
        config: 配置对象
        logger: 日志记录器
    """
    work_dir = config.global_config.work_dir

    logger.info("=" * 80)
    logger.info("开始初始化工作空间（Iteration 1）")
    logger.info("=" * 80)

    # =========================================================================
    # 步骤 1: 创建工作目录结构
    # =========================================================================
    logger.info("\n步骤 1: 创建工作目录结构")

    # 创建迭代 1 目录
    iter0_dir = work_dir / "iter_1"
    iter0_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  创建目录: {iter0_dir}")

    # =========================================================================
    # 步骤 2: 复制初始文件
    # =========================================================================
    logger.info("\n步骤 2: 复制初始文件")

    # 复制 nep.txt
    nep_dst = iter0_dir / "nep.txt"
    shutil.copy2(config.global_config.initial_nep_model, nep_dst)
    logger.info(
        f"  复制 NEP 模型: {config.global_config.initial_nep_model} -> {nep_dst}"
    )

    # 复制 nep.restart
    nep_restart_dst = iter0_dir / "nep.restart"
    shutil.copy2(config.global_config.initial_nep_restart, nep_restart_dst)
    logger.info(
        f"  复制 NEP restart: {config.global_config.initial_nep_restart} -> {nep_restart_dst}"
    )

    # 复制 train.xyz
    train_dst = iter0_dir / "train.xyz"
    shutil.copy2(config.global_config.initial_train_data, train_dst)
    logger.info(
        f"  复制训练数据: {config.global_config.initial_train_data} -> {train_dst}"
    )

    # 统计训练数据
    train_structures = read_trajectory(str(train_dst))
    logger.info(f"  训练集包含 {len(train_structures)} 个结构")

    # =========================================================================
    # 步骤 3: 生成活跃集
    # =========================================================================
    logger.info("\n步骤 3: 生成活跃集（MaxVol 选择）")

    # 预检查：确保每个元素类型都有足够的原子
    logger.info("  检查训练数据是否足够...")
    from collections import Counter

    # 统计每种元素的原子数
    all_symbols = []
    for structure in train_structures:
        all_symbols.extend(structure.get_chemical_symbols())

    element_counts = Counter(all_symbols)
    logger.info("  训练集中的元素统计:")
    for elem, count in sorted(element_counts.items()):
        logger.info(f"    {elem}: {count} 个原子")

    # 从 NEP 文件读取描述符维度和元素列表
    with open(nep_dst) as f:
        first_line = f.readline()
        parts = first_line.split()
        n_types = int(parts[1])
        nep_elements = parts[2 : 2 + n_types]

    # 检查每个元素是否有足够的原子
    # 注意：这里只是粗略估计，实际需要的数量取决于描述符维度
    logger.info(f"\n  NEP 模型中的元素: {', '.join(nep_elements)}")

    missing_elements = []
    insufficient_elements = []

    for elem in nep_elements:
        if elem not in element_counts:
            missing_elements.append(elem)
        elif element_counts[elem] < 10:  # 粗略检查，至少 10 个原子
            insufficient_elements.append((elem, element_counts[elem]))

    if missing_elements:
        error_msg = (
            f"\n{'=' * 80}\n"
            f"错误：训练数据中缺少某些元素\n"
            f"{'=' * 80}\n"
            f"缺少的元素: {', '.join(missing_elements)}\n"
            f"NEP 模型需要: {', '.join(nep_elements)}\n"
            f"训练集中有: {', '.join(element_counts.keys())}\n\n"
            f"解决方案：\n"
            f"  1. 在训练数据中添加包含这些元素的结构\n"
            f"  2. 或修改 NEP 模型以匹配当前的训练数据\n"
            f"{'=' * 80}\n"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if insufficient_elements:
        logger.warning("  警告：以下元素的原子数量可能不足：")
        for elem, count in insufficient_elements:
            logger.warning(f"    {elem}: 只有 {count} 个原子 (建议至少 10 个)")
        logger.warning("  如果 MaxVol 失败，需要添加更多的训练结构")

    try:
        active_set_result, selected_structures = select_active_set(
            trajectory=train_structures,
            nep_file=str(nep_dst),
            gamma_tol=config.selection.gamma_tol,
            batch_size=config.selection.batch_size,
        )

        logger.info("  活跃集生成成功")
        logger.info(f"  选中的结构数量: {len(selected_structures)}")

        # 统计每个元素的活跃环境数量
        total_envs = sum(len(inv) for inv in active_set_result.inverse_dict.values())
        logger.info(f"  活跃环境总数: {total_envs}")

        for element, inv_matrix in active_set_result.inverse_dict.items():
            logger.info(f"    元素 {element}: {len(inv_matrix)} 个活跃环境")

        # 保存活跃集文件
        asi_file = iter0_dir / "active_set.asi"
        write_asi_file(active_set_result.inverse_dict, str(asi_file))
        logger.info(f"  保存活跃集文件: {asi_file}")

        # 保存活跃集结构（可选，用于分析）
        active_xyz = iter0_dir / "active_set.xyz"
        write_trajectory(selected_structures, str(active_xyz))
        logger.info(f"  保存活跃集结构: {active_xyz}")

    except Exception as e:
        logger.error(f"  活跃集生成失败: {e}")
        raise

    # =========================================================================
    # 步骤 4: 准备 GPUMD 探索任务
    # =========================================================================
    logger.info("\n步骤 4: 准备 GPUMD 探索任务")

    # 创建 GPUMD 总目录
    gpumd_dir = iter0_dir / "gpumd"
    gpumd_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"  创建 GPUMD 目录: {gpumd_dir}")

    # 为每个探索条件创建子目录
    for cond in config.gpumd.conditions:
        cond_dir = gpumd_dir / cond.id
        cond_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"  创建探索条件: {cond.id}")

        # 复制结构文件
        structure_dst = cond_dir / "model.xyz"
        shutil.copy2(cond.structure_file, structure_dst)
        logger.info(f"    复制结构文件: {cond.structure_file.name}")

        # 复制 NEP 模型
        nep_gpumd = cond_dir / "nep.txt"
        shutil.copy2(nep_dst, nep_gpumd)
        logger.info("    复制 NEP 模型")

        # 复制活跃集
        asi_gpumd = cond_dir / "active_set.asi"
        shutil.copy2(asi_file, asi_gpumd)
        logger.info("    复制活跃集文件")

        # 写入 run.in
        run_in_file = cond_dir / "run.in"
        with open(run_in_file, "w") as f:
            f.write(cond.run_in_content)
        logger.info("    创建 run.in")

        # 写入作业脚本（自动添加 DONE 标记）
        job_script_file = cond_dir / "job.sh"
        with open(job_script_file, "w") as f:
            f.write(_ensure_done_marker(config.gpumd.job_script))
        logger.info("    创建作业脚本（已自动添加 DONE 标记）")

    # =========================================================================
    # 步骤 5: 创建状态文件
    # =========================================================================
    logger.info("\n步骤 5: 创建状态文件")

    # 创建 DONE 文件标记初始化完成
    done_file = iter0_dir / "DONE"
    done_file.touch()
    logger.info(f"  创建 DONE 文件: {done_file}")

    logger.info("\n" + "=" * 80)
    logger.info("初始化完成！")
    logger.info("=" * 80)
    logger.info("\n下一步: 提交 GPUMD 探索任务")
    logger.info(f"  任务目录: {gpumd_dir}")
    logger.info(
        f"  提交命令: cd <condition_dir> && {config.global_config.submit_command}"
    )


def main():
    """
    主函数：执行初始化流程
    """
    import sys

    if len(sys.argv) < 2:
        print("用法: learnep-init <config_file.yaml>")
        print("\n示例:")
        print("  learnep-init config.yaml")
        sys.exit(1)

    config_file = sys.argv[1]

    # 加载配置
    print("加载配置文件...")
    config = load_config(config_file)

    # 设置日志
    logger = setup_logger(config.global_config.log_file)

    # 执行初始化
    try:
        initialize_workspace(config, logger)
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise


if __name__ == "__main__":
    main()
