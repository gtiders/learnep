"""
主程序入口

完整的主动学习流程：
1. 初始化工作空间（迭代 1）- 使用用户提供的初始文件
2. 运行迭代循环（迭代 1, 2, 3, ...）
3. 直到收敛或达到最大迭代次数
"""

import sys
from pathlib import Path

from .config import load_config, print_config_summary
from .initialize import setup_logger, initialize_workspace
from .iteration import IterationManager


def main() -> None:
    """
    主函数：运行完整的主动学习流程
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="LearnEP - NEP 主动学习框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例:
  # 从头开始运行
  learnep config.yaml

  # 从指定迭代继续运行
  learnep config.yaml --start-iter 5

  # 仅初始化
  learnep-init config.yaml
        """,
    )
    parser.add_argument("config", type=str, help="配置文件路径")
    parser.add_argument(
        "--start-iter",
        type=int,
        default=1,
        help="起始迭代编号 (默认从 iter_1 开始)",
    )
    args = parser.parse_args()

    config_file = args.config
    start_iter = args.start_iter

    if not Path(config_file).exists():
        print(f"错误: 配置文件不存在: {config_file}")
        sys.exit(1)
    # 加载配置
    print("=" * 80)
    print("LearnEP - NEP 主动学习框架")
    print("=" * 80)
    print(f"\n加载配置文件: {config_file}")

    try:
        config = load_config(config_file)
    except Exception as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    # 打印配置摘要
    print_config_summary(config)

    # 设置日志
    logger = setup_logger(config.global_config.log_file)

    # 初始化逻辑：如果 iter_1 不存在，自动初始化
    work_dir = Path(config.global_config.work_dir)
    iter_1_dir = work_dir / "iter_1"

    if start_iter == 1 and not iter_1_dir.exists():
        logger.info("=" * 80)
        logger.info("检测到 iter_1 不存在，开始初始化工作空间")
        logger.info("=" * 80)

        try:
            initialize_workspace(config, logger)
        except Exception as e:
            logger.error(f"初始化失败: {e}")
            sys.exit(1)

    logger.info("=" * 80)
    logger.info("开始主动学习流程")
    logger.info("=" * 80)

    # 创建迭代管理器
    iteration_manager = IterationManager(config, logger)

    # 运行迭代循环
    max_iterations = config.global_config.max_iterations
    current_iter = start_iter

    while current_iter <= max_iterations:
        try:
            # 运行一次迭代
            should_continue = iteration_manager.run_iteration(current_iter)

            if not should_continue:
                # 收敛或失败
                logger.info("\n" + "=" * 80)
                logger.info(f"主动学习在迭代 {current_iter} 结束")
                logger.info("=" * 80)
                break

            # 继续下一轮
            current_iter += 1

        except KeyboardInterrupt:
            logger.warning("\n用户中断程序")
            logger.info(f"当前进度: 迭代 {current_iter}")
            logger.info(
                f"重启命令: python -m nep_auto.main {config_file} --start-iter {current_iter}"
            )
            sys.exit(0)

        except Exception as e:
            logger.error(f"迭代 {current_iter} 发生异常: {e}")
            logger.exception("详细错误信息:")
            sys.exit(1)

    else:
        # 达到最大迭代次数
        logger.info("\n" + "=" * 80)
        logger.info(f"达到最大迭代次数 ({max_iterations})")

        # 检查最后一轮的收敛状态
        final_iter_dir = config.global_config.work_dir / f"iter_{max_iterations}"
        large_gamma_file = final_iter_dir / "large_gamma.xyz"

        if large_gamma_file.exists():
            from .iteration import read_trajectory

            large_gamma_structs = read_trajectory(str(large_gamma_file))

            if len(large_gamma_structs) == 0:
                logger.info("✅ 模型已收敛：所有结构 gamma ≤ 收敛阈值")
                logger.info(f"   最终 NEP 模型: {final_iter_dir / 'nep.txt'}")
            else:
                logger.warning("⚠️  模型未完全收敛")
                logger.warning(f"   仍有 {len(large_gamma_structs)} 个高 gamma 结构")
                logger.warning("   建议解决方案：")
                logger.warning("   1. 增加 max_iterations 继续训练")
                logger.warning("   2. 或调整 gamma_high 阈值降低选择标准")
                logger.warning("   3. 或检查是否需要扩大探索空间")

        logger.info("=" * 80)

    logger.info("\n主动学习流程完成！")


if __name__ == "__main__":
    main()
