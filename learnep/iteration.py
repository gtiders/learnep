"""
主动学习迭代模块

实现主动学习的核心迭代循环，包括：
1. GPUMD 探索
2. 结构筛选
3. VASP DFT 标注
4. NEP 训练
5. 活跃集更新
"""

import shutil
import subprocess
import time
import logging
from pathlib import Path
from typing import List, Optional

from ase import Atoms

from .config import Config
from .maxvol import (
    select_active_set,
    select_candidates_maxvol,
    select_candidates_fps,
    read_trajectory,
    write_trajectory,
    write_asi_file,
)


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


class TaskManager:
    """任务管理器：提交和监控作业"""

    def __init__(self, config: Config, logger: logging.Logger):
        """
        初始化任务管理器

        参数:
            config: 配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.submit_command = config.global_config.submit_command
        self.check_interval = config.global_config.check_interval

    def submit_job(self, job_dir: Path) -> bool:
        """
        在指定目录提交作业

        参数:
            job_dir: 作业目录

        返回:
            是否提交成功
        """
        try:
            # 切换到作业目录并执行提交命令
            result = subprocess.run(
                self.submit_command,
                shell=True,
                cwd=job_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.logger.info(f"  作业已提交: {job_dir}")
                if result.stdout.strip():
                    self.logger.info(f"    输出: {result.stdout.strip()}")
                return True
            else:
                self.logger.error(f"  作业提交失败: {job_dir}")
                self.logger.error(f"    错误: {result.stderr}")
                return False

        except Exception as e:
            self.logger.error(f"  提交作业时发生异常: {e}")
            return False

    def wait_for_completion(
        self, job_dirs: List[Path], timeout: Optional[int] = None
    ) -> bool:
        """
        等待所有作业完成（通过检测 DONE 文件）

        参数:
            job_dirs: 作业目录列表
            timeout: 超时时间（秒），None 表示无限等待

        返回:
            是否所有作业都成功完成
        """
        start_time = time.time()
        pending_jobs = list(job_dirs)

        self.logger.info(f"等待 {len(pending_jobs)} 个作业完成...")

        while pending_jobs:
            # 检查超时
            if timeout and (time.time() - start_time) > timeout:
                self.logger.warning(
                    f"等待超时（{timeout} 秒），剩余 {len(pending_jobs)} 个作业"
                )
                return False

            # 检查每个作业的 DONE 文件
            completed = []
            for job_dir in pending_jobs:
                done_file = job_dir / "DONE"
                if done_file.exists():
                    completed.append(job_dir)
                    self.logger.info(f"  作业完成: {job_dir.name}")

            # 移除已完成的作业
            for job_dir in completed:
                pending_jobs.remove(job_dir)

            # 如果还有未完成的作业，等待一段时间后再检查
            if pending_jobs:
                time.sleep(self.check_interval)

        self.logger.info("所有作业已完成")
        return True


class IterationManager:
    """迭代管理器：管理主动学习循环"""

    def __init__(self, config: Config, logger: logging.Logger):
        """
        初始化迭代管理器

        参数:
            config: 配置对象
            logger: 日志记录器
        """
        self.config = config
        self.logger = logger
        self.work_dir = config.global_config.work_dir
        self.task_manager = TaskManager(config, logger)

    def run_gpumd(self, iter_num: int) -> bool:
        """
        运行 GPUMD 探索

        参数:
            iter_num: 当前迭代编号

        返回:
            是否成功
        """
        self.logger.info("=" * 80)
        self.logger.info(f"步骤 1: GPUMD 探索（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        gpumd_dir = iter_dir / "gpumd"

        # 检查是否已经运行过
        if (iter_dir / "large_gamma.xyz").exists():
            self.logger.info("GPUMD 探索已完成，跳过此步骤")
            return True

        # 如果GPUMD目录不存在，尝试准备它
        if not gpumd_dir.exists():
            self.logger.info("GPUMD 目录不存在，准备创建...")

            # 检查上一轮是否存在
            if iter_num > 1:
                # iter_2+ 从上一轮复制
                prev_iter_dir = self.work_dir / f"iter_{iter_num - 1}"
                if not prev_iter_dir.exists():
                    self.logger.error(f"上一轮目录不存在: {prev_iter_dir}")
                    self.logger.error("请确保从 iter_1 开始或使用 --start-iter 1")
                    return False

                # 复制必要文件
                for filename in ["nep.txt", "active_set.asi", "train.xyz"]:
                    src = prev_iter_dir / filename
                    if src.exists():
                        shutil.copy2(src, iter_dir / filename)
                        self.logger.info(f"  复制: {filename}")
                    else:
                        self.logger.error(f"  文件不存在: {src}")
                        return False

            elif iter_num == 1:
                # iter_1 从用户提供的初始文件获取
                self.logger.info("这是第一轮迭代，从配置文件获取初始文件...")
                iter_dir.mkdir(parents=True, exist_ok=True)

                # 复制初始 nep.txt
                nep_src = Path(self.config.global_config.initial_nep_model)
                if nep_src.exists():
                    shutil.copy2(nep_src, iter_dir / "nep.txt")
                    self.logger.info(f"  复制初始 NEP 模型: {nep_src.name}")
                else:
                    self.logger.error(f"  初始 NEP 模型不存在: {nep_src}")
                    return False

                # 复制初始 train.xyz
                train_src = Path(self.config.global_config.initial_train_data)
                if train_src.exists():
                    shutil.copy2(train_src, iter_dir / "train.xyz")
                    self.logger.info(f"  复制初始训练数据: {train_src.name}")
                else:
                    self.logger.error(f"  初始训练数据不存在: {train_src}")
                    return False

                # 生成活跃集
                self.logger.info("  从初始数据生成活跃集...")
                try:
                    train_structures = read_trajectory(str(iter_dir / "train.xyz"))
                    active_set_result, _ = select_active_set(
                        trajectory=train_structures,
                        nep_file=str(iter_dir / "nep.txt"),
                        gamma_tol=self.config.selection.gamma_tol,
                        batch_size=self.config.selection.batch_size,
                    )
                    write_asi_file(
                        active_set_result.inverse_dict,
                        str(iter_dir / "active_set.asi"),
                    )
                    total = sum(
                        len(inv) for inv in active_set_result.inverse_dict.values()
                    )
                    self.logger.info(f"  活跃集包含 {total} 个环境")
                except Exception as e:
                    self.logger.error(f"  生成活跃集失败: {e}")
                    return False

            else:
                self.logger.error("iter_num 必须 >= 1")
                return False

            # 创建 GPUMD 目录结构
            gpumd_dir.mkdir(parents=True, exist_ok=True)

            # 为每个条件创建目录
            for cond in self.config.gpumd.conditions:
                cond_dir = gpumd_dir / cond.id
                cond_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"  创建条件目录: {cond.id}")

                # 复制结构文件
                structure_dst = cond_dir / "model.xyz"
                shutil.copy2(cond.structure_file, structure_dst)

                # 复制 NEP 和活跃集
                shutil.copy2(iter_dir / "nep.txt", cond_dir / "nep.txt")
                shutil.copy2(iter_dir / "active_set.asi", cond_dir / "active_set.asi")

                # 写入 run.in
                with open(cond_dir / "run.in", "w") as f:
                    f.write(cond.run_in_content)

                # 写入作业脚本（自动添加 DONE 标记）
                with open(cond_dir / "job.sh", "w") as f:
                    f.write(_ensure_done_marker(self.config.gpumd.job_script))

            self.logger.info("GPUMD 目录准备完成")

        # 收集所有条件目录
        job_dirs = []
        for cond in self.config.gpumd.conditions:
            cond_dir = gpumd_dir / cond.id
            if not cond_dir.exists():
                self.logger.error(f"条件目录不存在: {cond_dir}")
                return False
            job_dirs.append(cond_dir)

        # 提交所有作业
        self.logger.info(f"提交 {len(job_dirs)} 个 GPUMD 作业...")
        for job_dir in job_dirs:
            if not self.task_manager.submit_job(job_dir):
                return False

        # 等待完成
        if not self.task_manager.wait_for_completion(
            job_dirs, timeout=self.config.gpumd.timeout
        ):
            return False

        # 合并所有 extrapolation_dump.xyz
        self.logger.info("\n合并高 Gamma 结构...")
        large_gamma_file = iter_dir / "large_gamma.xyz"

        all_structures = []
        for job_dir in job_dirs:
            dump_file = job_dir / "extrapolation_dump.xyz"
            if dump_file.exists():
                try:
                    structures = read_trajectory(str(dump_file))
                    all_structures.extend(structures)
                    self.logger.info(f"  {job_dir.name}: {len(structures)} 个结构")
                except Exception as e:
                    self.logger.warning(f"  读取 {dump_file} 失败: {e}")

        # 保存合并结果
        if all_structures:
            write_trajectory(all_structures, str(large_gamma_file))
            self.logger.info(f"总共收集到 {len(all_structures)} 个高 Gamma 结构")
            self.logger.info(f"保存到: {large_gamma_file}")
        else:
            # 创建空文件
            large_gamma_file.touch()
            self.logger.info("未收集到高 Gamma 结构（训练可能已收敛）")

        return True

    def select_structures(self, iter_num: int) -> List[Atoms]:
        """
        选择待标注的新结构

        支持两种独立的选择算法：
        - maxvol: 使用 MaxVol 算法选择能最大化描述符空间的结构
        - fps: 使用最远点采样选择多样性最大的结构

        参数:
            iter_num: 当前迭代编号

        返回:
            选中的结构列表
        """
        self.logger.info("=" * 80)
        self.logger.info(f"步骤 2: 结构筛选（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        train_file = iter_dir / "train.xyz"
        large_gamma_file = iter_dir / "large_gamma.xyz"
        nep_file = iter_dir / "nep.txt"

        # 检查输入文件
        if not large_gamma_file.exists():
            self.logger.error(f"large_gamma.xyz 不存在: {large_gamma_file}")
            return []

        # 检查是否为空
        if large_gamma_file.stat().st_size == 0:
            self.logger.info("large_gamma.xyz 为空，没有新结构需要标注")
            return []

        # 读取文件
        train_structures = read_trajectory(str(train_file))
        candidate_structures = read_trajectory(str(large_gamma_file))

        self.logger.info(f"训练集结构数: {len(train_structures)}")
        self.logger.info(f"候选结构数: {len(candidate_structures)}")

        max_structures = self.config.global_config.max_structures_per_iteration
        method = self.config.selection.method.lower()

        if method == "fps":
            # FPS 模式：直接使用最远点采样
            self.logger.info(f"\n使用 FPS 选择（目标: {max_structures} 个结构）...")

            selected = select_candidates_fps(
                candidate_trajectory=candidate_structures,
                nep_file=str(nep_file),
                max_count=max_structures,
                initial_min_distance=self.config.selection.fps_min_distance,
                show_progress=False,
            )
            self.logger.info(f"FPS 选中 {len(selected)} 个结构")

        else:
            # MaxVol 模式（默认）
            self.logger.info(f"\n使用 MaxVol 选择（目标: {max_structures} 个结构）...")

            selected = select_candidates_maxvol(
                train_trajectory=train_structures,
                candidate_trajectory=candidate_structures,
                nep_file=str(nep_file),
                max_count=max_structures,
                gamma_tol=self.config.selection.gamma_tol,
                batch_size=self.config.selection.batch_size,
                show_progress=False,
            )
            self.logger.info(f"MaxVol 选中 {len(selected)} 个结构")

        return selected

    def run_vasp(self, iter_num: int, structures: List[Atoms]) -> bool:
        """
        运行 VASP DFT 计算

        参数:
            iter_num: 当前迭代编号
            structures: 待计算的结构列表

        返回:
            是否成功
        """
        if not structures:
            self.logger.info("没有需要 DFT 标注的结构，跳过 VASP 步骤")
            return True

        self.logger.info("=" * 80)
        self.logger.info(f"步骤 3: VASP DFT 标注（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        vasp_dir = iter_dir / "vasp"
        vasp_dir.mkdir(parents=True, exist_ok=True)

        # 为每个结构创建计算目录
        job_dirs = []
        for i, structure in enumerate(structures):
            task_dir = vasp_dir / f"task_{i:04d}"
            task_dir.mkdir(parents=True, exist_ok=True)

            # 写入 POSCAR
            from ase.io import write as ase_write

            ase_write(str(task_dir / "POSCAR"), structure, format="vasp", direct=True)

            # 复制输入文件
            shutil.copy2(self.config.vasp.incar_file, task_dir / "INCAR")
            shutil.copy2(self.config.vasp.potcar_file, task_dir / "POTCAR")
            shutil.copy2(self.config.vasp.kpoints_file, task_dir / "KPOINTS")

            # 写入作业脚本（自动添加 DONE 标记）
            with open(task_dir / "job.sh", "w") as f:
                f.write(_ensure_done_marker(self.config.vasp.job_script))

            job_dirs.append(task_dir)

        self.logger.info(f"创建了 {len(job_dirs)} 个 VASP 计算任务")

        # 提交所有作业
        self.logger.info("\n提交 VASP 作业...")
        for job_dir in job_dirs:
            if not self.task_manager.submit_job(job_dir):
                return False

        # 等待完成
        if not self.task_manager.wait_for_completion(
            job_dirs, timeout=self.config.vasp.timeout
        ):
            return False

        # 收集结果并追加到训练集
        self.logger.info("\n收集 DFT 计算结果...")
        train_file = iter_dir / "train.xyz"
        new_structures = []
        failed_tasks = []  # 记录失败的任务

        for i, job_dir in enumerate(job_dirs):
            outcar_file = job_dir / "OUTCAR"

            if not outcar_file.exists():
                self.logger.warning(f"  任务 {i}: OUTCAR 文件不存在: {job_dir.name}")
                failed_tasks.append((i, job_dir.name, "OUTCAR不存在"))
                continue

            try:
                from ase.io import read as ase_read

                structure = ase_read(str(outcar_file), format="vasp-out")

                # 验证结构是否包含必要的信息
                if not hasattr(structure, "get_potential_energy"):
                    raise ValueError("结构缺少能量信息")

                # 尝试获取能量和力，确保数据完整
                _ = structure.get_potential_energy()
                forces = structure.get_forces()

                if forces is None or len(forces) == 0:
                    raise ValueError("结构缺少力信息")

                new_structures.append(structure)

            except Exception as e:
                self.logger.warning(f"  任务 {i}: 读取 {outcar_file.name} 失败: {e}")
                failed_tasks.append((i, job_dir.name, str(e)))

        # 统计结果
        total_tasks = len(job_dirs)
        success_count = len(new_structures)
        failed_count = len(failed_tasks)

        self.logger.info("\nDFT 计算统计:")
        self.logger.info(f"  总任务数: {total_tasks}")
        self.logger.info(f"  成功: {success_count}")
        self.logger.info(f"  失败: {failed_count}")

        if failed_tasks:
            self.logger.info("\n失败任务详情:")
            for task_id, task_name, reason in failed_tasks:
                self.logger.info(f"  - {task_name}: {reason}")

        if new_structures:
            # 追加到训练集
            existing = read_trajectory(str(train_file))
            all_structures = existing + new_structures
            write_trajectory(all_structures, str(train_file))
            self.logger.info(f"\n成功标注 {len(new_structures)} 个结构")
            self.logger.info(f"训练集更新为 {len(all_structures)} 个结构")
            return True
        else:
            self.logger.error("未成功收集到任何 DFT 结果")
            return False

    def run_nep(self, iter_num: int) -> bool:
        """
        运行 NEP 训练

        参数:
            iter_num: 当前迭代编号

        返回:
            是否成功
        """
        self.logger.info("=" * 80)
        self.logger.info(f"步骤 4: NEP 训练（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        nep_dir = iter_dir / "nep_train"
        nep_dir.mkdir(parents=True, exist_ok=True)

        # 复制训练数据
        train_file = iter_dir / "train.xyz"
        train_xyz_dst = nep_dir / "train.xyz"

        # 训练集修剪（可选）
        if self.config.nep.prune_train_set:
            self.logger.info("\n检查训练集大小...")

            # 读取 NEP 模型以获取描述符维度
            # 注意：这里需要先有 nep.txt，所以在 iter_1 时使用初始模型
            if iter_num == 1:
                nep_for_check = Path(self.config.global_config.initial_nep_model)
            else:
                prev_iter_dir = self.work_dir / f"iter_{iter_num - 1}"
                nep_for_check = prev_iter_dir / "nep.txt"

            if not nep_for_check.exists():
                self.logger.error(
                    f"  NEP 模型不存在，无法计算描述符维度: {nep_for_check}"
                )
                return False

            # 从 NEP 文件读取描述符维度
            try:
                from .maxvol import (
                    prune_training_set_maxvol,
                    read_trajectory,
                    write_trajectory,
                )

                # 读取 nep.txt 获取描述符维度
                with open(nep_for_check) as f:
                    _ = f.readline()  # 只需要验证文件可读，不使用内容
                    # 第三个参数通常是 basis_size，第四个是 l_max
                    # 描述符维度粗略估计：从 NEP 计算器获取
                    from pynep.calculate import NEP

                    temp_calc = NEP(str(nep_for_check))
                    # 计算一个小结构的描述符来获取维度
                    test_structures = read_trajectory(str(train_file))
                    if len(test_structures) > 0:
                        test_desc = temp_calc.get_property(
                            "descriptor", test_structures[0]
                        )
                        descriptor_dim = test_desc.shape[1]
                        self.logger.info(f"  描述符维度: {descriptor_dim}")

                        # 计算最大允许的结构数
                        max_structures = int(
                            descriptor_dim * self.config.nep.max_structures_factor
                        )
                        self.logger.info(
                            f"  最大结构数: {max_structures} "
                            f"(维度 {descriptor_dim} × {self.config.nep.max_structures_factor})"
                        )

                        # 读取训练集
                        train_structures = read_trajectory(str(train_file))
                        self.logger.info(f"  当前训练集大小: {len(train_structures)}")

                        if len(train_structures) > max_structures:
                            # 执行修剪
                            pruned_structures = prune_training_set_maxvol(
                                structures=train_structures,
                                nep_file=str(nep_for_check),
                                max_structures=max_structures,
                                show_progress=False,
                            )

                            # 保存修剪后的训练集
                            write_trajectory(pruned_structures, str(train_xyz_dst))
                            self.logger.info(
                                f"  ✓ 训练集已修剪: {len(train_structures)} → {len(pruned_structures)}"
                            )
                        else:
                            # 不需要修剪，直接复制
                            shutil.copy2(train_file, train_xyz_dst)
                            self.logger.info("  训练集大小适中，无需修剪")
                    else:
                        # 训练集为空
                        shutil.copy2(train_file, train_xyz_dst)

            except Exception as e:  # noqa: F841
                self.logger.warning("  训练集修剪失败: {e}")
                self.logger.warning("  回退到直接复制模式")
                shutil.copy2(train_file, train_xyz_dst)
        else:
            # 未启用修剪，直接复制
            shutil.copy2(train_file, train_xyz_dst)

        # 复制 nep.txt 和 nep.restart（用于继续训练）
        # iter_1: 从用户提供的初始文件
        # iter_2+: 从上一轮的训练结果
        if iter_num == 1:
            # 第一轮：从配置文件获取初始文件
            nep_src = Path(self.config.global_config.initial_nep_model)
            restart_src = Path(self.config.global_config.initial_nep_restart)

            if nep_src.exists():
                shutil.copy2(nep_src, nep_dir / "nep.txt")
                self.logger.info("  复制初始 nep.txt")
            else:
                self.logger.error(f"  初始 nep.txt 不存在: {nep_src}")
                return False

            if restart_src.exists():
                shutil.copy2(restart_src, nep_dir / "nep.restart")
                self.logger.info("  复制初始 nep.restart")
            else:
                self.logger.error(f"  初始 nep.restart 不存在: {restart_src}")
                return False

        else:
            # 后续轮次：从上一轮复制
            prev_iter_dir = self.work_dir / f"iter_{iter_num - 1}"

            nep_src = prev_iter_dir / "nep.txt"
            if nep_src.exists():
                shutil.copy2(nep_src, nep_dir / "nep.txt")
                self.logger.info("  复制上一轮的 nep.txt")
            else:
                self.logger.error(f"  上一轮的 nep.txt 不存在: {nep_src}")
                return False

            restart_src = prev_iter_dir / "nep.restart"
            if restart_src.exists():
                shutil.copy2(restart_src, nep_dir / "nep.restart")
                self.logger.info("  复制上一轮的 nep.restart")
            else:
                self.logger.warning(f"  上一轮的 nep.restart 不存在: {restart_src}")
                # nep.restart 不存在不算错误，可能是第一次训练

        # 写入 nep.in
        with open(nep_dir / "nep.in", "w") as f:
            f.write(self.config.nep.input_content)

        # 写入作业脚本（自动添加 DONE 标记）
        with open(nep_dir / "job.sh", "w") as f:
            f.write(_ensure_done_marker(self.config.nep.job_script))

        self.logger.info(f"NEP 训练目录: {nep_dir}")

        # 提交作业
        if not self.task_manager.submit_job(nep_dir):
            return False

        # 等待完成
        if not self.task_manager.wait_for_completion(
            [nep_dir], timeout=self.config.nep.timeout
        ):
            return False

        # 复制训练结果到迭代目录
        nep_txt = nep_dir / "nep.txt"
        nep_restart = nep_dir / "nep.restart"

        if nep_txt.exists():
            shutil.copy2(nep_txt, iter_dir / "nep.txt")
            self.logger.info("  复制训练后的 nep.txt")
        else:
            self.logger.error("NEP 训练失败：未生成 nep.txt")
            return False

        if nep_restart.exists():
            shutil.copy2(nep_restart, iter_dir / "nep.restart")
            self.logger.info("  复制训练后的 nep.restart")
        else:
            self.logger.warning("未生成 nep.restart（可能训练未收敛或不需要）")

        self.logger.info("NEP 训练完成")
        return True

    def update_active_set(self, iter_num: int) -> bool:
        """
        更新活跃集

        参数:
            iter_num: 当前迭代编号

        返回:
            是否成功
        """
        self.logger.info("=" * 80)
        self.logger.info(f"步骤 5: 更新活跃集（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        train_file = iter_dir / "train.xyz"
        nep_file = iter_dir / "nep.txt"

        # 读取训练集
        train_structures = read_trajectory(str(train_file))
        self.logger.info(f"训练集包含 {len(train_structures)} 个结构")

        # 生成活跃集
        try:
            active_set_result, selected_structures = select_active_set(
                trajectory=train_structures,
                nep_file=str(nep_file),
                gamma_tol=self.config.selection.gamma_tol,
                batch_size=self.config.selection.batch_size,
            )

            # 统计
            total_envs = sum(
                len(inv) for inv in active_set_result.inverse_dict.values()
            )
            self.logger.info(f"活跃环境总数: {total_envs}")
            for element, inv_matrix in active_set_result.inverse_dict.items():
                self.logger.info(f"  元素 {element}: {len(inv_matrix)} 个活跃环境")

            # 保存活跃集
            asi_file = iter_dir / "active_set.asi"
            write_asi_file(active_set_result.inverse_dict, str(asi_file))
            self.logger.info(f"保存活跃集文件: {asi_file}")

            return True

        except Exception as e:
            self.logger.error(f"活跃集生成失败: {e}")
            return False

    def prepare_next_gpumd(self, iter_num: int) -> bool:
        """
        准备下一轮 GPUMD 探索

        参数:
            iter_num: 当前迭代编号

        返回:
            是否成功
        """
        self.logger.info("=" * 80)
        self.logger.info(f"步骤 6: 准备下一轮 GPUMD 探索（迭代 {iter_num + 1}）")
        self.logger.info("=" * 80)

        curr_iter_dir = self.work_dir / f"iter_{iter_num}"
        next_iter_dir = self.work_dir / f"iter_{iter_num + 1}"
        next_iter_dir.mkdir(parents=True, exist_ok=True)

        # 复制文件到下一轮
        shutil.copy2(curr_iter_dir / "train.xyz", next_iter_dir / "train.xyz")
        shutil.copy2(curr_iter_dir / "nep.txt", next_iter_dir / "nep.txt")
        shutil.copy2(curr_iter_dir / "active_set.asi", next_iter_dir / "active_set.asi")

        # 复制 nep.restart（如果存在）
        nep_restart = curr_iter_dir / "nep.restart"
        if nep_restart.exists():
            shutil.copy2(nep_restart, next_iter_dir / "nep.restart")
            self.logger.info("  复制 nep.restart 到下一轮")

        # 创建 GPUMD 目录
        next_gpumd_dir = next_iter_dir / "gpumd"
        next_gpumd_dir.mkdir(parents=True, exist_ok=True)

        # 为每个条件创建目录
        for cond in self.config.gpumd.conditions:
            cond_dir = next_gpumd_dir / cond.id
            cond_dir.mkdir(parents=True, exist_ok=True)

            # 复制结构文件
            structure_dst = cond_dir / "model.xyz"
            shutil.copy2(cond.structure_file, structure_dst)

            # 复制 NEP 和活跃集
            shutil.copy2(next_iter_dir / "nep.txt", cond_dir / "nep.txt")
            shutil.copy2(next_iter_dir / "active_set.asi", cond_dir / "active_set.asi")

            # 写入 run.in
            with open(cond_dir / "run.in", "w") as f:
                f.write(cond.run_in_content)

            # 写入作业脚本（自动添加 DONE 标记）
            with open(cond_dir / "job.sh", "w") as f:
                f.write(_ensure_done_marker(self.config.gpumd.job_script))

        self.logger.info(f"准备完成: {next_gpumd_dir}")
        return True

    def run_iteration(self, iter_num: int) -> bool:
        """
        运行一次完整迭代

        参数:
            iter_num: 迭代编号

        返回:
            是否继续（True=继续，False=收敛或失败）
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"开始迭代 {iter_num}")
        self.logger.info("=" * 80)

        # 步骤 1: GPUMD 探索
        if not self.run_gpumd(iter_num):
            self.logger.error("GPUMD 探索失败")
            return False

        # 步骤 2: 结构筛选
        selected = self.select_structures(iter_num)

        # 检查是否收敛
        if len(selected) == 0:
            self.logger.info("\n" + "=" * 80)
            self.logger.info("未选中新结构 - 训练已收敛！")
            self.logger.info("=" * 80)
            return False

        # 保存待标注结构
        iter_dir = self.work_dir / f"iter_{iter_num}"
        to_add_file = iter_dir / "to_add.xyz"
        write_trajectory(selected, str(to_add_file))
        self.logger.info(f"保存待标注结构: {to_add_file}")

        # 步骤 3: VASP DFT 标注
        if not self.run_vasp(iter_num, selected):
            self.logger.error("VASP 标注失败")
            return False

        # 步骤 4: NEP 训练
        if not self.run_nep(iter_num):
            self.logger.error("NEP 训练失败")
            return False

        # 步骤 5: 更新活跃集
        if not self.update_active_set(iter_num):
            self.logger.error("活跃集更新失败")
            return False

        # 步骤 6: 准备下一轮
        if not self.prepare_next_gpumd(iter_num):
            self.logger.error("准备下一轮失败")
            return False

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"迭代 {iter_num} 完成")
        self.logger.info("=" * 80)

        return True
