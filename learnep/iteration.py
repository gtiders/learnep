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
        self.scheduler_type = config.global_config.scheduler_type.lower()
        self.job_status_command = config.global_config.job_status_command

        # 存储作业目录到任务号的映射
        self.job_ids: dict[Path, str] = {}

    def _parse_job_id(self, output: str) -> Optional[str]:
        """
        从提交命令的输出中解析任务号

        参数:
            output: 提交命令的标准输出

        返回:
            解析出的任务号，失败返回 None
        """
        output = output.strip()
        if not output:
            return None

        scheduler = self.scheduler_type

        if scheduler == "pbs":
            # PBS 格式: "12345.pbs01" 或直接 "12345"
            # 直接返回整行（去除空白）
            return output.split("\n")[0].strip()

        elif scheduler == "slurm":
            # SLURM 格式: "Submitted batch job 12345"
            import re

            match = re.search(r"Submitted batch job (\d+)", output)
            if match:
                return match.group(1)
            # 也可能直接返回数字
            if output.isdigit():
                return output

        elif scheduler == "lsf":
            # LSF 格式: "Job <12345> is submitted to queue <normal>"
            import re

            match = re.search(r"Job <(\d+)>", output)
            if match:
                return match.group(1)

        elif scheduler == "auto":
            # 自动检测：尝试各种格式
            import re

            # 尝试 SLURM 格式
            match = re.search(r"Submitted batch job (\d+)", output)
            if match:
                return match.group(1)

            # 尝试 LSF 格式
            match = re.search(r"Job <(\d+)>", output)
            if match:
                return match.group(1)

            # 尝试 PBS 格式（直接返回第一行）
            first_line = output.split("\n")[0].strip()
            if first_line:
                return first_line

        # 默认：返回第一行
        return output.split("\n")[0].strip()

    def _is_job_running(self, job_id: str) -> bool:
        """
        检查任务是否还在运行

        参数:
            job_id: 任务号

        返回:
            True 表示任务还在运行，False 表示任务已结束
        """
        try:
            # 构建查询命令
            cmd = self.job_status_command.format(job_id=job_id)

            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

            # 不同调度系统的判断逻辑
            if self.scheduler_type == "pbs":
                # PBS: 如果任务不存在，qstat 返回非零退出码
                # 或者输出包含 "Unknown Job Id"
                if result.returncode != 0:
                    return False
                if "Unknown Job Id" in result.stderr:
                    return False
                return True

            elif self.scheduler_type == "slurm":
                # SLURM: 任务不存在时 squeue 返回空或错误
                if result.returncode != 0:
                    return False
                # 检查输出是否包含任务号
                return job_id in result.stdout

            elif self.scheduler_type == "lsf":
                # LSF: 任务不存在时 bjobs 返回错误
                if result.returncode != 0:
                    return False
                return job_id in result.stdout

            else:
                # auto 或其他：检查退出码
                return result.returncode == 0

        except subprocess.TimeoutExpired:
            self.logger.warning(f"  检查任务状态超时: {job_id}")
            return True  # 超时时假设任务还在运行
        except Exception as e:
            self.logger.warning(f"  检查任务状态失败: {job_id}, 错误: {e}")
            return True  # 出错时假设任务还在运行

    def submit_job(self, job_dir: Path) -> Optional[str]:
        """
        在指定目录提交作业

        参数:
            job_dir: 作业目录

        返回:
            任务号，提交失败返回 None
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
                job_id = self._parse_job_id(result.stdout)
                if job_id:
                    self.job_ids[job_dir] = job_id
                    self.logger.info(f"  作业已提交: {job_dir.name} (任务号: {job_id})")
                else:
                    self.logger.info(f"  作业已提交: {job_dir.name} (无法解析任务号)")
                return job_id
            else:
                self.logger.error(f"  作业提交失败: {job_dir}")
                self.logger.error(f"    错误: {result.stderr}")
                return None

        except Exception as e:
            self.logger.error(f"  提交作业时发生异常: {e}")
            return None

    def wait_for_completion(
        self, job_dirs: List[Path], timeout: Optional[int] = None
    ) -> bool:
        """
        等待所有作业完成

        使用双重检测机制：
        1. 检测 DONE 文件是否存在
        2. 检测任务号是否还在队列中

        任一条件满足即视为任务完成。

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

            # 检查每个作业
            completed = []
            for job_dir in pending_jobs:
                # 检测方式1: DONE 文件
                done_file = job_dir / "DONE"
                if done_file.exists():
                    completed.append(job_dir)
                    self.logger.info(f"  作业完成 (DONE文件): {job_dir.name}")
                    continue

                # 检测方式2: 任务号状态
                job_id = self.job_ids.get(job_dir)
                if job_id and not self._is_job_running(job_id):
                    completed.append(job_dir)
                    self.logger.info(
                        f"  作业完成 (任务结束): {job_dir.name} ({job_id})"
                    )
                    continue

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
        self._bootstrap_mode = False  # 是否处于冷启动模式

    def is_bootstrap_mode(self, iter_dir: Path) -> bool:
        """
        检测是否需要使用冷启动模式。

        条件：训练数据不足以运行 MaxVol 算法。

        参数:
            iter_dir: 当前迭代目录

        返回:
            是否需要冷启动模式
        """
        if not self.config.bootstrap.enabled:
            return False

        train_file = iter_dir / "train.xyz"
        nep_file = iter_dir / "nep.txt"

        if not train_file.exists() or not nep_file.exists():
            return True  # 文件不存在，需要冷启动

        try:
            from .maxvol import check_data_sufficient, read_trajectory

            train_structures = read_trajectory(str(train_file))
            is_sufficient, stats = check_data_sufficient(
                train_structures, str(nep_file)
            )

            if not is_sufficient:
                self.logger.info("\n检测到训练数据不足，将使用冷启动模式")
                for elem, (count, dim) in stats.items():
                    status = "✓" if count >= dim else "✗"
                    self.logger.info(f"  {status} 元素 {elem}: {count}/{dim} 原子环境")
                return True

            return False

        except Exception as e:
            self.logger.warning(f"检测数据充足性时出错: {e}")
            return True  # 出错时假设需要冷启动

    def validate_bootstrap_config(self) -> bool:
        """
        验证冷启动配置是否有效。

        检查：
        1. 是否有冷启动条件
        2. 每个条件的 run.in 是否包含 dump_exyz

        返回:
            配置是否有效
        """
        if not self.config.bootstrap.conditions:
            self.logger.error("冷启动配置中没有定义任何条件 (bootstrap.conditions)")
            return False

        for cond in self.config.bootstrap.conditions:
            run_in = cond.run_in_content

            # 检查 dump_exyz
            if "dump_exyz" not in run_in:
                self.logger.error(
                    f"冷启动条件 '{cond.id}' 的 run.in 中必须包含 dump_exyz 命令\n"
                    f"当前 run.in:\n{run_in}\n"
                    f"请添加类似: dump_exyz 100 1 1"
                )
                return False

            # 警告：不应使用 compute_extrapolation
            if "compute_extrapolation" in run_in:
                self.logger.warning(
                    f"冷启动条件 '{cond.id}' 使用了 compute_extrapolation，"
                    f"冷启动模式下建议仅使用 dump_exyz"
                )

        return True

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
            if self.task_manager.submit_job(job_dir) is None:
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

    def run_bootstrap_gpumd(self, iter_num: int) -> bool:
        """
        运行冷启动模式的 GPUMD 探索。

        与正常模式不同，冷启动模式：
        - 使用单独的冷启动配置
        - 使用 dump_exyz 保存轨迹而不是 compute_extrapolation
        - 不计算 gamma 值

        参数:
            iter_num: 当前迭代编号

        返回:
            是否成功
        """
        self.logger.info("=" * 80)
        self.logger.info(f"冷启动模式：GPUMD 探索（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        bootstrap_dir = iter_dir / "bootstrap_gpumd"

        # 验证冷启动配置
        if not self.validate_bootstrap_config():
            return False

        # 创建目录
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        # 为每个条件创建目录
        job_dirs = []
        for cond in self.config.bootstrap.conditions:
            cond_dir = bootstrap_dir / cond.id
            cond_dir.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"  创建冷启动条件目录: {cond.id}")

            # 复制结构文件
            structure_dst = cond_dir / "model.xyz"
            shutil.copy2(cond.structure_file, structure_dst)

            # 复制 NEP 模型
            nep_src = iter_dir / "nep.txt"
            if nep_src.exists():
                shutil.copy2(nep_src, cond_dir / "nep.txt")
            else:
                # 从初始模型复制
                shutil.copy2(
                    self.config.global_config.initial_nep_model, cond_dir / "nep.txt"
                )

            # 写入 run.in
            with open(cond_dir / "run.in", "w") as f:
                f.write(cond.run_in_content)

            # 写入作业脚本
            with open(cond_dir / "job.sh", "w") as f:
                f.write(_ensure_done_marker(self.config.bootstrap.job_script))

            job_dirs.append(cond_dir)

        self.logger.info(f"创建了 {len(job_dirs)} 个冷启动 GPUMD 任务")

        # 提交所有作业
        self.logger.info("\n提交冷启动 GPUMD 作业...")
        for job_dir in job_dirs:
            if self.task_manager.submit_job(job_dir) is None:
                return False

        # 等待完成
        if not self.task_manager.wait_for_completion(
            job_dirs, timeout=self.config.bootstrap.timeout
        ):
            return False

        # 合并所有 dump.xyz
        self.logger.info("\n收集冷启动轨迹...")
        all_structures = []
        for job_dir in job_dirs:
            dump_file = job_dir / "dump.xyz"
            if dump_file.exists():
                try:
                    structures = read_trajectory(str(dump_file))
                    all_structures.extend(structures)
                    self.logger.info(f"  {job_dir.name}: {len(structures)} 个结构")
                except Exception as e:
                    self.logger.warning(f"  读取 {dump_file} 失败: {e}")

        self.logger.info(f"总共收集到 {len(all_structures)} 个冷启动结构")

        # 保存到 bootstrap_dump.xyz
        bootstrap_dump_file = iter_dir / "bootstrap_dump.xyz"
        if all_structures:
            write_trajectory(all_structures, str(bootstrap_dump_file))
            self.logger.info(f"保存到: {bootstrap_dump_file}")
        else:
            bootstrap_dump_file.touch()
            self.logger.warning("未收集到任何结构")

        return True

    def select_bootstrap_structures(self, iter_num: int) -> List[Atoms]:
        """
        冷启动模式的结构选择。

        流程：
        1. 读取 bootstrap_dump.xyz
        2. 过滤不合理的结构
        3. 使用 FPS 选择多样性结构

        参数:
            iter_num: 当前迭代编号

        返回:
            选中的结构列表
        """
        self.logger.info("=" * 80)
        self.logger.info(f"冷启动模式：结构选择（迭代 {iter_num}）")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        bootstrap_dump_file = iter_dir / "bootstrap_dump.xyz"
        nep_file = iter_dir / "nep.txt"

        # NEP 文件可能不在 iter_dir，尝试使用初始模型
        if not nep_file.exists():
            nep_file = Path(self.config.global_config.initial_nep_model)

        # 检查输入文件
        if not bootstrap_dump_file.exists():
            self.logger.error(f"bootstrap_dump.xyz 不存在: {bootstrap_dump_file}")
            return []

        if bootstrap_dump_file.stat().st_size == 0:
            self.logger.info("bootstrap_dump.xyz 为空")
            return []

        # 读取轨迹
        all_structures = read_trajectory(str(bootstrap_dump_file))
        self.logger.info(f"读取到 {len(all_structures)} 个结构")

        # 过滤不合理的结构
        self.logger.info("\n过滤不合理的结构...")
        from .maxvol import filter_reasonable_structures

        reasonable = filter_reasonable_structures(
            structures=all_structures,
            nep_file=str(nep_file),
            min_distance=self.config.bootstrap.filter.min_distance,
            max_force=self.config.bootstrap.filter.max_force,
            max_energy_deviation=self.config.bootstrap.filter.max_energy_per_atom,
            show_progress=False,
        )
        self.logger.info(f"过滤后剩余 {len(reasonable)} 个合理结构")

        if len(reasonable) == 0:
            self.logger.warning("过滤后没有合理结构，请检查过滤参数")
            return []

        # FPS 选择（如果超过上限）
        max_structures = self.config.global_config.max_structures_per_iteration

        if len(reasonable) <= max_structures:
            self.logger.info(
                f"合理结构数 ({len(reasonable)}) <= 上限 ({max_structures})，无需 FPS 筛选"
            )
            return reasonable

        self.logger.info(
            f"\nFPS 筛选（{len(reasonable)} → 最多 {max_structures} 个）..."
        )

        from .maxvol import apply_fps_filter

        selected = apply_fps_filter(
            structures=reasonable,
            nep_file=str(nep_file),
            max_count=max_structures,
            initial_min_distance=self.config.selection.fps_min_distance,
            show_progress=False,
        )
        self.logger.info(f"FPS 选中 {len(selected)} 个结构")

        return selected

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

        if method == "maxvol_fps":
            # MaxVol + FPS 模式：先 MaxVol 选择代表性结构，再用 FPS 二次筛选
            self.logger.info("\n使用 MaxVol + FPS 模式...")

            from .maxvol import select_extension_structures, apply_fps_filter

            # 第一步：MaxVol 选择
            self.logger.info("第一步: MaxVol 选择代表性结构...")
            selected = select_extension_structures(
                train_trajectory=train_structures,
                candidate_trajectory=candidate_structures,
                nep_file=str(nep_file),
                gamma_tol=self.config.selection.gamma_tol,
                batch_size=self.config.selection.batch_size,
            )
            self.logger.info(f"MaxVol 选中 {len(selected)} 个结构")

            # 第二步：FPS 筛选（如果超过限制）
            if len(selected) > max_structures:
                self.logger.info(f"第二步: FPS 筛选到 {max_structures} 个结构...")
                selected = apply_fps_filter(
                    structures=selected,
                    nep_file=str(nep_file),
                    max_count=max_structures,
                    initial_min_distance=self.config.selection.fps_min_distance,
                    show_progress=False,
                )
                self.logger.info(f"FPS 筛选后: {len(selected)} 个结构")

        else:
            # MaxVol 模式（默认）：直接选择最有价值的结构
            self.logger.info(f"\n使用 MaxVol 模式（上限: {max_structures} 个结构）...")

            from .maxvol import select_structures_maxvol

            selected = select_structures_maxvol(
                train_structures=train_structures,
                candidate_structures=candidate_structures,
                nep_file=str(nep_file),
                max_structures=max_structures,
                gamma_tol=self.config.selection.gamma_tol,
                batch_size=self.config.selection.batch_size,
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
            if self.task_manager.submit_job(job_dir) is None:
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
        if self.task_manager.submit_job(nep_dir) is None:
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

        自动检测是否需要使用冷启动模式：
        - 如果训练数据充足，使用正常模式（MaxVol + gamma 筛选）
        - 如果训练数据不足，使用冷启动模式（dump_exyz + FPS）

        参数:
            iter_num: 迭代编号

        返回:
            是否继续（True=继续，False=收敛或失败）
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"开始迭代 {iter_num}")
        self.logger.info("=" * 80)

        iter_dir = self.work_dir / f"iter_{iter_num}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        # 检测是否需要冷启动模式
        bootstrap_mode = self.is_bootstrap_mode(iter_dir)

        if bootstrap_mode:
            self.logger.info("\n*** 使用冷启动模式 ***\n")
            return self._run_bootstrap_iteration(iter_num)
        else:
            self.logger.info("\n*** 使用正常模式 ***\n")
            return self._run_normal_iteration(iter_num)

    def _run_normal_iteration(self, iter_num: int) -> bool:
        """运行正常模式迭代"""
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
        self.logger.info(f"迭代 {iter_num} 完成（正常模式）")
        self.logger.info("=" * 80)

        return True

    def _run_bootstrap_iteration(self, iter_num: int) -> bool:
        """运行冷启动模式迭代"""
        iter_dir = self.work_dir / f"iter_{iter_num}"

        # 确保有初始文件
        if iter_num == 1:
            # 复制初始 NEP 模型和训练数据
            nep_src = Path(self.config.global_config.initial_nep_model)
            train_src = Path(self.config.global_config.initial_train_data)

            if nep_src.exists():
                shutil.copy2(nep_src, iter_dir / "nep.txt")
                self.logger.info(f"复制初始 NEP 模型: {nep_src.name}")

            if train_src.exists():
                shutil.copy2(train_src, iter_dir / "train.xyz")
                self.logger.info(f"复制初始训练数据: {train_src.name}")

        # 步骤 1: 冷启动 GPUMD 探索
        if not self.run_bootstrap_gpumd(iter_num):
            self.logger.error("冷启动 GPUMD 探索失败")
            return False

        # 步骤 2: 冷启动结构选择
        selected = self.select_bootstrap_structures(iter_num)

        if len(selected) == 0:
            self.logger.warning("冷启动未选中任何结构，请检查配置")
            return False

        # 保存待标注结构
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

        # 步骤 5: 检查是否可以切换到正常模式
        nep_file = iter_dir / "nep.txt"
        train_file = iter_dir / "train.xyz"

        if nep_file.exists() and train_file.exists():
            from .maxvol import check_data_sufficient, read_trajectory as read_traj

            train_structures = read_traj(str(train_file))
            is_sufficient, stats = check_data_sufficient(
                train_structures, str(nep_file)
            )

            if is_sufficient:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("训练数据已充足，下一轮将切换到正常模式！")
                self.logger.info("=" * 80)

                # 生成活跃集
                if not self.update_active_set(iter_num):
                    self.logger.warning("活跃集更新失败，下一轮可能仍使用冷启动模式")
            else:
                self.logger.info("\n数据仍不足，下一轮继续使用冷启动模式")
                for elem, (count, dim) in stats.items():
                    status = "✓" if count >= dim else "✗"
                    self.logger.info(f"  {status} 元素 {elem}: {count}/{dim} 原子环境")

        # 步骤 6: 准备下一轮
        next_iter_dir = self.work_dir / f"iter_{iter_num + 1}"
        next_iter_dir.mkdir(parents=True, exist_ok=True)

        # 复制文件到下一轮
        if (iter_dir / "train.xyz").exists():
            shutil.copy2(iter_dir / "train.xyz", next_iter_dir / "train.xyz")
        if (iter_dir / "nep.txt").exists():
            shutil.copy2(iter_dir / "nep.txt", next_iter_dir / "nep.txt")
        if (iter_dir / "nep.restart").exists():
            shutil.copy2(iter_dir / "nep.restart", next_iter_dir / "nep.restart")
        if (iter_dir / "active_set.asi").exists():
            shutil.copy2(iter_dir / "active_set.asi", next_iter_dir / "active_set.asi")

        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"迭代 {iter_num} 完成（冷启动模式）")
        self.logger.info("=" * 80)

        return True
