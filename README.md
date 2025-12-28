# LearnEP - NEP 主动学习框架

**LearnEP** 是一个专为 NEP (Neuroevolution Potential) 势函数设计的高级主动学习（Active Learning）自动化框架。它深度集成了 `jaxvol` 库，利用自适应采样技术（Adaptive Sampling）和 MaxVol 算法，实现高效、鲁棒的训练-探索-筛选循环。

---

## 目录

1. [项目特点](#1-项目特点)
2. [快速开始](#2-快速开始)
3. [核心功能详解](#3-核心功能详解)
   - [断点续传与重启](#31-断点续传与重启---restart-from)
   - [冷启动与热启动机制](#32-冷启动与热启动机制)
   - [迭代精细控制](#33-迭代精细控制-iteration_control)
4. [配置指南](#4-配置指南)
5. [工作流全解](#5-工作流全解)
6. [文件与数据流](#6-文件与数据流)
7. [注意事项](#7-注意事项)
8. [测试与验证](#8-测试与验证)

---

## 1. 项目特点

*   **全自动化流程**: 自动管理从训练、MD探索、结构筛选、VASP计算到数据更新的全过程。
*   **JaxVol 深度集成**: 使用 JAX 加速的自适应采样，自动处理从秩亏（Rank-Deficient）到满秩（Full-Rank）的过渡，基于 Gamma 指标智能筛选高不确定性结构。
*   **鲁棒的作业调度**: 支持 PBS/Slurm 等调度器，具备“双重检查”（文件+队列状态）和超时自动取消机制，防止任务挂死。
*   **严格的数据流**: 采用 `next_iter` 机制，确保每一轮迭代的数据严格隔离且可追溯。

---

## 2. 快速开始

### 2.1 初始化
在工作目录下生成默认配置文件 `config.yaml`：
```bash
python -m learnep init --output config.yaml
```

### 2.2 运行
启动主动学习循环：
```bash
python -m learnep run config.yaml
```

### 2.3 查看状态
查看当前运行进度和最后一轮完成的迭代：
```bash
python -m learnep status
```

---

## 3. 核心功能详解

### 3.1 断点续传与重启 (`--restart-from`)
如果某轮迭代（例如第 5 轮）出现参数错误或需要调整策略，您可以使用重启功能：

```bash
python -m learnep run config.yaml --restart-from 5
```

**执行逻辑**:
1. **自动清理**: 系统会立即删除 `iter_005` 及其之后所有迭代（`iter_006`, ...）的文件夹，确保环境干净。
2. **状态重置**: 修改 `status.json`，将“最后完成迭代”重置为 4。
3. **重新开始**: 程序从第 5 轮重新开始运行。

### 3.2 冷启动与热启动机制
LearnEP 会根据 `iter_000` 初始文件的存在情况，智能决定启动模式：

*   **冷启动 (Cold Start)**:
    *   **触发条件**: 只有 `train.xyz`，没有模型文件。
    *   **行为**: 使用配置中的 `first_train_input` 参数从头开始训练 NEP 模型。
*   **热启动 (Hot Start)**:
    *   **触发条件**: 目录下同时存在 `nep.txt` (模型) 和 `nep.restart` (训练状态)。
    *   **行为**:
        *   **第 0 轮**: 直接使用已有的 `nep.txt` 进行 MD 探索，**跳过初始训练**，节省时间。
        *   **后续轮次**: 读取上一轮的模型和状态，进行增量微调（Fine-tuning）。
    *   **要求**: 必须严格同时提供这两个文件，缺一不可（否则视为无法继续训练，转为冷启动）。

### 3.3 迭代精细控制 (`iteration_control`)
您可以在 `config.yaml` 中为特定的迭代轮次覆盖全局设置。例如，在初始几轮使用较短的 MD 步数快速探索：

```yaml
iteration_control:
  enabled: true
  rules:
    - iterations: [0, 1, 2]  # 对第0,1,2轮生效
      gpumd:
        conditions:
          - id: "short_explore"
            run_in: "dump_exyz 100 0 0" # 跑得更短
```

---

## 4. 配置指南

配置文件 `config.yaml` 分为几个关键部分：

### Global (全局设置)
*   `work_dir`: 所有迭代文件夹生成的主目录。
*   `scheduler`: 作业调度器配置。
    *   `submit_cmd`: 提交命令 (如 `qsub {script}`)。
    *   `check_cmd`: 状态检查命令 (如 `qstat {job_id}`)。
    *   `cancel_cmd`: 取消命令 (如 `qdel {job_id}`)，用于超时清理。

### NEP (训练)
*   `train_input`: 标准训练参数（`nep.in` 内容）。
*   `first_train_input`: (可选) 仅用于冷启动第一轮的参数（通常步数更长）。
*   `job_script`: 训练任务的提交脚本模板。

### GPUMD (探索)
*   `conditions`: 定义多个探索条件（温度、压力等）。每个条件会生成一个独立的子任务。
*   `run_in`: GPUMD 的输入参数，必须包含 `dump_exyz` 以输出扩展 XYZ 格式（含力信息）。

### Selection (筛选 - JaxVol)
*   `mode`: `adaptive` (推荐) 或 `maxvol`.
*   `gamma`:
    *   `threshold`: Gamma 阈值，只有超过此值的结构才会被选中。
    *   `threshold_max`: 安全阈值，超过此值会触发警告或停止，防止非物理结构进入。
    *   `n_max_label`: 每轮最大送去 VASP 计算的结构数量（节省算力）。

### VASP (标号)
*   `input_files`: 需要复制到计算目录的文件 (`INCAR`, `POTCAR`, `KPOINTS`)。
*   `job_script`: VASP 计算脚本模板。
*   `timeout`: 超时时间（秒）。如果 VASP 任务卡住，超过此时间会强制取消并进入下一阶段。

---

## 5. 工作流全解

Orchestrator 严格按照以下顺序执行每一轮迭代（Step 0 到 Step N）：

1.  **准备阶段 (Prepare)**:
    *   创建 `iter_NNN` 目录。
    *   从上一轮的 `next_iter` 文件夹（或初始输入）复制 `nep.txt`, `nep.restart`, `train.xyz`, `active_set.asi`。
2.  **训练阶段 (Train)**:
    *   判断是否热启动。
    *   提交 NEP 训练任务。
    *   等待任务完成，生成新的 `nep.txt`。
3.  **探索阶段 (Explore)**:
    *   利用新模型并行运行多个 GPUMD MD 任务。
    *   生成轨迹文件 (`dump.xyz`)。
4.  **筛选阶段 (Selection)**:
    *   加载 `train.xyz` 构建/更新活跃集 (ASI)。
    *   使用 JaxVol 扫描 MD 轨迹，计算每个帧的 Gamma 值。
    *   筛选出高不确定性结构，去重并排序，保存为 `candidates.xyz`。
5.  **标号阶段 (Label)**:
    *   为筛选出的候选结构生成 VASP 任务 (`POSCAR`)。
    *   提交所有 DFT 计算任务。
    *   **超时监控**: 如果任务超时，自动 `kill` 并尝试收集已完成的部分。
6.  **更新阶段 (Update)**:
    *   收集 DFT 计算结果（能量、力、维里）。
    *   将新数据追加到 `train.xyz`。
    *   **准备下一轮**: 将最新的模型、数据和状态文件复制到 `next_iter` 子目录，供下一轮使用。

---

## 6. 文件与数据流

了解文件流向有助于排错。

| 阶段 | 输入文件 | 关键输出 | 备注 |
| :--- | :--- | :--- | :--- |
| **0. 初始** | 用户提供 | `train.xyz` (必须), `nep.txt` (可选) | 如有 `nep.txt` 则触发热启动 |
| **1. 训练** | `nep.in` | `nep.txt`, `nep.restart` | 模型和断点文件 |
| **2. 探索** | `run.in`, `nep.txt` | `dump.xyz` | 包含构型和力/能量预测 |
| **3. 筛选** | `dump.xyz`, `active_set.asi` | `candidates.xyz` | **关键产物**: 待计算结构 |
| **4. 标号** | `POSCAR` (from candidates) | `vasprun.xml` / `OUTCAR` | 真实 DFT 数据 |
| **5. 下一轮** | `train.xyz` (已合并新数据) | `iter_N/next_iter/` | 包含所有传递给 N+1 轮的文件 |

---

## 7. 注意事项

1.  **调度器命令**: 请务必根据您集群的实际情况（PBS, Slurm, LSF）修改 `config.yaml` 中的 `submit_cmd` 和 `check_cmd`。如果检查命令配置错误，程序可能会误判任务状态导致无限等待。
2.  **VASP 超时**: 建议设置合理的 `timeout`。如果 VASP 计算死循环，不仅浪费机时，还会阻塞整个主动学习流程。
3.  **结构合理性**: JaxVol 筛选出的结构可能是高能非物理结构。建议在 VASP 计算脚本中加入简单的预检查，或者在 `selection` 配置中调整 `gamma` 阈值防止选中极端离群点。
4.  **路径**: 所有路径建议使用**绝对路径**，避免因工作目录切换导致找不到文件。

---

## 8. 测试与验证

在正式运行前，强烈建议运行内置的 Mock 测试套件，它不需要真实的 VASP/GPU 环境即可验证流程逻辑。

```bash
# 运行完整测试
python tests/run_tests.py
```

测试包含：
*   **Cold Start**: 验证冷启动逻辑。
*   **Long Run**: 模拟 10 轮迭代，验证数据流稳定性。
*   **Restart**: 验证中断恢复和文件清理功能。

---

**LearnEP Team**
