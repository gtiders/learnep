"""
JAXVol 命令行工具入口。
提供 active (主动选择), extend (数据集扩展), gamma (外推计算) 等子命令。

功能说明与示例：

1. active (主动选择):
   从大规模训练集中或者候选集中，选出最具代表性的 Active Set（基底）。
   支持从零开始（亏秩状态）自动构建，自动处理从 Pivoted QR 到 MaxVol 的过渡。
   示例：
     python -m jaxvol.cli active --train-path all_structures.xyz --output active_set.xyz

2. extend (数据集扩展):
   在现有训练集基础上，从候选集中挑选出能最大程度增加信息量的新构型。
   通常用于 Active Learning 循环中，从新采样的 MD 轨迹中挑选结构加入训练。
   示例：
     python -m jaxvol.cli extend --base-path current_train.xyz --candidate-path new_candidates.xyz --output to_add.xyz

3. gamma (外推计算):
   计算给定轨迹中每个构型的外推等级 (Gamma)，并筛选出高风险/高新颖性构型。
   需配合已生成的 active_set.asi 文件使用。
   支持自动截断（上限爆炸检测）和自动终止（QR 模式下分数稳定停止）。
   示例：
     python -m jaxvol.cli gamma --input md_trajectory.xyz --threshold 1.0 --output high_gamma.xyz
"""
import typer
import jax
import os
from typing import Optional
from typing_extensions import Annotated

app = typer.Typer(help="JAXVol: Maximally Adaptive Volume Selection Tools")

def configure_jax(device: Optional[str]):
    # Note: jax_enable_x64 is now set in __init__.py globally.
    if device:
        print(f"Setting JAX Platform to: {device}")
        jax.config.update("jax_platform_name", device)

@app.command()
def active(
    train_path: Annotated[str, typer.Option(help="Path to training data")] = "train.xyz",
    nep_path: Annotated[str, typer.Option(help="Path to nep.txt")] = "nep.txt",
    output: Annotated[str, typer.Option(help="Output XYZ file")] = "active_set.xyz",
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 10000,
    mode: Annotated[str, typer.Option(help="Selection mode (adaptive/maxvol)")] = "adaptive",
    device: Annotated[Optional[str], typer.Option(help="Force JAX device (cpu/gpu)")] = None,
    gamma_min: Annotated[Optional[float], typer.Option(help="Min gamma threshold (Deprecated in adaptive mode)")] = None,
    gamma_max: Annotated[Optional[float], typer.Option(help="Max gamma threshold (Deprecated in adaptive mode)")] = None,
):
    """
    Select active set from training data.
    """
    print("--- Select Active Set ---")
    configure_jax(device)
    
    from .tools import get_B_projections, get_active_set
    from ase.io import read, write
    try:
        from pynep.io import load_nep, dump_nep
    except ImportError:
        print("PyNEP not found. Using ASE for IO only.")
        load_nep = read
        dump_nep = write
    
    print(f"Loading data from {train_path}...")
    try:
        traj = load_nep(train_path)
    except Exception as e:
        print(f"Failed to load with load_nep, trying ASE read: {e}")
        traj = read(train_path, index=":")

    B_projections, B_indices = get_B_projections(traj, nep_path)
    # Note: We keep gamma_min/max args in function signature for interface compatibility if needed,
    # but currently tools.get_active_set doesn't use them (as reverted). 
    # Passing them would fail if signature doesn't match. 
    # tools.get_active_set signature is: (B_projections, ..., mode)
    
    _, active_structs = get_active_set(
        B_projections, 
        B_indices, 
        write_asi=True, 
        batch_size=batch_size,
        mode=mode
    )
    
    print(f"Saving {len(active_structs)} structures to {output}...")
    out_traj = [traj[i] for i in active_structs]
    try:
        dump_nep(output, out_traj)
    except:
        write(output, out_traj)

@app.command()
def extend(
    base_path: Annotated[str, typer.Option(help="Current training set")] = "train.xyz",
    candidate_path: Annotated[str, typer.Option(help="Candidates")] = "large_gamma.xyz",
    nep_path: Annotated[str, typer.Option(help="Path to nep.txt")] = "nep.txt",
    output: Annotated[str, typer.Option(help="Output file")] = "to_add.xyz",
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 10000,
    mode: Annotated[str, typer.Option(help="Selection mode (adaptive/maxvol)")] = "adaptive",
    device: Annotated[Optional[str], typer.Option(help="Force JAX device (cpu/gpu)")] = None,
):
    """
    Extend training set with new candidates.
    """
    print("--- Extend Training Set ---")
    configure_jax(device)
    
    from .tools import get_B_projections, get_active_set
    from ase.io import read, write
    try:
        from pynep.io import load_nep, dump_nep
    except ImportError:
        load_nep = read
        dump_nep = write

    print("Loading datasets...")
    try:
        data1 = load_nep(base_path)
    except:
        data1 = read(base_path, index=":")
        
    try:
        data2 = load_nep(candidate_path)
    except:
        data2 = read(candidate_path, index=":")
        
    data = data1 + data2
    n_base = len(data1)
    
    print(f"Base size: {n_base}, Candidate size: {len(data2)}")
    
    B_projections, B_indices = get_B_projections(data, nep_path)
    
    _, active_structs = get_active_set(
        B_projections, 
        B_indices, 
        write_asi=False,
        batch_size=batch_size,
        mode=mode
    )
    
    new_indices = [i for i in active_structs if i >= n_base]
    print(f"Selected {len(new_indices)} new structures from candidates.")
    
    out = [data[i] for i in new_indices]
    try:
        dump_nep(output, out)
    except:
        write(output, out)

@app.command()
def gamma(
    input: Annotated[str, typer.Option(help="Input trajectory")] = "to_select.xyz",
    nep_path: Annotated[str, typer.Option(help="Path to nep.txt")] = "nep.txt",
    asi_path: Annotated[str, typer.Option(help="Path to active set inverse")] = "active_set.asi",
    output: Annotated[str, typer.Option(help="Output file")] = "large_gamma.xyz",
    threshold: Annotated[float, typer.Option(help="Min Gamma threshold for selection")] = 1.0,
    threshold_max: Annotated[Optional[float], typer.Option(help="Max Gamma cutoff (Stops scanning if exceeded)")] = None,
    auto_stop: Annotated[bool, typer.Option(help="Enable statistical auto-stopping in QR mode")] = False,
    std_tol: Annotated[float, typer.Option(help="Std Dev tolerance for auto-stop")] = 1e-4,
    device: Annotated[Optional[str], typer.Option(help="Force JAX device (cpu/gpu)")] = None,
):
    """
    Calculate gamma and scan trajectory.
    """
    print("--- Calculate Gamma & Scan Trajectory ---")
    configure_jax(device)
    
    from .tools import scan_trajectory_gamma
    from ase.io import read, write
    try:
        from pynep.io import load_nep, dump_nep
    except ImportError:
        load_nep = read
        dump_nep = write
        
    print(f"Loading trajectory from {input}...")
    try:
        traj = load_nep(input)
    except:
        traj = read(input, index=":")
        
    out_traj = scan_trajectory_gamma(
        traj, 
        nep_file=nep_path, 
        asi_file=asi_path,
        gamma_min=threshold,
        gamma_max=threshold_max,
        auto_stop_qr=auto_stop,
        std_tol=std_tol
    )
    
    print(f"Selected {len(out_traj)} structures fitting steps.")
    
    if output:
        try:
            dump_nep(output, out_traj)
        except:
            write(output, out_traj)

if __name__ == "__main__":
    app()
