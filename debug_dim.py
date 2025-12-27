from pynep.calculate import NEP
from ase.io import read
import numpy as np
import sys
from pathlib import Path


def check_descriptor_dim(nep_file, xyz_file):
    if not Path(nep_file).exists():
        print(f"Error: {nep_file} does not exist")
        return

    if not Path(xyz_file).exists():
        print(f"Error: {xyz_file} does not exist")
        return

    print(f"Loading NEP from {nep_file}...")
    calc = NEP(str(nep_file))

    print(f"Loading structure from {xyz_file}...")
    atoms = read(str(xyz_file), index=0)
    print(f"Structure has {len(atoms)} atoms")

    print("Calculating descriptor...")
    desc = calc.get_property("descriptor", atoms)
    print(f"Descriptor shape: {desc.shape}")
    print(f"Calculated dimension: {desc.shape[1]}")

    # 尝试读取 nep.txt 的第一行
    with open(nep_file) as f:
        line1 = f.readline().split()
        print(f"nep.txt header: {line1}")


if __name__ == "__main__":
    work_dir = Path("/cache/ybgao2024/KLiGeNEP")  # 从之前的日志中获取
    iter_1_dir = work_dir / "iter_1"

    nep_file = iter_1_dir / "nep.txt"
    xyz_file = iter_1_dir / "train.xyz"

    check_descriptor_dim(nep_file, xyz_file)
