import os
import shutil
import logging
from ase.io import read, write


class VASPTask:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_calculations(
        self, work_dir: str, candidates: list, input_files: list, job_tmpl: str
    ):
        """
        Prepare batch VASP calculations.
        candidates: List of ASE Atoms objects.
        input_files: List of paths to INCAR, POTCAR, etc.
        job_tmpl: Job script content.
        """
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        job_map = {}  # {script_path: sub_dir}

        for i, atoms in enumerate(candidates):
            # Prepare directory
            sub_dir = os.path.join(work_dir, f"calc_{i:04d}")
            os.makedirs(sub_dir, exist_ok=True)

            # Write POSCAR
            write(os.path.join(sub_dir, "POSCAR"), atoms, format="vasp")

            # Copy Inputs
            for inp in input_files:
                if os.path.exists(inp):
                    shutil.copy2(inp, sub_dir)
                else:
                    self.logger.warning(f"[VASP] Warning: Input file {inp} not found!")

            # Write Job Script
            # We must make sure the job script runs in the sub_dir
            # Usually we submit a script *inside* the sub_dir
            script_name = "vasp.sh"
            script_path = os.path.join(sub_dir, script_name)

            # Add cd command to script if not present?
            # Usually PBS_O_WORKDIR handles it if submitted from dir.
            # But safer to just put script there.
            with open(script_path, "w") as f:
                f.write(job_tmpl)

            job_map[script_path] = sub_dir

        return job_map

    def collect_results(self, work_dir: str) -> list:
        """
        Collect results from all subdirectories.
        Returns list of ASE Atoms with forces/energy in arrays/info.
        Filters out structures without valid forces or energy.
        """
        import numpy as np

        results = []
        failed_count = 0
        invalid_count = 0

        if not os.path.exists(work_dir):
            self.logger.warning(
                f"[VASP] Work dir {work_dir} needed for collection but not found."
            )
            return []

        subdirs = sorted([d for d in os.listdir(work_dir) if d.startswith("calc_")])
        self.logger.info(
            f"[VASP] Collecting results from {len(subdirs)} calculations..."
        )

        for d in subdirs:
            full_path = os.path.join(work_dir, d)
            try:
                # vasprun.xml is preferred for precision
                xml = os.path.join(full_path, "vasprun.xml")
                outcar = os.path.join(full_path, "OUTCAR")

                atoms = None
                source = None
                if os.path.exists(xml):
                    atoms = read(xml)
                    source = "vasprun.xml"
                elif os.path.exists(outcar):
                    atoms = read(outcar)
                    source = "OUTCAR"
                else:
                    self.logger.warning(f"[VASP] No output found in {d}")
                    failed_count += 1
                    continue

                # Extract forces and energy from calculator if present
                forces = None
                energy = None

                # Try to get from calculator first
                if atoms.calc is not None:
                    try:
                        forces = atoms.calc.get_forces()
                    except Exception:
                        pass
                    try:
                        energy = atoms.calc.get_potential_energy()
                    except Exception:
                        pass

                # Fallback to arrays/info if calculator didn't provide
                if forces is None and "forces" in atoms.arrays:
                    forces = atoms.arrays["forces"]
                if energy is None and "energy" in atoms.info:
                    energy = atoms.info["energy"]

                # Validate forces and energy
                has_valid_forces = forces is not None and not np.allclose(forces, 0.0)
                has_valid_energy = energy is not None and not np.isclose(energy, 0.0)

                if not has_valid_forces or not has_valid_energy:
                    self.logger.warning(
                        f"[VASP] {d}: Invalid data - forces_valid={has_valid_forces}, energy_valid={has_valid_energy}"
                    )
                    invalid_count += 1
                    continue

                # Ensure forces and energy are in arrays/info for extxyz write
                atoms.arrays["forces"] = forces
                atoms.info["energy"] = energy

                # Also store stress if available
                if atoms.calc is not None:
                    try:
                        stress = atoms.calc.get_stress()
                        if stress is not None:
                            atoms.info["stress"] = stress
                    except Exception:
                        pass

                # Detach calculator to prevent issues during write (e.g. extxyz trying to use it)
                # We already extracted what we need (forces, energy, stress)
                atoms.calc = None

                results.append(atoms)
                self.logger.debug(
                    f"[VASP] {d}: OK from {source} - E={energy:.4f} eV, max|F|={np.max(np.abs(forces)):.4f} eV/Å"
                )

            except Exception as e:
                self.logger.error(f"[VASP] Failed to parse {d}: {e}")
                failed_count += 1

        self.logger.info(
            f"[VASP] Collection complete: {len(results)} valid, {invalid_count} invalid, {failed_count} failed"
        )
        return results
