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
        Returns list of ASE Atoms with calculator results attached.
        """
        results = []
        # sort dirs
        if not os.path.exists(work_dir):
            self.logger.warning(
                f"[VASP] Work dir {work_dir} needed for collection but not found."
            )
            return []

        subdirs = sorted([d for d in os.listdir(work_dir) if d.startswith("calc_")])

        for d in subdirs:
            full_path = os.path.join(work_dir, d)
            # Try read OUTCAR or vasprun.xml
            try:
                # vasprun.xml is preferred for precision
                xml = os.path.join(full_path, "vasprun.xml")
                outcar = os.path.join(full_path, "OUTCAR")

                if os.path.exists(xml):
                    atoms = read(xml)
                    results.append(atoms)
                elif os.path.exists(outcar):
                    atoms = read(outcar)
                    results.append(atoms)
                else:
                    self.logger.warning(f"[VASP] No output found in {d}")
            except Exception as e:
                self.logger.error(f"[VASP] Failed to parse {d}: {e}")

        return results
