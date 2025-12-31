import os
import shutil
import logging


class GPUMDTask:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_run(
        self, work_dir: str, run_in_content: str, structure_file: str, job_script: str
    ) -> str:
        # Check structure file
        if not os.path.exists(structure_file):
            raise FileNotFoundError(f"Structure file not found: {structure_file}")

        os.makedirs(work_dir, exist_ok=True)

        # Validation: Must contain 'dump_exyz' for Active Learning Selection
        if "dump_exyz" not in run_in_content:
            self.logger.error(
                "[GPUMD] 'dump_exyz' command missing in run_in. It is required for JAXVol selection."
            )
            raise ValueError(
                f"[GPUMD] 'dump_exyz' command missing in run_in for {work_dir}. It is required for JAXVol selection."
            )

        # Write run.in
        with open(os.path.join(work_dir, "run.in"), "w") as f:
            f.write(run_in_content)

        # Copy/Link structure file to model.xyz (standard GPUMD input name usually requires specific naming or run.in config)
        # Assuming run.in uses "model.xyz" or the user specified structure file is manually handled?
        # The config usually specifies `structure_file`. We should copy it to the work dir.
        # Note: GPUMD usually expects `model.xyz` if not specified otherwise, OR run.in has `xyz_file structure.xyz`.
        # We will copy input structure to `model.xyz` as a convention or keep original name.
        # Let's copy to `model.xyz` to be safe/standard if user config implies it.
        # Wait, user config `structure_file: input/init.xyz`.
        # We copy `input/init.xyz` -> `work_dir/model.xyz`? Or strictly follow file?
        # Let's copy it to `work_dir` with its basename.

        src_struct = os.path.abspath(structure_file)
        dst_struct = os.path.join(
            work_dir, "model.xyz"
        )  # Enforce standard name for simplicity

        # Read src, write dst (in case it's ASE atoms object later, but now just file copy)

        if os.path.exists(src_struct):
            shutil.copy2(src_struct, dst_struct)

        # Write job script
        script_path = os.path.join(work_dir, "run.sh")
        with open(script_path, "w") as f:
            f.write(job_script)

        return script_path

    def get_trajectory_path(self, work_dir: str) -> str:
        """
        Return the path to the dumped trajectory.
        GPUMD 'dump_exyz' typically produces 'movie.xyz' or 'dump.xyz'.
        We check for common names.
        """
        candidates = ["movie.xyz", "dump.xyz", "trajectory.xyz"]
        for c in candidates:
            p = os.path.join(work_dir, c)
            if os.path.exists(p):
                return p
        self.logger.warning(f"[GPUMD] No trajectory file found in {work_dir}")
        return None
