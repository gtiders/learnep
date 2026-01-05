import os
import sys
import json
import shutil
import numpy as np
import logging
import re
from collections import defaultdict
from ase.io import read, write

from .config import Config
from .scheduler import JobRunner
from .tasks.nep import NEPTask
from .tasks.gpumd import GPUMDTask
from .tasks.vasp import VASPTask

# Import JAXVol Logic (Vendored)


class LearnEPOrchestrator:
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self._setup_logger()

        # Configure JAX Platform (Must be done before importing jaxvol)
        try:
            import jax

            sel_conf = self.config.data.get("selection", {})
            device = sel_conf.get("device", "cpu")
            # If user wants cpu, force it to avoid TPU probing warnings
            if device.lower() == "cpu":
                jax.config.update("jax_platform_name", "cpu")
                # Also hide the probe warning if possible, but platform_name=cpu is usually enough
            else:
                jax.config.update("jax_platform_name", device)
        except ImportError:
            pass

        self.scheduler = JobRunner(self.config.scheduler_config)
        self.nep_task = NEPTask(self.config)
        self.gpumd_task = GPUMDTask(self.config)
        self.vasp_task = VASPTask(self.config)

        self.work_dir = self.config.work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)

    def _setup_logger(self):
        log_path = os.path.join(self.config.work_dir, self.config.log_file)
        # Ensure log dir exists? Config usually says log filename, assumed in work_dir or abs.
        # Let's assume relative to work_dir if not abs.
        if not os.path.isabs(log_path):
            log_path = os.path.abspath(
                log_path
            )  # wait, above joins work_dir. Double join?
            # Config.log_file "test.log". work_dir="...". joined is correct.

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )
        self.logger = logging.getLogger("learnep")

    def run(self, restart_from: int = None):
        self.logger.info(f"--- LearnEP Started [Work Dir: {self.work_dir}] ---")

        # Handle manual restart request
        if restart_from is not None:
            self._handle_restart(restart_from)
            start_iter = restart_from
        else:
            # User Request: Pure run should not be affected by status.json logic blindly.
            # We default to 0.
            # If status.json exists, we just log it, we don't let it dictate starting point automatically
            # unless we implement a specific 'resume' flag. Default run = fresh start or explicit restart.
            status_iter = self._get_last_completed_iter()
            if status_iter >= 0:
                self.logger.info(
                    f"Status Record: Last completed iteration was {status_iter}. (Ignoring for current run)"
                )

            start_iter = 0

        self.logger.info(f"Starting from Iteration {start_iter}")

        # Convergence tracking
        consecutive_empty = 0
        convergence_threshold = self.config.data.get("global", {}).get(
            "convergence_empty_iters", 3
        )
        self.logger.info(
            f"[Config] Convergence: {convergence_threshold} consecutive empty iterations"
        )

        for n in range(start_iter, self.config.max_iterations):
            has_candidates = self._run_iteration(n)
            self._mark_iter_complete(n)

            if has_candidates:
                consecutive_empty = 0
            else:
                consecutive_empty += 1
                self.logger.info(
                    f"[Convergence] Empty iteration {consecutive_empty}/{convergence_threshold}"
                )

                if consecutive_empty >= convergence_threshold:
                    self.logger.info(
                        f"[CONVERGED] No candidates selected for {convergence_threshold} consecutive iterations. Stopping."
                    )
                    break

    def _handle_restart(self, start_n: int):
        """
        Clean up future iterations and reset status to allow restarting from start_n.
        """
        self.logger.info(
            f"[RESTART] Resetting state to start from Iteration {start_n}..."
        )

        # 1. Clean up directories iter >= start_n
        for n in range(
            start_n, self.config.max_iterations + 10
        ):  # Look ahead generously
            d = os.path.join(self.work_dir, f"iter_{n:03d}")
            if os.path.exists(d):
                self.logger.info(f"  Removing future dir: {d}")
                shutil.rmtree(d)

        # 2. Reset Status File
        # If we restart from N, then N-1 is the last completed one.
        new_last_completed = start_n - 1
        self._mark_iter_complete(new_last_completed)
        self.logger.info(f"  Status reset. Last completed: {new_last_completed}")

    def _run_iteration(self, n: int) -> bool:
        """
        Run a single iteration.
        Returns True if candidates were selected, False otherwise.
        """
        self.logger.info(f"\n=== Iteration {n} ===")
        iter_conf = self.config.get_iteration_config(n)
        iter_dir = os.path.join(self.work_dir, f"iter_{n:03d}")

        # 1. Prepare Directory & Data
        self._prepare_iteration_data(n, iter_dir, iter_conf)

        # 2. Training (Check/Train)
        # Returns absolute path to nep.txt
        nep_model_path = self._run_train(n, iter_dir, iter_conf)

        # 3. Exploration (GPUMD)
        traj_map = self._run_explore(n, iter_dir, iter_conf, nep_model_path)

        # 4. Selection (JaxVol)
        candidates = self._run_selection(
            n, iter_dir, iter_conf, traj_map, nep_model_path
        )

        if not candidates:
            self.logger.info(
                "No candidates selected. Iteration converged or exploration insufficient."
            )
            # Prepare next iter anyway (just forward model) to keep loop alive or allow manual check
            self._prep_next_from_paths(iter_dir, nep_model_path)
            return False  # No candidates

        # Save candidates to file
        for cand in candidates:
            if "gamma" in cand.arrays:
                cand.info["gamma_max"] = float(np.max(cand.arrays["gamma"]))

        candidates_path = os.path.join(iter_dir, "candidates.xyz")
        write(candidates_path, candidates, format="extxyz")
        self.logger.info(f"Saved {len(candidates)} candidates to {candidates_path}")

        # 5. Labeling (VASP)
        new_data = self._run_label(n, iter_dir, iter_conf, candidates)

        # 6. Update (Merge Data & Prep Next)
        self._run_update(n, iter_dir, new_data, nep_model_path)

        return True  # Candidates were found and processed

    def _prepare_iteration_data(self, n: int, iter_dir: str, conf: dict):
        os.makedirs(iter_dir, exist_ok=True)

        self.logger.info(f"Preparing data for iter_{n}...")

        # File Rename Mapping
        DEST_MAP = {
            "train_data": "train.xyz",
            "nep_model": "nep.txt",
            "nep_restart": "nep.restart",
        }

        # Source: prev_iter/next_iter OR initial_input
        if n == 0:
            # Initial Load
            init_inp = self.config.initial_input
            for k, f in init_inp.items():
                if f and os.path.exists(f):
                    # Determine destination name
                    dest_name = DEST_MAP.get(k, os.path.basename(f))
                    dest_path = os.path.join(iter_dir, dest_name)

                    self.logger.info(f"  Copying {f} -> {dest_path}")
                    shutil.copy2(f, dest_path)
                elif f:
                    self.logger.warning(
                        f"  [Warn] Initial input file not found: {f} (key={k})"
                    )
        else:
            prev_iter = n - 1
            prev_next_dir = os.path.join(
                self.work_dir, f"iter_{prev_iter:03d}", "next_iter"
            )

            if os.path.exists(prev_next_dir):
                files = os.listdir(prev_next_dir)
                self.logger.info(f"  Copying {len(files)} files from {prev_next_dir}")
                for f in files:
                    src = os.path.join(prev_next_dir, f)
                    dst = os.path.join(iter_dir, f)
                    shutil.copy2(src, dst)
            else:
                self.logger.warning(
                    f"Warning: Previous next_iter dir not found: {prev_next_dir}"
                )

    def _run_train(self, n: int, iter_dir: str, conf: dict) -> str:
        # User Request: separate sub-directory for NEP
        nep_work_dir = os.path.join(iter_dir, "nep")
        os.makedirs(nep_work_dir, exist_ok=True)

        model_path = os.path.join(nep_work_dir, "nep.txt")
        restart_path = os.path.join(nep_work_dir, "nep.restart")

        # Files at iter_dir root
        root_train = os.path.join(iter_dir, "train.xyz")
        root_model = os.path.join(iter_dir, "nep.txt")
        root_restart = os.path.join(iter_dir, "nep.restart")

        # --- DFT-First Mode Detection ---
        # At Iteration 0, if train.xyz has no forces (or all zeros) AND no nep.txt,
        # we need to label the structures via DFT first.
        if n == 0 and os.path.exists(root_train) and not os.path.exists(root_model):
            if self._needs_dft_labeling(root_train):
                self.logger.info(
                    "[DFT-First Mode] train.xyz has no valid forces. Running DFT labeling..."
                )
                unlabeled = read(root_train, index=":")
                # Use separate directory for initial DFT labeling to avoid conflict with candidate labeling
                labeled = self._run_label(
                    n, iter_dir, conf, unlabeled, label_subdir="labeling_init"
                )

                if labeled:
                    write(root_train, labeled)
                    self.logger.info(
                        f"[DFT-First Mode] Labeled {len(labeled)} structures. Updated train.xyz."
                    )
                else:
                    raise RuntimeError(
                        "[DFT-First Mode] DFT labeling failed. No labeled structures returned."
                    )

        # Copy files to sub-dir
        if os.path.exists(root_train):
            shutil.copy2(root_train, os.path.join(nep_work_dir, "train.xyz"))
        if os.path.exists(root_model):
            shutil.copy2(root_model, model_path)
        if os.path.exists(root_restart):
            shutil.copy2(root_restart, restart_path)

        # Decide Input Content
        # Check hot start in SUB_DIR
        if os.path.exists(restart_path) and os.path.exists(model_path):
            # Hot Start Detected
            if n == 0:
                self.logger.info(
                    "Hot Start at Iteration 0: Skipping initial training (using provided model)."
                )
                return model_path

            self.logger.info(f"Hot Start Training (Iteration {n})...")
            inp_content = conf["nep"]["train_input"]
        else:
            self.logger.info("Cold Start / First Training (Missing restart files)...")
            inp_content = conf["nep"].get(
                "first_train_input", conf["nep"]["train_input"]
            )

        job_script = conf["nep"]["job_script"].replace("{iter}", str(n))
        script = self.nep_task.prepare_train(nep_work_dir, inp_content, job_script)

        jid = self.scheduler.submit_job(script, nep_work_dir)

        # User Request: Timeout
        timeout = conf["nep"].get("timeout", None)  # Default None or set default?
        if timeout is None:
            timeout = 86400 * 3  # 3 days default safety

        self.scheduler.wait_jobs({jid: nep_work_dir}, timeout=timeout)

        # Verify result
        if not os.path.exists(model_path):
            raise RuntimeError(f"Training failed in iter {n}")

        return model_path

    def _needs_dft_labeling(self, train_path: str) -> bool:
        """
        Check if train.xyz needs DFT labeling.
        Returns True if ANY structure has:
          - No 'forces' key in atoms.arrays, OR all forces are zero
          - No 'energy' key in atoms.info, OR energy is zero
        """
        try:
            atoms_list = read(train_path, index=":")
            self.logger.info(
                f"[Validation] Checking {len(atoms_list)} structures in {train_path}..."
            )

            needs_labeling = False
            for i, atoms in enumerate(atoms_list):
                # Check forces
                has_forces = "forces" in atoms.arrays
                forces_valid = False
                if has_forces:
                    forces = atoms.arrays["forces"]
                    forces_valid = not np.allclose(forces, 0.0)

                # Check energy
                has_energy = "energy" in atoms.info
                energy_valid = False
                if has_energy:
                    energy = atoms.info["energy"]
                    energy_valid = not np.isclose(energy, 0.0)

                if not forces_valid or not energy_valid:
                    self.logger.warning(
                        f"[Validation] Structure {i}: forces_valid={forces_valid}, energy_valid={energy_valid}"
                    )
                    needs_labeling = True

            if needs_labeling:
                self.logger.info("[Validation] Some structures need DFT labeling.")
            else:
                self.logger.info(
                    "[Validation] All structures have valid forces and energy."
                )

            return needs_labeling

        except Exception as e:
            self.logger.warning(f"Error checking data in {train_path}: {e}")
            return True  # Conservative: assume needs labeling on error

    def _run_explore(self, n: int, iter_dir: str, conf: dict, nep_path: str):
        """
        Run GPUMD exploration and submit gamma scanning tasks.
        Returns traj_map: {cond_id: extrapolation.xyz path}
        """
        conditions = conf["gpumd"]["conditions"]
        job_script_tmpl = conf["gpumd"]["job_script"].replace("{iter}", str(n))

        explore_dir = os.path.join(iter_dir, "exploration")
        if not os.path.exists(explore_dir):
            os.makedirs(explore_dir)

        # Build Active Set FIRST (needed for scanning)
        asi_path = os.path.join(iter_dir, "active_set.asi")
        if not os.path.exists(asi_path):
            self.logger.info("Building Active Set for gamma scanning...")
            self._build_active_set(iter_dir, nep_path, conf)

        job_map = {}  # {jid: (cond_id, c_dir)}

        for cond in conditions:
            cond_id = cond["id"]
            c_dir = os.path.join(explore_dir, cond_id)
            os.makedirs(c_dir, exist_ok=True)

            # Copy NEP model passed from training/init
            if nep_path and os.path.exists(nep_path):
                shutil.copy2(nep_path, os.path.join(c_dir, "nep.txt"))
            else:
                self.logger.error(f"NEP model not found at {nep_path}")
                raise FileNotFoundError("NEP model missing for exploration.")

            script = self.gpumd_task.prepare_run(
                c_dir, cond["run_in"], cond["structure_file"], job_script_tmpl
            )

            jid = self.scheduler.submit_job(script, c_dir)
            job_map[jid] = (cond_id, c_dir)

        timeout = conf["gpumd"].get("timeout", 86400)  # Default 24h
        scan_timeout = conf["selection"].get("scan_timeout", 3600)  # Default 1h

        traj_map = {}  # {cond_id: extrapolation.xyz path}
        scan_job_map = {}  # {scan_jid: (cond_id, c_dir)}

        # Wait for all GPUMD jobs
        self.scheduler.wait_jobs(
            {jid: info[1] for jid, info in job_map.items()}, timeout=timeout
        )

        # Submit scan tasks for each completed GPUMD
        for jid, (cond_id, c_dir) in job_map.items():
            traj_path = self.gpumd_task.get_trajectory_path(c_dir)
            if not traj_path:
                self.logger.warning(f"[{cond_id}] No trajectory found, skipping scan")
                continue

            extrap_path = os.path.join(c_dir, "extrapolation.xyz")

            # Skip if extrapolation already exists (avoid re-computation)
            if os.path.exists(extrap_path):
                self.logger.info(
                    f"[{cond_id}] Extrapolation exists, skipping scan: {extrap_path}"
                )
                traj_map[cond_id] = extrap_path
                continue

            # Prepare and submit scan task
            try:
                # Remove DONE file left by GPUMD job to avoid interfering with scan job monitoring
                done_file = os.path.join(c_dir, "DONE")
                if os.path.exists(done_file):
                    os.remove(done_file)
                    self.logger.debug(f"[{cond_id}] Removed old DONE file before scan")

                scan_script = self._prepare_scan_task(
                    c_dir, traj_path, nep_path, asi_path, conf
                )
                scan_jid = self.scheduler.submit_job(scan_script, c_dir)
                scan_job_map[scan_jid] = (cond_id, c_dir)
                self.logger.info(f"[{cond_id}] Submitted scan task: {scan_jid}")
            except Exception as e:
                self.logger.error(f"[{cond_id}] Failed to submit scan task: {e}")
                continue

        # Wait for all scan tasks
        if scan_job_map:
            self.logger.info(f"Waiting for {len(scan_job_map)} scan tasks...")
            self.scheduler.wait_jobs(
                {jid: info[1] for jid, info in scan_job_map.items()},
                timeout=scan_timeout,
            )

        # Collect extrapolation results
        for scan_jid, (cond_id, c_dir) in scan_job_map.items():
            extrap_path = os.path.join(c_dir, "extrapolation.xyz")
            if os.path.exists(extrap_path):
                traj_map[cond_id] = extrap_path
                self.logger.info(f"[{cond_id}] Scan complete: {extrap_path}")
            else:
                self.logger.warning(
                    f"[{cond_id}] Scan failed: extrapolation.xyz not found"
                )

        return traj_map

    def _prepare_scan_task(
        self, work_dir: str, traj_path: str, nep_path: str, asi_path: str, conf: dict
    ) -> str:
        """
        Prepare gamma scanning job script with auto-injected Python path and parameters.
        """
        sel_conf = conf["selection"]
        gamma_conf = sel_conf.get("gamma", {})

        # Auto-detect current Python interpreter absolute path
        python_exe = sys.executable

        # Generate scan command with absolute paths
        output_path = os.path.join(work_dir, "extrapolation.xyz")
        min_dist = sel_conf.get("min_dist", None)

        scan_cmd = (
            f"{python_exe} -m learnep.jaxvol gamma "
            f"--input {os.path.abspath(traj_path)} "
            f"--nep-path {os.path.abspath(nep_path)} "
            f"--asi-path {os.path.abspath(asi_path)} "
            f"--output {os.path.abspath(output_path)} "
            f"--threshold {gamma_conf.get('threshold', 1.05)} "
            f"--threshold-max {gamma_conf.get('threshold_max', 20.0)}"
        )

        if min_dist is not None:
            scan_cmd += f" --min-dist {min_dist}"

        # Get job script template from config, auto-inject scan command
        job_tmpl = sel_conf.get("scan_job_script", self._default_scan_job_script())
        job_content = job_tmpl.replace("{scan_cmd}", scan_cmd)

        script_path = os.path.join(work_dir, "scan.sh")
        with open(script_path, "w") as f:
            f.write(job_content)

        return script_path

    def _default_scan_job_script(self) -> str:
        """Default job script template for gamma scanning."""
        return """#!/bin/bash
#PBS -N gamma_scan
#PBS -q gpu
#PBS -l nodes=1:ppn=1:gpus=1
cd $PBS_O_WORKDIR
{scan_cmd}
"""

    def _build_active_set(self, iter_dir: str, nep_path: str, conf: dict):
        """Build active set from train.xyz for gamma scanning."""
        from learnep.jaxvol.tools import get_B_projections, get_active_set

        sel_conf = conf["selection"]
        mode = sel_conf.get("mode", "adaptive")
        train_xyz = os.path.join(iter_dir, "train.xyz")
        asi_path = os.path.join(iter_dir, "active_set.asi")

        train_atoms = read(train_xyz, index=":")
        B_proj, B_idx = get_B_projections(train_atoms, nep_path)
        get_active_set(B_proj, B_idx, write_asi=True, asi_filename=asi_path, mode=mode)
        self.logger.info(f"Active Set saved to {asi_path}")

    def _run_selection(
        self, n: int, iter_dir: str, conf: dict, traj_map: dict, nep_path: str
    ) -> list:
        """
        Load pre-computed extrapolation results and perform MaxVol sub-selection.
        traj_map: {cond_id: extrapolation.xyz path} from _run_explore
        """
        from learnep.jaxvol.tools import get_B_projections, get_active_set

        sel_conf = conf["selection"]
        mode = sel_conf.get("mode", "adaptive")

        nep_txt = nep_path
        train_xyz = os.path.join(iter_dir, "train.xyz")
        asi_path = os.path.join(iter_dir, "active_set.asi")

        # Active Set should already be built by _run_explore, but check just in case
        if not os.path.exists(asi_path):
            self.logger.info("Active Set not found, building...")
            try:
                train_atoms = read(train_xyz, index=":")
                B_proj, B_idx = get_B_projections(train_atoms, nep_txt)
                get_active_set(
                    B_proj, B_idx, write_asi=True, asi_filename=asi_path, mode=mode
                )
            except Exception as e:
                self.logger.error(f"Selection Prep Failed: {e}")
                return []

        # Load pre-computed extrapolation results
        candidates = []
        temp_counts = defaultdict(int)

        # Helper to find config for cond_id
        cond_config_map = {c["id"]: c for c in conf["gpumd"]["conditions"]}

        for cond_id, extrap_file in traj_map.items():
            try:
                # Directly load pre-computed extrapolation results
                selected = read(extrap_file, index=":")
                self.logger.info(
                    f"[{cond_id}] Loaded {len(selected)} candidates from {extrap_file}"
                )
                candidates.extend(selected)

                # Track Temperature Source
                c_conf = cond_config_map.get(cond_id, {})
                run_in_txt = c_conf.get("run_in", "")
                temp_match = re.search(r"ensemble\s+\w+\s+(\d+)", run_in_txt)
                temp = "Unknown"
                if temp_match:
                    temp = f"{temp_match.group(1)}K"

                temp_counts[temp] += len(selected)

            except Exception as e:
                self.logger.error(f"Loading {extrap_file} failed: {e}")

        # Log Selection Statistics by Temperature
        if temp_counts:
            stats_msg = "\n[Selection] Candidate Sources by Temperature:"
            # Sort by temperature if possible (parse K)
            sorted_temps = sorted(
                temp_counts.keys(),
                key=lambda x: int(x[:-1]) if x[:-1].isdigit() else 99999,
            )
            for t in sorted_temps:
                stats_msg += f"\n  - {t}: {temp_counts[t]} structures"
            self.logger.info(stats_msg + "\n")

        if not candidates:
            return []

        # --- Improved Selection Strategy: MaxVol Sub-selection + Random Fallback ---
        # 1. Deduplication using MaxVol (Candidate Resampling)
        # Combine Train + Candidates to ensure we only pick candidates that add new information relative to Train

        self.logger.info(f"Initial candidates (High Gamma): {len(candidates)}")

        if not candidates:
            return []

        # Read Train Data again (or reuse train_atoms if available and safe)
        # To be safe and stateless, we reconstruct the list.
        # But wait, earlier we read train_atoms (L428). Is it still in memory/scope?
        # L428: train_atoms = read(train_xyz, index=":") -> It is in scope if L428 executed?
        # Yes, L428 is inside try-catch block.
        # Let's re-read to be absolutely safe or structure the code to ensure access.
        try:
            train_atoms = read(train_xyz, index=":")
        except Exception:
            train_atoms = []

        combined_data = train_atoms + candidates
        n_train = len(train_atoms)
        self.logger.info(
            f"Performing MaxVol Sub-selection on {n_train} existing + {len(candidates)} candidates..."
        )

        try:
            # We need B Projections for the combined set to rerun MaxVol
            B_comb, B_idx_comb = get_B_projections(combined_data, nep_txt)

            # Run MaxVol selection (no writing ASI, just get indices)
            # This corresponds to the 'extend' logic
            _, selected_indices_comb = get_active_set(
                B_comb, B_idx_comb, write_asi=False, mode=mode
            )

            # Filter: Keep only indices that belong to 'candidates' part
            # convert to set for O(1) lookup
            selected_set = set(selected_indices_comb)

            # Candidates start from index n_train
            final_candidates = []
            for i in range(len(candidates)):
                global_idx = n_train + i
                if global_idx in selected_set:
                    final_candidates.append(candidates[i])

            self.logger.info(
                f"MaxVol retained {len(final_candidates)} non-redundant candidates (from {len(candidates)})."
            )

        except Exception as e:
            self.logger.error(
                f"MaxVol Sub-selection failed: {e}. Falling back to raw candidates."
            )
            final_candidates = candidates

        # 2. Random Sampling (if still too many)
        n_max = sel_conf.get("n_max_label", 50)

        if len(final_candidates) > n_max:
            import random

            self.logger.info(
                f"Count ({len(final_candidates)}) > limit ({n_max}). Applying Random Sampling..."
            )
            random_seed = sel_conf.get("seed", 42)
            random.seed(random_seed)
            final_candidates = random.sample(final_candidates, n_max)
            self.logger.info(f"Randomly selected {len(final_candidates)} candidates.")
        else:
            self.logger.info(
                f"Count ({len(final_candidates)}) <= limit ({n_max}). Keeping all."
            )

        return final_candidates

    def _run_label(
        self,
        n: int,
        iter_dir: str,
        conf: dict,
        candidates: list,
        label_subdir: str = "labeling",
    ):
        label_dir = os.path.join(iter_dir, label_subdir)
        vasp_conf = conf["vasp"]

        job_tmpl = vasp_conf["job_script"].replace("{iter}", str(n))
        input_files = vasp_conf["input_files"]

        # Returns {script_path: sub_dir}
        prep_map = self.vasp_task.prepare_calculations(
            label_dir, candidates, input_files, job_tmpl
        )

        # Must submit jobs!
        job_map = {}
        for script, sub_dir in prep_map.items():
            jid = self.scheduler.submit_job(script, sub_dir)
            job_map[jid] = sub_dir

        timeout = vasp_conf.get("timeout", 86400)
        self.scheduler.wait_jobs(job_map, timeout=timeout)

        return self.vasp_task.collect_results(label_dir)

    def _run_update(self, n: int, iter_dir: str, new_data: list, nep_path: str):
        train_path = os.path.join(iter_dir, "train.xyz")

        # Read existing data
        existing = read(train_path, index=":") if os.path.exists(train_path) else []

        # Filter new_data to ensure all have valid forces and energy
        valid_new = []
        for atoms in new_data:
            has_forces = "forces" in atoms.arrays and not np.allclose(
                atoms.arrays["forces"], 0.0
            )
            has_energy = "energy" in atoms.info and not np.isclose(
                atoms.info["energy"], 0.0
            )
            if has_forces and has_energy:
                valid_new.append(atoms)

        if len(valid_new) < len(new_data):
            self.logger.warning(
                f"[Update] Filtered out {len(new_data) - len(valid_new)} invalid structures from new data"
            )

        # Append valid new data
        combined = existing + valid_new
        write(train_path, combined)

        self.logger.info(
            f"[Update] train.xyz: {len(existing)} existing + {len(valid_new)} new = {len(combined)} total"
        )

        # Prep Next Iter
        self._prep_next_from_paths(iter_dir, nep_path)

    def _prep_next_from_paths(self, iter_dir: str, nep_path: str):
        next_iter_dir = os.path.join(iter_dir, "next_iter")
        os.makedirs(next_iter_dir, exist_ok=True)

        # 1. New Model files
        if os.path.exists(nep_path):
            shutil.copy2(nep_path, os.path.join(next_iter_dir, "nep.txt"))

        nep_dir = os.path.dirname(nep_path)
        restart_path = os.path.join(nep_dir, "nep.restart")
        if os.path.exists(restart_path):
            shutil.copy2(restart_path, os.path.join(next_iter_dir, "nep.restart"))

        # 2. Updated Train Data
        train_path = os.path.join(iter_dir, "train.xyz")
        if os.path.exists(train_path):
            shutil.copy2(train_path, os.path.join(next_iter_dir, "train.xyz"))

        # Note: ASI files are NOT copied as they are regenerated each iteration from train.xyz

    def _get_last_completed_iter(self):
        try:
            with open(os.path.join(self.work_dir, "status.json"), "r") as f:
                return json.load(f).get("last_completed", -1)
        except Exception:
            return -1

    def _mark_iter_complete(self, n):
        with open(os.path.join(self.work_dir, "status.json"), "w") as f:
            json.dump({"last_completed": n}, f)
