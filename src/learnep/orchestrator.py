import os
import json
import shutil
import glob
import numpy as np
import logging
from ase.io import read, write

from .config import Config
from .scheduler import JobRunner
from .tasks.nep import NEPTask
from .tasks.gpumd import GPUMDTask
from .tasks.vasp import VASPTask

# Import JAXVol Logic (Vendored)
from learnep.jaxvol.tools import (
    scan_trajectory_gamma,
    get_B_projections,
    get_active_set,
)


class LearnEPOrchestrator:
    def __init__(self, config_path: str):
        self.config = Config(config_path)
        self._setup_logger()

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
            # Determine start iteration from status
            start_iter = self._get_last_completed_iter() + 1

        self.logger.info(f"Starting from Iteration {start_iter}")

        for n in range(start_iter, self.config.max_iterations):
            self._run_iteration(n)
            self._mark_iter_complete(n)

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

    def _run_iteration(self, n: int):
        self.logger.info(f"\n=== Iteration {n} ===")
        iter_conf = self.config.get_iteration_config(n)
        iter_dir = os.path.join(self.work_dir, f"iter_{n:03d}")

        # 1. Prepare Directory & Data
        self._prepare_iteration_data(n, iter_dir, iter_conf)

        # 2. Training (Check/Train)
        self._run_train(n, iter_dir, iter_conf)

        # 3. Exploration (GPUMD)
        traj_files = self._run_explore(n, iter_dir, iter_conf)

        # 4. Selection (JaxVol)
        candidates = self._run_selection(n, iter_dir, iter_conf, traj_files)

        if not candidates:
            self.logger.info(
                "No candidates selected. Iteration converged or exploration insufficient."
            )
            # Prepare next iter anyway (just forward model) or stop?
            # Forward model to keep loop alive or user intervention.
            self.nep_task.prepare_next_iter(iter_dir)
            return

        # Save candidates to file (User Request)
        # Ensure gamma info is preserved (Extended XYZ)
        # Also add summary stat to info for easy reading
        for cand in candidates:
            if "gamma" in cand.arrays:
                cand.info["gamma_max"] = float(np.max(cand.arrays["gamma"]))

        candidates_path = os.path.join(iter_dir, "candidates.xyz")
        write(candidates_path, candidates, format="extxyz")
        self.logger.info(f"Saved {len(candidates)} candidates to {candidates_path}")

        # 5. Labeling (VASP)
        new_data = self._run_label(n, iter_dir, iter_conf, candidates)

        # 6. Update (Merge Data & Prep Next)
        self._run_update(n, iter_dir, new_data)

    def _prepare_iteration_data(self, n: int, iter_dir: str, conf: dict):
        os.makedirs(iter_dir, exist_ok=True)

        self.logger.info(f"Preparing data for iter_{n}...")

        # Source: prev_iter/next_iter OR initial_input
        if n == 0:
            # Initial Load
            init_inp = self.config.initial_input
            # Copy train, nep.txt, etc. if they exist
            # Note: User might only provide train.xyz for cold start
            for k, f in init_inp.items():
                if f and os.path.exists(f):
                    shutil.copy2(f, os.path.join(iter_dir, os.path.basename(f)))
        else:
            prev_iter = n - 1
            prev_next_dir = os.path.join(
                self.work_dir, f"iter_{prev_iter:03d}", "next_iter"
            )

            if os.path.exists(prev_next_dir):
                for f in os.listdir(prev_next_dir):
                    shutil.copy2(
                        os.path.join(prev_next_dir, f), os.path.join(iter_dir, f)
                    )
            else:
                self.logger.warning(
                    f"Warning: Previous next_iter dir not found: {prev_next_dir}"
                )

    def _run_train(self, n: int, iter_dir: str, conf: dict):
        model_path = os.path.join(iter_dir, "nep.txt")
        restart_path = os.path.join(
            iter_dir, "nep.restart"
        )  # Only needed for hot start

        # Decide Input Content
        # Strict Hot Start Check: need BOTH nep.restart AND nep.txt to continue.
        # If user provides model but no restart => assume we can't hot start training properly (NEP usage).
        # Actually usually nep.restart is the key for CONTINUATION.
        if os.path.exists(restart_path) and os.path.exists(model_path):
            # Hot Start Detected
            if n == 0:
                self.logger.info(
                    "Hot Start at Iteration 0: Skipping initial training (using provided model for exploration)."
                )
                return

            self.logger.info(f"Hot Start Training (Iteration {n})...")
            inp_content = conf["nep"]["train_input"]
        else:
            self.logger.info("Cold Start / First Training (Missing restart files)...")
            # If first_train_input not defined, fallback to train_input
            inp_content = conf["nep"].get(
                "first_train_input", conf["nep"]["train_input"]
            )

        job_script = conf["nep"]["job_script"].replace("{iter}", str(n))

        script = self.nep_task.prepare_train(iter_dir, inp_content, job_script)

        jid = self.scheduler.submit_job(script, iter_dir)
        self.scheduler.wait_jobs({jid: iter_dir})

        # Verify result
        if not os.path.exists(os.path.join(iter_dir, "nep.txt")):
            raise RuntimeError(f"Training failed in iter {n}")

    def _run_explore(self, n: int, iter_dir: str, conf: dict):
        conditions = conf["gpumd"]["conditions"]
        job_script_tmpl = conf["gpumd"]["job_script"].replace("{iter}", str(n))

        explore_dir = os.path.join(iter_dir, "exploration")
        if not os.path.exists(explore_dir):
            os.makedirs(explore_dir)

        job_map = {}
        traj_files = []

        for cond in conditions:
            cond_id = cond["id"]
            c_dir = os.path.join(explore_dir, cond_id)

            script = self.gpumd_task.prepare_run(
                c_dir, cond["run_in"], cond["structure_file"], job_script_tmpl
            )

            jid = self.scheduler.submit_job(script, c_dir)
            job_map[jid] = c_dir

        self.scheduler.wait_jobs(job_map)

        # Collect trajectories
        for c_dir in job_map.values():
            t = self.gpumd_task.get_trajectory_path(c_dir)
            if t:
                traj_files.append(t)

        return traj_files

    def _run_selection(
        self, n: int, iter_dir: str, conf: dict, traj_files: list
    ) -> list:
        sel_conf = conf["selection"]
        mode = sel_conf.get("mode", "adaptive")
        gamma_conf = sel_conf.get("gamma", {})

        # Asi file comes from previous iteration training data (technically get_active_set from train.xyz)
        # But wait, jaxvol usually builds active set from *train.xyz*.
        # So we need to build active set from current train.xyz first.
        train_xyz = os.path.join(iter_dir, "train.xyz")
        nep_txt = os.path.join(iter_dir, "nep.txt")

        self.logger.info("Building Active Set from current train.xyz...")
        try:
            train_atoms = read(train_xyz, index=":")
            B_proj, B_idx = get_B_projections(train_atoms, nep_txt)
            # This generates 'active_set.asi' in CWD. We should handle paths.
            # get_active_set writes 'active_set.asi' by default.
            # We explicitly pass filename.
            asi_path = os.path.join(iter_dir, "active_set.asi")

            get_active_set(
                B_proj, B_idx, write_asi=True, asi_filename=asi_path, mode=mode
            )
        except Exception as e:
            self.logger.error(f"Selection Prep Failed: {e}")
            return []

        # Scan Trajectories
        candidates = []
        for t_file in traj_files:
            try:
                traj = read(t_file, index=":")  # or load_nep

                selected = scan_trajectory_gamma(
                    traj,
                    nep_file=nep_txt,
                    asi_file=asi_path,
                    gamma_min=gamma_conf.get("threshold", 0.05),
                    gamma_max=gamma_conf.get("threshold_max", 20.0),
                    auto_stop_qr=gamma_conf.get("auto_stop", False),
                    std_tol=gamma_conf.get("std_tol", 1e-4),
                )
                candidates.extend(selected)
            except Exception as e:
                self.logger.error(f"Scanning {t_file} failed: {e}")

        # Limit count
        n_max = sel_conf.get("n_max_label", 50)
        if len(candidates) > n_max:
            # Simple truncation or FPS?
            # For now truncation (greedy top gamma logic handled inside scan? no scan returns all valid).
            # We should sort by gamma? scan_trajectory_gamma returns them in temporal order.
            # Ideally we sort by gamma.
            candidates.sort(key=lambda x: max(x.arrays.get("gamma", [0])), reverse=True)
            candidates = candidates[:n_max]

        return candidates

    def _run_label(self, n: int, iter_dir: str, conf: dict, candidates: list):
        label_dir = os.path.join(iter_dir, "labeling")
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

    def _run_update(self, n: int, iter_dir: str, new_data: list):
        train_path = os.path.join(iter_dir, "train.xyz")

        # Append
        existing = read(train_path, index=":") if os.path.exists(train_path) else []
        combined = existing + new_data
        write(train_path, combined)  # update in place for this iter

        self.logger.info(f"Updated train.xyz: {len(existing)} -> {len(combined)}")

        # Prep Next Iter
        self.nep_task.prepare_next_iter(iter_dir)

    def _get_last_completed_iter(self):
        try:
            with open(os.path.join(self.work_dir, "status.json"), "r") as f:
                return json.load(f).get("last_completed", -1)
        except:  # noqa: E722
            return -1

    def _mark_iter_complete(self, n):
        with open(os.path.join(self.work_dir, "status.json"), "w") as f:
            json.dump({"last_completed": n}, f)
