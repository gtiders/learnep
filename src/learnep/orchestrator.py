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
        # Returns absolute path to nep.txt
        nep_model_path = self._run_train(n, iter_dir, iter_conf)

        # 3. Exploration (GPUMD)
        traj_files = self._run_explore(n, iter_dir, iter_conf, nep_model_path)

        # 4. Selection (JaxVol)
        candidates = self._run_selection(
            n, iter_dir, iter_conf, traj_files, nep_model_path
        )

        if not candidates:
            self.logger.info(
                "No candidates selected. Iteration converged or exploration insufficient."
            )
            # Prepare next iter anyway (just forward model) to keep loop alive or allow manual check
            self._prep_next_from_paths(iter_dir, nep_model_path)
            return

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
        # If user says "no limit on wait time", we add one?
        # User said "seems no limit on wait time". We should add one.
        if timeout is None:
            timeout = 86400 * 3  # 3 days default safety

        self.scheduler.wait_jobs({jid: nep_work_dir}, timeout=timeout)

        # Verify result
        if not os.path.exists(model_path):
            raise RuntimeError(f"Training failed in iter {n}")

        return model_path

    def _run_explore(self, n: int, iter_dir: str, conf: dict, nep_path: str):
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
            job_map[jid] = c_dir

        timeout = conf["gpumd"].get("timeout", 86400)  # Default 24h
        self.scheduler.wait_jobs(job_map, timeout=timeout)

        # Collect trajectories
        for c_dir in job_map.values():
            t = self.gpumd_task.get_trajectory_path(c_dir)
            if t:
                traj_files.append(t)

        return traj_files

    def _run_selection(
        self, n: int, iter_dir: str, conf: dict, traj_files: list, nep_path: str
    ) -> list:
        sel_conf = conf["selection"]
        mode = sel_conf.get("mode", "adaptive")
        gamma_conf = sel_conf.get("gamma", {})

        # Use passed nep_path
        nep_txt = nep_path
        train_xyz = os.path.join(iter_dir, "train.xyz")

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
                    qr_threshold=gamma_conf.get("qr_threshold", 1e-4),
                    qr_threshold_max=gamma_conf.get("qr_threshold_max", 0.5),
                    auto_stop_qr=False,
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

    def _run_update(self, n: int, iter_dir: str, new_data: list, nep_path: str):
        train_path = os.path.join(iter_dir, "train.xyz")

        # Append
        existing = read(train_path, index=":") if os.path.exists(train_path) else []
        combined = existing + new_data
        write(train_path, combined)  # update in place for this iter

        self.logger.info(f"Updated train.xyz: {len(existing)} -> {len(combined)}")

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

        # 3. ASI files (at iter_dir root)
        for asi in glob.glob(os.path.join(iter_dir, "*.asi")):
            shutil.copy2(asi, os.path.join(next_iter_dir, os.path.basename(asi)))

    def _get_last_completed_iter(self):
        try:
            with open(os.path.join(self.work_dir, "status.json"), "r") as f:
                return json.load(f).get("last_completed", -1)
        except:  # noqa: E722
            return -1

    def _mark_iter_complete(self, n):
        with open(os.path.join(self.work_dir, "status.json"), "w") as f:
            json.dump({"last_completed": n}, f)
