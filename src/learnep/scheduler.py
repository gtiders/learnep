import subprocess
import time
import re
import os
from typing import Dict, Any
import logging


class JobRunner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.submit_cmd = config.get("submit_cmd", "qsub {script}")
        self.check_cmd = config.get("check_cmd", "qstat {job_id}")
        self.cancel_cmd = config.get("cancel_cmd", None)
        self.logger = logging.getLogger(__name__)
        self.check_interval = config.get("check_interval", 60)

    def submit_job(self, script_path: str, work_dir: str) -> str:
        """
        Submit a job script.
        Automatically appends 'touch DONE' to the script content before submission if not present.
        Returns: Job ID string
        """
        # Auto-inject 'touch DONE' logic
        # Read script, check if DONE is touched, if not append it.
        # Note: Ideally we modify the file on disk or assuming the caller prepares it?
        # The user requirement was "you need to automatically inject touch DONE".
        # We can append it to the file.

        with open(script_path, "r") as f:
            content = f.read()

        if "touch DONE" not in content:
            with open(script_path, "a") as f:
                f.write("\n\n# Auto-injected by LearnEP\ntouch DONE\n")

        # Prepare command
        # Support placeholders: {script}, {cwd}
        script_name = os.path.basename(script_path)
        abs_work_dir = os.path.abspath(work_dir)

        cmd_str = self.submit_cmd
        if "{script}" in cmd_str:
            cmd_str = cmd_str.replace("{script}", script_name)
        else:
            cmd_str = f"{cmd_str} {script_name}"

        if "{cwd}" in cmd_str:
            cmd_str = cmd_str.replace("{cwd}", abs_work_dir)

        # Check if {cwd} was used - if so, the command handles directory itself
        use_local_cwd = "{cwd}" not in self.submit_cmd

        self.logger.info(
            f"Submitting: {cmd_str}" + (f" in {work_dir}" if use_local_cwd else "")
        )
        try:
            # Run submission
            result = (
                subprocess.check_output(
                    cmd_str,
                    shell=True,
                    cwd=work_dir if use_local_cwd else None,
                    stderr=subprocess.STDOUT,
                )
                .decode("utf-8")
                .strip()
            )

            # Extract Job ID
            # Heuristic: usually the last non-empty line contains the ID, or is the ID
            job_id = result.splitlines()[-1].strip()
            # Simple fallback for PBS: 12345.server
            match = re.search(r"(\d+[\w\.-]*)", job_id)
            if match:
                job_id = match.group(1)

            self.logger.info(f"Job submitted. ID: {job_id}")
            return job_id
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Submission failed: {e.output.decode()}")
            raise e

    def wait_jobs(self, job_map: Dict[str, str], timeout: int = None):
        """
        Monitor a batch of jobs.
        job_map: {job_id: work_dir}
        timeout: seconds. If exceeded, cancel all remaining.
        Dual Check:
          1. Check if DONE file exists in work_dir.
          2. Check if Job ID is in queue.

        Cleanup Policy:
          If a job is considered "Done" (by any criteria), we explicitly send a cancel/kill signal
          to the scheduler (e.g. qdel) to ensure it is removed from the system, ignoring errors.
        """
        start_time = time.time()
        remaining_jobs = set(job_map.keys())

        self.logger.info(f"Waiting for {len(remaining_jobs)} jobs...")

        while remaining_jobs:
            # Check Timeout
            if timeout and (time.time() - start_time > timeout):
                self.logger.warning(
                    f"Batch TIMEOUT ({timeout}s). Cancelling remaining {len(remaining_jobs)} jobs."
                )
                self.cancel_batch(list(remaining_jobs))
                break

            done_jobs = set()
            for job_id in remaining_jobs:
                wd = job_map[job_id]
                done_file = os.path.join(wd, "DONE")

                # Check 1: DONE file
                if os.path.exists(done_file):
                    self.logger.info(
                        f"Job {job_id} COMPLETED (DONE file found). Performing cleanup..."
                    )
                    self.cancel_batch([job_id])  # Cleanup
                    done_jobs.add(job_id)
                    continue

                # Check 2: Queue Status
                if not self._is_job_running(job_id):
                    self.logger.info(
                        f"Job {job_id} DISAPPEARED from queue. Performing cleanup..."
                    )
                    self.cancel_batch([job_id])  # Cleanup
                    done_jobs.add(job_id)

            remaining_jobs -= done_jobs

            if remaining_jobs:
                time.sleep(self.check_interval)

    def _is_job_running(self, job_id: str) -> bool:
        """Check if job is still in queue system."""
        if "{job_id}" in self.check_cmd:
            cmd = self.check_cmd.format(job_id=job_id)
        else:
            cmd = f"{self.check_cmd} {job_id}"

        try:
            # If command fails (e.g. qstat returns strict error for finished job), it's not running.
            subprocess.check_call(
                cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def cancel_batch(self, job_ids: list):
        """Fire-and-forget cancellation."""
        if not self.cancel_cmd:
            return

        for jid in job_ids:
            # Handle template formatting if placeholder exists
            if "{job_id}" in self.cancel_cmd:
                cmd = self.cancel_cmd.replace("{job_id}", jid)
            else:
                cmd = f"{self.cancel_cmd} {jid}"

            self.logger.info(f"Cancelling {jid}: {cmd}")
            try:
                subprocess.run(cmd, shell=True, check=False, timeout=5)
            except Exception as e:
                self.logger.warning(f"Cancel failed for {jid} (ignored): {e}")
