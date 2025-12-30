import os
import shutil
import logging


class NEPTask:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def prepare_train(self, work_dir: str, train_input: str, job_script: str):
        """
        Prepare directory for NEP training.
        """
        if not os.path.exists(work_dir):
            os.makedirs(work_dir)

        # Write nep.in
        with open(os.path.join(work_dir, "nep.in"), "w") as f:
            f.write(train_input)

        # Write Job Script
        script_path = os.path.join(work_dir, "train.sh")
        with open(script_path, "w") as f:
            f.write(job_script)

        return script_path

    def prepare_next_iter(self, current_work_dir: str):
        """
        Prepare the `next_iter` folder for the next cycle.
        Copy model files and train.xyz to a subdirectory.
        """
        next_iter_dir = os.path.join(current_work_dir, "next_iter")
        if not os.path.exists(next_iter_dir):
            os.makedirs(next_iter_dir, exist_ok=True)

        # Files to forward
        # User Logic: nep.txt, nep.restart, asi.
        # System Logic: train.xyz is also strictly required for training accumulation.
        files_to_copy = ["nep.txt", "nep.restart", "train.xyz"]

        # Copy explicit files
        for f in files_to_copy:
            src = os.path.join(current_work_dir, f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(next_iter_dir, f))
            else:
                # nep.restart might not exist in cold start or first iter
                if f != "nep.restart":
                    self.logger.warning(f"Warning: {f} not found in {current_work_dir}")

        # Copy ASI files (glob)
        import glob

        for asi in glob.glob(os.path.join(current_work_dir, "*.asi")):
            shutil.copy2(asi, os.path.join(next_iter_dir, os.path.basename(asi)))
