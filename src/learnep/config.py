import yaml
import copy
from typing import Dict, Any


class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.data = self._load_config()
        self._validate()

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _validate(self):
        """Basic validation of configuration structure."""
        required_sections = ["global", "vasp", "nep", "gpumd", "selection"]
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required section in config: {section}")

        g = self.data["global"]
        if "work_dir" not in g:
            raise ValueError("Global section must specify 'work_dir'")

    @property
    def global_settings(self) -> Dict[str, Any]:
        return self.data["global"]

    @property
    def work_dir(self) -> str:
        return self.data["global"]["work_dir"]

    @property
    def log_file(self) -> str:
        return self.data["global"].get("log_file", "learnep.log")

    @property
    def max_iterations(self) -> int:
        return self.data["global"].get("max_iterations", 50)

    @property
    def scheduler_config(self) -> Dict[str, Any]:
        return self.data["global"]["scheduler"]

    @property
    def initial_input(self) -> Dict[str, str]:
        return self.data["global"].get("initial_input", {})

    def get_iteration_config(self, iter_num: int) -> Dict[str, Any]:
        """
        Get configuration for a specific iteration number.
        Merge base config with any overrides from `iteration_control`.
        """
        current_config = copy.deepcopy(self.data)

        control = self.data.get("iteration_control", {})

        # Check explicit enable switch
        if not control.get("enabled", False):
            return current_config

        rules = control.get("rules", [])

        for rule in rules:
            target_iterations = rule.get("iterations", [])

            if iter_num in target_iterations:
                print(f"[Config] Applying override for iteration {iter_num}...")
                self._recursive_update(current_config, rule)

        return current_config

    def _recursive_update(self, d: Dict, u: Dict):
        """
        Recursive dictionary update (merge).
        """
        for k, v in u.items():
            if k == "iterations":
                continue

            if isinstance(v, dict):
                d[k] = self._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
