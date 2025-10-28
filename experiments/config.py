from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConfigManager:
    """Manages ~/.experiments layout, global config.json, and per-project state.

    Structure:
      - ~/.experiments/config.json
      - ~/.experiments/logs/
      - ~/.experiments/projects/{project}/
          - project.json
          - canceled_jobs.json
          - launched_jobs.json
          - stages/{stage}/jobs.json
    """

    def __init__(self) -> None:
        self.config_dir: Path = Path.home() / ".experiments"
        self.config_file: Path = self.config_dir / "config.json"
        self.logs_dir: Path = self.config_dir / "logs"
        self.projects_dir: Path = self.config_dir / "projects"

    # ---------- Global config ----------
    def ensure_config(self) -> Dict[str, Any]:
        """Ensure config directory and file exist, return config dict."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)

        if not self.config_file.exists():
            default_config: Dict[str, Any] = {
                "log_directory": str(self.logs_dir),
                "default_partition": "general",
                # Partition-specific defaults; users may customize
                # Use "*" as a wildcard to set defaults for all partitions
                # Specific partition configs override the "*" defaults
                "default_slurm_args": {
                    "*": {"time": "2-00:00:00", "cpus": 1, "requeue": False},
                    "array": {"cpus": 4, "requeue": True},
                },
                # Defaults applied when initializing a new project.json
                # "{project_name}" in values will be replaced on init
                "project_defaults": {
                    "name": "{project_name}",
                },
            }
            self.save_config(default_config)
            return default_config

        # Load existing config and ensure keys
        config = self.load_config()
        if "default_partition" not in config:
            config["default_partition"] = "general"
        if "default_slurm_args" not in config:
            config["default_slurm_args"] = {}
        if "project_defaults" not in config:
            config["project_defaults"] = {"name": "{project_name}"}
        # Persist any migrations
        self.save_config(config)
        return config

    def load_config(self) -> Dict[str, Any]:
        with open(self.config_file, "r") as f:
            return json.load(f)

    def save_config(self, config: Dict[str, Any]) -> None:
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=2)

    # ---------- Project layout ----------
    def get_projects_dir(self) -> Path:
        return self.projects_dir

    def get_project_dir(self, project: str) -> Path:
        d = self.projects_dir / project
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_project_file(self, project: str) -> Path:
        return self.get_project_dir(project) / "project.json"

    def get_project_stages_dir(self, project: str) -> Path:
        d = self.get_project_dir(project) / "stages"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def get_stage_dir(self, project: str, stage: str) -> Path:
        d = self.get_project_stages_dir(project) / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------- Project state files ----------
    def get_canceled_jobs_file(self, project: str) -> Path:
        return self.get_project_dir(project) / "canceled_jobs.json"

    def get_launched_jobs_file(self, project: str) -> Path:
        return self.get_project_dir(project) / "launched_jobs.json"

    # ---------- Project config.json ----------
    def _apply_project_template(self, obj: Any, project_name: str) -> Any:
        if isinstance(obj, str):
            return obj.replace("{project_name}", project_name)
        if isinstance(obj, list):
            return [self._apply_project_template(v, project_name) for v in obj]
        if isinstance(obj, dict):
            return {k: self._apply_project_template(v, project_name) for k, v in obj.items()}
        return obj

    def ensure_project(self, project: str) -> Dict[str, Any]:
        """Ensure project directory and project.json exist; return project config."""
        self.get_project_dir(project)
        stages_dir = self.get_project_stages_dir(project)
        stages_dir.mkdir(parents=True, exist_ok=True)

        pfile = self.get_project_file(project)
        if not pfile.exists():
            # Build defaults from global config "project_defaults"
            global_conf = self.ensure_config()
            defaults = dict(global_conf.get("project_defaults", {}))
            # Always include name
            defaults.setdefault("name", "{project_name}")
            project_conf = {
                "config": self._apply_project_template(defaults, project),
            }
            with open(pfile, "w") as f:
                json.dump(project_conf, f, indent=2)
            return project_conf

        with open(pfile, "r") as f:
            return json.load(f)

    def load_project_config(self, project: str) -> Dict[str, Any]:
        self.ensure_project(project)
        with open(self.get_project_file(project), "r") as f:
            return json.load(f)

    def save_project_config(self, project: str, project_conf: Dict[str, Any]) -> None:
        with open(self.get_project_file(project), "w") as f:
            json.dump(project_conf, f, indent=2)

    # ---------- Jobs history (per project) ----------
    def save_job_info(self, project: str, stage: str, job_info: Dict[str, Any]) -> None:
        stage_dir = self.get_stage_dir(project, stage)
        jobs_file = stage_dir / "jobs.json"
        jobs: List[Dict[str, Any]] = []
        if jobs_file.exists():
            with open(jobs_file, "r") as f:
                jobs = json.load(f)
        jobs.append(job_info)
        with open(jobs_file, "w") as f:
            json.dump(jobs, f, indent=2)

    def load_jobs(self, project: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        jobs: List[Dict[str, Any]] = []
        if stage:
            stage_dir = self.get_stage_dir(project, stage)
            jobs_file = stage_dir / "jobs.json"
            if jobs_file.exists():
                with open(jobs_file, "r") as f:
                    stage_jobs = json.load(f)
                    for job in stage_jobs:
                        job["stage"] = stage
                    jobs.extend(stage_jobs)
        else:
            proj_stages_dir = self.get_project_stages_dir(project)
            if proj_stages_dir.exists():
                for stage_dir in proj_stages_dir.iterdir():
                    if stage_dir.is_dir():
                        jobs_file = stage_dir / "jobs.json"
                        if jobs_file.exists():
                            with open(jobs_file, "r") as f:
                                stage_jobs = json.load(f)
                                for job in stage_jobs:
                                    job["stage"] = stage_dir.name
                                jobs.extend(stage_jobs)
        return jobs

    # ---------- Cancel tracking ----------
    def load_canceled_jobs(self, project: str) -> set:
        p = self.get_canceled_jobs_file(project)
        if p.exists():
            with open(p, "r") as f:
                return set(json.load(f))
        return set()

    def save_canceled_job(self, project: str, job_id: str) -> None:
        canceled = self.load_canceled_jobs(project)
        canceled.add(job_id)
        p = self.get_canceled_jobs_file(project)
        with open(p, "w") as f:
            json.dump(sorted(canceled), f, indent=2)

    # ---------- Launched (per project) ----------
    def load_launched_jobs(self, project: str) -> Dict[str, Any]:
        p = self.get_launched_jobs_file(project)
        if p.exists():
            with open(p, "r") as f:
                return json.load(f)
        return {}

    def save_launched_jobs(self, project: str, mapping: Dict[str, Any]) -> None:
        p = self.get_launched_jobs_file(project)
        with open(p, "w") as f:
            json.dump(mapping, f, indent=2)

    # ---------- Helpers ----------
    def expand_path(self, p: str) -> str:
        return str(Path(os.path.expanduser(p)).resolve())

