# Specification: `experiments/config.py`

## Purpose
Manages the persistent `~/.experiments/` directory layout, global configuration, per-project state files, job history, and cancel tracking. All filesystem state used by the `experiments` framework is centralized through this class.

## Directory Layout

```
~/.experiments/
├── config.json                  # Global config
├── logs/                        # Default log directory for Slurm output
└── projects/
    └── {project}/
        ├── project.json         # Per-project config
        ├── canceled_jobs.json   # Set of canceled job IDs
        ├── launched_jobs.json   # relpath -> {job_id, array_index} mapping
        └── stages/
            └── {stage}/
                └── jobs.json    # Array of job info dicts for this stage
```

---

## Class: `ConfigManager`

### Constructor: `__init__(self) -> None`
Sets four `Path` attributes:
- `self.config_dir` = `~/.experiments`
- `self.config_file` = `~/.experiments/config.json`
- `self.logs_dir` = `~/.experiments/logs`
- `self.projects_dir` = `~/.experiments/projects`

---

### Global Config Methods

#### `ensure_config(self) -> Dict[str, Any]`
- Creates `config_dir`, `logs_dir`, `projects_dir` with `mkdir(parents=True, exist_ok=True)`.
- If `config.json` does **not** exist, writes a default config and returns it:
  ```json
  {
    "log_directory": "~/.experiments/logs",
    "default_partition": "general",
    "default_slurm_args": {
      "*": {"time": "2-00:00:00", "cpus": 1, "requeue": false},
      "array": {"cpus": 4, "requeue": true}
    },
    "project_defaults": {
      "name": "{project_name}"
    }
  }
  ```
- If `config.json` **does** exist, loads it and ensures keys `default_partition`, `default_slurm_args`, and `project_defaults` are present (migrates if missing), then saves and returns.

#### `load_config(self) -> Dict[str, Any]`
- Reads and returns `config.json` as a dict.

#### `save_config(self, config: Dict[str, Any]) -> None`
- Writes `config` to `config.json` with `indent=2`.

---

### Project Layout Methods

#### `get_projects_dir(self) -> Path`
- Returns `self.projects_dir`.

#### `get_project_dir(self, project: str) -> Path`
- Returns `projects_dir / project`, creating it if necessary.

#### `get_project_file(self, project: str) -> Path`
- Returns `get_project_dir(project) / "project.json"`.

#### `get_project_stages_dir(self, project: str) -> Path`
- Returns `get_project_dir(project) / "stages"`, creating it if necessary.

#### `get_stage_dir(self, project: str, stage: str) -> Path`
- Returns `get_project_stages_dir(project) / stage`, creating it if necessary.

---

### Project State File Methods

#### `get_canceled_jobs_file(self, project: str) -> Path`
- Returns `get_project_dir(project) / "canceled_jobs.json"`.

#### `get_launched_jobs_file(self, project: str) -> Path`
- Returns `get_project_dir(project) / "launched_jobs.json"`.

---

### Project Config Methods

#### `_apply_project_template(self, obj: Any, project_name: str) -> Any`
- Recursively replaces `{project_name}` in strings, lists, and dicts with the actual `project_name`.
- Non-string/list/dict values pass through unchanged.

#### `ensure_project(self, project: str) -> Dict[str, Any]`
- Creates project directory and stages directory.
- If `project.json` does not exist:
  - Reads global config's `project_defaults` section.
  - Applies template substitution of `{project_name}`.
  - Writes `{"config": <templated_defaults>}` to `project.json`.
  - Returns this dict.
- If `project.json` exists, loads and returns it.

#### `load_project_config(self, project: str) -> Dict[str, Any]`
- Calls `ensure_project(project)` first (ensuring directory exists).
- Reads and returns `project.json`.

#### `save_project_config(self, project: str, project_conf: Dict[str, Any]) -> None`
- Writes `project_conf` to `project.json` with `indent=2`.

---

### Job History Methods (per project)

#### `save_job_info(self, project: str, stage: str, job_info: Dict[str, Any]) -> None`
- Appends `job_info` to `stages/{stage}/jobs.json` (creating file if needed).

#### `load_jobs(self, project: str, stage: Optional[str] = None) -> List[Dict[str, Any]]`
- If `stage` is provided: loads `stages/{stage}/jobs.json`, adds `"stage"` key to each job dict, returns list.
- If `stage` is `None`: iterates all stage directories, loads all `jobs.json` files, adds `"stage"` key from directory name, returns combined list.

---

### Cancel Tracking Methods

#### `load_canceled_jobs(self, project: str) -> set`
- Returns set of job IDs from `canceled_jobs.json`, or empty set if file doesn't exist.

#### `save_canceled_job(self, project: str, job_id: str) -> None`
- Loads existing canceled set, adds `job_id`, saves sorted list back.

---

### Launched Jobs Tracking Methods

#### `load_launched_jobs(self, project: str) -> Dict[str, Any]`
- Returns dict from `launched_jobs.json`, or empty dict if file doesn't exist.

#### `save_launched_jobs(self, project: str, mapping: Dict[str, Any]) -> None`
- Writes `mapping` to `launched_jobs.json`.

---

### Helper Methods

#### `expand_path(self, p: str) -> str`
- Expands `~` and resolves to absolute path string.
