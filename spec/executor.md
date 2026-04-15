# Specification: `experiments/executor.py`

## Purpose
Core execution engine for the `experiments` framework. Defines the task compilation model (Task + TaskBlocks), the abstract Executor with topological ordering, and concrete executors for printing commands (PrintExecutor) and submitting Slurm array jobs (SlurmExecutor).

---

## Helper: `_Progress`

### Purpose
Simple progress status printer to stderr (no external dependencies).

### Constructor: `__init__(self, total: int, desc: str)`
- Stores total and description, renders immediately.

### Methods
- `set(desc)` — Updates description and re-renders.
- `advance(n=1)` — Increments count (capped at total) and re-renders.
- `close()` — Prints newline to stderr.
- `_render()` — Writes `\r{desc} [{count}/{total}]` to stderr.

---

## Function: `dquote(s: str) -> str`
- Double-quotes a string, escaping `\`, `"`, and backticks.
- Does **NOT** escape `$`, allowing environment variable expansion inside the quoted string.

---

## Class: `Directive(ABC)`

### Purpose
Abstract wrapper for executor-specific values that pass through artifact hashing. The inner `_value` is accessed via `__call__()` or proxied via `__getattr__`.

### Constructor: `__init__(self, value: Any)`
- Stores `self._value = value`.

### Methods
- `__str__()` → `str(self._value)`
- `__call__()` → returns `self._value`
- `__getattr__(name)` → proxies to `self._value.__getattr__(name)`
- `__setattr__(name, value)` → proxies to `self._value.__setattr__(name, value)` (except for `_value` itself)

---

## Class: `IgnoreHash(Directive)`

### Purpose
A `Directive` subclass that causes the wrapped value to be **excluded** from artifact hash computation. When `Artifact.get_hash()` encounters an `IgnoreHash` value, `atom()` returns `None`, skipping that field.

---

## Class: `Task`

### Purpose
A compiled unit of work for a single artifact. Holds a list of `TaskBlock`s and metadata about paths.

### Constructor
```python
Task(artifact_path: str, code_path: str, artifact: Artifact | None = None, gs_path: str | None = None)
```
- `self.blocks: List[TaskBlock] = []`
- `self.artifact_path`, `self.gs_path`, `self.code_path`, `self.artifact`

### Methods (all append to `self.blocks`)

| Method | Block Created | Description |
|--------|--------------|-------------|
| `create_file(path, content)` | `CreateFileTaskBlock` | Creates a file with string or bytes content |
| `create_yaml_file(path, content)` | `CreateFileTaskBlock` | Serializes dict to YAML, then creates file |
| `create_json_file(path, content)` | `CreateFileTaskBlock` | Serializes dict to JSON, then creates file |
| `run_command(command, vargs, kwargs, vformat, kwformat, flagformat)` | `CommandTaskBlock` | Runs a shell command with formatted args |
| `upload_to_gs(path, gs_path, directory, contents, no_fail)` | `UploadToGSTaskBlock` | gsutil cp upload |
| `download_from_gs(gs_path, path, directory, skip_existing, contents, no_fail)` | `DownloadFromGSTaskBlock` | gsutil cp download |
| `download(url, local_path, skip_existing)` | `DownloadTaskBlock` | curl download |
| `ensure_directory(path)` | `EnsureDirectoryTaskBlock` | mkdir -p |
| `download_hf_model(model_name, local_dir, skip_existing)` | `DownloadHFModelTaskBlock` | hf download command |
| `rsync_to_gs(path, gs_path, ...)` | `RsyncToGSTaskBlock` | gsutil rsync upload |
| `rsync_from_gs(gs_path, path, ...)` | `RsyncFromGSTaskBlock` | gsutil rsync download |
| `set_env(name, value, from_command)` | `SetEnvTaskBlock` | export variable |

---

## Class: `TaskBlock(ABC)`

### Purpose
Abstract base for a single shell command block.

### Abstract Method: `execute(self) -> str | None`
- Returns a shell command string, or `None` to skip.

---

## TaskBlock Subclasses

### `CommandTaskBlock`
- **Constructor**: `(command, vargs, kwargs, vformat, kwformat, flagformat)`
  - Defaults: `vformat='{v}'`, `kwformat="--{k} '{v}'"`, `flagformat='--{k}'`
- **`execute()`**: Builds command string:
  1. Starts with `self.command`.
  2. Appends positional args formatted with `vformat`.
  3. Appends keyword args: `None` values are omitted; `True` booleans use `flagformat`; `False` booleans are omitted entirely; other values use `kwformat`.

### `CreateFileTaskBlock`
- **Constructor**: `(path, content, mkdirs=True, mode=None)`
- **`execute()`**:
  1. Converts content to bytes (UTF-8 if string).
  2. Base64-encodes the content.
  3. Generates: `mkdir -p parent && base64 -d > path << '___B64___'\n{b64}\n___B64___`
  4. Optionally appends `chmod {mode} path`.

### `UploadToGSTaskBlock`
- **Constructor**: `(path, gs_path, directory=False, contents=True, no_fail=False)`
- **`execute()`**:
  - Computes a lockfile path from SHA-256 of local path.
  - Adjusts source/dest paths based on `contents` flag (trailing slashes/wildcards).
  - For directories: `gsutil -m cp -r src dest`; for files: `gsutil cp src dest`.
  - Wraps in `flock -x lockfile -c "..."` for exclusive locking.
  - If `no_fail`: appends `|| true`.

### `DownloadFromGSTaskBlock`
- **Constructor**: `(gs_path, path, directory=False, skip_existing=True, contents=True, no_fail=False)`
- **`execute()`**:
  - Similar lock-based approach to upload.
  - Creates parent directory with `mkdir -p`.
  - If `skip_existing`: wraps in `if [ ! -e path ]; then ... else echo "Skipping" fi`.
  - The entire if/else block is wrapped in the flock.

### `DownloadTaskBlock`
- **Constructor**: `(url, local_path, mkdirs=True, skip_existing=True)`
- **`execute()`**: `curl -L url -o local_path` with optional `mkdir -p` and skip-existing guard.

### `EnsureDirectoryTaskBlock`
- **Constructor**: `(path)`
- **`execute()`**: `mkdir -p -- "path"`

### `DownloadHFModelTaskBlock`
- **Constructor**: `(model_name, local_dir, mkdirs=True, skip_existing=True)`
- **`execute()`**: `hf download "model_name" --local-dir "local_dir"` with optional mkdir and skip guard.

### `RsyncToGSTaskBlock`
- **Constructor**: `(path, gs_path, delete=False, checksum=False, contents=None, check_exists=False, no_fail=False)`
- **`execute()`**: `gsutil -m rsync -r [-d] [-c] src dest` with flock, optional existence check, and no_fail.

### `RsyncFromGSTaskBlock`
- **Constructor**: `(gs_path, path, delete=False, checksum=False, skip_existing=True, contents=None, check_exists=False, no_fail=False)`
- **`execute()`**: Same pattern as RsyncToGS but reversed direction, with skip_existing guard.

### `SetEnvTaskBlock`
- **Constructor**: `(name, value, from_command=False)`
- **`execute()`**:
  - If `from_command`: `export NAME=$(command)`
  - Else: `export NAME="value"` (with dquote allowing `$VAR` expansion)

---

## Function: `_find_artifact_dependencies(value: Any) -> Iterable[Artifact]`
- Recursively traverses a data structure (dicts, lists, tuples, sets, ArtifactSets) to yield all `Artifact` instances found. Used to build the dependency graph.

---

## Slurm Config Normalization Helpers

### `_to_bool(value)` → Coerces string/int to bool if unambiguous.
### `_to_int_if_numeric(value)` → Converts digit-only strings to int.
### `_norm_key(key)` → Lowercases and replaces hyphens with underscores.
### `_parse_gres_gpu(gres_value)` → Parses GRES strings like `"gpu:2"` or `"gpu:a100:2"` into `{total, types}`.
### `_normalize_slurm_config(raw)` → Comprehensive normalization:
- Lowercases keys, converts hyphens to underscores.
- Coerces booleans and numeric strings.
- Maps `cpus` → `cpus_per_task`.
- Handles `exclusive` with yes/no/user/mcs semantics.
- **Drops `array` key** (managed internally by executor).

---

## Class: `Executor` (Base)

### Purpose
Abstract base executor providing stage registration, topological ordering, artifact filtering, and the execution pipeline. Subclasses implement `compile_artifact()` and `launch()`.

### Constructor: `__init__(self)`
- `self._stages: Dict[str, List[Artifact]] = {}`
- `self._verbose_filtering: bool = True`

### Method: `stage(self, name: str, artifacts: Iterable[Artifact] | ArtifactSet) -> None`
- Registers a named stage. Single `Artifact` instances are auto-wrapped in `ArtifactSet`.

### Method: `auto_cli(self) -> None`
- Parses `sys.argv` for stage names, calls `self.execute(selected)`.

### Method: `execute(self, stages, head=None, tail=None, rerun=False, artifacts=None, jobs=None) -> None`
**Pipeline:**
1. Validates stages exist, normalizes to all if empty.
2. Validates head/tail (mutually exclusive, must be positive).
3. Collects unique artifacts across all stages (deduped by `id()`).
4. Computes topological ordering → `all_tiers`.
5. Filters tiers to selected stages.
6. Optionally filters by artifact class names.
7. Filters out artifacts where `should_skip()` returns `True` (unless `rerun=True`).
8. Reports skipped artifacts to stderr.
9. Applies head/tail filtering (flattens tiers, slices, re-filters).
10. Builds tier-to-stages mapping.
11. Compiles artifacts into tasks: `[[self.compile_artifact(a) for a in tier] for tier in tiers]`.
12. Calls `self.launch(task_tiers, tier_to_stages, jobs)`.

### Method: `compute_topological_ordering(self, artifacts) -> List[List[Artifact]]`
- Uses Kahn's algorithm with layering.
- Returns tiers where tier N has no dependencies on tiers N+1...
- **Dependency detection**: Scans all attributes of each artifact via `_find_artifact_dependencies()`. If an attribute value is (or contains) another `Artifact` in the set, an edge is created.
- **Cycle detection**: Raises `ValueError` if no ready artifacts exist but some remain.
- **Order preservation**: Within a tier, artifacts appear in original input order.

### Abstract Method: `launch(self, tiers, tier_to_stages=None, jobs=None) -> None`
### Abstract Method: `compile_artifact(self, artifact) -> Task`

### Internal Helper Methods
- `_validate_and_normalize_stages(stages)` — Returns validated stage list.
- `_collect_unique_artifacts()` — Dedupes by `id()`.
- `_build_artifact_to_stages_mapping()` — Returns `Dict[int, Set[str]]`.
- `_filter_tiers_by_stages(tiers, selected_stages)` — Filters tiers.
- `_filter_tiers_by_artifact_class(tiers, artifact_classes)` — Filters by class name.
- `_filter_skipped_artifacts(tiers, rerun)` — Returns `(executable_tiers, skipped_list)`.
- `_apply_head_tail_filter(tiers, head, tail)` — Flattens, slices, re-tiers.
- `_build_tier_to_stages_mapping(tiers, selected_stages)` — Returns `Dict[int, List[str]]`.

---

## Class: `PrintExecutor(Executor)`

### Purpose
Prints all shell commands to stdout. Useful for debugging or piping to `bash` directly.

### Constructor
```python
PrintExecutor(artifact_path=None, code_path=None, gs_path=None, setup_command=None)
```
- Reads project/global config from `ConfigManager` to fill defaults.
- Sets `_verbose_filtering = False`.

### `compile_artifact(artifact)` → Creates `Task`, calls `artifact.construct(task)`, returns task.

### `launch(tiers, ...)` → Prints:
1. `#!/usr/bin/env bash`
2. Setup command (if any)
3. `set -euo pipefail`
4. `export EXPERIMENTS_PROJECT_CONF="..."`
5. For each task: `export EXPERIMENTS_EXPERIMENT_CONF="..."` + all block commands.

---

## Class: `SlurmExecutor(Executor)`

### Purpose
Submits tasks as Slurm array jobs. Each tier becomes one or more array jobs. Tasks with different resource requirements within a tier are submitted as separate jobs. Dependencies between tiers are tracked via `--dependency=afterok:...`.

### Constructor
```python
SlurmExecutor(artifact_path=None, code_path=None, project=None, gs_path=None,
              default_slurm_args=None, dry_run=False, setup_command=None)
```
- Initializes `Project.init(project)` if needed.
- Reads global/project config for defaults.
- Attributes include:
  - `dry_run`, `setup_command`, `external_dependencies`, `cli_slurm_overrides`
  - `force_launch`, `array_throttle`, `split_jobs`
  - `_active_jobs`, `_launched_map`, `recent_job_ids`

### `auto_cli()` → Delegates to `cli.auto_cli(self)`.

### `compile_artifact(artifact)` → Same as `PrintExecutor`.

### `launch(tiers, tier_to_stages=None, jobs=None)`
**Pipeline:**
1. Pre-computes total group count for progress bar.
2. Prints banner (DRY RUN or actual).
3. Scans running jobs via `squeue` → `_active_jobs`.
4. Loads and prunes launched jobs mapping.
5. For each tier:
   - Calls `_submit_tier(tier_index, tier, artifact_to_job_id, stage_names, jobs, progress)`.
   - Updates `artifact_to_job_id` with new mappings.
6. Prints dry-run summary or launch summary.
7. Persists launched mapping.
8. Stores `recent_job_ids`.

### `_submit_tier(...)` → Returns `Dict[int, str]` (artifact_id → job_id)
1. Groups tasks by requirements (`_group_tasks_by_requirements`).
2. For each group:
   - Filters out already-running tasks (unless `force_launch`).
   - Optionally splits into chunks if `split_jobs` is set.
   - For each chunk: builds sbatch header, builds script body, writes to tempfile, submits via `sbatch --parsable`.
   - Saves job info to `ConfigManager`.
   - Updates launched map.

### Slurm Config Resolution (`_build_slurm_config`)
**Priority order (lowest to highest):**
1. `default_slurm_args` from executor constructor
2. Global config `default_slurm_args["*"]` (wildcard)
3. Global config `default_slurm_args["{partition}"]` (partition-specific)
4. Project config `default_slurm_args["*"]`
5. Project config `default_slurm_args["{partition}"]` or flat key-value pairs
6. Artifact `get_requirements()` return value
7. CLI `--slurm` overrides

Then: normalization, default partition fallback, cpus_per_task inference from GPUs.

### sbatch Script Generation (`_build_sbatch_header` + `_build_script_body`)

**Header** includes `#SBATCH` directives for:
- `--job-name`, `--output`, `--error` (optional), `--chdir`
- `--dependency=afterok:...` (from dependency graph + external deps)
- `--partition`, `--time`, `--time-min`, `--account`, `--qos`
- `--nodes`, `--ntasks`, `--cpus-per-task`
- `--mem`, `--mem-per-cpu`
- `--gpus`, `--gres`, `--constraint`
- `--exclude`, `--nodelist`
- `--requeue`, `--signal`, `--open-mode`
- `--mail-type`, `--mail-user`, `--exclusive`
- `--array=0-N[%throttle]`
- Generic pass-through for unknown keys.

**Body** structure:
```bash
{setup_command}
set -euo pipefail
export EXPERIMENTS_PROJECT_CONF="..."
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    export EXPERIMENTS_EXPERIMENT_CONF="..."
    {commands for task 0}
    ;;
  1)
    ...
    ;;
  *)
    echo "Invalid SLURM_ARRAY_TASK_ID" >&2
    exit 1
    ;;
esac
```

**When `jobs` limit is specified** and fewer than total tasks:
- Tasks are distributed round-robin across `jobs` groups.
- `set +e` is used within grouped cases (so one failure doesn't stop the group).
- Each group runs its tasks sequentially.

### Running Job Tracking
- `_get_active_jobs()` → Runs `squeue -u $USER -o "%A %a %T"`, returns `Set[(job_id, array_index)]`.
- `_load_launched_jobs()` / `_save_launched_jobs()` → Delegates to `ConfigManager`.
- `_prune_launched_jobs()` → Removes entries not in active jobs set.
- `_should_launch_task(task)` → Returns `False` if task's artifact relpath is already mapped to an active job.

### `VALID_REQUIREMENT_KEYS`
Set of allowed keys from `get_requirements()`:
```python
{'output', 'error', 'separate_error', 'partition', 'time', 'time_min', 'account',
 'qos', 'chdir', 'nodes', 'ntasks', 'cpus', 'cpus_per_task', 'mem', 'mem_per_cpu',
 'gpus', 'gres', 'constraint', 'exclude', 'nodelist', 'requeue', 'signal',
 'open_mode', 'mail_type', 'mail_user'}
```

---

## Module-Level Helpers

### `_is_json_scalar(x)` → `True` for str/int/float/bool/None.
### `_json_sanitize(obj)` → Recursively strips non-serializable values.
### `_safe_json_dumps(obj)` → Sanitize then `json.dumps` with compact separators. Returns `"{}"` on failure.
### `_artifact_experiment_conf(artifact)` → Builds a JSON-safe dict from `as_dict()` + all `@property` values + `relpath`.
