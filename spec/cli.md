# Specification: `experiments/cli.py`

## Purpose
Implements the command-line interface for experiment management. Provides subcommands for launching, canceling, relaunching, inspecting logs, viewing history, printing commands, and exporting experiment data.

---

## Function: `auto_cli(executor: SlurmExecutor) -> None`
- Creates an `ExperimentCLI` instance and calls `cli.run()`.
- This is the main entry point for CLI usage.

---

## Class: `ExperimentCLI`

### Constructor: `__init__(self, executor: SlurmExecutor)`
- Stores `self.executor`.
- Creates `self.config_manager = ConfigManager()`.
- Loads global config via `self.config_manager.ensure_config()`.
- Initializes `Project` if executor has a project name and Project isn't already initialized.

---

### Method: `run(self) -> None`
Parses `sys.argv` using `argparse` with subcommands:

#### Subcommand: `launch`
- **Positional**: `stages` (optional, 0 or more)
- **Options**: `--head N`, `--tail N`, `--rerun`, `--reverse`, `--exclude STAGE...`, `--artifact ARTIFACT...`, `--jobs N`, `--dependency JOBID...`, `--slurm KEY=VALUE...`, `--force-launch`, `--throttle N`, `--splitjobs N`

#### Subcommand: `drylaunch`
- Same options as `launch`. Calls `self.launch(...)` with `dry_run=True`.

#### Subcommand: `relaunch`
- Same options as `launch` + `--cancelafter`.
- Cancels then launches (or launches then cancels if `--cancelafter`).

#### Subcommand: `cancel`
- **Positional**: `stages` (optional, 0 or more)

#### Subcommand: `cat`
- **Positional**: `job_spec` (e.g., "12345" or "12345_0"), optional `array_index`.

#### Subcommand: `history`
- No arguments.

#### Subcommand: `print`
- **Positional**: `stages` (optional)
- **Options**: `--head`, `--tail`, `--rerun`, `--reverse`, `--exclude`, `--artifact`, `--jobs`

#### Subcommand: `printlines`
- Same as `print` + `--output-dir DIR`.

#### Subcommand: `export`
- **Positional**: `stages` (optional)
- **Options**: `--artifact ARTIFACT...`, `--check_exists`, `--exists`, `-f/--file FILE` (required)

---

### Method: `launch(self, stages, dry_run=False, head=None, tail=None, rerun=False, reverse=False, exclude=None, artifacts=None, jobs=None, dependency=None, slurm_overrides=None, force_launch=False, throttle=None, split_jobs=None)`

**Behavior:**
1. Sets `executor.dry_run`.
2. Parses `--slurm` overrides (KEY=VALUE format) with:
   - Key normalization (lowercase, hyphens → underscores).
   - Boolean coercion for recognized true/false strings.
   - Integer coercion for numeric values.
   - **Rejects `array=...`** with error.
3. Handles throttle (CLI `--throttle` > `--slurm throttle=N`).
4. Sets executor attributes: `cli_slurm_overrides`, `force_launch`, `array_throttle`, `split_jobs`, `external_dependencies`.
5. Computes selected stages (provided or all).
6. Applies `--exclude` and `--reverse`.
7. Calls `executor.execute(selected, head, tail, rerun, artifacts, jobs)`.

---

### Method: `cancel(self, stages, exclude_job_ids=None)`

**Behavior:**
1. Determines stages to cancel (provided or all).
2. Loads jobs for each stage from `ConfigManager`.
3. Loads already-canceled job set.
4. Filters: `job_ids - canceled_jobs - exclude_job_ids`.
5. Runs `scancel job_id` for each job.
6. Marks successful cancellations in `ConfigManager`.
7. Prints summary.

---

### Method: `relaunch(self, stages, ..., cancel_after=False)`

**Behavior (default: `cancel_after=False`):**
1. Computes selected stages (with exclude/reverse).
2. Cancels selected stages.
3. Launches with same parameters.

**Behavior (`cancel_after=True`):**
1. Launches first.
2. Gets `recent_job_ids` from executor.
3. Cancels selected stages, excluding the newly launched job IDs.

---

### Method: `cat(self, job_spec, array_index=None)`
- Parses `job_spec` (splits on `_` for array index).
- Finds log files in log directory matching pattern `*_{job_id}_{array_index}.log`.
- Prints all matching log file contents.

---

### Method: `history(self)`
- Iterates all stage directories under the project.
- Loads `jobs.json` from each, deduplicates by job_id.
- Sorts by timestamp (oldest first).
- Prints formatted table: Job ID, Job Name, Stage, Tasks, Submitted.

---

### Method: `print_commands(self, stages, ...)`
- Creates a `PrintExecutor` with same paths as `SlurmExecutor`.
- Copies `_stages` from the Slurm executor.
- Calls `print_executor.execute(...)` to output commands to stdout.

---

### Method: `print_lines(self, stages, ..., output_dir=None)`
- Creates individual `.sh` script files (one per task).
- Each script contains: shebang, setup command, `set -euo pipefail`, project config export, experiment config export, task commands.
- Scripts are made executable (`chmod 0o755`).
- Prints `bash /path/to/job_NNNN.sh` lines to stdout.
- If `--output-dir` not provided, creates temp dir in `/tmp`.

---

### Method: `export(self, stages, artifacts=None, exists_only=False, check_exists=False, output_file=None)`

**Behavior:**
1. Validates: `output_file` required; `--exists` requires `--check_exists`.
2. Collects unique artifacts from selected stages.
3. Optionally filters by artifact class names.
4. If `check_exists`: evaluates `artifact.exists` for each (with progress bar).
5. If `exists_only`: filters to only existing artifacts.
6. For each artifact, builds export dict:
   - `stage`, `artifact_type`, `hash`
   - All fields from `as_dict()` (converted to JSON-safe values).
   - `exists` (if checked).
   - All `@property` values from the artifact class.
   - Artifact references formatted as `"ClassName(hash)"`.
7. Builds final export: `{project_config, global_config, artifacts}`.
8. Writes to local file or uploads to `gs://` path.

---

## Important Behaviors

### `--slurm` Override Parsing
- Format: `KEY=VALUE` pairs.
- `array` is explicitly forbidden (managed by executor).
- Values are coerced: `"true"/"yes"/"y"/"1"` → `True`, `"false"/"no"/"n"/"0"/""` → `False`, digit strings → `int`.

### Stage Selection
- If no stages specified, all executor stages are used.
- `--exclude` removes stages by name.
- `--reverse` reverses the stage order.

### Artifact Reference Format in Export
- Artifact values in `as_dict()` that are `Artifact` instances → `"ClassName(hash)"`.
- `ArtifactSet` values → list of `"ClassName(hash)"` strings.
