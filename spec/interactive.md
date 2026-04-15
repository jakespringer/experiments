# Specification: `experiments/scripts/interactive.py`

## Purpose
Submits an interactive Slurm job with sensible defaults. Generates an sbatch script that runs a `sleep` command for the duration of the job, allowing users to attach with `srun --jobid=JOBID --pty bash`.

---

## Module-Level Constants

### `EXCLUDED_NODES`
- A list of node names to always exclude from scheduling.
- Default: empty list (with commented-out examples).
- Editable by the user to exclude problematic nodes.

---

## Functions

### `parse_gpus(gpu_arg) -> Tuple[Optional[str], int]`
- Parses a GPU argument which can be an integer or `"TYPE:COUNT"` string.
- **Returns** `(gpu_string_for_sbatch, gpu_count_int)`.
- Examples:
  - `"4"` → `("4", 4)`
  - `"H100:4"` → `("H100:4", 4)`
  - `None` → `(None, 0)`
  - `"H100"` (just type, no count) → `("H100", 1)`

### `parse_args() -> argparse.Namespace`
- Parses command-line arguments:
  - `--partition/-p` (default: `"general"`)
  - `--time/-t` (default: `None`; resolved later based on partition)
  - `--gpus/-g` (default: `None`; supports `int` or `TYPE:COUNT`)
  - `--cpus/-c` (default: `None`; resolved to GPUs + 2)
  - `--mem` (default: `None`; resolved to `64G + 16G × GPU_count`)
  - `--job-name/-J` (default: `"i"`)
  - `--account/-A` (default: `None`)
  - `--qos` (default: `None`)
  - `--constraint/-C` (default: `None`)
  - `--nodelist/-w` (default: `None`)
  - `--exclude/-x` (default: `None`; combined with `EXCLUDED_NODES`)
  - `--dry-run/-n` (flag; print without executing)

### `time_to_seconds(time_str) -> int`
- Converts Slurm time format (`D-HH:MM:SS` or `HH:MM:SS` or `MM:SS` or `SS`) to total seconds.
- Supports optional `D-` day prefix separated by hyphen.

### `build_sbatch_script(args) -> str`
- Builds the complete sbatch script as a string.
- **Default resolution**:
  - Time: `"0-02:00:00"` for `debug` partition, `"2-00:00:00"` otherwise.
  - CPUs: `gpu_count + 2`.
  - Memory: `64 + (16 × gpu_count)` GB.
- **Script structure**:
  1. Shebang line.
  2. `#SBATCH` directives (job-name, partition, time, cpus-per-task, gpus, mem, optional account/qos/constraint/nodelist/exclude, nodes=1, output=/tmp/{job_name}_%j.log).
  3. Echo statements printing hostname, date, job ID, and connection instructions.
  4. `sleep {seconds}` command for the entire time limit.
- Exclude list: merges `EXCLUDED_NODES` with user `--exclude` argument.

### `main() -> int`
- Parses args, builds script, prints script to stderr.
- If `--dry-run`: prints `[Dry run - not executing]` and returns 0.
- Otherwise: submits via `sbatch` (piping script to stdin).
- On success: extracts job ID from stdout, prints connection instructions to stderr.
- On `KeyboardInterrupt`: prints `Interrupted.` and returns 130.
- Returns sbatch's exit code.

---

## Important Behaviors
- All informational output goes to stderr; only sbatch's stdout goes to stdout.
- The job itself just sleeps — it's a placeholder for interactive use via `srun --pty`.
- Output log is written to `/tmp/{job_name}_%j.log`.
- Always requests `--nodes=1` (single-node only).
