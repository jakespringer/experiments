# Specification: `experiments/scripts/batch_local.py`

## Purpose
GPU Job Runner — a terminal dashboard for parallel GPU job execution. Reads commands from stdin and executes them in parallel across GPUs with a live-updating Rich dashboard showing progress, timing, job status, and live output.

## Dependencies
- `argparse`, `os`, `sys`, `threading`, `queue`, `subprocess`, `time`, `select`, `fcntl`, `tty`, `termios`
- `dataclasses`, `datetime`, `enum`, `typing`, `collections.deque`
- `torch` (for `torch.cuda.device_count()`)
- `rich` (Console, Live, Table, Panel, Layout, Text, Style, Align, box)

---

## Module-Level Constants

### `COLORS`
- Dark color scheme dict with keys: `bg`, `border`, `border_active`, `text`, `text_dim`, `text_bright`, `accent`, `success`, `error`, `warning`, `running`.
- Uses black and dark gray tones.

---

## Enum: `JobStatus`
- `PENDING = "pending"`
- `RUNNING = "running"`
- `COMPLETED = "completed"`
- `ERRORED = "errored"`

---

## Dataclass: `Job`

### Fields
- `index: int` — Job index.
- `command: str` — Shell command string.
- `status: JobStatus` — Default `PENDING`.
- `start_time: Optional[datetime]` — Default `None`.
- `end_time: Optional[datetime]` — Default `None`.
- `exit_code: Optional[int]` — Default `None`.
- `worker_id: Optional[int]` — Default `None`.
- `gpus: str` — GPU assignment string (default `""`).

### Properties
- `duration -> Optional[timedelta]`: Elapsed time (uses `datetime.now()` if still running).
- `duration_str -> str`: Formatted duration string (`Xh XXm XXs`, `Xm XXs`, or `Xs`).
- `short_command -> str`: Command truncated to 50 characters with `...` suffix if longer.

---

## Dataclass: `JobManager`

### Purpose
Thread-safe job state manager. Maintains job list, tracks output for the currently selected job, and provides statistics.

### Fields
- `jobs: list[Job]` — All jobs.
- `lock: threading.RLock` — Protects job state.
- `start_time: Optional[datetime]` — Set when first job starts.
- `completed_durations: list[float]` — Durations (seconds) of completed jobs.
- `current_output_job: Optional[int]` — Index of job whose output is currently displayed.
- `output_lines: deque` — Rolling buffer of output lines (maxlen=500).
- `output_lock: threading.Lock` — Protects output state.

### Methods

#### `add_job(self, index, command)`
- Appends a new `Job` under `self.lock`.

#### `get_job(self, index) -> Optional[Job]`
- Finds job by index under `self.lock`.

#### `start_job(self, index, worker_id, gpus)`
- Sets job to `RUNNING` with timestamp and worker/GPU info.
- Sets `start_time` on first job start.
- Auto-selects first running job for output display if none selected.

#### `complete_job(self, index, exit_code)`
- Under `self.lock`:
  - Sets status to `COMPLETED` (exit_code=0) or `ERRORED`.
  - Records duration.
  - Picks next running job (sorted by worker_id, then index) for output display.
- Under `self.output_lock`:
  - If completed job was current output: switches to next running job.

#### `append_output(self, job_index, line)`
- Under `self.output_lock`: appends line if `job_index` matches `current_output_job`.

#### `get_output_lines(self, max_lines=50) -> Tuple[Optional[int], list[str]]`
- Returns `(current_output_job, last_N_lines)`.

#### `switch_to_job(self, job_index, output_dir)`
- Under `self.output_lock`: sets new current job, clears output buffer, loads existing output from `{output_dir}/{job_index}.txt`.

#### `cycle_job(self, direction, output_dir)`
- Cycles through running jobs by `direction` (+1 next, -1 previous).
- Sorts running jobs by `worker_id`.
- Wraps around at boundaries.
- Loads output from file for the new job.

### Properties
- `stats -> dict`: Returns `{total, pending, running, completed, errored, finished}`.
- `running_jobs -> list[Job]`: Jobs with status `RUNNING`.
- `finished_jobs -> list[Job]`: Jobs with status `COMPLETED` or `ERRORED`, sorted by `end_time` descending.
- `elapsed_time -> timedelta`: Time since first job started.
- `estimated_total_time -> Optional[timedelta]`: Estimates total time based on average completed duration, accounting for running and pending jobs divided by parallelism.

---

## Class: `Dashboard`

### Purpose
Rich-based live dashboard with split-pane layout showing progress, running jobs, completed jobs, and live output.

### Constructor: `__init__(self, manager, output_path, num_parallel, console)`
- Sets up a two-column `Layout`:
  - **Left**: header (3 lines), progress (10 lines), running (flex), completed (flex).
  - **Right**: output panel (full height).

### Methods

#### `get_output_panel_height(self) -> int`
- Returns `max(terminal_height - 5, 10)`.

#### `format_timedelta(self, td) -> str`
- Formats timedelta as `H:MM:SS` or `MM:SS`.

#### `create_header(self) -> Panel`
- Title: "GPU JOB RUNNER" with worker count.

#### `create_progress_panel(self) -> Panel`
- Progress bar (`█`/`░`), percentage, stats (total/pending/running/done/failed), elapsed/estimated times, output path.

#### `create_running_table(self) -> Panel`
- Table with columns: selection marker (`>`), JOB index, Worker ID, GPU assignment, elapsed TIME, CMD (truncated).
- Current output job highlighted with bold styling.
- Title: "running (up/down to switch)".

#### `create_finished_table(self) -> Panel`
- Table showing last 15 finished jobs with: JOB index, STATUS (OK/FAIL), EXIT code, TIME, CMD.

#### `create_output_panel(self) -> Panel`
- Shows live output from the currently selected job.
- Auto-scrolls by showing only the last N lines that fit.
- Truncates long lines to half the terminal width.
- Shows "Waiting for jobs to start..." or "Waiting for output..." when appropriate.

#### `generate(self) -> Layout`
- Updates all layout sections and returns the layout.

---

## Functions

### `parse_arguments() -> argparse.Namespace`
- `--parallel`: Number of parallel commands (required if `--gpus` not specified).
- `--gpus`: GPUs per job (if specified, `--parallel` is inferred from available GPUs).
- `--first-gpu`: First GPU index to use (default: 0).
- `--output`: Output directory for stdout/stderr files.
- `--refresh-rate`: Dashboard refresh rate in seconds (default: 0.1).

### `read_commands() -> List[Tuple[int, str]]`
- Reads commands from stdin, one per line.
- Skips blank lines and lines starting with `#`.
- Returns list of `(line_index, command_string)`.

### `worker(worker_id, first_gpu, num_gpus, job_queue, output_dir, manager)`
- Worker thread function.
- Computes GPU assignment: `first_gpu + worker_id * num_gpus` through `first_gpu + (worker_id + 1) * num_gpus - 1`.
- Processes jobs from queue until empty.
- For each job:
  1. Calls `manager.start_job(...)`.
  2. Sets `CUDA_VISIBLE_DEVICES` and `PYTHONUNBUFFERED=1`.
  3. Runs command via `subprocess.Popen` with `shell=True`, `stdout=PIPE`, `stderr=STDOUT`.
  4. Sets stdout to non-blocking mode via `fcntl`.
  5. Polls for output using `select` with 50ms timeout.
  6. Writes output to both file and `manager.append_output`.
  7. Drains remaining output after process terminates.
  8. Writes header/footer with metadata to output file.
  9. Calls `manager.complete_job(...)`.

### `create_output_directory(output_path: Optional[str]) -> str`
- If `output_path` is `None`: uses `/home/jspringe/slurm/local_outputs/{N+1}` (auto-incrementing).
- Creates directory with `os.makedirs(..., exist_ok=True)`.
- Raises `RuntimeError` if default base doesn't exist and no `--output` specified.

### `main()`
1. Validates that either `--parallel` or `--gpus` is specified.
2. If `--gpus > 0`: detects available GPUs via `torch.cuda.device_count()`.
   - Infers `num_parallel = (available_gpus - first_gpu) // gpus`.
   - If `--parallel` also given: uses `min(parallel, inferred)`.
3. Reads commands from stdin.
4. Creates output directory and `JobManager`.
5. Starts worker threads.
6. Sets up keyboard input via `/dev/tty` (non-blocking, raw mode):
   - Up arrow: cycle to previous running job.
   - Down arrow: cycle to next running job.
7. Runs Rich `Live` display loop:
   - Updates dashboard, checks keyboard, sleeps for refresh rate.
   - Continues until all threads finish.
8. Restores terminal settings in `finally`.
9. Joins threads with 1-second timeout.
10. Prints final summary: completed/failed/time/output path.
11. Exits with code 1 if any jobs errored, 0 otherwise.

---

## Important Behaviors
- **Thread safety**: Job state protected by `RLock`, output state by separate `Lock`. Locking order: never hold `output_lock` while acquiring `self.lock`.
- **Non-blocking I/O**: Worker uses `fcntl` to set subprocess stdout to non-blocking, reads with `select` for responsive output.
- **Keyboard input**: Reads from `/dev/tty` directly (since stdin is used for commands pipe). Falls back gracefully if no controlling terminal.
- **Output files**: Each job writes to `{output_dir}/{index}.txt` with header/footer metadata.
- **GPU assignment**: Sequential GPU blocks — worker N gets GPUs `[first_gpu + N*gpus_per_job, ...)`.
