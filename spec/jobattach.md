# Specification: `experiments/scripts/jobattach.py`

## Purpose
Follows Slurm job log files in real-time (like `tail -f`). Discovers log files from the experiments framework's job history, supports array jobs, handles truncation/rotation, and can discover new log files as array tasks start.

---

## Functions

### `find_job_info(job_id: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]`
- Searches all projects under `~/.experiments/projects/` for a job matching `job_id`.
- Iterates: `projects_dir / {project} / stages / {stage} / jobs.json`.
- Returns `(job_info_dict, project_name, stage_name)` on match.
- Returns `(None, None, None)` if not found.
- Silently skips invalid/unreadable JSON files.

### `_pattern_for(job_id: str, log_pattern: str, array_id: Optional[str]) -> Tuple[Path, str]`
- Builds a filename pattern and directory from a log pattern string.
- Replaces `%A` with `job_id` in the filename.
- Replaces `%a` with `array_id` if provided, or `*` (glob wildcard) if `None`.
- Returns `(log_dir, filename_pattern)`.

### `find_log_file(log_pattern: str, job_id: str, array_id: str = None) -> List[Path]`
- Finds log files matching the pattern using `log_dir.glob(filename_pattern)`.
- Returns sorted list of matching paths, or empty list if directory doesn't exist.

---

## Class: `Follower`

### Purpose
Follows a single file for appended lines, with partial-line buffering. Handles truncation detection and file reopening.

### Constructor: `__init__(self, path: Path, from_start: bool, prefix: Optional[str])`
- `path`: File to follow.
- `prefix`: Optional prefix prepended to each output line (e.g., filename for disambiguation).
- `from_start`: If `True`, starts reading from beginning; if `False`, seeks to end.
- Initializes `_fh` (file handle), `_pos` (file position), `_buf` (partial line buffer).

### Method: `_open(self, from_start: bool)`
- Opens file in text mode with `encoding='utf-8', errors='replace'`.
- Seeks to beginning or end based on `from_start`.
- Updates `_pos`.

### Method: `_maybe_reopen(self)`
- Detects truncation by comparing current file size to last known position.
- If file is smaller than `_pos`: closes and reopens from start (file was truncated/rotated).
- If file is temporarily missing: does nothing (may reappear later).

### Method: `read_new(self)`
- If file handle is `None` and file exists: opens from end.
- Calls `_maybe_reopen()` to check for truncation.
- Reads all new data in a loop.
- Splits on newlines; emits complete lines immediately with optional prefix.
- Retains trailing partial line in `_buf` for next read.
- Flushes stdout after each line for interactive feel.

### Method: `finalize(self)`
- Emits any remaining partial line in `_buf`.
- Closes file handle.

---

## Function: `follow_logs(job_id, log_pattern, array_id, from_start, interval)`

### Purpose
Main follow loop. Follows one or many log files, discovering new ones over time (useful for array tasks that start at different times).

### Behavior
1. Builds glob pattern via `_pattern_for`.
2. Creates `Follower` instances for all initially matching files.
3. If no files found and `array_id` specified: exits with error.
4. If no files found and no `array_id`: prints waiting message.
5. After initial discovery, sets `from_start = False` for subsequently discovered files.
6. Main loop (until SIGINT):
   - Periodically re-discovers new files via `ensure_followers()`.
   - Calls `read_new()` on each follower.
   - Sleeps for `interval` seconds between polls.
7. On exit: calls `finalize()` on all followers.
8. SIGINT is handled via `signal.signal` to set a `stop` flag for clean shutdown.

### Prefix Behavior
- If multiple files or `array_id is None`: uses filename as prefix for disambiguation.
- If single file with known array_id: no prefix.

---

## Function: `main()`

### CLI Arguments
- **Positional**: `job_spec` — Job ID or `job_id_arrayindex` (e.g., `12345` or `12345_0`).
- `--from-start` — Print existing content first (default: start at end).
- `--interval` — Polling interval in seconds (default: `0.5`).

### Behavior
1. Parses `job_spec`: splits on `_` to extract `job_id` and optional `array_id`.
2. Calls `find_job_info(job_id)` to locate job in project history.
3. Extracts `log_file` pattern from job info.
4. Prints project/stage context to stderr.
5. Calls `follow_logs(...)` with parsed parameters.
6. Exits with error if job not found or no log pattern available.

---

## Important Behaviors
- Uses polling (not inotify) for cross-platform compatibility and simplicity.
- Handles file truncation/rotation by detecting shrinkage and reopening.
- New array task logs are discovered automatically during following.
- All error/status messages go to stderr; log content goes to stdout.
- Partial lines are buffered and not emitted until a newline is received.
