# Specification: `experiments/scripts/jobcat.py`

## Purpose
Displays Slurm job log file contents (like `cat`). Searches the experiments framework's job history to find log file patterns, resolves them to actual files, and prints their contents.

---

## Functions

### `find_job_info(job_id: str) -> Tuple[Optional[Dict], Optional[str], Optional[str]]`
- Identical implementation to `jobattach.py`'s `find_job_info`.
- Searches `~/.experiments/projects/{project}/stages/{stage}/jobs.json` for matching `job_id`.
- Returns `(job_info_dict, project_name, stage_name)` or `(None, None, None)`.

### `find_log_file(log_pattern: str, job_id: str, array_id: str = None) -> List[Path]`
- Extracts directory and filename pattern from `log_pattern`.
- Replaces `%A` with `job_id` in filename.
- Replaces `%a` with `array_id` (if specified) or `*` (wildcard).
- Returns sorted list of matching files via `glob`, or empty list if directory doesn't exist.

### `print_log_file(log_file: Path, show_header: bool = True) -> bool`
- If `show_header=True`: prints a banner with `=` separators and the filename.
- Reads and prints the entire file content.
- Ensures output ends with a newline.
- Returns `True` on success, `False` on `IOError`.

---

## Function: `main()`

### CLI Arguments
- **Positional**: `job_spec` — Job ID or `job_id_arrayindex` (e.g., `12345` or `12345_0`).

### Behavior
1. Parses `job_spec`: splits on `_` to extract `job_id` and optional `array_id`.
2. Calls `find_job_info(job_id)` to locate job.
3. Extracts `log_file` pattern from job info.
4. Calls `find_log_file(...)` to find matching log files.
5. **If no files found**:
   - Prints error message with expected pattern.
   - If log directory doesn't exist: reports it.
   - Otherwise: lists up to 10 files matching the job ID in the directory (as hints).
   - Exits with code 1.
6. **If files found**:
   - Shows header only if multiple files (`len(log_files) > 1`).
   - Prints project/stage info if available and multiple files.
   - Iterates and prints each file with blank lines between them.
   - Exits with code 0 if all files read successfully, code 1 if any failed.

---

## Important Behaviors
- Unlike `jobattach`, this prints content once and exits (no following).
- Headers with `=` separators are only shown when multiple log files are displayed.
- When a single file is found, it's printed without any header — just raw content.
- Error messages go to stderr; log content goes to stdout.
- Helpful diagnostics when files are not found (directory existence check, glob for similar files).
