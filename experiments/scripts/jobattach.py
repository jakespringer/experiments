#!/usr/bin/env python3
import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def find_job_info(job_id: str):
    """Search all projects for job information.

    Args:
        job_id: The Slurm job ID to search for

    Returns:
        Tuple of (job_info dict, project_name, stage_name) or (None, None, None) if not found
    """
    experiments_dir = Path.home() / ".experiments"
    projects_dir = experiments_dir / "projects"

    if not projects_dir.exists():
        return None, None, None

    # Search through all projects and stages
    for project_dir in projects_dir.iterdir():
        stages_dir = project_dir / "stages"
        if not project_dir.is_dir():
            continue

        for stage_dir in stages_dir.iterdir():
            if not stage_dir.is_dir():
                continue

            jobs_file = stage_dir / "jobs.json"
            if not jobs_file.exists():
                continue

            try:
                with open(jobs_file, 'r') as f:
                    jobs = json.load(f)

                for job in jobs:
                    if job.get('job_id') == job_id:
                        return job, project_dir.name, stage_dir.name
            except (json.JSONDecodeError, IOError):
                # Skip invalid or unreadable files
                continue

    return None, None, None


def _pattern_for(job_id: str, log_pattern: str, array_id: Optional[str]):
    """Build filename pattern and directory to search."""
    log_path = Path(log_pattern)
    log_dir = log_path.parent
    filename_pattern = log_path.name.replace('%A', job_id)
    if array_id is not None:
        filename_pattern = filename_pattern.replace('%a', array_id)
    else:
        filename_pattern = filename_pattern.replace('%a', '*')
    return log_dir, filename_pattern


def find_log_file(log_pattern: str, job_id: str, array_id: str = None):
    """Find log file(s) matching the pattern."""
    log_dir, filename_pattern = _pattern_for(job_id, log_pattern, array_id)
    if not log_dir.exists():
        return []
    return sorted(log_dir.glob(filename_pattern))


class Follower:
    """Follow a single file for appended lines, with partial-line buffering."""

    def __init__(self, path: Path, from_start: bool, prefix: Optional[str]):
        self.path = path
        self.prefix = prefix
        self._fh = None
        self._pos = 0
        self._buf = ""  # hold partial line across reads
        self._open(from_start)

    def _open(self, from_start: bool):
        # Open in text mode, tolerate encoding glitches
        self._fh = open(self.path, 'r', encoding='utf-8', errors='replace')
        if from_start:
            self._pos = 0
            self._fh.seek(0, os.SEEK_SET)
        else:
            self._fh.seek(0, os.SEEK_END)
            self._pos = self._fh.tell()

    def _maybe_reopen(self):
        """Handle truncation/rotation by re-seeking or reopening."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            # File temporarily missing (rotation). Try to reopen later.
            return
        if size < self._pos:
            # Truncated/rotated
            try:
                self._fh.close()
            except Exception:
                pass
            self._open(from_start=True)

    def read_new(self):
        """Read and print any newly appended lines."""
        if self._fh is None:
            # Try to open if file reappeared
            if self.path.exists():
                self._open(from_start=False)
            else:
                return

        self._maybe_reopen()

        while True:
            chunk = self._fh.read()
            if not chunk:
                # no new data
                break
            self._pos = self._fh.tell()
            self._buf += chunk
            # Emit full lines; keep trailing partial line in buffer
            lines = self._buf.split('\n')
            self._buf = lines.pop()  # last piece (maybe empty if ended on '\n')
            for line in lines:
                if self.prefix:
                    print(f"[{self.prefix}] {line}")
                else:
                    print(line)
                # ensure immediate flush for interactive feel
                sys.stdout.flush()

    def finalize(self):
        """If we end and have a partial line, emit it."""
        if self._buf:
            if self.prefix:
                print(f"[{self.prefix}] {self._buf}")
            else:
                print(self._buf)
            sys.stdout.flush()
            self._buf = ""
        try:
            if self._fh:
                self._fh.close()
        except Exception:
            pass


def follow_logs(job_id: str, log_pattern: str, array_id: Optional[str], from_start: bool, interval: float):
    """Follow one or many log files, optionally discovering new ones over time."""
    log_dir, filename_pattern = _pattern_for(job_id, log_pattern, array_id)

    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}", file=sys.stderr)
        sys.exit(1)

    followers: Dict[Path, Follower] = {}

    def ensure_followers():
        new_paths = sorted(log_dir.glob(filename_pattern))
        for p in new_paths:
            if p not in followers:
                prefix = None
                if len(new_paths) > 1 or array_id is None:
                    # show short filename; for arrays this helps disambiguate
                    prefix = p.name
                try:
                    followers[p] = Follower(p, from_start=from_start, prefix=prefix)
                    # After first one, subsequent discoveries start from end by default
                except IOError as e:
                    print(f"Warning: cannot open {p}: {e}", file=sys.stderr)

    # Initial set
    ensure_followers()

    if not followers:
        # If user targeted a specific array id, fail immediately.
        if array_id is not None:
            print(f"Error: No log files found for job {job_id} (array task {array_id})", file=sys.stderr)
            print(f"Expected pattern: {log_pattern}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"Waiting for logs matching {filename_pattern} in {log_dir} ...", file=sys.stderr)

    # On first pass, we only want --from-start for files opened initially.
    # For subsequently discovered files, force from_start=False.
    from_start = False

    stop = False

    def handle_sigint(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while not stop:
            # Discover new files (useful for array tasks starting later)
            ensure_followers()
            # Read any new data
            for f in list(followers.values()):
                f.read_new()
            time.sleep(interval)
    finally:
        for f in followers.values():
            f.finalize()


def main():
    parser = argparse.ArgumentParser(
        description='Follow Slurm job logs from experiments (like tail -f).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jobattach 12345              # Follow all array tasks for job 12345 as they appear
  jobattach 12345_0            # Follow only array task 0 of job 12345
  jobattach --from-start 12345 # Print existing content, then keep following
"""
    )
    parser.add_argument(
        'job_spec',
        help='Job ID or job_id_arrayindex (e.g., 12345 or 12345_0)'
    )
    parser.add_argument(
        '--from-start', action='store_true',
        help='Print existing file content first (default: start at end and only show new lines)'
    )
    parser.add_argument(
        '--interval', type=float, default=0.5,
        help='Polling interval in seconds (default: 0.5)'
    )

    args = parser.parse_args()

    # Parse job spec
    job_spec = args.job_spec
    array_id = None

    if '_' in job_spec:
        parts = job_spec.split('_', 1)
        job_id = parts[0]
        array_id = parts[1]
    else:
        job_id = job_spec

    # Find job information
    job_info, project_name, stage_name = find_job_info(job_id)

    if job_info is None:
        print(f"Error: Job {job_id} not found in any project", file=sys.stderr)
        print(f"\nSearched in: {Path.home() / '.experiments' / 'projects'}", file=sys.stderr)
        sys.exit(1)

    # Get log file pattern from job info
    log_pattern = job_info.get('log_file')

    if not log_pattern:
        print(f"Error: No log file information found for job {job_id}", file=sys.stderr)
        sys.exit(1)

    if project_name and stage_name:
        print(f"Following logs for job {job_id}  (Project: {project_name}, Stage: {stage_name})", file=sys.stderr)
    else:
        print(f"Following logs for job {job_id}", file=sys.stderr)

    follow_logs(
        job_id=job_id,
        log_pattern=log_pattern,
        array_id=array_id,
        from_start=args.from_start,
        interval=args.interval,
    )


if __name__ == '__main__':
    main()