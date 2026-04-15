#!/usr/bin/env python3
"""
GPU Job Runner - A terminal dashboard for parallel GPU job execution.

Pipe commands via stdin and watch them execute in parallel across GPUs with
a live-updating dashboard showing progress, timing, and job status.
"""

import argparse
import os
import re
import sys
import threading
import queue
import subprocess
import time
import select
import fcntl
import tty
import termios
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
from collections import deque
import torch

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.style import Style
from rich.align import Align
from rich import box

# Strip ANSI escapes from subprocess output so they don't corrupt Rich rendering
_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]|\x1b[()][AB012]|\x1b\][^\x07]*\x07|\r')

# Dark color scheme - black and dark grays only
COLORS = {
    "bg": "black",
    "border": "grey15",
    "border_active": "grey23",
    "text": "grey37",
    "text_dim": "grey19",
    "text_bright": "grey46",
    "accent": "dark_cyan",
    "success": "green4",
    "error": "red3",
    "warning": "orange4",
    "running": "dodger_blue2",
    "yielding": "dark_goldenrod",
    "selected_bg": "light_sky_blue1",
    "selected_fg": "black",
}


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    YIELDING = "yielding"
    COMPLETED = "completed"
    ERRORED = "errored"


@dataclass
class Job:
    index: int
    command: str
    status: JobStatus = JobStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    exit_code: Optional[int] = None
    worker_id: Optional[int] = None
    gpus: str = ""

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time is None:
            return None
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def duration_str(self) -> str:
        d = self.duration
        if d is None:
            return "-"
        total_seconds = int(d.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h{minutes:02d}m{seconds:02d}s"
        elif minutes > 0:
            return f"{minutes}m{seconds:02d}s"
        else:
            return f"{seconds}s"

    def short_command(self, max_len: int = 50) -> str:
        if len(self.command) <= max_len:
            return self.command
        return self.command[: max_len - 3] + "..."


@dataclass
class JobManager:
    jobs: list[Job] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    start_time: Optional[datetime] = None
    completed_durations: list[float] = field(default_factory=list)
    current_output_job: Optional[int] = None
    output_lines: deque = field(default_factory=lambda: deque(maxlen=500))
    output_lock: threading.Lock = field(default_factory=threading.Lock)
    scroll_offset: int = 0  # 0 = follow tail, >0 = scrolled up N lines from bottom

    def add_job(self, index: int, command: str):
        with self.lock:
            self.jobs.append(Job(index=index, command=command))

    def get_job(self, index: int) -> Optional[Job]:
        with self.lock:
            for job in self.jobs:
                if job.index == index:
                    return job
        return None

    def start_job(self, index: int, worker_id: int, gpus: str):
        with self.lock:
            if self.start_time is None:
                self.start_time = datetime.now()
            for job in self.jobs:
                if job.index == index:
                    job.status = JobStatus.RUNNING
                    job.start_time = datetime.now()
                    job.worker_id = worker_id
                    job.gpus = gpus
                    break
        # Auto-select first running job for output display
        with self.output_lock:
            if self.current_output_job is None:
                self.current_output_job = index
                self.output_lines.clear()
                self.scroll_offset = 0

    def yield_job(self, index: int):
        """Mark a job as yielding — GPU work done, finalizing."""
        with self.lock:
            for job in self.jobs:
                if job.index == index:
                    job.status = JobStatus.YIELDING
                    break

    def complete_job(self, index: int, exit_code: int):
        # Update job state under self.lock, and compute "next" while still under self.lock.
        next_running_index: Optional[int] = None
    
        with self.lock:
            for job in self.jobs:
                if job.index == index:
                    job.status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.ERRORED
                    job.end_time = datetime.now()
                    job.exit_code = exit_code
                    if job.duration:
                        self.completed_durations.append(job.duration.total_seconds())
                    break
    
            # Pick a deterministic next active job (by worker_id then index)
            running = [j for j in self.jobs if j.status in (JobStatus.RUNNING, JobStatus.YIELDING)]
            running.sort(key=lambda j: (j.worker_id is None, j.worker_id or 0, j.index))
            next_running_index = running[0].index if running else None
    
        # Now touch output state under output_lock (and do NOT call anything that grabs self.lock here).
        with self.output_lock:
            if self.current_output_job == index:
                self.current_output_job = next_running_index
                self.output_lines.clear()
                self.scroll_offset = 0

    def append_output(self, job_index: int, line: str):
        with self.output_lock:
            if self.current_output_job == job_index:
                self.output_lines.append(line)
                # Keep scroll position stable when scrolled up
                if self.scroll_offset > 0:
                    self.scroll_offset += 1

    def scroll_output(self, delta: int, page_size: int = 1):
        """Scroll output view. Positive delta = scroll up (back), negative = scroll down (forward)."""
        with self.output_lock:
            total = len(self.output_lines)
            new_offset = self.scroll_offset + delta
            self.scroll_offset = max(0, min(new_offset, max(0, total - page_size)))

    def get_output_lines(self, max_lines: int = 50) -> tuple[Optional[int], list[str]]:
        with self.output_lock:
            all_lines = list(self.output_lines)
            total = len(all_lines)
            if self.scroll_offset == 0 or self.scroll_offset >= total:
                return self.current_output_job, all_lines[-max_lines:]
            end = total - self.scroll_offset
            start = max(0, end - max_lines)
            return self.current_output_job, all_lines[start:end]

    def switch_to_job(self, job_index: int, output_dir: str):
        """Switch to monitoring a specific job, loading its output from file."""
        with self.output_lock:
            self.current_output_job = job_index
            self.output_lines.clear()
            self.scroll_offset = 0
            # Load existing output from file
            output_file = os.path.join(output_dir, f"{job_index}.txt")
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        for line in f:
                            self.output_lines.append(line.rstrip('\n'))
                except Exception:
                    pass

    @property
    def stats(self) -> dict:
        with self.lock:
            total = len(self.jobs)
            pending = sum(1 for j in self.jobs if j.status == JobStatus.PENDING)
            running = sum(1 for j in self.jobs if j.status == JobStatus.RUNNING)
            yielding = sum(1 for j in self.jobs if j.status == JobStatus.YIELDING)
            completed = sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)
            errored = sum(1 for j in self.jobs if j.status == JobStatus.ERRORED)
            return {
                "total": total,
                "pending": pending,
                "running": running,
                "yielding": yielding,
                "completed": completed,
                "errored": errored,
                "finished": completed + errored,
            }

    @property
    def running_jobs(self) -> list[Job]:
        with self.lock:
            return [j for j in self.jobs if j.status in (JobStatus.RUNNING, JobStatus.YIELDING)]

    @property
    def finished_jobs(self) -> list[Job]:
        with self.lock:
            return sorted(
                [j for j in self.jobs if j.status in (JobStatus.COMPLETED, JobStatus.ERRORED)],
                key=lambda j: j.end_time or datetime.now(),
                reverse=True,
            )

    @property
    def elapsed_time(self) -> timedelta:
        if self.start_time is None:
            return timedelta(0)
        return datetime.now() - self.start_time

    @property
    def estimated_total_time(self) -> Optional[timedelta]:
        with self.lock:
            if not self.completed_durations:
                return None
    
            total = len(self.jobs)
            pending = sum(1 for j in self.jobs if j.status == JobStatus.PENDING)
            running = sum(1 for j in self.jobs if j.status == JobStatus.RUNNING)
            completed = sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)
            errored = sum(1 for j in self.jobs if j.status == JobStatus.ERRORED)
            finished = completed + errored
    
            if finished == 0:
                return None
    
            avg_duration = sum(self.completed_durations) / len(self.completed_durations)
            running_contribution = running * avg_duration * 0.5
            pending_contribution = pending * avg_duration
            parallel = max(running, 1)
            estimated_remaining = (running_contribution + pending_contribution) / parallel
    
            start_time = self.start_time
    
        # compute elapsed outside the lock
        elapsed = (datetime.now() - start_time) if start_time else timedelta(0)
        return elapsed + timedelta(seconds=estimated_remaining)

class Dashboard:
    def __init__(self, manager: JobManager, output_path: str, num_parallel: int, console: Console):
        self.manager = manager
        self.output_path = output_path
        self.num_parallel = num_parallel
        self.console = console
        self.running_scroll_top = 0
        self.completed_scroll_top = 0

    def _running_panel_size(self) -> int:
        """Dynamic size for the running panel based on num_parallel."""
        needed = self.num_parallel + 3  # 2 borders + 1 header + num_parallel rows
        max_size = max((self.console.size.height - 13) // 2, 5)
        return max(4, min(needed, max_size))

    def _visible_rows(self, panel: str) -> int:
        """Number of job rows visible in a panel (excluding borders and header)."""
        if panel == "running":
            return self._running_panel_size() - 3
        else:
            used = 13 + self._running_panel_size()
            remaining = self.console.size.height - used
            return max(remaining - 3, 1)

    def navigate_job(self, direction: int):
        """Cycle through jobs: running first, then finished. direction: -1 for up, 1 for down."""
        running = self._get_sorted_running()
        finished = self._get_sorted_finished()
        navigable = [j.index for j in running] + [j.index for j in finished]
        if not navigable:
            return

        current = self.manager.current_output_job
        if current is None or current not in navigable:
            new_index = navigable[0]
        else:
            pos = navigable.index(current)
            new_pos = max(0, min(pos + direction, len(navigable) - 1))
            new_index = navigable[new_pos]

        if new_index != current:
            self.manager.switch_to_job(new_index, self.output_path)

    def _get_sorted_running(self) -> list[Job]:
        return sorted(
            self.manager.running_jobs,
            key=lambda j: (j.status == JobStatus.YIELDING, j.worker_id or 0),
        )

    def _get_sorted_finished(self) -> list[Job]:
        return self.manager.finished_jobs

    def get_output_panel_height(self) -> int:
        """Calculate how many lines fit in the output panel."""
        # Panel borders (2) + safety margin (1) to prevent Rich from cropping bottom lines
        terminal_height = self.console.size.height
        return max(terminal_height - 3, 5)

    def format_timedelta(self, td: Optional[timedelta]) -> str:
        if td is None:
            return "--:--"
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"

    def create_header(self) -> Panel:
        text = Text()
        text.append("GPU JOB RUNNER", style=f"bold {COLORS['text_bright']}")
        text.append("  |  ", style=COLORS["text_dim"])
        text.append(f"{self.num_parallel}", style=COLORS["accent"])
        text.append(" workers", style=COLORS["text_dim"])
        text.append("  |  ", style=COLORS["text_dim"])
        text.append("↑↓", style=COLORS["accent"])
        text.append(" select job  ", style=COLORS["text_dim"])
        text.append("PgUp/Dn", style=COLORS["accent"])
        text.append(" scroll output", style=COLORS["text_dim"])
        return Panel(
            Align.center(text),
            box=box.HEAVY,
            border_style=COLORS["border"],
            padding=(0, 0),
        )

    def create_progress_panel(self) -> Panel:
        stats = self.manager.stats
        elapsed = self.format_timedelta(self.manager.elapsed_time)
        estimated = self.format_timedelta(self.manager.estimated_total_time)
        progress_pct = (stats["finished"] / stats["total"] * 100) if stats["total"] > 0 else 0

        bar_width = 30
        filled = int(bar_width * stats["finished"] / max(stats["total"], 1))
        bar = "█" * filled + "░" * (bar_width - filled)

        text = Text()
        text.append(f" {bar[:filled]}", style=COLORS["success"])
        text.append(f"{bar[filled:]}", style=COLORS["text_dim"])
        text.append(f" {progress_pct:5.1f}%\n\n", style=COLORS["text"])
        
        text.append(f" Total     {stats['total']:>5}", style=COLORS["text_dim"])
        text.append(f"     Elapsed   {elapsed:>10}\n", style=COLORS["text_dim"])
        text.append(f" Pending   {stats['pending']:>5}", style=COLORS["warning"])
        text.append(f"     Estimate  {estimated:>10}\n", style=COLORS["text_dim"])
        text.append(f" Running   {stats['running']:>5}", style=COLORS["running"])
        text.append(f"     Output    {self.output_path}\n", style=COLORS["text_dim"])
        text.append(f" Yielding  {stats['yielding']:>5}", style=COLORS["yielding"] if stats["yielding"] > 0 else COLORS["text_dim"])
        text.append(f"\n", style=COLORS["text_dim"])
        text.append(f" Done      {stats['completed']:>5}", style=COLORS["success"])
        text.append(f"\n", style=COLORS["text_dim"])
        text.append(f" Failed    {stats['errored']:>5}", style=COLORS["error"] if stats["errored"] > 0 else COLORS["text_dim"])

        return Panel(
            text,
            title=f"[{COLORS['text_dim']}]progress[/]",
            box=box.ROUNDED,
            border_style=COLORS["border"],
            padding=(0, 1),
        )

    def create_running_table(self) -> Panel:
        table = Table(
            box=None,
            show_header=True,
            header_style=f"bold {COLORS['text_dim']}",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("", width=1)  # Selection marker
        table.add_column("JOB", style=COLORS["text"], width=4, justify="right")
        table.add_column("W", style=COLORS["text_dim"], width=3, justify="right")
        table.add_column("GPU", style=COLORS["text_dim"], width=6)
        table.add_column("TIME", style=COLORS["running"], width=9, justify="right")
        table.add_column("CMD", style=COLORS["text_dim"], no_wrap=True)

        all_running = self._get_sorted_running()
        visible_rows = self._visible_rows("running")

        # Scroll-into-view: ensure selected job is visible
        selected_pos = None
        for i, job in enumerate(all_running):
            if job.index == self.manager.current_output_job:
                selected_pos = i
                break
        if selected_pos is not None:
            if selected_pos < self.running_scroll_top:
                self.running_scroll_top = selected_pos
            elif selected_pos >= self.running_scroll_top + visible_rows:
                self.running_scroll_top = selected_pos - visible_rows + 1
        max_top = max(0, len(all_running) - visible_rows)
        self.running_scroll_top = max(0, min(self.running_scroll_top, max_top))

        visible_jobs = all_running[self.running_scroll_top:self.running_scroll_top + visible_rows]
        for job in visible_jobs:
            is_selected = job.index == self.manager.current_output_job
            time_color = COLORS["yielding"] if job.status == JobStatus.YIELDING else COLORS["running"]
            bg = COLORS["selected_bg"] if is_selected else ""
            if is_selected:
                sel = f"bold on {bg}"
                marker = Text("▶", style=f"{COLORS['accent']} on {bg}")
                job_text = Text(str(job.index), style=f"bold {COLORS['selected_fg']} on {bg}")
                worker_text = Text(str(job.worker_id), style=f"{COLORS['selected_fg']} on {bg}")
                gpu_text = Text(job.gpus, style=f"{COLORS['selected_fg']} on {bg}")
                time_text = Text(job.duration_str, style=f"bold {time_color} on {bg}")
                cmd_text = Text(job.short_command(), style=f"{COLORS['selected_fg']} on {bg}")
            else:
                marker = Text(" ")
                job_text = Text(str(job.index), style=COLORS["text"])
                worker_text = Text(str(job.worker_id), style=COLORS["text_dim"])
                gpu_text = Text(job.gpus, style=COLORS["text_dim"])
                time_text = Text(job.duration_str, style=time_color)
                cmd_text = Text(job.short_command(), style=COLORS["text_dim"])

            table.add_row(
                marker,
                job_text,
                worker_text,
                gpu_text,
                time_text,
                cmd_text,
            )

        scroll_info = ""
        if len(all_running) > visible_rows:
            top = self.running_scroll_top + 1
            bot = min(self.running_scroll_top + visible_rows, len(all_running))
            scroll_info = f" {top}-{bot}/{len(all_running)}"
        border_style = COLORS["border"]
        title = f"[{COLORS['text_dim']}]running{scroll_info}[/]"
        return Panel(
            table,
            title=title,
            box=box.ROUNDED,
            border_style=border_style,
            padding=(0, 0),
        )

    def create_finished_table(self) -> Panel:
        table = Table(
            box=None,
            show_header=True,
            header_style=f"bold {COLORS['text_dim']}",
            expand=True,
            padding=(0, 1),
        )
        table.add_column("", width=1)  # Selection marker
        table.add_column("JOB", style=COLORS["text"], width=5, justify="right")
        table.add_column("STATUS", width=6, justify="center")
        table.add_column("EXIT", width=4, justify="right")
        table.add_column("TIME", style=COLORS["text_dim"], width=9, justify="right")
        table.add_column("CMD", style=COLORS["text_dim"], no_wrap=True)

        all_finished = self._get_sorted_finished()
        visible_rows = self._visible_rows("completed")

        # Scroll-into-view: ensure selected job is visible
        selected_pos = None
        for i, job in enumerate(all_finished):
            if job.index == self.manager.current_output_job:
                selected_pos = i
                break
        if selected_pos is not None:
            if selected_pos < self.completed_scroll_top:
                self.completed_scroll_top = selected_pos
            elif selected_pos >= self.completed_scroll_top + visible_rows:
                self.completed_scroll_top = selected_pos - visible_rows + 1
        max_top = max(0, len(all_finished) - visible_rows)
        self.completed_scroll_top = max(0, min(self.completed_scroll_top, max_top))

        visible_jobs = all_finished[self.completed_scroll_top:self.completed_scroll_top + visible_rows]
        for job in visible_jobs:
            is_selected = job.index == self.manager.current_output_job
            bg = COLORS["selected_bg"] if is_selected else ""
            if job.status == JobStatus.COMPLETED:
                status_label, status_color = "OK", COLORS["success"]
            else:
                status_label, status_color = "FAIL", COLORS["error"]

            if is_selected:
                marker = Text("▶", style=f"{COLORS['accent']} on {bg}")
                job_text = Text(str(job.index), style=f"bold {COLORS['selected_fg']} on {bg}")
                status = Text(status_label, style=f"bold {status_color} on {bg}")
                exit_text = Text(str(job.exit_code), style=f"{status_color} on {bg}")
                time_text = Text(job.duration_str, style=f"{COLORS['selected_fg']} on {bg}")
                cmd_text = Text(job.short_command(), style=f"{COLORS['selected_fg']} on {bg}")
            else:
                marker = Text(" ")
                job_text = Text(str(job.index))
                status = Text(status_label, style=status_color)
                exit_text = Text(str(job.exit_code), style=status_color)
                time_text = Text(job.duration_str, style=COLORS["text_dim"])
                cmd_text = Text(job.short_command(), style=COLORS["text_dim"])

            table.add_row(
                marker,
                job_text,
                status,
                exit_text,
                time_text,
                cmd_text,
            )

        stats = self.manager.stats
        scroll_info = ""
        if len(all_finished) > visible_rows:
            top = self.completed_scroll_top + 1
            bot = min(self.completed_scroll_top + visible_rows, len(all_finished))
            scroll_info = f" {top}-{bot}/{len(all_finished)}"
        border_style = COLORS["border"]
        return Panel(
            table,
            title=f"[{COLORS['text_dim']}]completed {stats['finished']}/{stats['total']}{scroll_info}[/]",
            box=box.ROUNDED,
            border_style=border_style,
            padding=(0, 0),
        )

    def create_output_panel(self) -> Panel:
        # Calculate how many lines can fit in the panel
        visible_lines = self.get_output_panel_height()
        job_index, lines = self.manager.get_output_lines(max_lines=visible_lines)
        
        if job_index is None:
            content = Text("Waiting for jobs to start...", style=COLORS["text_dim"])
            title = f"[{COLORS['text_dim']}]output[/]"
        else:
            job = self.manager.get_job(job_index)
            cmd_short = job.command[:60] + "..." if job and len(job.command) > 60 else (job.command if job else "")
            from rich.markup import escape as _markup_escape
            if job and job.status == JobStatus.COMPLETED:
                status_tag = f"[{COLORS['success']}]OK[/]"
            elif job and job.status == JobStatus.ERRORED:
                status_tag = f"[{COLORS['error']}]FAIL[/]"
            elif job and job.status == JobStatus.YIELDING:
                status_tag = f"[{COLORS['yielding']}]yielding[/]"
            else:
                status_tag = f"[{COLORS['running']}]running[/]"
            scroll_tag = "" if self.manager.scroll_offset == 0 else f" [{COLORS['warning']}]SCROLLED[/]"
            title = f"[{COLORS['text_dim']}]job {job_index}[/] {status_tag}{scroll_tag} [{COLORS['text_dim']}]{_markup_escape(cmd_short)}[/]"

            if not lines:
                content = Text("Waiting for output...", style=COLORS["text_dim"])
            else:
                content = Text()
                # Only show the last N lines that fit (auto-scroll effect)
                # Output panel gets 3/5 of the screen width
                panel_width = (self.console.size.width * 3 // 5) - 4  # minus borders/padding
                for i, line in enumerate(lines):
                    clean = _ANSI_ESCAPE.sub('', line)
                    display_line = clean[:panel_width] if len(clean) > panel_width else clean
                    content.append(display_line, style=COLORS["text"])
                    if i < len(lines) - 1:
                        content.append("\n")

        border_style = COLORS["border"]
        return Panel(
            content,
            title=title,
            box=box.ROUNDED,
            border_style=border_style,
            padding=(0, 1),
        )

    def generate(self) -> Layout:
        layout = Layout()
        layout.split_row(
            Layout(name="left", ratio=2),
            Layout(name="right", ratio=3),
        )
        layout["left"].split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=10),
            Layout(name="running", size=self._running_panel_size()),
            Layout(name="completed", ratio=1),
        )
        layout["header"].update(self.create_header())
        layout["progress"].update(self.create_progress_panel())
        layout["running"].update(self.create_running_table())
        layout["completed"].update(self.create_finished_table())
        layout["right"].update(self.create_output_panel())
        return layout


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run multiple commands in parallel on separate GPUs with a live dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help="Number of commands to run in parallel. Required if --gpus is not specified.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=0,
        help="Number of GPUs to assign per job. If specified, --parallel is inferred.",
    )
    parser.add_argument(
        "--first-gpu",
        type=int,
        default=0,
        help="The first GPU index to use (default: 0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Directory to store stdout and stderr of each job.",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=0.1,
        help="Dashboard refresh rate in seconds (default: 0.1).",
    )
    return parser.parse_args()


def read_commands():
    """Read commands from stdin."""
    commands = []
    for idx, line in enumerate(sys.stdin):
        cmd = line.strip()
        if cmd and not cmd.startswith("#"):
            commands.append((idx, cmd))
    return commands


def _monitor_yielded_process(
    process: subprocess.Popen,
    job_index: int,
    output_file,
    output_buffer: str,
    manager: JobManager,
):
    """Background monitor for a process whose job has yielded GPU time."""
    try:
        while True:
            ret = process.poll()
            try:
                ready, _, _ = select.select([process.stdout], [], [], 0.05)
                if ready:
                    chunk = process.stdout.read()
                    if chunk:
                        output_buffer += chunk
                        while '\n' in output_buffer:
                            line, output_buffer = output_buffer.split('\n', 1)
                            output_file.write(line + '\n')
                            output_file.flush()
                            manager.append_output(job_index, line)
            except (IOError, OSError):
                pass

            if ret is not None:
                try:
                    remaining = process.stdout.read()
                    if remaining:
                        output_buffer += remaining
                except (IOError, OSError):
                    pass
                if output_buffer:
                    for line in output_buffer.splitlines():
                        output_file.write(line + '\n')
                        output_file.flush()
                        manager.append_output(job_index, line)

                output_file.write(f"\n{'='*60}\n")
                output_file.write(f"Finished: {datetime.now().isoformat()}\n")
                output_file.write(f"Exit code: {ret}\n")
                output_file.write(f"{'='*60}\n")
                output_file.close()
                manager.complete_job(job_index, ret)
                return

            time.sleep(0.01)
    except Exception as e:
        try:
            output_file.write(f"\n\nBackground monitor error: {e}\n")
            output_file.close()
        except Exception:
            pass
        manager.complete_job(job_index, 1)


def worker(
    worker_id: int,
    first_gpu: int,
    num_gpus: int,
    job_queue: queue.Queue,
    output_dir: str,
    manager: JobManager,
    bg_threads: list,
):
    """Worker thread that processes jobs from the queue."""
    if num_gpus > 0:
        start_gpu = first_gpu + worker_id * num_gpus
        end_gpu = start_gpu + num_gpus
        gpu_indices = ",".join(map(str, range(start_gpu, end_gpu)))
    else:
        gpu_indices = ""

    while True:
        try:
            index, command = job_queue.get_nowait()
        except queue.Empty:
            break

        manager.start_job(index, worker_id, gpu_indices if num_gpus > 0 else "-")

        env = os.environ.copy()
        if num_gpus > 0:
            env["CUDA_VISIBLE_DEVICES"] = gpu_indices
        # Force unbuffered output for Python subprocesses
        env["PYTHONUNBUFFERED"] = "1"

        # Yield file: child writes "true" to signal GPU work is done
        yield_file = os.path.join(output_dir, f"yield_{index}.txt")
        with open(yield_file, "w") as yf:
            yf.write("false")
        env["BATCH_LOCAL_YIELD_FILE"] = yield_file

        output_file = os.path.join(output_dir, f"{index}.txt")
        exit_code = 1
        yielded = False

        with open(output_file, "w") as f:
            f.write(f"{'='*60}\n")
            f.write(f"Job #{index}\n")
            f.write(f"Command: {command}\n")
            f.write(f"Worker: {worker_id}\n")
            f.write(f"GPUs: {gpu_indices if num_gpus > 0 else 'N/A'}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.flush()

            try:
                # stderr=subprocess.STDOUT merges stderr into stdout so both are captured
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    env=env,
                    text=True,
                    bufsize=1,  # Line buffered
                )
                
                # Set stdout to non-blocking mode
                fd = process.stdout.fileno()
                flags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
                
                # Actively poll for process completion and output
                output_buffer = ""
                last_yield_check = time.time()
                while True:
                    # Check if process has finished
                    ret = process.poll()
                    
                    # Try to read any available output (non-blocking)
                    try:
                        ready, _, _ = select.select([process.stdout], [], [], 0.05)
                        if ready:
                            chunk = process.stdout.read()
                            if chunk:
                                output_buffer += chunk
                                # Process complete lines
                                while '\n' in output_buffer:
                                    line, output_buffer = output_buffer.split('\n', 1)
                                    f.write(line + '\n')
                                    f.flush()
                                    manager.append_output(index, line)
                    except (IOError, OSError):
                        pass  # No data available

                    # Check yield file every ~1 second
                    now = time.time()
                    if now - last_yield_check >= 1.0:
                        last_yield_check = now
                        try:
                            with open(yield_file, "r") as yf:
                                if yf.read().strip().lower() == "true":
                                    manager.yield_job(index)
                                    dup_f = open(output_file, "a")
                                    t = threading.Thread(
                                        target=_monitor_yielded_process,
                                        args=(process, index, dup_f, output_buffer, manager),
                                        daemon=True,
                                    )
                                    t.start()
                                    bg_threads.append(t)
                                    yielded = True
                                    break
                        except Exception:
                            pass

                    # If process has terminated, drain remaining output and exit
                    if ret is not None:
                        # Final read to drain any remaining output
                        try:
                            remaining = process.stdout.read()
                            if remaining:
                                output_buffer += remaining
                        except (IOError, OSError):
                            pass
                        
                        # Process any remaining buffered content
                        if output_buffer:
                            for line in output_buffer.splitlines():
                                f.write(line + '\n')
                                f.flush()
                                manager.append_output(index, line)
                        
                        exit_code = ret
                        break
                    
                    # Small sleep to prevent busy-waiting
                    time.sleep(0.01)
                
            except Exception as e:
                error_msg = f"\n\nError executing command: {e}\n"
                f.write(error_msg)
                manager.append_output(index, error_msg)
                exit_code = 1

            if not yielded:
                f.write(f"\n{'='*60}\n")
                f.write(f"Finished: {datetime.now().isoformat()}\n")
                f.write(f"Exit code: {exit_code}\n")
                f.write(f"{'='*60}\n")

        if not yielded:
            manager.complete_job(index, exit_code)
        job_queue.task_done()


def create_output_directory(output_path: Optional[str]) -> str:
    """Create and return the output directory path."""
    if output_path is None:
        base_dir = '/home/jspringe/slurm/local_outputs'
        if not os.path.exists(base_dir):
            raise RuntimeError(f"Must provide --output or create {base_dir}")
        existing = os.listdir(base_dir)
        next_id = max((int(d) for d in existing if d.isdigit()), default=0) + 1
        output_path = os.path.join(base_dir, str(next_id))

    os.makedirs(output_path, exist_ok=True)
    return output_path


def main():
    args = parse_arguments()
    console = Console()

    if args.gpus == 0 and args.parallel is None:
        console.print(
            f"[{COLORS['error']}]Error: Either --parallel or --gpus must be specified.[/]"
        )
        sys.exit(1)

    if args.gpus > 0:
        try:
            available_gpus = torch.cuda.device_count()
        except Exception:
            available_gpus = 0

        if available_gpus == 0:
            console.print(f"[{COLORS['warning']}]Warning: No GPUs detected.[/]")
            if args.parallel is None:
                console.print(f"[{COLORS['error']}]Error: --parallel required without GPUs.[/]")
                sys.exit(1)
            num_parallel = args.parallel
            args.gpus = 0
        else:
            inferred_parallel = (available_gpus - args.first_gpu) // args.gpus
            num_parallel = min(args.parallel, inferred_parallel) if args.parallel else inferred_parallel
            if num_parallel <= 0:
                console.print(f"[{COLORS['error']}]Error: Not enough GPUs.[/]")
                sys.exit(1)
    else:
        num_parallel = args.parallel

    if sys.stdin.isatty():
        console.print(f"[{COLORS['text_dim']}]Waiting for commands on stdin...[/]")

    commands = read_commands()
    if not commands:
        console.print(f"[{COLORS['error']}]No commands to execute.[/]")
        sys.exit(1)

    output_path = create_output_directory(args.output)
    manager = JobManager()
    for idx, cmd in commands:
        manager.add_job(idx, cmd)

    job_queue = queue.Queue()
    for cmd in commands:
        job_queue.put(cmd)

    dashboard = Dashboard(manager, output_path, num_parallel, console)

    bg_threads = []
    threads = []
    for worker_id in range(num_parallel):
        t = threading.Thread(
            target=worker,
            args=(worker_id, args.first_gpu, args.gpus, job_queue, output_path, manager, bg_threads),
            daemon=True,
        )
        t.start()
        threads.append(t)

    # Set up keyboard input handling
    # Reconnect to /dev/tty for keyboard input (stdin may be a pipe)
    old_settings = None
    old_flags = None
    tty_fd = None
    keyboard_enabled = False
    
    try:
        tty_fd = os.open('/dev/tty', os.O_RDONLY | os.O_NONBLOCK)
        old_settings = termios.tcgetattr(tty_fd)
        new_settings = termios.tcgetattr(tty_fd)
        new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO  # Disable canonical mode and echo
        termios.tcsetattr(tty_fd, termios.TCSANOW, new_settings)
        keyboard_enabled = True
    except Exception:
        pass  # May fail if no controlling terminal

    def check_keyboard():
        """Check for key presses. Returns list of key names."""
        if not keyboard_enabled or tty_fd is None:
            return []
        try:
            ready, _, _ = select.select([tty_fd], [], [], 0)
            if not ready:
                return []
            data = os.read(tty_fd, 256).decode('utf-8', errors='ignore')
            keys = []
            i = 0
            while i < len(data):
                if data[i] == '\x1b' and i + 1 < len(data) and data[i + 1] == '[':
                    # CSI escape sequence: \x1b[ <params> <final_byte>
                    i += 2
                    param = ''
                    while i < len(data) and (data[i].isdigit() or data[i] == ';'):
                        param += data[i]
                        i += 1
                    if i < len(data):
                        final = data[i]
                        i += 1
                        if final == 'A':
                            keys.append('up')
                        elif final == 'B':
                            keys.append('down')
                        elif final == 'C':
                            keys.append('right')
                        elif final == 'D':
                            keys.append('left')
                        elif final == 'Z':
                            keys.append('shift_tab')
                        elif final == '~':
                            if param == '5':
                                keys.append('page_up')
                            elif param == '6':
                                keys.append('page_down')
                        elif final == 'H':
                            keys.append('home')
                        elif final == 'F':
                            keys.append('end')
                elif data[i] == '\t':
                    keys.append('tab')
                    i += 1
                else:
                    i += 1
            return keys
        except Exception:
            return []

    try:
        with Live(
            dashboard.generate(),
            console=console,
            auto_refresh=False,
            screen=True,
        ) as live:
            last_render = time.monotonic()
            render_interval = args.refresh_rate
            while any(t.is_alive() for t in threads) or any(t.is_alive() for t in list(bg_threads)):
                page_size = max(dashboard.get_output_panel_height() // 2, 5)
                dirty = False
                for key in check_keyboard():
                    if key == 'up':
                        dashboard.navigate_job(-1)
                    elif key == 'down':
                        dashboard.navigate_job(1)
                    elif key == 'page_up':
                        manager.scroll_output(page_size)
                    elif key == 'page_down':
                        manager.scroll_output(-page_size)
                    elif key == 'home':
                        manager.scroll_output(999999)
                    elif key == 'end':
                        manager.scroll_output(-999999)
                    else:
                        continue
                    dirty = True

                now = time.monotonic()
                if dirty or now - last_render >= render_interval:
                    live.update(dashboard.generate(), refresh=True)
                    last_render = now

                time.sleep(0.015)
            live.update(dashboard.generate(), refresh=True)
            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        # Restore terminal settings
        try:
            if old_settings is not None and tty_fd is not None:
                termios.tcsetattr(tty_fd, termios.TCSADRAIN, old_settings)
            if tty_fd is not None:
                os.close(tty_fd)
        except Exception:
            pass

    for t in threads:
        t.join(timeout=1.0)
    for t in bg_threads:
        t.join(timeout=5.0)

    # Final summary
    stats = manager.stats
    console.print()
    console.print(f"[{COLORS['text_bright']}]Completed:[/] {stats['completed']}  "
                  f"[{COLORS['error']}]Failed:[/] {stats['errored']}  "
                  f"[{COLORS['text_dim']}]Time:[/] {dashboard.format_timedelta(manager.elapsed_time)}  "
                  f"[{COLORS['text_dim']}]Output:[/] {output_path}")

    sys.exit(1 if stats["errored"] > 0 else 0)


if __name__ == "__main__":
    main()
