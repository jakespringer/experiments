#!/usr/bin/env python3
"""
GPU Job Runner - A terminal dashboard for parallel GPU job execution.

Pipe commands via stdin and watch them execute in parallel across GPUs with
a live-updating dashboard showing progress, timing, and job status.
"""

import argparse
import os
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
}


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
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

    @property
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
    
            # Pick a deterministic next running job (by worker_id then index)
            running = [j for j in self.jobs if j.status == JobStatus.RUNNING]
            running.sort(key=lambda j: (j.worker_id is None, j.worker_id or 0, j.index))
            next_running_index = running[0].index if running else None
    
        # Now touch output state under output_lock (and do NOT call anything that grabs self.lock here).
        with self.output_lock:
            if self.current_output_job == index:
                self.current_output_job = next_running_index
                self.output_lines.clear()

    def append_output(self, job_index: int, line: str):
        with self.output_lock:
            if self.current_output_job == job_index:
                self.output_lines.append(line)

    def get_output_lines(self, max_lines: int = 50) -> tuple[Optional[int], list[str]]:
        with self.output_lock:
            return self.current_output_job, list(self.output_lines)[-max_lines:]

    def switch_to_job(self, job_index: int, output_dir: str):
        """Switch to monitoring a specific job, loading its output from file."""
        with self.output_lock:
            self.current_output_job = job_index
            self.output_lines.clear()
            # Load existing output from file
            output_file = os.path.join(output_dir, f"{job_index}.txt")
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        for line in f:
                            self.output_lines.append(line.rstrip('\n'))
                except Exception:
                    pass

    def cycle_job(self, direction: int, output_dir: str):
        """Cycle through running jobs. direction: 1 for next, -1 for previous."""
        running = self.running_jobs
        if not running:
            return
        
        with self.output_lock:
            current = self.current_output_job
            # Get list of running job indices
            running_indices = [j.index for j in sorted(running, key=lambda j: j.worker_id or 0)]
            
            if current is None or current not in running_indices:
                # Select first running job
                new_index = running_indices[0]
            else:
                # Find current position and cycle
                try:
                    pos = running_indices.index(current)
                    new_pos = (pos + direction) % len(running_indices)
                    new_index = running_indices[new_pos]
                except ValueError:
                    new_index = running_indices[0]
            
            if new_index != current:
                self.current_output_job = new_index
                self.output_lines.clear()
                # Load existing output from file
                output_file = os.path.join(output_dir, f"{new_index}.txt")
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
            completed = sum(1 for j in self.jobs if j.status == JobStatus.COMPLETED)
            errored = sum(1 for j in self.jobs if j.status == JobStatus.ERRORED)
            return {
                "total": total,
                "pending": pending,
                "running": running,
                "completed": completed,
                "errored": errored,
                "finished": completed + errored,
            }

    @property
    def running_jobs(self) -> list[Job]:
        with self.lock:
            return [j for j in self.jobs if j.status == JobStatus.RUNNING]

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
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self):
        self.layout.split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )
        self.layout["left"].split_column(
            Layout(name="header", size=3),
            Layout(name="progress", size=10),
            Layout(name="running", ratio=1),
            Layout(name="completed", ratio=1),
        )

    def get_output_panel_height(self) -> int:
        """Calculate how many lines fit in the output panel."""
        # Get terminal height, subtract 2 for panel borders, 1 for title, 2 for padding
        terminal_height = self.console.size.height
        return max(terminal_height - 5, 10)

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

        running_jobs = self.manager.running_jobs
        for job in sorted(running_jobs, key=lambda j: j.worker_id or 0):
            is_selected = job.index == self.manager.current_output_job
            if is_selected:
                marker = Text(">", style=f"bold {COLORS['text_bright']}")
                job_text = Text(str(job.index), style=f"bold {COLORS['text_bright']}")
                time_text = Text(job.duration_str, style=f"bold {COLORS['running']}")
                cmd_text = Text(job.short_command, style=COLORS["text"])
            else:
                marker = Text(" ")
                job_text = Text(str(job.index), style=COLORS["text"])
                time_text = Text(job.duration_str, style=COLORS["running"])
                cmd_text = Text(job.short_command, style=COLORS["text_dim"])
            
            table.add_row(
                marker,
                job_text,
                str(job.worker_id),
                job.gpus,
                time_text,
                cmd_text,
            )

        title = f"[{COLORS['text_dim']}]running (up/down to switch)[/]"
        return Panel(
            table,
            title=title,
            box=box.ROUNDED,
            border_style=COLORS["running"],
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
        table.add_column("JOB", style=COLORS["text"], width=5, justify="right")
        table.add_column("STATUS", width=6, justify="center")
        table.add_column("EXIT", width=4, justify="right")
        table.add_column("TIME", style=COLORS["text_dim"], width=9, justify="right")
        table.add_column("CMD", style=COLORS["text_dim"], no_wrap=True)

        finished_jobs = self.manager.finished_jobs[:15]
        for job in finished_jobs:
            if job.status == JobStatus.COMPLETED:
                status = Text("OK", style=COLORS["success"])
                exit_style = COLORS["success"]
            else:
                status = Text("FAIL", style=COLORS["error"])
                exit_style = COLORS["error"]

            table.add_row(
                str(job.index),
                status,
                Text(str(job.exit_code or 0), style=exit_style),
                job.duration_str,
                job.short_command,
            )

        stats = self.manager.stats
        return Panel(
            table,
            title=f"[{COLORS['text_dim']}]completed {stats['finished']}/{stats['total']}[/]",
            box=box.ROUNDED,
            border_style=COLORS["border"],
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
            title = f"[{COLORS['text_dim']}]job {job_index}[/] [{COLORS['text_dim']}]{cmd_short}[/]"
            
            if not lines:
                content = Text("Waiting for output...", style=COLORS["text_dim"])
            else:
                content = Text()
                # Only show the last N lines that fit (auto-scroll effect)
                for line in lines:
                    # Truncate long lines to panel width
                    panel_width = (self.console.size.width // 2) - 4  # Half screen minus borders/padding
                    display_line = line[:panel_width] if len(line) > panel_width else line
                    content.append(display_line + "\n", style=COLORS["text"])

        return Panel(
            content,
            title=title,
            box=box.ROUNDED,
            border_style=COLORS["border_active"],
            padding=(0, 1),
        )

    def generate(self) -> Layout:
        self.layout["header"].update(self.create_header())
        self.layout["progress"].update(self.create_progress_panel())
        self.layout["running"].update(self.create_running_table())
        self.layout["completed"].update(self.create_finished_table())
        self.layout["right"].update(self.create_output_panel())
        return self.layout


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


def worker(
    worker_id: int,
    first_gpu: int,
    num_gpus: int,
    job_queue: queue.Queue,
    output_dir: str,
    manager: JobManager,
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

        output_file = os.path.join(output_dir, f"{index}.txt")
        exit_code = 1

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

            f.write(f"\n{'='*60}\n")
            f.write(f"Finished: {datetime.now().isoformat()}\n")
            f.write(f"Exit code: {exit_code}\n")
            f.write(f"{'='*60}\n")

        manager.complete_job(index, exit_code)
        job_queue.task_done()


def create_output_directory(output_path: Optional[str]) -> str:
    """Create and return the output directory path."""
    if output_path is None:
        base_dir = '/home/jspringe/slurm/local_outputs'
        if not os.path.exists(base_dir):
            raise RuntimeError(f"Must provide --output or create {base_dir}")
        existing = os.listdir(base_dir)
        output_path = os.path.join(base_dir, str(len(existing) + 1))

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

    threads = []
    for worker_id in range(num_parallel):
        t = threading.Thread(
            target=worker,
            args=(worker_id, args.first_gpu, args.gpus, job_queue, output_path, manager),
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
        """Check for arrow key presses. Returns 'up', 'down', or None."""
        if not keyboard_enabled or tty_fd is None:
            return None
        try:
            ready, _, _ = select.select([tty_fd], [], [], 0)
            if ready:
                ch = os.read(tty_fd, 1).decode('utf-8', errors='ignore')
                if ch == '\x1b':  # Escape sequence
                    ch2 = os.read(tty_fd, 1).decode('utf-8', errors='ignore')
                    if ch2 == '[':
                        ch3 = os.read(tty_fd, 1).decode('utf-8', errors='ignore')
                        if ch3 == 'A':
                            return 'up'
                        elif ch3 == 'B':
                            return 'down'
        except Exception:
            pass
        return None

    try:
        with Live(
            dashboard.generate(),
            console=console,
            refresh_per_second=1 / args.refresh_rate,
            screen=True,
        ) as live:
            while any(t.is_alive() for t in threads):
                # Check for keyboard input
                key = check_keyboard()
                if key == 'up':
                    manager.cycle_job(-1, output_path)
                elif key == 'down':
                    manager.cycle_job(1, output_path)
                
                live.update(dashboard.generate())
                time.sleep(args.refresh_rate)
            live.update(dashboard.generate())
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
