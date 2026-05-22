"""Local execution of experiment stages with GPU/CPU-aware scheduling.

Adds a ``LocalExecutor`` that mirrors ``PrintExecutor``'s compilation step
(building per-task bash scripts from artifacts) but spawns the resulting
scripts as subprocesses on the host. Tasks within the selected DAG run in
parallel up to the configured GPU and CPU limits. GPU indices are allocated
explicitly via ``CUDA_VISIBLE_DEVICES`` so heterogeneous-resource jobs can
share the GPU pool safely.

Dependency semantics match ``launch``: a task only starts after every
prerequisite that's still in the run completes successfully (cross-stage
dependencies on artifacts excluded from the run are assumed satisfied, just
like Slurm ``afterok`` skips dependencies absent from the submission set).
A failed prerequisite cascades — dependents are marked errored without
running.
"""
from __future__ import annotations

import fcntl
import os
import select
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .artifact import Artifact
from .batch import _parse_gpu_count
from .config import ConfigManager
from .executor import (
    Executor,
    Task,
    _artifact_experiment_conf,
    _is_raw_block,
    _safe_json_dumps,
    dquote,
)
from .project import Project


def _detect_gpus() -> int:
    """Return the number of physical GPUs visible on the host.

    Prefers ``nvidia-smi`` (truth source on the node) and falls back to
    ``torch.cuda.device_count()``. Returns 0 if neither is available.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return sum(1 for line in result.stdout.splitlines() if line.strip())
    except Exception:
        pass
    try:
        import torch  # type: ignore

        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _extract_gpu_count(reqs: Dict[str, Any]) -> int:
    if reqs.get("gpus") is not None:
        return _parse_gpu_count(reqs["gpus"])
    gres = reqs.get("gres")
    if gres and str(gres).startswith("gpu"):
        return _parse_gpu_count(gres)
    return 0


def _extract_cpu_count(reqs: Dict[str, Any]) -> int:
    val = reqs.get("cpus_per_task", reqs.get("cpus", 1))
    try:
        return max(1, int(val))
    except (TypeError, ValueError):
        return 1


def _default_output_dir() -> str:
    """Pick an output directory in the user's slurm outputs dir if it exists,
    else fall back to a fresh tempdir under ``/tmp``."""
    base = "/home/jspringe/slurm/local_outputs"
    if os.path.isdir(base):
        existing = [d for d in os.listdir(base) if d.isdigit()]
        next_id = max((int(d) for d in existing), default=0) + 1
        path = os.path.join(base, str(next_id))
        os.makedirs(path, exist_ok=True)
        return path
    return tempfile.mkdtemp(prefix="exp_local_")


class _TaskNode:
    """A scheduled task with its resource requirements and prerequisites."""

    __slots__ = (
        "index",
        "artifact_id",
        "task",
        "gpu_count",
        "cpu_count",
        "prerequisites",
        "script_path",
    )

    def __init__(
        self,
        index: int,
        artifact_id: int,
        task: Task,
        gpu_count: int,
        cpu_count: int,
        prerequisites: Set[int],
    ) -> None:
        self.index = index
        self.artifact_id = artifact_id
        self.task = task
        self.gpu_count = gpu_count
        self.cpu_count = cpu_count
        self.prerequisites = prerequisites
        self.script_path: Optional[str] = None


class LocalExecutor(Executor):
    """Run tasks locally with DAG-aware GPU/CPU scheduling.

    The executor reuses ``Executor.execute()`` for stage validation,
    topological ordering, exists/skip filtering, and head/tail filtering.
    Once tasks are compiled, ``launch()`` builds a DAG over the selected
    tasks and dispatches them as subprocesses with explicit GPU index
    allocation.
    """

    def __init__(
        self,
        artifact_path: Optional[str] = None,
        code_path: Optional[str] = None,
        gs_path: Optional[str] = None,
        setup_command: Optional[str] = None,
        first_gpu: int = 0,
        max_gpus: Optional[int] = None,
        max_cpus: Optional[int] = None,
        output_dir: Optional[str] = None,
        refresh_rate: float = 0.1,
        no_dashboard: bool = False,
    ) -> None:
        super().__init__()
        mgr = ConfigManager()
        global_conf = mgr.ensure_config()
        proj_name = Project.name
        proj_conf: Dict[str, Any] = {}
        if proj_name is not None:
            try:
                proj_conf = mgr.load_project_config(proj_name).get("config", {})
            except Exception:
                proj_conf = {}
        if artifact_path is None:
            artifact_path = proj_conf.get("artifact_path") or str(
                mgr.get_project_dir(proj_name or "default") / "artifacts"
            )
        if code_path is None:
            code_path = proj_conf.get("code_path") or str(Path.cwd())
        self.artifact_path = Path(artifact_path)
        self.code_path = Path(code_path)
        self.gs_path = gs_path
        self.setup_command = setup_command
        self.first_gpu = first_gpu
        self.max_gpus = max_gpus
        self.max_cpus = max_cpus
        self.output_dir = output_dir
        self.refresh_rate = refresh_rate
        self.no_dashboard = no_dashboard
        self._verbose_filtering = False
        self._global_config = global_conf

    def compile_artifact(self, artifact: Artifact) -> Task:
        log_dir = self._global_config.get(
            "log_directory", str(Path.home() / ".experiments" / "logs")
        )
        task = Task(
            artifact_path=str(self.artifact_path),
            code_path=str(self.code_path),
            artifact=artifact,
            gs_path=self.gs_path,
            log_dir=log_dir,
        )
        artifact.construct(task)
        return task

    def launch(
        self,
        tiers: List[List[Task]],
        tier_to_stages: Optional[Dict[int, List[str]]] = None,
        jobs: Optional[int] = None,
    ) -> None:
        # ---- resource pool ----
        detected = _detect_gpus()
        first = max(0, self.first_gpu)
        avail = max(0, detected - first)
        if self.max_gpus is None:
            num_gpus = avail
        else:
            num_gpus = min(self.max_gpus, avail)
        gpu_pool = list(range(first, first + num_gpus))
        num_cpus = self.max_cpus if self.max_cpus is not None else (os.cpu_count() or 1)

        # ---- output directory ----
        if self.output_dir is None:
            output_dir = _default_output_dir()
        else:
            output_dir = self.output_dir
            os.makedirs(output_dir, exist_ok=True)

        # ---- flatten + build DAG ----
        all_tasks = [t for tier in tiers for t in tier]
        if not all_tasks:
            print("No tasks to run.", file=sys.stderr)
            return

        artifact_id_to_index: Dict[int, int] = {}
        for i, task in enumerate(all_tasks):
            if task.artifact is not None:
                artifact_id_to_index[id(task.artifact)] = i

        # Build a member→producer map so prereq edges that target a
        # filtered-out member (e.g. an individual ``JudgedResponses``
        # or per-spec ``MsmResponses`` whose ``..._individual`` stage
        # isn't in the active selection) get retargeted onto the
        # containing batched producer (``BatchedJudgedResponses``,
        # ``BatchedModelResponses``, ``ArtifactBatch``). Without this,
        # downstream tasks have no prereq edge and race the producer.
        member_to_producer_id: Dict[int, int] = {}
        for i, task in enumerate(all_tasks):
            if task.artifact is None:
                continue
            try:
                contained = task.artifact.contained_artifacts()
            except Exception:
                contained = []
            for child in contained:
                member_to_producer_id[id(child)] = id(task.artifact)

        nodes: List[_TaskNode] = []
        for i, task in enumerate(all_tasks):
            artifact_id = id(task.artifact) if task.artifact is not None else id(task)
            prereqs: Set[int] = set()
            if task.artifact is not None:
                try:
                    deps = task.artifact.get_direct_dependencies()
                except Exception:
                    deps = []
                for dep in deps:
                    dep_id = id(dep)
                    if dep_id == artifact_id:
                        continue
                    if dep_id in artifact_id_to_index:
                        prereqs.add(dep_id)
                        continue
                    # Filtered-out member: route the edge to its
                    # batched producer if one is in the run plan.
                    producer_id = member_to_producer_id.get(dep_id)
                    if producer_id is not None and producer_id in artifact_id_to_index and producer_id != artifact_id:
                        prereqs.add(producer_id)
            reqs: Dict[str, Any] = {}
            if task.artifact is not None and hasattr(task.artifact, "get_requirements"):
                try:
                    reqs = dict(task.artifact.get_requirements())  # type: ignore[attr-defined]
                except Exception:
                    reqs = {}
                if "cpus" in reqs and "cpus_per_task" not in reqs:
                    reqs["cpus_per_task"] = reqs.pop("cpus")
            gpu_count = _extract_gpu_count(reqs)
            cpu_count = _extract_cpu_count(reqs)
            nodes.append(
                _TaskNode(
                    index=i,
                    artifact_id=artifact_id,
                    task=task,
                    gpu_count=gpu_count,
                    cpu_count=cpu_count,
                    prerequisites=prereqs,
                )
            )

        # Validate that each task fits in the pool at all (else we'd deadlock).
        for node in nodes:
            label = (
                node.task.artifact.relpath
                if node.task.artifact is not None
                else f"task-{node.index}"
            )
            if node.gpu_count > num_gpus:
                print(
                    f"Error: Task {label} requires {node.gpu_count} GPU(s) "
                    f"but only {num_gpus} are available "
                    f"(detected={detected}, first_gpu={first}, max_gpus={self.max_gpus}).",
                    file=sys.stderr,
                )
                sys.exit(1)
            if node.cpu_count > num_cpus:
                print(
                    f"Error: Task {label} requires {node.cpu_count} CPU(s) "
                    f"but only {num_cpus} are available.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # ---- write per-task scripts ----
        for node in nodes:
            node.script_path = self._write_task_script(node, output_dir)

        # ---- launch dashboard / scheduler ----
        from .scripts.batch_local import JobManager  # lazy: pulls rich, etc.

        manager = JobManager()
        for node in nodes:
            manager.add_job(node.index, self._task_summary(node))

        gpu_range = (
            f"{first}-{first + num_gpus - 1}" if num_gpus > 0 else "none"
        )
        print(
            f"Running {len(nodes)} task(s) locally  "
            f"GPUs={num_gpus} (indices {gpu_range})  "
            f"CPUs={num_cpus}  Output={output_dir}",
            file=sys.stderr,
        )
        if tier_to_stages:
            stages = sorted({s for ss in tier_to_stages.values() for s in ss})
            if stages:
                print(f"Stages: {', '.join(stages)}", file=sys.stderr)
        print("", file=sys.stderr)

        scheduler = _LocalScheduler(
            nodes=nodes,
            gpu_pool=gpu_pool,
            max_cpus=num_cpus,
            output_dir=output_dir,
            manager=manager,
        )

        # If we don't have a TTY, fall back to plain mode regardless of flag.
        use_dashboard = (not self.no_dashboard) and sys.stdout.isatty()
        scheduler.run(use_dashboard=use_dashboard, refresh_rate=self.refresh_rate)

    def _write_task_script(self, node: _TaskNode, output_dir: str) -> str:
        """Compile a task's blocks into a self-contained bash script."""
        task = node.task
        lines: List[str] = ["#!/usr/bin/env bash"]
        if self.setup_command:
            lines.append(self.setup_command)
            lines.append("")
        lines.append("set -euo pipefail")
        lines.append("")

        try:
            proj_conf = ConfigManager().load_project_config(Project.name or "").get(
                "config", {}
            )
        except Exception:
            proj_conf = {}
        lines.append(
            f"export EXPERIMENTS_PROJECT_CONF={dquote(_safe_json_dumps(proj_conf))}"
        )
        if task.artifact is not None:
            exp_conf = _artifact_experiment_conf(task.artifact)
            lines.append(
                f"export EXPERIMENTS_EXPERIMENT_CONF={dquote(_safe_json_dumps(exp_conf))}"
            )
        lines.append("")

        for block in task.blocks:
            cmd = block.execute()
            if not cmd:
                continue
            if _is_raw_block(block):
                lines.append(cmd)
            else:
                escaped = (
                    cmd.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("`", "\\`")
                    .replace("$", "\\$")
                )
                lines.append(f'echo "+ {escaped}" >&2')
                lines.append(cmd)

        sh_dir = os.path.join(output_dir, "sh")
        os.makedirs(sh_dir, exist_ok=True)
        path = os.path.join(sh_dir, f"{node.index}.sh")
        with open(path, "w") as f:
            f.write("\n".join(lines))
            f.write("\n")
        os.chmod(path, 0o755)
        return path

    @staticmethod
    def _task_summary(node: _TaskNode) -> str:
        if node.task.artifact is not None:
            cls = node.task.artifact.__class__.__name__
            try:
                rel = node.task.artifact.relpath
            except Exception:
                rel = ""
            return f"{cls} ({rel}) [g{node.gpu_count}c{node.cpu_count}]"
        return f"task-{node.index} [g{node.gpu_count}c{node.cpu_count}]"


class _LocalScheduler:
    """DAG-aware scheduler with explicit GPU index allocation.

    Single dedicated scheduler thread owns ``running`` / ``completed_ids`` /
    ``failed_ids``; the dashboard runs on the main thread and reads job
    state via ``JobManager`` (which has its own internal locks).
    """

    def __init__(
        self,
        nodes: List[_TaskNode],
        gpu_pool: List[int],
        max_cpus: int,
        output_dir: str,
        manager: Any,  # JobManager (lazy import)
    ) -> None:
        self.nodes = nodes
        self.gpu_pool = list(gpu_pool)
        self.max_cpus = max_cpus
        self.output_dir = output_dir
        self.manager = manager
        self.lock = threading.RLock()
        self.completed_ids: Set[int] = set()
        self.failed_ids: Set[int] = set()
        # Each entry: dict(node, process, gpus, cpus, output_file, output_buffer)
        self.running: List[Dict[str, Any]] = []
        self.done_event = threading.Event()
        self.aborted = False

    # -- resource accounting -------------------------------------------------
    def _free_gpus(self) -> List[int]:
        used: Set[int] = set()
        for e in self.running:
            used.update(e["gpus"])
        return [g for g in self.gpu_pool if g not in used]

    def _used_cpus(self) -> int:
        return sum(e["cpus"] for e in self.running)

    # -- main loop -----------------------------------------------------------
    def _scheduler_loop(self) -> None:
        pending: List[_TaskNode] = list(self.nodes)
        try:
            while True:
                with self.lock:
                    # Cascade prerequisite failures (afterok semantics).
                    for node in list(pending):
                        if node.prerequisites & self.failed_ids:
                            self._mark_skipped_by_failure(node)
                            pending.remove(node)

                    # Stop dispatching new tasks if user aborted; let
                    # in-flight processes finish so we can reap them cleanly.
                    if not self.aborted:
                        progressed = True
                        while progressed:
                            progressed = False
                            free = self._free_gpus()
                            used = self._used_cpus()
                            for node in list(pending):
                                if not node.prerequisites.issubset(self.completed_ids):
                                    continue
                                if node.gpu_count > len(free):
                                    continue
                                if node.cpu_count + used > self.max_cpus:
                                    continue
                                allocated = free[: node.gpu_count]
                                free = free[node.gpu_count :]
                                used += node.cpu_count
                                self._launch(node, allocated)
                                pending.remove(node)
                                progressed = True

                    finished = (not pending or self.aborted) and not self.running

                if finished:
                    return

                self._poll_running()
                time.sleep(0.05)
        finally:
            self.done_event.set()

    # -- dispatch / cancel ---------------------------------------------------
    def _mark_skipped_by_failure(self, node: _TaskNode) -> None:
        path = os.path.join(self.output_dir, f"{node.index}.txt")
        try:
            with open(path, "w") as f:
                f.write("Skipped: one or more prerequisite tasks failed.\n")
        except Exception:
            pass
        self.failed_ids.add(node.artifact_id)
        # Show as immediate fail in the dashboard.
        self.manager.start_job(node.index, worker_id=node.index, gpus="-")
        self.manager.complete_job(node.index, exit_code=1)

    def _launch(self, node: _TaskNode, allocated: List[int]) -> None:
        env = os.environ.copy()
        if allocated:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, allocated))
        env["PYTHONUNBUFFERED"] = "1"

        # Yield file: child writes "true" to release its GPU/CPU
        # accounting while keeping the subprocess running (e.g. so
        # FinetunedModel can let another GPU job start during its
        # upload). Mirrors ``batch_local``'s worker contract.
        yield_file = os.path.join(self.output_dir, f"yield_{node.index}.txt")
        with open(yield_file, "w") as yf:
            yf.write("false")
        env["BATCH_LOCAL_YIELD_FILE"] = yield_file

        out_path = os.path.join(self.output_dir, f"{node.index}.txt")
        f = open(out_path, "w")
        f.write("=" * 60 + "\n")
        f.write(f"Job #{node.index}\n")
        if node.task.artifact is not None:
            f.write(
                f"Artifact: {node.task.artifact.__class__.__name__} "
                f"({node.task.artifact.relpath})\n"
            )
        f.write(
            "GPUs: "
            + (",".join(map(str, allocated)) if allocated else "N/A")
            + "\n"
        )
        f.write(f"CPUs: {node.cpu_count}\n")
        f.write(f"Started: {datetime.now().isoformat()}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            ["bash", node.script_path],  # type: ignore[list-item]
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        try:
            fd = proc.stdout.fileno()  # type: ignore[union-attr]
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        except Exception:
            pass

        gpu_str = ",".join(map(str, allocated)) if allocated else "-"
        self.manager.start_job(node.index, worker_id=node.index, gpus=gpu_str)
        self.running.append(
            {
                "node": node,
                "process": proc,
                "gpus": list(allocated),
                "cpus": node.cpu_count,
                "output_file": f,
                "output_buffer": "",
                "yield_file": yield_file,
                "yielded": False,
                "last_yield_check": 0.0,
            }
        )

    # -- output draining + reaping ------------------------------------------
    def _poll_running(self) -> None:
        with self.lock:
            entries = list(self.running)

        for entry in entries:
            proc = entry["process"]
            node = entry["node"]
            of = entry["output_file"]

            # Drain available stdout bytes.
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 0)
                if ready:
                    chunk = proc.stdout.read()
                    if chunk:
                        entry["output_buffer"] += chunk
                        while "\n" in entry["output_buffer"]:
                            line, entry["output_buffer"] = entry[
                                "output_buffer"
                            ].split("\n", 1)
                            of.write(line + "\n")
                            of.flush()
                            self.manager.append_output(node.index, line)
            except (IOError, OSError):
                pass

            # Yield check: a running child can write "true" to its
            # ``BATCH_LOCAL_YIELD_FILE`` to release GPU/CPU accounting
            # while still finishing post-GPU work (e.g. uploads). Poll
            # at ~1 Hz to keep overhead low.
            if not entry["yielded"]:
                now = time.time()
                if now - entry["last_yield_check"] >= 1.0:
                    entry["last_yield_check"] = now
                    try:
                        with open(entry["yield_file"], "r") as yf:
                            yielded = yf.read().strip().lower() == "true"
                    except Exception:
                        yielded = False
                    if yielded:
                        with self.lock:
                            entry["yielded"] = True
                            entry["gpus"] = []
                            entry["cpus"] = 0
                        self.manager.yield_job(node.index)

            ret = proc.poll()
            if ret is not None:
                # Final drain.
                try:
                    remaining = proc.stdout.read()
                    if remaining:
                        entry["output_buffer"] += remaining
                except (IOError, OSError):
                    pass
                if entry["output_buffer"]:
                    for line in entry["output_buffer"].splitlines():
                        of.write(line + "\n")
                        of.flush()
                        self.manager.append_output(node.index, line)
                of.write("\n" + "=" * 60 + "\n")
                of.write(f"Finished: {datetime.now().isoformat()}\n")
                of.write(f"Exit code: {ret}\n")
                of.write("=" * 60 + "\n")
                of.close()

                with self.lock:
                    if entry in self.running:
                        self.running.remove(entry)
                    self.manager.complete_job(node.index, ret)
                    if ret == 0:
                        self.completed_ids.add(node.artifact_id)
                    else:
                        self.failed_ids.add(node.artifact_id)

    # -- abort (Ctrl-C handler) ---------------------------------------------
    def abort(self) -> None:
        with self.lock:
            self.aborted = True
            for entry in list(self.running):
                try:
                    entry["process"].terminate()
                except Exception:
                    pass

    # -- main entry points ---------------------------------------------------
    def run(self, use_dashboard: bool = True, refresh_rate: float = 0.1) -> None:
        thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        thread.start()

        try:
            if use_dashboard:
                self._run_dashboard(refresh_rate)
            else:
                self._run_plain()
        except KeyboardInterrupt:
            self.abort()

        # Wait for the scheduler to drain in-flight processes.
        thread.join(timeout=30.0)

        stats = self.manager.stats
        print("", file=sys.stderr)
        print(
            f"Completed: {stats['completed']}  Failed: {stats['errored']}  "
            f"Output: {self.output_dir}",
            file=sys.stderr,
        )
        if stats["errored"] > 0:
            sys.exit(1)

    def _run_plain(self) -> None:
        # Periodically print short status; honour Ctrl-C.
        last_state: tuple = ()
        while not self.done_event.is_set():
            self.done_event.wait(timeout=2.0)
            stats = self.manager.stats
            state = (stats["pending"], stats["running"], stats["completed"], stats["errored"])
            if state != last_state:
                print(
                    f"[runlocal] pending={stats['pending']} "
                    f"running={stats['running']} "
                    f"done={stats['completed']} "
                    f"failed={stats['errored']}",
                    file=sys.stderr,
                )
                last_state = state

    def _run_dashboard(self, refresh_rate: float) -> None:
        # Lazy imports keep the heavy `rich` / terminal stack out of the
        # rest of the experiments CLI.
        from rich.console import Console
        from rich.live import Live
        from .scripts.batch_local import Dashboard

        console = Console()
        # Estimate panel height: max possible concurrent tasks is bounded by
        # GPUs / smallest-GPU-task and CPUs / smallest-CPU-task. Use a
        # generous estimate; the dashboard scrolls if exceeded.
        gpu_costs = [n.gpu_count for n in self.nodes if n.gpu_count > 0]
        if gpu_costs and self.gpu_pool:
            num_parallel = max(1, len(self.gpu_pool) // min(gpu_costs))
        else:
            num_parallel = max(1, min(self.max_cpus, len(self.nodes)))
        dashboard = Dashboard(self.manager, self.output_dir, num_parallel, console)

        old_settings = None
        tty_fd: Optional[int] = None
        keyboard_enabled = False
        try:
            import termios

            tty_fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            old_settings = termios.tcgetattr(tty_fd)
            new_settings = termios.tcgetattr(tty_fd)
            new_settings[3] = new_settings[3] & ~termios.ICANON & ~termios.ECHO
            termios.tcsetattr(tty_fd, termios.TCSANOW, new_settings)
            keyboard_enabled = True
        except Exception:
            pass

        def check_keyboard() -> List[str]:
            if not keyboard_enabled or tty_fd is None:
                return []
            try:
                ready, _, _ = select.select([tty_fd], [], [], 0)
                if not ready:
                    return []
                data = os.read(tty_fd, 256).decode("utf-8", errors="ignore")
                keys: List[str] = []
                i = 0
                while i < len(data):
                    if (
                        data[i] == "\x1b"
                        and i + 1 < len(data)
                        and data[i + 1] == "["
                    ):
                        i += 2
                        param = ""
                        while i < len(data) and (data[i].isdigit() or data[i] == ";"):
                            param += data[i]
                            i += 1
                        if i < len(data):
                            final = data[i]
                            i += 1
                            if final == "A":
                                keys.append("up")
                            elif final == "B":
                                keys.append("down")
                            elif final == "~":
                                if param == "5":
                                    keys.append("page_up")
                                elif param == "6":
                                    keys.append("page_down")
                            elif final == "H":
                                keys.append("home")
                            elif final == "F":
                                keys.append("end")
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
                while not self.done_event.is_set():
                    page_size = max(dashboard.get_output_panel_height() // 2, 5)
                    dirty = False
                    for k in check_keyboard():
                        if k == "up":
                            dashboard.navigate_job(-1)
                            dirty = True
                        elif k == "down":
                            dashboard.navigate_job(1)
                            dirty = True
                        elif k == "page_up":
                            self.manager.scroll_output(page_size)
                            dirty = True
                        elif k == "page_down":
                            self.manager.scroll_output(-page_size)
                            dirty = True
                        elif k == "home":
                            self.manager.scroll_output(999999)
                            dirty = True
                        elif k == "end":
                            self.manager.scroll_output(-999999)
                            dirty = True

                    now = time.monotonic()
                    if dirty or now - last_render >= refresh_rate:
                        live.update(dashboard.generate(), refresh=True)
                        last_render = now
                    time.sleep(0.015)

                live.update(dashboard.generate(), refresh=True)
                time.sleep(0.3)
        finally:
            try:
                if old_settings is not None and tty_fd is not None:
                    import termios

                    termios.tcsetattr(tty_fd, termios.TCSADRAIN, old_settings)
                if tty_fd is not None:
                    os.close(tty_fd)
            except Exception:
                pass
