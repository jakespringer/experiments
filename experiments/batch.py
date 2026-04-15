"""ArtifactBatch — run multiple artifacts in parallel within a single Slurm job."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .artifact import Artifact, ArtifactSet
from .executor import TaskBlock


class _RawShellBlock(TaskBlock):
    """A TaskBlock that emits a shell line verbatim.

    Unlike ``CommandTaskBlock``, this is not wrapped in echo-escaping by the
    script builder, so bash syntax like arrays, process substitution, and
    heredocs pass through unmangled.
    """

    def __init__(self, line: str) -> None:
        self._line = line

    def execute(self) -> str:
        return self._line


# ---------------------------------------------------------------------------
# Resource parsing / combining helpers
# ---------------------------------------------------------------------------

def _parse_mem(mem_str: str) -> int:
    """Parse memory string like '128G', '64000M' to megabytes."""
    mem_str = str(mem_str).strip().upper()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$', mem_str)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2)
    multipliers = {'': 1, 'K': 1 / 1024, 'M': 1, 'G': 1024, 'T': 1024 * 1024}
    return int(val * multipliers.get(unit, 1))


def _format_mem(mb: int) -> str:
    """Format megabytes back to human-readable string."""
    if mb >= 1024 and mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"


def _parse_gpu_count(gpu_val: Any) -> int:
    """Extract GPU count from various formats: '4', 'A100:4', 'gpu:4', 'gpu:a100:4'."""
    if not gpu_val:
        return 0
    s = str(gpu_val)
    parts = s.split(':')
    try:
        return int(parts[-1])
    except ValueError:
        return 0


def _parse_gpu_type(gpu_val: Any) -> str | None:
    """Extract GPU type if present: 'A100:4' -> 'A100', '4' -> None."""
    if not gpu_val:
        return None
    s = str(gpu_val)
    parts = s.split(':')
    if len(parts) >= 2:
        for p in parts[:-1]:
            if p.lower() != 'gpu':
                return p
    return None


def combine_requirements(req_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine resource requirements from multiple artifacts.

    Strategy:
    - GPUs, CPUs, memory: sum
    - Partition, time, constraint, etc.: take from first non-None
    - gres: parse and sum GPU counts
    """
    if not req_list:
        return {}

    req_list = [r for r in req_list if r] or [{}]
    combined: Dict[str, Any] = {}

    # --- Summable: GPUs (--gpus flag) ---
    gpu_counts: List[int] = []
    gpu_type: str | None = None
    for r in req_list:
        if 'gpus' in r:
            gpu_counts.append(_parse_gpu_count(r['gpus']))
            if gpu_type is None:
                gpu_type = _parse_gpu_type(r['gpus'])
    if gpu_counts:
        total_gpus = sum(gpu_counts)
        combined['gpus'] = f"{gpu_type}:{total_gpus}" if gpu_type else total_gpus

    # --- Summable: gres ---
    gres_gpu_counts: List[int] = []
    gres_gpu_type: str | None = None
    for r in req_list:
        if 'gres' in r:
            parts = str(r['gres']).split(':')
            if parts[0] == 'gpu':
                if len(parts) == 2:
                    try:
                        gres_gpu_counts.append(int(parts[1]))
                    except ValueError:
                        pass
                elif len(parts) >= 3:
                    if gres_gpu_type is None:
                        gres_gpu_type = parts[1]
                    try:
                        gres_gpu_counts.append(int(parts[2]))
                    except ValueError:
                        pass
    if gres_gpu_counts:
        total_gres = sum(gres_gpu_counts)
        if gres_gpu_type:
            combined['gres'] = f"gpu:{gres_gpu_type}:{total_gres}"
        else:
            combined['gres'] = f"gpu:{total_gres}"

    # --- Summable: CPUs ---
    cpu_counts = [r.get('cpus', r.get('cpus_per_task', 0)) for r in req_list]
    cpu_counts = [c for c in cpu_counts if c]
    if cpu_counts:
        # Use whichever key the children use
        cpu_key = 'cpus' if any('cpus' in r for r in req_list) else 'cpus_per_task'
        combined[cpu_key] = sum(cpu_counts)

    # --- Summable: Memory ---
    mem_values = [r.get('mem') for r in req_list if r.get('mem')]
    if mem_values:
        total_mb = sum(_parse_mem(str(m)) for m in mem_values)
        combined['mem'] = _format_mem(total_mb)

    # --- Non-summable: take first ---
    for key in ('partition', 'time', 'time_min', 'account', 'qos', 'constraint',
                'exclude', 'nodelist', 'requeue', 'signal', 'nodes',
                'mail_type', 'mail_user'):
        for r in req_list:
            if key in r:
                combined[key] = r[key]
                break

    return combined


# ---------------------------------------------------------------------------
# ArtifactBatch
# ---------------------------------------------------------------------------

class ArtifactBatch(Artifact):
    """Groups multiple artifacts to run in parallel within a single Slurm job.

    Usage::

        batch = ArtifactBatch([artifact_a, artifact_b, artifact_c])
        executor.stage('my_stage', ArtifactSet([batch]))

    Or use ``--autobatch N`` on the CLI to create batches automatically.
    """

    # We manually manage fields instead of using @dataclass(frozen=True)
    # because the base Artifact class is not a dataclass.

    def __init__(self, artifacts: list | tuple, batch_index: int = 0):
        self._batch_artifacts: Tuple[Artifact, ...] = tuple(artifacts)
        self._batch_index: int = batch_index

    # -- dict / hash support --------------------------------------------------

    def as_dict(self) -> Dict[str, Any]:
        # Represent child artifacts as an ordered string of hashes so the
        # base get_hash() preserves insertion order (the default tuple
        # hashing sorts elements, which would make [A,B] == [B,A]).
        ordered_hashes = '|'.join(a.get_hash() for a in self._batch_artifacts)
        return {
            'artifact_order': ordered_hashes,
            'batch_index': self._batch_index,
        }

    @property
    def relpath(self) -> str:
        return f"ArtifactBatch/{self.get_hash()}"

    # -- skip / exists --------------------------------------------------------

    @property
    def exists(self) -> bool:
        return all(a.exists for a in self._batch_artifacts)

    def should_skip(self) -> bool:
        return all(a.should_skip() for a in self._batch_artifacts)

    # -- requirements ---------------------------------------------------------

    def get_requirements(self) -> Dict[str, Any]:
        """Combine requirements from all non-skipped children."""
        reqs = []
        for a in self._batch_artifacts:
            # Only request resources for children that still need to run.
            if a.should_skip():
                continue
            if hasattr(a, 'get_requirements'):
                reqs.append(a.get_requirements())
            else:
                reqs.append({})
        if not reqs:
            # All children are done — return minimal requirements.
            return {}
        return combine_requirements(reqs)

    # -- dependency discovery -------------------------------------------------

    def get_direct_dependencies(self) -> List[Artifact]:
        """Return transitive dependencies of children (not the children themselves)."""
        from .executor import _find_artifact_dependencies

        children_set = set(id(a) for a in self._batch_artifacts)
        deps: List[Artifact] = []
        seen: set = set()
        for child in self._batch_artifacts:
            for attr_value in vars(child).values():
                for dep in _find_artifact_dependencies(attr_value):
                    dep_id = id(dep)
                    if dep_id not in children_set and dep_id not in seen:
                        seen.add(dep_id)
                        deps.append(dep)
        return deps

    # -- construct (parallel launcher) ----------------------------------------

    def construct(self, task: 'Task'):
        from .executor import Task as _Task, _artifact_experiment_conf, _safe_json_dumps, dquote
        import json
        import base64 as _b64

        # If every child is already done, emit nothing.
        if self.should_skip():
            task.run_command('echo "ArtifactBatch: all children already exist, nothing to do"')
            return

        # Build child commands by constructing each child into its own Task.
        # Children that should be skipped are recorded but get no commands,
        # so they don't consume a GPU slot or launch a process.
        child_scripts: List[List[str]] = []
        child_skip: List[bool] = []
        for child in self._batch_artifacts:
            if child.should_skip():
                child_scripts.append([])
                child_skip.append(True)
                continue
            child_skip.append(False)
            child_task = _Task(
                artifact_path=task.artifact_path,
                code_path=task.code_path,
                artifact=child,
                gs_path=task.gs_path,
                log_dir=task.log_dir,
            )
            child.construct(child_task)
            cmds = []
            for block in child_task.blocks:
                cmd = block.execute()
                if cmd:
                    cmds.append(cmd)
            child_scripts.append(cmds)

        # Compute GPU partitioning: (offset, count) per child — skip children
        # that are already done so they don't consume GPU slots.
        gpu_offsets: List[Tuple[int, int]] = []
        offset = 0
        for i, child in enumerate(self._batch_artifacts):
            if child_skip[i]:
                gpu_offsets.append((0, 0))
                continue
            reqs = child.get_requirements() if hasattr(child, 'get_requirements') else {}
            child_gpus = _parse_gpu_count(reqs.get('gpus', 0))
            if child_gpus == 0:
                gres_val = str(reqs.get('gres', ''))
                if gres_val.startswith('gpu'):
                    child_gpus = _parse_gpu_count(gres_val)
            gpu_offsets.append((offset, child_gpus))
            offset += child_gpus

        total_gpus = offset

        # Use the configured log directory (from the executor / global config),
        # falling back to ~/.experiments/logs.
        log_dir = task.log_dir or "${HOME}/.experiments/logs"

        # Helper: emit a raw shell line directly into the task.
        def emit(line: str) -> None:
            task.blocks.append(_RawShellBlock(line))

        # Build children config for the Python batch runner.
        # Each child's commands are serialized as a base64-encoded bash script.
        children_config = []
        for i, (child, cmds) in enumerate(zip(self._batch_artifacts, child_scripts)):
            if child_skip[i] or not cmds:
                continue

            gpu_offset, gpu_count = gpu_offsets[i]

            # Build the child's shell script
            script_lines = ['#!/bin/bash', 'set -e']

            # Per-child experiment config
            exp_conf = _artifact_experiment_conf(child)
            exp_conf_str = _safe_json_dumps(exp_conf)
            script_lines.append(f'export EXPERIMENTS_EXPERIMENT_CONF={dquote(exp_conf_str)}')

            # All child commands
            for cmd in cmds:
                script_lines.append(cmd)

            script = '\n'.join(script_lines) + '\n'

            children_config.append({
                'index': i,
                'relpath': child.relpath,
                'script_b64': _b64.b64encode(script.encode()).decode(),
                'gpu_offset': gpu_offset,
                'gpu_count': gpu_count,
            })

        if not children_config:
            task.run_command('echo "ArtifactBatch: all children skipped or have no commands"')
            return

        config_b64 = _b64.b64encode(json.dumps(children_config).encode()).decode()

        # --- Emit the bash wrapper that invokes the Python batch runner ---

        emit('set +eu')
        emit('')

        # Log skipped children
        for i, child in enumerate(self._batch_artifacts):
            if child_skip[i]:
                emit(f'# child {i}: {child.relpath} (SKIPPED)')

        # Discover GPUs
        if total_gpus > 0:
            emit("ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null "
                 "| paste -sd ',')")
            emit('if [ -z "$ALL_GPUS" ]; then')
            emit(f'  echo "[batch] WARNING: nvidia-smi returned no GPUs but {total_gpus} were requested" >&2')
            emit('fi')
        else:
            emit('ALL_GPUS=""')
        emit('')

        # Export runtime variables for the Python launcher
        emit(f'export BATCH_LOG_DIR="{log_dir}"')
        emit('export BATCH_JOB_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"')
        emit('export BATCH_ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"')
        emit('export BATCH_ALL_GPUS="${ALL_GPUS:-}"')
        emit('')

        # Log child log paths to the main Slurm log for debugging
        emit(f'echo "[batch] Log directory: {log_dir}" >&2')
        for cfg in children_config:
            emit(f'echo "[batch] child {cfg["index"]}: {cfg["relpath"]}'
                 f' -> {log_dir}/batch_${{BATCH_JOB_ID}}_${{BATCH_ARRAY_ID}}_child_{cfg["index"]}.out" >&2')
        emit('')

        # Write config to a temp file via base64 decode (single line, no heredoc)
        emit('BATCH_CONFIG=$(mktemp /tmp/batch_config_XXXXXX.json)')
        emit(f'echo "{config_b64}" | base64 -d > "$BATCH_CONFIG"')
        emit('trap \'rm -f "$BATCH_CONFIG"\' EXIT')
        emit('')

        # Run the Python batch runner
        emit('python -m experiments.scripts.batch_runner "$BATCH_CONFIG"')
        emit('BATCH_RC=$?')
        emit('')

        # Cleanup (also handled by trap)
        emit('rm -f "$BATCH_CONFIG"')
        emit('')

        # Propagate failure
        emit('if [ "$BATCH_RC" -ne 0 ]; then')
        emit('  exit "$BATCH_RC"')
        emit('fi')
        emit('')
        emit('# Restore strict mode')
        emit('set -eu')
