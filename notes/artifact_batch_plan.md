# ArtifactBatch Implementation Plan

## Goal

Implement `ArtifactBatch` — a mechanism that groups N artifacts together so they run **in parallel within a single Slurm job**, combining their resource requirements (summing GPUs, CPUs, memory). This enables efficient packing of small jobs (e.g., 8 single-GPU finetuning runs) into one multi-GPU Slurm allocation.

---

## 1. Design Overview

### Core Idea

`ArtifactBatch` is an `Artifact` subclass that wraps a list of artifacts. From the framework's perspective it looks like a single artifact: it has one `construct()`, one `get_requirements()`, one `relpath`, and it occupies one slot in a Slurm array job. Internally, its `construct()` emits a parallel launcher script that runs all child artifacts simultaneously, each on its own subset of GPUs.

```
ArtifactBatch([artifact_0, artifact_1, ..., artifact_7])
  ├── get_requirements() → sum of all children's requirements
  ├── construct(task)    → parallel launcher script
  ├── relpath            → "ArtifactBatch/<hash>"
  ├── exists             → all children exist
  └── should_skip()      → all children should_skip
```

### Key Properties

- **Transparent to the framework**: ArtifactBatch *is* an Artifact. Stages, ArtifactSets, topological sort, dependency tracking — everything works unchanged.
- **Dependencies preserved**: If any child has dependencies, the batch inherits them. The batch's construct accesses parent artifact paths the same way children would.
- **Parallel execution within a single job**: Children run as background processes with `CUDA_VISIBLE_DEVICES` partitioning, with a wait-all at the end.
- **Per-child logging**: Each child's stdout/stderr is tee'd to a separate log file, enabling `jobcat jobid_arrayidx_childidx` lookups.

---

## 2. Implementation Steps

### Step 1: `ArtifactBatch` class in `experiments/batch.py`

Create a new file `experiments/batch.py` with the `ArtifactBatch` class.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from .artifact import Artifact, ArtifactSet
from .executor import Task, IgnoreHash

@dataclass(frozen=True)
class ArtifactBatch(Artifact):
    """Groups multiple artifacts to run in parallel within a single Slurm job."""

    artifacts: Tuple[Artifact, ...]  # frozen → must be tuple
    batch_index: int = 0             # disambiguator when auto-batching creates multiple batches

    def __init__(self, artifacts, batch_index=0):
        # Frozen dataclass workaround for converting list → tuple
        object.__setattr__(self, 'artifacts', tuple(artifacts))
        object.__setattr__(self, 'batch_index', batch_index)

    @property
    def relpath(self) -> str:
        return f"ArtifactBatch/{self.get_hash()}"

    @property
    def exists(self) -> bool:
        return all(a.exists for a in self.artifacts)

    def should_skip(self) -> bool:
        return all(a.should_skip() for a in self.artifacts)

    def get_requirements(self) -> Dict[str, Any]:
        """Combine requirements from all children by summing resources."""
        return combine_requirements([a.get_requirements() for a in self.artifacts])

    def construct(self, task: Task):
        """Build a parallel launcher script that runs all children concurrently."""
        # Details in Step 3 below
        ...
```

**Key design decisions:**
- Use `Tuple[Artifact, ...]` (not `List`) since the dataclass is frozen and we need hashability.
- Override `__init__` to accept `list` and auto-convert to `tuple`.
- `batch_index` field provides uniqueness when auto-batching creates multiple batches from the same artifact types.

### Step 2: Resource Combination (`combine_requirements`)

In `experiments/batch.py`, implement the resource-combining logic:

```python
import re

def _parse_mem(mem_str: str) -> int:
    """Parse memory string like '128G', '64000M' to megabytes."""
    mem_str = str(mem_str).strip().upper()
    m = re.match(r'^(\d+(?:\.\d+)?)\s*([KMGT]?)B?$', mem_str)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2)
    multipliers = {'': 1, 'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024*1024}
    return int(val * multipliers.get(unit, 1))

def _format_mem(mb: int) -> str:
    """Format megabytes back to human-readable string."""
    if mb >= 1024 and mb % 1024 == 0:
        return f"{mb // 1024}G"
    return f"{mb}M"

def _parse_gpu_count(gpu_val) -> int:
    """Extract GPU count from various formats: '4', 'A100:4', 'gpu:4', 'gpu:a100:4'."""
    s = str(gpu_val)
    parts = s.split(':')
    # Try last part as integer
    try:
        return int(parts[-1])
    except ValueError:
        return 0

def _parse_gpu_type(gpu_val) -> str | None:
    """Extract GPU type if present: 'A100:4' → 'A100', '4' → None."""
    s = str(gpu_val)
    parts = s.split(':')
    if len(parts) >= 2:
        # 'A100:4' or 'gpu:A100:4'
        for p in parts[:-1]:
            if p.lower() != 'gpu':
                return p
    return None

def combine_requirements(req_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Combine resource requirements from multiple artifacts.

    Strategy:
    - GPUs, CPUs, memory: sum
    - Partition, time, constraint, account, qos: take from first (all should match for batching)
    - gres: parse and sum GPU counts
    - Everything else: take from first non-None value
    """
    if not req_list:
        return {}

    # Filter out empty dicts
    req_list = [r for r in req_list if r] or [{}]

    combined = {}

    # === Summable resources ===

    # GPUs (--gpus flag)
    gpu_counts = []
    gpu_type = None
    for r in req_list:
        if 'gpus' in r:
            gpu_counts.append(_parse_gpu_count(r['gpus']))
            if gpu_type is None:
                gpu_type = _parse_gpu_type(r['gpus'])
    if gpu_counts:
        total_gpus = sum(gpu_counts)
        if gpu_type:
            combined['gpus'] = f"{gpu_type}:{total_gpus}"
        else:
            combined['gpus'] = total_gpus

    # gres (--gres flag) — parse gpu:TYPE:COUNT or gpu:COUNT
    gres_gpu_counts = []
    gres_gpu_type = None
    for r in req_list:
        if 'gres' in r:
            gres_str = str(r['gres'])
            parts = gres_str.split(':')
            if parts[0] == 'gpu':
                if len(parts) == 2:
                    try: gres_gpu_counts.append(int(parts[1]))
                    except: pass
                elif len(parts) >= 3:
                    if gres_gpu_type is None:
                        gres_gpu_type = parts[1]
                    try: gres_gpu_counts.append(int(parts[2]))
                    except: pass
    if gres_gpu_counts:
        total_gres_gpus = sum(gres_gpu_counts)
        if gres_gpu_type:
            combined['gres'] = f"gpu:{gres_gpu_type}:{total_gres_gpus}"
        else:
            combined['gres'] = f"gpu:{total_gres_gpus}"

    # CPUs
    cpu_key = 'cpus' if any('cpus' in r for r in req_list) else 'cpus_per_task'
    cpu_counts = [r.get('cpus', r.get('cpus_per_task', 0)) for r in req_list]
    cpu_counts = [c for c in cpu_counts if c]
    if cpu_counts:
        combined[cpu_key] = sum(cpu_counts)

    # Memory
    mem_values = [r.get('mem') for r in req_list if r.get('mem')]
    if mem_values:
        total_mb = sum(_parse_mem(str(m)) for m in mem_values)
        combined['mem'] = _format_mem(total_mb)

    # === Non-summable: take first ===
    first_with = lambda key: next((r[key] for r in req_list if key in r), None)

    for key in ['partition', 'time', 'time_min', 'account', 'qos', 'constraint',
                'exclude', 'nodelist', 'requeue', 'signal', 'nodes',
                'mail_type', 'mail_user']:
        val = first_with(key)
        if val is not None:
            combined[key] = val

    return combined
```

### Step 3: Parallel Launcher in `construct()`

The `construct()` method generates a bash script that:
1. Determines the total number of GPUs allocated to this Slurm job
2. Partitions GPUs across child artifacts based on each child's GPU needs
3. Launches each child's commands as a background process with isolated `CUDA_VISIBLE_DEVICES`
4. Tees each child's output to a separate log file (for `jobcat` integration)
5. Waits for all children, tracking exit codes
6. Exits with non-zero if any child failed

```python
def construct(self, task: Task):
    n = len(self.artifacts)

    # Create a sub-task for each child to get its commands
    child_scripts = []
    for i, child_artifact in enumerate(self.artifacts):
        child_task = Task(
            artifact_path=task.artifact_path,
            code_path=task.code_path,
            artifact=child_artifact,
            gs_path=task.gs_path,
        )
        child_artifact.construct(child_task)

        # Collect the shell commands from the child task
        commands = []
        for block in child_task.blocks:
            cmd = block.execute()
            if cmd:
                commands.append(cmd)
        child_scripts.append(commands)

    # Compute GPU partitioning
    gpu_offsets = []
    offset = 0
    for child_artifact in self.artifacts:
        reqs = child_artifact.get_requirements() if hasattr(child_artifact, 'get_requirements') else {}
        child_gpus = _parse_gpu_count(reqs.get('gpus', reqs.get('gres', 0)))
        if child_gpus == 0:
            # Check gres
            gres = reqs.get('gres', '')
            parts = str(gres).split(':')
            if len(parts) >= 2:
                try: child_gpus = int(parts[-1])
                except: child_gpus = 0
        gpu_offsets.append((offset, child_gpus))
        offset += child_gpus

    # Build the experiment config export for each child
    from .executor import _artifact_experiment_conf, _safe_json_dumps, dquote

    # Generate the batch log directory
    batch_log_dir = f"{task.artifact_path}/{self.relpath}/logs"
    task.ensure_directory(batch_log_dir)

    # Build parallel launcher script
    pids_var = "BATCH_PIDS"
    exit_codes_var = "BATCH_EXIT_CODES"

    # Initialize tracking
    task.run_command(f"{pids_var}=()")
    task.run_command(f"{exit_codes_var}=()")

    # Determine all available GPUs from SLURM allocation
    task.run_command('ALL_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | tr "\\n" "," | sed "s/,$//")')
    task.run_command('IFS="," read -ra GPU_ARRAY <<< "$ALL_GPUS"')

    for i, (child_artifact, commands) in enumerate(zip(self.artifacts, child_scripts)):
        if not commands:
            continue

        gpu_offset, gpu_count = gpu_offsets[i]

        # Build GPU index list for this child
        # E.g., if offset=2, count=2, picks GPU_ARRAY[2] and GPU_ARRAY[3]
        gpu_indices = " ".join(f"${{GPU_ARRAY[{gpu_offset + g}]}}" for g in range(gpu_count)) if gpu_count > 0 else ""

        log_file = f"{batch_log_dir}/child_{i}.log"

        # Export per-child experiment config
        exp_conf = _artifact_experiment_conf(child_artifact)
        exp_conf_str = _safe_json_dumps(exp_conf)

        # Build subshell: set CUDA_VISIBLE_DEVICES, export config, run commands
        subshell_parts = []
        if gpu_count > 0:
            # Dynamically compute CSV of GPU indices
            subshell_parts.append(
                f'CHILD_GPUS=({gpu_indices}); '
                f'export CUDA_VISIBLE_DEVICES=$(IFS=","; echo "${{CHILD_GPUS[*]}}")'
            )
        else:
            subshell_parts.append('export CUDA_VISIBLE_DEVICES=""')

        subshell_parts.append(f'export EXPERIMENTS_EXPERIMENT_CONF={dquote(exp_conf_str)}')

        for cmd in commands:
            subshell_parts.append(cmd)

        subshell_script = " && ".join(subshell_parts)

        # Launch in background, tee output to log file
        task.run_command(f'( {subshell_script} ) > >(tee -a {dquote(log_file)}) 2>&1 &')
        task.run_command(f'{pids_var}+=($!)')

    # Wait for all children and collect exit codes
    task.run_command(f"""
BATCH_FAILED=0
for pid in "${{{pids_var}[@]}}"; do
    wait "$pid" || BATCH_FAILED=1
done
if [ "$BATCH_FAILED" -ne 0 ]; then
    echo "ERROR: One or more batch children failed" >&2
    exit 1
fi
""".strip())
```

**Important implementation notes:**

- The subshell approach `( ... ) &` ensures each child runs in its own process group with its own `CUDA_VISIBLE_DEVICES`.
- `tee` captures output to per-child log files while still streaming to the job's main stdout (visible in Slurm logs).
- The wait loop at the end ensures the Slurm job doesn't exit until all children finish, and propagates failure.

### Step 4: Dependency Discovery for Batched Artifacts

The framework's `_find_artifact_dependencies()` already recursively scans attributes. Since `ArtifactBatch.artifacts` is a tuple of `Artifact` instances, the existing traversal in `_find_artifact_dependencies` will discover them via the `(list, tuple, set)` branch:

```python
elif isinstance(value, (list, tuple, set)):
    for v in value:
        yield from _find_artifact_dependencies(v)
```

**This means dependencies just work** — if child artifacts reference other artifacts, the topological sort correctly places the batch after those dependencies.

**However**, there is a subtlety: the child artifacts themselves should NOT also appear as standalone artifacts in any stage. They are "absorbed" into the batch. The batch replaces them. If the user manually creates batches, they should register the batches (not the children) in stages. The `--autobatch` flag handles this automatically (see Step 6).

### Step 5: Integration with Existing Structure

#### 5a. Update `__init__.py` exports

Add `ArtifactBatch` and `combine_requirements` to `experiments/__init__.py`:

```python
from .batch import ArtifactBatch, combine_requirements
```

And add to `__all__`.

#### 5b. `ArtifactBatch` works with `ArtifactSet` and `stage()`

Since `ArtifactBatch` is an `Artifact`, it can be used anywhere an artifact is expected:

```python
# Manual batch creation
batch = ArtifactBatch([artifact1, artifact2, artifact3, artifact4])
executor.stage('finetune', ArtifactSet([batch]))

# Or mixed with normal artifacts
executor.stage('finetune', ArtifactSet([batch1, batch2, single_artifact]))
```

#### 5c. The `exists` and `should_skip` behavior

- `ArtifactBatch.exists` → True iff ALL children exist
- `ArtifactBatch.should_skip()` → True iff ALL children should_skip

This means:
- If 7 of 8 children are done, the batch still runs (but we could optimize later to skip completed children within the batch)
- The `--rerun` flag bypasses this as usual

**Future optimization**: In `construct()`, wrap each child's commands in `if ! child.exists; then ... fi` so completed children are skipped even when the batch runs. This is a nice-to-have.

### Step 6: `--autobatch N` CLI Flag

Add `--autobatch N` to the launch/drylaunch/relaunch commands in `cli.py`.

**Implementation in `ExperimentCLI.launch()`:**

After resolving stages and before calling `executor.execute()`, transform the executor's stages by batching:

```python
def _apply_autobatch(self, batch_size: int):
    """Replace artifacts in each stage with ArtifactBatch groups of size batch_size."""
    from .batch import ArtifactBatch

    for stage_name, artifacts in self.executor._stages.items():
        if len(artifacts) <= 1:
            continue

        batched = []
        for i in range(0, len(artifacts), batch_size):
            chunk = artifacts[i:i + batch_size]
            if len(chunk) == 1:
                batched.append(chunk[0])  # Don't wrap single artifacts
            else:
                batched.append(ArtifactBatch(chunk, batch_index=i // batch_size))

        self.executor._stages[stage_name] = batched
```

**CLI argument addition** (in all launch/drylaunch/relaunch parsers):

```python
parser.add_argument(
    '--autobatch',
    type=int,
    metavar='N',
    help='Automatically batch every N artifacts within each stage to run in parallel on a single job'
)
```

**Integration point** — in `ExperimentCLI.launch()`, after resolving stages but before calling `self.executor.execute()`:

```python
if autobatch is not None and autobatch > 1:
    self._apply_autobatch(autobatch)
```

**Important**: `--autobatch` only groups artifacts **within the same stage**. Cross-stage batching doesn't make sense because stages have dependency ordering. Also, auto-batching must happen **before** the topological sort, because the batch objects need to be in the stages for the topo sort to correctly handle their dependencies.

**Dependency handling in autobatch**: When we batch artifacts, their child dependencies are still discovered. But the child artifacts themselves are no longer in any stage — only the batch is. The batch references them via its `artifacts` tuple, which means the topo sort sees the batch depending on whatever the children depend on. The children's own dependencies (e.g., a `PretrainedModel` that a `FinetunedModel` depends on) are found via recursive attribute scanning of the batch's children.

However, there's a subtlety: `_find_artifact_dependencies` scans `vars(artifact)` on the batch, which yields `{'artifacts': (child1, child2, ...)}`. It then enters the tuple branch and discovers child1, child2, etc. But child1, child2 are NOT in any stage (the batch replaced them). So `_build_dependency_graph` would raise `ValueError: ... not in the artifact set`.

**Fix**: We need to handle this. Two options:

**(A) Don't yield the child artifacts themselves as dependencies, only their transitive dependencies.** This means the batch shouldn't "depend on" its own children — it *contains* them. We need `_find_artifact_dependencies` to special-case `ArtifactBatch`: when scanning an `ArtifactBatch`, skip its `artifacts` field (since those are contained, not depended upon), and instead scan the children's attributes for transitive dependencies.

**(B) Simpler: override `as_dict()` on `ArtifactBatch` to exclude the `artifacts` field from dependency scanning**, and instead manually add transitive dependencies.

Actually, the cleanest approach:

**(C) Override the dependency discovery on `ArtifactBatch`.** Add a method `get_dependencies()` that the framework calls instead of (or in addition to) attribute scanning. For the batch, it returns the union of all children's dependencies (excluding the children themselves).

Let me go with a simpler variant of (A): **modify `_find_artifact_dependencies`** to handle `ArtifactBatch` specially:

```python
def _find_artifact_dependencies(value: Any) -> Iterable[Artifact]:
    from .batch import ArtifactBatch

    if isinstance(value, ArtifactBatch):
        # Don't yield the batch's children as dependencies — they're contained, not depended upon.
        # Instead, yield the transitive dependencies of all children.
        for child in value.artifacts:
            for attr_value in vars(child).values():
                yield from _find_artifact_dependencies(attr_value)
    elif isinstance(value, Artifact):
        yield value
    elif isinstance(value, ArtifactSet):
        for item in value:
            yield from _find_artifact_dependencies(item)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _find_artifact_dependencies(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _find_artifact_dependencies(v)
```

Wait — but the batch IS an artifact and gets yielded when it appears as a field value on some other artifact. The issue is when the framework scans the batch's OWN attributes. Let me re-think:

The framework does this in `_build_dependency_graph`:
```python
for artifact in artifacts:  # artifacts = all unique artifacts across all stages
    for attr_value in vars(artifact).values():
        for dependency in _find_artifact_dependencies(attr_value):
            # dependency must be in artifact_by_id
```

When `artifact` is an `ArtifactBatch`, `vars(artifact)` yields `{'artifacts': (child1, child2, ...), 'batch_index': 0}`. Then `_find_artifact_dependencies` enters the tuple `(child1, child2, ...)`, finds each child as an `Artifact`, and yields them. But child1, child2 are NOT in `artifact_by_id` (only the batch is).

**Solution**: Override `_build_dependency_graph` behavior for `ArtifactBatch`. The cleanest way: **add a method to `Artifact` that returns its direct dependencies**, and override it in `ArtifactBatch`:

```python
# In Artifact base class:
def get_direct_dependencies(self) -> List['Artifact']:
    """Return artifacts that this artifact directly depends on.

    Default: scan all attribute values recursively.
    Override in subclasses (e.g., ArtifactBatch) for custom behavior.
    """
    deps = []
    for attr_value in vars(self).values():
        deps.extend(_find_artifact_dependencies(attr_value))
    return deps

# In ArtifactBatch:
def get_direct_dependencies(self) -> List[Artifact]:
    """Return transitive dependencies of all children (not the children themselves)."""
    deps = []
    for child in self.artifacts:
        for attr_value in vars(child).values():
            for dep in _find_artifact_dependencies(attr_value):
                # Only yield if it's not one of our own children
                if dep not in self.artifacts:
                    deps.append(dep)
    return deps
```

Then update `_build_dependency_graph` to use `artifact.get_direct_dependencies()` instead of inline scanning.

### Step 7: Per-Child Logging and `jobcat` Integration

#### Log File Layout

When an `ArtifactBatch` runs as Slurm job `12345` array index `3`, the Slurm log goes to:
```
~/.experiments/logs/tier-0_12345_3.out    # main job log (interleaved output from all children)
```

The batch's `construct()` creates per-child logs at:
```
<artifact_path>/ArtifactBatch/<hash>/logs/child_0.log
<artifact_path>/ArtifactBatch/<hash>/logs/child_1.log
...
<artifact_path>/ArtifactBatch/<hash>/logs/child_7.log
```

#### `jobcat` Extension

Extend the `jobcat` tool and CLI `cat` command to support a **third-level index**:

```
jobcat 12345_3         → shows main Slurm log for job 12345, array index 3 (as before)
jobcat 12345_3_0       → shows child 0's log within batch at array index 3
jobcat 12345_3_5       → shows child 5's log
```

**Implementation in `jobcat.py`**:

When the job_spec has two underscores (e.g., `12345_3_5`):
1. Parse as `job_id=12345`, `array_id=3`, `child_id=5`
2. Find the Slurm job log as usual to identify the batch
3. Look up the batch's artifact path from the job info
4. Read the child log file at `<artifact_path>/ArtifactBatch/<hash>/logs/child_{child_id}.log`

**Alternative approach** (simpler, preferred): Store child log paths in the job info when submitting. Or better yet, use a convention based on the Slurm log directory:

In `construct()`, instead of writing child logs to the artifact path, write them alongside the Slurm log:

```
~/.experiments/logs/tier-0_12345_3.out         # main log
~/.experiments/logs/tier-0_12345_3_child0.out  # child 0
~/.experiments/logs/tier-0_12345_3_child1.out  # child 1
...
```

This makes `jobcat` trivial — when given `12345_3_0`, it looks for `*_12345_3_child0.out` in the log directory.

**Even simpler**: The batch's `construct()` generates log paths using `$SLURM_JOB_ID` and `$SLURM_ARRAY_TASK_ID`:

```bash
BATCH_LOG_DIR="$LOG_DIR"  # same dir as SLURM output
CHILD_LOG="$BATCH_LOG_DIR/tier-X_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_child${i}.out"
```

But `construct()` doesn't know the Slurm job name or log dir at construct time. We can pass it through the task or use a known convention.

**Cleanest approach**: The batch's `construct()` computes the child log path from the SLURM_OUTPUT variable:

```bash
# In the generated script:
MAIN_LOG="$SLURM_OUTPUT"  # This is set by Slurm
CHILD_LOG="${MAIN_LOG%.out}_child${i}.out"
```

Actually, `$SLURM_OUTPUT` isn't always available. Better: have the executor inject the log pattern into the task environment. Or simplest: use a fixed convention:

```bash
LOG_DIR="${HOME}/.experiments/logs"
CHILD_LOG="${LOG_DIR}/batch_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_child_${i}.out"
```

Then in `jobcat`:
```python
if child_id is not None:
    pattern = f"batch_{job_id}_{array_id}_child_{child_id}.out"
    # search in log_dir
```

### Step 8: Handling `exists` Optimization Within Batches

When a batch runs but some children already exist, we should skip those children to avoid redundant work. In `construct()`:

```python
for i, (child_artifact, commands) in enumerate(zip(self.artifacts, child_scripts)):
    if child_artifact.should_skip():
        # Emit a skip message
        task.run_command(f'echo "Skipping child {i} ({child_artifact.relpath}): already exists"')
        continue
    # ... launch as usual
```

**Note**: This check happens at script-generation time (in the launcher Python process), not at Slurm execution time. This is fine for most use cases. If we wanted runtime skip checks, we'd need to emit shell-level existence checks, which is more complex.

---

## 3. Complete File Changes

### New Files

| File | Purpose |
|------|---------|
| `experiments/batch.py` | `ArtifactBatch` class, `combine_requirements()` function |

### Modified Files

| File | Changes |
|------|---------|
| `experiments/__init__.py` | Export `ArtifactBatch`, `combine_requirements` |
| `experiments/executor.py` | (1) Add `get_direct_dependencies()` to `_build_dependency_graph` flow. (2) Modify `_find_artifact_dependencies` or the dependency graph builder to handle `ArtifactBatch` correctly. |
| `experiments/artifact.py` | Add `get_direct_dependencies()` method to `Artifact` base class |
| `experiments/cli.py` | Add `--autobatch N` flag to launch/drylaunch/relaunch parsers and wire it through |
| `experiments/scripts/jobcat.py` | Support `jobid_arrayidx_childidx` three-part spec |

### No Changes Needed

| File | Why |
|------|-----|
| `finetuning_valley_stages/finetune.py` | FinetunedModel already has `get_requirements()`. No changes needed. |
| `finetuning_valley_stages/launcher.py` | No changes needed — `--autobatch` is handled by the CLI framework. |
| Any stage files | Stage definitions stay the same. `--autobatch` transforms them at launch time. |

---

## 4. Motivating Example Walkthrough

### Before (current behavior)

```bash
python launcher.py launch oci_finetune
# Submits N individual Slurm jobs, each requesting 1 GPU, 8 CPUs, 128G mem
# If N=24, that's 24 separate jobs in the Slurm queue
```

### After (with autobatch)

```bash
python launcher.py launch oci_finetune --autobatch 8
# Groups artifacts into batches of 8
# Each batch requests 8 GPUs, 64 CPUs, 1024G mem (or capped)
# Submits 3 jobs (24/8 = 3), each running 8 finetuning jobs in parallel
# CUDA_VISIBLE_DEVICES is partitioned: child 0 gets GPU 0, child 1 gets GPU 1, etc.
```

### Manual batch creation (alternative API)

```python
from experiments import ArtifactBatch, ArtifactSet

# Group first 8 models into a batch
batch1 = ArtifactBatch(finetuned_models[:8])
batch2 = ArtifactBatch(finetuned_models[8:16])
batch3 = ArtifactBatch(finetuned_models[16:24])

executor.stage('oci_finetune', ArtifactSet([batch1, batch2, batch3]))
```

### Viewing logs

```bash
# Job 12345 is the batch for array index 0 (batch1)
jobcat 12345_0       # Full interleaved log
jobcat 12345_0_3     # Just child 3's log within this batch
```

---

## 5. Edge Cases and Considerations

### 5a. Heterogeneous Requirements Within a Batch

If artifacts in a batch have different GPU requirements (e.g., one needs 2 GPUs, another needs 1), the GPU partitioning logic handles this:

```python
# Child 0: 2 GPUs → CUDA_VISIBLE_DEVICES=0,1
# Child 1: 1 GPU  → CUDA_VISIBLE_DEVICES=2
# Child 2: 1 GPU  → CUDA_VISIBLE_DEVICES=3
# Total allocated: 4 GPUs
```

### 5b. CPU-Only Artifacts

If an artifact needs 0 GPUs, it gets `CUDA_VISIBLE_DEVICES=""` and runs normally. The batch still sums CPUs and memory correctly.

### 5c. Batches With Dependencies

If child artifacts have different dependencies, the batch depends on the union of all children's dependencies. The topological sort places the batch after ALL of those dependencies complete.

### 5d. Memory Limits

Summing memory (e.g., 8 × 128G = 1024G) might exceed node limits. Users should be aware of this. We could add a `--max-mem` option or warning, but for now, trust the user to set appropriate batch sizes.

### 5e. Partial Failures

The current design fails the whole batch if any child fails. This is the simplest correct behavior — Slurm sees the job as failed, and rerunning the batch reruns all children (except those that already exist, if we implement Step 8).

### 5f. Time Limits

The batch uses the same time limit as individual artifacts since all children run in parallel. This is correct — the batch should take roughly the same wall time as a single artifact.

---

## 6. Implementation Order

1. **`experiments/batch.py`** — ArtifactBatch class with `combine_requirements`, `construct`, `get_requirements`, `exists`, `should_skip`, `get_direct_dependencies`
2. **`experiments/artifact.py`** — Add `get_direct_dependencies()` to base `Artifact`
3. **`experiments/executor.py`** — Update `_build_dependency_graph` to use `get_direct_dependencies()`
4. **`experiments/__init__.py`** — Add exports
5. **`experiments/cli.py`** — Add `--autobatch` flag and `_apply_autobatch` method
6. **`experiments/scripts/jobcat.py`** — Support three-part job spec
7. **Test** with `drylaunch` on the finetuning-valley launcher

---

## 7. Testing Strategy

Since we can't run code locally, testing is done by:

1. **`drylaunch` with `--autobatch`**: Verify the generated scripts look correct, batches have combined requirements, dependency ordering is preserved.
2. **Manual inspection**: Read generated sbatch scripts to verify GPU partitioning, child log paths, background process management.
3. **Small-scale test**: `python launcher.py launch oci_finetune --autobatch 2 --head 4` to launch 2 batches of 2, verify logs work.
