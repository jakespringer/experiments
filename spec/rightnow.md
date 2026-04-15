# Specification: `experiments/scripts/rightnow.py`

## Purpose
SLURM Resource Estimator and Job Diagnostic CLI. Estimates the largest job parameters (time, CPUs, memory) that can run immediately for various GPU counts, considering fairshare policy, priorities, backfill scheduling, and partition limits. Also provides a "blame" mode that diagnoses why specific pending jobs aren't running.

## Dependencies
- `argparse`, `re`, `subprocess`, `sys`, `time`
- `dataclasses`, `datetime`, `typing`
- `rich` (Console, Table, Progress, Panel, Text, Live, Layout, box)

---

## Dataclasses

### `NodeResources`
- **Fields**: `name`, `state`, `cpus`, `cpus_alloc`, `memory` (MB), `memory_alloc` (MB), `gpus`, `gpus_alloc`, `partition`, `partitions: List[str]`, `gpu_type: Optional[str]`, `features: Set[str]`.
- **Properties**:
  - `cpus_free`: `max(0, cpus - cpus_alloc)`.
  - `memory_free`: `max(0, memory - memory_alloc)`.
  - `gpus_free`: `max(0, gpus - gpus_alloc)`.
  - `is_usable`: `True` if state contains `idle` or `mix` and does NOT contain `drain`, `down`, `maint`, `reboot`, or `reserved`.

### `RunningJob`
- **Fields**: `job_id`, `user`, `cpus`, `gpus`, `gpu_type: Optional[str]`, `memory` (MB), `nodes: List[str]`, `end_time: Optional[datetime]`, `partition`.

### `PendingJob`
- **Fields**: `job_id`, `user`, `priority: int`, `cpus`, `gpus`, `gpu_type: Optional[str]`, `memory` (MB), `time_limit_minutes`, `partition`, `state`, `reason`, `start_time: Optional[datetime]`, `start_time_available: bool`.
- **Method**: `can_run_on_node(node)` — Checks partition match, GPU type match (case-insensitive), and resource capacity (cpus, memory, gpus).

### `JobBrief`
- **Fields**: `job_id`, `priority`, `state`, `start_time: Optional[datetime]`, `reason`, `user`, `partition`.

### `JobDetail`
- **Fields**: `job_id`, `state`, `reason`, `priority`, `partition`, `user`, `gpus`, `gpu_type: Optional[str]`, `cpus`, `memory`, `time_limit_minutes`, `start_time: Optional[datetime]`, `submit_time: Optional[datetime]`.

### `UserPriority`
- **Fields**: `fairshare`, `account`, `norm_shares`, `raw_usage`, `effective_usage`.

### `PriorityWeights`
- **Fields**: `fairshare`, `age`, `job_size`, `partition`, `qos` — all `int`, defaulting to 1000 (qos=0).

### `PartitionInfo`
- **Fields**: `max_time_minutes`, `priority_factor`, `max_nodes: Optional`, `max_jobs: Optional`, `total_nodes: Optional`.

### `ResourceEvent`
- **Fields**: `time: Optional[datetime]`, `delta_cpus`, `delta_gpus`, `delta_mem`, `kind` (`'release'` or `'alloc'`), `label`.

### `NodeSimState`
- **Fields**: `node: NodeResources`, `free_cpus`, `free_gpus`, `free_mem`, `events: List[ResourceEvent]`, `scheduled: List[Tuple]`, `running: List[RunningJob]`.

### `PartitionEvent`
- **Fields**: `time: Optional[datetime]`, `delta_jobs`, `delta_nodes`, `delta_gpus`, `kind`, `label`.

### `PartitionSimState`
- **Fields**: `partition`, `max_jobs`, `max_nodes`, `max_gpus`, `free_jobs`, `free_nodes`, `free_gpus`, `events: List[PartitionEvent]`.

---

## Exception: `SlurmQueryError`
- Raised when SLURM commands fail or time out.

---

## Class: `SlurmBase`

### Purpose
Base class with SLURM command execution and output parsing utilities. Used by both `SlurmEstimator` and `SlurmBlame`.

### Methods

#### `_run_command(self, cmd: List[str]) -> str`
- Runs subprocess with `capture_output=True, text=True, check=True, timeout=30`.
- Raises `SlurmQueryError` on `CalledProcessError` or `TimeoutExpired`.

#### Static Methods (Parsing)
- `_parse_slurm_time(time_str) -> Optional[datetime]`: Parses ISO format or `%Y-%m-%dT%H:%M:%S`. Returns `None` for N/A values.
- `_parse_time_limit(time_str) -> int`: Converts time limit to minutes. Supports `D-HH:MM:SS`, `HH:MM:SS`, `MM:SS`, `SS`. Returns 10080 for `UNLIMITED`. Defaults to 60 on error.
- `_parse_memory(mem_str) -> int`: Converts memory string (with T/G/M/K suffixes) to MB.
- `_parse_gres_string(gres) -> Tuple[int, Optional[str]]`: Parses `gpu:TYPE:COUNT` or `gpu:COUNT`. Returns `(count, gpu_type)`.
- `_parse_req_gpus(spec) -> Tuple[int, Optional[str]]`: Parses GPU requests from `ReqTRES`/`TresPerNode` strings. Handles both colon and equals separators. Falls back to `_parse_gres_string`.
- `_parse_req_mem(spec) -> int`: Parses `mem=VALUE` from comma-separated TRES string.
- `_parse_alloc_gpus(tres) -> int`: Extracts allocated GPU count from `AllocTRES` string via regex.
- `_expand_node_list(node_list) -> List[str]`: Expands SLURM node range notation (e.g., `node[001-003,005]` → `['node001', 'node002', 'node003', 'node005']`).

---

## Class: `SlurmEstimator(SlurmBase)`

### Purpose
Main estimator class. Queries SLURM for cluster state and estimates maximum job parameters for various GPU counts.

### Constructor: `__init__(self, max_gpus=16, verbose=False, partition=None, user=None, gpu_type=None)`
- `max_gpus`: capped at 8.
- `filter_partition`, `filter_user`, `filter_gpu_type`: optional filters.
- Initializes empty lists for `nodes`, `running_jobs`, `pending_jobs`.
- State: `partition_limits`, `partition_priority`, `partition_info`, `allowed_partitions`.
- `current_user`: from `whoami` command (or `filter_user` if specified).

### Query Methods

#### `query_nodes(self) -> None`
- Runs `scontrol show node -o`.
- Parses each node line into `NodeResources`:
  - CPUs: from `CPUEfctv` or `CPUTot`.
  - Memory: `RealMemory` and `AllocMem` (with fallback allocation estimate from CPU ratio).
  - GPUs: from `Gres` field via `_parse_gres_string`.
  - GPU alloc: from `AllocTRES` via `_parse_alloc_gpus`, capped at total GPUs.
  - Features: from `AvailableFeatures` or `ActiveFeatures`.
  - GPU type inference from features if not in GRES.
- Stores results in `self.nodes`.

#### `_get_node_gpu_alloc(self, node_name, total_gpus) -> int`
- Queries individual node's `AllocTRES` for GPU count.
- Heuristic fallback: full allocation for `alloc` state, half for `mix`.

#### `query_running_jobs(self) -> None`
- Runs `squeue -h -t RUNNING -o '%i|%u|%C|%b|%m|%N|%e|%P'`.
- Parses into `RunningJob` objects with GRES parsing for GPUs, node list expansion, and end time parsing.

#### `query_pending_jobs(self) -> None`
- Two-phase query:
  1. `squeue -h -t PENDING -o '%i|%u|%Q|%C|%b|%m|%l|%P|%T|%r'` — main job data.
  2. `squeue -h -t PENDING --start -o '%i|%S'` — start time estimates.
- Merges start times into job data.
- Creates `PendingJob` objects.

#### `query_user_priority(self) -> None`
- Runs `sprio --noheader -u {user}` and `sacctmgr show user {user} format=Account,DefaultAccount withassoc --noheader`.
- Parses fairshare, norm shares, raw usage, effective usage.
- Creates `UserPriority` object.

#### `query_user_partitions(self) -> None`
- Queries `sacctmgr show associations` for the current user to determine allowed partitions.
- Sets `self.allowed_partitions` if any are found (or `None` if unrestricted).

#### `query_partitions(self) -> None`
- Runs `scontrol show partition -o`.
- Parses `MaxTime`, `PriorityJobFactor`, `MaxNodes`, `MaxJobs`, `TotalNodes` for each partition.
- Populates `self.partition_limits`, `self.partition_priority`, `self.partition_info`.

### Static Method: `_infer_gpu_type_from_features(features) -> Optional[str]`
- Guesses GPU type from node feature strings.
- Checks against known prefixes: `h100`, `a100`, `v100`, `l40s`, `l40`, `a40`, `a30`, `t4`, `rtx6000`, `rtxa6000`, `rtx8000`.
- Falls back to regex pattern `[a-z]{1,4}\d{2,3}`.

### Core Estimation

#### `calculate_max_time_for_gpus(self, gpu_count) -> Dict`
- Determines maximum job parameters for a given GPU count across all accessible partitions.
- **Algorithm per partition**:
  1. Filters nodes by partition, GPU type, and allowed partitions.
  2. Filters to usable nodes with enough GPUs.
  3. Computes total CPUs and memory from eligible nodes.
  4. Calculates `priority_factor` from partition priority and user fairshare.
  5. Identifies "competing" pending jobs (higher priority, same partition/GPU type).
  6. Determines partition-level gates (MaxTime limits, partition resource exhaustion).
  7. **If no competing jobs**: status = `available` with max time 10080 minutes.
  8. **If competing jobs exist**: finds earliest competing start time.
     - If gate time available and `max_time > 0`: status = `backfill`.
     - If `max_time <= 0`: status = `blocked`.
     - If no gate time available: status = `uncertain` (default 120 min).
  9. Applies partition time cap and user preference adjustments.
- Selects best result across partitions based on: non-unavailable preferred, then priority factor, then free GPUs.
- Returns dict with: `status`, `max_time_minutes`, `max_cpus`, `max_memory`, `reason`, `has_asterisk`, `partition`, `gpu_type`, `all_partitions`, `all_gpu_types`.

#### `estimate_all_configurations(self, show_progress=True) -> Dict[int, Dict]`
- Runs `calculate_max_time_for_gpus` for GPU counts `[1, 2, 4, 8, ...]` (powers of 2 up to `max_gpus`).
- Shows Rich progress spinner if `show_progress=True`.

#### `refresh_data(self) -> None`
- Re-queries all SLURM data: nodes, running jobs, pending jobs, user priority, user partitions, partitions.
- Resets timestamp and increments update counter.

#### `generate_display(self, results) -> Panel`
- Generates a Rich `Panel` containing:
  - **Header**: Cluster summary (node count, free GPUs, job counts, filters, update time).
  - **Body**: Table with columns: GPUs, Partitions, GPU Types, Status (colored), Max Time, Max CPUs, Max Memory, Notes.
  - **Footer**: Legend (Available/Backfill/Uncertain/Blocked) and "Press Ctrl+C to exit".
- Status formatting: green checkmark for available, yellow lightning for backfill, orange question mark for uncertain, red X for blocked.
- Time formatting: days (if ≥24h) or hours, with asterisk for incomplete data.

---

## Class: `SlurmBlame(SlurmBase)`

### Purpose
Diagnoses why specific pending jobs aren't running by simulating resource availability and scheduling.

### Constructor: `__init__(self, estimator: SlurmEstimator, blockers: int = 5)`
- Takes a pre-loaded `SlurmEstimator` instance.
- `blockers`: number of higher-priority jobs to analyze per node.

### Methods

#### `snapshot_queue(self) -> List[JobBrief]`
- Runs `squeue -h -t R,PD -o '%i|%Q|%T|%S|%r|%u|%P'`.
- Parses into `JobBrief` objects sorted by priority (descending).

#### `predict_start(self, job_id) -> Optional[datetime]`
- Runs `squeue -h --start -j {job_id} -o '%i|%S'` to get SLURM's predicted start time.

#### `fetch_job_detail(self, job_id) -> JobDetail`
- Runs `scontrol show job -o {job_id}`.
- Parses all fields into `JobDetail` with multi-source GPU/memory resolution:
  - GPUs: `ReqTRES` → `Gres` → `AllocTRES` fallback chain.
  - Memory: `ReqTRES` → `MinMemoryCPU × cpus` → `MinMemoryNode` → `ReqMem` fallback chain.

#### Static Helpers
- `_req_from_detail(detail) -> Tuple[int, int, int]`: Returns `(cpus, gpus, memory)`.
- `_req_from_pending(job) -> Tuple[int, int, int]`: Returns `(cpus, gpus, memory)`.
- `_fits(free, req) -> bool`: Component-wise comparison (all free ≥ req).
- `_apply_delta(free, event, node) -> Tuple[int, int, int]`: Applies resource event, clamped to `[0, node.max]`.
- `_per_node_usage(job, node) -> Tuple[int, int, int]`: Divides multi-node job resources across nodes (ceiling division).

#### Node/Partition Analysis

##### `_node_supports(self, node, detail) -> bool`
- Checks: usable state, partition match, GPU type match (case-insensitive), resource capacity.

##### `_job_can_use_node(self, job, node) -> bool`
- Delegates to `node.is_usable` and `job.can_run_on_node(node)`.

##### `_build_node_states(self, detail) -> Dict[str, NodeSimState]`
- Builds per-node simulation state for nodes that support the target job.
- Populates resource events from running jobs (release events at `end_time`).
- Estimates per-node resource usage for multi-node jobs.

##### `_build_partition_state(self, detail) -> Optional[PartitionSimState]`
- Computes partition-level resource state: max_jobs, max_nodes, max_gpus, free counts.
- Creates release events for each running job in the partition.

##### `_earliest_partition_start(self, state, req_gpus, req_nodes, now) -> Tuple[Optional[datetime], List[PartitionEvent]]`
- Simulates partition-level resource availability over time.
- Processes events chronologically, applying deltas until requirements fit.
- Returns earliest time partition can accommodate job, plus event trace.

##### `_earliest_start(self, state, req, now) -> Tuple[Optional[datetime], List[ResourceEvent]]`
- Same as `_earliest_partition_start` but for per-node resources.

##### `_select_competing_jobs(self, detail, node_states, ahead_job_ids) -> List[PendingJob]`
- Identifies pending jobs that compete for the same resources.
- If `ahead_job_ids` is provided: filters to only those jobs.
- Otherwise: filters by higher priority.
- Must be able to run on at least one candidate node.

##### `_schedule_competing_jobs(self, detail, node_states, partition_state, competing, now) -> Dict`
- Simulates scheduling competing jobs onto nodes.
- For each competing job: finds best (earliest-available) node, allocates resources.
- Updates node simulation state with allocation events.
- Updates partition simulation state if applicable.
- Returns per-node schedule map.

##### `_compute_node_blockers(self, detail, ahead_job_ids=None) -> List[Dict]`
- Main diagnostic method.
- Builds node and partition simulation states.
- Selects and schedules competing jobs.
- For each candidate node:
  - Computes earliest start time.
  - Collects reasons: running jobs with unknown end times, release events, partition constraints, queued competing jobs.
- Returns top 3 nodes sorted by earliest start time, each with `{node, start, reasons}`.

#### `diagnose(self, job_id, queue) -> Dict`
- Entry point for job diagnosis.
- Fetches job detail, determines queue position and ahead-of-me jobs.
- Classifies status: `running`, `waiting-priority`, `waiting-resources`, `blocked-dependency`, `blocked-error`, `blocked-resources`, `unknown`.
- Calls `_analyze_resources` for recommendations and `_compute_node_blockers` for blockers.
- Returns comprehensive diagnostic dict with: `job_id`, `user`, `partition`, `state`, `status`, `reason`, `eta`, `priority`, `blockers`, `resources`, `recommendations`, `capacity_trace`, `node_blockers`.

#### `_analyze_resources(self, detail) -> Tuple[List[str], bool, bool, List[str]]`
- Analyzes whether any node has capacity for the job.
- Checks: GPU type mismatch, GPU count, CPU count, memory.
- If no capacity: generates recommendations (max CPUs, max memory, available GPU types, per-node GPU limits).
- If capacity exists but not free: generates alternative configurations (different partitions, GPU types, reduced resources).
- Returns `(recommendations, has_any_capacity, has_free_now, trace)`.

---

## Helper Functions

### `_format_eta(dt) -> str`
- Formats datetime as relative time: "imminent", "X min", "X.X h", "X.X d".

### `_format_blocker_time(dt, now=None) -> str`
- Formats datetime as `HH:MM{am/pm}` with optional `+N` day offset.

### `_render_blame_table(diag) -> Panel`
- Renders a diagnostic dict as a Rich table with columns: State, Priority, ETA, Reason, Blockers, Resources, Recommendations.
- Status icons: green checkmark (running), yellow hourglass (waiting resources), yellow arrow (waiting priority), red warning (dependency/error/unschedulable).
- Node blockers shown with hierarchical bullet-point formatting, deduplicating repeated reasons.

---

## CLI Functions

### `run_available(args)`
- **Single snapshot mode** (default):
  - Creates `SlurmEstimator`, refreshes data, estimates configurations, prints display.
  - Adjusts terminal size environment variables for proper rendering.
- **Monitor mode** (`--monitor`):
  - Creates estimator, enters Rich `Live` display loop.
  - Refreshes every `args.n` seconds (default 5.0).
  - Stops on `KeyboardInterrupt`.

### `run_blame(args)`
- Creates `SlurmEstimator`, refreshes data.
- Creates `SlurmBlame` inspector with `args.blockers` (default 5).
- Snapshots queue.
- For each `job_id` in `args.job_ids`: diagnoses and renders blame table.

### `build_parser() -> argparse.ArgumentParser`
- Creates top-level parser with two subcommands:
- **`available`**:
  - `--max-gpus` (default 8, capped at 8)
  - `--monitor/-m` (enable auto-refresh)
  - `-n SECONDS` (refresh interval, default 5.0)
  - `--verbose/-v`
  - `--partition/-p`
  - `--user/-u`
  - `--gpu/-g` (GPU type filter)
- **`blame`**:
  - `job_ids` (positional, one or more)
  - `--blockers/-b` (default 5)

### `main()`
- If no subcommand or unknown first arg: defaults to `available`.
- Routes to `run_available` or `run_blame`.

---

## Important Behaviors

### Status Categories
- **`available`**: No competing higher-priority jobs; can run at full partition time limit.
- **`backfill`**: Competing jobs exist but a time window exists before the first one would start.
- **`uncertain`**: Start times unavailable for competing jobs; conservative 120-minute estimate.
- **`blocked`**: No available time window (competing job starts too soon).
- **`unavailable`**: No suitable partition or nodes found.

### Asterisk (`*`) Indicator
- Appended to time estimates when data is incomplete (missing start times, unknown end times).

### Priority Factor Calculation
- `partition_priority × max(0.01, user_fairshare)`.
- Used to rank partitions when multiple are available.

### Partition Caps
- `max_time_minutes` from partition limits applied to all estimates.
- `max_cpus` capped at 128, `max_memory` capped at 512000 MB.

### Best Partition Selection
- Prefers non-unavailable status.
- Then highest priority factor.
- Then most free GPUs.
