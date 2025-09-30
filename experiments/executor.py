from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set

import argparse
import base64
import hashlib
import os
from pathlib import Path
from shlex import quote as shquote
import subprocess
import tempfile

import yaml

from .artifact import Artifact, ArtifactSet

class Directive(ABC):
    """Wrapper for executor-specific values that should pass-through behavior."""

    def __init__(self, value: Any) -> None:
        self._value = value

    def __str__(self) -> str:
        return str(self._value)

    def __call__(self) -> Any:
        return self._value

    def __getattr__(self, name: str) -> Any:
        return self._value.__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_value":
            object.__setattr__(self, "_value", value)
        else:
            self._value.__setattr__(name, value)

class IgnoreHash(Directive):
    pass

class Task:
    """A compiled unit of work consisting of shell blocks for an artifact."""

    def __init__(
        self,
        artifact_path: str,
        code_path: str,
        artifact: Artifact | None = None,
        gs_path: str | None = None,
    ) -> None:
        self.blocks: List[TaskBlock] = []
        self.artifact_path = artifact_path
        self.gs_path = gs_path
        self.code_path = code_path
        self.artifact = artifact
    
    def create_file(self, path: str, content: str | bytes) -> None:
        """Add a file creation block to this task."""
        self.blocks.append(CreateFileTaskBlock(path, content))
    
    def create_yaml_file(self, path: str, content: Dict[str, Any]) -> None:
        """Add a YAML file creation block to this task.
        
        The content dictionary will be serialized to YAML format.
        """
        yaml_content = yaml.dump(content, default_flow_style=False, sort_keys=False)
        self.blocks.append(CreateFileTaskBlock(path, yaml_content))
    
    def run_command(
        self,
        command: str,
        vargs: Sequence[str] | None = None,
        kwargs: Dict[str, Any] | None = None,
        vformat: str | None = None,
        kwformat: str | None = None,
    ) -> None:
        """Add a command execution block to this task."""
        self.blocks.append(CommandTaskBlock(command, vargs, kwargs, vformat, kwformat))

    def upload_to_gs(self, path: str, gs_path: str) -> None:
        """Add a Google Cloud Storage upload block to this task."""
        self.blocks.append(UploadToGSTaskBlock(path, gs_path))

    def download_from_gs(self, gs_path: str, path: str) -> None:
        """Add a Google Cloud Storage download block to this task."""
        self.blocks.append(DownloadFromGSTaskBlock(gs_path, path))

class TaskBlock(ABC):
    """A block that yields a shell command line when executed."""

    @abstractmethod
    def execute(self) -> str | None:
        raise NotImplementedError

class CommandTaskBlock(TaskBlock):
    """Executes a shell command with optional positional and keyword arguments."""

    def __init__(
        self,
        command: str,
        vargs: Sequence[str] | None = None,
        kwargs: Dict[str, Any] | None = None,
        vformat: str | None = None,
        kwformat: str | None = None,
    ) -> None:
        self.command = command
        self.vargs = vargs or []
        self.kwargs = kwargs or {}
        self.vformat = vformat or '{v}'
        self.kwformat = kwformat or '--{k} \'{v}\''

    def execute(self) -> str:
        """Build the complete command string with formatted arguments."""
        parts = [self.command]
        
        # Add positional arguments
        if self.vargs:
            vargs_str = ' '.join(self.vformat.format(v=arg) for arg in self.vargs)
            parts.append(vargs_str)
        
        # Add keyword arguments
        if self.kwargs:
            kwargs_str = ' '.join(
                self.kwformat.format(k=k, v=v) for k, v in self.kwargs.items()
            )
            parts.append(kwargs_str)
        
        return ' '.join(parts)

class CreateFileTaskBlock(TaskBlock):
    """Creates a file with the specified content using base64 encoding."""

    def __init__(
        self,
        path: str | Path,
        content: str | bytes,
        mkdirs: bool = True,
        mode: int | None = None,
    ) -> None:
        self.path = str(path)
        self.content = content
        self.mkdirs = mkdirs
        self.mode = mode

    def execute(self) -> str:
        """Generate shell commands to create the file with proper content."""
        # Convert content to bytes
        if isinstance(self.content, str):
            payload = self.content.encode('utf-8')
        else:
            payload = self.content

        # Encode content as base64 for safe shell transmission
        b64 = base64.b64encode(payload).decode("ascii")
        delim = "___B64___"
        
        parts = []
        
        # Create parent directory if needed
        if self.mkdirs:
            parent = os.path.dirname(self.path) or "."
            parts.append(f"mkdir -p -- {shquote(parent)}")

        # Write file using base64 decoding
        parts.append(f"base64 -d > {shquote(self.path)} << '{delim}'\n{b64}\n{delim}")

        # Set file permissions if specified
        if self.mode is not None:
            parts.append(f"chmod {self.mode} {shquote(self.path)}")
        
        return " && ".join(parts)

class UploadToGSTaskBlock(TaskBlock):
    """Uploads a file or directory to Google Cloud Storage with locking."""

    def __init__(
        self,
        path: str,
        gs_path: str,
        directory: bool = False,
    ) -> None:
        self.path = path
        self.gs_path = gs_path  # Should be gs://bucket/path format
        self.directory = directory
    
    def execute(self) -> str:
        """Generate a locked gcloud storage upload command."""
        # Create a lockfile based on the local path to prevent concurrent access
        path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:10]
        lockfile = f"/tmp/{path_hash}.lock"

        # Build the gcloud command
        recursive_flag = "-r " if self.directory else ""
        gcloud_cmd = f"gcloud storage cp {recursive_flag}{shquote(self.path)} {shquote(self.gs_path)}"

        # Wrap in an exclusive file lock to prevent race conditions
        return f"flock -x {shquote(lockfile)} -c {shquote(gcloud_cmd)}"


class DownloadFromGSTaskBlock(TaskBlock):
    """Downloads a file or directory from Google Cloud Storage with locking."""

    def __init__(
        self,
        gs_path: str,
        path: str,
        directory: bool = False,
    ) -> None:
        self.gs_path = gs_path  # Should be gs://bucket/path format
        self.path = path  # Local destination path
        self.directory = directory

    def execute(self) -> str:
        """Generate a locked gcloud storage download command."""
        # Create a lockfile based on the local path to prevent concurrent access
        path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:10]
        lockfile = f"/tmp/{path_hash}.lock"

        parts = []
        
        # Create necessary directories before download
        if self.directory:
            parts.append(f"mkdir -p -- {shquote(self.path)}")
            recursive_flag = "-r "
        else:
            parent = os.path.dirname(self.path) or "."
            parts.append(f"mkdir -p -- {shquote(parent)}")
            recursive_flag = ""

        # Build and lock the gcloud command
        gcloud_cmd = f"gcloud storage cp {recursive_flag}{shquote(self.gs_path)} {shquote(self.path)}"
        parts.append(f"flock -x {shquote(lockfile)} -c {shquote(gcloud_cmd)}")
        
        return " && ".join(parts)


def _find_artifact_dependencies(value: Any) -> Iterable[Artifact]:
    """Recursively find all Artifact instances within a data structure.
    
    This traverses dictionaries, lists, tuples, and sets to discover artifact
    dependencies declared in artifact attributes.
    """
    if isinstance(value, Artifact):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _find_artifact_dependencies(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _find_artifact_dependencies(v)


class Executor:
    """Base executor providing staging, ordering, and launching of tasks."""

    def __init__(self) -> None:
        self._stages: Dict[str, List[Artifact]] = {}

    def stage(self, name: str, artifacts: Iterable[Artifact] | ArtifactSet) -> None:
        """Register a named stage containing artifacts to execute."""
        self._stages[name] = list(artifacts)

    def auto_cli(self) -> None:
        """Parse command-line arguments and execute selected stages."""
        parser = argparse.ArgumentParser(description="Experiment executor")
        parser.add_argument(
            "stages",
            nargs="*",
            help="Optional stage names to run; omit to run all registered stages",
        )
        args = parser.parse_args()
        selected = list(args.stages) if args.stages else list(self._stages.keys())
        self.execute(selected)

    def execute(self, stages: List[str]) -> None:
        """Execute the specified stages (or all stages if empty list)."""
        # Validate inputs
        if not self._stages:
            print("No stages registered.")
            return

        stages = self._validate_and_normalize_stages(stages)
        unique_artifacts = self._collect_unique_artifacts()
        
        if not unique_artifacts:
            print("No artifacts registered.")
            return

        # Compute execution order via topological sort
        all_tiers = self.compute_topological_ordering(unique_artifacts)
        
        # Filter to only artifacts in selected stages
        filtered_tiers = self._filter_tiers_by_stages(all_tiers, stages)
        
        if not filtered_tiers:
            print("No artifacts matched the selected stages.")
            return

        # Compile and launch
        task_tiers = [[self.compile_artifact(a) for a in tier] for tier in filtered_tiers]
        self.launch(task_tiers)

    def _validate_and_normalize_stages(self, stages: List[str]) -> List[str]:
        """Validate stage names and return normalized list (all stages if empty)."""
        if stages:
            unknown = [s for s in stages if s not in self._stages]
            if unknown:
                raise ValueError(f"Unknown stage(s): {unknown}")
            return stages
        else:
            return list(self._stages.keys())

    def _collect_unique_artifacts(self) -> List[Artifact]:
        """Collect all artifacts from all stages, deduplicated by identity."""
        seen_ids: Set[int] = set()
        unique_artifacts: List[Artifact] = []
        
        for artifacts in self._stages.values():
            for artifact in artifacts:
                artifact_id = id(artifact)
                if artifact_id not in seen_ids:
                    seen_ids.add(artifact_id)
                    unique_artifacts.append(artifact)
        
        return unique_artifacts

    def _filter_tiers_by_stages(
        self,
        tiers: List[List[Artifact]],
        selected_stages: List[str],
    ) -> List[List[Artifact]]:
        """Filter artifact tiers to only include artifacts in selected stages."""
        # Build membership map: artifact ID -> set of stage names
        artifact_to_stages: Dict[int, Set[str]] = {}
        for stage_name, artifacts in self._stages.items():
            for artifact in artifacts:
                artifact_to_stages.setdefault(id(artifact), set()).add(stage_name)

        # Filter each tier
        selected_set = set(selected_stages)
        filtered_tiers: List[List[Artifact]] = []
        
        for tier in tiers:
            # Keep artifacts that belong to at least one selected stage
            filtered_tier = [
                a for a in tier
                if artifact_to_stages.get(id(a), set()) & selected_set
            ]
            if filtered_tier:
                filtered_tiers.append(filtered_tier)
        
        return filtered_tiers

    def compute_topological_ordering(
        self,
        artifacts: Sequence[Artifact],
    ) -> List[List[Artifact]]:
        """Compute execution tiers using topological sort.
        
        Returns a list of tiers where each tier contains artifacts that can be
        executed in parallel. Artifacts in tier N+1 may depend on artifacts in
        tier N, but not vice versa.
        
        Uses Kahn's algorithm with layering to detect cycles and preserve order.
        """
        artifact_by_id: Dict[int, Artifact] = {id(a): a for a in artifacts}
        
        # Build dependency graph
        dependents: Dict[int, Set[int]] = {id(a): set() for a in artifacts}
        num_dependencies: Dict[int, int] = {id(a): 0 for a in artifacts}
        
        self._build_dependency_graph(
            artifacts,
            artifact_by_id,
            dependents,
            num_dependencies,
        )
        
        # Perform layered topological sort (Kahn's algorithm)
        return self._kahn_layered_sort(
            artifacts,
            artifact_by_id,
            dependents,
            num_dependencies,
        )

    def _build_dependency_graph(
        self,
        artifacts: Sequence[Artifact],
        artifact_by_id: Dict[int, Artifact],
        dependents: Dict[int, Set[int]],
        num_dependencies: Dict[int, int],
    ) -> None:
        """Scan artifact attributes to build the dependency graph."""
        for artifact in artifacts:
            artifact_id = id(artifact)
            
            # Scan all attributes for artifact references
            for attr_value in vars(artifact).values():
                for dependency in _find_artifact_dependencies(attr_value):
                    dependency_id = id(dependency)
                    
                    # Validate dependency is in the artifact set
                    if dependency_id not in artifact_by_id:
                        raise ValueError(
                            f"Artifact {artifact} depends on {dependency}, "
                            "which is not in the artifact set. "
                            "All dependencies must be explicitly included."
                        )
                    
                    # Add edge: dependency -> artifact (artifact depends on dependency)
                    if artifact_id not in dependents[dependency_id]:
                        dependents[dependency_id].add(artifact_id)
                        num_dependencies[artifact_id] += 1

    def _kahn_layered_sort(
        self,
        artifacts: Sequence[Artifact],
        artifact_by_id: Dict[int, Artifact],
        dependents: Dict[int, Set[int]],
        num_dependencies: Dict[int, int],
    ) -> List[List[Artifact]]:
        """Perform Kahn's algorithm with layering to produce execution tiers."""
        tiers: List[List[Artifact]] = []
        remaining = set(artifact_by_id.keys())
        processed = 0

        while remaining:
            # Find all artifacts with no remaining dependencies
            # Preserve original input order within each tier
            ready_ids = [
                id(artifact)
                for artifact in artifacts
                if id(artifact) in remaining and num_dependencies[id(artifact)] == 0
            ]
            
            if not ready_ids:
                raise ValueError("Cycle detected in artifact dependencies")

            # Add this tier
            tiers.append([artifact_by_id[aid] for aid in ready_ids])

            # Process this tier: remove from graph and update dependents
            for artifact_id in ready_ids:
                remaining.remove(artifact_id)
                processed += 1
                
                # Decrease dependency count for all dependents
                for dependent_id in dependents[artifact_id]:
                    num_dependencies[dependent_id] -= 1

        # Sanity check: all artifacts should have been processed
        if processed != len(artifact_by_id):
            raise ValueError("Graph processing error: not all artifacts were processed")

        return tiers

    def launch(self, tiers: List[List[Task]]) -> None:
        raise NotImplementedError

    def compile_artifact(self, artifact: Artifact) -> Task:
        raise NotImplementedError


class PrintExecutor(Executor):
    """Executor that prints planned shell commands for inspection."""

    def __init__(
        self,
        artifact_path: str,
        code_path: str,
        gs_path: str | None = None,
    ) -> None:
        super().__init__()
        self.artifact_path = Path(artifact_path)
        self.code_path = Path(code_path)
        self.gs_path = gs_path

    def compile_artifact(self, artifact: Artifact) -> Task:
        """Compile an artifact into a task by calling its construct method."""
        task = Task(
            artifact_path=str(self.artifact_path),
            code_path=str(self.code_path),
            artifact=artifact,
            gs_path=self.gs_path,
        )
        artifact.construct(task)
        return task

    def launch(self, tiers: List[List[Task]]) -> None:
        """Print all shell commands that would be executed."""
        for tier in tiers:
            for task in tier:
                for block in task.blocks:
                    command = block.execute()
                    if command:
                        print(command)


class SlurmExecutor(Executor):
    """Executor that submits tasks as Slurm array jobs.

    Each tier is submitted as a separate Slurm array job, with the array index
    mapping to the task position within that tier. Tasks with different resource
    requirements are submitted as separate parallel jobs.
    """

    def __init__(
        self,
        artifact_path: str,
        code_path: str,
        gs_path: str | None = None,
        default_slurm_args: Dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.artifact_path = Path(artifact_path)
        self.code_path = Path(code_path)
        self.gs_path = gs_path
        self.default_slurm_args = default_slurm_args or {}
        self.dry_run = dry_run
        self._next_fake_job_id = 1000  # For dry run mode
        self._dry_run_jobs: List[Dict[str, Any]] = []  # Store job info for dry run summary

    def auto_cli(self) -> None:
        """Parse command-line arguments including --dry flag and execute selected stages."""
        parser = argparse.ArgumentParser(description="Experiment executor (Slurm)")
        parser.add_argument(
            "stages",
            nargs="*",
            help="Optional stage names to run; omit to run all registered stages",
        )
        parser.add_argument(
            "--dry",
            action="store_true",
            help="Dry run mode: show what would be submitted without actually submitting jobs",
        )
        args = parser.parse_args()
        
        # Set dry run mode based on CLI flag
        self.dry_run = args.dry
        
        selected = list(args.stages) if args.stages else list(self._stages.keys())
        self.execute(selected)

    def compile_artifact(self, artifact: Artifact) -> Task:
        """Compile an artifact into a task by calling its construct method."""
        task = Task(
            artifact_path=str(self.artifact_path),
            code_path=str(self.code_path),
            artifact=artifact,
            gs_path=self.gs_path,
        )
        artifact.construct(task)
        return task

    def launch(self, tiers: List[List[Task]]) -> None:
        """Submit each tier as one or more Slurm array jobs with proper dependencies."""
        previous_tier_job_ids: List[str] = []
        
        if self.dry_run:
            print("\n" + "=" * 100)
            print("DRY RUN MODE - No jobs will be submitted")
            print("=" * 100 + "\n")
        
        for tier_index, tier in enumerate(tiers):
            if not tier:
                continue
            
            # Submit this tier (may create multiple jobs if requirements differ)
            # and get all job IDs created
            current_tier_job_ids = self._submit_tier(
                tier_index,
                tier,
                dependency_job_ids=previous_tier_job_ids,
            )
            
            # These jobs become dependencies for the next tier
            previous_tier_job_ids = current_tier_job_ids
        
        # Print summary for dry run
        if self.dry_run:
            self._print_dry_run_summary()

    def _submit_tier(
        self,
        tier_index: int,
        tier: List[Task],
        dependency_job_ids: List[str],
    ) -> List[str]:
        """Submit a tier as one or more Slurm array jobs.
        
        If tasks have different requirements, they are grouped and submitted
        as separate jobs. All jobs in this tier depend on all jobs from the
        previous tier.
        
        Returns:
            List of job IDs created for this tier
        """
        # Group tasks by their requirements
        task_groups = self._group_tasks_by_requirements(tier)
        
        submitted_job_ids: List[str] = []
        
        # Submit a separate job for each requirement group
        for group_index, (requirements, tasks) in enumerate(task_groups.items()):
            job_name = f"tier-{tier_index}"
            if len(task_groups) > 1:
                job_name += f"-grp{group_index}"
            
            # Get Slurm configuration for this group
            slurm_config = self._build_slurm_config(requirements)
            
            # Generate the sbatch script
            sbatch_header = self._build_sbatch_header(
                job_name,
                tasks,
                slurm_config,
                dependency_job_ids,
            )
            script_body = self._build_script_body(tasks)
            
            # Write script to temporary file
            script_path = self._write_script(tier_index, group_index, sbatch_header, script_body)
            
            # Submit the job (or fake it in dry run mode)
            job_id = self._submit_sbatch_script(script_path)
            submitted_job_ids.append(job_id)
            
            # Store job info for dry run summary
            if self.dry_run:
                # Collect artifact class names
                artifact_classes = [
                    task.artifact.__class__.__name__ 
                    for task in tasks 
                    if task.artifact is not None
                ]
                
                self._dry_run_jobs.append({
                    'job_id': job_id,
                    'job_name': job_name,
                    'tier': tier_index,
                    'group': group_index,
                    'num_tasks': len(tasks),
                    'artifact_classes': artifact_classes,
                    'config': slurm_config,
                    'dependencies': list(dependency_job_ids),
                    'script_path': script_path,
                })
            
            # Print for inspection (more detailed in non-dry-run mode)
            if not self.dry_run:
                print('=' * 100)
                print(f"Submitted job {job_id}: {job_name}")
                print('\n'.join(sbatch_header))
                print('\n\n')
                print('\n'.join(script_body))
                print(f"Script: {script_path}")
                print()
        
        return submitted_job_ids

    def _group_tasks_by_requirements(
        self,
        tier: List[Task],
    ) -> Dict[str, List[Task]]:
        """Group tasks by their resource requirements.
        
        Returns:
            Dictionary mapping requirement signature (as frozen string) to list of tasks
        """
        groups: Dict[str, List[Task]] = defaultdict(list)
        
        for task in tier:
            # Extract requirements from artifact
            if task.artifact is not None and hasattr(task.artifact, 'get_requirements'):
                reqs = dict(task.artifact.get_requirements())  # type: ignore[attr-defined]
            else:
                reqs = {}
            
            # Create a hashable signature from requirements
            # Sort keys for consistency
            req_signature = str(sorted(reqs.items()))
            groups[req_signature].append(task)
        
        return groups

    def _build_slurm_config(self, requirements_signature: str) -> Dict[str, Any]:
        """Build Slurm configuration from a requirements signature.
        
        Args:
            requirements_signature: String representation of sorted requirements dict
        
        Returns:
            Complete Slurm configuration with defaults applied
        """
        # Parse requirements from signature
        # The signature is str(sorted(reqs.items()))
        try:
            reqs = dict(eval(requirements_signature)) if requirements_signature != "[]" else {}
        except Exception:
            reqs = {}
        
        # Merge with defaults
        config: Dict[str, Any] = {}
        config.update(self.default_slurm_args)
        config.update(reqs)
        
        # Apply reasonable defaults
        config.setdefault('partition', 'general')
        config.setdefault('time', '1-00:00:00')
        config.setdefault('cpus', 1)
        
        return config

    def _build_sbatch_header(
        self,
        job_name: str,
        tasks: List[Task],
        config: Dict[str, Any],
        dependency_job_ids: List[str],
    ) -> List[str]:
        """Build the #SBATCH header lines for the script.
        
        Args:
            job_name: Name for this job
            tasks: List of tasks in this job
            config: Slurm configuration dictionary
            dependency_job_ids: Job IDs that must complete before this job starts
        """
        lines = ["#!/usr/bin/env bash"]
        
        # Job name
        lines.append(f"#SBATCH --job-name={job_name}")
        
        # Dependencies: wait for all previous tier jobs to complete successfully
        if dependency_job_ids:
            dependency_str = ":".join(dependency_job_ids)
            lines.append(f"#SBATCH --dependency=afterok:{dependency_str}")
        
        # Resource specifications
        if 'partition' in config:
            lines.append(f"#SBATCH --partition={config['partition']}")
        if 'time' in config:
            lines.append(f"#SBATCH --time={config['time']}")
        if 'cpus' in config:
            lines.append(f"#SBATCH --cpus-per-task={config['cpus']}")
        
        # GPU handling: distinguish between model-specific (GRES) and count-only
        gpus_val = config.get('gpus')
        if gpus_val is not None:
            gpus_str = str(gpus_val)
            if ':' in gpus_str:
                # Format like "a100:2" - use GRES syntax
                lines.append(f"#SBATCH --gres=gpu:{gpus_str}")
            else:
                # Just a number - use simple GPU count
                lines.append(f"#SBATCH --gpus={gpus_str}")
        
        # Array specification
        max_index = len(tasks) - 1
        lines.append(f"#SBATCH --array=0-{max_index}")
        
        return lines

    def _build_script_body(self, tier: List[Task]) -> List[str]:
        """Build the shell script body with case statement for array tasks."""
        lines = [
            "set -euo pipefail",
            "case \"${SLURM_ARRAY_TASK_ID:-0}\" in",
        ]
        
        # Add a case for each task
        for idx, task in enumerate(tier):
            lines.append(f"  {idx})")
            for block in task.blocks:
                command = block.execute()
                if command:
                    lines.append(f"    {command}")
            lines.append("    ;;")
        
        # Add error handler for invalid IDs
        lines.extend([
            "  *)",
            "    echo \"Invalid SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}\" >&2",
            "    exit 1",
            "    ;;",
            "esac",
        ])
        
        return lines

    def _write_script(
        self,
        tier_index: int,
        group_index: int,
        header: List[str],
        body: List[str],
    ) -> str:
        """Write the complete script to a temporary file and return the path."""
        with tempfile.NamedTemporaryFile(
            mode='w',
            prefix=f"slurm_tier_{tier_index}_grp{group_index}_",
            suffix='.sh',
            delete=False,
        ) as f:
            f.write("\n".join(header))
            f.write("\n\n")
            f.write("\n".join(body))
            return f.name

    def _submit_sbatch_script(self, script_path: str) -> str:
        """Submit a script via sbatch and return the job ID.
        
        In dry run mode, returns a fake job ID without actually submitting.
        
        Args:
            script_path: Path to the sbatch script file
            
        Returns:
            Job ID as a string
            
        Raises:
            RuntimeError: If submission fails or job ID cannot be parsed
        """
        # In dry run mode, return a fake job ID
        if self.dry_run:
            fake_id = str(self._next_fake_job_id)
            self._next_fake_job_id += 1
            return fake_id
        
        # Actually submit the job
        try:
            # Use --parsable to get just the job ID
            result = subprocess.run(
                ["sbatch", "--parsable", script_path],
                capture_output=True,
                text=True,
                check=True,
            )
            
            # Parse output: "jobid" or "jobid;cluster"
            output = result.stdout.strip()
            if not output:
                raise RuntimeError("sbatch returned empty output")
            
            # Extract job ID (first part before semicolon if present)
            job_id = output.split(';')[0]
            
            if not job_id:
                raise RuntimeError(f"Could not parse job ID from sbatch output: {output}")
            
            return job_id
            
        except subprocess.CalledProcessError as e:
            error_msg = f"sbatch failed: {e.stderr}"
            raise RuntimeError(error_msg) from e
        except Exception as e:
            raise RuntimeError(f"Error submitting sbatch script: {e}") from e

    def _print_dry_run_summary(self) -> None:
        """Print a nice summary of jobs that would be submitted in dry run mode."""
        if not self._dry_run_jobs:
            print("\nNo jobs to submit.\n")
            return
        
        print("\n" + "=" * 100)
        print("DRY RUN SUMMARY")
        print("=" * 100 + "\n")
        
        # Overall statistics
        total_jobs = len(self._dry_run_jobs)
        total_tasks = sum(job['num_tasks'] for job in self._dry_run_jobs)
        
        print(f"Total jobs: {total_jobs}")
        print(f"Total tasks: {total_tasks}")
        print()
        
        # Count artifacts by class
        artifact_counts: Dict[str, int] = defaultdict(int)
        for job in self._dry_run_jobs:
            for artifact_class in job['artifact_classes']:
                artifact_counts[artifact_class] += 1
        
        if artifact_counts:
            print("Artifact types:")
            for artifact_class, count in sorted(artifact_counts.items()):
                print(f"  {artifact_class}: {count} task(s)")
            print()
        
        # List jobs by tier
        jobs_by_tier: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for job in self._dry_run_jobs:
            jobs_by_tier[job['tier']].append(job)
        
        print("Job submission plan:")
        print()
        
        for tier_index in sorted(jobs_by_tier.keys()):
            tier_jobs = jobs_by_tier[tier_index]
            print(f"Tier {tier_index}:")
            
            for job in tier_jobs:
                print(f"  Job {job['job_id']}: {job['job_name']}")
                print(f"    Tasks: {job['num_tasks']}")
                
                # Show unique artifact types in this job
                unique_artifacts = sorted(set(job['artifact_classes']))
                if unique_artifacts:
                    print(f"    Artifacts: {', '.join(unique_artifacts)}")
                
                # Show key resource requirements
                config = job['config']
                resources = []
                if config.get('partition'):
                    resources.append(f"partition={config['partition']}")
                if config.get('cpus'):
                    resources.append(f"cpus={config['cpus']}")
                if config.get('gpus'):
                    resources.append(f"gpus={config['gpus']}")
                if config.get('time'):
                    resources.append(f"time={config['time']}")
                
                if resources:
                    print(f"    Resources: {', '.join(resources)}")
                
                # Show dependencies
                if job['dependencies']:
                    deps_str = ', '.join(job['dependencies'])
                    print(f"    Dependencies: {deps_str}")
                else:
                    print(f"    Dependencies: None (first tier)")
                
                print(f"    Script: {job['script_path']}")
                print()
        
        print("=" * 100)
        print("To actually submit these jobs, run without the --dry flag")
        print("=" * 100 + "\n")