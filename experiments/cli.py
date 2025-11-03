"""Command-line interface for experiment management."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .executor import SlurmExecutor
from .config import ConfigManager
from .project import Project


class ExperimentCLI:
    """Command-line interface for experiments."""
    
    def __init__(self, executor: SlurmExecutor):
        self.executor = executor
        self.config_manager = ConfigManager()
        self.config = self.config_manager.ensure_config()
        # Ensure Project context is initialized
        if Project.name is None and getattr(self.executor, 'project', None):
            Project.init(self.executor.project)  # type: ignore[arg-type]
    
    def run(self) -> None:
        """Parse arguments and run the appropriate command."""
        parser = argparse.ArgumentParser(
            description="Experiment executor CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        subparsers = parser.add_subparsers(dest='command', help='Command to run')
        
        # launch command
        launch_parser = subparsers.add_parser('launch', help='Launch experiment stages')
        launch_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to run (omit for all stages)'
        )
        launch_parser.add_argument(
            '--head',
            type=int,
            metavar='N',
            help='Only launch the first N artifacts'
        )
        launch_parser.add_argument(
            '--tail',
            type=int,
            metavar='N',
            help='Only launch the last N artifacts'
        )
        launch_parser.add_argument(
            '--rerun',
            action='store_true',
            help='Ignore exists check and rerun all artifacts'
        )
        launch_parser.add_argument(
            '--reverse',
            action='store_true',
            help='Launch stages in reverse order (respects dependencies)'
        )
        launch_parser.add_argument(
            '--exclude',
            nargs='+',
            metavar='STAGE',
            help='Stage names to exclude from execution'
        )
        launch_parser.add_argument(
            '--artifact',
            nargs='+',
            metavar='ARTIFACT',
            help='Artifact class names to include (filters by type)'
        )
        launch_parser.add_argument(
            '--jobs',
            type=int,
            metavar='N',
            help='Combine tasks into N parallel jobs by running multiple tasks sequentially in each job'
        )
        launch_parser.add_argument(
            '--dependency',
            nargs='+',
            metavar='JOBID',
            help='Job IDs that all launched jobs should depend on'
        )
        launch_parser.add_argument(
            '--slurm',
            nargs='*',
            metavar='KEY=VALUE',
            help='Override Slurm args for all jobs (e.g., time=12:00:00 gpus=4 exclude=node[01-04] nodelist=node05)'
        )
        launch_parser.add_argument(
            '--force-launch',
            action='store_true',
            help='Ignore running check and launch jobs anyway'
        )
        launch_parser.add_argument(
            '--throttle',
            type=int,
            metavar='N',
            help='Limit concurrent array tasks to N (adds %%N to --array specification)'
        )
        launch_parser.add_argument(
            '--splitjobs',
            type=int,
            metavar='N',
            help='Split large array submissions into multiple sbatch calls with at most N indices per submission'
        )
        
        # drylaunch command
        drylaunch_parser = subparsers.add_parser('drylaunch', help='Dry run: show what would be launched')
        drylaunch_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to run (omit for all stages)'
        )
        drylaunch_parser.add_argument(
            '--head',
            type=int,
            metavar='N',
            help='Only launch the first N artifacts'
        )
        drylaunch_parser.add_argument(
            '--tail',
            type=int,
            metavar='N',
            help='Only launch the last N artifacts'
        )
        drylaunch_parser.add_argument(
            '--rerun',
            action='store_true',
            help='Ignore exists check and rerun all artifacts'
        )
        drylaunch_parser.add_argument(
            '--reverse',
            action='store_true',
            help='Launch stages in reverse order (respects dependencies)'
        )
        drylaunch_parser.add_argument(
            '--exclude',
            nargs='+',
            metavar='STAGE',
            help='Stage names to exclude from execution'
        )
        drylaunch_parser.add_argument(
            '--artifact',
            nargs='+',
            metavar='ARTIFACT',
            help='Artifact class names to include (filters by type)'
        )
        drylaunch_parser.add_argument(
            '--jobs',
            type=int,
            metavar='N',
            help='Combine tasks into N parallel jobs by running multiple tasks sequentially in each job'
        )
        drylaunch_parser.add_argument(
            '--dependency',
            nargs='+',
            metavar='JOBID',
            help='Job IDs that all launched jobs should depend on'
        )
        drylaunch_parser.add_argument(
            '--slurm',
            nargs='*',
            metavar='KEY=VALUE',
            help='Override Slurm args for all jobs (no submission) (e.g., time=12:00:00 gpus=4 exclude=node[01-04] nodelist=node05)'
        )
        drylaunch_parser.add_argument(
            '--throttle',
            type=int,
            metavar='N',
            help='Limit concurrent array tasks to N (adds %%N to --array specification)'
        )
        drylaunch_parser.add_argument(
            '--splitjobs',
            type=int,
            metavar='N',
            help='Split large array submissions into multiple sbatch calls with at most N indices per submission'
        )
        
        # cancel command
        cancel_parser = subparsers.add_parser('cancel', help='Cancel jobs for stages')
        cancel_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to cancel (omit for all stages)'
        )
        
        # cat command
        cat_parser = subparsers.add_parser('cat', help='Print log file for a job')
        cat_parser.add_argument('job_spec', type=str, help='Job ID or job_id_arrayindex (e.g., 12345 or 12345_0)')
        cat_parser.add_argument('array_index', type=int, nargs='?', help='Optional: array index if not in job_spec')
        
        # history command
        subparsers.add_parser('history', help='Show launch history')
        
        # print command
        print_parser = subparsers.add_parser('print', help='Print commands to run sequentially (can be piped to bash)')
        print_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to run (omit for all stages)'
        )
        print_parser.add_argument(
            '--head',
            type=int,
            metavar='N',
            help='Only print the first N artifacts'
        )
        print_parser.add_argument(
            '--tail',
            type=int,
            metavar='N',
            help='Only print the last N artifacts'
        )
        print_parser.add_argument(
            '--rerun',
            action='store_true',
            help='Ignore exists check and rerun all artifacts'
        )
        print_parser.add_argument(
            '--reverse',
            action='store_true',
            help='Print stages in reverse order (respects dependencies)'
        )
        print_parser.add_argument(
            '--exclude',
            nargs='+',
            metavar='STAGE',
            help='Stage names to exclude from execution'
        )
        print_parser.add_argument(
            '--artifact',
            nargs='+',
            metavar='ARTIFACT',
            help='Artifact class names to include (filters by type)'
        )
        print_parser.add_argument(
            '--jobs',
            type=int,
            metavar='N',
            help='Combine tasks into N parallel jobs by running multiple tasks sequentially in each job'
        )
        
        # printlines command
        printlines_parser = subparsers.add_parser('printlines', help='Print one line per job to execute individual bash scripts')
        printlines_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to run (omit for all stages)'
        )
        printlines_parser.add_argument(
            '--head',
            type=int,
            metavar='N',
            help='Only print the first N artifacts'
        )
        printlines_parser.add_argument(
            '--tail',
            type=int,
            metavar='N',
            help='Only print the last N artifacts'
        )
        printlines_parser.add_argument(
            '--rerun',
            action='store_true',
            help='Ignore exists check and rerun all artifacts'
        )
        printlines_parser.add_argument(
            '--reverse',
            action='store_true',
            help='Print stages in reverse order (respects dependencies)'
        )
        printlines_parser.add_argument(
            '--exclude',
            nargs='+',
            metavar='STAGE',
            help='Stage names to exclude from execution'
        )
        printlines_parser.add_argument(
            '--artifact',
            nargs='+',
            metavar='ARTIFACT',
            help='Artifact class names to include (filters by type)'
        )
        printlines_parser.add_argument(
            '--jobs',
            type=int,
            metavar='N',
            help='Combine tasks into N parallel jobs by running multiple tasks sequentially in each job'
        )
        printlines_parser.add_argument(
            '--output-dir',
            type=str,
            metavar='DIR',
            help='Directory to write job scripts (default: creates temp dir in /tmp)'
        )
        
        # export command
        export_parser = subparsers.add_parser('export', help='Export project and artifact information to JSON')
        export_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to export (omit for all stages)'
        )
        export_parser.add_argument(
            '--artifact',
            nargs='+',
            metavar='ARTIFACT',
            help='Artifact class names to include (filters by type)'
        )
        export_parser.add_argument(
            '--exists',
            action='store_true',
            help='Only export artifacts that exist'
        )
        export_parser.add_argument(
            '-f', '--file',
            type=str,
            required=True,
            metavar='FILE',
            help='Output JSON file path'
        )
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        # Route to appropriate handler
        if args.command == 'launch':
            self.launch(
                args.stages,
                dry_run=False,
                head=getattr(args, 'head', None),
                tail=getattr(args, 'tail', None),
                rerun=getattr(args, 'rerun', False),
                reverse=getattr(args, 'reverse', False),
                exclude=getattr(args, 'exclude', None),
                artifacts=getattr(args, 'artifact', None),
                jobs=getattr(args, 'jobs', None),
                dependency=getattr(args, 'dependency', None),
                slurm_overrides=getattr(args, 'slurm', None),
                force_launch=getattr(args, 'force_launch', False),
                throttle=getattr(args, 'throttle', None),
                split_jobs=getattr(args, 'splitjobs', None),
            )
        elif args.command == 'drylaunch':
            self.launch(
                args.stages,
                dry_run=True,
                head=getattr(args, 'head', None),
                tail=getattr(args, 'tail', None),
                rerun=getattr(args, 'rerun', False),
                reverse=getattr(args, 'reverse', False),
                exclude=getattr(args, 'exclude', None),
                artifacts=getattr(args, 'artifact', None),
                jobs=getattr(args, 'jobs', None),
                dependency=getattr(args, 'dependency', None),
                slurm_overrides=getattr(args, 'slurm', None),
                force_launch=False,
                throttle=getattr(args, 'throttle', None),
                split_jobs=getattr(args, 'splitjobs', None),
            )
        elif args.command == 'cancel':
            self.cancel(args.stages)
        elif args.command == 'cat':
            self.cat(args.job_spec, args.array_index)
        elif args.command == 'history':
            self.history()
        elif args.command == 'print':
            self.print_commands(args.stages, head=getattr(args, 'head', None), tail=getattr(args, 'tail', None), rerun=getattr(args, 'rerun', False), reverse=getattr(args, 'reverse', False), exclude=getattr(args, 'exclude', None), artifacts=getattr(args, 'artifact', None), jobs=getattr(args, 'jobs', None))
        elif args.command == 'printlines':
            self.print_lines(args.stages, head=getattr(args, 'head', None), tail=getattr(args, 'tail', None), rerun=getattr(args, 'rerun', False), reverse=getattr(args, 'reverse', False), exclude=getattr(args, 'exclude', None), artifacts=getattr(args, 'artifact', None), jobs=getattr(args, 'jobs', None), output_dir=getattr(args, 'output_dir', None))
        elif args.command == 'export':
            self.export(args.stages, artifacts=getattr(args, 'artifact', None), exists_only=getattr(args, 'exists', False), output_file=args.file)
    
    def launch(self, stages: List[str], dry_run: bool = False, head: Optional[int] = None, tail: Optional[int] = None, rerun: bool = False, reverse: bool = False, exclude: Optional[List[str]] = None, artifacts: Optional[List[str]] = None, jobs: Optional[int] = None, dependency: Optional[List[str]] = None, slurm_overrides: Optional[List[str]] = None, force_launch: bool = False, throttle: Optional[int] = None, split_jobs: Optional[int] = None) -> None:
        """Launch experiment stages."""
        # Apply config settings to executor
        self.executor.dry_run = dry_run
        
        # Apply CLI slurm overrides (highest priority)
        overrides: Dict[str, Any] = {}
        if slurm_overrides:
            for item in slurm_overrides:
                if '=' in item:
                    k, v = item.split('=', 1)
                    overrides[k.strip()] = v.strip()
        
        # Handle throttle: CLI --throttle takes precedence over --slurm throttle=N
        throttle_value = None
        if throttle is not None:
            # Explicit --throttle flag has highest priority
            throttle_value = throttle
        elif 'throttle' in overrides:
            # Fall back to --slurm throttle=N
            try:
                throttle_value = int(overrides['throttle'])
            except (ValueError, TypeError):
                pass
            # Remove from overrides since we handle it separately
            overrides.pop('throttle', None)
        
        if hasattr(self.executor, 'cli_slurm_overrides'):
            self.executor.cli_slurm_overrides = overrides
        if hasattr(self.executor, 'force_launch'):
            self.executor.force_launch = bool(force_launch)
        if hasattr(self.executor, 'array_throttle'):
            self.executor.array_throttle = throttle_value
        # Set split_jobs on the executor
        if hasattr(self.executor, 'split_jobs'):
            self.executor.split_jobs = split_jobs
        
        # Set external dependencies on the executor
        if hasattr(self.executor, 'external_dependencies'):
            self.executor.external_dependencies = dependency or []
        
        # Execute stages
        selected = stages if stages else list(self.executor._stages.keys())
        
        # Exclude specified stages
        if exclude:
            selected = [s for s in selected if s not in exclude]
        
        if reverse:
            selected = list(reversed(selected))
        self.executor.execute(selected, head=head, tail=tail, rerun=rerun, artifacts=artifacts, jobs=jobs)
    
    def cancel(self, stages: List[str]) -> None:
        """Cancel jobs for the specified stages."""
        if not Project.name:
            print("Error: No project specified in executor")
            sys.exit(1)
        
        # Determine which stages to cancel
        stages_to_cancel = stages if stages else list(self.executor._stages.keys())
        
        print(f"Cancelling jobs for stages: {', '.join(stages_to_cancel)}")
        print()
        
        # Load jobs for each stage
        all_jobs_to_cancel = []
        for stage in stages_to_cancel:
            jobs = self.config_manager.load_jobs(Project.name, stage)  # type: ignore[arg-type]
            all_jobs_to_cancel.extend(jobs)
        
        if not all_jobs_to_cancel:
            print("No jobs found to cancel.")
            return
        
        # Load already canceled jobs
        canceled_jobs = self.config_manager.load_canceled_jobs(Project.name)  # type: ignore[arg-type]
        
        # Group by job_id (since array jobs share one ID)
        job_ids = set()
        for job in all_jobs_to_cancel:
            if 'job_id' in job:
                job_ids.add(job['job_id'])
        
        # Filter out already canceled jobs
        jobs_to_cancel = job_ids - canceled_jobs
        
        if not jobs_to_cancel:
            print("No new jobs to cancel.")
            return
        
        # Cancel each job
        successfully_canceled = []
        failed_to_cancel = []
        
        for job_id in sorted(jobs_to_cancel):
            try:
                subprocess.run(['scancel', job_id], check=True, capture_output=True)
                print(f"Cancelled job {job_id}")
                successfully_canceled.append(job_id)
                # Mark as canceled
                self.config_manager.save_canceled_job(Project.name, job_id)  # type: ignore[arg-type]
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode().strip()
                print(f"Failed to cancel job {job_id}: {error_msg}")
                failed_to_cancel.append(job_id)
        
        print()
        print(f"Successfully cancelled {len(successfully_canceled)} job(s)")
        if failed_to_cancel:
            print(f"Failed to cancel {len(failed_to_cancel)} job(s)")
    
    def cat(self, job_spec: str, array_index: Optional[int] = None) -> None:
        """Print log file for a job.
        
        Args:
            job_spec: Job ID or job_id_arrayindex (e.g., "12345" or "12345_0")
            array_index: Optional array index if not in job_spec
        """
        # Parse job spec
        if '_' in job_spec:
            parts = job_spec.split('_')
            job_id = parts[0]
            array_index = int(parts[1])
        else:
            job_id = job_spec
        
        # Get log directory
        log_dir = Path(self.config.get('log_directory', str(Path.home() / ".experiments" / "logs")))
        
        if not log_dir.exists():
            print(f"Log directory does not exist: {log_dir}")
            return
        
        # Find log files matching this job ID
        if array_index is not None:
            # Look for specific array task log
            pattern = f"*_{job_id}_{array_index}.log"
        else:
            # Look for all logs for this job
            pattern = f"*_{job_id}_*.log"
        
        matching_logs = sorted(log_dir.glob(pattern))
        
        if not matching_logs:
            print(f"No log files found for job {job_id}" + (f" array index {array_index}" if array_index is not None else ""))
            print(f"Searched in: {log_dir}")
            print(f"Pattern: {pattern}")
            return
        
        # Print logs
        print("=" * 80)
        print(f"Job ID: {job_id}" + (f", Array Index: {array_index}" if array_index is not None else ""))
        print(f"Found {len(matching_logs)} log file(s)")
        print("=" * 80)
        print()
        
        for log_file in matching_logs:
            print(f"--- {log_file.name} ---")
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                    print(content)
                    if content and not content.endswith('\n'):
                        print()
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
            print()
    
    def history(self) -> None:
        """Print launch history."""
        if not Project.name:
            print("Error: No project specified in executor")
            sys.exit(1)
        
        # Collect all unique jobs across all stages
        all_jobs = []
        seen_job_ids = set()
        
        project_dir = self.config_manager.get_project_dir(Project.name)  # type: ignore[arg-type]
        if not project_dir.exists():
            print("No jobs found in history.")
            return
        
        stages_root = project_dir / "stages"
        if stages_root.exists():
            for stage_dir in sorted(stages_root.iterdir()):
                if stage_dir.is_dir():
                    jobs_file = stage_dir / "jobs.json"
                    if jobs_file.exists():
                        with open(jobs_file, 'r') as f:
                            import json
                            stage_jobs = json.load(f)
                            for job in stage_jobs:
                                job_id = job.get('job_id')
                                # Only add unique jobs (avoid duplicates)
                                if job_id and job_id not in seen_job_ids:
                                    seen_job_ids.add(job_id)
                                    job['stage'] = stage_dir.name
                                    all_jobs.append(job)
        
        if not all_jobs:
            print("No jobs found in history.")
            return
        
        # Sort by timestamp (oldest first, most recent at bottom)
        all_jobs.sort(key=lambda j: j.get('timestamp', ''))
        
        print("=" * 100)
        print(f"Job History for Project: {Project.name}")
        print("=" * 100)
        print()
        print(f"Total jobs: {len(all_jobs)}")
        print()
        
        # Print jobs
        print(f"{'Job ID':<12} {'Job Name':<30} {'Stage':<20} {'Tasks':<6} {'Submitted':<20}")
        print("-" * 100)
        
        for job in all_jobs:
            job_id = job.get('job_id', 'N/A')
            job_name = job.get('job_name', 'unknown')
            stage = job.get('stage', 'unknown')
            num_tasks = job.get('num_tasks', 'N/A')
            timestamp = job.get('timestamp', 'unknown')
            
            # Truncate long names
            if len(job_name) > 30:
                job_name = job_name[:27] + "..."
            if len(stage) > 20:
                stage = stage[:17] + "..."
            
            print(f"{job_id:<12} {job_name:<30} {stage:<20} {num_tasks:<6} {timestamp:<20}")
        
        print()
        print(f"Use 'cat <job_id>' or 'cat <job_id>_<array_index>' to view logs")
        print("=" * 100)
    
    def print_commands(self, stages: List[str], head: Optional[int] = None, tail: Optional[int] = None, rerun: bool = False, reverse: bool = False, exclude: Optional[List[str]] = None, artifacts: Optional[List[str]] = None, jobs: Optional[int] = None) -> None:
        """Print commands to run sequentially (can be piped to bash)."""
        from .executor import PrintExecutor
        
        # Create a PrintExecutor with the same paths as the SlurmExecutor
        print_executor = PrintExecutor(
            gs_path=self.executor.gs_path,
            setup_command=self.executor.setup_command,
        )
        
        # Copy stage information from the SlurmExecutor
        print_executor._stages = self.executor._stages
        
        # Execute stages using PrintExecutor (will print commands to stdout)
        selected = stages if stages else list(self.executor._stages.keys())
        
        # Exclude specified stages
        if exclude:
            selected = [s for s in selected if s not in exclude]
        
        if reverse:
            selected = list(reversed(selected))
        print_executor.execute(selected, head=head, tail=tail, rerun=rerun, artifacts=artifacts, jobs=jobs)
    
    def print_lines(self, stages: List[str], head: Optional[int] = None, tail: Optional[int] = None, rerun: bool = False, reverse: bool = False, exclude: Optional[List[str]] = None, artifacts: Optional[List[str]] = None, jobs: Optional[int] = None, output_dir: Optional[str] = None) -> None:
        """Print one line per job to execute individual bash scripts."""
        import tempfile
        from .executor import PrintExecutor
        
        # Create output directory for scripts
        if output_dir is None:
            # Create a temporary directory
            output_dir = tempfile.mkdtemp(prefix='experiment_scripts_')
        else:
            # Use provided directory
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            output_dir = str(output_dir_path)
        
        # Create a PrintExecutor with the same paths as the SlurmExecutor
        print_executor = PrintExecutor(
            gs_path=self.executor.gs_path,
            setup_command=self.executor.setup_command,
        )
        
        # Copy stage information from the SlurmExecutor
        print_executor._stages = self.executor._stages
        
        # Get selected stages
        selected = stages if stages else list(self.executor._stages.keys())
        
        # Exclude specified stages
        if exclude:
            selected = [s for s in selected if s not in exclude]
        
        if reverse:
            selected = list(reversed(selected))
        
        # Validate and normalize stages
        selected = print_executor._validate_and_normalize_stages(selected)
        unique_artifacts = print_executor._collect_unique_artifacts()
        
        if not unique_artifacts:
            print("No artifacts registered.", file=sys.stderr)
            return
        
        # Compute execution order via topological sort
        all_tiers = print_executor.compute_topological_ordering(unique_artifacts)
        
        # Filter to only artifacts in selected stages
        filtered_tiers = print_executor._filter_tiers_by_stages(all_tiers, selected)
        
        if not filtered_tiers:
            print("No artifacts matched the selected stages.", file=sys.stderr)
            return
        
        # Filter by artifact class names if specified
        if artifacts:
            filtered_tiers = print_executor._filter_tiers_by_artifact_class(filtered_tiers, artifacts)
            
            if not filtered_tiers:
                print(f"No artifacts matched the specified types: {', '.join(artifacts)}", file=sys.stderr)
                return
        
        # Filter out artifacts that should be skipped unless rerun flag is set
        executable_tiers, skipped_artifacts = print_executor._filter_skipped_artifacts(filtered_tiers, rerun=rerun)
        
        if not executable_tiers:
            print("All artifacts already exist. Nothing to execute.", file=sys.stderr)
            return
        
        # Apply head/tail filtering if requested
        if head is not None or tail is not None:
            executable_tiers = print_executor._apply_head_tail_filter(executable_tiers, head, tail)
            
            if not executable_tiers:
                print("No artifacts remain after filtering.", file=sys.stderr)
                return
        
        # Compile tasks
        task_tiers = [[print_executor.compile_artifact(a) for a in tier] for tier in executable_tiers]
        
        # Generate individual script files and print execution lines
        task_index = 0
        for tier in task_tiers:
            for task in tier:
                # Generate script content for this task
                script_lines = []
                script_lines.append("#!/usr/bin/env bash")
                
                # Add setup commands if provided
                if print_executor.setup_command:
                    script_lines.append(print_executor.setup_command)
                    script_lines.append("")
                
                script_lines.append("set -euo pipefail")
                script_lines.append("")
                
                # Export project configuration
                try:
                    proj_conf = self.config_manager.load_project_config(Project.name or "").get("config", {})
                except Exception:
                    proj_conf = {}
                proj_conf_str = json.dumps(proj_conf, separators=(",", ":"))
                from .executor import dquote
                script_lines.append(f"export EXPERIMENTS_PROJECT_CONF={dquote(proj_conf_str)}")
                script_lines.append("")
                
                # Export per-task experiment config
                if task.artifact is not None:
                    from .executor import _artifact_experiment_conf, _safe_json_dumps
                    exp_conf = _artifact_experiment_conf(task.artifact)
                    exp_conf_str = _safe_json_dumps(exp_conf)
                    script_lines.append(f"export EXPERIMENTS_EXPERIMENT_CONF={dquote(exp_conf_str)}")
                
                # Add task commands
                for block in task.blocks:
                    command = block.execute()
                    if command:
                        script_lines.append(command)
                
                # Write script to file
                script_filename = f"job_{task_index:04d}.sh"
                script_path = Path(output_dir) / script_filename
                
                with open(script_path, 'w') as f:
                    f.write("\n".join(script_lines))
                    f.write("\n")
                
                # Make script executable
                script_path.chmod(0o755)
                
                # Print execution line to stdout
                print(f"bash {script_path}")
                
                task_index += 1
    
    def export(self, stages: List[str], artifacts: Optional[List[str]] = None, exists_only: bool = False, output_file: str = None) -> None:
        """Export project configuration and artifacts to JSON file.
        
        Args:
            stages: List of stage names to export (empty for all)
            artifacts: Optional list of artifact class names to filter by
            exists_only: If True, only export artifacts that exist
            output_file: Path to output JSON file
        """
        if not output_file:
            print("Error: Output file is required", file=sys.stderr)
            sys.exit(1)
        
        # Determine which stages to export
        selected_stages = stages if stages else list(self.executor._stages.keys())
        
        # Build artifact-to-stages mapping
        artifact_to_stages: Dict[int, List[str]] = {}
        for stage_name, stage_artifacts in self.executor._stages.items():
            if stage_name not in selected_stages:
                continue
            for artifact in stage_artifacts:
                artifact_id = id(artifact)
                if artifact_id not in artifact_to_stages:
                    artifact_to_stages[artifact_id] = []
                artifact_to_stages[artifact_id].append(stage_name)
        
        # Collect unique artifacts
        seen_ids = set()
        unique_artifacts = []
        for stage_name in selected_stages:
            if stage_name not in self.executor._stages:
                print(f"Warning: Unknown stage '{stage_name}'", file=sys.stderr)
                continue
            for artifact in self.executor._stages[stage_name]:
                artifact_id = id(artifact)
                if artifact_id not in seen_ids:
                    seen_ids.add(artifact_id)
                    unique_artifacts.append(artifact)
        
        # Filter by artifact class if specified
        if artifacts:
            artifact_class_set = set(artifacts)
            unique_artifacts = [
                a for a in unique_artifacts
                if a.__class__.__name__ in artifact_class_set
            ]
        
        # Filter by exists if specified
        if exists_only:
            unique_artifacts = [a for a in unique_artifacts if a.exists]
        
        # Helper function to safely convert values to JSON
        def safe_json_value(value: Any) -> Any:
            """Convert value to JSON-safe format, return None if not convertible."""
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (list, tuple)):
                try:
                    return [safe_json_value(v) for v in value]
                except:
                    return None
            if isinstance(value, dict):
                try:
                    return {k: safe_json_value(v) for k, v in value.items()}
                except:
                    return None
            if isinstance(value, Path):
                return str(value)
            # For other types, try str() conversion
            try:
                return str(value)
            except:
                return None
        
        # Export artifacts
        exported_artifacts = []
        for artifact in unique_artifacts:
            artifact_id = id(artifact)
            artifact_dict = {
                'stage': artifact_to_stages.get(artifact_id, []),
                'artifact_type': artifact.__class__.__name__,
            }
            
            # Get data from as_dict()
            try:
                as_dict_data = artifact.as_dict()
                for key, value in as_dict_data.items():
                    artifact_dict[key] = safe_json_value(value)
            except Exception as e:
                artifact_dict['_as_dict_error'] = str(e)
            
            # Get @property attributes
            for attr_name in dir(artifact.__class__):
                try:
                    attr_value = getattr(artifact.__class__, attr_name)
                    if isinstance(attr_value, property):
                        # Try to get the property value
                        try:
                            prop_value = getattr(artifact, attr_name)
                            artifact_dict[attr_name] = safe_json_value(prop_value)
                        except Exception:
                            artifact_dict[attr_name] = None
                except:
                    pass
            
            exported_artifacts.append(artifact_dict)
        
        # Build export data
        export_data = {
            'project_config': {},
            'global_config': {},
            'artifacts': exported_artifacts
        }
        
        # Get project config
        if Project.name:
            try:
                proj_conf = self.config_manager.load_project_config(Project.name).get('config', {})
                export_data['project_config'] = safe_json_value(proj_conf)
            except Exception:
                pass
        
        # Get global config
        try:
            export_data['global_config'] = safe_json_value(self.config)
        except Exception:
            pass
        
        # Write to file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported {len(exported_artifacts)} artifact(s) to {output_file}", file=sys.stderr)
        if artifacts:
            print(f"  Filtered by artifact types: {', '.join(artifacts)}", file=sys.stderr)
        if exists_only:
            print(f"  Filtered to only existing artifacts", file=sys.stderr)
        print(f"  Stages: {', '.join(selected_stages)}", file=sys.stderr)


def auto_cli(executor: SlurmExecutor) -> None:
    """Main entry point for CLI."""
    cli = ExperimentCLI(executor)
    cli.run()
