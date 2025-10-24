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
            help='Override Slurm args for all jobs (e.g., time=12:00:00 gpus=4)'
        )
        launch_parser.add_argument(
            '--force-launch',
            action='store_true',
            help='Ignore running check and launch jobs anyway'
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
            help='Override Slurm args for all jobs (no submission)'
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
            )
        elif args.command == 'cancel':
            self.cancel(args.stages)
        elif args.command == 'cat':
            self.cat(args.job_spec, args.array_index)
        elif args.command == 'history':
            self.history()
        elif args.command == 'print':
            self.print_commands(args.stages, head=getattr(args, 'head', None), tail=getattr(args, 'tail', None), rerun=getattr(args, 'rerun', False), reverse=getattr(args, 'reverse', False), exclude=getattr(args, 'exclude', None), artifacts=getattr(args, 'artifact', None), jobs=getattr(args, 'jobs', None))
    
    def launch(self, stages: List[str], dry_run: bool = False, head: Optional[int] = None, tail: Optional[int] = None, rerun: bool = False, reverse: bool = False, exclude: Optional[List[str]] = None, artifacts: Optional[List[str]] = None, jobs: Optional[int] = None, dependency: Optional[List[str]] = None, slurm_overrides: Optional[List[str]] = None, force_launch: bool = False) -> None:
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
        if hasattr(self.executor, 'cli_slurm_overrides'):
            self.executor.cli_slurm_overrides = overrides
        if hasattr(self.executor, 'force_launch'):
            self.executor.force_launch = bool(force_launch)
        
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


def auto_cli(executor: SlurmExecutor) -> None:
    """Main entry point for CLI."""
    cli = ExperimentCLI(executor)
    cli.run()
