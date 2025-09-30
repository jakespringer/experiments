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


class ConfigManager:
    """Manages experiment configuration and state."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".experiments"
        self.config_file = self.config_dir / "config.json"
        self.projects_dir = self.config_dir / "projects"
    
    def ensure_config(self) -> Dict[str, Any]:
        """Ensure config directory and file exist, return config."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.config_file.exists():
            # Create default config
            default_config = {
                "log_directory": str(self.config_dir / "logs"),
                "default_slurm_args": {
                    "general": {
                        "time": "2-00:00:00",
                        "cpus": 1,
                        "requeue": False
                    },
                    "array": {
                        "time": "2-00:00:00",
                        "cpus": 4,
                        "requeue": True
                    },
                    "cpu": {
                        "time": "1-00:00:00",
                        "cpus": 1,
                        "requeue": False
                    }
                }
            }
            self.save_config(default_config)
            
            # Create log directory
            Path(default_config["log_directory"]).mkdir(parents=True, exist_ok=True)
            
            return default_config
        
        return self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_file, 'r') as f:
            return json.load(f)
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_project_dir(self, project: str) -> Path:
        """Get project directory, creating it if needed."""
        project_dir = self.projects_dir / project
        project_dir.mkdir(parents=True, exist_ok=True)
        return project_dir
    
    def get_stage_dir(self, project: str, stage: str) -> Path:
        """Get stage directory, creating it if needed."""
        stage_dir = self.get_project_dir(project) / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        return stage_dir
    
    def save_job_info(self, project: str, stage: str, job_info: Dict[str, Any]) -> None:
        """Save job information for a stage."""
        stage_dir = self.get_stage_dir(project, stage)
        jobs_file = stage_dir / "jobs.json"
        
        # Load existing jobs
        jobs = []
        if jobs_file.exists():
            with open(jobs_file, 'r') as f:
                jobs = json.load(f)
        
        # Add new job info
        jobs.append(job_info)
        
        # Save
        with open(jobs_file, 'w') as f:
            json.dump(jobs, f, indent=2)
    
    def load_jobs(self, project: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load job information for a stage or all stages."""
        jobs = []
        
        if stage:
            # Load jobs for specific stage
            stage_dir = self.get_stage_dir(project, stage)
            jobs_file = stage_dir / "jobs.json"
            if jobs_file.exists():
                with open(jobs_file, 'r') as f:
                    stage_jobs = json.load(f)
                    for job in stage_jobs:
                        job['stage'] = stage
                    jobs.extend(stage_jobs)
        else:
            # Load jobs for all stages
            project_dir = self.get_project_dir(project)
            for stage_dir in project_dir.iterdir():
                if stage_dir.is_dir():
                    jobs_file = stage_dir / "jobs.json"
                    if jobs_file.exists():
                        with open(jobs_file, 'r') as f:
                            stage_jobs = json.load(f)
                            for job in stage_jobs:
                                job['stage'] = stage_dir.name
                            jobs.extend(stage_jobs)
        
        return jobs


class ExperimentCLI:
    """Command-line interface for experiments."""
    
    def __init__(self, executor: SlurmExecutor):
        self.executor = executor
        self.config_manager = ConfigManager()
        self.config = self.config_manager.ensure_config()
    
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
        
        # drylaunch command
        drylaunch_parser = subparsers.add_parser('drylaunch', help='Dry run: show what would be launched')
        drylaunch_parser.add_argument(
            'stages',
            nargs='*',
            help='Stage names to run (omit for all stages)'
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
        
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        # Route to appropriate handler
        if args.command == 'launch':
            self.launch(args.stages, dry_run=False)
        elif args.command == 'drylaunch':
            self.launch(args.stages, dry_run=True)
        elif args.command == 'cancel':
            self.cancel(args.stages)
        elif args.command == 'cat':
            self.cat(args.job_spec, args.array_index)
        elif args.command == 'history':
            self.history()
    
    def launch(self, stages: List[str], dry_run: bool = False) -> None:
        """Launch experiment stages."""
        # Apply config settings to executor
        self.executor.dry_run = dry_run
        self.executor.config_manager = self.config_manager
        self.executor.config = self.config
        
        # Get default slurm args for partitions from config
        if 'default_slurm_args' in self.config:
            self.executor.default_slurm_args_by_partition = self.config['default_slurm_args']
        
        # Execute stages
        selected = stages if stages else list(self.executor._stages.keys())
        self.executor.execute(selected)
    
    def cancel(self, stages: List[str]) -> None:
        """Cancel jobs for the specified stages."""
        if not self.executor.project:
            print("Error: No project specified in executor")
            sys.exit(1)
        
        # Determine which stages to cancel
        stages_to_cancel = stages if stages else list(self.executor._stages.keys())
        
        print(f"Cancelling jobs for stages: {', '.join(stages_to_cancel)}")
        print()
        
        # Load jobs for each stage
        all_jobs_to_cancel = []
        for stage in stages_to_cancel:
            jobs = self.config_manager.load_jobs(self.executor.project, stage)
            all_jobs_to_cancel.extend(jobs)
        
        if not all_jobs_to_cancel:
            print("No jobs found to cancel.")
            return
        
        # Group by job_id (since array jobs share one ID)
        job_ids = set()
        for job in all_jobs_to_cancel:
            if 'job_id' in job:
                job_ids.add(job['job_id'])
        
        # Cancel each job
        for job_id in sorted(job_ids):
            try:
                subprocess.run(['scancel', job_id], check=True, capture_output=True)
                print(f"Cancelled job {job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to cancel job {job_id}: {e.stderr.decode()}")
        
        print(f"\nCancelled {len(job_ids)} job(s)")
    
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
        if not self.executor.project:
            print("Error: No project specified in executor")
            sys.exit(1)
        
        # Collect all unique jobs across all stages
        all_jobs = []
        seen_job_ids = set()
        
        project_dir = self.config_manager.get_project_dir(self.executor.project)
        if not project_dir.exists():
            print("No jobs found in history.")
            return
        
        for stage_dir in sorted(project_dir.iterdir()):
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
        print(f"Job History for Project: {self.executor.project}")
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


def auto_cli(executor: SlurmExecutor) -> None:
    """Main entry point for CLI."""
    cli = ExperimentCLI(executor)
    cli.run()
