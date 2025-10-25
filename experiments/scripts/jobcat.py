#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def find_job_info(job_id: str):
    """Search all projects for job information.
    
    Args:
        job_id: The Slurm job ID to search for
        
    Returns:
        Tuple of (job_info dict, project_name, stage_name) or (None, None, None) if not found
    """
    experiments_dir = Path.home() / ".experiments"
    projects_dir = experiments_dir / "projects" 
    
    if not projects_dir.exists():
        return None, None, None
    
    # Search through all projects and stages
    for project_dir in projects_dir.iterdir():
        stages_dir = project_dir / "stages"

        if not project_dir.is_dir():
            continue
            
        for stage_dir in stages_dir.iterdir():
            if not stage_dir.is_dir():
                continue
                
            jobs_file = stage_dir / "jobs.json"
            if not jobs_file.exists():
                continue
                
            try:
                with open(jobs_file, 'r') as f:
                    jobs = json.load(f)
                    
                for job in jobs:
                    if job.get('job_id') == job_id:
                        return job, project_dir.name, stage_dir.name
            except (json.JSONDecodeError, IOError):
                # Skip invalid or unreadable files
                continue
    
    return None, None, None


def find_log_file(log_pattern: str, job_id: str, array_id: str = None):
    """Find log file(s) matching the pattern.
    
    Args:
        log_pattern: Log file pattern from job info (e.g., /path/to/logs/job_%A_%a.out)
        job_id: The Slurm job ID
        array_id: Optional array task ID
        
    Returns:
        List of matching log file paths
    """
    # Replace Slurm placeholders
    # %A = job ID, %a = array task ID
    # The pattern might be something like: /path/to/logs/tier-0_%A_%a.out
    
    # Extract the directory and filename pattern
    log_path = Path(log_pattern)
    log_dir = log_path.parent
    filename_pattern = log_path.name
    
    if not log_dir.exists():
        return []
    
    # Replace %A with actual job ID
    filename_pattern = filename_pattern.replace('%A', job_id)
    
    if array_id is not None:
        # Looking for specific array task
        filename_pattern = filename_pattern.replace('%a', array_id)
        matching_files = list(log_dir.glob(filename_pattern))
    else:
        # Looking for all array tasks - replace %a with wildcard
        filename_pattern = filename_pattern.replace('%a', '*')
        matching_files = list(log_dir.glob(filename_pattern))
    
    return sorted(matching_files)


def print_log_file(log_file: Path, show_header: bool = True):
    """Print the contents of a log file.
    
    Args:
        log_file: Path to the log file
        show_header: Whether to print a header with the filename
    """
    if show_header:
        print("=" * 80)
        print(f"Log file: {log_file}")
        print("=" * 80)
        print()
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            print(content, end='')
            if content and not content.endswith('\n'):
                print()
    except IOError as e:
        print(f"Error reading {log_file}: {e}", file=sys.stderr)
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Display Slurm job logs from experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jobcat 12345              # Show all array tasks for job 12345
  jobcat 12345_0            # Show only array task 0 of job 12345
        """
    )
    parser.add_argument(
        'job_spec',
        help='Job ID or job_id_arrayindex (e.g., 12345 or 12345_0)'
    )
    
    args = parser.parse_args()
    
    # Parse job spec
    job_spec = args.job_spec
    array_id = None
    
    if '_' in job_spec:
        parts = job_spec.split('_', 1)
        job_id = parts[0]
        array_id = parts[1]
    else:
        job_id = job_spec
    
    # Find job information
    job_info, project_name, stage_name = find_job_info(job_id)
    
    if job_info is None:
        print(f"Error: Job {job_id} not found in any project", file=sys.stderr)
        print(f"\nSearched in: {Path.home() / '.experiments' / 'projects'}", file=sys.stderr)
        sys.exit(1)
    
    # Get log file pattern from job info
    log_pattern = job_info.get('log_file')
    
    if not log_pattern:
        print(f"Error: No log file information found for job {job_id}", file=sys.stderr)
        sys.exit(1)
    
    # Find matching log files
    log_files = find_log_file(log_pattern, job_id, array_id)
    
    if not log_files:
        print(f"Error: No log files found for job {job_id}", file=sys.stderr)
        if array_id:
            print(f"       Array task: {array_id}", file=sys.stderr)
        print(f"\nExpected pattern: {log_pattern}", file=sys.stderr)
        
        # Try to provide helpful information
        log_path = Path(log_pattern)
        log_dir = log_path.parent
        if not log_dir.exists():
            print(f"Log directory does not exist: {log_dir}", file=sys.stderr)
        else:
            # Show what files exist in the directory
            all_files = list(log_dir.glob(f"*{job_id}*"))
            if all_files:
                print(f"\nFiles in {log_dir} matching job ID:", file=sys.stderr)
                for f in sorted(all_files)[:10]:  # Show first 10
                    print(f"  - {f.name}", file=sys.stderr)
        
        sys.exit(1)
    
    # Print log file(s)
    show_header = len(log_files) > 1
    
    if show_header:
        print(f"Found {len(log_files)} log file(s) for job {job_id}")
        if project_name and stage_name:
            print(f"Project: {project_name}, Stage: {stage_name}")
        print()
    
    success = True
    for i, log_file in enumerate(log_files):
        if i > 0:
            print()  # Blank line between multiple files
        
        if not print_log_file(log_file, show_header=show_header):
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()