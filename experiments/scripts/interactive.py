#!/usr/bin/env python3
"""Submit an interactive Slurm job with sensible defaults."""

import argparse
import re
import subprocess
import sys

# ============================================================================
# EXCLUDED NODES - Edit this list to exclude problematic nodes
# ============================================================================
EXCLUDED_NODES = [
    # "node001",
    # "node002",
]
# ============================================================================


def parse_gpus(gpu_arg):
    """Parse GPU argument which can be an int or 'TYPE:COUNT' string.
    
    Returns (gpu_string_for_sbatch, gpu_count_int).
    Examples:
        "4" -> ("4", 4)
        "H100:4" -> ("H100:4", 4)
        4 -> ("4", 4)
    """
    if gpu_arg is None:
        return (None, 0)
    
    gpu_str = str(gpu_arg)
    
    if ":" in gpu_str:
        # Format: TYPE:COUNT (e.g., "H100:4")
        parts = gpu_str.split(":")
        try:
            count = int(parts[-1])
            return (gpu_str, count)
        except ValueError:
            # If we can't parse count, assume 1
            return (gpu_str, 1)
    else:
        # Just a number
        try:
            count = int(gpu_str)
            return (gpu_str, count)
        except ValueError:
            # Just a type name, assume 1
            return (gpu_str, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Submit an interactive Slurm job",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--partition", "-p",
        default="general",
        help="Partition to submit to",
    )
    parser.add_argument(
        "--time", "-t",
        default=None,
        help="Time limit (default: 2-00:00:00, or 0-02:00:00 for debug partition)",
    )
    parser.add_argument(
        "--gpus", "-g",
        default=None,
        help="Number of GPUs, or TYPE:COUNT (e.g., 4 or H100:4)",
    )
    parser.add_argument(
        "--cpus", "-c",
        type=int,
        default=None,
        help="Number of CPUs (default: GPUs + 2)",
    )
    parser.add_argument(
        "--mem",
        default=None,
        help="Memory (default: 64G + 16G per GPU)",
    )
    parser.add_argument(
        "--job-name", "-J",
        default="interactive",
        help="Job name",
    )
    parser.add_argument(
        "--account", "-A",
        default=None,
        help="Account to charge",
    )
    parser.add_argument(
        "--qos",
        default=None,
        help="Quality of Service",
    )
    parser.add_argument(
        "--constraint", "-C",
        default=None,
        help="Node constraint",
    )
    parser.add_argument(
        "--nodelist", "-w",
        default=None,
        help="Specific nodes to use",
    )
    parser.add_argument(
        "--exclude", "-x",
        default=None,
        help="Nodes to exclude (in addition to EXCLUDED_NODES)",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print the command without executing",
    )
    return parser.parse_args()


def time_to_seconds(time_str):
    """Convert Slurm time format (D-HH:MM:SS or HH:MM:SS) to seconds."""
    days = 0
    if "-" in time_str:
        day_part, time_part = time_str.split("-", 1)
        days = int(day_part)
    else:
        time_part = time_str
    
    parts = time_part.split(":")
    if len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        hours, minutes, seconds = 0, int(parts[0]), int(parts[1])
    else:
        hours, minutes, seconds = 0, 0, int(parts[0])
    
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def build_sbatch_script(args):
    """Build the sbatch script content."""
    lines = ["#!/usr/bin/env bash"]
    
    # Parse GPU specification
    gpu_str, gpu_count = parse_gpus(args.gpus)
    
    # Job name
    lines.append(f"#SBATCH --job-name={args.job_name}")
    
    # Partition
    lines.append(f"#SBATCH --partition={args.partition}")
    
    # Time limit - default depends on partition
    if args.time is not None:
        time_limit = args.time
    elif args.partition == "debug":
        time_limit = "0-02:00:00"
    else:
        time_limit = "2-00:00:00"
    lines.append(f"#SBATCH --time={time_limit}")
    
    # CPUs - default is GPUs + 2
    cpus = args.cpus if args.cpus is not None else gpu_count + 2
    lines.append(f"#SBATCH --cpus-per-task={cpus}")
    
    # GPUs
    if gpu_str is not None and gpu_count > 0:
        lines.append(f"#SBATCH --gpus={gpu_str}")
    
    # Memory - default is 64G + 16G per GPU
    if args.mem is not None:
        mem = args.mem
    else:
        mem_gb = 64 + (16 * gpu_count)
        mem = f"{mem_gb}G"
    lines.append(f"#SBATCH --mem={mem}")
    
    # Optional: account
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    
    # Optional: QoS
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    
    # Optional: constraint
    if args.constraint:
        lines.append(f"#SBATCH --constraint={args.constraint}")
    
    # Optional: nodelist
    if args.nodelist:
        lines.append(f"#SBATCH --nodelist={args.nodelist}")
    
    # Exclude nodes - combine hardcoded list with user-specified
    exclude_nodes = list(EXCLUDED_NODES)
    if args.exclude:
        exclude_nodes.extend(args.exclude.split(","))
    if exclude_nodes:
        lines.append(f"#SBATCH --exclude={','.join(exclude_nodes)}")
    
    # Output file
    lines.append(f"#SBATCH --output={args.job_name}_%j.log")
    
    # Script body - sleep for the duration
    sleep_seconds = time_to_seconds(time_limit)
    lines.append("")
    lines.append(f"echo \"Interactive job started on $(hostname) at $(date)\"")
    lines.append(f"echo \"Job ID: $SLURM_JOB_ID\"")
    lines.append(f"echo \"Sleeping for {sleep_seconds} seconds ({time_limit})...\"")
    lines.append(f"echo \"Connect with: srun --jobid=$SLURM_JOB_ID --pty bash\"")
    lines.append(f"sleep {sleep_seconds}")
    
    return "\n".join(lines)


def main():
    args = parse_args()
    script = build_sbatch_script(args)
    
    # Print the script
    print("=== SBATCH Script ===", file=sys.stderr)
    print(script, file=sys.stderr)
    print("=====================", file=sys.stderr)
    
    if args.dry_run:
        print("\n[Dry run - not executing]", file=sys.stderr)
        return 0
    
    # Submit via sbatch
    print("\nSubmitting job...", file=sys.stderr)
    try:
        result = subprocess.run(
            ["sbatch"],
            input=script,
            text=True,
            capture_output=True,
        )
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        
        if result.returncode == 0:
            # Extract job ID and print connection instructions
            match = re.search(r"(\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                print(f"\nConnect with: srun --jobid={job_id} --pty bash", file=sys.stderr)
        
        return result.returncode
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())