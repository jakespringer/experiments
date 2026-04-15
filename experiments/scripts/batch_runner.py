"""Parallel batch child launcher with real-time log file streaming.

Usage:
    python -m experiments.scripts.batch_runner <config.json>

Expects environment variables:
    BATCH_LOG_DIR   — directory for child log files
    BATCH_JOB_ID    — Slurm job ID (for log file naming)
    BATCH_ARRAY_ID  — Slurm array task ID (for log file naming)
    BATCH_ALL_GPUS  — comma-separated GPU indices (optional)

The config.json contains a list of child descriptors, each with:
    index       — child index
    relpath     — artifact relpath (for display)
    script_b64  — base64-encoded bash script to run
    gpu_offset  — starting index into the GPU list
    gpu_count   — number of GPUs for this child
"""

import base64
import json
import os
import subprocess
import sys
import tempfile
import threading


def run_child(child, log_dir, job_id, array_id, all_gpus, lock, results):
    idx = child['index']
    relpath = child['relpath']
    script = base64.b64decode(child['script_b64']).decode()
    gpu_offset = child['gpu_offset']
    gpu_count = child['gpu_count']

    log_path = os.path.join(log_dir, f'batch_{job_id}_{array_id}_child_{idx}.out')

    # Create the log file immediately so it is visible to jobcat while the
    # child is still running.  We keep it open for the lifetime of the child
    # and flush after every line.
    os.makedirs(os.path.dirname(log_path) or log_dir, exist_ok=True)
    log_file = open(log_path, 'w')
    try:
        log_file.write(f'[child {idx}] START: {relpath}\n')
        log_file.flush()
        os.fsync(log_file.fileno())

        fd, script_path = tempfile.mkstemp(suffix='.sh', prefix=f'batch_child_{idx}_')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(script)
            os.chmod(script_path, 0o755)

            # Set up per-child environment
            env = os.environ.copy()
            # Force line-buffered / unbuffered stdout in child processes so
            # that output streams into the log file in real time.
            env['PYTHONUNBUFFERED'] = '1'
            if gpu_count > 0 and all_gpus:
                gpu_indices = all_gpus[gpu_offset:gpu_offset + gpu_count]
                if len(gpu_indices) < gpu_count:
                    with lock:
                        print(
                            f'[batch] WARNING: child {idx} requested {gpu_count} GPUs '
                            f'(offset {gpu_offset}) but only {len(gpu_indices)} available '
                            f'(total GPUs: {len(all_gpus)})',
                            file=sys.stderr, flush=True,
                        )
                env['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_indices)
            else:
                env['CUDA_VISIBLE_DEVICES'] = ''

            with lock:
                print(
                    f'[batch] Starting child {idx}: {relpath}\n'
                    f'[batch]   log: {log_path}',
                    file=sys.stderr, flush=True,
                )

            # Use stdbuf to force line-buffered stdout for non-Python
            # processes (C/C++ programs, etc.) that ignore PYTHONUNBUFFERED.
            proc = subprocess.Popen(
                ['stdbuf', '-oL', 'bash', script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )

            for line in iter(proc.stdout.readline, b''):
                decoded = line.decode('utf-8', errors='replace')
                log_file.write(decoded)
                log_file.flush()

            rc = proc.wait()

            if rc != 0:
                log_file.write(f'[child {idx}] FAILED with exit code {rc}\n')
                log_file.flush()
                with lock:
                    print(
                        f'[batch] Child {idx} (pid {proc.pid}) FAILED with exit code {rc}',
                        file=sys.stderr, flush=True,
                    )
                    results[idx] = rc
            else:
                log_file.write(f'[child {idx}] DONE (exit 0)\n')
                log_file.flush()
                with lock:
                    print(
                        f'[batch] Child {idx} (pid {proc.pid}) finished successfully',
                        file=sys.stderr, flush=True,
                    )
                    results[idx] = rc
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
    except Exception as e:
        log_file.write(f'[child {idx}] EXCEPTION: {e}\n')
        log_file.flush()
        with lock:
            print(f'[batch] Child {idx} EXCEPTION: {e}', file=sys.stderr, flush=True)
            results[idx] = 1
    finally:
        log_file.close()


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m experiments.scripts.batch_runner <config.json>', file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)

    if not config:
        print('[batch] WARNING: config is empty, no children to run', file=sys.stderr)
        return

    log_dir = os.environ['BATCH_LOG_DIR']
    job_id = os.environ['BATCH_JOB_ID']
    array_id = os.environ['BATCH_ARRAY_ID']
    all_gpus_str = os.environ.get('BATCH_ALL_GPUS', '')
    all_gpus = (
        [g.strip() for g in all_gpus_str.split(',') if g.strip()]
        if all_gpus_str else []
    )

    os.makedirs(log_dir, exist_ok=True)

    # Print summary of all children and their log paths up front so the
    # main Slurm log captures where every child's output will live.
    print(f'[batch] Launching {len(config)} children (job={job_id}, array={array_id})', file=sys.stderr, flush=True)
    for child in config:
        child_log = os.path.join(log_dir, f'batch_{job_id}_{array_id}_child_{child["index"]}.out')
        print(f'[batch]   child {child["index"]}: {child["relpath"]} -> {child_log}', file=sys.stderr, flush=True)

    lock = threading.Lock()
    results = {}

    threads = []
    for child in config:
        t = threading.Thread(
            target=run_child,
            args=(child, log_dir, job_id, array_id, all_gpus, lock, results),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if len(results) != len(config):
        missing = [c['index'] for c in config if c['index'] not in results]
        print(f'ERROR: {len(missing)} children did not report a result: {missing}', file=sys.stderr, flush=True)
        sys.exit(1)
    if any(rc != 0 for rc in results.values()):
        print('ERROR: One or more batch children failed', file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
