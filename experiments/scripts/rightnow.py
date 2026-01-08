#!/usr/bin/env python3
"""
SLURM Resource Estimator - Estimates the largest job parameters that can run RIGHT NOW
considering fairshare policy, priorities, and backfill scheduling.

This tool implements SLURM's backfill scheduling logic to determine what job
parameters would allow immediate scheduling.

Author: AI Assistant
License: MIT
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.layout import Layout
    from rich import box
except ImportError:
    print("Error: The 'rich' library is required. Install it with: pip install rich")
    sys.exit(1)


@dataclass
class NodeResources:
    """Represents available resources on a node"""
    name: str
    state: str
    cpus: int
    cpus_alloc: int
    memory: int  # MB
    memory_alloc: int  # MB
    gpus: int
    gpus_alloc: int
    partition: str
    partitions: List[str]
    gpu_type: Optional[str]
    features: Set[str]

    @property
    def cpus_free(self) -> int:
        return max(0, self.cpus - self.cpus_alloc)

    @property
    def memory_free(self) -> int:
        return max(0, self.memory - self.memory_alloc)

    @property
    def gpus_free(self) -> int:
        return max(0, self.gpus - self.gpus_alloc)

    @property
    def is_usable(self) -> bool:
        state_lower = self.state.lower()
        return ('idle' in state_lower or 'mix' in state_lower) and not any(
            x in state_lower for x in ['drain', 'down', 'maint', 'reboot', 'reserved']
        )


@dataclass
class RunningJob:
    """Represents a running job"""
    job_id: str
    user: str
    cpus: int
    gpus: int
    gpu_type: Optional[str]
    memory: int  # MB
    nodes: List[str]
    end_time: Optional[datetime]
    partition: str


@dataclass
class PendingJob:
    """Represents a pending job in the queue"""
    job_id: str
    user: str
    priority: int
    cpus: int
    gpus: int
    gpu_type: Optional[str]
    memory: int  # MB
    time_limit_minutes: int
    partition: str
    state: str
    reason: str
    start_time: Optional[datetime]
    start_time_available: bool

    def can_run_on_node(self, node: "NodeResources") -> bool:
        partition_match = self.partition in node.partitions if self.partition else True
        gpu_type_match = True
        if self.gpu_type and node.gpu_type:
            gpu_type_match = self.gpu_type.lower() == node.gpu_type.lower()
        return (
            partition_match
            and gpu_type_match
            and node.cpus >= self.cpus
            and node.memory >= self.memory
            and node.gpus >= self.gpus
        )


@dataclass
class JobBrief:
    job_id: str
    priority: int
    state: str
    start_time: Optional[datetime]
    reason: str
    user: str
    partition: str


@dataclass
class JobDetail:
    job_id: str
    state: str
    reason: str
    priority: int
    partition: str
    user: str
    gpus: int
    gpu_type: Optional[str]
    cpus: int
    memory: int
    time_limit_minutes: int
    start_time: Optional[datetime]
    submit_time: Optional[datetime]


@dataclass
class UserPriority:
    fairshare: float
    account: str
    norm_shares: float
    raw_usage: int
    effective_usage: float


@dataclass
class PriorityWeights:
    fairshare: int = 1000
    age: int = 1000
    job_size: int = 1000
    partition: int = 1000
    qos: int = 0


@dataclass
class PartitionInfo:
    max_time_minutes: int
    priority_factor: float
    max_nodes: Optional[int] = None
    max_jobs: Optional[int] = None
    total_nodes: Optional[int] = None


@dataclass
class ResourceEvent:
    time: Optional[datetime]
    delta_cpus: int
    delta_gpus: int
    delta_mem: int
    kind: str  # 'release' or 'alloc'
    label: str


@dataclass
class NodeSimState:
    node: NodeResources
    free_cpus: int
    free_gpus: int
    free_mem: int
    events: List[ResourceEvent] = field(default_factory=list)
    scheduled: List[Tuple[str, int, Optional[datetime]]] = field(default_factory=list)
    running: List[RunningJob] = field(default_factory=list)


@dataclass
class PartitionEvent:
    time: Optional[datetime]
    delta_jobs: int
    delta_nodes: int
    delta_gpus: int
    kind: str  # 'release' or 'alloc'
    label: str


@dataclass
class PartitionSimState:
    partition: str
    max_jobs: int
    max_nodes: int
    max_gpus: int
    free_jobs: int
    free_nodes: int
    free_gpus: int
    events: List[PartitionEvent] = field(default_factory=list)

class SlurmQueryError(Exception):
    """Raised when SLURM commands fail"""
    pass


class SlurmBase:
    def _run_command(self, cmd: List[str]) -> str:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise SlurmQueryError(f"Command failed: {' '.join(cmd)}\nError: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise SlurmQueryError(f"Command timed out: {' '.join(cmd)}")

    @staticmethod
    def _parse_slurm_time(time_str: str) -> Optional[datetime]:
        if not time_str or time_str in ['N/A', 'Unknown', 'None', ''] or time_str.startswith('N/A'):
            return None
        try:
            if 'T' in time_str:
                return datetime.fromisoformat(time_str)
            return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except Exception:
            return None

    @staticmethod
    def _parse_time_limit(time_str: str) -> int:
        if not time_str or time_str in ['UNLIMITED', 'NOT_SET']:
            return 10080
        try:
            if '-' in time_str:
                days, time = time_str.split('-')
                parts = time.split(':')
                h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
                return int(days) * 24 * 60 + h * 60 + m
            parts = time_str.split(':')
            if len(parts) == 3:
                h, m, _ = map(int, parts)
                return h * 60 + m
            if len(parts) == 2:
                m, _ = map(int, parts)
                return m
            if len(parts) == 1:
                return int(parts[0])
            return 60
        except Exception:
            return 60

    @staticmethod
    def _parse_memory(mem_str: str) -> int:
        if not mem_str or mem_str in ['0', 'N/A']:
            return 0
        try:
            if 'T' in mem_str:
                return int(float(mem_str.replace('T', '')) * 1024 * 1024)
            if 'G' in mem_str:
                return int(float(mem_str.replace('G', '')) * 1024)
            if 'M' in mem_str:
                return int(float(mem_str.replace('M', '')))
            if 'K' in mem_str:
                return int(float(mem_str.replace('K', '')) / 1024)
            return int(mem_str)
        except Exception:
            return 0

    @staticmethod
    def _parse_gres_string(gres: str) -> Tuple[int, Optional[str]]:
        if not gres or gres in ['(null)', 'N/A', 'n/a']:
            return 0, None
        entry = gres.split(',')[0].strip()
        if not entry.startswith('gpu'):
            return 0, None
        entry = entry.split('(')[0].strip()
        parts = entry.split(':')
        gpu_type = None
        count = 0
        if len(parts) == 3:
            _, gpu_type, count_str = parts
            try:
                count = int(count_str)
            except Exception:
                count = 0
        elif len(parts) == 2:
            _, tail = parts
            if tail.isdigit():
                count = int(tail)
            else:
                gpu_type = tail
        elif len(parts) == 1:
            count = 0
        return count, gpu_type

    @staticmethod
    def _parse_req_gpus(spec: str) -> Tuple[int, Optional[str]]:
        """Parse requested GPUs from ReqTRES/TresPerNode/Gres-style strings."""
        if not spec or spec in ['(null)', 'N/A', 'n/a']:
            return 0, None
        counts: Dict[str, int] = {}
        types: Set[str] = set()
        # Match both colon and equals separators, e.g., gres/gpu:L40:1 or gres/gpu:A100=4
        for m in re.findall(r'gres/gpu(?::([A-Za-z0-9_\-+]+))?[:=](\d+)', spec, flags=re.IGNORECASE):
            maybe_type, count_str = m
            key = (maybe_type or '_generic').lower()
            try:
                count = int(count_str)
            except Exception:
                continue
            counts[key] = max(counts.get(key, 0), count)
            if maybe_type:
                types.add(maybe_type)
        if counts:
            generic = counts.get('_generic', 0)
            typed_keys = [k for k in counts if k != '_generic']
            if typed_keys:
                max_typed = max(counts[k] for k in typed_keys)
                gpu_type = typed_keys[0].upper() if len(typed_keys) == 1 else None
                total = max(max_typed, generic)  # avoid double-counting generic + typed
                return total, gpu_type
            return generic, None
        # Fallback to classic gres parsing (gpu:TYPE:COUNT or gpu:COUNT)
        return SlurmBase._parse_gres_string(spec)

    @staticmethod
    def _parse_req_mem(spec: str) -> int:
        """Parse requested memory (MB) from ReqTRES-style strings."""
        if not spec or spec in ['(null)', 'N/A', 'n/a']:
            return 0
        for part in spec.split(','):
            if part.strip().startswith('mem='):
                return SlurmBase._parse_memory(part.split('=', 1)[1])
        return 0

    @staticmethod
    def _parse_alloc_gpus(tres: str) -> int:
        if not tres:
            return 0
        matches = re.findall(r'gres/gpu(?::[A-Za-z0-9_]+)?=(\d+)', tres, flags=re.IGNORECASE)
        if matches:
            try:
                return max(int(m) for m in matches)
            except Exception:
                return 0
        return 0

    @staticmethod
    def _expand_node_list(node_list: str) -> List[str]:
        if not node_list or node_list in ['(null)', 'N/A']:
            return []
        nodes: List[str] = []
        if '[' in node_list:
            prefix = node_list.split('[')[0]
            range_part = node_list.split('[')[1].rstrip(']')
            for part in range_part.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    for i in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{i:0{len(start)}d}")
                else:
                    nodes.append(f"{prefix}{part}")
        else:
            nodes.append(node_list)
        return nodes


class SlurmEstimator(SlurmBase):
    """Main class for estimating SLURM job parameters"""
    
    def __init__(
        self,
        max_gpus: int = 16,
        verbose: bool = False,
        partition: Optional[str] = None,
        user: Optional[str] = None,
        gpu_type: Optional[str] = None
    ):
        self.console = Console()
        self.max_gpus = min(max_gpus, 8)
        self.verbose = verbose
        self.filter_partition = partition
        self.filter_user = user
        self.filter_gpu_type = gpu_type.lower() if gpu_type else None
        self.nodes: List[NodeResources] = []
        self.running_jobs: List[RunningJob] = []
        self.pending_jobs: List[PendingJob] = []
        self.user_priority: Optional[UserPriority] = None
        self.priority_weights: PriorityWeights = PriorityWeights()
        self.partition_limits: Dict[str, int] = {}
        self.partition_priority: Dict[str, float] = {}
        self.partition_info: Dict[str, PartitionInfo] = {}
        self.allowed_partitions: Optional[Set[str]] = None
        self.current_user = self._get_current_user()
        if self.filter_user:
            self.current_user = self.filter_user
        self.now = datetime.now()
        self.last_update = None
        self.update_count = 0
        self.error_message = None
        
    def _get_current_user(self) -> str:
        """Get current username"""
        try:
            result = subprocess.run(['whoami'], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except:
            return "unknown"
    
    def _run_command(self, cmd: List[str]) -> str:
        """Run a shell command and return output"""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
            return result.stdout
        except subprocess.CalledProcessError as e:
            raise SlurmQueryError(f"Command failed: {' '.join(cmd)}\nError: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise SlurmQueryError(f"Command timed out: {' '.join(cmd)}")
    
    def _parse_slurm_time(self, time_str: str) -> Optional[datetime]:
        """Parse SLURM time format to datetime"""
        if not time_str or time_str in ['N/A', 'Unknown', 'None', '']:
            return None
        
        try:
            if 'T' in time_str:
                return datetime.fromisoformat(time_str)
            return datetime.strptime(time_str, '%Y-%m-%dT%H:%M:%S')
        except:
            return None
    
    def _parse_time_limit(self, time_str: str) -> int:
        """Convert time limit string to minutes"""
        if not time_str or time_str in ['UNLIMITED', 'NOT_SET']:
            return 10080
        
        try:
            if '-' in time_str:
                days, time = time_str.split('-')
                parts = time.split(':')
                h, m = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
                return int(days) * 24 * 60 + h * 60 + m
            else:
                parts = time_str.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 60 + m
                elif len(parts) == 2:
                    m, s = map(int, parts)
                    return m
                elif len(parts) == 1:
                    return int(parts[0])
                else:
                    return 60
        except:
            return 60
    
    def _parse_memory(self, mem_str: str) -> int:
        """Parse memory string to MB"""
        if not mem_str or mem_str in ['0', 'N/A']:
            return 0
        
        try:
            if 'T' in mem_str:
                return int(float(mem_str.replace('T', '')) * 1024 * 1024)
            elif 'G' in mem_str:
                return int(float(mem_str.replace('G', '')) * 1024)
            elif 'M' in mem_str:
                return int(float(mem_str.replace('M', '')))
            elif 'K' in mem_str:
                return int(float(mem_str.replace('K', '')) / 1024)
            else:
                return int(mem_str)
        except:
            return 0
    
    def _parse_gres_string(self, gres: str) -> Tuple[int, Optional[str]]:
        """Parse a GRES string like gpu:A100:4 or gpu:4 into count and type"""
        if not gres or gres in ['(null)', 'N/A', 'n/a']:
            return 0, None
        
        # Take first entry if multiple comma-separated
        entry = gres.split(',')[0].strip()
        if not entry.startswith('gpu'):
            return 0, None
        
        # Patterns: gpu:<type>:<n> or gpu:<n>
        entry = entry.split('(')[0].strip()
        parts = entry.split(':')
        gpu_type = None
        count = 0
        
        if len(parts) == 3:
            _, gpu_type, count_str = parts
            try:
                count = int(count_str)
            except:
                count = 0
        elif len(parts) == 2:
            _, tail = parts
            if tail.isdigit():
                count = int(tail)
            else:
                gpu_type = tail
        elif len(parts) == 1:
            # Just "gpu"
            count = 0
        
        return count, gpu_type

    @staticmethod
    def _infer_gpu_type_from_features(features: Set[str]) -> Optional[str]:
        """Guess GPU type from node features when GRES lacks it."""
        if not features:
            return None
        known_prefixes = (
            'h100',
            'a100',
            'v100',
            'l40s',
            'l40',
            'a40',
            'a30',
            't4',
            'rtx6000',
            'rtxa6000',
            'rtx8000',
        )
        for feat in features:
            lower = feat.lower()
            for prefix in known_prefixes:
                if lower.startswith(prefix):
                    return prefix.upper()
        for feat in features:
            lower = feat.lower()
            if re.match(r'[a-z]{1,4}\d{2,3}', lower):
                return lower.upper()
        return None
    
    def _parse_alloc_gpus(self, tres: str) -> int:
        """Parse allocated GPU count from an AllocTRES string"""
        if not tres:
            return 0
        # Look for gres/gpu or gres/gpu:<type>
        matches = re.findall(r'gres/gpu(?::[A-Za-z0-9_]+)?=(\d+)', tres, flags=re.IGNORECASE)
        if matches:
            try:
                return max(int(m) for m in matches)
            except:
                return 0
        return 0
    
    def query_nodes(self) -> None:
        """Query node information using scontrol show node"""
        try:
            cmd = ['scontrol', 'show', 'node', '-o']
            output = self._run_command(cmd)
            
            self.nodes = []
            for line in output.strip().split('\n'):
                if not line:
                    continue
                
                fields = {}
                for token in line.split():
                    if '=' in token:
                        key, val = token.split('=', 1)
                        fields[key] = val
                
                node_name = fields.get('NodeName', 'unknown')
                state = fields.get('State', 'unknown')
                
                cpus_total = 0
                try:
                    cpus_total = int(fields.get('CPUEfctv') or fields.get('CPUTot') or 0)
                except:
                    cpus_total = 0
                
                try:
                    cpus_alloc = int(fields.get('CPUAlloc') or 0)
                except:
                    cpus_alloc = 0
                
                memory_total = self._parse_memory(fields.get('RealMemory', '0'))
                memory_alloc = self._parse_memory(fields.get('AllocMem', '0'))
                if memory_alloc == 0 and cpus_total > 0:
                    memory_alloc = int(memory_total * (cpus_alloc / max(cpus_total, 1)))
                
                partitions_raw = fields.get('Partitions', '')
                partitions = [p for p in partitions_raw.split(',') if p]
                partition = partitions[0] if partitions else ''
                
                gpus_total, gpu_type = self._parse_gres_string(fields.get('Gres', ''))
                gpus_alloc = self._parse_alloc_gpus(fields.get('AllocTRES', ''))
                gpus_alloc = min(gpus_alloc, gpus_total) if gpus_total else gpus_alloc
                features_raw = fields.get('AvailableFeatures') or fields.get('ActiveFeatures') or ''
                features = {f.strip().lower() for f in re.split(r'[,\s]+', features_raw) if f.strip()}
                if not gpu_type:
                    inferred_gpu = self._infer_gpu_type_from_features(features)
                    if inferred_gpu:
                        gpu_type = inferred_gpu
                
                node = NodeResources(
                    name=node_name,
                    state=state,
                    cpus=cpus_total,
                    cpus_alloc=cpus_alloc,
                    memory=memory_total,
                    memory_alloc=memory_alloc,
                    gpus=gpus_total,
                    gpus_alloc=gpus_alloc,
                    partition=partition,
                    partitions=partitions,
                    gpu_type=gpu_type,
                    features=features
                )
                
                self.nodes.append(node)
        
        except SlurmQueryError as e:
            self.error_message = f"Error querying nodes: {e}"
    
    def _get_node_gpu_alloc(self, node_name: str, total_gpus: int) -> int:
        """Get allocated GPU count for a node"""
        try:
            cmd = ['scontrol', 'show', 'node', node_name]
            output = self._run_command(cmd)
            
            alloc_match = re.search(r'AllocTRES=.*gres/gpu=(\d+)', output)
            if alloc_match:
                return int(alloc_match.group(1))
            
            if 'alloc' in output.lower() or 'mix' in output.lower():
                return total_gpus if 'alloc' in output.lower() else total_gpus // 2
            
            return 0
        except:
            return 0
    
    def query_running_jobs(self) -> None:
        """Query running jobs and their end times"""
        try:
            cmd = ['squeue', '-h', '-t', 'RUNNING', '-o', '%i|%u|%C|%b|%m|%N|%e|%P']
            output = self._run_command(cmd)
            
            self.running_jobs = []
            for line in output.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) < 8:
                    continue
                
                job_id = parts[0].strip()
                user = parts[1].strip()
                cpus = int(parts[2].strip()) if parts[2].strip() else 1
                
                gpu_str = parts[3].strip()
                gpus, gpu_type = self._parse_gres_string(gpu_str)
                
                memory = self._parse_memory(parts[4].strip())
                node_list = parts[5].strip()
                nodes = self._expand_node_list(node_list)
                end_time = self._parse_slurm_time(parts[6].strip())
                partition = parts[7].strip()
                
                job = RunningJob(
                    job_id=job_id,
                    user=user,
                    cpus=cpus,
                    gpus=gpus,
                    gpu_type=gpu_type,
                    memory=memory,
                    nodes=nodes,
                    end_time=end_time,
                    partition=partition
                )
                
                self.running_jobs.append(job)
        
        except SlurmQueryError as e:
            self.error_message = f"Error querying running jobs: {e}"
    
    def _expand_node_list(self, node_list: str) -> List[str]:
        """Expand SLURM node list notation"""
        if not node_list or node_list in ['(null)', 'N/A']:
            return []
        
        nodes = []
        if '[' in node_list:
            prefix = node_list.split('[')[0]
            range_part = node_list.split('[')[1].rstrip(']')
            
            for part in range_part.split(','):
                if '-' in part:
                    start, end = part.split('-')
                    for i in range(int(start), int(end) + 1):
                        nodes.append(f"{prefix}{i:0{len(start)}d}")
                else:
                    nodes.append(f"{prefix}{part}")
        else:
            nodes.append(node_list)
        
        return nodes
    
    def query_pending_jobs(self) -> None:
        """Query pending jobs with priorities and start times"""
        try:
            cmd = ['squeue', '-h', '-t', 'PENDING', '-o', '%i|%u|%Q|%C|%b|%m|%l|%P|%T|%r']
            output = self._run_command(cmd)
            
            pending_job_data = {}
            for line in output.strip().split('\n'):
                if not line:
                    continue
                
                parts = line.split('|')
                if len(parts) < 10:
                    continue
                
                job_id = parts[0].strip()
                user = parts[1].strip()
                priority = int(parts[2].strip()) if parts[2].strip() and parts[2].strip().isdigit() else 0
                cpus = int(parts[3].strip()) if parts[3].strip() else 1
                
                gpu_str = parts[4].strip()
                gpus, gpu_type = self._parse_req_gpus(gpu_str)
                
                memory = self._parse_memory(parts[5].strip())
                time_limit = self._parse_time_limit(parts[6].strip())
                partition = parts[7].strip()
                state = parts[8].strip()
                reason = parts[9].strip()
                
                pending_job_data[job_id] = {
                    'user': user,
                    'priority': priority,
                    'cpus': cpus,
                    'gpus': gpus,
                    'gpu_type': gpu_type,
                    'memory': memory,
                    'time_limit_minutes': time_limit,
                    'partition': partition,
                    'state': state,
                    'reason': reason
                }
            
            # Query start times
            try:
                cmd = ['squeue', '-h', '-t', 'PENDING', '--start', '-o', '%i|%S']
                output = self._run_command(cmd)
                
                for line in output.strip().split('\n'):
                    if not line:
                        continue
                    
                    parts = line.split('|')
                    if len(parts) < 2:
                        continue
                    
                    job_id = parts[0].strip()
                    start_time_str = parts[1].strip()
                    start_time = self._parse_slurm_time(start_time_str)
                    
                    if job_id in pending_job_data:
                        pending_job_data[job_id]['start_time'] = start_time
                        pending_job_data[job_id]['start_time_available'] = start_time is not None
            except:
                pass
            
            self.pending_jobs = []
            for job_id, data in pending_job_data.items():
                job = PendingJob(
                    job_id=job_id,
                    user=data['user'],
                    priority=data['priority'],
                    cpus=data['cpus'],
                    gpus=data['gpus'],
                    gpu_type=data['gpu_type'],
                    memory=data['memory'],
                    time_limit_minutes=data['time_limit_minutes'],
                    partition=data['partition'],
                    state=data['state'],
                    reason=data['reason'],
                    start_time=data.get('start_time'),
                    start_time_available=data.get('start_time_available', False)
                )
                self.pending_jobs.append(job)
            
            self.pending_jobs.sort(key=lambda x: x.priority, reverse=True)
        
        except SlurmQueryError as e:
            self.error_message = f"Error querying pending jobs: {e}"
    
    def query_user_priority(self) -> None:
        """Query user's fairshare and priority (using true FairShare column)"""
        try:
            cmd = ['sshare', '-a', '-P']
            output = self._run_command(cmd)
            
            lines = output.strip().split('\n')
            if len(lines) < 2:
                return
            
            for line in lines[1:]:
                parts = line.split('|')
                if len(parts) < 7:
                    continue
                
                account = parts[0].strip()
                user = parts[1].strip()
                norm_shares = parts[3].strip()
                raw_usage = parts[4].strip()
                eff_usage = parts[5].strip()
                fairshare_val = parts[6].strip()
                
                if user == self.current_user:
                    self.user_priority = UserPriority(
                        fairshare=float(fairshare_val) if fairshare_val else 0.5,
                        account=account,
                        norm_shares=float(norm_shares) if norm_shares else 1.0,
                        raw_usage=int(raw_usage) if raw_usage else 0,
                        effective_usage=float(eff_usage) if eff_usage else 0.0
                    )
                    break
        
        except SlurmQueryError as e:
            if not self.user_priority:
                self.user_priority = UserPriority(
                    fairshare=0.5,
                    account="default",
                    norm_shares=1.0,
                    raw_usage=0,
                    effective_usage=0.5
                )
    
    def query_user_partitions(self) -> None:
        """Query partitions the user can access"""
        try:
            cmd = ['sacctmgr', 'show', 'assoc', 'user', self.current_user, '-P', '-n']
            output = self._run_command(cmd)
            parts = set()
            for line in output.strip().split('\n'):
                if not line:
                    continue
                cols = line.split('|')
                if len(cols) < 4:
                    continue
                partition = cols[3].strip()
                # Empty partition means ALL partitions on many clusters
                if partition == '' or partition.lower() == 'all':
                    self.allowed_partitions = None
                    return
                parts.add(partition)
            self.allowed_partitions = parts if parts else None
        except Exception:
            self.allowed_partitions = None
    
    def query_partitions(self) -> None:
        """Query partition settings including MaxTime"""
        try:
            cmd = ['scontrol', 'show', 'partition', '-o']
            output = self._run_command(cmd)
            limits: Dict[str, int] = {}
            priorities: Dict[str, float] = {}
            infos: Dict[str, PartitionInfo] = {}

            def _parse_int(val: Optional[str]) -> Optional[int]:
                if val is None or val == '':
                    return None
                try:
                    return int(val)
                except Exception:
                    return None

            for line in output.strip().split('\n'):
                if not line:
                    continue
                fields = {}
                for token in line.split():
                    if '=' in token:
                        key, val = token.split('=', 1)
                        fields[key] = val
                name = fields.get('PartitionName')
                max_time_str = fields.get('MaxTime')
                priority_factor = fields.get('PriorityJobFactor')
                max_nodes = _parse_int(fields.get('MaxNodes'))
                total_nodes = _parse_int(fields.get('TotalNodes'))
                max_jobs = _parse_int(fields.get('MaxJobs'))
                if name and max_time_str:
                    max_minutes = self._parse_time_limit(max_time_str)
                    limits[name] = max_minutes
                    pf = float(priority_factor) if priority_factor else 0.0
                    priorities[name] = pf
                    infos[name] = PartitionInfo(
                        max_time_minutes=max_minutes,
                        priority_factor=pf,
                        max_nodes=max_nodes,
                        max_jobs=max_jobs,
                        total_nodes=total_nodes
                    )
            self.partition_limits = limits
            self.partition_priority = priorities
            self.partition_info = infos
        except Exception:
            self.partition_limits = {}
            self.partition_priority = {}
            self.partition_info = {}
    
    def estimate_my_job_priority(self, gpu_count: int, cpus: int) -> int:
        """Estimate priority for a hypothetical job from this user"""
        if not self.user_priority:
            return 0
        
        fairshare_raw = max(0.0, min(1.0, self.user_priority.fairshare))
        fairshare_priority = self.priority_weights.fairshare * (1.0 - fairshare_raw)
        age_priority = 0
        job_size_factor = min(1.0, 1.0 / (1 + gpu_count * 0.1))
        job_size_priority = self.priority_weights.job_size * job_size_factor
        
        total_priority = fairshare_priority + age_priority + job_size_priority
        return int(total_priority)
    
    def calculate_max_time_for_gpus(self, gpu_count: int) -> Dict[str, any]:
        """Calculate maximum time limit for immediate scheduling"""
        candidate_partitions: Set[str] = set()
        if self.filter_partition:
            candidate_partitions.add(self.filter_partition)
        else:
            for n in self.nodes:
                for p in n.partitions:
                    if p:
                        candidate_partitions.add(p)
        
        if self.allowed_partitions is not None:
            candidate_partitions = {p for p in candidate_partitions if p in self.allowed_partitions}
            if self.filter_partition and self.filter_partition not in candidate_partitions:
                return {
                    'status': 'unavailable',
                    'max_time_minutes': 0,
                    'max_cpus': 0,
                    'max_memory': 0,
                    'reason': f'User not allowed on partition {self.filter_partition}',
                    'has_asterisk': False,
                    'partition': self.filter_partition
                }
        
        if not candidate_partitions:
            return {
                'status': 'unavailable',
                'max_time_minutes': 0,
                'max_cpus': 0,
                'max_memory': 0,
                'reason': 'No permitted partitions found',
                'has_asterisk': False,
                'partition': self.filter_partition or '—',
                'all_partitions': [],
                'all_gpu_types': []
            }
        
        sorted_parts = sorted(
            candidate_partitions,
            key=lambda p: self.partition_priority.get(p, 0.0),
            reverse=True
        )
        fairshare_est = self.user_priority.fairshare if self.user_priority else None
        
        all_partitions: Set[str] = set()
        all_gpu_types: Set[str] = set()
        best_result = None
        best_free_gpus = -1
        best_priority_factor = -1.0
        for part in sorted_parts:
            usable_nodes: List[NodeResources] = []
            for n in self.nodes:
                if not n.is_usable or n.gpus_free < gpu_count:
                    continue
                if part not in n.partitions:
                    continue
                if self.filter_gpu_type:
                    if not n.gpu_type or n.gpu_type.lower() != self.filter_gpu_type:
                        continue
                usable_nodes.append(n)
            
            if not usable_nodes:
                result = {
                    'status': 'unavailable',
                    'max_time_minutes': 0,
                    'max_cpus': 0,
                    'max_memory': 0,
                    'reason': f'No nodes with {gpu_count} free GPUs',
                    'has_asterisk': False,
                    'partition': part,
                    'gpu_type': self.filter_gpu_type or "unknown"
                }
                if best_result is None:
                    best_result = result
                continue
            
            total_cpus = sum(n.cpus_free for n in usable_nodes)
            total_memory = sum(n.memory_free for n in usable_nodes)
            all_partitions.add(part)
            for n in usable_nodes:
                if n.gpu_type:
                    all_gpu_types.add(n.gpu_type)
            priority_factor = self.partition_priority.get(part, 0.0)
            part_info = self.partition_info.get(part)
            
            estimated_cpus = min(gpu_count * 8, total_cpus)
            my_priority = self.estimate_my_job_priority(gpu_count, estimated_cpus)
            
            def apply_preferences(res: Dict[str, any]) -> Dict[str, any]:
                res_reason = (res.get('reason') or '').rstrip('; ')
                if fairshare_est is not None:
                    suffix = f"; fairshare≈{fairshare_est:.3f}"
                    res_reason = (res_reason + suffix).strip('; ')
                    res['fairshare'] = fairshare_est
                res['reason'] = res_reason or res.get('reason') or ''
                if res.get('status') in ['unavailable', 'blocked'] or res.get('max_time_minutes', 0) <= 0:
                    res['max_cpus'] = 0
                    res['max_memory'] = 0
                    return res

                min_cpus = max(gpu_count, 1)
                capped_cpus = min(total_cpus, res.get('max_cpus', 0))
                if capped_cpus < min_cpus and total_cpus > 0:
                    res['reason'] = (res['reason'] + "; cpus limited").strip('; ')
                if total_cpus > 0:
                    res['max_cpus'] = min(total_cpus, max(min_cpus, capped_cpus))
                else:
                    res['max_cpus'] = 0

                mem_floor = gpu_count * 32 * 1024
                mem_cap = min(total_memory, res.get('max_memory', 0))
                mem_target = max(mem_floor, mem_cap)
                if mem_target > total_memory and total_memory > 0:
                    mem_target = total_memory
                    res['reason'] = (res['reason'] + "; memory limited").strip('; ')
                elif total_memory <= 0:
                    mem_target = 0
                    res['reason'] = (res['reason'] + "; memory unavailable").strip('; ')
                elif mem_cap < mem_floor and mem_target == mem_floor:
                    res['reason'] = (res['reason'] + "; raised memory for 32GB/GPU").strip('; ')
                res['max_memory'] = mem_target
                return res
            
            filtered_pending: List[PendingJob] = []
            for j in self.pending_jobs:
                if j.gpus <= 0:
                    continue
                if j.partition and j.partition != part:
                    continue
                if self.filter_gpu_type and j.gpu_type and j.gpu_type.lower() != self.filter_gpu_type:
                    continue
                filtered_pending.append(j)
            
            higher_priority_jobs = [j for j in filtered_pending if j.priority > my_priority]
        
            gpu_type_for_partition = self.filter_gpu_type
            if not gpu_type_for_partition:
                for node in usable_nodes:
                    if node.gpu_type:
                        gpu_type_for_partition = node.gpu_type
                        break
            
            partition_cap = self.partition_limits.get(part)
            
            def apply_partition_cap(result: Dict[str, any]) -> Dict[str, any]:
                if partition_cap is not None and partition_cap >= 0:
                    if result['max_time_minutes'] > partition_cap:
                        result['max_time_minutes'] = partition_cap
                        # annotate reason to show cap
                        hours_cap = partition_cap / 60
                        result['reason'] += f"; capped by partition ({hours_cap:.1f}h)"
                    result['partition_max_minutes'] = partition_cap
                return result
            
            running_in_part = [r for r in self.running_jobs if r.partition == part]
            part_gate = None
            gate_reason = None
            if part_info:
                if part_info.max_jobs and part_info.max_jobs > 0 and len(running_in_part) >= part_info.max_jobs:
                    earliest_end = min((r.end_time for r in running_in_part if r.end_time), default=None)
                    if earliest_end:
                        part_gate = earliest_end
                        gate_reason = f"partition max_jobs {len(running_in_part)}/{part_info.max_jobs}"
                    else:
                        candidate_result = {
                            'status': 'blocked',
                            'max_time_minutes': 0,
                            'max_cpus': 0,
                            'max_memory': 0,
                            'reason': f"Partition max_jobs reached ({len(running_in_part)}/{part_info.max_jobs})",
                            'has_asterisk': True,
                            'partition': part,
                            'gpu_type': gpu_type_for_partition
                        }
                        candidate_result = apply_preferences(apply_partition_cap(candidate_result))
                        free_gpus_here = sum(n.gpus_free for n in usable_nodes if n.is_usable)
                        should_update = (
                            best_result is None
                            or (candidate_result.get('status') != 'unavailable' and best_result.get('status') == 'unavailable')
                            or priority_factor > best_priority_factor
                            or (priority_factor == best_priority_factor and free_gpus_here > best_free_gpus)
                        )
                        if should_update:
                            best_priority_factor = priority_factor
                            best_free_gpus = free_gpus_here
                            best_result = candidate_result
                        continue
                if part_info.max_nodes and part_info.max_nodes > 0:
                    used_nodes = {n for r in running_in_part for n in r.nodes}
                    if len(used_nodes) >= part_info.max_nodes:
                        earliest_end = min((r.end_time for r in running_in_part if r.end_time), default=None)
                        if earliest_end:
                            part_gate = max(part_gate, earliest_end) if part_gate else earliest_end
                            gate_reason = (gate_reason + "; " if gate_reason else "") + f"partition max_nodes {len(used_nodes)}/{part_info.max_nodes}"
                        else:
                            candidate_result = {
                                'status': 'blocked',
                                'max_time_minutes': 0,
                                'max_cpus': 0,
                                'max_memory': 0,
                                'reason': f"Partition max_nodes reached ({len(used_nodes)}/{part_info.max_nodes})",
                                'has_asterisk': True,
                                'partition': part,
                                'gpu_type': gpu_type_for_partition
                            }
                            candidate_result = apply_preferences(apply_partition_cap(candidate_result))
                            free_gpus_here = sum(n.gpus_free for n in usable_nodes if n.is_usable)
                            should_update = (
                                best_result is None
                                or (candidate_result.get('status') != 'unavailable' and best_result.get('status') == 'unavailable')
                                or priority_factor > best_priority_factor
                                or (priority_factor == best_priority_factor and free_gpus_here > best_free_gpus)
                            )
                            if should_update:
                                best_priority_factor = priority_factor
                                best_free_gpus = free_gpus_here
                                best_result = candidate_result
                            continue

            if not higher_priority_jobs and not part_gate:
                candidate_result = {
                    'status': 'available',
                    'max_time_minutes': 10080,
                    'max_cpus': min(total_cpus, 128),
                    'max_memory': min(total_memory, 512000),
                    'reason': 'No higher-priority jobs blocking',
                    'has_asterisk': False,
                    'partition': part,
                    'gpu_type': gpu_type_for_partition
                }
                candidate_result = apply_preferences(apply_partition_cap(candidate_result))
            else:
                competing_jobs = []
                for job in higher_priority_jobs:
                    for node in usable_nodes:
                        if job.can_run_on_node(node) or job.gpus <= gpu_count:
                            competing_jobs.append(job)
                            break
                
                if not competing_jobs:
                    if part_gate:
                        gate_time = part_gate
                        time_until_start = (gate_time - self.now).total_seconds() / 60
                        max_time = max(0, int(time_until_start) - 5)
                        if max_time > 0:
                            candidate_result = {
                                'status': 'backfill',
                                'max_time_minutes': max_time,
                                'max_cpus': min(total_cpus, 128),
                                'max_memory': min(total_memory, 512000),
                                'reason': gate_reason or 'Partition limit window',
                                'has_asterisk': True,
                                'earliest_start': gate_time,
                                'partition': part,
                                'gpu_type': gpu_type_for_partition
                            }
                        else:
                            candidate_result = {
                                'status': 'blocked',
                                'max_time_minutes': 0,
                                'max_cpus': 0,
                                'max_memory': 0,
                                'reason': gate_reason or 'Partition limit reached',
                                'has_asterisk': True,
                                'partition': part,
                                'gpu_type': gpu_type_for_partition
                            }
                    else:
                        candidate_result = {
                            'status': 'available',
                            'max_time_minutes': 10080,
                            'max_cpus': min(total_cpus, 128),
                            'max_memory': min(total_memory, 512000),
                            'reason': "Higher-priority jobs don't compete",
                            'has_asterisk': False,
                            'partition': part,
                            'gpu_type': gpu_type_for_partition
                        }
                    candidate_result = apply_preferences(apply_partition_cap(candidate_result))
                else:
                    earliest_start = None
                    has_asterisk = False
                    
                    for job in competing_jobs:
                        if job.start_time and job.start_time_available:
                            if earliest_start is None or job.start_time < earliest_start:
                                earliest_start = job.start_time
                        else:
                            has_asterisk = True
                    
                    gate_time = part_gate or earliest_start
                    gate_has_asterisk = has_asterisk or (part_gate is not None and gate_reason is None and earliest_start is None)
                    gate_reason_text = gate_reason or (f'Backfill before job {competing_jobs[0].job_id}' if competing_jobs else 'Backfill window')
                    if gate_time:
                        time_until_start = (gate_time - self.now).total_seconds() / 60
                        max_time = max(0, int(time_until_start) - 5)
                        
                        if max_time > 0:
                            candidate_result = {
                                'status': 'backfill',
                                'max_time_minutes': max_time,
                                'max_cpus': min(total_cpus, 128),
                                'max_memory': min(total_memory, 512000),
                                'reason': gate_reason_text,
                                'has_asterisk': gate_has_asterisk,
                                'earliest_start': gate_time,
                                'partition': part,
                                'gpu_type': gpu_type_for_partition
                            }
                            candidate_result = apply_preferences(apply_partition_cap(candidate_result))
                        else:
                            candidate_result = {
                                'status': 'blocked',
                                'max_time_minutes': 0,
                                'max_cpus': 0,
                                'max_memory': 0,
                                'reason': f'Blocked by job {competing_jobs[0].job_id}',
                                'has_asterisk': has_asterisk,
                                'partition': part,
                                'gpu_type': gpu_type_for_partition
                            }
                            candidate_result = apply_preferences(apply_partition_cap(candidate_result))
                    else:
                        candidate_result = {
                            'status': 'uncertain',
                            'max_time_minutes': 120,
                            'max_cpus': min(total_cpus, 128),
                            'max_memory': min(total_memory, 512000),
                            'reason': 'Start times unavailable, uncertain*',
                            'has_asterisk': True,
                            'partition': part,
                            'gpu_type': gpu_type_for_partition
                        }
                        candidate_result = apply_preferences(apply_partition_cap(candidate_result))
            
            free_gpus_here = sum(n.gpus_free for n in usable_nodes if n.is_usable)
            should_update = (
                best_result is None
                or (candidate_result.get('status') != 'unavailable' and best_result.get('status') == 'unavailable')
                or priority_factor > best_priority_factor
                or (priority_factor == best_priority_factor and free_gpus_here > best_free_gpus)
            )
            if should_update:
                best_priority_factor = priority_factor
                best_free_gpus = free_gpus_here
                best_result = candidate_result
        
        final_result = best_result if best_result else {
            'status': 'unavailable',
            'max_time_minutes': 0,
            'max_cpus': 0,
            'max_memory': 0,
            'reason': 'No suitable partition found',
            'has_asterisk': False,
            'partition': self.filter_partition or '—',
            'gpu_type': self.filter_gpu_type or "—",
        }
        final_result['all_partitions'] = sorted(all_partitions)
        final_result['all_gpu_types'] = sorted(all_gpu_types)
        return final_result
    
    def estimate_all_configurations(self, show_progress: bool = True) -> Dict[int, Dict]:
        """Estimate resources for all GPU configurations"""
        results = {}
        
        gpu_counts = []
        gpu = 1
        while gpu <= self.max_gpus:
            gpu_counts.append(gpu)
            gpu *= 2
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.console,
                transient=True
            ) as progress:
                task = progress.add_task("[cyan]Analyzing...", total=len(gpu_counts))
                
                for gpu_count in gpu_counts:
                    results[gpu_count] = self.calculate_max_time_for_gpus(gpu_count)
                    progress.advance(task)
        else:
            for gpu_count in gpu_counts:
                results[gpu_count] = self.calculate_max_time_for_gpus(gpu_count)
        
        return results
    
    def refresh_data(self) -> None:
        """Refresh all data from SLURM"""
        self.now = datetime.now()
        self.error_message = None
        
        self.query_nodes()
        self.query_running_jobs()
        self.query_pending_jobs()
        self.query_user_priority()
        self.query_user_partitions()
        self.query_partitions()
        
        self.last_update = datetime.now()
        self.update_count += 1
    
    def generate_display(self, results: Dict[int, Dict]) -> Panel:
        """Generate the display panel with all results"""
        # Create main table
        title = Text()
        title.append("SLURM Resource Estimator\n", style="bold white")
        title.append("Maximum job parameters that can run ", style="bold")
        title.append("RIGHT NOW", style="bold green")
        
        if self.user_priority:
            subtitle = f"User: {self.current_user} | "
            subtitle += f"Fairshare: {self.user_priority.fairshare:.3f} | "
            subtitle += f"Account: {self.user_priority.account}"
        else:
            subtitle = f"User: {self.current_user}"
        if self.filter_partition:
            subtitle += f" | Partition: {self.filter_partition}"
        if self.filter_gpu_type:
            subtitle += f" | GPU: {self.filter_gpu_type}"
        
        table = Table(
            title=title,
            caption=subtitle,
            box=box.ROUNDED,
            show_header=True,
            show_lines=True,
            header_style="bold cyan",
            title_style="bold",
            caption_style="italic dim",
            expand=True
        )
        
        table.add_column("GPUs", justify="center", style="bold yellow", width=6)
        table.add_column("Partitions", justify="center", width=18)
        table.add_column("GPU Types", justify="center", width=18)
        table.add_column("Status", justify="center", width=14)
        table.add_column("Max Time", justify="right", style="magenta", width=14)
        table.add_column("Max CPUs", justify="right", style="cyan", width=10)
        table.add_column("Max Memory", justify="right", style="blue", width=12)
        table.add_column("Notes", justify="left", style="dim")
        
        for gpu_count in sorted(results.keys()):
            result = results[gpu_count]
            
            status = result['status']
            if status == 'available':
                status_text = "[bold green]✓ Available[/]"
            elif status == 'backfill':
                status_text = "[bold yellow]⚡ Backfill[/]"
            elif status == 'uncertain':
                status_text = "[bold orange]? Uncertain[/]"
            else:
                status_text = "[bold red]✗ Blocked[/]"
            
            max_time_min = result['max_time_minutes']
            if max_time_min > 0:
                hours = max_time_min / 60
                if hours >= 24:
                    days = hours / 24
                    time_text = f"{days:.1f} days"
                else:
                    time_text = f"{hours:.1f} hours"
                
                if result.get('has_asterisk'):
                    time_text += "*"
            else:
                time_text = "—"
            
            cpus_text = str(result['max_cpus']) if result['max_cpus'] > 0 else "—"
            
            memory_gb = result['max_memory'] / 1024
            memory_text = f"{memory_gb:.0f} GB" if result['max_memory'] > 0 else "—"
            
            notes = result['reason']
            part_list = result.get('all_partitions') or []
            part_text = ", ".join(part_list) if part_list else (result.get('partition') or "—")
            gpu_list = result.get('all_gpu_types') or []
            gpu_type_text = ", ".join(gpu_list) if gpu_list else (result.get('gpu_type') or self.filter_gpu_type or "—")
            
            table.add_row(
                str(gpu_count),
                part_text,
                gpu_type_text,
                status_text,
                time_text,
                cpus_text,
                memory_text,
                notes
            )
        
        # Create layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=7),
            Layout(name="body"),
            Layout(name="footer", size=7)
        )
        
        # Header with summary
        summary_lines = []
        filtered_nodes = [
            n for n in self.nodes
            if (not self.filter_partition or self.filter_partition in n.partitions) and
               (not self.filter_gpu_type or (n.gpu_type and n.gpu_type.lower() == self.filter_gpu_type)) and
               (self.allowed_partitions is None or any(p in self.allowed_partitions for p in n.partitions))
        ]
        summary_lines.append(f"Nodes: {len(filtered_nodes)} total, {sum(1 for n in filtered_nodes if n.is_usable)} usable")
        total_free_gpus = sum(n.gpus_free for n in filtered_nodes if n.is_usable)
        summary_lines.append(f"Free GPUs: {total_free_gpus}")
        summary_lines.append(f"Jobs: {len(self.running_jobs)} running, {len(self.pending_jobs)} pending")
        if self.filter_partition or self.filter_gpu_type:
            filters = []
            if self.filter_partition:
                filters.append(f"partition={self.filter_partition}")
            if self.filter_gpu_type:
                filters.append(f"gpu={self.filter_gpu_type}")
            summary_lines.append("Filters: " + ", ".join(filters))
        
        if self.last_update:
            time_str = self.last_update.strftime("%H:%M:%S")
            summary_lines.append(f"Last update: {time_str} (refresh #{self.update_count})")
        
        if self.error_message:
            summary_lines.append(f"[yellow]⚠ {self.error_message}[/]")
        
        summary_panel = Panel(
            "\n".join(summary_lines),
            title="[bold]Cluster Summary[/]",
            border_style="blue",
            padding=(0, 2)
        )
        
        # Footer with legend
        legend_text = (
            "[bold green]✓[/] Available  "
            "[bold yellow]⚡[/] Backfill  "
            "[bold orange]?[/] Uncertain  "
            "[bold red]✗[/] Blocked\n"
            "[dim]* = Incomplete data  |  Press Ctrl+C to exit[/]"
        )
        
        legend_panel = Panel(
            legend_text,
            border_style="dim",
            padding=(0, 2)
        )
        
        layout["header"].update(summary_panel)
        layout["body"].update(table)
        layout["footer"].update(legend_panel)
        
        return Panel(layout, border_style="cyan", padding=(1, 2))


class SlurmBlame(SlurmBase):
    def __init__(self, estimator: SlurmEstimator, blockers: int = 5):
        self.console = Console()
        self.blockers = blockers
        self.estimator = estimator

    def snapshot_queue(self) -> List[JobBrief]:
        cmd = ['squeue', '-h', '-t', 'R,PD', '-o', '%i|%Q|%T|%S|%r|%u|%P']
        output = self._run_command(cmd)
        jobs: List[JobBrief] = []
        for line in output.strip().split('\n'):
            if not line:
                continue
            parts = line.split('|')
            if len(parts) < 7:
                continue
            job_id = parts[0].strip()
            priority = int(parts[1]) if parts[1].strip().isdigit() else 0
            state = parts[2].strip()
            start_time = self._parse_slurm_time(parts[3].strip())
            reason = parts[4].strip()
            user = parts[5].strip()
            partition = parts[6].strip()
            jobs.append(JobBrief(
                job_id=job_id,
                priority=priority,
                state=state,
                start_time=start_time,
                reason=reason,
                user=user,
                partition=partition
            ))
        jobs.sort(key=lambda j: j.priority, reverse=True)
        return jobs

    def predict_start(self, job_id: str) -> Optional[datetime]:
        try:
            cmd = ['squeue', '-h', '--start', '-j', job_id, '-o', '%i|%S']
            output = self._run_command(cmd)
            for line in output.strip().split('\n'):
                parts = line.split('|')
                if len(parts) < 2 or parts[0].strip() != job_id:
                    continue
                return self._parse_slurm_time(parts[1].strip())
        except SlurmQueryError:
            return None
        return None

    def fetch_job_detail(self, job_id: str) -> JobDetail:
        cmd = ['scontrol', 'show', 'job', '-o', job_id]
        output = self._run_command(cmd)
        line = output.strip().split('\n')[0] if output else ""
        fields: Dict[str, str] = {}
        for token in line.split():
            if '=' in token:
                key, val = token.split('=', 1)
                fields[key] = val
        state = fields.get('JobState', 'unknown')
        reason = fields.get('Reason', '')
        priority = int(fields.get('Priority', '0') or 0)
        partition = fields.get('Partition', '')
        user_field = fields.get('UserId', '')
        user = user_field.split('(')[0] if '(' in user_field else user_field
        cpus = int(fields.get('NumCPUs') or fields.get('ReqCPUS') or 0)
        req_tres = fields.get('ReqTRES', '') or fields.get('TRES', '')
        memory = self._parse_req_mem(req_tres)
        if memory <= 0:
            if 'MinMemoryCPU' in fields and cpus > 0:
                memory = self._parse_memory(fields['MinMemoryCPU']) * cpus
            elif 'MinMemoryNode' in fields:
                memory = self._parse_memory(fields['MinMemoryNode'])
            elif 'ReqMem' in fields:
                memory = self._parse_memory(fields['ReqMem'])
        gpus, gpu_type = self._parse_req_gpus(req_tres)
        if gpus == 0:
            gres_count, gres_type = self._parse_gres_string(fields.get('Gres', ''))
            gpus = gres_count
            gpu_type = gpu_type or gres_type
        alloc_gpus = self._parse_alloc_gpus(fields.get('TRES', ''))
        if gpus == 0 and alloc_gpus > 0:
            gpus = alloc_gpus
        time_limit_minutes = self._parse_time_limit(fields.get('TimeLimit', ''))
        start_time = self._parse_slurm_time(fields.get('StartTime', ''))
        submit_time = self._parse_slurm_time(fields.get('SubmitTime', ''))
        return JobDetail(
            job_id=job_id,
            state=state,
            reason=reason,
            priority=priority,
            partition=partition,
            user=user,
            gpus=gpus,
            gpu_type=gpu_type,
            cpus=cpus,
            memory=memory,
            time_limit_minutes=time_limit_minutes,
            start_time=start_time,
            submit_time=submit_time
        )

    @staticmethod
    def _req_from_detail(detail: JobDetail) -> Tuple[int, int, int]:
        return detail.cpus, detail.gpus, detail.memory

    @staticmethod
    def _req_from_pending(job: PendingJob) -> Tuple[int, int, int]:
        return job.cpus, job.gpus, job.memory

    @staticmethod
    def _fits(free: Tuple[int, int, int], req: Tuple[int, int, int]) -> bool:
        return free[0] >= req[0] and free[1] >= req[1] and free[2] >= req[2]

    @staticmethod
    def _apply_delta(
        free: Tuple[int, int, int],
        event: ResourceEvent,
        node: NodeResources
    ) -> Tuple[int, int, int]:
        fc = max(0, min(node.cpus, free[0] + event.delta_cpus))
        fg = max(0, min(node.gpus, free[1] + event.delta_gpus))
        fm = max(0, min(node.memory, free[2] + event.delta_mem))
        return fc, fg, fm

    @staticmethod
    def _per_node_usage(job: RunningJob, node: NodeResources) -> Tuple[int, int, int]:
        parts = max(1, len(job.nodes)) if job.nodes else 1
        cpus = min(node.cpus, (job.cpus + parts - 1) // parts)
        gpus = min(node.gpus, (job.gpus + parts - 1) // parts)
        memory = min(node.memory, (job.memory + parts - 1) // parts)
        return cpus, gpus, memory

    def _node_supports(self, node: NodeResources, detail: JobDetail) -> bool:
        if not node.is_usable:
            return False
        if detail.partition and detail.partition not in node.partitions:
            return False
        if detail.gpu_type:
            if not node.gpu_type:
                return False
            if node.gpu_type.lower() != detail.gpu_type.lower():
                return False
        return (
            node.cpus >= detail.cpus
            and node.memory >= detail.memory
            and node.gpus >= detail.gpus
        )

    def _job_can_use_node(self, job: PendingJob, node: NodeResources) -> bool:
        if not node.is_usable:
            return False
        return job.can_run_on_node(node)

    def _build_node_states(self, detail: JobDetail) -> Dict[str, NodeSimState]:
        states: Dict[str, NodeSimState] = {}
        for node in self.estimator.nodes:
            if not self._node_supports(node, detail):
                continue
            state = NodeSimState(
                node=node,
                free_cpus=node.cpus_free,
                free_gpus=node.gpus_free,
                free_mem=node.memory_free,
            )
            for r in self.estimator.running_jobs:
                if node.name not in r.nodes:
                    continue
                state.running.append(r)
                if r.end_time:
                    cpus, gpus, mem = self._per_node_usage(r, node)
                    label = f"{r.job_id} ({r.user})"
                    state.events.append(ResourceEvent(
                        time=r.end_time,
                        delta_cpus=cpus,
                        delta_gpus=gpus,
                        delta_mem=mem,
                        kind='release',
                        label=label
                    ))
            states[node.name] = state
        return states

    def _build_partition_state(self, detail: JobDetail) -> Optional[PartitionSimState]:
        if not detail.partition:
            return None
        part_info = self.estimator.partition_info.get(detail.partition)
        nodes = [n for n in self.estimator.nodes if detail.partition in n.partitions]
        if not nodes:
            return None
        total_nodes = part_info.total_nodes or len(nodes) if part_info else len(nodes)
        max_nodes = part_info.max_nodes or total_nodes if part_info else total_nodes
        max_jobs = part_info.max_jobs if part_info and part_info.max_jobs else 10**6
        gpu_totals = sorted((n.gpus for n in nodes), reverse=True)
        max_gpus = sum(gpu_totals[:max_nodes]) if gpu_totals else 0

        running_in_part = [r for r in self.estimator.running_jobs if r.partition == detail.partition]
        used_nodes = {n for r in running_in_part for n in r.nodes}
        used_nodes_count = len(used_nodes)
        used_gpus = sum(r.gpus for r in running_in_part)

        free_jobs = max(0, max_jobs - len(running_in_part))
        free_nodes = max(0, max_nodes - used_nodes_count)
        free_gpus = max(0, max_gpus - used_gpus)

        events: List[PartitionEvent] = []
        for r in running_in_part:
            if not r.end_time:
                continue
            nodes_used = max(1, len(r.nodes)) if r.nodes else 1
            events.append(PartitionEvent(
                time=r.end_time,
                delta_jobs=1,
                delta_nodes=nodes_used,
                delta_gpus=r.gpus,
                kind='release',
                label=f"{r.job_id} ({r.user})"
            ))

        return PartitionSimState(
            partition=detail.partition,
            max_jobs=max_jobs,
            max_nodes=max_nodes,
            max_gpus=max_gpus,
            free_jobs=free_jobs,
            free_nodes=free_nodes,
            free_gpus=free_gpus,
            events=events
        )

    @staticmethod
    def _partition_fits(free: Tuple[int, int, int], req: Tuple[int, int, int]) -> bool:
        return free[0] >= req[0] and free[1] >= req[1] and free[2] >= req[2]

    @staticmethod
    def _partition_apply_delta(
        free: Tuple[int, int, int],
        event: PartitionEvent,
        state: PartitionSimState
    ) -> Tuple[int, int, int]:
        fj = max(0, min(state.max_jobs, free[0] + event.delta_jobs))
        fn = max(0, min(state.max_nodes, free[1] + event.delta_nodes))
        fg = max(0, min(state.max_gpus, free[2] + event.delta_gpus))
        return fj, fn, fg

    def _earliest_partition_start(
        self,
        state: PartitionSimState,
        req_gpus: int,
        req_nodes: int,
        now: datetime
    ) -> Tuple[Optional[datetime], List[PartitionEvent]]:
        req = (1, req_nodes, req_gpus)
        free = (state.free_jobs, state.free_nodes, state.free_gpus)
        if self._partition_fits(free, req):
            return now, []
        sentinel = now + timedelta(days=3650)
        events = sorted(
            state.events,
            key=lambda e: ((e.time or sentinel), 0 if e.kind == 'release' else 1)
        )
        trace: List[PartitionEvent] = []
        current = free
        for ev in events:
            if ev.time and ev.time <= now:
                trace.append(ev)
                continue
            current = self._partition_apply_delta(current, ev, state)
            trace.append(ev)
            if self._partition_fits(current, req):
                return ev.time or now, trace
        return None, trace

    def _earliest_start(
        self,
        state: NodeSimState,
        req: Tuple[int, int, int],
        now: datetime
    ) -> Tuple[Optional[datetime], List[ResourceEvent]]:
        free = (state.free_cpus, state.free_gpus, state.free_mem)
        if self._fits(free, req):
            return now, []
        sentinel = now + timedelta(days=3650)
        events = sorted(
            state.events,
            key=lambda e: ((e.time or sentinel), 0 if e.kind == 'release' else 1)
        )
        trace: List[ResourceEvent] = []
        current = free
        for ev in events:
            if ev.time and ev.time <= now:
                trace.append(ev)
                continue
            current = self._apply_delta(current, ev, state.node)
            trace.append(ev)
            if self._fits(current, req):
                return ev.time or now, trace
        return None, trace

    def _select_competing_jobs(
        self,
        detail: JobDetail,
        node_states: Dict[str, NodeSimState],
        ahead_job_ids: Optional[Set[str]]
    ) -> List[PendingJob]:
        competing: List[PendingJob] = []
        for j in self.estimator.pending_jobs:
            if j.job_id == detail.job_id:
                continue
            if ahead_job_ids is not None:
                if j.job_id not in ahead_job_ids:
                    continue
            else:
                if j.priority <= detail.priority:
                    continue
            if any(self._job_can_use_node(j, st.node) for st in node_states.values()):
                competing.append(j)
        competing.sort(key=lambda j: (-j.priority, j.job_id))
        return competing

    def _schedule_competing_jobs(
        self,
        detail: JobDetail,
        node_states: Dict[str, NodeSimState],
        partition_state: Optional[PartitionSimState],
        competing: List[PendingJob],
        now: datetime
    ) -> Dict[str, List[Tuple[str, int, Optional[datetime]]]]:
        scheduled_map: Dict[str, List[Tuple[str, int, Optional[datetime]]]] = {name: [] for name in node_states.keys()}

        for job in competing:
            req = self._req_from_pending(job)
            req_nodes = 1
            part_start = None
            if partition_state:
                part_start, _ = self._earliest_partition_start(partition_state, req[1], req_nodes, now)
                if part_start is None:
                    continue
            best_state = None
            best_time = None
            for st in node_states.values():
                if not self._job_can_use_node(job, st.node):
                    continue
                start, _ = self._earliest_start(st, req, now)
                if start is None:
                    continue
                if best_time is None or start < best_time:
                    best_time = start
                    best_state = st
            if not best_state or best_time is None:
                continue
            start = best_time
            if part_start:
                start = max(start, part_start)
            label = f"{job.job_id} (prio {job.priority})"
            if start <= now:
                best_state.free_cpus = max(0, best_state.free_cpus - req[0])
                best_state.free_gpus = max(0, best_state.free_gpus - req[1])
                best_state.free_mem = max(0, best_state.free_mem - req[2])
                best_state.events.append(ResourceEvent(
                    time=now,
                    delta_cpus=-req[0],
                    delta_gpus=-req[1],
                    delta_mem=-req[2],
                    kind='alloc',
                    label=label
                ))
            else:
                best_state.events.append(ResourceEvent(
                    time=start,
                    delta_cpus=-req[0],
                    delta_gpus=-req[1],
                    delta_mem=-req[2],
                    kind='alloc',
                    label=label
                ))
            best_state.scheduled.append((job.job_id, job.priority, start))
            if partition_state:
                alloc_event = PartitionEvent(
                    time=start,
                    delta_jobs=-1,
                    delta_nodes=-req_nodes,
                    delta_gpus=-req[1],
                    kind='alloc',
                    label=label
                )
                if start <= now:
                    fj, fn, fg = self._partition_apply_delta(
                        (partition_state.free_jobs, partition_state.free_nodes, partition_state.free_gpus),
                        alloc_event,
                        partition_state
                    )
                    partition_state.free_jobs = fj
                    partition_state.free_nodes = fn
                    partition_state.free_gpus = fg
                else:
                    partition_state.events.append(alloc_event)
            scheduled_map[best_state.node.name].append((job.job_id, job.priority, start))

        return scheduled_map

    def _compute_node_blockers(self, detail: JobDetail, ahead_job_ids: Optional[Set[str]] = None) -> List[Dict[str, any]]:
        base_states = self._build_node_states(detail)
        if not base_states:
            return []

        node_states = {
            name: NodeSimState(
                node=st.node,
                free_cpus=st.free_cpus,
                free_gpus=st.free_gpus,
                free_mem=st.free_mem,
                events=list(st.events),
                scheduled=[],
                running=list(st.running),
            )
            for name, st in base_states.items()
        }

        now = self.estimator.now or datetime.now()
        partition_state = self._build_partition_state(detail)
        competing = self._select_competing_jobs(detail, base_states, ahead_job_ids)
        schedule_map = self._schedule_competing_jobs(detail, node_states, partition_state, competing, now)

        req = self._req_from_detail(detail)
        blockers: List[Dict[str, any]] = []

        part_trace_for_detail: List[PartitionEvent] = []
        part_start_for_detail: Optional[datetime] = None
        if partition_state:
            part_start_for_detail, part_trace_for_detail = self._earliest_partition_start(
                partition_state,
                req[1],
                1,
                now,
            )

        sentinel = now + timedelta(days=3650)

        ahead_per_node: Dict[str, List[PendingJob]] = {}
        if competing:
            for name, base in base_states.items():
                node_jobs = [
                    j for j in competing
                    if self._job_can_use_node(j, base.node)
                ]
                if node_jobs:
                    ahead_per_node[name] = node_jobs

        for name, st in node_states.items():
            start, trace = self._earliest_start(st, req, now)
            reasons: List[str] = []

            if start is None and st.running:
                unknown = [r for r in st.running if not r.end_time]
                if unknown:
                    labels = ", ".join(r.job_id for r in unknown)
                    reasons.append(f"running jobs with unknown end: {labels}")

            for ev in trace:
                if ev.kind != "release":
                    continue
                when = _format_blocker_time(ev.time, now) if ev.time else "unknown"
                reasons.append(f"running {ev.label} ends ~{when}")

            if partition_state:
                if part_start_for_detail:
                    start = max(start, part_start_for_detail) if start else part_start_for_detail
                for ev in part_trace_for_detail:
                    when = _format_blocker_time(ev.time, now) if ev.time else "unknown"
                    if ev.kind == "release":
                        reasons.append(f"partition: running {ev.label} ends ~{when}")
                    else:
                        reasons.append(f"partition: queued {ev.label} starts ~{when}")
                if part_start_for_detail is None:
                    if partition_state.max_jobs == 0:
                        reasons.append("partition max_jobs reached")
                    elif partition_state.max_nodes == 0:
                        reasons.append("partition max_nodes reached")
                    elif partition_state.max_gpus <= 0:
                        reasons.append("partition GPU capacity exhausted")

            node_jobs = ahead_per_node.get(name, [])
            if node_jobs:
                scheduled = {sid: t for sid, _, t in schedule_map.get(name, st.scheduled or [])}

                def job_key(j: PendingJob) -> Tuple[datetime, int, str]:
                    sched_time = scheduled.get(j.job_id)
                    base_time = j.start_time or sentinel
                    return (sched_time or base_time, -j.priority, j.job_id)

                for j in sorted(node_jobs, key=job_key)[: self.blockers]:
                    sched_time = scheduled.get(j.job_id) or j.start_time
                    when = _format_blocker_time(sched_time, now) if sched_time else "unknown"
                    prefix = "queued"
                    if j.job_id in scheduled:
                        prefix = "queued (scheduled)"
                    reasons.append(f"{prefix} {j.job_id} (prio {j.priority}) ahead here ~{when}")

            if not reasons and start is None:
                reasons.append("insufficient capacity or missing end times")

            blockers.append({
                "node": st.node.name,
                "start": start,
                "reasons": reasons,
            })

        blockers.sort(key=lambda b: b["start"] or sentinel)
        return blockers[:3]

    def diagnose(self, job_id: str, queue: List[JobBrief]) -> Dict[str, any]:
        try:
            detail = self.fetch_job_detail(job_id)
        except SlurmQueryError as e:
            return {
                'job_id': job_id,
                'status': 'error',
                'reason': str(e),
                'blockers': [],
                'eta': None,
                'state': 'UNKNOWN',
                'user': 'unknown',
                'partition': '',
                'priority': 0,
                'resources': {},
            }
        
        queue_entry = next((q for q in queue if q.job_id == job_id), None)
        ahead_job_ids: Optional[Set[str]] = None
        if queue:
            ahead_job_ids = set()
            for q in queue:
                if q.job_id == job_id:
                    break
                if q.state.upper().startswith('PD'):
                    ahead_job_ids.add(q.job_id)
        state = (queue_entry.state if queue_entry else detail.state) or detail.state
        reason = (queue_entry.reason if queue_entry and queue_entry.reason else detail.reason) or ''
        priority = queue_entry.priority if queue_entry else detail.priority
        partition = detail.partition or (queue_entry.partition if queue_entry else '')
        eta = queue_entry.start_time or detail.start_time or self.predict_start(job_id)
        gpu_type = detail.gpu_type.lower() if detail.gpu_type else None
        
        status = 'unknown'
        upper_state = state.upper()
        if upper_state.startswith('R') or upper_state.startswith('CG'):
            status = 'running'
        else:
            rlow = reason.lower()
            if 'priority' in rlow or 'prio' in rlow:
                status = 'waiting-priority'
            elif any(key in rlow for key in ['resource', 'node', 'partition', 'license', 'tres', 'memory', 'cpu']):
                status = 'waiting-resources'
            elif 'depend' in rlow:
                status = 'blocked-dependency'
            elif any(key in rlow for key in ['hold', 'admin', 'error']):
                status = 'blocked-error'
            else:
                status = 'unknown'
        
        resources = {
            'gpus': detail.gpus,
            'gpu_type': detail.gpu_type,
            'cpus': detail.cpus,
            'memory': detail.memory,
        }
        
        recs, has_any_capacity, has_free_now, trace = self._analyze_resources(detail)
        node_blockers = self._compute_node_blockers(detail, ahead_job_ids)
        
        if status != 'running':
            if not has_any_capacity:
                status = 'blocked-resources'
            elif not has_free_now and status not in ['waiting-priority', 'blocked-dependency', 'blocked-error']:
                status = 'waiting-resources'
        
        return {
            'job_id': job_id,
            'user': detail.user,
            'partition': partition,
            'state': state,
            'status': status,
            'reason': reason,
            'eta': eta,
            'priority': priority,
            'blockers': [],
            'running_blocker': None,
            'resources': resources,
            'recommendations': recs,
            'capacity_trace': trace,
            'node_blockers': node_blockers,
        }

    def _analyze_resources(self, detail: JobDetail) -> Tuple[List[str], bool, bool, List[str]]:
        recs: List[str] = []
        trace: List[str] = []
        partition = detail.partition
        gpu_type = detail.gpu_type.lower() if detail.gpu_type else None
        nodes = [n for n in self.estimator.nodes if (not partition or partition in n.partitions)]
        max_gpus_per_node = max((n.gpus for n in nodes), default=0)

        reason_counts = {
            'gpu_mismatch': 0,
            'gpus_total': 0,
            'cpus_total': 0,
            'memory_total': 0,
            'unusable': 0,
        }
        unknown_gpu_type = 0
        inferred_gpu_types = 0
        usable_nodes = sum(1 for n in nodes if n.is_usable)
        gpu_match_nodes = 0
        capacity_nodes: List[Tuple[NodeResources, Optional[str]]] = []
        free_now_nodes: List[NodeResources] = []
        seen_gpu_types: Set[str] = set()

        for n in nodes:
            resolved_gpu_type = n.gpu_type
            if not resolved_gpu_type:
                resolved_gpu_type = self._infer_gpu_type_from_features(getattr(n, 'features', set()))
                if resolved_gpu_type:
                    inferred_gpu_types += 1
            if resolved_gpu_type:
                seen_gpu_types.add(resolved_gpu_type.lower())

            if not n.is_usable:
                reason_counts['unusable'] += 1

            if gpu_type:
                if resolved_gpu_type and resolved_gpu_type.lower() != gpu_type:
                    reason_counts['gpu_mismatch'] += 1
                    continue
                if not resolved_gpu_type:
                    unknown_gpu_type += 1

            gpu_match_nodes += 1
            if n.gpus < detail.gpus:
                reason_counts['gpus_total'] += 1
                continue
            if n.cpus < detail.cpus:
                reason_counts['cpus_total'] += 1
                continue
            if n.memory < detail.memory:
                reason_counts['memory_total'] += 1
                continue

            capacity_nodes.append((n, resolved_gpu_type))
            if (
                n.is_usable
                and n.gpus_free >= detail.gpus
                and n.cpus_free >= detail.cpus
                and n.memory_free >= detail.memory
            ):
                free_now_nodes.append(n)

        has_any_capacity = len(capacity_nodes) > 0
        has_free_now = len(free_now_nodes) > 0

        trace.append(f"nodes: {len(nodes)} (usable {usable_nodes}) in partition={partition or 'any'}")
        if gpu_type:
            seen = ", ".join(sorted(seen_gpu_types)) if seen_gpu_types else "none"
            trace.append(
                f"gpu match {gpu_match_nodes}/{len(nodes)} for {gpu_type}; inferred {inferred_gpu_types}, unknown {unknown_gpu_type}; seen {seen}"
            )
        trace.append(f"fit totals: {len(capacity_nodes)}; free now: {len(free_now_nodes)}")
        blocked_bits: List[str] = []
        for key, label in [
            ('gpu_mismatch', 'gpu type'),
            ('gpus_total', 'gpu count'),
            ('cpus_total', 'cpu'),
            ('memory_total', 'memory'),
        ]:
            if reason_counts[key]:
                blocked_bits.append(f"{label} {reason_counts[key]}")
        if reason_counts['unusable']:
            blocked_bits.append(f"unusable {reason_counts['unusable']}")
        if blocked_bits:
            trace.append("exclusions: " + ", ".join(blocked_bits))
        if detail.gpus > max_gpus_per_node and max_gpus_per_node > 0:
            trace.append(
                f"request {detail.gpus} GPUs exceeds per-node max {max_gpus_per_node}; multi-node fit not modeled"
            )

        if partition:
            part_cap = self.estimator.partition_limits.get(partition)
            if part_cap and detail.time_limit_minutes > part_cap:
                hours_cap = part_cap / 60
                recs.append(f"partition={partition}; timelimit≤{hours_cap:.1f}h")

        if not has_any_capacity:
            max_cpus = max((n.cpus for n in nodes), default=0)
            max_mem = max((n.memory for n in nodes), default=0)
            if detail.cpus > max_cpus and max_cpus > 0:
                recs.append(f"partition={partition or 'any'}; cpus≤{max_cpus}")
            if detail.memory > max_mem and max_mem > 0:
                recs.append(f"partition={partition or 'any'}; memory≤{max_mem/1024:.0f}GB")
            if gpu_type and seen_gpu_types:
                recs.append(f"available GPU types: {', '.join(sorted(seen_gpu_types))}")
            if detail.gpus > max_gpus_per_node and max_gpus_per_node > 0:
                recs.append(f"gpus≤{max_gpus_per_node} (per-node limit) or allow multi-node")
            return recs, has_any_capacity, has_free_now, trace

        alternatives: List[str] = []
        alt_seen = set()

        def format_alt(part: str, gpu_t: Optional[str], cpus_val: int, mem_mb: int, tl_min: int) -> str:
            mem_gb = max(1, int((mem_mb + 1023) // 1024))
            tl_hours = tl_min / 60 if tl_min else 0
            segments = [
                f"partition={part}",
                f"gpu_type={gpu_t}" if gpu_t else "gpu_type=any",
                f"gpus={detail.gpus}",
                f"cpus={cpus_val}",
                f"memory={mem_gb}GB",
            ]
            if tl_min:
                segments.append(f"timelimit≤{tl_hours:.1f}h")
            return "; ".join(segments)

        def consider_node(n: NodeResources, part: str, allow_gpu_switch: bool, resolved_gpu: Optional[str]) -> None:
            nonlocal alternatives
            if detail.partition and part and part != detail.partition:
                return
            if self.estimator.filter_partition and part and part != self.estimator.filter_partition:
                return
            gpu_t = detail.gpu_type
            if not gpu_t and resolved_gpu:
                gpu_t = resolved_gpu
            if gpu_type and resolved_gpu and resolved_gpu.lower() != gpu_type:
                if not allow_gpu_switch:
                    return
                gpu_t = resolved_gpu
            cpus_needed = min(detail.cpus, n.cpus_free)
            mem_needed = min(detail.memory, n.memory_free)
            part_cap = self.estimator.partition_limits.get(part)
            time_needed = detail.time_limit_minutes
            if part_cap:
                time_needed = min(time_needed, part_cap)

            if n.gpus_free < detail.gpus or cpus_needed <= 0 or mem_needed <= 0:
                return
            if (
                detail.cpus <= n.cpus_free
                and detail.memory <= n.memory_free
                and detail.time_limit_minutes <= (part_cap or detail.time_limit_minutes)
            ):
                return

            alt = format_alt(part, gpu_t, cpus_needed, mem_needed, time_needed)
            if alt not in alt_seen and len(alternatives) < 4:
                alt_seen.add(alt)
                alternatives.append(alt)

        for n, resolved_gpu in capacity_nodes:
            if not n.is_usable:
                continue
            consider_node(n, partition or n.partition, allow_gpu_switch=False, resolved_gpu=resolved_gpu)

        for n in self.estimator.nodes:
            if not n.is_usable or n.gpus_free < detail.gpus:
                continue
            resolved_gpu = n.gpu_type or self._infer_gpu_type_from_features(getattr(n, 'features', set()))
            consider_node(n, n.partition, allow_gpu_switch=True, resolved_gpu=resolved_gpu)
            if len(alternatives) >= 4:
                break

        recs.extend(alternatives)
        return recs, has_any_capacity, has_free_now, trace


def _format_eta(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    now = datetime.now()
    delta = dt - now
    minutes = int(delta.total_seconds() // 60)
    if minutes <= 0:
        return "imminent"
    if minutes < 120:
        return f"{minutes} min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f} h"
    days = hours / 24
    return f"{days:.1f} d"


def _format_blocker_time(dt: Optional[datetime], now: Optional[datetime] = None) -> str:
    if not dt:
        return "unknown"
    base = dt.strftime("%I:%M%p").lstrip("0").lower()
    now = now or datetime.now()
    day_offset = (dt.date() - now.date()).days
    if day_offset > 0:
        base += f"+{day_offset}"
    return base


def _render_blame_table(diag: Dict[str, any]) -> Panel:
    status = diag.get('status', 'unknown')
    status_map = {
        'running': "[bold green]✓ Running[/]",
        'waiting-resources': "[bold yellow]⏳ Waiting resources[/]",
        'waiting-priority': "[bold yellow]⏫ Waiting priority[/]",
        'blocked-dependency': "[bold red]⚠ Dependency[/]",
        'blocked-error': "[bold red]⚠ Error[/]",
        'blocked-resources': "[bold red]⚠ Unschedulable[/]",
        'error': "[bold red]⚠ Query error[/]",
        'unknown': "[bold orange]? Unknown[/]",
    }
    status_text = status_map.get(status, "[bold orange]? Unknown[/]")
    
    eta_text = _format_eta(diag.get('eta'))
    reason = diag.get('reason') or "No reason reported"
    resources = diag.get('resources') or {}
    mem_mb = resources.get('memory', 0)
    mem_gb = f"{mem_mb/1024:.0f} GB" if mem_mb else "—"
    res_text = f"{resources.get('gpus', 0)} GPU(s), {resources.get('cpus', 0)} CPU(s), {mem_gb}"
    if resources.get('gpu_type'):
        res_text += f", {resources['gpu_type']}"
    
    node_blockers = diag.get('node_blockers') or []
    reason_seen: Dict[str, str] = {}
    node_lines: List[str] = []
    now = datetime.now()
    for entry in node_blockers:
        start_text = _format_eta(entry.get('start'))
        node_name = entry.get('node', '?')
        reasons = entry.get('reasons') or ["No blockers on this node"]
        deduped: List[str] = []
        for r in reasons:
            if r in reason_seen and reason_seen[r] != node_name:
                deduped.append(f"{reason_seen[r]}…")
            else:
                reason_seen[r] = node_name
                deduped.append(r)
        reason_text = "\n".join(f"• {r}" for r in deduped)
        node_lines.append(f"[bold]{node_name}[/] ({start_text})\n{reason_text}")
    blocker_text = "\n\n".join(node_lines) if node_lines else "None ahead"
    
    table = Table(
        title=f"Job {diag.get('job_id', '?')} ({diag.get('user', 'unknown')})",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        title_style="bold",
        expand=True,
    )
    table.add_column("State", justify="center", width=14)
    table.add_column("Priority", justify="right", width=9)
    table.add_column("ETA", justify="center", width=10)
    table.add_column("Reason", justify="left")
    table.add_column("Blockers", justify="left", width=38)
    table.add_column("Resources", justify="left", width=22)
    table.add_column("Recommendations", justify="left", width=28)
    
    recs = diag.get('recommendations') or []
    rec_text = "\n".join(f"- {r}" for r in recs) if recs else "—"
    
    table.add_row(
        status_text,
        str(diag.get('priority', 0)),
        eta_text,
        reason,
        blocker_text,
        res_text,
        rec_text,
    )
    return Panel(table, border_style="cyan")


def run_available(args: argparse.Namespace) -> None:
    console = Console()
    if not args.monitor:
        import os
        import shutil
        cols, lines = shutil.get_terminal_size()
        os.environ["COLUMNS"] = str(cols)
        os.environ["LINES"] = str(max(1, lines - 6))
        
        try:
            estimator = SlurmEstimator(
                max_gpus=args.max_gpus,
                verbose=args.verbose,
                partition=args.partition,
                user=args.user,
                gpu_type=args.gpu_type
            )
            
            estimator.refresh_data()
            results = estimator.estimate_all_configurations(show_progress=True)
            display = estimator.generate_display(results)
            console.print(display)
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            sys.exit(1)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/] {e}", style="red")
            if args.verbose:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)
    else:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]SLURM Resource Estimator - Monitor Mode[/]\n"
            f"[dim]Refreshing every {args.n:.1f} seconds...[/]",
            border_style="cyan"
        ))
        console.print()
        
        estimator = SlurmEstimator(
            max_gpus=args.max_gpus,
            verbose=False,
            partition=args.partition,
            user=args.user,
            gpu_type=args.gpu_type
        )
        
        try:
            with Live(console=console, refresh_per_second=2, screen=False) as live:
                while True:
                    estimator.refresh_data()
                    results = estimator.estimate_all_configurations(show_progress=False)
                    display = estimator.generate_display(results)
                    live.update(display)
                    time.sleep(args.n)
        except KeyboardInterrupt:
            console.print("\n[green]Monitor mode stopped by user[/green]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n[bold red]Error:[/] {e}", style="red")
            if args.verbose:
                import traceback
                console.print(traceback.format_exc())
            sys.exit(1)


def run_blame(args: argparse.Namespace) -> None:
    console = Console()
    estimator = SlurmEstimator(max_gpus=8, verbose=False)
    try:
        estimator.refresh_data()
    except Exception as e:
        console.print(f"[bold red]Error refreshing cluster data:[/] {e}")
    inspector = SlurmBlame(estimator=estimator, blockers=args.blockers)
    try:
        queue = inspector.snapshot_queue()
    except SlurmQueryError as e:
        console.print(f"[bold red]Error querying queue:[/] {e}")
        sys.exit(1)
    
    for job_id in args.job_ids:
        diag = inspector.diagnose(job_id, queue)
        table = _render_blame_table(diag)
        console.print(table)
        console.print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SLURM diagnostic CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    
    avail = subparsers.add_parser(
        "available",
        help="Estimate job parameters that can run right now",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s available                # Single snapshot
  %(prog)s available --monitor      # Monitor mode (refresh every 5s)
  %(prog)s available --monitor -n 10
  %(prog)s available --max-gpus 32 -v
        """,
    )
    avail.add_argument('--max-gpus', type=int, default=8, help='Maximum GPUs to analyze (default: 8, capped at 8)')
    avail.add_argument('--monitor', '-m', action='store_true', help='Enable monitor mode with auto-refresh')
    avail.add_argument('-n', type=float, default=5.0, metavar='SECONDS', help='Refresh interval in seconds (default: 5.0)')
    avail.add_argument('--verbose', '-v', action='store_true', help='Show detailed progress information')
    avail.add_argument('--partition', '-p', type=str, default=None, help='Restrict analysis to a single partition')
    avail.add_argument('--user', '-u', type=str, default=None, help='Evaluate as this user (default: current user)')
    avail.add_argument('--gpu', '-g', dest='gpu_type', type=str, default=None, help='Restrict to GPU type (e.g., A100_40GB, H200)')
    
    blame = subparsers.add_parser(
        "blame",
        help="Diagnose why specific jobs are not running",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    blame.add_argument('job_ids', nargs='+', help='Job IDs to diagnose')
    blame.add_argument('--blockers', '-b', type=int, default=5, help='Number of higher-priority jobs to list')
    
    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] not in ['available', 'blame']:
        argv = ['available'] + argv
    args = parser.parse_args(argv)
    
    if args.command == 'blame':
        run_blame(args)
    else:
        run_available(args)


if __name__ == '__main__':
    main()
