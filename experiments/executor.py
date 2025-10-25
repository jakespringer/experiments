from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Set

import argparse
import base64
import hashlib
import os
import subprocess
import sys
import json
import tempfile
import yaml

from pathlib import Path
from shlex import quote as shquote

from .artifact import Artifact, ArtifactSet
from .config import ConfigManager
from .project import Project

# Optional pretty progress bar (stderr). Falls back gracefully if unavailable.
try:
    from tqdm import tqdm as _tqdm  # type: ignore
except Exception:  # pragma: no cover
    _tqdm = None


class _Progress:
    """Minimal progress helper that writes to stderr (pretty with tqdm if available)."""

    def __init__(self, total: int, desc: str) -> None:
        self._total = max(0, int(total))
        self._count = 0
        self._desc = desc
        if _tqdm is not None and self._total > 0:
            self._bar = _tqdm(
                total=self._total,
                desc=desc,
                file=sys.stderr,
                bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} steps",
                leave=False,
            )
        else:
            self._bar = None
            self._render()

    def set(self, desc: str) -> None:
        self._desc = desc
        if self._bar is not None:
            self._bar.set_description(desc)
        else:
            self._render()

    def advance(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)
        else:
            self._count = min(self._total, self._count + n)
            self._render()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
        else:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _render(self) -> None:
        sys.stderr.write(f"\r{self._desc} [{self._count}/{self._total}]")
        sys.stderr.flush()


def dquote(s: str) -> str:
    """Quote a string using double quotes, allowing environment variable expansion.
    
    Only escapes characters that would break the double-quoted string:
    backslash, double quote, and backtick. Dollar signs are NOT escaped,
    allowing $VAR to expand.
    """
    # Escape backslash first (must be first!)
    s = s.replace('\\', '\\\\')
    # Escape double quotes
    s = s.replace('"', '\\"')
    # Escape backticks (command substitution)
    s = s.replace('`', '\\`')
    return f'"{s}"'

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
    
    def create_json_file(self, path: str, content: Dict[str, Any]) -> None:
        """Add a JSON file creation block to this task.
        
        The content dictionary will be serialized to JSON format.
        """
        json_content = json.dumps(content, indent=2, sort_keys=False)
        self.blocks.append(CreateFileTaskBlock(path, json_content))
    
    def run_command(
        self,
        command: str,
        vargs: Sequence[str] | None = None,
        kwargs: Dict[str, Any] | None = None,
        vformat: str | None = None,
        kwformat: str | None = None,
        flagformat: str | None = None,
    ) -> None:
        """Add a command execution block to this task."""
        self.blocks.append(CommandTaskBlock(command, vargs, kwargs, vformat, kwformat, flagformat))

    def upload_to_gs(self, path: str, gs_path: str, directory: bool = False, contents: bool | None = True) -> None:
        """Add a Google Cloud Storage upload block to this task."""
        self.blocks.append(UploadToGSTaskBlock(path, gs_path, directory=directory, contents=contents))

    def download_from_gs(self, gs_path: str, path: str, directory: bool = False, skip_existing: bool = True, contents: bool | None = True) -> None:
        """Add a Google Cloud Storage download block to this task."""
        self.blocks.append(DownloadFromGSTaskBlock(gs_path, path, directory=directory, skip_existing=skip_existing, contents=contents))

    def download(self, url: str, local_path: str, skip_existing: bool = True) -> None:
        """Add a web URL download block to this task."""
        self.blocks.append(DownloadTaskBlock(url, local_path, skip_existing=skip_existing))

    def ensure_directory(self, path: str) -> None:
        """Add a directory creation block to this task."""
        self.blocks.append(EnsureDirectoryTaskBlock(path))

    def download_hf_model(self, model_name: str, local_dir: str, skip_existing: bool = True) -> None:
        """Add a Hugging Face model download block to this task."""
        self.blocks.append(DownloadHFModelTaskBlock(model_name, local_dir, skip_existing=skip_existing))

    def rsync_to_gs(self, path: str, gs_path: str, delete: bool = False, checksum: bool = False, contents: bool | None = True, check_exists: bool = False) -> None:
        """Add a Google Cloud Storage rsync upload block to this task."""
        self.blocks.append(RsyncToGSTaskBlock(path, gs_path, delete=delete, checksum=checksum, contents=contents, check_exists=check_exists))

    def rsync_from_gs(self, gs_path: str, path: str, delete: bool = False, checksum: bool = False, skip_existing: bool = True, contents: bool | None = True, check_exists: bool = False) -> None:
        """Add a Google Cloud Storage rsync download block to this task."""
        self.blocks.append(RsyncFromGSTaskBlock(gs_path, path, delete=delete, checksum=checksum, skip_existing=skip_existing, contents=contents, check_exists=check_exists))

    def set_env(self, name: str, value: str, from_command: bool = False) -> None:
        """Add an environment variable export block to this task.
        
        Args:
            name: The environment variable name
            value: The value to set (or command to run if from_command=True)
            from_command: If True, treat value as a command and set the variable
                         to the command's stdout
        """
        self.blocks.append(SetEnvTaskBlock(name, value, from_command=from_command))

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
        flagformat: str | None = None,
    ) -> None:
        self.command = command
        self.vargs = vargs or []
        self.kwargs = kwargs or {}
        self.vformat = vformat or '{v}'
        self.kwformat = kwformat or '--{k} \'{v}\''
        self.flagformat = flagformat or '--{k}'

    def execute(self) -> str:
        """Build the complete command string with formatted arguments."""
        parts = [self.command]
        
        # Add positional arguments
        if self.vargs:
            vargs_str = ' '.join(self.vformat.format(v=arg) for arg in self.vargs)
            parts.append(vargs_str)
        
        # Add keyword arguments
        if self.kwargs:
            formatted_kwargs: List[str] = []
            for k, v in self.kwargs.items():
                # Omit None and False values entirely
                if v is None:
                    continue
                if isinstance(v, bool):
                    if v:
                        formatted_kwargs.append(self.flagformat.format(k=k))
                    # If False, omit entirely
                    continue
                formatted_kwargs.append(self.kwformat.format(k=k, v=v))
            if formatted_kwargs:
                parts.append(' '.join(formatted_kwargs))
        
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
            parts.append(f"mkdir -p -- {dquote(parent)}")

        # Write file using base64 decoding
        parts.append(f"base64 -d > {dquote(self.path)} << '{delim}'\n{b64}\n{delim}")

        # Set file permissions if specified
        if self.mode is not None:
            parts.append(f"chmod {self.mode} {dquote(self.path)}")
        
        return " && ".join(parts)

class UploadToGSTaskBlock(TaskBlock):
    """Uploads a file or directory to Google Cloud Storage with locking."""

    def __init__(
        self,
        path: str,
        gs_path: str,
        directory: bool = False,
        contents: bool | None = True,
    ) -> None:
        self.path = path
        self.gs_path = gs_path  # Should be gs://bucket/path format
        self.directory = directory
        self.contents = contents  # If True, add trailing slashes; if False, remove; if None, leave
    
    def execute(self) -> str:
        """Generate a locked gsutil upload command."""
        # Create a lockfile based on the local path to prevent concurrent access
        path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:10]
        lockfile = f"/tmp/{path_hash}.lock"

        # Prepare source and destination paths, respecting contents flag
        source_path = self.path
        dest_path = self.gs_path
        if self.directory:
            if self.contents is True:
                # Copy contents: use wildcard on source, no trailing slash on dest
                source_path = source_path.rstrip('/') + '/*'
                dest_path = dest_path.rstrip('/')
            elif self.contents is False:
                source_path = source_path.rstrip('/')
                dest_path = dest_path.rstrip('/')

        # Build the gsutil command
        if self.directory:
            # Use -m for parallel operations and -r for recursive
            gsutil_cmd = f"gsutil -m cp -r {dquote(source_path)} {dquote(dest_path)}"
        else:
            gsutil_cmd = f"gsutil cp {dquote(source_path)} {dquote(dest_path)}"

        # Wrap in an exclusive file lock to prevent race conditions
        return f"flock -x {dquote(lockfile)} -c {dquote(gsutil_cmd)}"


class DownloadFromGSTaskBlock(TaskBlock):
    """Downloads a file or directory from Google Cloud Storage with locking."""

    def __init__(
        self,
        gs_path: str,
        path: str,
        directory: bool = False,
        skip_existing: bool = True,
        contents: bool | None = True,
    ) -> None:
        self.gs_path = gs_path  # Should be gs://bucket/path format
        self.path = path  # Local destination path
        self.directory = directory
        self.skip_existing = skip_existing
        self.contents = contents  # If True, add trailing slashes; if False, remove; if None, leave

    def execute(self) -> str:
        """Generate a locked gsutil download command."""
        # Create a lockfile based on the local path to prevent concurrent access
        path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:10]
        lockfile = f"/tmp/{path_hash}.lock"

        # Build the inner command (existence check + download)
        inner_parts = []
        
        # If skip_existing is enabled, check if path already exists
        if self.skip_existing:
            inner_parts.append(f"[ ! -e {dquote(self.path)} ]")
        
        # Prepare source and destination paths, respecting contents flag
        source_path = self.gs_path
        dest_path = self.path
        if self.directory:
            if self.contents is True:
                # Copy contents: use wildcard on source, no trailing slash on dest
                source_path = source_path.rstrip('/') + '/*'
                dest_path = dest_path.rstrip('/')
            elif self.contents is False:
                source_path = source_path.rstrip('/')
                dest_path = dest_path.rstrip('/')

        # Create necessary directories before download
        if self.directory:
            inner_parts.append(f"mkdir -p -- {dquote(self.path.rstrip('/'))}")
            # Use -m for parallel operations and -r for recursive
            gsutil_cmd = f"gsutil -m cp -r {dquote(source_path)} {dquote(dest_path)}"
        else:
            parent = os.path.dirname(self.path) or "."
            inner_parts.append(f"mkdir -p -- {dquote(parent)}")
            gsutil_cmd = f"gsutil cp {dquote(source_path)} {dquote(dest_path)}"
        
        inner_parts.append(gsutil_cmd)
        
        # Combine inner parts
        inner_cmd = " && ".join(inner_parts)
        
        # Wrap entire command (including existence check) in flock
        if self.skip_existing:
            # Add skip message for when file exists
            locked_cmd = f"flock -x {dquote(lockfile)} -c {dquote(inner_cmd)} || echo \"Skipping download, {self.path} already exists\""
        else:
            locked_cmd = f"flock -x {dquote(lockfile)} -c {dquote(inner_cmd)}"
        
        return locked_cmd


class DownloadTaskBlock(TaskBlock):
    """Downloads a file from a web URL using curl."""

    def __init__(
        self,
        url: str,
        local_path: str,
        mkdirs: bool = True,
        skip_existing: bool = True,
    ) -> None:
        self.url = url
        self.local_path = local_path
        self.mkdirs = mkdirs
        self.skip_existing = skip_existing

    def execute(self) -> str:
        """Generate shell command to download the file."""
        parts = []
        
        # Create parent directory if needed
        if self.mkdirs:
            parent = os.path.dirname(self.local_path) or "."
            parts.append(f"mkdir -p -- {dquote(parent)}")
        
        # Build the curl command
        # -L: follow redirects
        # -o: output file
        curl_cmd = f"curl -L {dquote(self.url)} -o {dquote(self.local_path)}"
        parts.append(curl_cmd)
        
        download_cmd = " && ".join(parts)
        
        # If skip_existing is enabled, check if file already exists
        if self.skip_existing:
            # Skip download if file exists
            return f"[ ! -e {dquote(self.local_path)} ] && {{ {download_cmd}; }} || echo \"Skipping download, {self.local_path} already exists\""
        else:
            return download_cmd


class EnsureDirectoryTaskBlock(TaskBlock):
    """Creates a directory with mkdir -p."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def execute(self) -> str:
        """Generate shell command to create the directory."""
        return f"mkdir -p -- {dquote(self.path)}"


class DownloadHFModelTaskBlock(TaskBlock):
    """Downloads a Hugging Face model to a local directory."""

    def __init__(
        self,
        model_name: str,
        local_dir: str,
        mkdirs: bool = True,
        skip_existing: bool = True,
    ) -> None:
        self.model_name = model_name
        self.local_dir = local_dir
        self.mkdirs = mkdirs
        self.skip_existing = skip_existing

    def execute(self) -> str:
        """Generate shell command to download the HF model."""
        parts = []
        
        # Create directory if needed
        if self.mkdirs:
            parts.append(f"mkdir -p -- {dquote(self.local_dir)}")
        
        # Build the hf download command
        hf_cmd = f"hf download {dquote(self.model_name)} --local-dir {dquote(self.local_dir)}"
        parts.append(hf_cmd)
        
        download_cmd = " && ".join(parts)
        
        # If skip_existing is enabled, check if directory already exists and is non-empty
        if self.skip_existing:
            # Skip download if directory exists and is not empty
            return f"[ ! -e {dquote(self.local_dir)} ] && {{ {download_cmd}; }} || echo \"Skipping download, {self.local_dir} already exists\""
        else:
            return download_cmd


class RsyncToGSTaskBlock(TaskBlock):
    """Syncs a local directory to Google Cloud Storage using gsutil rsync with locking."""

    def __init__(
        self,
        path: str,
        gs_path: str,
        delete: bool = False,
        checksum: bool = False,
        contents: bool | None = None,
        check_exists: bool = False,
    ) -> None:
        self.path = path
        self.gs_path = gs_path  # Should be gs://bucket/path format
        self.delete = delete  # If True, delete files in destination not in source
        self.checksum = checksum  # If True, use checksum comparison instead of mtime
        self.contents = contents  # If True, add trailing slashes; if False, remove them; if None, leave as-is
        self.check_exists = check_exists  # If True, only rsync if remote path exists
    
    def execute(self) -> str:
        """Generate a locked gsutil rsync command."""
        # Create a lockfile based on the local path to prevent concurrent access
        path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:10]
        lockfile = f"/tmp/{path_hash}.lock"

        # Handle trailing slashes based on contents parameter
        source_path = self.path
        dest_path = self.gs_path
        if self.contents is True:
            # Add trailing slashes to sync directory contents
            if not source_path.endswith('/'):
                source_path = source_path + '/'
            if not dest_path.endswith('/'):
                dest_path = dest_path + '/'
        elif self.contents is False:
            # Remove trailing slashes
            source_path = source_path.rstrip('/')
            dest_path = dest_path.rstrip('/')
        # If contents is None, leave paths as-is

        # Build the inner command (existence check + rsync)
        inner_parts = []
        
        # If check_exists is enabled, check if the remote path exists
        if self.check_exists:
            inner_parts.append(f"gsutil -q ls {dquote(dest_path)} > /dev/null 2>&1")
        
        # Build the gsutil rsync command
        # -r for recursive sync, -m for parallel operations
        gsutil_cmd_parts = ["gsutil", "-m", "rsync", "-r"]
        
        # Add optional flags
        if self.delete:
            gsutil_cmd_parts.append("-d")  # Delete files in dest not in source
        if self.checksum:
            gsutil_cmd_parts.append("-c")  # Use checksum instead of mtime
        
        # Add source and destination
        gsutil_cmd_parts.extend([dquote(source_path), dquote(dest_path)])
        gsutil_cmd = " ".join(gsutil_cmd_parts)
        
        inner_parts.append(gsutil_cmd)
        
        # Combine inner parts
        inner_cmd = " && ".join(inner_parts)
        
        # Wrap entire command (including existence check) in flock
        if self.check_exists:
            # Add skip message for when remote path doesn't exist
            locked_cmd = f"flock -x {dquote(lockfile)} -c {dquote(inner_cmd)} || echo \"Skipping rsync, remote path {dest_path} does not exist\""
        else:
            locked_cmd = f"flock -x {dquote(lockfile)} -c {dquote(inner_cmd)}"
        
        return locked_cmd


class RsyncFromGSTaskBlock(TaskBlock):
    """Syncs from Google Cloud Storage to a local directory using gsutil rsync with locking."""

    def __init__(
        self,
        gs_path: str,
        path: str,
        delete: bool = False,
        checksum: bool = False,
        skip_existing: bool = True,
        contents: bool | None = None,
        check_exists: bool = False,
    ) -> None:
        self.gs_path = gs_path  # Should be gs://bucket/path format
        self.path = path  # Local destination path
        self.delete = delete  # If True, delete files in destination not in source
        self.checksum = checksum  # If True, use checksum comparison instead of mtime
        self.skip_existing = skip_existing
        self.contents = contents  # If True, add trailing slashes; if False, remove them; if None, leave as-is
        self.check_exists = check_exists  # If True, only rsync if remote path exists

    def execute(self) -> str:
        """Generate a locked gsutil rsync command."""
        # Create a lockfile based on the local path to prevent concurrent access
        path_hash = hashlib.sha256(self.path.encode("utf-8")).hexdigest()[:10]
        lockfile = f"/tmp/{path_hash}.lock"

        # Handle trailing slashes based on contents parameter
        source_path = self.gs_path
        dest_path = self.path
        if self.contents is True:
            # Add trailing slashes to sync directory contents
            if not source_path.endswith('/'):
                source_path = source_path + '/'
            if not dest_path.endswith('/'):
                dest_path = dest_path + '/'
        elif self.contents is False:
            # Remove trailing slashes
            source_path = source_path.rstrip('/')
            dest_path = dest_path.rstrip('/')
        # If contents is None, leave paths as-is

        # Build the inner command (all checks + mkdir + rsync)
        inner_parts = []
        
        # If skip_existing is enabled, check if local path already exists
        if self.skip_existing:
            inner_parts.append(f"[ ! -e {dquote(self.path.rstrip('/'))} ]")
        
        # If check_exists is enabled, check if the remote path exists
        if self.check_exists:
            inner_parts.append(f"gsutil -q ls {dquote(source_path)} > /dev/null 2>&1")
        
        # Create necessary directory before sync
        # Use the original path (without trailing slash) for mkdir
        inner_parts.append(f"mkdir -p -- {dquote(self.path.rstrip('/'))}")

        # Build the gsutil rsync command
        # -r for recursive sync, -m for parallel operations
        gsutil_cmd_parts = ["gsutil", "-m", "rsync", "-r"]
        
        # Add optional flags
        if self.delete:
            gsutil_cmd_parts.append("-d")  # Delete files in dest not in source
        if self.checksum:
            gsutil_cmd_parts.append("-c")  # Use checksum instead of mtime
        
        # Add source and destination (with potential trailing slashes)
        gsutil_cmd_parts.extend([dquote(source_path), dquote(dest_path)])
        gsutil_cmd = " ".join(gsutil_cmd_parts)
        
        inner_parts.append(gsutil_cmd)
        
        # Combine inner parts
        inner_cmd = " && ".join(inner_parts)
        
        # Wrap entire command (including all checks) in flock
        locked_cmd = f"flock -x {dquote(lockfile)} -c {dquote(inner_cmd)}"
        
        # Add appropriate skip message based on what checks are enabled
        if self.skip_existing and self.check_exists:
            return f"{locked_cmd} || echo \"Skipping sync (path exists or remote unavailable)\""
        elif self.skip_existing:
            return f"{locked_cmd} || echo \"Skipping sync, {self.path} already exists\""
        elif self.check_exists:
            return f"{locked_cmd} || echo \"Skipping rsync, remote path {source_path} does not exist\""
        else:
            return locked_cmd


class SetEnvTaskBlock(TaskBlock):
    """Sets an environment variable by exporting it."""

    def __init__(self, name: str, value: str, from_command: bool = False) -> None:
        self.name = name
        self.value = value
        self.from_command = from_command

    def execute(self) -> str:
        """Generate shell command to export an environment variable."""
        if self.from_command:
            # Set variable to the stdout of the command (strip trailing whitespace)
            # Use $(...) for command substitution
            return f"export {self.name}=$({self.value})"
        else:
            # Use dquote to allow variable expansion in the value
            return f"export {self.name}={dquote(self.value)}"


def _find_artifact_dependencies(value: Any) -> Iterable[Artifact]:
    """Recursively find all Artifact instances within a data structure.
    
    This traverses dictionaries, lists, tuples, sets, and ArtifactSets to discover
    artifact dependencies declared in artifact attributes.
    """
    if isinstance(value, Artifact):
        yield value
    elif isinstance(value, ArtifactSet):
        # ArtifactSet may contain multiple artifacts that are dependencies
        for item in value:
            yield from _find_artifact_dependencies(item)
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
        self._verbose_filtering: bool = True  # Whether to print filter messages

    def stage(self, name: str, artifacts: Iterable[Artifact] | ArtifactSet) -> None:
        """Register a named stage containing artifacts to execute.
        
        Accepts a single `Artifact`, an `ArtifactSet`, or any iterable of `Artifact`s.
        A single `Artifact` is wrapped into an `ArtifactSet` for consistency.
        """
        if isinstance(artifacts, Artifact):
            artifacts = ArtifactSet([artifacts])
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

    def execute(self, stages: List[str], head: int | None = None, tail: int | None = None, rerun: bool = False, artifacts: List[str] | None = None, jobs: int | None = None) -> None:
        """Execute the specified stages (or all stages if empty list).
        
        Args:
            stages: List of stage names to execute
            head: If provided, only execute the first N artifacts
            tail: If provided, only execute the last N artifacts
            rerun: If True, ignore exists check and run all artifacts            artifacts: If provided, only execute artifacts with these class names
            jobs: If provided, maximum number of jobs to launch in each group
        """
        # Validate inputs
        if not self._stages:
            print("No stages registered.", file=sys.stderr)
            return

        # Validate head/tail arguments
        if head is not None and tail is not None:
            print("Error: Cannot specify both --head and --tail", file=sys.stderr)
            return
        
        if head is not None and head <= 0:
            print(f"Error: --head must be positive, got {head}", file=sys.stderr)
            return
        
        if tail is not None and tail <= 0:
            print(f"Error: --tail must be positive, got {tail}", file=sys.stderr)
            return

        stages = self._validate_and_normalize_stages(stages)
        unique_artifacts = self._collect_unique_artifacts()
        
        if not unique_artifacts:
            print("No artifacts registered.", file=sys.stderr)
            return

        # Compute execution order via topological sort
        all_tiers = self.compute_topological_ordering(unique_artifacts)
        
        # Filter to only artifacts in selected stages
        filtered_tiers = self._filter_tiers_by_stages(all_tiers, stages)
        
        if not filtered_tiers:
            print("No artifacts matched the selected stages.", file=sys.stderr)
            return
        
        # Filter by artifact class names if specified
        if artifacts:
            filtered_tiers = self._filter_tiers_by_artifact_class(filtered_tiers, artifacts)
            
            if not filtered_tiers:
                print(f"No artifacts matched the specified types: {', '.join(artifacts)}", file=sys.stderr)
                return

        # Filter out artifacts that should be skipped (e.g., already exist)
        # unless rerun flag is set
        executable_tiers, skipped_artifacts = self._filter_skipped_artifacts(filtered_tiers, rerun=rerun)
        
        # Report skipped artifacts
        if skipped_artifacts:
            # Build mapping of artifact ID to stage names for display
            artifact_to_stages = self._build_artifact_to_stages_mapping()
            
            print(f"Skipping {len(skipped_artifacts)} artifact(s) that already exist:", file=sys.stderr)
            for artifact in skipped_artifacts:
                stage_names = sorted(artifact_to_stages.get(id(artifact), set()) & set(stages))
                stage_str = f" [stages: {', '.join(stage_names)}]" if stage_names else ""
                print(f"  - {artifact.__class__.__name__} ({artifact.relpath}){stage_str}", file=sys.stderr)
            print(file=sys.stderr)
        
        if not executable_tiers:
            print("All artifacts already exist. Nothing to execute.", file=sys.stderr)
            return

        # Apply head/tail filtering if requested
        if head is not None or tail is not None:
            executable_tiers = self._apply_head_tail_filter(executable_tiers, head, tail)
            
            if not executable_tiers:
                limit_type = "head" if head is not None else "tail"
                limit_value = head if head is not None else tail
                print(f"No artifacts remain after applying --{limit_type} {limit_value}", file=sys.stderr)
                return

        # Build mapping of tier index to stage names
        tier_to_stages = self._build_tier_to_stages_mapping(executable_tiers, stages)
        
        # Compile and launch
        task_tiers = [[self.compile_artifact(a) for a in tier] for tier in executable_tiers]
        self.launch(task_tiers, tier_to_stages=tier_to_stages, jobs=jobs)

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

    def _build_artifact_to_stages_mapping(self) -> Dict[int, Set[str]]:
        """Build a mapping from artifact ID to the set of stage names it belongs to."""
        artifact_to_stages: Dict[int, Set[str]] = {}
        for stage_name, artifacts in self._stages.items():
            for artifact in artifacts:
                artifact_to_stages.setdefault(id(artifact), set()).add(stage_name)
        return artifact_to_stages

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
    
    def _filter_tiers_by_artifact_class(
        self,
        tiers: List[List[Artifact]],
        artifact_classes: List[str],
    ) -> List[List[Artifact]]:
        """Filter artifact tiers to only include artifacts with matching class names.
        
        Args:
            tiers: List of artifact tiers
            artifact_classes: List of artifact class names to include
            
        Returns:
            Filtered list of artifact tiers containing only matching artifacts
        """
        artifact_class_set = set(artifact_classes)
        filtered_tiers: List[List[Artifact]] = []
        
        for tier in tiers:
            # Keep artifacts whose class name matches one of the specified names
            filtered_tier = [
                a for a in tier
                if a.__class__.__name__ in artifact_class_set
            ]
            if filtered_tier:
                filtered_tiers.append(filtered_tier)
        
        return filtered_tiers
    
    def _filter_skipped_artifacts(
        self,
        tiers: List[List[Artifact]],
        rerun: bool = False,
    ) -> tuple[List[List[Artifact]], List[Artifact]]:
        """Filter out artifacts that should be skipped.
        
        Args:
            tiers: List of artifact tiers
            rerun: If True, don't skip any artifacts (ignore exists check)
            
        Returns:
            Tuple of (executable_tiers, skipped_artifacts) where:
            - executable_tiers: Tiers with skipped artifacts removed
            - skipped_artifacts: List of artifacts that were skipped
        """
        skipped_artifacts: List[Artifact] = []
        executable_tiers: List[List[Artifact]] = []
        
        # If rerun is True, don't skip anything
        if rerun:
            return tiers, skipped_artifacts
        
        for tier in tiers:
            executable_tier = []
            for artifact in tier:
                if artifact.should_skip():
                    skipped_artifacts.append(artifact)
                else:
                    executable_tier.append(artifact)
            
            # Only include non-empty tiers
            if executable_tier:
                executable_tiers.append(executable_tier)
        
        return executable_tiers, skipped_artifacts
    
    def _apply_head_tail_filter(
        self,
        tiers: List[List[Artifact]],
        head: int | None,
        tail: int | None,
    ) -> List[List[Artifact]]:
        """Apply head or tail filtering to artifact tiers.
        
        Args:
            tiers: List of artifact tiers
            head: If provided, keep only the first N artifacts
            tail: If provided, keep only the last N artifacts
            
        Returns:
            Filtered list of artifact tiers
        """
        # Flatten tiers to get ordered list of artifacts
        all_artifacts = [artifact for tier in tiers for artifact in tier]
        
        # Apply head/tail filter
        if head is not None:
            selected_artifacts = all_artifacts[:head]
            if self._verbose_filtering:
                print(f"Limiting to first {head} artifact(s) (out of {len(all_artifacts)} total)", file=sys.stderr)
        elif tail is not None:
            selected_artifacts = all_artifacts[-tail:]
            if self._verbose_filtering:
                print(f"Limiting to last {tail} artifact(s) (out of {len(all_artifacts)} total)", file=sys.stderr)
        else:
            return tiers
        
        if not selected_artifacts:
            return []
        
        if self._verbose_filtering:
            print(file=sys.stderr)
        
        # Create a set of selected artifact IDs for fast lookup
        selected_ids = {id(a) for a in selected_artifacts}
        
        # Filter tiers to only include selected artifacts
        filtered_tiers: List[List[Artifact]] = []
        for tier in tiers:
            filtered_tier = [a for a in tier if id(a) in selected_ids]
            if filtered_tier:
                filtered_tiers.append(filtered_tier)
        
        return filtered_tiers
    
    def _build_tier_to_stages_mapping(
        self,
        tiers: List[List[Artifact]],
        selected_stages: List[str],
    ) -> Dict[int, List[str]]:
        """Build mapping from tier index to list of stage names.
        
        Args:
            tiers: Filtered list of artifact tiers
            selected_stages: List of selected stage names
            
        Returns:
            Dictionary mapping tier index to list of stage names containing
            artifacts in that tier
        """
        # Build membership map: artifact ID -> set of stage names
        artifact_to_stages: Dict[int, Set[str]] = {}
        for stage_name, artifacts in self._stages.items():
            for artifact in artifacts:
                artifact_to_stages.setdefault(id(artifact), set()).add(stage_name)
        
        # Build tier to stages mapping
        tier_to_stages: Dict[int, List[str]] = {}
        selected_set = set(selected_stages)
        
        for tier_index, tier in enumerate(tiers):
            # Collect all stage names for artifacts in this tier
            stages_in_tier: Set[str] = set()
            for artifact in tier:
                stages_in_tier.update(
                    artifact_to_stages.get(id(artifact), set()) & selected_set
                )
            tier_to_stages[tier_index] = sorted(stages_in_tier)
        
        return tier_to_stages

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

    def launch(self, tiers: List[List[Task]], tier_to_stages: Dict[int, List[str]] | None = None, jobs: int | None = None) -> None:
        raise NotImplementedError

    def compile_artifact(self, artifact: Artifact) -> Task:
        raise NotImplementedError


class PrintExecutor(Executor):
    """Executor that prints planned shell commands for inspection."""

    def __init__(
        self,
        artifact_path: str | None = None,
        code_path: str | None = None,
        gs_path: str | None = None,
        setup_command: str | None = None,
    ) -> None:
        super().__init__()
        mgr = ConfigManager()
        global_conf = mgr.ensure_config()
        proj_name = Project.name
        proj_conf: Dict[str, Any] = {}
        if proj_name is not None:
            try:
                proj_conf = mgr.load_project_config(proj_name).get("config", {})
            except Exception:
                proj_conf = {}
        if artifact_path is None:
            artifact_path = proj_conf.get("artifact_path") or str(mgr.get_project_dir(proj_name or "default") / "artifacts")
        if code_path is None:
            code_path = proj_conf.get("code_path") or str(Path.cwd())
        self.artifact_path = Path(artifact_path)
        self.code_path = Path(code_path)
        self.gs_path = gs_path
        self.setup_command = setup_command
        self._verbose_filtering = False  # Don't print filter messages for print command
        self._global_config = global_conf

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

    def launch(self, tiers: List[List[Task]], tier_to_stages: Dict[int, List[str]] | None = None, jobs: int | None = None) -> None:
        """Print all shell commands that would be executed."""
        # Print bash safety header for proper error handling
        print("#!/usr/bin/env bash")
        print("set -euo pipefail")
        print()
        
        # Export project configuration
        try:
            proj_conf = ConfigManager().load_project_config(Project.name or "").get("config", {})
        except Exception:
            proj_conf = {}
        import json as _json
        proj_conf_str = _json.dumps(proj_conf, separators=(",", ":"))
        print(f"export EXPERIMENTS_PROJECT_CONF={dquote(proj_conf_str)}")
        print()

        # Run setup commands if provided
        if self.setup_command:
            print(self.setup_command)
            print()
        
        for tier in tiers:
            for task in tier:
                # Export per-task experiment config
                if task.artifact is not None:
                    exp_conf = _artifact_experiment_conf(task.artifact)
                    exp_conf_str = _safe_json_dumps(exp_conf)
                    print(f"export EXPERIMENTS_EXPERIMENT_CONF={dquote(exp_conf_str)}")
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
        artifact_path: str | None = None,
        code_path: str | None = None,
        project: str | None = None,
        gs_path: str | None = None,
        default_slurm_args: Dict[str, Any] | None = None,
        dry_run: bool = False,
        setup_command: str | None = None,
    ) -> None:
        super().__init__()
        # Initialize project context if provided
        if project and (Project.name is None or Project.name != project):
            Project.init(project)
        mgr = ConfigManager()
        global_conf = mgr.ensure_config()
        proj_name = Project.name or project or "default"
        try:
            proj_conf: Dict[str, Any] = mgr.load_project_config(proj_name).get("config", {})
        except Exception:
            proj_conf = {}
        if artifact_path is None:
            artifact_path = proj_conf.get("artifact_path") or str(mgr.get_project_dir(proj_name) / "artifacts")
        if code_path is None:
            code_path = proj_conf.get("code_path") or str(Path.cwd())
        self.artifact_path = Path(artifact_path)
        self.code_path = Path(code_path)
        self.project = Project.name or project
        self.gs_path = gs_path
        self.default_slurm_args = default_slurm_args or {}
        self.default_slurm_args_by_partition: Dict[str, Dict[str, Any]] = {}
        self.dry_run = dry_run
        self.setup_command = setup_command
        self._next_fake_job_id = 1000  # For dry run mode
        self._dry_run_jobs: List[Dict[str, Any]] = []  # Store job info for dry run summary
        self.external_dependencies: List[str] = []  # External job IDs to depend on
        self.cli_slurm_overrides: Dict[str, Any] = {}  # Highest-priority overrides from CLI
        self.force_launch: bool = False
        self._active_jobs: Set[tuple] = set()
        self._launched_map: Dict[str, Any] = {}
        self._global_config: Dict[str, Any] = global_conf

    def auto_cli(self) -> None:
        """Launch the CLI interface for this executor."""
        from .cli import auto_cli
        auto_cli(self)

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

    def launch(self, tiers: List[List[Task]], tier_to_stages: Dict[int, List[str]] | None = None, jobs: int | None = None) -> None:
        """Submit each tier as one or more Slurm array jobs with proper dependencies.
        
        Args:
            tiers: List of task tiers to execute
            tier_to_stages: Optional mapping from tier index to list of stage names
        """
        # Map artifact ID to the job ID that produces it
        artifact_to_job_id: Dict[int, str] = {}
        all_job_ids: List[str] = []
        
        # Pre-compute group count for progress bar
        total_groups = 0
        for _tier in tiers:
            if not _tier:
                continue
            _groups = self._group_tasks_by_requirements(_tier)
            for _, _tasks in _groups.items():
                _filtered = _tasks
                if not self.force_launch:
                    _filtered = [t for t in _tasks if self._should_launch_task(t)]
                if _filtered:
                    total_groups += 1

        progress = _Progress(total=2 + total_groups, desc="Preparing launch")

        if self.dry_run:
            print("\n" + "=" * 100, file=sys.stderr)
            print("DRY RUN MODE - No jobs will be submitted", file=sys.stderr)
            if hasattr(self, 'external_dependencies') and self.external_dependencies:
                print(f"External dependencies: {', '.join(self.external_dependencies)}", file=sys.stderr)
            print("=" * 100 + "\n", file=sys.stderr)
        else:
            print("\n" + "=" * 80, file=sys.stderr)
            print("Launching Jobs", file=sys.stderr)
            if hasattr(self, 'external_dependencies') and self.external_dependencies:
                print(f"External dependencies: {', '.join(self.external_dependencies)}", file=sys.stderr)
            print("=" * 80 + "\n", file=sys.stderr)

        # Load and prune launched jobs mapping, and capture active jobs from squeue
        progress.set("Scanning running jobs")
        self._active_jobs = self._get_active_jobs()
        progress.advance(1)
        progress.set("Pruning launch history")
        self._launched_map = self._load_launched_jobs()
        self._prune_launched_jobs()
        progress.advance(1)

        for tier_index, tier in enumerate(tiers):
            if not tier:
                continue
            
            # Determine stage names for this tier
            stage_names = tier_to_stages.get(tier_index, []) if tier_to_stages else []
            
            # Submit this tier (may create multiple jobs if requirements differ)
            # and get mapping of artifacts to their job IDs
            job_mappings = self._submit_tier(
                tier_index,
                tier,
                artifact_to_job_id=artifact_to_job_id,
                stage_names=stage_names,
                jobs=jobs,
                progress=progress,
            )
            
            # Track all submitted jobs (deduplicate since multiple artifacts can map to same job)
            tier_job_ids = sorted(set(job_mappings.values()))
            all_job_ids.extend(tier_job_ids)
            
            # Update artifact-to-job mapping with new jobs from this tier
            artifact_to_job_id.update(job_mappings)
        
        # Print summary for dry run
        if self.dry_run:
            self._print_dry_run_summary()
        else:
            # Print final summary for actual launch
            self._print_launch_summary(all_job_ids)

        # Persist launched mapping after launch
        if not self.dry_run:
            self._save_launched_jobs(self._launched_map)

        progress.set("Done")
        progress.close()

    def _submit_tier(
        self,
        tier_index: int,
        tier: List[Task],
        artifact_to_job_id: Dict[int, str],
        stage_names: List[str] | None = None,
        jobs: int | None = None,
        progress: _Progress | None = None,
    ) -> Dict[int, str]:
        """Submit a tier as one or more Slurm array jobs.
        
        If tasks have different requirements, they are grouped and submitted
        as separate jobs. Each job only depends on the specific jobs that
        produce artifacts it needs.
        
        Args:
            tier_index: Index of the tier
            tier: List of tasks in the tier
            artifact_to_job_id: Mapping of artifact IDs to job IDs that produce them
            stage_names: Optional list of stage names for this tier
        
        Returns:
            Dictionary mapping artifact IDs to job IDs for tasks in this tier
        """
        from datetime import datetime
        

        # Group tasks by their requirements
        task_groups = self._group_tasks_by_requirements(tier)
        
        # Track which artifacts are produced by which job in this tier
        new_mappings: Dict[int, str] = {}
        
        # Submit a separate job for each requirement group
        for group_index, (requirements, tasks) in enumerate(task_groups.items()):
            # Optionally filter out tasks that already have running jobs (unless force)
            if not self.force_launch:
                tasks = [t for t in tasks if self._should_launch_task(t)]
            if not tasks:
                continue
            job_name = f"tier-{tier_index}"
            if len(task_groups) > 1:
                job_name += f"-grp{group_index}"
            
            # Get Slurm configuration for this group
            slurm_config = self._build_slurm_config(requirements)
            
            # Compute dependencies for this specific group
            dependency_job_ids = self._compute_dependencies_for_tasks(tasks, artifact_to_job_id)
            
            # Add external dependencies if specified
            if hasattr(self, 'external_dependencies') and self.external_dependencies:
                dependency_job_ids = list(set(dependency_job_ids) | set(self.external_dependencies))
                dependency_job_ids.sort()

            if progress is not None:
                progress.set(f"Submitting tier {tier_index} group {group_index + 1}/{len(task_groups)}")
            
            # Generate the sbatch script
            sbatch_header = self._build_sbatch_header(
                job_name,
                tasks,
                slurm_config,
                dependency_job_ids,
                jobs_limit=jobs,
            )
            script_body = self._build_script_body(tasks, jobs_limit=jobs)
            
            # Write script to temporary file
            script_path = self._write_script(tier_index, group_index, sbatch_header, script_body)
            
            # Submit the job (or fake it in dry run mode)
            job_id = self._submit_sbatch_script(script_path)
            if progress is not None:
                progress.advance(1)
            
            # Map all artifacts in this group to this job ID
            for task in tasks:
                if task.artifact is not None:
                    new_mappings[id(task.artifact)] = job_id
            
            # Extract log file from header
            log_file = None
            for line in sbatch_header:
                if line.startswith("#SBATCH --output="):
                    log_file = line.split("=", 1)[1]
                    break
            
            # Collect artifact class names and check existence
            artifact_classes = [
                task.artifact.__class__.__name__ 
                for task in tasks 
                if task.artifact is not None
            ]
            
            # Count how many tasks are complete vs remaining
            num_complete = sum(
                1 for task in tasks 
                if task.artifact is not None and task.artifact.exists
            )
            num_remaining = len(tasks) - num_complete
            
            # Determine which stage this group belongs to
            group_stage = self._determine_group_stage(tasks, stage_names)
            
            # Build job info
            job_info = {
                'job_id': job_id,
                'job_name': job_name,
                'tier': tier_index,
                'group': group_index,
                'stage': group_stage,
                'num_tasks': len(tasks),
                'num_complete': num_complete,
                'num_remaining': num_remaining,
                'artifact_classes': artifact_classes,
                'config': slurm_config,
                'dependencies': list(dependency_job_ids),
                'script_path': script_path,
                'log_file': log_file,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            # Store job info for dry run summary
            if self.dry_run:
                self._dry_run_jobs.append(job_info)
            
            # Save job info to persistent storage (per-project)
            if self.project and not self.dry_run and group_stage:
                ConfigManager().save_job_info(self.project, group_stage, job_info)

            # Record relpath -> {job_id, array_index} mappings for launched tasks
            if not self.dry_run:
                # Determine array index mapping for each task
                if jobs is not None and jobs > 0 and jobs < len(tasks):
                    # Round-robin assignment used in _build_script_body
                    for idx, task in enumerate(tasks):
                        if task.artifact is None:
                            continue
                        array_index = idx % jobs
                        self._launched_map[task.artifact.relpath] = {
                            'job_id': job_id,
                            'array_index': array_index,
                        }
                else:
                    for idx, task in enumerate(tasks):
                        if task.artifact is None:
                            continue
                        self._launched_map[task.artifact.relpath] = {
                            'job_id': job_id,
                            'array_index': idx,
                        }
            
            # Print brief summary (not in dry run mode)
            if not self.dry_run:
                # Get unique artifact types
                artifact_types = sorted(set(artifact_classes))
                artifact_summary = ', '.join(artifact_types) if artifact_types else 'N/A'
                
                stage_str = f" [{group_stage}]" if group_stage else ""
                print(f"✓ Submitted job {job_id}: {job_name}{stage_str}", file=sys.stderr)
                print(f"  Tasks: {len(tasks)}, Artifacts: {artifact_summary}", file=sys.stderr)
                print(f"  Config: partition={slurm_config.get('partition', 'N/A')}, "
                      f"cpus={slurm_config.get('cpus', 'N/A')}, "
                      f"gpus={slurm_config.get('gpus', 'N/A')}, "
                      f"time={slurm_config.get('time', 'N/A')}", file=sys.stderr)
                print(file=sys.stderr)
        
        return new_mappings
    
    def _compute_dependencies_for_tasks(
        self,
        tasks: List[Task],
        artifact_to_job_id: Dict[int, str],
    ) -> List[str]:
        """Compute the set of job IDs that these tasks depend on.
        
        Args:
            tasks: List of tasks to compute dependencies for
            artifact_to_job_id: Mapping of artifact IDs to job IDs that produce them
        
        Returns:
            Sorted list of unique job IDs that these tasks depend on
        """
        dependency_job_ids: Set[str] = set()
        
        for task in tasks:
            if task.artifact is None:
                continue
            
            # Find all artifacts this task depends on
            for attr_value in vars(task.artifact).values():
                for dependency in _find_artifact_dependencies(attr_value):
                    dependency_id = id(dependency)
                    # If we've already submitted a job for this dependency, add it
                    if dependency_id in artifact_to_job_id:
                        dependency_job_ids.add(artifact_to_job_id[dependency_id])
        
        return sorted(dependency_job_ids)
    
    def _determine_group_stage(self, tasks: List[Task], stage_names: List[str]) -> str | None:
        """Determine which stage this group of tasks belongs to.
        
        Args:
            tasks: List of tasks in this group
            stage_names: Possible stage names for this tier
            
        Returns:
            The stage name that best represents this group, or None
        """
        if not tasks or not stage_names:
            return None
        
        # Count which stage each artifact in this group belongs to
        stage_counts: Dict[str, int] = defaultdict(int)
        
        for task in tasks:
            if task.artifact is not None:
                # Find which stage(s) contain this artifact
                artifact_id = id(task.artifact)
                for stage_name in stage_names:
                    if stage_name in self._stages:
                        stage_artifacts = self._stages[stage_name]
                        if any(id(a) == artifact_id for a in stage_artifacts):
                            stage_counts[stage_name] += 1
        
        # Return the stage with the most artifacts in this group
        if stage_counts:
            return max(stage_counts.items(), key=lambda x: x[1])[0]
        
        # Fallback to first stage name
        return stage_names[0] if stage_names else None

    # Valid requirement keys that can be returned from get_requirements()
    VALID_REQUIREMENT_KEYS = {
        # Log files
        'output', 'error', 'separate_error',
        # Basic specifications
        'partition', 'time', 'account', 'qos', 'chdir',
        # Node and task resources
        'nodes', 'ntasks', 'cpus', 'cpus_per_task',
        # Memory
        'mem', 'mem_per_cpu',
        # GPU resources
        'gpus', 'gres', 'constraint',
        # Job control
        'requeue', 'signal', 'open_mode',
        # Email notifications
        'mail_type', 'mail_user',
    }

    def _get_task_requirements(self, task: Task) -> Dict[str, Any]:
        """Extract and normalize requirements from a task's artifact.
        
        Returns:
            Dictionary of requirements with validation and normalization applied.
        """
        if task.artifact is None or not hasattr(task.artifact, 'get_requirements'):
            return {}
        
        reqs = dict(task.artifact.get_requirements())  # type: ignore[attr-defined]
        
        # Validate that all keys are recognized
        invalid_keys = set(reqs.keys()) - self.VALID_REQUIREMENT_KEYS
        if invalid_keys:
            artifact_name = task.artifact.__class__.__name__
            raise ValueError(
                f"Invalid requirement key(s) in {artifact_name}.get_requirements(): {invalid_keys}. "
                f"Valid keys are: {sorted(self.VALID_REQUIREMENT_KEYS)}"
            )
        
        # Normalize: convert 'cpus' to 'cpus_per_task' for consistency
        if 'cpus' in reqs and 'cpus_per_task' not in reqs:
            reqs['cpus_per_task'] = reqs.pop('cpus')
        elif 'cpus' in reqs and 'cpus_per_task' in reqs:
            # Both specified - use cpus_per_task, ignore cpus
            reqs.pop('cpus')
        
        return reqs
    
    def _group_tasks_by_requirements(
        self,
        tier: List[Task],
    ) -> Dict[tuple, List[Task]]:
        """Group tasks by their resource requirements.
        
        Returns:
            Dictionary mapping requirement tuple to list of tasks with those requirements.
        """
        groups: Dict[tuple, List[Task]] = defaultdict(list)
        
        for task in tier:
            reqs = self._get_task_requirements(task)
            # Create a hashable key from sorted requirements
            # Use frozenset of items for proper hashing
            req_key = tuple(sorted(reqs.items()))
            groups[req_key].append(task)
        
        return groups

    def _build_slurm_config(self, requirements_tuple: tuple) -> Dict[str, Any]:
        """Build Slurm configuration from normalized requirements.
        
        Args:
            requirements_tuple: Tuple of (key, value) pairs from task requirements
        
        Returns:
            Complete Slurm configuration with defaults applied
        """
        # Convert tuple back to dict
        reqs = dict(requirements_tuple)
        
        # Start with executor-level defaults
        config: Dict[str, Any] = {}
        config.update(self.default_slurm_args)

        # Global config defaults
        global_defaults = dict(self._global_config.get('default_slurm_args', {}))
        default_partition = self._global_config.get('default_partition', 'general')

        # Project overrides
        project_overrides: Dict[str, Any] = {}
        if Project.name:
            try:
                _pconf = ConfigManager().load_project_config(Project.name).get('config', {})
                project_overrides = dict(_pconf.get('default_slurm_args', {}))
                default_partition = _pconf.get('default_partition', default_partition)
            except Exception:
                project_overrides = {}

        # Determine partition
        partition = reqs.get('partition', config.get('partition', default_partition))
        
        # Apply partition-specific defaults from global
        if isinstance(global_defaults, dict) and partition in global_defaults:
            part_defaults = global_defaults.get(partition, {})
            if isinstance(part_defaults, dict):
                config.update(part_defaults)

        # Apply project-level overrides
        if isinstance(project_overrides, dict):
            if partition in project_overrides and isinstance(project_overrides[partition], dict):
                config.update(project_overrides[partition])
            else:
                for k, v in project_overrides.items():
                    if isinstance(v, (str, int, float, bool)):
                        config[k] = v

        # Apply artifact-specific requirements
        config.update(reqs)

        # Apply CLI overrides (highest priority)
        if getattr(self, 'cli_slurm_overrides', None):
            config.update(self.cli_slurm_overrides)
        
        # Apply final defaults only if not set
        config.setdefault('partition', default_partition)
        config.setdefault('time', '2-00:00:00')
        
        # Set default cpus_per_task based on GPUs if not specified
        if 'cpus_per_task' not in config and 'gpus' in config:
            # Try to extract GPU count
            gpus_val = str(config['gpus'])
            if ':' in gpus_val:
                # Format like "A6000:4" - extract number after colon
                try:
                    gpu_count = int(gpus_val.split(':')[-1])
                    config['cpus_per_task'] = gpu_count
                except ValueError:
                    config['cpus_per_task'] = 1
            else:
                # Try to parse as integer
                try:
                    config['cpus_per_task'] = int(gpus_val)
                except ValueError:
                    # Single GPU name like "A6000"
                    config['cpus_per_task'] = 1
        
        # Normalize cpus -> cpus_per_task even if provided via defaults/overrides
        if 'cpus' in config and 'cpus_per_task' not in config:
            config['cpus_per_task'] = config.pop('cpus')
        elif 'cpus' in config and 'cpus_per_task' in config:
            config.pop('cpus')

        # Final fallback for cpus_per_task
        config.setdefault('cpus_per_task', 1)
        
        return config

    # ---- Running job tracking helpers ----
    def _get_active_jobs(self) -> Set[tuple]:
        """Return set of (job_id, array_index_or_None) for current user's squeue."""
        try:
            user = os.environ.get('USER', '')
            cmd = ["squeue", "-h"]
            if user:
                cmd += ["-u", user]
            cmd += ["-o", "%A %a %T"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            active: Set[tuple] = set()
            for line in result.stdout.splitlines():
                parts = line.strip().split(None, 2)
                if not parts:
                    continue
                job_id = parts[0]
                array_idx_str = parts[1] if len(parts) > 1 else "N/A"
                array_index = None if array_idx_str in ("N/A", "-1") else array_idx_str
                active.add((job_id, array_index))
            return active
        except Exception:
            return set()

    def _load_launched_jobs(self) -> Dict[str, Any]:
        try:
            if self.project:
                return ConfigManager().load_launched_jobs(self.project)
        except Exception:
            return {}
        return {}

    def _save_launched_jobs(self, mapping: Dict[str, Any]) -> None:
        try:
            if self.project:
                ConfigManager().save_launched_jobs(self.project, mapping)
        except Exception:
            pass

    def _prune_launched_jobs(self) -> None:
        """Remove entries from launched map that are no longer active."""
        if not self._launched_map:
            return
        pruned: Dict[str, Any] = {}
        for relpath, info in self._launched_map.items():
            job_id = str(info.get('job_id', ''))
            arr = info.get('array_index', None)
            key = (job_id, None if arr is None else str(arr))
            if key in self._active_jobs or (job_id, None) in self._active_jobs:
                pruned[relpath] = info
        self._launched_map = pruned
        if not self.dry_run and self.project:
            self._save_launched_jobs(self._launched_map)

    def _should_launch_task(self, task: Task) -> bool:
        if task.artifact is None:
            return True
        rel = task.artifact.relpath
        info = self._launched_map.get(rel)
        if not info:
            return True
        job_id = str(info.get('job_id', ''))
        arr = info.get('array_index', None)
        key_specific = (job_id, None if arr is None else str(arr))
        key_any = (job_id, None)
        return not (key_specific in self._active_jobs or key_any in self._active_jobs)

    def _build_sbatch_header(
        self,
        job_name: str,
        tasks: List[Task],
        config: Dict[str, Any],
        dependency_job_ids: List[str],
        jobs_limit: int | None = None,
    ) -> List[str]:
        """Build the #SBATCH header lines for the script.
        
        Args:
            job_name: Name for this job
            tasks: List of tasks in this job
            config: Slurm configuration dictionary
            dependency_job_ids: Job IDs that must complete before this job starts
            jobs_limit: If specified, combine tasks to create this many array indices
        """
        lines = ["#!/usr/bin/env bash"]
        
        # Job name
        lines.append(f"#SBATCH --job-name={job_name}")
        
        # Log file paths
        log_dir = self._global_config.get('log_directory', str(Path.home() / ".experiments" / "logs"))
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Use custom output/error paths if provided, otherwise use defaults
        if 'output' in config:
            lines.append(f"#SBATCH --output={config['output']}")
        else:
            output_file = f"{log_dir}/{job_name}_%A_%a.out"
            lines.append(f"#SBATCH --output={output_file}")
        
        if 'error' in config:
            lines.append(f"#SBATCH --error={config['error']}")
        elif config.get('separate_error', False):
            # Only add separate error file if explicitly requested
            error_file = f"{log_dir}/{job_name}_%A_%a.err"
            lines.append(f"#SBATCH --error={error_file}")
        
        # Working directory
        if 'chdir' in config:
            lines.append(f"#SBATCH --chdir={config['chdir']}")
        
        # Dependencies: wait for all previous tier jobs to complete successfully
        # Also include any external dependencies specified via CLI
        all_dependencies = list(dependency_job_ids)
        if hasattr(self, 'external_dependencies') and self.external_dependencies:
            all_dependencies.extend(self.external_dependencies)
        
        if all_dependencies:
            dependency_str = ":".join(all_dependencies)
            lines.append(f"#SBATCH --dependency=afterok:{dependency_str}")
        
        # Basic resource specifications
        if 'partition' in config:
            lines.append(f"#SBATCH --partition={config['partition']}")
        if 'time' in config:
            lines.append(f"#SBATCH --time={config['time']}")
        if 'account' in config:
            lines.append(f"#SBATCH --account={config['account']}")
        if 'qos' in config:
            lines.append(f"#SBATCH --qos={config['qos']}")
        
        # Node and task specifications
        if 'nodes' in config:
            lines.append(f"#SBATCH --nodes={config['nodes']}")
        if 'ntasks' in config:
            lines.append(f"#SBATCH --ntasks={config['ntasks']}")
        if 'cpus_per_task' in config:
            lines.append(f"#SBATCH --cpus-per-task={config['cpus_per_task']}")
        
        # Memory specifications
        if 'mem' in config:
            lines.append(f"#SBATCH --mem={config['mem']}")
        if 'mem_per_cpu' in config:
            lines.append(f"#SBATCH --mem-per-cpu={config['mem_per_cpu']}")
        
        # GPU handling
        # Support modern --gpus flag
        if 'gpus' in config:
            gpus_str = str(config['gpus'])
            lines.append(f"#SBATCH --gpus={gpus_str}")
        # Support --gres for older clusters
        if 'gres' in config:
            lines.append(f"#SBATCH --gres={config['gres']}")
        # Support constraint for specific hardware
        if 'constraint' in config:
            lines.append(f"#SBATCH --constraint={config['constraint']}")
        
        # Job control flags
        if config.get('requeue', False):
            lines.append(f"#SBATCH --requeue")
        if 'signal' in config:
            lines.append(f"#SBATCH --signal={config['signal']}")
        if 'open_mode' in config:
            lines.append(f"#SBATCH --open-mode={config['open_mode']}")
        
        # Email notifications
        if 'mail_type' in config:
            lines.append(f"#SBATCH --mail-type={config['mail_type']}")
        if 'mail_user' in config:
            lines.append(f"#SBATCH --mail-user={config['mail_user']}")
        
        # Array specification
        # If jobs_limit is specified and less than number of tasks, use it
        if jobs_limit is not None and jobs_limit > 0 and jobs_limit < len(tasks):
            max_index = jobs_limit - 1
        else:
            max_index = len(tasks) - 1
        lines.append(f"#SBATCH --array=0-{max_index}")
        
        return lines

    def _build_script_body(self, tier: List[Task], jobs_limit: int | None = None) -> List[str]:
        """Build the shell script body with case statement for array tasks.
        
        Args:
            tier: List of tasks to execute
            jobs_limit: If specified, combine tasks to create this many array indices
        """
        lines = [
            "set -euo pipefail",
        ]
        
        # Add setup commands if provided
        if self.setup_command:
            lines.append("")
            lines.append(self.setup_command)
            lines.append("")
        
        # Export project configuration for all tasks
        try:
            proj_conf = ConfigManager().load_project_config(Project.name or "").get("config", {})
        except Exception:
            proj_conf = {}
        proj_conf_str = _safe_json_dumps(proj_conf)
        lines.append(f"export EXPERIMENTS_PROJECT_CONF={dquote(proj_conf_str)}")
        lines.append("")
        
        lines.append("case \"${SLURM_ARRAY_TASK_ID:-0}\" in")
        
        # Redistribute tasks if jobs_limit is specified
        if jobs_limit is not None and jobs_limit > 0 and jobs_limit < len(tier):
            # Create jobs_limit groups, distribute tasks round-robin
            task_groups: List[List[Task]] = [[] for _ in range(jobs_limit)]
            for idx, task in enumerate(tier):
                group_idx = idx % jobs_limit
                task_groups[group_idx].append(task)
            
            # Generate a case for each group
            for group_idx, tasks in enumerate(task_groups):
                if not tasks:
                    continue
                lines.append(f"  {group_idx})")
                for task in tasks:
                    if task.artifact is not None:
                        exp_conf = _artifact_experiment_conf(task.artifact)
                        exp_conf_str = _safe_json_dumps(exp_conf)
                        lines.append(f"    export EXPERIMENTS_EXPERIMENT_CONF={dquote(exp_conf_str)}")
                    for block in task.blocks:
                        command = block.execute()
                        if command:
                            # Print the command before executing it (escape for double quotes)
                            escaped_command = command.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`')
                            lines.append(f'    echo "+ {escaped_command}" >&2')
                            lines.append(f"    {command}")
                lines.append("    ;;")
        else:
            # Original behavior: one case per task
            for idx, task in enumerate(tier):
                lines.append(f"  {idx})")
                if task.artifact is not None:
                    exp_conf = _artifact_experiment_conf(task.artifact)
                    exp_conf_str = _safe_json_dumps(exp_conf)
                    lines.append(f"    export EXPERIMENTS_EXPERIMENT_CONF={dquote(exp_conf_str)}")
                for block in task.blocks:
                    command = block.execute()
                    if command:
                        # Print the command before executing it (escape for double quotes)
                        escaped_command = command.replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`')
                        lines.append(f'    echo "+ {escaped_command}" >&2')
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
            print("\nNo jobs to submit.\n", file=sys.stderr)
            return
        
        print("\n" + "=" * 100, file=sys.stderr)
        print("DRY RUN SUMMARY", file=sys.stderr)
        print("=" * 100 + "\n", file=sys.stderr)
        
        # Overall statistics
        total_jobs = len(self._dry_run_jobs)
        total_tasks = sum(job['num_tasks'] for job in self._dry_run_jobs)
        total_complete = sum(job.get('num_complete', 0) for job in self._dry_run_jobs)
        total_remaining = sum(job.get('num_remaining', 0) for job in self._dry_run_jobs)
        
        print(f"Total jobs: {total_jobs}", file=sys.stderr)
        print(f"Total tasks: {total_tasks}", file=sys.stderr)
        print(f"Complete: {total_complete}, Remaining: {total_remaining}", file=sys.stderr)
        print(file=sys.stderr)
        
        # Count artifacts by class
        artifact_counts: Dict[str, int] = defaultdict(int)
        for job in self._dry_run_jobs:
            for artifact_class in job['artifact_classes']:
                artifact_counts[artifact_class] += 1
        
        if artifact_counts:
            print("Artifact types:", file=sys.stderr)
            for artifact_class, count in sorted(artifact_counts.items()):
                print(f"  {artifact_class}: {count} task(s)", file=sys.stderr)
            print(file=sys.stderr)
        
        # List jobs by tier
        jobs_by_tier: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for job in self._dry_run_jobs:
            jobs_by_tier[job['tier']].append(job)
        
        print("Job submission plan:", file=sys.stderr)
        print(file=sys.stderr)
        
        for tier_index in sorted(jobs_by_tier.keys()):
            tier_jobs = jobs_by_tier[tier_index]
            print(f"Tier {tier_index}:", file=sys.stderr)
            
            for job in tier_jobs:
                print(f"  Job {job['job_id']}: {job['job_name']}", file=sys.stderr)
                
                # Show stage if available
                if job.get('stage'):
                    print(f"    Stage: {job['stage']}", file=sys.stderr)
                
                print(f"    Tasks: {job['num_tasks']}", file=sys.stderr)
                
                # Show unique artifact types in this job
                unique_artifacts = sorted(set(job['artifact_classes']))
                if unique_artifacts:
                    print(f"    Artifacts: {', '.join(unique_artifacts)}", file=sys.stderr)
                
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
                    print(f"    Resources: {', '.join(resources)}", file=sys.stderr)
                
                # Show dependencies
                if job['dependencies']:
                    deps_str = ', '.join(job['dependencies'])
                    print(f"    Dependencies: {deps_str}", file=sys.stderr)
                else:
                    print(f"    Dependencies: None (first tier)", file=sys.stderr)
                
                print(f"    Script: {job['script_path']}", file=sys.stderr)
                print(file=sys.stderr)
        
        print("=" * 100, file=sys.stderr)
        print("To actually submit these jobs, run without the --dry flag", file=sys.stderr)
        print("=" * 100 + "\n", file=sys.stderr)
    
    def _print_launch_summary(self, job_ids: List[str]) -> None:
        """Print a brief summary of jobs that were actually launched."""
        if not job_ids:
            print("\nNo jobs were submitted.\n", file=sys.stderr)
            return
        
        print("=" * 80, file=sys.stderr)
        print("Launch Summary", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(file=sys.stderr)
        print(f"Total jobs submitted: {len(job_ids)}", file=sys.stderr)
        print(f"Job IDs: {', '.join(job_ids)}", file=sys.stderr)
        print(file=sys.stderr)
        print("Use 'python <script> history' to view full details", file=sys.stderr)
        print("Use 'python <script> cat <job_id>' to view logs", file=sys.stderr)
        print("=" * 80 + "\n", file=sys.stderr)


# ---- Helpers for JSON-safe serialization ----
def _is_json_scalar(x: Any) -> bool:
    return isinstance(x, (str, int, float, bool)) or x is None


def _json_sanitize(obj: Any) -> Any:
    if _is_json_scalar(obj):
        return obj
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            try:
                val = _json_sanitize(v)
                if val is not None:
                    out[str(k)] = val
            except Exception:
                continue
        return out
    if isinstance(obj, (list, tuple)):
        out_list: List[Any] = []
        for v in obj:
            try:
                val = _json_sanitize(v)
                out_list.append(val)
            except Exception:
                continue
        return out_list
    return None


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(_json_sanitize(obj), separators=(",", ":"))
    except Exception:
        return "{}"


def _artifact_experiment_conf(artifact: Artifact) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    try:
        data.update(artifact.as_dict())
    except Exception:
        data = {}
    try:
        for name, val in artifact.__class__.__dict__.items():
            if isinstance(val, property) and val.fget is not None:
                try:
                    data[name] = getattr(artifact, name)
                except Exception:
                    pass
    except Exception:
        pass
    try:
        data["relpath"] = artifact.relpath
    except Exception:
        pass
    return _json_sanitize(data)