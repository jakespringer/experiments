"""RUNLIB: Utility layer for experiments.

Provides configuration access, temporary workspaces, free port finding,
and high-performance Google Cloud Storage transfers with convenience I/O.
"""

import atexit
import json
import os
import shutil
import socket
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# Global caches
_project_config_cache: Optional[Dict[str, Any]] = None
_experiment_config_cache: Optional[Dict[str, Any]] = None
_gcs_available: Optional[bool] = None
_storage_module = None
_transfer_manager_module = None
_exceptions_module = None
_retry_module = None
_crc32c_module = None


class _ConfigView:
    """Read-only view of configuration dictionary."""
    
    def __init__(self, data: Dict[str, Any]):
        self._data = data
    
    def __getitem__(self, key: str) -> Any:
        return self._data[key]
    
    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __getattr__(self, name: str) -> Any:
        return self._data[name]

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_data",):
            object.__setattr__(self, name, value)
        else:
            raise AttributeError("Cannot set attributes on _ConfigView")
    
    def get(self, key: str, default=None) -> Any:
        return self._data.get(key, default)
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()
    
    def __repr__(self):
        return f"_ConfigView({self._data!r})"


# ============================================================================
# Configuration accessors
# ============================================================================

def _load_env_json(var_name: str) -> Dict[str, Any]:
    """Helper to read and parse JSON from env var.
    
    Args:
        var_name: Name of environment variable.
        
    Returns:
        Parsed JSON dictionary.
        
    Raises:
        RuntimeError: If env var is missing or contains invalid JSON.
    """
    value = os.environ.get(var_name)
    if value is None:
        raise RuntimeError(f"Environment variable {var_name} is not set")
    
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse {var_name} as JSON: {e}")


def get_project_config() -> _ConfigView:
    """Returns cached project-level config from EXPERIMENTS_PROJECT_CONF env var.
    
    Returns:
        Read-only view of project configuration.
        
    Raises:
        RuntimeError: If env var missing or invalid JSON.
    """
    global _project_config_cache
    if _project_config_cache is None:
        _project_config_cache = _load_env_json("EXPERIMENTS_PROJECT_CONF")
    return _ConfigView(_project_config_cache)


def get_experiment_config() -> _ConfigView:
    """Returns cached experiment-level config from EXPERIMENTS_EXPERIMENT_CONF env var.
    
    Returns:
        Read-only view of experiment configuration.
        
    Raises:
        RuntimeError: If env var missing or invalid JSON.
    """
    global _experiment_config_cache
    if _experiment_config_cache is None:
        _experiment_config_cache = _load_env_json("EXPERIMENTS_EXPERIMENT_CONF")
    return _ConfigView(_experiment_config_cache)


def get_relpath() -> str:
    """Returns the relpath attribute from experiment config.
    
    Returns:
        The relpath string.
        
    Raises:
        RuntimeError: If relpath is not present in experiment config.
    """
    config = get_experiment_config()
    if "relpath" not in config:
        raise RuntimeError("relpath not found in experiment configuration")
    return config["relpath"]


# ============================================================================
# Temporary workspace helpers
# ============================================================================

class TemporaryWorkspace:
    """Context manager for temporary workspace directories."""
    
    def __init__(self, path: Path):
        self.path = path
        self._removed = False
    
    def cleanup(self) -> None:
        """Recursively removes the directory if not already removed; idempotent."""
        if not self._removed and self.path.exists():
            shutil.rmtree(self.path)
            self._removed = True
    
    def __enter__(self) -> Path:
        """Returns path to support with statement."""
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensures cleanup on context exit."""
        self.cleanup()


def _choose_tmp_root(filesystem: Optional[Union[str, Path]]) -> Path:
    """Chooses base temp root.
    
    Args:
        filesystem: Explicit directory to use, or None for automatic selection.
        
    Returns:
        Path to use as temporary root.
    """
    if filesystem is not None:
        path = Path(filesystem)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    # Try /dev/shm if writable
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK):
        return shm
    
    # Fall back to system temp
    import tempfile
    return Path(tempfile.gettempdir())


def make_temp_dir(prefix: str = "exp_", filesystem: Optional[Union[str, Path]] = None) -> Path:
    """Creates a temporary directory under a preferred root.
    
    Args:
        prefix: Prefix for directory name.
        filesystem: Explicit directory to use, or None for automatic selection.
        
    Returns:
        Path to created directory.
    """
    import tempfile
    root = _choose_tmp_root(filesystem)
    temp_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    
    # Register cleanup at exit
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
    
    return temp_dir


@contextmanager
def temporary_workspace(prefix: str = "exp_", filesystem: Optional[Union[str, Path]] = None) -> Iterator[Path]:
    """Context manager producing a temporary directory that is always cleaned up on exit.
    
    Args:
        prefix: Prefix for directory name.
        filesystem: Explicit directory to use, or None for automatic selection.
        
    Yields:
        Path to temporary directory.
    """
    workspace = TemporaryWorkspace(make_temp_dir(prefix=prefix, filesystem=filesystem))
    try:
        yield workspace.path
    finally:
        workspace.cleanup()


# ============================================================================
# Networking helper
# ============================================================================

def get_free_port() -> int:
    """Returns an available TCP port on 127.0.0.1.
    
    Returns:
        Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# ============================================================================
# GCS utility internals
# ============================================================================

def _require_gcs() -> None:
    """Lazily imports google-cloud-storage and related modules.
    
    Raises:
        RuntimeError: If google-cloud-storage is not available.
    """
    global _gcs_available, _storage_module, _transfer_manager_module
    global _exceptions_module, _retry_module, _crc32c_module
    
    if _gcs_available is not None:
        if not _gcs_available:
            raise RuntimeError(
                "google-cloud-storage is not available. "
                "Install with: pip install google-cloud-storage"
            )
        return
    
    try:
        from google.cloud import storage
        from google.cloud.storage import transfer_manager
        _storage_module = storage
        _transfer_manager_module = transfer_manager
        
        try:
            from google.api_core import exceptions as api_exceptions
            from google.api_core import retry as api_retry
            _exceptions_module = api_exceptions
            _retry_module = api_retry
        except ImportError:
            _exceptions_module = None
            _retry_module = None
        
        try:
            import google_crc32c
            _crc32c_module = google_crc32c
        except ImportError:
            _crc32c_module = None
        
        _gcs_available = True
    except ImportError:
        _gcs_available = False
        raise RuntimeError(
            "google-cloud-storage is not available. "
            "Install with: pip install google-cloud-storage"
        )


def _get_storage_client():
    """Ensures GCS libs present and returns a storage.Client().
    
    Returns:
        A google.cloud.storage.Client instance.
    """
    _require_gcs()
    return _storage_module.Client()


def _get_gcs_exceptions():
    """Returns a tuple of retryable google.api_core.exceptions classes.
    
    Returns:
        Tuple of exception classes, or (Exception,) if unavailable.
    """
    _require_gcs()
    if _exceptions_module is None:
        return (Exception,)
    
    return (
        _exceptions_module.TooManyRequests,
        _exceptions_module.ServiceUnavailable,
        _exceptions_module.InternalServerError,
        _exceptions_module.RequestRangeNotSatisfiable,
    )


def _build_gcs_retry(attempts: int):
    """Builds a google-api-core Retry object.
    
    Args:
        attempts: Number of retry attempts.
        
    Returns:
        Retry object or None if unavailable or attempts <= 0.
    """
    _require_gcs()
    if _retry_module is None or attempts <= 0:
        return None
    
    exceptions = _get_gcs_exceptions()
    predicate = lambda exc: isinstance(exc, exceptions)
    
    return _retry_module.Retry(
        predicate=predicate,
        initial=1.0,
        maximum=32.0,
        multiplier=2.0,
        deadline=None,
    )


def _parse_gs_path(gs_url: str) -> Tuple[str, str]:
    """Splits gs://bucket/key... into (bucket, key).
    
    Args:
        gs_url: GCS URL starting with gs://.
        
    Returns:
        Tuple of (bucket_name, key).
        
    Raises:
        ValueError: If not a gs:// URL.
    """
    if not gs_url.startswith("gs://"):
        raise ValueError(f"Not a gs:// URL: {gs_url}")
    
    path = gs_url[5:]  # Remove gs://
    if "/" in path:
        bucket, key = path.split("/", 1)
    else:
        bucket = path
        key = ""
    
    return bucket, key


def _is_gs_path(path: str) -> bool:
    """True if path starts with gs://.
    
    Args:
        path: Path to check.
        
    Returns:
        True if path is a GCS path.
    """
    return path.startswith("gs://")


def _ensure_dir(path: Union[str, Path]) -> None:
    """mkdir -p equivalent.
    
    Args:
        path: Directory path to create.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def _sleep(seconds: float) -> None:
    """Thin wrapper around time.sleep.
    
    Args:
        seconds: Number of seconds to sleep.
    """
    time.sleep(seconds)


def _with_retries(fn, retry: int) -> Any:
    """Executes fn with up to retry + 1 attempts.
    
    Args:
        fn: Callable to execute.
        retry: Number of retries (0 means no retries).
        
    Returns:
        Result of fn.
        
    Raises:
        Last exception encountered.
    """
    exceptions = _get_gcs_exceptions()
    last_error = None
    
    for attempt in range(retry + 1):
        try:
            return fn()
        except exceptions as e:
            last_error = e
            if attempt < retry:
                _sleep(min(2 ** attempt, 32))
            else:
                raise
    
    if last_error:
        raise last_error


def _infer_remote_is_dir(bucket, key: str) -> bool:
    """Heuristics to determine if remote key is a directory.
    
    Args:
        bucket: GCS bucket object.
        key: Object key.
        
    Returns:
        True if key appears to be a directory.
    """
    # Explicit trailing / or /.
    if key.endswith("/") or key.endswith("/."):
        return True
    
    # Check if exact blob exists
    blob = bucket.blob(key)
    if blob.exists():
        return False
    
    # Check if any blobs exist with this prefix
    prefix = key if key.endswith("/") else key + "/"
    blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
    return len(blobs) > 0


def _directory_flag_for_local(path, directory: Union[str, bool]) -> bool:
    """Resolves whether path should be treated as a directory.
    
    Args:
        path: Local path.
        directory: "auto", True, or False.
        
    Returns:
        True if path should be treated as directory.
    """
    if directory == "auto":
        path_str = str(path)
        if path_str.endswith("/*"):
            return True
        return Path(path_str.rstrip("/*")).is_dir()
    return bool(directory)


def _directory_flag_for_remote(bucket, key: str, directory: Union[str, bool]) -> bool:
    """Resolves whether remote is a directory.
    
    Args:
        bucket: GCS bucket object.
        key: Object key.
        directory: "auto", True, or False.
        
    Returns:
        True if remote should be treated as directory.
    """
    if directory == "auto":
        return _infer_remote_is_dir(bucket, key)
    return bool(directory)


def _parse_local_contents_spec(path: Union[str, Path]) -> Tuple[Path, bool]:
    """Returns (base_path, contents_only) where trailing /* encodes contents-only.
    
    Args:
        path: Local path possibly ending with /*.
        
    Returns:
        Tuple of (base_path, contents_only).
    """
    path_str = str(path)
    if path_str.endswith("/*"):
        return Path(path_str[:-2]), True
    return Path(path_str), False


def _parse_remote_contents_spec(remote_path: str) -> Tuple[str, str, bool]:
    """Returns (bucket, key, contents_only) where trailing /. encodes contents-only.
    
    Args:
        remote_path: GCS path possibly ending with /.
        
    Returns:
        Tuple of (bucket, key, contents_only).
    """
    bucket, key = _parse_gs_path(remote_path)
    
    if key.endswith("/."):
        return bucket, key[:-2], True
    
    return bucket, key, False


def _list_remote_tree(client, bucket, prefix: str, retry_obj=None) -> Dict[str, Any]:
    """Lists all blobs under prefix into a name->blob mapping.
    
    Args:
        client: Storage client.
        bucket: Bucket object.
        prefix: Prefix to list under.
        retry_obj: Optional retry object.
        
    Returns:
        Dictionary mapping blob names to blob objects.
    """
    blobs = {}
    for blob in bucket.list_blobs(prefix=prefix, retry=retry_obj):
        blobs[blob.name] = blob
    return blobs


def _crc32c_local(path: Path) -> Optional[str]:
    """Computes base64-encoded CRC32C for local file.
    
    Args:
        path: Path to local file.
        
    Returns:
        Base64-encoded CRC32C or None if unavailable.
    """
    if _crc32c_module is None:
        return None
    
    import base64
    
    hasher = _crc32c_module.Checksum()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    
    return base64.b64encode(hasher.digest()).decode("ascii")


# ============================================================================
# Transfer operations
# ============================================================================

def download_from_gs(
    remote_path: str,
    local_path: Union[str, Path],
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8
) -> None:
    """Downloads a single file or a directory tree from GCS to local filesystem.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        local_path: Local destination path.
        directory: "auto", True, or False to indicate directory semantics.
        concurrent: Use concurrent transfers (may be no-op).
        ensure_contents: If True, places files directly under local_path.
        retry: Number of retries for transient failures.
        max_workers: Parallelism for transfer manager.
    """
    _require_gcs()
    
    def _download():
        client = _get_storage_client()
        bucket_name, key, contents_only = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)
        
        retry_obj = _build_gcs_retry(retry)
        
        # Determine if remote is a directory
        is_dir = _directory_flag_for_remote(bucket, key, directory)
        
        if not is_dir:
            # Single file download
            local_file = Path(local_path)
            _ensure_dir(local_file.parent)
            
            blob = bucket.blob(key)
            blob.download_to_filename(str(local_file), retry=retry_obj)
        else:
            # Directory download
            prefix = key if key.endswith("/") else key + "/"
            if key and not key.endswith("/") and not contents_only:
                prefix = key + "/"
            
            blobs = _list_remote_tree(client, bucket, prefix, retry_obj)
            
            if not blobs:
                return
            
            # Determine destination root
            local_root = Path(local_path)
            
            if not ensure_contents and not contents_only:
                # Nest under remote leaf name
                leaf = key.rstrip("/").split("/")[-1] if key else ""
                if leaf:
                    local_root = local_root / leaf
            
            _ensure_dir(local_root)
            
            # Prepare blob names and destination paths
            blob_names = []
            for blob_name in blobs:
                relative = blob_name[len(prefix):] if blob_name.startswith(prefix) else blob_name
                if relative:
                    blob_names.append(blob_name)
            
            if not blob_names:
                return
            
            # Use transfer manager for concurrent download
            results = _transfer_manager_module.download_many_to_path(
                bucket,
                blob_names,
                destination_directory=str(local_root),
                max_workers=max_workers
            )
            
            # Check for errors
            for name, result in zip(blob_names, results):
                if isinstance(result, Exception):
                    raise result
    
    _with_retries(_download, retry)


def upload_to_gs(
    local_path: Union[str, Path],
    remote_path: str,
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8
) -> None:
    """Uploads a single file or a directory tree from local filesystem to GCS.
    
    Args:
        local_path: Local source path.
        remote_path: GCS destination path (gs://bucket/key).
        directory: "auto", True, or False to indicate directory semantics.
        concurrent: Use concurrent transfers (may be no-op).
        ensure_contents: If True, uploads directory contents under remote_path.
        retry: Number of retries for transient failures.
        max_workers: Parallelism for transfer manager.
    """
    _require_gcs()
    
    def _upload():
        client = _get_storage_client()
        bucket_name, key, contents_only = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)
        
        retry_obj = _build_gcs_retry(retry)
        
        local_base, local_contents_only = _parse_local_contents_spec(local_path)
        
        # Determine if local is a directory
        is_dir = _directory_flag_for_local(local_base, directory)
        
        if not is_dir:
            # Single file upload
            blob = bucket.blob(key)
            blob.upload_from_filename(str(local_base), retry=retry_obj)
        else:
            # Directory upload
            files_to_upload = []
            
            for item in local_base.rglob("*"):
                if item.is_file():
                    relative = item.relative_to(local_base)
                    files_to_upload.append((str(item), str(relative)))
            
            if not files_to_upload:
                return
            
            # Determine remote prefix
            remote_prefix = key
            if not ensure_contents and not contents_only and not local_contents_only:
                # Nest under local base name
                leaf = local_base.name
                remote_prefix = f"{key}/{leaf}" if key else leaf
            
            if remote_prefix and not remote_prefix.endswith("/"):
                remote_prefix += "/"
            
            # Build mapping of source files to destination blob names
            blob_names = []
            source_files = []
            
            for source_file, relative in files_to_upload:
                blob_name = remote_prefix + str(relative).replace("\\", "/")
                blob_names.append(blob_name)
                source_files.append(source_file)
            
            # Create staging directory with proper structure
            import tempfile
            with tempfile.TemporaryDirectory() as staging_dir:
                staging_path = Path(staging_dir)
                
                for source_file, blob_name in zip(source_files, blob_names):
                    # Extract relative path from blob_name
                    rel_path = blob_name[len(remote_prefix):] if blob_name.startswith(remote_prefix) else blob_name
                    dest = staging_path / rel_path
                    _ensure_dir(dest.parent)
                    
                    # Try hard link, fall back to symlink, then copy
                    try:
                        os.link(source_file, dest)
                    except (OSError, NotImplementedError):
                        try:
                            os.symlink(source_file, dest)
                        except (OSError, NotImplementedError):
                            shutil.copy2(source_file, dest)
                
                # Prepare filenames for transfer manager
                filenames = [str(staging_path / blob_name[len(remote_prefix):]) 
                            for blob_name in blob_names]
                
                # Upload using transfer manager
                results = _transfer_manager_module.upload_many_from_filenames(
                    bucket,
                    filenames,
                    source_directory=str(staging_path),
                    max_workers=max_workers,
                    blob_name_prefix=remote_prefix
                )
                
                # Check for errors
                for name, result in zip(filenames, results):
                    if isinstance(result, Exception):
                        raise result
    
    _with_retries(_upload, retry)


def sync_to_gs(
    local_path: Union[str, Path],
    remote_path: str,
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    delete: bool = False,
    checksum: bool = False,
    retry: int = 0,
    max_workers: int = 8
) -> None:
    """Synchronizes a local directory or single file to GCS.
    
    Args:
        local_path: Local source path.
        remote_path: GCS destination path (gs://bucket/key).
        directory: "auto", True, or False to indicate directory semantics.
        concurrent: Use concurrent transfers (may be no-op).
        ensure_contents: If True, syncs directory contents under remote_path.
        delete: If True, removes remote blobs not present locally.
        checksum: If True, uses CRC32C for comparison instead of size.
        retry: Number of retries for transient failures.
        max_workers: Parallelism for transfer manager.
    """
    _require_gcs()
    
    def _sync():
        client = _get_storage_client()
        bucket_name, key, contents_only = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)
        
        retry_obj = _build_gcs_retry(retry)
        
        local_base, local_contents_only = _parse_local_contents_spec(local_path)
        
        # Determine if local is a directory
        is_dir = _directory_flag_for_local(local_base, directory)
        
        if not is_dir:
            # Single file sync - just upload
            blob = bucket.blob(key)
            blob.upload_from_filename(str(local_base), retry=retry_obj)
            return
        
        # Directory sync
        local_files = {}
        for item in local_base.rglob("*"):
            if item.is_file():
                relative = item.relative_to(local_base)
                local_files[str(relative)] = item
        
        # Determine remote prefix
        remote_prefix = key
        if not ensure_contents and not contents_only and not local_contents_only:
            leaf = local_base.name
            remote_prefix = f"{key}/{leaf}" if key else leaf
        
        if remote_prefix and not remote_prefix.endswith("/"):
            remote_prefix += "/"
        
        # List remote blobs
        remote_blobs = _list_remote_tree(client, bucket, remote_prefix, retry_obj)
        
        # Determine files to upload
        to_upload = []
        
        for relative_path, local_file in local_files.items():
            blob_name = remote_prefix + str(relative_path).replace("\\", "/")
            
            if blob_name not in remote_blobs:
                # New file
                to_upload.append((str(local_file), blob_name))
            else:
                # Check if changed
                blob = remote_blobs[blob_name]
                local_size = local_file.stat().st_size
                
                if checksum:
                    local_crc = _crc32c_local(local_file)
                    if local_crc and hasattr(blob, "crc32c") and blob.crc32c:
                        if local_crc != blob.crc32c:
                            to_upload.append((str(local_file), blob_name))
                    elif local_size != blob.size:
                        to_upload.append((str(local_file), blob_name))
                else:
                    if local_size != blob.size:
                        to_upload.append((str(local_file), blob_name))
        
        # Upload changed files
        if to_upload:
            for local_file, blob_name in to_upload:
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file, retry=retry_obj)
        
        # Delete remote files not in local
        if delete:
            local_blob_names = {
                remote_prefix + str(rel).replace("\\", "/")
                for rel in local_files.keys()
            }
            
            for blob_name in remote_blobs:
                if blob_name not in local_blob_names:
                    blob = bucket.blob(blob_name)
                    blob.delete(retry=retry_obj)
    
    _with_retries(_sync, retry)


def sync_from_gs(
    remote_path: str,
    local_path: Union[str, Path],
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    delete: bool = False,
    checksum: bool = False,
    retry: int = 0,
    max_workers: int = 8
) -> None:
    """Synchronizes from GCS to local directory.
    
    Args:
        remote_path: GCS source path (gs://bucket/key).
        local_path: Local destination path.
        directory: "auto", True, or False to indicate directory semantics.
        concurrent: Use concurrent transfers (may be no-op).
        ensure_contents: If True, syncs contents directly under local_path.
        delete: If True, removes local files not present remotely.
        checksum: If True, uses CRC32C for comparison instead of size.
        retry: Number of retries for transient failures.
        max_workers: Parallelism for transfer manager.
    """
    _require_gcs()
    
    def _sync():
        client = _get_storage_client()
        bucket_name, key, contents_only = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)
        
        retry_obj = _build_gcs_retry(retry)
        
        # Determine if remote is a directory
        is_dir = _directory_flag_for_remote(bucket, key, directory)
        
        if not is_dir:
            # Single file sync
            local_file = Path(local_path)
            _ensure_dir(local_file.parent)
            
            blob = bucket.blob(key)
            blob.download_to_filename(str(local_file), retry=retry_obj)
            return
        
        # Directory sync
        prefix = key if key.endswith("/") else key + "/"
        remote_blobs = _list_remote_tree(client, bucket, prefix, retry_obj)
        
        # Determine local root
        local_root = Path(local_path)
        if not ensure_contents and not contents_only:
            leaf = key.rstrip("/").split("/")[-1] if key else ""
            if leaf:
                local_root = local_root / leaf
        
        _ensure_dir(local_root)
        
        # Build mapping of remote to local
        to_download = []
        
        for blob_name, blob in remote_blobs.items():
            relative = blob_name[len(prefix):] if blob_name.startswith(prefix) else blob_name
            if not relative:
                continue
            
            local_file = local_root / relative
            
            if not local_file.exists():
                to_download.append((blob_name, local_file))
            else:
                # Check if changed
                local_size = local_file.stat().st_size
                
                if checksum:
                    local_crc = _crc32c_local(local_file)
                    if local_crc and hasattr(blob, "crc32c") and blob.crc32c:
                        if local_crc != blob.crc32c:
                            to_download.append((blob_name, local_file))
                    elif local_size != blob.size:
                        to_download.append((blob_name, local_file))
                else:
                    if local_size != blob.size:
                        to_download.append((blob_name, local_file))
        
        # Download changed files
        if to_download:
            for blob_name, local_file in to_download:
                _ensure_dir(local_file.parent)
                blob = bucket.blob(blob_name)
                blob.download_to_filename(str(local_file), retry=retry_obj)
        
        # Delete local files not in remote
        if delete:
            remote_relatives = {
                blob_name[len(prefix):] if blob_name.startswith(prefix) else blob_name
                for blob_name in remote_blobs.keys()
            }
            
            for item in local_root.rglob("*"):
                if item.is_file():
                    relative = str(item.relative_to(local_root))
                    if relative not in remote_relatives:
                        item.unlink()
    
    _with_retries(_sync, retry)


def pop_from_gs(
    remote_path: str,
    local_path: Union[str, Path],
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8
) -> None:
    """Downloads from GCS and then deletes the remote object(s).
    
    Args:
        remote_path: GCS source path (gs://bucket/key).
        local_path: Local destination path.
        directory: "auto", True, or False to indicate directory semantics.
        concurrent: Use concurrent transfers (may be no-op).
        ensure_contents: If True, places files directly under local_path.
        retry: Number of retries for transient failures.
        max_workers: Parallelism for transfer manager.
    """
    # Download first
    download_from_gs(
        remote_path, local_path, directory, concurrent,
        ensure_contents, retry, max_workers
    )
    
    # Then delete
    _require_gcs()
    
    def _delete():
        client = _get_storage_client()
        bucket_name, key, _ = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)
        
        retry_obj = _build_gcs_retry(retry)
        
        # Determine if remote is a directory
        is_dir = _directory_flag_for_remote(bucket, key, directory)
        
        if not is_dir:
            blob = bucket.blob(key)
            blob.delete(retry=retry_obj)
        else:
            # Delete all blobs under prefix
            prefix = key if key.endswith("/") else key + "/"
            blobs = list(bucket.list_blobs(prefix=prefix, retry=retry_obj))
            
            for blob in blobs:
                blob.delete(retry=retry_obj)
    
    _with_retries(_delete, retry)


def push_to_gs(
    local_path: Union[str, Path],
    remote_path: str,
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8
) -> None:
    """Uploads to GCS and then deletes the local source.
    
    Args:
        local_path: Local source path.
        remote_path: GCS destination path (gs://bucket/key).
        directory: "auto", True, or False to indicate directory semantics.
        concurrent: Use concurrent transfers (may be no-op).
        ensure_contents: If True, uploads directory contents under remote_path.
        retry: Number of retries for transient failures.
        max_workers: Parallelism for transfer manager.
    """
    # Upload first
    upload_to_gs(
        local_path, remote_path, directory, concurrent,
        ensure_contents, retry, max_workers
    )
    
    # Then delete local
    local_base, contents_only = _parse_local_contents_spec(local_path)
    
    if contents_only:
        # Only delete contents, preserve directory
        for item in local_base.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    else:
        # Delete entire path
        if local_base.is_file():
            local_base.unlink()
        elif local_base.is_dir():
            shutil.rmtree(local_base)


def check_exists_gs(
    remote: Union[str, Sequence[str]],
    max_ancestors: Optional[int] = None
) -> Union[bool, Dict[str, bool]]:
    """Existence checks for GCS paths.
    
    Args:
        remote: Single GCS path or sequence of paths.
        max_ancestors: Optional limit for grouping prefix depth.
        
    Returns:
        Boolean if single string, dict mapping path to boolean if sequence.
    """
    _require_gcs()
    
    client = _get_storage_client()
    
    if isinstance(remote, str):
        # Single path check
        bucket_name, key, contents_only = _parse_remote_contents_spec(remote)
        bucket = client.bucket(bucket_name)
        
        if contents_only or key.endswith("/"):
            # Check for any children
            prefix = key.rstrip("/") + "/" if key else ""
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            return len(blobs) > 0
        else:
            # Check exact blob or children under key/
            blob = bucket.blob(key)
            if blob.exists():
                return True
            
            # Check for children
            prefix = key + "/"
            blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
            return len(blobs) > 0
    else:
        # Multiple paths - group by bucket and prefix
        results = {}
        
        # Group by bucket
        by_bucket = {}
        for path in remote:
            bucket_name, key, contents_only = _parse_remote_contents_spec(path)
            if bucket_name not in by_bucket:
                by_bucket[bucket_name] = []
            by_bucket[bucket_name].append((path, key, contents_only))
        
        # Check each bucket's paths
        for bucket_name, paths in by_bucket.items():
            bucket = client.bucket(bucket_name)
            
            # For simplicity, check each path individually
            # A more optimized implementation would group by prefix
            for path, key, contents_only in paths:
                if contents_only or key.endswith("/"):
                    prefix = key.rstrip("/") + "/" if key else ""
                    blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
                    results[path] = len(blobs) > 0
                else:
                    blob = bucket.blob(key)
                    if blob.exists():
                        results[path] = True
                    else:
                        prefix = key + "/"
                        blobs = list(bucket.list_blobs(prefix=prefix, max_results=1))
                        results[path] = len(blobs) > 0
        
        return results


# ============================================================================
# Convenience I/O, listing, and deletion
# ============================================================================

def gs_read_text(remote_path: str, encoding: str = "utf-8", retry: int = 0) -> str:
    """Downloads blob content as text.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        encoding: Text encoding.
        retry: Number of retries for transient failures.
        
    Returns:
        Blob content as string.
    """
    _require_gcs()
    
    def _read():
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        retry_obj = _build_gcs_retry(retry)
        content = blob.download_as_bytes(retry=retry_obj)
        return content.decode(encoding)
    
    return _with_retries(_read, retry)


def gs_read_bytes(remote_path: str, retry: int = 0) -> bytes:
    """Downloads blob content as bytes.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        retry: Number of retries for transient failures.
        
    Returns:
        Blob content as bytes.
    """
    _require_gcs()
    
    def _read():
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        retry_obj = _build_gcs_retry(retry)
        return blob.download_as_bytes(retry=retry_obj)
    
    return _with_retries(_read, retry)


def gs_write_text(remote_path: str, text: str, encoding: str = "utf-8", retry: int = 0) -> None:
    """Uploads text as a blob.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        text: Text content to upload.
        encoding: Text encoding.
        retry: Number of retries for transient failures.
    """
    _require_gcs()
    
    def _write():
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        retry_obj = _build_gcs_retry(retry)
        blob.upload_from_string(
            text,
            content_type=f"text/plain; charset={encoding}",
            retry=retry_obj
        )
    
    _with_retries(_write, retry)


def gs_write_bytes(remote_path: str, data: bytes, content_type: str = "application/octet-stream", retry: int = 0) -> None:
    """Uploads bytes as a blob.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        data: Bytes content to upload.
        content_type: MIME content type.
        retry: Number of retries for transient failures.
    """
    _require_gcs()
    
    def _write():
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        retry_obj = _build_gcs_retry(retry)
        blob.upload_from_string(data, content_type=content_type, retry=retry_obj)
    
    _with_retries(_write, retry)


@contextmanager
def gs_open(remote_path: str, mode: str = "rb", retry: int = 0):
    """Context manager that opens a blob for streaming I/O.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        mode: Open mode (e.g., 'rb', 'wb', 'r', 'w').
        retry: Number of retries for transient failures.
        
    Yields:
        File-like object for blob I/O.
    """
    _require_gcs()
    
    def _open():
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(key)
        
        return blob.open(mode)
    
    file_obj = _with_retries(_open, retry)
    try:
        yield file_obj
    finally:
        file_obj.close()


def gs_list(
    remote_prefix: str,
    recursive: bool = True,
    delimiter: Optional[str] = None,
    max_results: Optional[int] = None
) -> List[str]:
    """Lists blob names under a prefix.
    
    Args:
        remote_prefix: GCS path prefix (gs://bucket/prefix).
        recursive: If True, lists recursively; if False and delimiter is None, uses '/'.
        delimiter: Optional delimiter for non-recursive listing.
        max_results: Optional limit on number of results.
        
    Returns:
        List of blob names (without bucket prefix).
    """
    _require_gcs()
    
    client = _get_storage_client()
    bucket_name, prefix = _parse_gs_path(remote_prefix)
    bucket = client.bucket(bucket_name)
    
    if not recursive and delimiter is None:
        delimiter = "/"
    
    blobs = bucket.list_blobs(
        prefix=prefix,
        delimiter=delimiter,
        max_results=max_results
    )
    
    return [blob.name for blob in blobs]


def gs_delete(remote_path: str, recursive: bool = False) -> None:
    """Deletes a single blob or all blobs under a prefix.
    
    Args:
        remote_path: GCS path (gs://bucket/key).
        recursive: If True or path ends with /, deletes all blobs under prefix.
    """
    _require_gcs()
    
    client = _get_storage_client()
    bucket_name, key = _parse_gs_path(remote_path)
    bucket = client.bucket(bucket_name)
    
    if recursive or key.endswith("/"):
        # Delete all blobs under prefix
        prefix = key.rstrip("/") + "/" if key and not key.endswith("/") else key
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        for blob in blobs:
            blob.delete()
    else:
        # Delete single blob
        blob = bucket.blob(key)
        blob.delete()

