# Specification: `experiments/runlib.py`

## Purpose
Runtime utility layer for experiments. Provides configuration access (via environment variables), temporary workspace management, free port discovery, and high-performance Google Cloud Storage (GCS) transfers with convenience I/O functions. This module is used at runtime by experiment tasks — it reads configuration injected by the executor into environment variables.

## Dependencies
- `atexit`, `json`, `os`, `shutil`, `socket`, `time`
- `contextlib.contextmanager`, `pathlib.Path`
- `typing` (various)
- **Optional**: `google.cloud.storage`, `google.cloud.storage.transfer_manager`, `google.api_core.exceptions`, `google.api_core.retry`, `google_crc32c`

---

## Module-Level Global Caches
- `_project_config_cache`: Cached parsed project config dict (from env var).
- `_experiment_config_cache`: Cached parsed experiment config dict (from env var).
- `_gcs_available`: `None` (unchecked), `True`, or `False` — GCS availability flag.
- `_storage_module`, `_transfer_manager_module`, `_exceptions_module`, `_retry_module`, `_crc32c_module`: Lazily imported GCS library modules.

---

## Class: `_ConfigView`

### Purpose
Read-only wrapper around a configuration dictionary. Prevents accidental mutation while providing dict-like and attribute-style access.

### Constructor: `__init__(self, data: Dict[str, Any])`
- Stores `data` as `self._data` via `object.__setattr__`.

### Access Methods
- `__getitem__(key)` → `self._data[key]`
- `__contains__(key)` → `key in self._data`
- `__getattr__(name)` → `self._data[name]` (attribute-style access)
- `__setattr__(name, value)` → raises `AttributeError` for all names except `"_data"`.
- `get(key, default=None)` → `self._data.get(key, default)`
- `keys()`, `values()`, `items()` → delegate to `self._data`.
- `__repr__()` → `f"_ConfigView({self._data!r})"`.

---

## Configuration Accessors

### `_load_env_json(var_name: str) -> Dict[str, Any]`
- Reads environment variable `var_name` and parses as JSON.
- Raises `RuntimeError` if variable is missing or JSON is invalid.

### `get_project_config() -> _ConfigView`
- Returns cached `_ConfigView` of project config from `EXPERIMENTS_PROJECT_CONF` env var.
- Parses JSON on first call, caches thereafter.
- Raises `RuntimeError` if env var missing or invalid.

### `get_experiment_config() -> _ConfigView`
- Returns cached `_ConfigView` of experiment config from `EXPERIMENTS_EXPERIMENT_CONF` env var.
- Same caching and error behavior as `get_project_config`.

### `get_relpath() -> str`
- Returns `config["relpath"]` from experiment config.
- Raises `RuntimeError` if `relpath` key not present.

---

## Temporary Workspace Helpers

### Class: `TemporaryWorkspace`

#### Constructor: `__init__(self, path: Path)`
- Stores `self.path = path` and `self._removed = False`.

#### Method: `cleanup(self) -> None`
- If not already removed and path exists: runs `shutil.rmtree(self.path)` and sets `_removed = True`.
- Idempotent — safe to call multiple times.

#### Context Manager Protocol
- `__enter__` → returns `self.path`.
- `__exit__` → calls `self.cleanup()`.

### `_choose_tmp_root(filesystem: Optional[Union[str, Path]]) -> Path`
- If `filesystem` is provided: creates it with `mkdir(parents=True, exist_ok=True)` and returns it.
- Otherwise: tries `/dev/shm` if writable, falls back to `tempfile.gettempdir()`.

### `make_temp_dir(prefix: str = "exp_", filesystem=None) -> Path`
- Creates a temporary directory under `_choose_tmp_root(filesystem)` using `tempfile.mkdtemp`.
- Registers `shutil.rmtree` cleanup via `atexit`.
- Returns the `Path` to the created directory.

### `temporary_workspace(prefix: str = "exp_", filesystem=None) -> Iterator[Path]`
- Context manager that creates a `TemporaryWorkspace` wrapping `make_temp_dir(...)`.
- Yields the workspace path.
- Guarantees cleanup in `finally` block via `workspace.cleanup()`.

---

## Networking Helper

### `get_free_port() -> int`
- Binds a TCP socket to `127.0.0.1:0`, retrieves the assigned port, closes the socket.
- Returns the port number.

---

## GCS Utility Internals

### `_require_gcs() -> None`
- Lazily imports `google.cloud.storage` and related modules.
- Sets `_gcs_available = True` on success, `False` on failure.
- On subsequent calls: returns immediately if available, raises `RuntimeError` if not.
- Also attempts to import `google.api_core.exceptions`, `google.api_core.retry`, and `google_crc32c` (all optional).

### `_get_storage_client()`
- Calls `_require_gcs()` and returns `storage.Client()`.

### `_get_gcs_exceptions() -> Tuple`
- Returns tuple of retryable exception classes: `(TooManyRequests, ServiceUnavailable, InternalServerError, RequestRangeNotSatisfiable)`.
- Falls back to `(Exception,)` if `google.api_core.exceptions` not available.

### `_build_gcs_retry(attempts: int)`
- Builds a `google.api_core.retry.Retry` object with exponential backoff (initial=1.0, maximum=32.0, multiplier=2.0).
- Returns `None` if retry module unavailable or `attempts <= 0`.

### `_parse_gs_path(gs_url: str) -> Tuple[str, str]`
- Splits `gs://bucket/key` into `(bucket, key)`.
- Raises `ValueError` if URL doesn't start with `gs://`.

### `_is_gs_path(path: str) -> bool`
- Returns `True` if `path` starts with `gs://`.

### `_ensure_dir(path: Union[str, Path]) -> None`
- `mkdir -p` equivalent: `Path(path).mkdir(parents=True, exist_ok=True)`.

### `_sleep(seconds: float) -> None`
- Thin wrapper around `time.sleep`.

### `_with_retries(fn, retry: int) -> Any`
- Executes `fn` with up to `retry + 1` attempts.
- On retryable exceptions: waits `min(2^attempt, 32)` seconds before retrying.
- Re-raises the last exception if all attempts fail.

### `_infer_remote_is_dir(bucket, key: str) -> bool`
- Returns `True` if `key` ends with `/` or `/.`.
- Checks if exact blob exists (returns `False` if so).
- Checks if any blobs exist with `key/` prefix (returns `True` if so).

### `_directory_flag_for_local(path, directory) -> bool`
- If `directory == "auto"`: returns `True` if path ends with `/*` or is a directory.
- Otherwise: returns `bool(directory)`.

### `_directory_flag_for_remote(bucket, key, directory) -> bool`
- If `directory == "auto"`: delegates to `_infer_remote_is_dir`.
- Otherwise: returns `bool(directory)`.

### `_parse_local_contents_spec(path) -> Tuple[Path, bool]`
- If path ends with `/*`: returns `(base_path, True)` (contents-only mode).
- Otherwise: returns `(path, False)`.

### `_parse_remote_contents_spec(remote_path) -> Tuple[str, str, bool]`
- Parses GCS path; if key ends with `/.`: strips it and returns `(bucket, key, True)`.
- Otherwise: returns `(bucket, key, False)`.

### `_list_remote_tree(client, bucket, prefix, retry_obj=None) -> Dict[str, Any]`
- Lists all blobs under `prefix` into `{name: blob}` mapping.

### `_crc32c_local(path: Path) -> Optional[str]`
- Computes base64-encoded CRC32C of a local file using `google_crc32c`.
- Returns `None` if `google_crc32c` is not available.
- Reads file in 8192-byte chunks.

---

## Transfer Operations

### `download_from_gs(remote_path, local_path, directory="auto", concurrent=True, ensure_contents=True, retry=0, max_workers=8)`
- Downloads a single file or directory tree from GCS to local filesystem.
- **Single file**: downloads blob directly to `local_path`, ensuring parent directory exists.
- **Directory**: lists all blobs under prefix, maps to local paths, uses `transfer_manager.download_many` for concurrent downloads.
- If `ensure_contents=True` or contents_only: places files directly under `local_path`.
- If `ensure_contents=False` and not contents_only: nests under remote leaf name.
- Checks download results for errors and re-raises.
- Wraps entire operation in `_with_retries`.

### `upload_to_gs(local_path, remote_path, directory="auto", concurrent=True, ensure_contents=True, retry=0, max_workers=8)`
- Uploads a single file or directory tree from local filesystem to GCS.
- **Single file**: uploads blob from filename.
- **Directory**: collects all files via `rglob("*")`, uses `transfer_manager.upload_many_from_filenames` for concurrent uploads.
- If `ensure_contents=False` and not contents_only: nests under local base name in remote prefix.
- Checks upload results for errors and re-raises.
- Wraps entire operation in `_with_retries`.

### `sync_to_gs(local_path, remote_path, directory="auto", concurrent=True, ensure_contents=True, delete=False, checksum=False, retry=0, max_workers=8)`
- Synchronizes local directory to GCS (incremental upload).
- **Single file**: just uploads.
- **Directory sync**:
  1. Collects local files via `rglob("*")`.
  2. Lists remote blobs under prefix.
  3. Compares: new files are uploaded; existing files are compared by size (or CRC32C if `checksum=True`).
  4. Uploads changed/new files one-by-one via `blob.upload_from_filename`.
  5. If `delete=True`: removes remote blobs not present locally.
- Wraps entire operation in `_with_retries`.

### `sync_from_gs(remote_path, local_path, directory="auto", concurrent=True, ensure_contents=True, delete=False, checksum=False, retry=0, max_workers=8)`
- Synchronizes GCS to local directory (incremental download).
- **Single file**: just downloads.
- **Directory sync**:
  1. Lists remote blobs under prefix.
  2. Checks local existence and compares by size (or CRC32C if `checksum=True`).
  3. Downloads changed/new files one-by-one.
  4. If `delete=True`: removes local files not present remotely via `item.unlink()`.
- Wraps entire operation in `_with_retries`.

### `pop_from_gs(remote_path, local_path, directory="auto", concurrent=True, ensure_contents=True, retry=0, max_workers=8)`
- Downloads from GCS then deletes the remote object(s).
- Calls `download_from_gs(...)` first.
- Then deletes: single blob or all blobs under prefix (for directories).
- Wraps delete operation in `_with_retries`.

### `push_to_gs(local_path, remote_path, directory="auto", concurrent=True, ensure_contents=True, retry=0, max_workers=8)`
- Uploads to GCS then deletes the local source.
- Calls `upload_to_gs(...)` first.
- Then deletes local:
  - If contents_only (`/*` suffix): deletes contents but preserves directory.
  - Otherwise: deletes entire path (`unlink` for files, `shutil.rmtree` for directories).

---

## Existence Check

### `check_exists_gs(remote, max_ancestors=None) -> Union[bool, Dict[str, bool]]`
- **Single path** (`str`):
  - If path indicates directory (trailing `/` or `/.`): checks for any children under prefix.
  - Otherwise: checks exact blob existence, then falls back to checking for children.
  - Returns `bool`.
- **Multiple paths** (`Sequence[str]`):
  - Groups paths by bucket name.
  - Checks each path individually within each bucket.
  - Returns `Dict[str, bool]` mapping each path to its existence status.
- Note: `max_ancestors` parameter is accepted but not used in the implementation.

---

## Convenience I/O

### `gs_read_text(remote_path, encoding="utf-8", retry=0) -> str`
- Downloads blob content as bytes, decodes with specified encoding.
- Wraps in `_with_retries`.

### `gs_read_bytes(remote_path, retry=0) -> bytes`
- Downloads blob content as raw bytes.
- Wraps in `_with_retries`.

### `gs_write_text(remote_path, text, encoding="utf-8", retry=0) -> None`
- Uploads text string as a blob with content type `text/plain; charset={encoding}`.
- Wraps in `_with_retries`.

### `gs_write_bytes(remote_path, data, content_type="application/octet-stream", retry=0) -> None`
- Uploads raw bytes as a blob with specified content type.
- Wraps in `_with_retries`.

### `gs_open(remote_path, mode="rb", retry=0)`
- Context manager that opens a GCS blob for streaming I/O.
- Supports modes like `'rb'`, `'wb'`, `'r'`, `'w'`.
- Opens via `blob.open(mode)`, yields the file-like object, closes in `finally`.
- Open operation wraps in `_with_retries`.

---

## Listing and Deletion

### `gs_list(remote_prefix, recursive=True, delimiter=None, max_results=None) -> List[str]`
- Lists blob names under a GCS prefix.
- If `recursive=False` and no delimiter specified: uses `"/"` as delimiter (non-recursive listing).
- Returns list of blob name strings.

### `gs_delete(remote_path, recursive=False) -> None`
- **Single blob** (`recursive=False` and no trailing `/`): deletes the exact blob.
- **Recursive** (`recursive=True` or trailing `/`): lists all blobs under prefix and deletes each.

---

## Important Behaviors

### Environment Variable Contract
- `EXPERIMENTS_PROJECT_CONF`: JSON-encoded project configuration (set by executor before task runs).
- `EXPERIMENTS_EXPERIMENT_CONF`: JSON-encoded experiment configuration (set by executor before task runs).
- Both must be valid JSON strings; missing or invalid values raise `RuntimeError`.

### GCS Lazy Loading
- GCS libraries are only imported on first use (`_require_gcs`).
- If GCS is unavailable, all GCS functions raise `RuntimeError` with an install hint.
- Optional dependencies (`google.api_core`, `google_crc32c`) degrade gracefully — retries and checksums are skipped if unavailable.

### Directory Semantics
- `ensure_contents=True` (default): directory contents placed directly under the target path.
- `ensure_contents=False`: contents nested under the source directory's leaf name.
- Trailing `/*` on local paths and trailing `/.` on remote paths encode "contents-only" mode.

### Retry Strategy
- Exponential backoff: `min(2^attempt, 32)` seconds between attempts.
- Retries only on recognized transient GCS exceptions.
- The `retry` parameter specifies number of *retries* (not total attempts), so total attempts = `retry + 1`.
