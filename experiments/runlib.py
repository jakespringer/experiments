from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import io
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

# Use the same attribute-access view as Project to keep behavior consistent
from .project import _ConfigView


# -----------------------------
# Config accessors (cached)
# -----------------------------

_CACHED_PROJECT_CONF: _ConfigView | None = None
_CACHED_EXPERIMENT_CONF: _ConfigView | None = None


def _load_env_json(var_name: str) -> Dict[str, Any]:
    raw = os.environ.get(var_name)
    if not raw:
        raise RuntimeError(
            f"Missing {var_name}. This code must be launched via the experiments launcher."
        )
    try:
        return json.loads(raw)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Invalid JSON in {var_name}: {e}") from e


def get_project_config() -> _ConfigView:
    global _CACHED_PROJECT_CONF
    if _CACHED_PROJECT_CONF is None:
        data = _load_env_json("EXPERIMENTS_PROJECT_CONF")
        _CACHED_PROJECT_CONF = _ConfigView(data)
    return _CACHED_PROJECT_CONF


def get_experiment_config() -> _ConfigView:
    global _CACHED_EXPERIMENT_CONF
    if _CACHED_EXPERIMENT_CONF is None:
        data = _load_env_json("EXPERIMENTS_EXPERIMENT_CONF")
        _CACHED_EXPERIMENT_CONF = _ConfigView(data)
    return _CACHED_EXPERIMENT_CONF


def get_relpath() -> str:
    cfg = get_experiment_config()
    try:
        return str(getattr(cfg, "relpath"))
    except Exception:
        raise RuntimeError("Experiment config does not contain 'relpath'")


# -----------------------------
# Temporary directory helpers
# -----------------------------

@dataclass
class TemporaryWorkspace:
    path: Path
    _removed: bool = False

    def cleanup(self) -> None:
        if not self._removed and self.path.exists():
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            finally:
                self._removed = True

    def __enter__(self) -> Path:
        return self.path

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.cleanup()


def _choose_tmp_root(filesystem: Optional[Union[str, Path]]) -> Path:
    if filesystem:
        root = Path(filesystem)
        root.mkdir(parents=True, exist_ok=True)
        return root
    shm = Path("/dev/shm")
    if shm.exists() and os.access(shm, os.W_OK | os.X_OK):
        return shm
    return Path(tempfile.gettempdir())


def make_temp_dir(prefix: str = "exp_", filesystem: Optional[Union[str, Path]] = None) -> Path:
    root = _choose_tmp_root(filesystem)
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=str(root)))
    ws = TemporaryWorkspace(path)
    atexit.register(ws.cleanup)
    return path


@contextlib.contextmanager
def temporary_workspace(prefix: str = "exp_", filesystem: Optional[Union[str, Path]] = None) -> Iterator[Path]:
    path = make_temp_dir(prefix=prefix, filesystem=filesystem)
    ws = TemporaryWorkspace(Path(path))
    try:
        yield ws.path
    finally:
        ws.cleanup()


# -----------------------------
# Networking helpers
# -----------------------------

def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# -----------------------------
# Google Cloud Storage helpers
# -----------------------------

_GCS_IMPORTED = False
_GCS_ERR: Optional[Exception] = None


def _require_gcs() -> None:
    global _GCS_IMPORTED, _GCS_ERR
    if _GCS_IMPORTED:
        return
    try:
        # Lazy import
        from google.cloud import storage  # type: ignore  # noqa: F401
        from google.cloud.storage import transfer_manager  # type: ignore  # noqa: F401
        from google.api_core import retry as _retry  # type: ignore  # noqa: F401
        from google.api_core import exceptions as _gexc  # type: ignore  # noqa: F401
        _GCS_IMPORTED = True
    except Exception as e:  # pragma: no cover
        _GCS_ERR = e
        raise RuntimeError(
            "google-cloud-storage is required for GCS operations. Install with 'pip install google-cloud-storage'."
        ) from e


def _get_storage_client():
    _require_gcs()
    from google.cloud import storage  # type: ignore

    return storage.Client()


def _get_gcs_exceptions():
    """Return a tuple of retryable google.api_core exception classes if available."""
    try:
        from google.api_core import exceptions as gexc  # type: ignore

        retryables = []
        for name in (
            "ServiceUnavailable",
            "InternalServerError",
            "TooManyRequests",
            "DeadlineExceeded",
            "GatewayTimeout",
        ):
            cls = getattr(gexc, name, None)
            if cls is not None:
                retryables.append(cls)
        # Fallback to base class if none found
        base = getattr(gexc, "GoogleAPICallError", None)
        if not retryables and base is not None:
            retryables.append(base)
        return tuple(retryables) if retryables else (Exception,)
    except Exception:
        return (Exception,)


def _build_gcs_retry(attempts: int):
    """Build a google.api_core.retry.Retry with a retryable predicate.

    If google.api_core.retry is unavailable or attempts <= 1, return None.
    """
    if attempts is None or int(attempts) <= 0:
        return None
    try:
        from google.api_core import retry as gretry  # type: ignore
        from google.api_core import exceptions as gexc  # type: ignore

        def predicate(exc: BaseException) -> bool:
            retryable_types = _get_gcs_exceptions()
            return isinstance(exc, retryable_types)

        # Configure a bounded backoff; attempts are enforced by our wrapper around
        # transfer_manager calls; for direct API calls we rely on deadline to cap.
        # Rough deadline: 2s per attempt with jitter handled by library
        deadline = 2.0 * (int(attempts) + 1)
        return gretry.Retry(predicate=predicate, initial=1.0, maximum=5.0, multiplier=2.0, deadline=deadline)
    except Exception:
        return None


def _parse_gs_path(gs_url: str) -> Tuple[str, str]:
    if not gs_url.startswith("gs://"):
        raise ValueError(f"Not a gs:// URL: {gs_url}")
    without = gs_url[5:]
    parts = without.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _is_gs_path(path: str) -> bool:
    return path.startswith("gs://")


def _ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _with_retries(fn, retry: int) -> Any:
    attempts = max(0, int(retry)) + 1
    last_exc: Optional[BaseException] = None
    retryable = _get_gcs_exceptions()
    for i in range(attempts):
        try:
            return fn()
        except retryable as e:  # type: ignore[misc]
            last_exc = e
            if i < attempts - 1:
                _sleep(2.0)
            else:
                raise
    return None


def _infer_remote_is_dir(bucket, key: str) -> bool:
    from google.cloud import storage  # type: ignore

    if key.endswith("/"):
        return True
    # Treat trailing '/.' as contents of a directory
    if key.endswith("/."):
        return True
    blob = bucket.blob(key)
    if blob.exists():
        return False
    prefix = key.rstrip("/") + "/"
    iterator = bucket.client.list_blobs(bucket, prefix=prefix, max_results=1)  # type: ignore[arg-type]
    for _ in iterator:
        return True
    return False


def _directory_flag_for_local(path: Union[str, Path], directory: Union[str, bool]) -> bool:
    if directory == "auto":
        # Support trailing '/*' to denote contents of directory
        p = str(path)
        if isinstance(path, (str, Path)) and str(p).endswith("/*"):
            return True
        return Path(path).is_dir()
    return bool(directory)


def _directory_flag_for_remote(bucket, key: str, directory: Union[str, bool]) -> bool:
    if directory == "auto":
        return _infer_remote_is_dir(bucket, key)
    return bool(directory)


def _parse_local_contents_spec(path: Union[str, Path]) -> Tuple[Path, bool]:
    """Return (base_path, contents_only) where '/*' means contents-only."""
    pstr = str(path)
    if pstr.endswith("/*"):
        base = Path(pstr[:-2])
        return base, True
    return Path(path), False


def _parse_remote_contents_spec(remote_path: str) -> Tuple[str, str, bool]:
    """Return (bucket, key, contents_only) where trailing '/.' means contents-only."""
    bucket, key = _parse_gs_path(remote_path)
    if key.endswith("/."):
        key = key[:-2]
        return bucket, key, True
    return bucket, key, False


def download_from_gs(
    remote_path: str,
    local_path: Union[str, Path],
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8,
) -> None:
    """Download file or directory from GCS.

    ensure_contents semantics:
      - file → write exactly to local_path
      - directory → copy contents under local_path (no extra nesting)
    """

    def _impl() -> None:
        client = _get_storage_client()
        from google.cloud.storage import transfer_manager  # type: ignore

        bucket_name, key, remote_contents = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)

        # If remote specified '/.', force contents semantics
        if remote_contents:
            directory = True
            ensure_contents_flag = True
        else:
            ensure_contents_flag = ensure_contents

        is_dir = _directory_flag_for_remote(bucket, key, directory)
        local_path_p = Path(local_path)

        if is_dir:
            # Determine destination root honoring ensure_contents_flag
            prefix = key.rstrip("/") + "/"
            dest_root = local_path_p if ensure_contents_flag else (local_path_p / Path(key.rstrip("/")).name)
            _ensure_dir(dest_root)
            if concurrent:
                retry_obj = _build_gcs_retry(retry)
                blob_names = [b.name for b in client.list_blobs(bucket, prefix=prefix, retry=retry_obj)]
                if not blob_names:
                    return
                transfer_manager.download_many_to_path(
                    bucket,
                    blob_names,
                    destination_directory=str(dest_root),
                    blob_name_prefix=prefix,
                    max_workers=max(1, int(max_workers)),
                )
            else:
                retry_obj = _build_gcs_retry(retry)
                for blob in client.list_blobs(bucket, prefix=prefix, retry=retry_obj):
                    rel = blob.name[len(prefix) :]
                    dest = dest_root / rel
                    _ensure_dir(dest.parent)
                    blob.download_to_filename(dest.as_posix(), retry=retry_obj)
        else:
            _ensure_dir(Path(local_path_p).parent)
            blob = bucket.blob(key)
            if concurrent:
                # Chunked concurrent download for large files
                transfer_manager.download_chunks_concurrently(
                    blob,
                    str(local_path_p),
                    max_workers=max(1, int(max_workers)),
                )
            else:
                retry_obj = _build_gcs_retry(retry)
                blob.download_to_filename(str(local_path_p), retry=retry_obj)

    _with_retries(_impl, retry)


def upload_to_gs(
    local_path: Union[str, Path],
    remote_path: str,
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8,
) -> None:
    def _impl() -> None:
        client = _get_storage_client()
        from google.cloud.storage import transfer_manager  # type: ignore

        local_base, local_contents = _parse_local_contents_spec(local_path)
        bucket_name, key, remote_contents = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)

        # Derive final flags
        ensure_contents_flag = bool(ensure_contents or remote_contents or local_contents)

        is_dir = _directory_flag_for_local(local_base, directory)

        if is_dir:
            if not local_base.is_dir():
                raise ValueError(f"Expected a directory at {local_base}")
            dest_prefix = key.rstrip("/") + "/" if ensure_contents_flag else key.rstrip("/") + f"/{local_base.name}/"
            if concurrent:
                transfer_manager.upload_many_from_directory(
                    bucket,
                    source_directory=str(local_base),
                    destination_directory=dest_prefix,
                    max_workers=max(1, int(max_workers)),
                )
            else:
                for path in local_base.rglob("*"):
                    if path.is_file():
                        rel = path.relative_to(local_base).as_posix()
                        blob = bucket.blob(dest_prefix + rel)
                        _ensure_dir(path.parent)
                        retry_obj = _build_gcs_retry(retry)
                        blob.upload_from_filename(path.as_posix(), retry=retry_obj)
        else:
            if not local_base.is_file():
                raise ValueError(f"Expected a file at {local_base}")
            blob = bucket.blob(key)
            retry_obj = _build_gcs_retry(retry)
            blob.upload_from_filename(local_base.as_posix(), retry=retry_obj)

    _with_retries(_impl, retry)


def _list_remote_tree(client, bucket, prefix: str, retry_obj=None) -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for b in client.list_blobs(bucket, prefix=prefix, retry=retry_obj):
        items[b.name] = b
    return items


def _crc32c_local(path: Path) -> Optional[str]:
    try:
        import google_crc32c  # type: ignore

        checksum = google_crc32c.Checksum()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                checksum.update(chunk)
        # google-cloud-storage uses base64-encoded crc32c; google-crc32c returns int digest
        # The Blob.crc32c property is a base64 string; comparing that exactly requires encoding.
        # To avoid strict dependency, return None here and fall back unless the library supports base64.
        # As an approximation we return None if base64 conversion is unavailable.
        try:
            import base64

            digest_int = checksum.digest()
            # google-crc32c v1.5+ returns int; convert to 4-byte big-endian
            if isinstance(digest_int, int):
                b4 = digest_int.to_bytes(4, byteorder="big", signed=False)
            else:
                b4 = digest_int  # type: ignore[assignment]
            return base64.b64encode(b4).decode("ascii")
        except Exception:
            return None
    except Exception:
        return None


def sync_to_gs(
    local_path: Union[str, Path],
    remote_path: str,
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    delete: bool = False,
    checksum: bool = False,
    retry: int = 0,
    max_workers: int = 8,
) -> None:
    def _impl() -> None:
        client = _get_storage_client()

        local_base, local_contents = _parse_local_contents_spec(local_path)
        bucket_name, key, remote_contents = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)

        ensure_contents_flag = bool(ensure_contents or remote_contents or local_contents)

        is_dir = _directory_flag_for_local(local_base, directory)
        if not is_dir:
            return upload_to_gs(local_base, remote_path, directory=False, concurrent=concurrent, ensure_contents=True, retry=retry, max_workers=max_workers)

        if not local_base.is_dir():
            raise ValueError(f"Expected a directory at {local_base}")

        dest_prefix = key.rstrip("/") + "/" if ensure_contents_flag else key.rstrip("/") + f"/{local_base.name}/"

        retry_obj = _build_gcs_retry(retry)
        remote_map = _list_remote_tree(client, bucket, dest_prefix, retry_obj=retry_obj)

        to_upload: List[Path] = []
        for path in local_base.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(local_base).as_posix()
            remote_key = dest_prefix + rel
            remote_blob = remote_map.get(remote_key)
            if remote_blob is None:
                to_upload.append(path)
                continue
            if checksum:
                local_crc = _crc32c_local(path)
                if local_crc and getattr(remote_blob, "crc32c", None) != local_crc:
                    to_upload.append(path)
                    continue
            else:
                try:
                    if path.stat().st_size != int(getattr(remote_blob, "size", 0)):
                        to_upload.append(path)
                        continue
                except Exception:
                    to_upload.append(path)

        def _upload_one(p: Path) -> None:
            rel = p.relative_to(local_base).as_posix()
            blob = bucket.blob(dest_prefix + rel)
            blob.upload_from_filename(p.as_posix(), retry=_build_gcs_retry(retry))

        if concurrent and to_upload:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
                list(ex.map(_upload_one, to_upload))
        else:
            for p in to_upload:
                _upload_one(p)

        if delete:
            local_keys = {dest_prefix + p.relative_to(local_base).as_posix() for p in local_base.rglob("*") if p.is_file()}
            extra_remote = set(remote_map.keys()) - local_keys
            for rk in extra_remote:
                from google.api_core import exceptions as gexc  # type: ignore
                try:
                    bucket.blob(rk).delete(retry=_build_gcs_retry(retry))
                except gexc.NotFound:
                    continue

    _with_retries(_impl, retry)


def sync_from_gs(
    remote_path: str,
    local_path: Union[str, Path],
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    delete: bool = False,
    checksum: bool = False,
    retry: int = 0,
    max_workers: int = 8,
) -> None:
    def _impl() -> None:
        client = _get_storage_client()

        bucket_name, key, remote_contents = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)

        if remote_contents:
            directory = True
            ensure_contents_flag = True
        else:
            ensure_contents_flag = ensure_contents

        is_dir = _directory_flag_for_remote(bucket, key, directory)
        local_p = Path(local_path)

        if not is_dir:
            return download_from_gs(remote_path, local_p, directory=False, concurrent=concurrent, ensure_contents=True, retry=retry, max_workers=max_workers)

        # Determine destination root honoring ensure_contents_flag
        prefix = key.rstrip("/") + "/"
        dest_root = local_p if ensure_contents_flag else (local_p / Path(key.rstrip("/")).name)
        _ensure_dir(dest_root)
        retry_obj = _build_gcs_retry(retry)
        remote_map = _list_remote_tree(client, bucket, prefix, retry_obj=retry_obj)

        def _dest_for(remote_key: str) -> Path:
            rel = remote_key[len(prefix) :]
            return dest_root / rel

        to_download: List[Tuple[str, str]] = []  # (remote_key, local_path)
        for rk, blob in remote_map.items():
            dest = _dest_for(rk)
            if dest.exists() and dest.is_file():
                if checksum and getattr(blob, "crc32c", None):
                    local_crc = _crc32c_local(dest)
                    if local_crc and local_crc == blob.crc32c:
                        continue
                else:
                    try:
                        if dest.stat().st_size == int(getattr(blob, "size", 0)):
                            continue
                    except Exception:
                        pass
            _ensure_dir(dest.parent)
            to_download.append((rk, dest.as_posix()))

        def _download_one(item: Tuple[str, str]) -> None:
            rk, lp = item
            bucket.blob(rk).download_to_filename(lp, retry=retry_obj)

        if concurrent and to_download:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
                list(ex.map(_download_one, to_download))
        else:
            for item in to_download:
                _download_one(item)

        if delete:
            # Remove local files not present remotely
            remote_rel = {rk[len(prefix) : ] for rk in remote_map.keys()}
            for path in dest_root.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(dest_root).as_posix()
                    if rel not in remote_rel:
                        try:
                            path.unlink()
                        except Exception:
                            pass

    _with_retries(_impl, retry)


def pop_from_gs(
    remote_path: str,
    local_path: Union[str, Path],
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8,
) -> None:
    def _impl() -> None:
        client = _get_storage_client()
        download_from_gs(remote_path, local_path, directory=directory, concurrent=concurrent, ensure_contents=ensure_contents, retry=retry, max_workers=max_workers)
        bucket_name, key, remote_contents = _parse_remote_contents_spec(remote_path)
        bucket = client.bucket(bucket_name)
        # If remote specified '/.', treat as directory contents
        if remote_contents or _directory_flag_for_remote(bucket, key, directory):
            prefix = key.rstrip("/") + "/"
            from google.api_core import exceptions as gexc  # type: ignore
            retry_obj = _build_gcs_retry(retry)
            for b in client.list_blobs(bucket, prefix=prefix, retry=retry_obj):
                try:
                    b.delete(retry=retry_obj)
                except gexc.NotFound:
                    continue
        else:
            from google.api_core import exceptions as gexc  # type: ignore
            blob = bucket.blob(key)
            try:
                blob.delete(retry=_build_gcs_retry(retry))
            except gexc.NotFound:
                pass

    _with_retries(_impl, retry)


def push_to_gs(
    local_path: Union[str, Path],
    remote_path: str,
    directory: Union[str, bool] = "auto",
    concurrent: bool = True,
    ensure_contents: bool = True,
    retry: int = 0,
    max_workers: int = 8,
) -> None:
    def _impl() -> None:
        # Parse specs to honor '/*' and '/.' semantics
        local_base, local_contents = _parse_local_contents_spec(local_path)
        _, _, remote_contents = _parse_remote_contents_spec(remote_path)

        upload_to_gs(local_path, remote_path, directory=directory, concurrent=concurrent, ensure_contents=ensure_contents or local_contents or remote_contents, retry=retry, max_workers=max_workers)

        # Deletion semantics: if local had '/*', delete only contents, preserve directory
        if _directory_flag_for_local(local_base, directory):
            if local_contents:
                try:
                    for child in local_base.iterdir():
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            with contextlib.suppress(Exception):
                                child.unlink()
                except Exception:
                    pass
            else:
                try:
                    shutil.rmtree(local_base, ignore_errors=True)
                except Exception:
                    pass
        else:
            try:
                Path(local_base).unlink()
            except Exception:
                pass

    _with_retries(_impl, retry)


def check_exists(remote: Union[str, Sequence[str]], max_ancestors: Optional[int] = None) -> Union[bool, Dict[str, bool]]:
    def _impl_one(path: str) -> bool:
        client = _get_storage_client()
        bucket_name, key, remote_contents = _parse_remote_contents_spec(path)
        bucket = client.bucket(bucket_name)
        retry_obj = _build_gcs_retry(1)
        # If '/.' provided, check for any children under the directory
        if remote_contents:
            prefix = key.rstrip("/") + "/"
            it = client.list_blobs(bucket, prefix=prefix, max_results=1, retry=retry_obj)
            return any(True for _ in it)
        else:
            blob = bucket.blob(key)
            if blob.exists(retry=retry_obj):
                return True
            prefix = key.rstrip("/") + "/"
            it = client.list_blobs(bucket, prefix=prefix, max_results=1, retry=retry_obj)
            return any(True for _ in it)

    if isinstance(remote, str):
        return _with_retries(lambda: _impl_one(remote), retry=0)

    # List[str] case: group by bucket and ancestor to reduce calls
    paths = list(remote)
    client = _get_storage_client()

    # Group by bucket
    by_bucket: Dict[str, List[str]] = {}
    for p in paths:
        bucket, key = _parse_gs_path(p)
        by_bucket.setdefault(bucket, []).append(key)

    results: Dict[str, bool] = {p: False for p in paths}

    for bucket_name, keys in by_bucket.items():
        bucket = client.bucket(bucket_name)

        # Build grouping prefixes
        groups: Dict[str, List[str]] = {}
        if max_ancestors is None:
            # Use common prefix if any; fallback to top-level segments
            def common_prefix(ss: List[str]) -> str:
                if not ss:
                    return ""
                split = [s.split("/") for s in ss]
                prefix_parts: List[str] = []
                for parts in zip(*split):
                    if all(part == parts[0] for part in parts):
                        prefix_parts.append(parts[0])
                    else:
                        break
                return "/".join(prefix_parts) + ("/" if prefix_parts else "")

            cp = common_prefix(keys)
            if cp:
                groups[cp] = keys
            else:
                for k in keys:
                    head = (k.split("/", 1)[0] + "/") if "/" in k else ""
                    groups.setdefault(head, []).append(k)
        else:
            depth = max(0, int(max_ancestors))
            for k in keys:
                parts = k.split("/")
                if depth >= len(parts):
                    prefix = ""
                else:
                    prefix = "/".join(parts[: len(parts) - depth])
                    if prefix:
                        prefix += "/"
                groups.setdefault(prefix, []).append(k)

        # List blobs once per group
            for prefix, member_keys in groups.items():
                listed = {b.name for b in client.list_blobs(bucket, prefix=prefix, retry=_build_gcs_retry(1))}
            for k in member_keys:
                    # Support '/.' key membership mapping
                    contents_only = False
                    if k.endswith("/."):
                        k0 = k[:-2]
                        contents_only = True
                    else:
                        k0 = k

                    if not contents_only and k0 in listed:
                        results[f"gs://{bucket_name}/{k}"] = True
                    else:
                        # If key not present, check if any children exist
                        if any(name.startswith(k0.rstrip("/") + "/") for name in listed):
                            results[f"gs://{bucket_name}/{k}"] = True

    return results


# --------------
# Convenience IO
# --------------

def gs_read_text(remote_path: str, encoding: str = "utf-8", retry: int = 0) -> str:
    def _impl() -> str:
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        blob = client.bucket(bucket_name).blob(key)
        return blob.download_as_text(encoding=encoding, retry=_build_gcs_retry(retry))

    return _with_retries(_impl, retry)


def gs_read_bytes(remote_path: str, retry: int = 0) -> bytes:
    def _impl() -> bytes:
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        blob = client.bucket(bucket_name).blob(key)
        return blob.download_as_bytes(retry=_build_gcs_retry(retry))

    return _with_retries(_impl, retry)


def gs_write_text(remote_path: str, text: str, encoding: str = "utf-8", retry: int = 0) -> None:
    data = text.encode(encoding)
    return gs_write_bytes(remote_path, data, content_type="text/plain; charset=" + encoding, retry=retry)


def gs_write_bytes(remote_path: str, data: bytes, content_type: str = "application/octet-stream", retry: int = 0) -> None:
    def _impl() -> None:
        client = _get_storage_client()
        bucket_name, key = _parse_gs_path(remote_path)
        blob = client.bucket(bucket_name).blob(key)
        blob.upload_from_string(data, content_type=content_type, retry=_build_gcs_retry(retry))

    _with_retries(_impl, retry)


@contextlib.contextmanager
def gs_open(remote_path: str, mode: str = "rb", retry: int = 0):  # type: ignore[override]
    client = _get_storage_client()
    bucket_name, key = _parse_gs_path(remote_path)
    blob = client.bucket(bucket_name).blob(key)

    def _open():
        return blob.open(mode)

    f = _with_retries(_open, retry)
    try:
        yield f
    finally:
        with contextlib.suppress(Exception):
            f.close()


def gs_list(remote_prefix: str, recursive: bool = True, delimiter: Optional[str] = None, max_results: Optional[int] = None) -> List[str]:
    client = _get_storage_client()
    bucket_name, key = _parse_gs_path(remote_prefix)
    bucket = client.bucket(bucket_name)
    params: Dict[str, Any] = {}
    if not recursive and delimiter is None:
        delimiter = "/"
    if delimiter:
        params["delimiter"] = delimiter
    if max_results is not None:
        params["max_results"] = int(max_results)
    names: List[str] = []
    for b in client.list_blobs(bucket, prefix=key, **params):
        names.append(b.name)
    return names


def gs_delete(remote_path: str, recursive: bool = False) -> None:
    client = _get_storage_client()
    bucket_name, key = _parse_gs_path(remote_path)
    bucket = client.bucket(bucket_name)
    if recursive or key.endswith("/"):
        prefix = key.rstrip("/") + "/"
        for b in client.list_blobs(bucket, prefix=prefix):
            with contextlib.suppress(Exception):
                b.delete()
    else:
        with contextlib.suppress(Exception):
            bucket.blob(key).delete()


