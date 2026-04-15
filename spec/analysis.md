# Specification: `experiments/analysis.py`

## Purpose
Provides utilities for loading, querying, filtering, and analyzing exported experiment data (JSON files produced by the `export` CLI command). Supports optional pandas integration for DataFrame-based analysis.

## Dependencies
- `json`, `re`, `pathlib.Path`
- **Optional**: `pandas` (detected at import time via `HAS_PANDAS` flag)

---

## Module-Level Constants

### `_ARTIFACT_REF_PATTERN`
- Regex: `r'^([A-Za-z_][A-Za-z0-9_]*)\(([a-f0-9]+)\)$'`
- Matches artifact reference strings like `"MyArtifact(abc123def0)"`.

---

## Internal Functions

### `_parse_artifact_reference(value: str) -> Optional[tuple[str, str]]`
- If `value` matches `_ARTIFACT_REF_PATTERN`, returns `(artifact_type, hash)`.
- Returns `None` for non-strings or non-matching strings.

### `_build_artifact_map(artifacts: List[Dict]) -> Dict[str, Dict]`
- Builds `{hash: artifact_dict}` from a list of artifact dicts.
- Keyed on `artifact["hash"]`.

### `_resolve_artifact_references(value, artifact_map) -> Any`
- Recursively resolves artifact reference strings in a value:
  - If string matches pattern AND map contains matching hash AND type matches → returns the full artifact dict.
  - Lists → recursively resolve each element.
  - Dicts → recursively resolve each value.
  - Other types → pass through.

---

## Public Functions

### `load_export(file_path, resolve=False) -> Dict[str, Any]`
- Loads a JSON export file. Returns dict with keys: `project_config`, `global_config`, `artifacts`.
- If `resolve=True`: builds artifact map, then resolves all artifact reference strings in every artifact to actual artifact dicts.
- Raises: `FileNotFoundError`, `json.JSONDecodeError`.

### `get_project_config(file_path) -> Dict[str, Any]`
- Returns `data["project_config"]` from the export.

### `get_global_config(file_path) -> Dict[str, Any]`
- Returns `data["global_config"]` from the export.

### `get_artifacts(file_path, resolve=False) -> List[Dict]`
- Returns `data["artifacts"]` from the export.
- `resolve` parameter passed through to `load_export`.

### `get_stages(file_path) -> List[str]`
- Collects all unique stage names from artifacts' `"stage"` field (which can be a list or string).
- Returns sorted list.

### `get_artifact_types(file_path) -> List[str]`
- Collects all unique `"artifact_type"` values.
- Returns sorted list.

### `filter_artifacts(artifacts, stage=None, artifact_types=None, exists=None) -> List[Dict]`
- Filters a list of artifact dicts by:
  - **stage**: Artifact's `"stage"` list must intersect with provided stage(s).
  - **artifact_types**: Artifact's `"artifact_type"` must be in the set.
  - **exists**: Artifact's `"exists"` must match the boolean.
- All filters are optional (`None` = no filter).

### `load_artifacts_df(file_path, stage=None, artifact_types=None, exists=None, flatten=True, resolve=False) -> pd.DataFrame`
- Loads artifacts, applies filters, returns as pandas DataFrame.
- If `flatten=True`: uses `flatten_dict` from `utils` to flatten nested dicts with dot-separated keys.
- Raises `ImportError` if pandas not available.

### `summarize_export(file_path) -> None`
- Prints a formatted summary to stdout:
  - Project configuration key-value pairs.
  - Total artifact count.
  - Per-stage artifact counts.
  - Per-type artifact counts.
  - Existence status counts (exists / does not exist / unknown).

### `count_by_stage(file_path) -> Dict[str, int]`
- Returns `{stage_name: count}` mapping.

### `count_by_type(file_path) -> Dict[str, int]`
- Returns `{artifact_type: count}` mapping.

### `get_artifact_by_relpath(file_path, relpath, resolve=False) -> Optional[Dict]`
- Finds first artifact where `artifact["relpath"] == relpath`.

### `get_artifact_by_hash(file_path, artifact_hash, resolve=False) -> Optional[Dict]`
- Finds first artifact where `artifact["hash"] == artifact_hash`.

### `export_to_csv(file_path, output_file, stage=None, artifact_types=None, exists=None, flatten=True, resolve=False) -> None`
- Loads artifacts as DataFrame, writes to CSV, prints count message.
- Requires pandas.
