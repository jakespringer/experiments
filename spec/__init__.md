# Specification: `experiments/__init__.py`

## Purpose
Package-level entry point that re-exports the public API surface of the `experiments` library.

## Imports and Re-exports

### From `.artifact`
- `Artifact` — Base class for task artifacts
- `ArtifactSet` — Collection wrapper with functional utilities
- `ArgumentProduct` — Utility for building cartesian products of named argument values

### From `.executor`
- `Executor` — Base executor class (abstract)
- `SlurmExecutor` — Slurm-based executor for submitting array jobs
- `PrintExecutor` — Executor that prints shell commands to stdout
- `Task` — Compiled unit of work for an artifact

### From `.cli`
- `auto_cli` — Main entry point for the CLI interface

### From `.utils`
- `flatten_dict` — Flattens nested dicts with dot-separated keys

### From `.project`
- `Project` — Static project context manager

### From `.` (submodule)
- `analysis` — Analysis module (imported as a whole submodule)

## `__all__`

```python
__all__ = [
    'Artifact', 'ArtifactSet', 'ArgumentProduct',
    'Executor', 'SlurmExecutor', 'PrintExecutor', 'Task',
    'auto_cli',
    'flatten_dict',
    'Project',
    'analysis',
]
```

## Notes
- The `analysis` module is imported as a namespace, not individual functions. Users access it as `experiments.analysis.load_export(...)`, etc.
- No logic is performed in this file; it is purely re-exporting symbols.
