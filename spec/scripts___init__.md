# Specification: `experiments/scripts/__init__.py`

## Purpose
Package marker for the `scripts` subpackage. Contains only a comment indicating the package provides console entry points.

---

## Contents
- Single comment line: `# Scripts package for console entry points`
- No imports, no exports, no executable code.

## Role
Makes `experiments/scripts/` importable as a Python package. The individual script modules (`batch_local`, `interactive`, `jobattach`, `jobcat`, `jsonl_viewer`, `rightnow`) are intended to be invoked as standalone CLI entry points rather than imported programmatically.
