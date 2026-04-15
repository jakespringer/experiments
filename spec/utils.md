# Specification: `experiments/utils.py`

## Purpose
Provides general-purpose utility functions for the experiments framework.

---

## Function: `flatten_dict(d, parent_key='', sep='.')`

### Purpose
Recursively flattens a nested dictionary into a single-level dictionary with dot-separated keys.

### Parameters
- **`d`** (`dict`): Dictionary to flatten (may be arbitrarily nested).
- **`parent_key`** (`str`, default `''`): Key prefix for recursive calls. Used internally to build compound keys during recursion.
- **`sep`** (`str`, default `'.'`): Separator string inserted between nested key components.

### Returns
- `dict`: A flat dictionary where nested keys are joined by `sep`.

### Algorithm
1. Iterates over all `(k, v)` pairs in `d`.
2. Constructs `new_key`:
   - If `parent_key` is non-empty: `f"{parent_key}{sep}{k}"`.
   - Otherwise: `k` (the key itself).
3. If `v` is a `dict`: recursively calls `flatten_dict(v, new_key, sep=sep)` and extends the result items.
4. Otherwise: appends `(new_key, v)` as a leaf item.
5. Returns `dict(items)`.

### Example
```python
>>> nested = {'a': {'b': 1, 'c': {'d': 2}}, 'e': 3}
>>> flatten_dict(nested)
{'a.b': 1, 'a.c.d': 2, 'e': 3}
```

### Important Behaviors
- Only `dict` values trigger recursion; lists, tuples, and other iterables are treated as leaf values.
- The separator is configurable — e.g., `sep='/'` would produce `'a/b'` instead of `'a.b'`.
- If `parent_key` is provided on the initial call, all output keys will be prefixed with it.
- Empty nested dicts produce no output keys (they are iterated but yield no items).
