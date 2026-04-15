# Specification: `experiments/artifact.py`

## Purpose
Defines the core data model for experiment artifacts, artifact collections, and parameter products. Artifacts are the fundamental unit of work in the framework — each artifact represents a task that can be executed, skipped, hashed, and ordered via dependency analysis.

---

## Class: `Artifact`

### Purpose
Base class for user-defined experiment artifacts. Subclasses are expected to be Python `@dataclass` classes. Users override `construct(task)` to define what shell commands an artifact runs, and optionally override `exists` or `should_skip()` for skip logic, and `get_requirements()` for Slurm resource requirements.

### Properties

#### `relpath -> str`
- Returns `"{ClassName}/{hash}"` where `ClassName` is `self.__class__.__name__` and `hash` is `self.get_hash()`.
- Used as a unique identifier for the artifact's output location.

#### `exists -> bool`
- Default: returns `False`.
- Intended to be overridden by subclasses to check if the artifact's output already exists (e.g., on disk, in GCS).
- When `True`, the artifact is skipped during execution.

### Methods

#### `should_skip(self) -> bool`
- Returns `self.exists`.
- Provides a centralized skip-condition check. Can be overridden for custom skip logic beyond existence.

#### `as_dict(self) -> Dict[str, Any]`
- If `self` is a dataclass, returns `{field.name: getattr(self, field.name)}` for all fields.
- Otherwise, returns all non-underscore-prefixed attributes from `vars(self)`.
- Performs **shallow** extraction — preserves `Artifact` references in values.

#### `get_hash(self) -> str`
- Computes a deterministic 10-character hex hash (SHA-256 prefix) of the artifact's data.
- **Caching**: Stores result in `self._experiments_hash_cache`; returns cached value on subsequent calls.
- **Algorithm**:
  1. Calls `self.as_dict()` to get field data.
  2. Defines inner function `atom(value)` that recursively converts values to hashable strings:
     - `IgnoreHash` → `None` (excluded from hash)
     - `Artifact` → `dependency.get_hash()` (recursive)
     - `ArtifactSet` → sorted concatenation of member hashes
     - `Directive` → `str(value)`
     - Scalars (`str`, `int`, `float`, `bool`, `None`) → `str(value)`
     - `list`/`tuple` → `'[' + ','.join(sorted(element_hashes)) + ']'` (recursive)
     - `dict` → `'{' + ','.join(sorted("k:v_hash")) + '}'` (recursive)
     - Other types → attempts `str(value)`, raises `TypeError` on failure
  3. Filters out `None` atoms, sorts items by key.
  4. Builds payload: `"key1=val1|key2=val2|..."` (pipe-separated, sorted by key).
  5. Returns first 10 hex characters of SHA-256 of UTF-8 encoded payload.

### Important Behaviors
- **Sorting in hashing**: Both list/tuple elements and dict key-value pairs are **sorted** for deterministic output. This means order of list elements does NOT affect the hash — only membership matters.
- **IgnoreHash fields**: Any field wrapped in `IgnoreHash(value)` is excluded from the hash computation entirely, allowing metadata that shouldn't affect identity.
- **Circular references**: If artifact A references artifact B which references A, `get_hash()` will infinitely recurse (no cycle detection).

---

## Class: `ArtifactSet(Sequence[Any])`

### Purpose
A typed collection wrapper with functional-programming-style utilities. Supports iteration, indexing, length, concatenation, cartesian products, mapping, filtering, and flat-mapping.

### Constructor: `__init__(self, items: Iterable[Any])`
- Stores `list(items)` internally as `self._items`.

### Sequence Protocol
- `__iter__` → iterates `self._items`
- `__len__` → `len(self._items)`
- `__getitem__(idx)` → `self._items[idx]`

### Operator: `__add__(self, other: ArtifactSet) -> ArtifactSet`
- Returns a new `ArtifactSet` containing items from both sets concatenated.
- Returns `NotImplemented` if `other` is not an `ArtifactSet`.

### Static Method: `_cartesian_dicts(params: Dict[str, Any]) -> List[Dict[str, Any]]`
- For each value in `params`:
  - If it's a `list` or `tuple`, use as-is.
  - Otherwise, wrap in a singleton list.
- Returns `itertools.product(*value_lists)` as a list of dicts.

### Class Method: `from_product(*, cls=None, params) -> ArtifactSet`
- **`params` is `ArgumentProduct`**: calls `params.to_dicts()` to expand.
- **`params` is `dict`**: calls `_cartesian_dicts(params)` to expand.
- **`params` is other sequence**: uses `list(params)` directly.
- If `cls` is provided AND all expanded items are dicts: instantiates `cls(**x)` for each.
- Otherwise wraps as-is.

### Static Method: `join_product(*sets: ArtifactSet) -> ArtifactSet`
- Returns cartesian product of all input sets as tuples.
- If no sets given, returns empty `ArtifactSet`.

### Method: `map(self, fn) -> ArtifactSet`
- Applies `fn` to each item. If item is a tuple, splats it: `fn(*item)`.
- Returns new `ArtifactSet` of results.

### Method: `filter(self, predicate) -> ArtifactSet`
- Keeps items where `predicate(item)` or `predicate(*item)` (if tuple) returns `True`.
- Returns new `ArtifactSet`.

### Method: `map_flatten(self, fn) -> ArtifactSet`
- Like `map`, but if `fn` returns an `ArtifactSet`, its items are flattened into the result.
- Non-`ArtifactSet` returns are treated as single items.

### Method: `map_reduce(self, map_fn, reduce_fn) -> Any`
- Applies `map_fn` to each item (with tuple splatting), collects results into a list.
- Passes the list to `reduce_fn` and returns its result.

---

## Class: `ArgumentProduct`

### Purpose
Utility for building cartesian products of named argument values, used as input to `ArtifactSet.from_product`.

### Constructor: `__init__(self, **params: Any)`
- Stores `params` as `self._params`.

### Class Method: `from_dict(cls, params: Dict[str, Any]) -> ArgumentProduct`
- Creates an `ArgumentProduct` from a dict.

### Method: `to_dicts(self) -> Iterable[Dict[str, Any]]`
- Same logic as `ArtifactSet._cartesian_dicts`:
  - Wraps non-list/tuple values in singleton lists.
  - Yields `dict(zip(keys, prod))` for each element of `itertools.product(*value_lists)`.

### Method: `map(self, fn) -> List[Any]`
- Applies `fn` to each dict from `to_dicts()` and returns the list of results.
