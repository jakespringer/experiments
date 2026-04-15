# Specification: `experiments/project.py`

## Purpose
Provides a static, process-global project context (`Project`) that is initialized once and then accessed throughout execution. Wraps project configuration from `project.json` for convenient attribute access.

---

## Class: `_ConfigView(SimpleNamespace)`

### Purpose
A read-write attribute-access wrapper around a dict. Extends `SimpleNamespace` for attribute-style access while maintaining a backing `_data` dict.

### Constructor: `__init__(self, data: Dict[str, Any])`
- Calls `super().__init__(**data)` to populate `SimpleNamespace` attributes.
- Stores `self._data = dict(data)` as a private backing store.

### Method: `to_dict(self) -> Dict[str, Any]`
- Returns `dict(self._data)`.

### Method: `__getattr__(self, name: str) -> Any`
- First falls back to `self._data[name]`.
- If not found, raises `AttributeError` with a helpful message including the path to `project.json` where the user should add the missing key.

### Method: `__setattr__(self, name: str, value: Any) -> None`
- If `name == "_data"`, uses `object.__setattr__` directly (avoids recursion).
- Otherwise, updates both `self._data[name]` and the `SimpleNamespace` attribute.

---

## Class: `Project`

### Purpose
Static (class-level) project context. All attributes and methods are `@classmethod` or class variables. Not meant to be instantiated.

### Class Variables
- `name: str | None = None` — Current project name.
- `config: _ConfigView | None = None` — Current project config wrapped in `_ConfigView`.

### Class Method: `init(cls, project_name: str) -> None`
**Behavior:**
1. Creates a `ConfigManager` instance.
2. Checks if `project.json` exists for `project_name`.
3. Calls `mgr.ensure_project(project_name)` (creates project.json if missing).
4. **If the file was just created** (didn't exist before step 3):
   - Prints a message to stderr instructing the user to edit the new `project.json`.
   - Calls `sys.exit(1)` to halt execution.
5. If the file already existed:
   - Reads `"config"` section from `project.json`.
   - Sets `cls.name` from `config["name"]` (falling back to `project_name`).
   - Sets `cls.config` to `_ConfigView(conf)`.

### Class Method: `update(cls, key: str, value: Any) -> None`
**Behavior:**
1. Raises `RuntimeError` if `cls.name is None` (must call `init` first).
2. Loads project config from disk.
3. Sets `config[key] = value` in the `"config"` section.
4. Saves updated config to disk.
5. Updates `cls.config` in-memory (creates new `_ConfigView` if `cls.config` is `None`, otherwise uses `__setattr__`).

### Important Behaviors
- `Project.init()` is meant to be called **once** at startup. Calling it again with a different project name will overwrite the class state.
- The exit-on-create behavior ensures users fill in necessary configuration before first run.
- `Project.config.<key>` provides convenient attribute access to project.json values.
