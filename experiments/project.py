from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict

from .config import ConfigManager


class _ConfigView(SimpleNamespace):
    """Attribute access wrapper for a dict that also exposes dict-like APIs."""

    def __init__(self, data: Dict[str, Any]):
        super().__init__(**data)
        self._data = dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def __getattr__(self, name: str) -> Any:  # fallback to dict
        if name in self._data:
            return self._data[name]
        return super().__getattribute__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_data",):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value
            try:
                super().__setattr__(name, value)
            except Exception:
                object.__setattr__(self, name, value)


class Project:
    """Static project context.

    Usage:
        Project.init("my-project")
        Project.name -> str
        Project.config.<key> -> value from project.json "config"
        Project.update(key, value) -> persists into project.json
    """

    name: str | None = None
    config: _ConfigView | None = None

    @classmethod
    def init(cls, project_name: str) -> None:
        mgr = ConfigManager()
        proj = mgr.ensure_project(project_name)
        conf = proj.get("config", {})
        cls.name = conf.get("name", project_name)
        cls.config = _ConfigView(conf)

    @classmethod
    def update(cls, key: str, value: Any) -> None:
        if cls.name is None:
            raise RuntimeError("Project.init must be called before update")
        mgr = ConfigManager()
        data = mgr.load_project_config(cls.name)
        conf = data.get("config", {})
        conf[key] = value
        data["config"] = conf
        mgr.save_project_config(cls.name, data)
        # update in-memory
        if cls.config is None:
            cls.config = _ConfigView(conf)
        else:
            cls.config.__setattr__(key, value)

