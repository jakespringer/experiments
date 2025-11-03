from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Sequence, Tuple, Union
import dataclasses
import hashlib


class Artifact:
    """Base Artifact.

    Minimal base for dataclass-style task artifacts. Subclasses are expected to
    be dataclasses defined in user code. Only core surfaces are provided here;
    advanced behavior (paths, materialization) will be added later.
    """

    @property
    def relpath(self) -> str:
        """Return the resolved artifact path.
        """
        class_name = self.__class__.__name__
        hash = self.get_hash()
        return f"{class_name}/{hash}"
    
    @property
    def exists(self) -> bool:
        """Check if this artifact already exists.
        
        Override this property in subclasses to implement custom existence checks.
        When exists returns True, the artifact will be skipped during execution.
        
        Returns:
            False by default (artifact needs to be executed)
        """
        return False
    
    def should_skip(self) -> bool:
        """Determine if this artifact should be skipped during execution.
        
        This method provides a centralized place to check various skip conditions.
        Currently checks if the artifact exists, but can be extended in the future
        with additional skip conditions.
        
        Override this method in subclasses for custom skip logic, or simply
        override the 'exists' property for existence-based skipping.
        
        Returns:
            True if the artifact should be skipped, False otherwise
        """
        return self.exists

    def as_dict(self) -> Dict[str, Any]:
        # Shallow extraction to preserve Artifact references
        if dataclasses.is_dataclass(self):
            return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        else:
            # Fallback: best-effort public attributes
            return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def get_hash(self) -> str:
        data = self.as_dict()
        
        from .executor import Directive, IgnoreHash

        def atom(value: Any) -> Union[str, None]:
            if isinstance(value, IgnoreHash):
                return None
            if isinstance(value, Artifact):
                return value.get_hash()
            if isinstance(value, ArtifactSet):
                hashes = [item.get_hash() for item in value if isinstance(item, Artifact)]
                if len(hashes) != len(value):
                    raise TypeError("ArtifactSet must contain only Artifacts for hashing")
                return ''.join(sorted(hashes))
            if isinstance(value, Directive):
                return str(value)
            if isinstance(value, (str, int, float, bool)) or value is None:
                return str(value)
            if isinstance(value, (list, tuple)):
                # Recursively hash list/tuple elements
                elements = []
                for item in value:
                    item_hash = atom(item)
                    if item_hash is not None:
                        elements.append(item_hash)
                # Use sorted for deterministic hashing
                return '[' + ','.join(sorted(elements)) + ']'
            if isinstance(value, dict):
                # Recursively hash dict key-value pairs
                pairs = []
                for k, v in value.items():
                    v_hash = atom(v)
                    if v_hash is not None:
                        pairs.append(f"{k}:{v_hash}")
                # Sort by key for deterministic hashing
                return '{' + ','.join(sorted(pairs)) + '}'
            # For other types, try str() conversion
            try:
                return str(value)
            except:
                raise TypeError(f"Unsupported value type in artifact hashing: {type(value).__name__}")

        items = [(k, a) for k, v in data.items() if (a := atom(v)) is not None]
        items.sort(key=lambda kv: kv[0])
        payload = '|'.join(f"{k}={v}" for k, v in items)
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:10]

class ArtifactSet(Sequence[Any]):
    """A collection wrapper with simple functional utilities.

    Supports construction from cartesian products and mapping, as well as join
    products between sets.
    """

    def __init__(self, items: Iterable[Any]):
        self._items: List[Any] = list(items)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Any:
        return self._items[idx]

    def __add__(self, other: "ArtifactSet") -> "ArtifactSet":
        """Add two ArtifactSets together to create a union.
        
        Args:
            other: Another ArtifactSet to combine with this one
            
        Returns:
            A new ArtifactSet containing all items from both sets
        """
        if not isinstance(other, ArtifactSet):
            return NotImplemented
        return ArtifactSet(list(self._items) + list(other._items))

    @staticmethod
    def _cartesian_dicts(params: Dict[str, Any]) -> List[Dict[str, Any]]:
        keys = list(params.keys())
        value_lists: List[List[Any]] = []
        for value in params.values():
            if isinstance(value, (list, tuple)):
                value_lists.append(list(value))
            else:
                value_lists.append([value])
        return [dict(zip(keys, prod)) for prod in itertools.product(*value_lists)]

    @classmethod
    def from_product(
        aset_cls,
        *,
        cls: Callable[..., Any] | None = None,
        params: Union[Dict[str, Any], "ArgumentProduct", Sequence[Any]],
    ) -> "ArtifactSet":
        """Build an ArtifactSet from a parameter product.

        - If params is a dict, compute the cartesian product of values (scalars
          treated as singletons).
        - If params is an ArgumentProduct, expand it to a list of dicts.
        - If params is a sequence of pre-built objects, wrap them directly.

        If "cls_" is provided and the expanded items are dicts, instantiate
        objects via "cls_(**variables)".
        """

        # Expand into a list of items or variable dicts
        if isinstance(params, ArgumentProduct):
            expanded: List[Any] = list(params.to_dicts())
        elif isinstance(params, dict):
            expanded = ArtifactSet._cartesian_dicts(params)
        else:
            expanded = list(params)

        # If we have variable dicts and a constructor, build objects
        if cls is not None and expanded and all(isinstance(x, dict) for x in expanded):
            built = [cls(**x) for x in expanded]
            return ArtifactSet(built)

        # Already objects (or dicts without cls_) — wrap as-is
        return ArtifactSet(expanded)

    @staticmethod
    def join_product(*sets: "ArtifactSet") -> "ArtifactSet":
        """Cartesian join across one or more ArtifactSets."""
        if not sets:
            return ArtifactSet([])
        pools: List[List[Any]] = [list(s) for s in sets]
        return ArtifactSet(list(itertools.product(*pools)))

    def map(self, fn: Callable[..., Any]) -> "ArtifactSet":
        """Map a function across the items.

        - If an item is a tuple, it will be splatted into the function.
        - Otherwise, the item is passed as a single argument.
        """
        mapped: List[Any] = []
        for item in self._items:
            if isinstance(item, tuple):
                mapped.append(fn(*item))
            else:
                mapped.append(fn(item))
        return ArtifactSet(mapped)

    def filter(self, predicate: Callable[..., bool]) -> "ArtifactSet":
        """Filter items by a predicate.

        - If an item is a tuple, it will be splatted into the predicate.
        - Otherwise, the item is passed as a single argument.

        Returns a new ArtifactSet containing only items for which the
        predicate returns True.
        """
        filtered: List[Any] = []
        for item in self._items:
            if isinstance(item, tuple):
                if predicate(*item):
                    filtered.append(item)
            else:
                if predicate(item):
                    filtered.append(item)
        return ArtifactSet(filtered)

    def map_flatten(self, fn: Callable[..., "ArtifactSet"]) -> "ArtifactSet":
        """Map a function across items and flatten the results.

        The function should return an ArtifactSet for each item. All resulting
        ArtifactSets will be flattened into a single ArtifactSet.

        - If an item is a tuple, it will be splatted into the function.
        - Otherwise, the item is passed as a single argument.
        """
        flattened: List[Any] = []
        for item in self._items:
            if isinstance(item, tuple):
                result = fn(*item)
            else:
                result = fn(item)
            
            if isinstance(result, ArtifactSet):
                flattened.extend(result)
            else:
                # If not an ArtifactSet, treat as a single item
                flattened.append(result)
        return ArtifactSet(flattened)

    def map_reduce(self, map_fn: Callable[..., Any], reduce_fn: Callable[[List[Any]], Any]) -> Any:
        """Map a function across items and reduce the results.

        First applies map_fn to each item, then applies reduce_fn to the list of results.

        Args:
            map_fn: Function to apply to each item (or tuple of items)
            reduce_fn: Function to reduce the mapped results to a single value

        Returns:
            The reduced value
        """
        mapped: List[Any] = []
        for item in self._items:
            if isinstance(item, tuple):
                mapped.append(map_fn(*item))
            else:
                mapped.append(map_fn(item))
        return reduce_fn(mapped)


class ArgumentProduct:
    """Utility to build cartesian products of named argument values."""

    def __init__(self, **params: Any) -> None:
        self._params: Dict[str, Any] = params

    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> "ArgumentProduct":
        return cls(**params)

    def to_dicts(self) -> Iterable[Dict[str, Any]]:
        keys = list(self._params.keys())
        value_lists: List[List[Any]] = []
        for value in self._params.values():
            if isinstance(value, (list, tuple)):
                value_lists.append(list(value))
            else:
                value_lists.append([value])
        for prod in itertools.product(*value_lists):
            yield dict(zip(keys, prod))

    def map(self, fn: Callable[[Dict[str, Any]], Any]) -> List[Any]:
        return [fn(variables) for variables in self.to_dicts()]