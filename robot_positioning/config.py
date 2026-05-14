from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from dotenv import dotenv_values

T = TypeVar("T")


class EnvHelper:
    """Centralized access to environment-backed configuration."""

    def __init__(self, env_file: str | os.PathLike[str] | None = None, overrides: dict[str, str] | None = None):
        self.env_file = Path(env_file) if env_file else None
        values: dict[str, str | None] = {}
        if self.env_file and self.env_file.exists():
            values.update(dotenv_values(self.env_file))
        values.update({key: value for key, value in os.environ.items() if value is not None})
        if overrides:
            values.update(overrides)
        self._values = {key: value for key, value in values.items() if value is not None}

    def get_val(self, key: str, cast: Callable[[str], T] = str, default: T | None = None, required: bool = False) -> T:
        raw_value = self._values.get(key)
        if raw_value in (None, ""):
            if required and default is None:
                raise KeyError(f"Missing required configuration value: {key}")
            return default  # type: ignore[return-value]
        if cast is bool:
            return self._to_bool(raw_value)  # type: ignore[return-value]
        return cast(raw_value)

    def get_list(self, key: str, default: Iterable[str] | None = None) -> list[str]:
        raw_value = self._values.get(key)
        if raw_value in (None, ""):
            return list(default or [])
        return [item.strip() for item in raw_value.split(",") if item.strip()]

    @staticmethod
    def _to_bool(value: str) -> bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
