# SPDX-License-Identifier: Apache-2.0
"""Common benchmark registry types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkSpec:
    """Registry entry for one benchmark dataset."""

    name: str
    aliases: tuple[str, ...]
    default_split: str
    default_source: str
    load_samples: Callable[..., list[Any]]
    evaluate: Callable[[list[dict[str, Any]]], tuple[dict[str, Any], list[dict[str, Any]]]]
    local_names: tuple[str, ...] = ()

    def source_for(self, data_root: str | Path | None = None) -> str:
        if data_root is None:
            return self.default_source

        root = Path(data_root).expanduser()
        for name in self.local_names:
            path = root / name
            if path.exists():
                return str(path)
        return self.default_source
