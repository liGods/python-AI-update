from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TranspositionEntry:
    depth: int
    value: float


class TranspositionTable(dict[object, TranspositionEntry]):
    def get_at_depth(self, key: object, depth: int) -> float | None:
        entry = self.get(key)
        return entry.value if entry is not None and entry.depth >= depth else None
