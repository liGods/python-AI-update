from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class SearchBudget:
    milliseconds: int
    started: float = 0.0

    def __post_init__(self) -> None:
        self.started = perf_counter()

    @property
    def expired(self) -> bool:
        return (perf_counter() - self.started) * 1000 >= max(0, self.milliseconds)

    @property
    def elapsed_ms(self) -> float:
        return round((perf_counter() - self.started) * 1000, 3)
