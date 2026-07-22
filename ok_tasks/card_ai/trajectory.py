from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

from ok_tasks.card_ai.schema import SCHEMA_VERSION, TrajectoryEvent


class TrajectoryWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: TrajectoryEvent | dict) -> None:
        self.extend((event,))

    def extend(self, events: Iterable[TrajectoryEvent | dict]) -> None:
        if self.path.suffix == ".gz":
            stream = gzip.open(self.path, mode="at", encoding="utf-8", compresslevel=1)
        else:
            stream = self.path.open(mode="a", encoding="utf-8")
        with stream:
            for event in events:
                value = event.to_dict() if isinstance(event, TrajectoryEvent) else dict(event)
                value.setdefault("schema_version", SCHEMA_VERSION)
                stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_trajectory(path: str | Path) -> Iterator[dict]:
    source = Path(path)
    if not source.is_file():
        return
    opener = gzip.open if source.suffix == ".gz" else Path.open
    arguments = {"mode": "rt", "encoding": "utf-8"}
    with opener(source, **arguments) as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError as error:
                raise ValueError(f"轨迹第 {line_number} 行不是有效 JSON: {source}") from error
            if int(value.get("schema_version", 0)) > SCHEMA_VERSION:
                raise ValueError(f"轨迹版本高于当前程序支持版本: {value.get('schema_version')}")
            yield value


def atomic_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
