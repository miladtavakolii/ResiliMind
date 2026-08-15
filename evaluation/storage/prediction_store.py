from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


class PredictionStore:
    """
    Stores and loads raw model predictions.

    Predictions are stored separately from gold
    benchmark data to prevent contamination.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, predictions: list[dict]) -> None:
        """Save predictions as JSONL."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            for item in predictions:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self) -> Iterator[dict]:
        """Load predictions from JSONL."""
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)
