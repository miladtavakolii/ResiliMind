from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


class PredictionStore:
    """Stores and loads raw model predictions.

    Predictions are stored separately from gold benchmark data to prevent contamination.
    """

    def __init__(self, path: Path) -> None:
        """Initialize the prediction store with a file path.

        Args:
            path: Path to the JSONL prediction file.
        """
        self.path = path

    def save(self, predictions: Iterable[dict[str, Any]]) -> None:
        """Save predictions to a JSON Lines file.

        Args:
            predictions: Iterable of prediction dictionaries to serialize.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        with self.path.open("w", encoding="utf-8") as file:
            for item in predictions:
                file.write(json.dumps(item, ensure_ascii=False) + "\n")

    def load(self) -> Iterator[dict[str, Any]]:
        """Yield predictions iteratively from the JSON Lines file.

        Yields:
            Parsed prediction dictionaries from each non-empty line.
        """
        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    yield json.loads(line)
