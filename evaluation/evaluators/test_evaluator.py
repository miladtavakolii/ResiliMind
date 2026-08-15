from __future__ import annotations

from typing import Any

from evaluation.evaluators.base import BaseEvaluator


class DummyEvaluator(BaseEvaluator):
    """Temporary dummy evaluator used for pipeline and framework testing.

    Attributes:
        name (str): Identifier name for the evaluator.
    """

    name: str = "dummy"

    def evaluate(self, gold: Any, prediction: Any) -> dict[str, bool]:
        """Execute a stub evaluation returning a constant success status.

        Args:
            gold: Ground truth data (ignored).
            prediction: System prediction output (ignored).

        Returns:
            dict[str, bool]: A dictionary indicating mock success.
        """
        return {"success": True}
