from evaluation.evaluators.base import BaseEvaluator


class DummyEvaluator(BaseEvaluator):
    """Temporary evaluator used for framework validation."""

    name = "dummy"

    def evaluate(self, gold, prediction):
        return {"success": True}
