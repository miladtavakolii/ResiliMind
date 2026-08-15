from __future__ import annotations

from typing import Any, Sequence

from evaluation.evaluators.base import BaseEvaluator


class ExtractionEvaluator(BaseEvaluator):
    """Evaluates the Extractor Agent performance.

    This evaluator measures whether the system correctly identifies
    active resilience graph nodes and their associated signal polarity.

    Evaluation levels:
        1. Node extraction: Precision, Recall, F1, Jaccard similarity
        2. Signal polarity: Accuracy over correctly extracted nodes
    """

    name: str = "extraction"

    def evaluate(self, gold: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """Evaluate extracted resilience signals against ground truth.

        Args:
            gold: EvaluationGold object containing ground truth extractions.
            prediction: Workflow output dictionary containing extracted signals.

        Returns:
            dict[str, Any]: Extraction metrics containing node detection stats
                (counts, precision, recall, f1, jaccard) and polarity metrics.
        """
        gold_signals = gold.extraction.active_signals
        predicted_signals = prediction.get("extraction", {}).get("signals", [])

        gold_nodes = {signal.node_id for signal in gold_signals}
        predicted_nodes = {
            signal.get("node_id")
            for signal in predicted_signals
            if signal.get("node_id")
        }

        intersection = gold_nodes & predicted_nodes

        precision = self._safe_div(len(intersection), len(predicted_nodes))
        recall = self._safe_div(len(intersection), len(gold_nodes))
        f1 = self._safe_div(2 * precision * recall, precision + recall)
        jaccard = self._safe_div(len(intersection), len(gold_nodes | predicted_nodes))

        polarity = self._evaluate_polarity(gold_signals, predicted_signals)

        return {
            "node_detection": {
                "gold_count": len(gold_nodes),
                "prediction_count": len(predicted_nodes),
                "matched": len(intersection),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "jaccard": jaccard,
            },
            "polarity": polarity,
        }

    def _evaluate_polarity(
        self,
        gold_signals: Sequence[Any],
        predicted_signals: Sequence[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Evaluate signal polarity for matched nodes.

        Args:
            gold_signals: Sequence of ground truth signal objects.
            predicted_signals: Sequence of predicted signal dictionaries.

        Returns:
            dict[str, float | int]: Polarity evaluation metrics containing accuracy
                and the count of matched nodes.
        """
        gold_map = {signal.node_id: signal.detected_signal for signal in gold_signals}
        pred_map = {
            signal.get("node_id"): signal.get("detected_signal")
            for signal in predicted_signals
        }

        matched_nodes = gold_map.keys() & pred_map.keys()
        if not matched_nodes:
            return {"accuracy": 0.0, "matched_nodes": 0}

        correct = sum(1 for node_id in matched_nodes if gold_map[node_id] == pred_map[node_id])

        return {
            "accuracy": correct / len(matched_nodes),
            "matched_nodes": len(matched_nodes),
        }

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        """Safely divide two numbers, returning 0.0 if the denominator is zero.

        Args:
            numerator: The division numerator.
            denominator: The division denominator.

        Returns:
            float: The result of the division, or 0.0 if denominator is 0.
        """
        if denominator == 0:
            return 0.0
        return numerator / denominator
