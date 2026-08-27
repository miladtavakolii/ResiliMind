from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from evaluation.evaluators.base import BaseEvaluator

class ExtractionEvaluator(BaseEvaluator):
    """Evaluates the Extractor Agent performance.

    This evaluator measures whether the system correctly identifies
    active resilience graph nodes and their associated signal polarity.

    Evaluation levels:
        1. Node extraction: Precision, Recall, F1, Jaccard
        2. Signal polarity: Accuracy
        3. Evidence alignment: Exact Match, Substring Match, Token F1
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
        evidence = self._evaluate_evidence(gold_signals, predicted_signals)

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
            "evidence": evidence,
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
    
    def _evaluate_evidence(
        self,
        gold_signals: Sequence[Any],
        predicted_signals: Sequence[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Evaluate evidence spans for nodes correctly identified by the extractor.

        Evidence is evaluated only on matched node IDs so that node-detection
        failures and evidence-quality failures remain separate metrics.

        Args:
            gold_signals: Sequence of ground-truth signal objects with node_id and evidence.
            predicted_signals: Sequence of predicted signal dictionaries.

        Returns:
            dict[str, float | int]: Dictionary containing exact match, substring match,
                token F1 scores, and matched node counts.
        """
        gold_map = {signal.node_id: signal.evidence or "" for signal in gold_signals}
        predicted_map = {
            signal.get("node_id"): signal.get("evidence", "")
            for signal in predicted_signals
            if signal.get("node_id")
        }

        matched_nodes = gold_map.keys() & predicted_map.keys()
        if not matched_nodes:
            return {
                "exact_match": 0.0,
                "substring_match": 0.0,
                "token_f1": 0.0,
                "matched_nodes": 0,
            }

        exact_matches = 0
        substring_matches = 0
        token_f1_scores: list[float] = []

        for node_id in matched_nodes:
            gold_evidence = self._normalize_evidence(gold_map[node_id])
            predicted_evidence = self._normalize_evidence(predicted_map[node_id])

            if not gold_evidence:
                continue

            if gold_evidence == predicted_evidence:
                exact_matches += 1

            if (
                gold_evidence in predicted_evidence
                or predicted_evidence in gold_evidence
            ):
                substring_matches += 1

            token_f1_scores.append(
                self._token_f1(gold_evidence, predicted_evidence)
            )

        evaluated = len(token_f1_scores)
        if evaluated == 0:
            return {
                "exact_match": 0.0,
                "substring_match": 0.0,
                "token_f1": 0.0,
                "matched_nodes": len(matched_nodes),
            }

        return {
            "exact_match": exact_matches / evaluated,
            "substring_match": substring_matches / evaluated,
            "token_f1": sum(token_f1_scores) / evaluated,
            "matched_nodes": len(matched_nodes),
        }

    @staticmethod
    def _normalize_evidence(text: str) -> str:
        """Normalize whitespace and case for evidence comparison.

        Args:
            text: Raw evidence string to normalize.

        Returns:
            str: Normalized lowercase evidence string with collapsed whitespace.
        """
        return " ".join(text.strip().lower().split())

    @staticmethod
    def _token_f1(gold: str, prediction: str) -> float:
        """Calculate token-level F1 for two evidence spans.

        Args:
            gold: Normalized ground-truth evidence text.
            prediction: Normalized predicted evidence text.

        Returns:
            float: Harmonic mean of token-level precision and recall.
        """
        gold_tokens = gold.split()
        prediction_tokens = prediction.split()

        if not gold_tokens or not prediction_tokens:
            return 0.0

        gold_counts = Counter(gold_tokens)
        prediction_counts = Counter(prediction_tokens)

        overlap = sum(
            min(gold_counts[token], prediction_counts[token])
            for token in (gold_counts.keys() & prediction_counts.keys())
        )

        if overlap == 0:
            return 0.0

        precision = overlap / len(prediction_tokens)
        recall = overlap / len(gold_tokens)

        return 2 * precision * recall / (precision + recall)

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
