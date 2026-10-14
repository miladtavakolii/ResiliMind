from __future__ import annotations

from statistics import mean
from typing import Any

from evaluation.schemas import CaseEvaluationResult
from evaluation.metrics.classification import classification_report


class EvaluationAggregator:
    """Aggregates per-case evaluation results into dataset-level metrics."""

    def aggregate(self, results: list[CaseEvaluationResult]) -> dict[str, Any]:
        """Generate overall evaluation statistics across all evaluated cases.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, Any]: Aggregated metrics for safety, assessment, routing,
                and response evaluation alongside dataset size.
        """
        return {
            "dataset_size": len(results),
            "safety": self._aggregate_safety(results),
            "extraction": self._aggregate_extraction(results),
            "assessment": self._aggregate_assessment(results),
            "routing": self._aggregate_routing(results),
            "response": self._aggregate_response(results),
        }

    def _aggregate_extraction(self, results: list[CaseEvaluationResult]) -> dict[str, Any]:
        """Aggregate extraction metrics across all cases.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, Any]: Dictionary containing mean precision, recall, f1,
                and jaccard scores for node detection.
        """
        metrics = [
            m for result in results if (m := result.metrics.get("extraction"))
        ]

        if not metrics:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}

        polarity_metrics = [m["polarity"] for m in metrics if m.get("polarity", {}).get("matched_nodes", 0) > 0]
        polarity_correct = sum(m["correct"] for m in polarity_metrics)
        polarity_total = sum(m["matched_nodes"] for m in polarity_metrics)
        polarity_accuracy = (polarity_correct / polarity_total if polarity_total else 0.0)

        evidence = [m["evidence"] for m in metrics if m.get("evidence", {}).get("matched_nodes", 0) > 0]

        nodes = [m["node_detection"] for m in metrics]

        gold_count = sum(n["gold_count"] for n in nodes)
        prediction_count = sum(n["prediction_count"] for n in nodes)
        matched = sum(n["matched"] for n in nodes)

        micro_precision = (
            matched / prediction_count
            if prediction_count
            else 0.0
        )
        micro_recall = (
            matched / gold_count
            if gold_count
            else 0.0
        )
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if micro_precision + micro_recall
            else 0.0
        )

        return {
            "node_detection": {
                "precision": mean(n["precision"] for n in nodes),
                "recall": mean(n["recall"] for n in nodes),
                "f1": mean(n["f1"] for n in nodes),
                "jaccard": mean(n["jaccard"] for n in nodes),
            },
            "polarity_accuracy": polarity_accuracy,
            "evidence": {
                "exact_match": mean(e["exact_match"] for e in evidence),
                "substring_match": mean(e["substring_match"] for e in evidence),
                "token_f1": mean(e["token_f1"] for e in evidence),
            },
            "micro": {
                "precision": micro_precision,
                "recall": micro_recall,
                "f1": micro_f1,
            }
        }

    def _aggregate_safety(self, results: list[CaseEvaluationResult]) -> dict[str, Any]:
        """Calculate aggregate safety metrics across evaluation results.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, Any]: Dictionary containing binary safety metrics and multiclass
                risk category classification metrics.
        """
        tp, tn, fp, fn = 0, 0, 0, 0
        pairs = []

        for result in results:
            if not (metric := result.metrics.get("safety")):
                continue

            matrix = metric.get("confusion_matrix", {})
            tp += int(matrix.get("tp", 0))
            tn += int(matrix.get("tn", 0))
            fp += int(matrix.get("fp", 0))
            fn += int(matrix.get("fn", 0))

            if category := metric.get("category"):
                expected = category.get("expected")
                predicted = category.get("predicted") or "SAFE"
                if expected:
                    pairs.append((str(expected), str(predicted)))

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall)
            else 0.0
        )

        base_labels = ["SAFE", "SELF_HARM", "VIOLENCE", "SEVERE_ABUSE"]
        extra_labels = sorted({
            label for pair in pairs for label in pair if label not in base_labels
        })
        category_labels = base_labels + extra_labels
        category_matrix = {
            actual: {pred: 0 for pred in category_labels}
            for actual in category_labels
        }
        for expected, predicted in pairs:
            if expected in category_matrix and predicted in category_matrix[expected]:
                category_matrix[expected][predicted] += 1

        classification = classification_report(pairs)
        category_total = len(pairs)
        category_correct = sum(1 for exp, pred in pairs if exp == pred)

        return {
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "risk_category": {
                "accuracy": classification["accuracy"],
                "macro_f1": classification["macro"]["f1"],
                "confusion_matrix": category_matrix,
                "correct": category_correct,
                "total": category_total,
            },
        }

    def _aggregate_assessment(self, results: list[CaseEvaluationResult]) -> dict[str, Any]:
        """Calculate aggregate assessment regression and status classification metrics across evaluation results.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, Any]: Dictionary containing MAE/RMSE on evaluated nodes,
                coverage against gold nodes, and status classification metrics.
        """
        metrics = [
            m for result in results if (m := result.metrics.get("assessment"))
        ]

        status_labels = ("GREEN", "YELLOW", "RED")

        if not metrics:
            return {
                "mean_mae": 0.0,
                "evaluated_nodes_mae": 0.0,
                "mean_rmse": 0.0,
                "matched_nodes": 0,
                "gold_nodes": 0,
                "coverage": 0.0,
                "status_accuracy": 0.0,
                "status_macro_f1": 0.0,
                "status": {
                    "accuracy": 0.0,
                    "macro_f1": 0.0,
                    "confusion_matrix": {
                        actual: {pred: 0 for pred in status_labels}
                        for actual in status_labels
                    },
                    "correct": 0,
                    "total": 0,
                },
            }

        absolute_error_sum = sum(m.get("absolute_error_sum", 0.0) for m in metrics)
        squared_error_sum = sum(m.get("squared_error_sum", 0.0) for m in metrics)
        observation_count = sum(m.get("observation_count", 0) for m in metrics)
        matched_nodes = sum(m.get("matched_nodes", 0) for m in metrics)
        gold_nodes = sum(m.get("gold_nodes", 0) for m in metrics)

        mae = absolute_error_sum / observation_count if observation_count else 0.0
        rmse = (
            (squared_error_sum / observation_count) ** 0.5
            if observation_count
            else 0.0
        )
        coverage = matched_nodes / gold_nodes if gold_nodes else 1.0

        total_status_matrix = {
            actual: {pred: 0 for pred in status_labels}
            for actual in status_labels
        }
        status_pairs = []

        for m in metrics:
            status = m.get("status", {})
            matrix = status.get("confusion_matrix")
            if matrix:
                for actual in status_labels:
                    for pred in status_labels:
                        total_status_matrix[actual][pred] += matrix.get(actual, {}).get(pred, 0)

            pairs = status.get("pairs")
            if pairs:
                status_pairs.extend(pairs)
            elif matrix:
                for actual in status_labels:
                    for pred in status_labels:
                        count = matrix.get(actual, {}).get(pred, 0)
                        if count > 0:
                            status_pairs.extend([(actual, pred)] * count)

        classification = classification_report(status_pairs)
        status_total = len(status_pairs)
        status_correct = sum(
            total_status_matrix[label][label] for label in status_labels
        )
        status_accuracy = (
            status_correct / status_total if status_total else 0.0
        )
        status_macro_f1 = classification["macro"]["f1"]

        return {
            "mean_mae": mae,
            "evaluated_nodes_mae": mae,
            "mean_rmse": rmse,
            "matched_nodes": matched_nodes,
            "gold_nodes": gold_nodes,
            "coverage": coverage,
            "status_accuracy": status_accuracy,
            "status_macro_f1": status_macro_f1,
            "status": {
                "accuracy": status_accuracy,
                "macro_f1": status_macro_f1,
                "confusion_matrix": total_status_matrix,
                "correct": status_correct,
                "total": status_total,
            },
        }

    def _aggregate_routing(self, results: list[CaseEvaluationResult]) -> dict[str, float]:
        """Calculate aggregate workflow routing accuracy across evaluation results.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, float]: Dictionary containing overall routing accuracy.
        """
        labels = ("advisor", "questioner", "emergency_response", "unknown")
        matrix = {actual: {pred: 0 for pred in labels} for actual in labels}

        pairs = []
        total, correct = 0, 0
        for result in results:
            metric = result.metrics.get("routing")
            if not metric:
                continue

            expected, predicted = metric.get("expected"), metric.get("predicted")
            if expected in labels:
                predicted = predicted if predicted in labels else "unknown"
                pairs.append((expected, predicted))
                matrix[expected][predicted] += 1
                total += 1
                if expected == predicted:
                    correct += 1

        classification = classification_report(pairs)
        return {
            "accuracy": classification["accuracy"],
            "macro": classification["macro"],
            "per_class": classification["per_class"],
            "confusion_matrix": matrix,
        }

    def _aggregate_response(self, results: list[CaseEvaluationResult]) -> dict[str, float]:
        """Calculate aggregate qualitative response scores from LLM judge outputs.

        Args:
            results: List of per-case evaluation results.

        Returns:
            dict[str, float]: Dictionary containing average judge evaluation score.
        """
        scores = [
            metric.get("overall", 0)
            for result in results
            if (metric := result.metrics.get("response"))
        ]
        return {"average_score": mean(scores) if scores else 0}
