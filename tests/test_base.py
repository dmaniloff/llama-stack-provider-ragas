"""Unit tests for RagasEvaluatorBase._get_metrics."""

import pytest

from llama_stack_provider_ragas.base import RagasEvaluatorBase
from llama_stack_provider_ragas.constants import METRIC_MAPPING
from llama_stack_provider_ragas.errors import RagasEvaluationError

pytestmark = pytest.mark.unit


class ConcreteEvaluator(RagasEvaluatorBase):
    """Minimal concrete subclass for testing."""

    async def run_eval(self, request): ...

    async def job_status(self, request): ...

    async def job_cancel(self, request): ...

    async def job_result(self, request): ...


class TestGetMetrics:
    def setup_method(self):
        self.evaluator = ConcreteEvaluator()

    def test_valid_scoring_functions(self):
        metrics = self.evaluator._get_metrics(["answer_relevancy", "faithfulness"])
        assert len(metrics) == 2
        assert metrics[0] is METRIC_MAPPING["answer_relevancy"]
        assert metrics[1] is METRIC_MAPPING["faithfulness"]

    def test_unknown_scoring_function_skipped(self):
        metrics = self.evaluator._get_metrics(
            ["answer_relevancy", "nonexistent_metric"]
        )
        assert len(metrics) == 1
        assert metrics[0] is METRIC_MAPPING["answer_relevancy"]

    def test_all_unknown_falls_back_to_defaults(self):
        metrics = self.evaluator._get_metrics(["nonexistent_metric"])
        assert len(metrics) == len(self.evaluator._DEFAULT_METRICS)
        for metric, name in zip(metrics, self.evaluator._DEFAULT_METRICS, strict=False):
            assert metric is METRIC_MAPPING[name]

    def test_empty_list_falls_back_to_defaults(self):
        metrics = self.evaluator._get_metrics([])
        assert len(metrics) == len(self.evaluator._DEFAULT_METRICS)

    def test_default_metric_drift_skipped(self, monkeypatch):
        monkeypatch.setattr(
            ConcreteEvaluator,
            "_DEFAULT_METRICS",
            ["answer_relevancy", "nonexistent_default"],
        )
        metrics = self.evaluator._get_metrics([])
        assert len(metrics) == 1
        assert metrics[0] is METRIC_MAPPING["answer_relevancy"]

    def test_all_defaults_invalid_raises(self, monkeypatch):
        monkeypatch.setattr(
            ConcreteEvaluator,
            "_DEFAULT_METRICS",
            ["nonexistent_1", "nonexistent_2"],
        )
        with pytest.raises(RagasEvaluationError, match="No valid default metrics"):
            self.evaluator._get_metrics([])
