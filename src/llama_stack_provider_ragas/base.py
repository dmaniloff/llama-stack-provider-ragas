"""
Base provider for Ragas evaluation with common functionality.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from .compat import (
    Benchmark,
    BenchmarkConfig,
    BenchmarksProtocolPrivate,
    Eval,
    EvaluateResponse,
    EvaluateRowsRequest,
    Job,
    JobCancelRequest,
    JobResultRequest,
    JobStatusRequest,
    RunEvalRequest,
    ScoringResult,
    json_schema_type,
)
from .constants import METRIC_MAPPING
from .errors import RagasEvaluationError

logger = logging.getLogger(__name__)


@json_schema_type
class EmptyEvaluateResponse(EvaluateResponse):
    generations: list[dict[str, Any]] = []
    scores: dict[str, ScoringResult] = {}


class RagasEvaluatorBase(Eval, BenchmarksProtocolPrivate, ABC):
    """Base class for Ragas evaluators with common functionality."""

    def __init__(self):
        self.benchmarks: dict[str, Benchmark] = {}

    def _get_metrics(self, scoring_functions: list[str]) -> list:
        """Get the list of metrics to run based on scoring functions.

        Args:
            scoring_functions: List of scoring function names to use

        Returns:
            List of metrics (unconfigured - ragas_evaluate will configure them)
        """
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        metrics = []

        for metric_name in scoring_functions:
            if metric_name in METRIC_MAPPING:
                metric = METRIC_MAPPING[metric_name]
                metrics.append(metric)
            else:
                logger.warning(f"Unknown metric: {metric_name}")

        if not metrics:
            # Use default metrics if none specified or all invalid
            logger.info("Using default metrics")
            metrics = [
                answer_relevancy,
                context_precision,
                faithfulness,
                context_recall,
            ]

        return metrics

    def _validate_eval_candidate(self, benchmark_config: BenchmarkConfig) -> None:
        """Validate that the eval candidate is supported."""
        eval_candidate = benchmark_config.eval_candidate
        if eval_candidate.type != "model":
            raise RagasEvaluationError(
                "Ragas currently only supports model candidates. "
                "We will add support for agents soon!"
            )

    def _get_benchmark(self, benchmark_id: str) -> Benchmark:
        """Get benchmark by ID with error handling."""
        if benchmark_id not in self.benchmarks:
            raise RagasEvaluationError(f"Benchmark {benchmark_id} not found")
        return self.benchmarks[benchmark_id]

    @abstractmethod
    async def run_eval(self, request: RunEvalRequest) -> Job:
        """Run an evaluation on a benchmark."""
        ...

    async def evaluate_rows(self, request: EvaluateRowsRequest) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark."""
        raise NotImplementedError(
            "evaluate_rows is not implemented, use run_eval instead"
        )

    @abstractmethod
    async def job_status(self, request: JobStatusRequest) -> Job:
        """Get the status of a job."""
        ...

    @abstractmethod
    async def job_cancel(self, request: JobCancelRequest) -> None:
        """Cancel a job."""
        ...

    @abstractmethod
    async def job_result(self, request: JobResultRequest) -> EvaluateResponse:
        """Get the result of a job."""
        ...

    async def register_benchmark(self, task_def: Benchmark) -> None:
        """Register a benchmark for evaluation."""
        self.benchmarks[task_def.identifier] = task_def
        logger.info(f"Registered benchmark: {task_def.identifier}")

    async def unregister_benchmark(self, benchmark_id: str) -> None:
        """Unregister a benchmark."""
        removed = self.benchmarks.pop(benchmark_id, None)
        if removed is not None:
            logger.info(f"Unregistered benchmark: {benchmark_id}")
        else:
            logger.info(f"Benchmark not found (nothing to unregister): {benchmark_id}")
