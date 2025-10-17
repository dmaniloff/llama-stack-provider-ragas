import asyncio
import functools as ft
import logging
from typing import Any

from llama_stack.apis.benchmarks import Benchmark
from llama_stack.apis.common.job_types import Job, JobStatus
from llama_stack.apis.datasetio import DatasetIO
from llama_stack.apis.eval import BenchmarkConfig, Eval, EvaluateResponse
from llama_stack.apis.inference import Inference
from llama_stack.apis.scoring import ScoringResult
from llama_stack.providers.datatypes import BenchmarksProtocolPrivate
from llama_stack.schema_utils import json_schema_type
from ragas import EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    Metric,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from ..config import RagasProviderInlineConfig
from ..constants import METRIC_MAPPING
from ..errors import RagasEvaluationError
from ..logging_utils import render_dataframe_as_table
from .wrappers_inline import LlamaStackInlineEmbeddings, LlamaStackInlineLLM

logger = logging.getLogger(__name__)


@json_schema_type
class RagasEvaluationJob(Job):
    """Ragas evaluation job. Keeps track of the evaluation result."""

    # TODO: maybe propose this change to Job itself
    result: EvaluateResponse | None
    eval_config: RagasProviderInlineConfig


# TODO: maybe unify in __init__.py
@json_schema_type
class EmptyEvaluateResponse(EvaluateResponse):
    generations: list[dict[str, Any]] = []
    scores: dict[str, ScoringResult] = {}


class RagasEvaluatorInline(Eval, BenchmarksProtocolPrivate):
    def __init__(
        self,
        config: RagasProviderInlineConfig,
        datasetio_api: DatasetIO,
        inference_api: Inference,
    ):
        self.config = config
        self.datasetio_api = datasetio_api
        self.inference_api = inference_api
        self.evaluation_jobs: dict[str, RagasEvaluationJob] = {}
        self.benchmarks: dict[str, Benchmark] = {}

    async def run_eval(
        self,
        benchmark_id: str,
        benchmark_config: BenchmarkConfig,
    ) -> Job:
        eval_candidate = benchmark_config.eval_candidate
        if eval_candidate.type != "model":
            raise RagasEvaluationError(
                "Ragas currently only supports model candidates. "
                "We will add support for agents soon!"
            )

        model_id = benchmark_config.eval_candidate.model
        sampling_params = eval_candidate.sampling_params

        # for now, inline evals are hardcoded to run with max_workers=1
        ragas_run_config = RunConfig(max_workers=1)

        llm_wrapper = LlamaStackInlineLLM(
            self.inference_api, model_id, sampling_params, run_config=ragas_run_config
        )
        embeddings_wrapper = LlamaStackInlineEmbeddings(
            self.inference_api, self.config.embedding_model, run_config=ragas_run_config
        )

        task_def = self.benchmarks[benchmark_id]  # TODO: add error handling
        dataset_id = task_def.dataset_id
        scoring_functions = task_def.scoring_functions
        metrics = self._get_metrics(scoring_functions)
        eval_dataset = await self._prepare_dataset(
            dataset_id, benchmark_config.num_examples
        )

        ragas_evaluation_task = asyncio.create_task(
            self._run_ragas_evaluation(
                eval_dataset,
                llm_wrapper,
                embeddings_wrapper,
                metrics,
                ragas_run_config,
            )
        )

        job_id = str(len(self.evaluation_jobs))
        job = RagasEvaluationJob(
            job_id=job_id,
            status=JobStatus.in_progress,
            result=None,
            eval_config=self.config,
        )
        ragas_evaluation_task.add_done_callback(
            ft.partial(self._handle_evaluation_completion, job)
        )
        self.evaluation_jobs[job_id] = job
        return job

    def _get_metrics(self, scoring_functions: list[str]) -> list[Metric]:
        """Get the list of metrics to run based on scoring functions.

        Args:
            scoring_functions: List of scoring function names to use

        Returns:
            List of metrics (unconfigured - ragas_evaluate will configure them)
        """
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

    async def _prepare_dataset(
        self, dataset_id: str, limit: int = -1
    ) -> EvaluationDataset:
        all_rows = await self.datasetio_api.iterrows(
            dataset_id=dataset_id,
            limit=limit,
        )
        return EvaluationDataset.from_list(all_rows.data)

    async def _run_ragas_evaluation(
        self,
        eval_dataset: EvaluationDataset,
        llm_wrapper: LlamaStackInlineLLM,
        embeddings_wrapper: LlamaStackInlineEmbeddings,
        metrics: list[Metric],
        ragas_run_config: RunConfig,
    ) -> EvaluateResponse:
        result = await asyncio.to_thread(
            ragas_evaluate,
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm_wrapper,
            embeddings=embeddings_wrapper,
            experiment_name=self.config.ragas_config.experiment_name,
            run_config=ragas_run_config,
            raise_exceptions=self.config.ragas_config.raise_exceptions,
            column_map=self.config.ragas_config.column_map,
            show_progress=self.config.ragas_config.show_progress,
            batch_size=self.config.ragas_config.batch_size,
        )
        result_df = result.to_pandas()
        table_output = render_dataframe_as_table(result_df, "Ragas Evaluation Results")
        logger.info(f"Ragas evaluation completed:\n{table_output}")

        # Convert scores to ScoringResult format
        scores = {}
        for metric_name in [m.name for m in metrics]:
            metric_scores = result[metric_name]
            score_rows = [{"score": score} for score in metric_scores]

            if metric_scores:
                aggregated_score = sum(metric_scores) / len(metric_scores)
            else:
                aggregated_score = 0.0

            scores[metric_name] = ScoringResult(
                score_rows=score_rows,
                aggregated_results={metric_name: aggregated_score},
            )

        logger.info(f"Evaluation completed. Scores: {scores}")
        return EvaluateResponse(generations=eval_dataset.to_list(), scores=scores)

    def _handle_evaluation_completion(
        self, job: RagasEvaluationJob, task: asyncio.Task
    ):
        try:
            result = task.result()
        except Exception as e:
            logger.error(f"Evaluation task failed: {e}")
            job.status = JobStatus.failed
        else:
            job.status = JobStatus.completed
            job.result = result

    async def evaluate_rows(
        self,
        benchmark_id: str,
        input_rows: list[dict[str, Any]],
        scoring_functions: list[str],
        benchmark_config: BenchmarkConfig,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark."""
        raise NotImplementedError(
            "evaluate_rows is not implemented, use run_eval instead"
        )

    async def job_status(self, benchmark_id: str, job_id: str) -> Job:
        """Get the status of a job.

        Args:
            benchmark_id: The ID of the benchmark to run the evaluation on.
            job_id: The ID of the job to get the status of.

        Returns:
            The status of the evaluation job.
        """
        if (job := self.evaluation_jobs.get(job_id)) is None:
            raise RagasEvaluationError(f"Job {job_id} not found")

        return job

    async def job_cancel(self, benchmark_id: str, job_id: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, benchmark_id: str, job_id: str) -> EvaluateResponse:
        job = await self.job_status(benchmark_id, job_id)

        if job.status == JobStatus.completed:
            return job.result
        elif job.status == JobStatus.failed:
            logger.warning(f"Job {job_id} failed")
        else:
            logger.warning(f"Job {job_id} is still running")

        # TODO: propose enhancement to EvaluateResponse to include a status?
        return EmptyEvaluateResponse()

    async def register_benchmark(self, task_def: Benchmark) -> None:
        self.benchmarks[task_def.identifier] = task_def
        logger.info(f"Registered benchmark: {task_def.identifier}")
