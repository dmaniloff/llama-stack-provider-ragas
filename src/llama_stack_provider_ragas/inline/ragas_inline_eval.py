import asyncio
import functools as ft
import logging

from ragas import EvaluationDataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import Metric
from ragas.run_config import RunConfig

from ..base import EmptyEvaluateResponse, RagasEvaluatorBase
from ..compat import (
    Benchmark,
    DatasetIO,
    EvaluateResponse,
    EvaluateRowsRequest,
    Inference,
    IterRowsRequest,
    Job,
    JobCancelRequest,
    JobResultRequest,
    JobStatus,
    JobStatusRequest,
    RunEvalRequest,
    ScoringResult,
    json_schema_type,
)
from ..config import RagasProviderInlineConfig
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


class RagasEvaluatorInline(RagasEvaluatorBase):
    def __init__(
        self,
        config: RagasProviderInlineConfig,
        datasetio_api: DatasetIO,
        inference_api: Inference,
    ):
        super().__init__()
        self.config = config
        self.datasetio_api = datasetio_api
        self.inference_api = inference_api
        self.evaluation_jobs: dict[str, RagasEvaluationJob] = {}

    async def run_eval(
        self,
        request: RunEvalRequest,
    ) -> Job:
        benchmark_id = request.benchmark_id
        benchmark_config = request.benchmark_config

        # Use base class validation
        self._validate_eval_candidate(benchmark_config)

        model_id = benchmark_config.eval_candidate.model
        sampling_params = benchmark_config.eval_candidate.sampling_params

        # for now, inline evals are hardcoded to run with max_workers=1
        ragas_run_config = RunConfig(max_workers=1)

        llm_wrapper = LlamaStackInlineLLM(
            self.inference_api, model_id, sampling_params, run_config=ragas_run_config
        )
        embeddings_wrapper = LlamaStackInlineEmbeddings(
            self.inference_api, self.config.embedding_model, run_config=ragas_run_config
        )

        task_def = self._get_benchmark(benchmark_id)
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

    async def _prepare_dataset(
        self, dataset_id: str, limit: int | None = None
    ) -> EvaluationDataset:
        all_rows = await self.datasetio_api.iterrows(
            IterRowsRequest(
                dataset_id=dataset_id,
                limit=limit,
            )
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
        request: EvaluateRowsRequest,
    ) -> EvaluateResponse:
        """Evaluate a list of rows on a benchmark."""
        raise NotImplementedError(
            "evaluate_rows is not implemented, use run_eval instead"
        )

    async def job_status(self, request: JobStatusRequest) -> Job:
        """Get the status of a job.

        Args:
            request: The job status request containing benchmark_id and job_id.

        Returns:
            The status of the evaluation job.
        """
        if (job := self.evaluation_jobs.get(request.job_id)) is None:
            raise RagasEvaluationError(f"Job {request.job_id} not found")

        return job

    async def job_cancel(self, request: JobCancelRequest) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def job_result(self, request: JobResultRequest) -> EvaluateResponse:
        job = await self.job_status(
            JobStatusRequest(benchmark_id=request.benchmark_id, job_id=request.job_id)
        )

        if job.status == JobStatus.completed:
            return job.result
        elif job.status == JobStatus.failed:
            logger.warning(f"Job {request.job_id} failed")
        else:
            logger.warning(f"Job {request.job_id} is still running")

        # TODO: propose enhancement to EvaluateResponse to include a status?
        return EmptyEvaluateResponse()

    async def register_benchmark(self, benchmark: Benchmark) -> None:
        self.benchmarks[benchmark.identifier] = benchmark
        logger.info(f"Registered benchmark: {benchmark.identifier}")
