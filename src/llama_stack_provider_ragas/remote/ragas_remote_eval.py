import logging
import uuid

import pandas as pd
import requests
from pydantic import BaseModel

from ..base import EmptyEvaluateResponse, RagasEvaluatorBase
from ..compat import (
    Benchmark,
    BenchmarkConfig,
    EvaluateResponse,
    Job,
    JobCancelRequest,
    JobResultRequest,
    JobStatus,
    JobStatusRequest,
    RunEvalRequest,
    ScoringResult,
    json_schema_type,
)
from ..config import (
    KubeflowConfig,
    RagasConfig,
    RagasProviderRemoteConfig,
)
from ..constants import AVAILABLE_METRICS
from ..errors import RagasEvaluationError
from ..logging_utils import render_dataframe_as_table

logger = logging.getLogger(__name__)


@json_schema_type
class RagasEvaluationJobRuntimeConfig(BaseModel):
    benchmark_config: BenchmarkConfig
    embedding_model: str
    benchmark: Benchmark
    ragas_config: RagasConfig
    kubeflow_config: KubeflowConfig


@json_schema_type
class RagasEvaluationJob(Job):
    """Llama Stack Job with some additional information."""

    runtime_config: RagasEvaluationJobRuntimeConfig
    kubeflow_run_id: str | None = None
    result: EvaluateResponse | None

    @property
    def result_s3_location(self) -> str:
        return f"{self.runtime_config.kubeflow_config.results_s3_prefix.rstrip('/')}/{self.job_id}/results.jsonl"


class RagasEvaluatorRemote(RagasEvaluatorBase):
    """Execute Ragas evaluations using Kubeflow Pipelines."""

    def __init__(
        self,
        config: RagasProviderRemoteConfig,
    ):
        super().__init__()
        self.config = config
        self.evaluation_jobs: dict[str, RagasEvaluationJob] = {}
        self._kfp_client = None

    @property
    def kfp_client(self):
        """Lazy-initialized KFP client. Deferred to eval runtime."""
        if self._kfp_client is None:
            try:
                import kfp

                token = self._get_kfp_token()
                if not token:
                    raise RagasEvaluationError(
                        "No token found. Please run `oc login` and try again."
                    )

                # the kfp.Client handles the healthz endpoint poorly, run a pre-flight check manually
                response = requests.get(
                    f"{self.config.kubeflow_config.pipelines_endpoint}/apis/v2beta1/healthz",
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {token}",
                    },
                    timeout=5,
                )
                response.raise_for_status()

                self._kfp_client = kfp.Client(
                    host=self.config.kubeflow_config.pipelines_endpoint,
                    existing_token=token,
                )
            except ImportError as e:
                raise RagasEvaluationError(
                    "Kubeflow Pipelines SDK not available. Install with: pip install .[remote]"
                ) from e
            except requests.exceptions.RequestException as e:
                raise RagasEvaluationError(
                    f"Failed to connect to Kubeflow Pipelines server at {self.config.kubeflow_config.pipelines_endpoint}, "
                    "do you need a new token?"
                ) from e
            except Exception as e:
                raise RagasEvaluationError(
                    "Failed to initialize Kubeflow Pipelines client."
                ) from e

        return self._kfp_client

    def _get_kfp_token(self) -> str:
        if self.config.kubeflow_config.pipelines_api_token:
            logger.info("Using KUBEFLOW_PIPELINES_TOKEN from config")
            return str(
                self.config.kubeflow_config.pipelines_api_token.get_secret_value()
            )

        try:
            from .kubeflow.utils import _load_kube_config

            kube_config = _load_kube_config()
            token = str(kube_config.api_key["authorization"].split(" ")[-1])
        except ImportError as e:
            raise RagasEvaluationError(
                "Kubernetes client is not installed. Install with: pip install .[remote]"
            ) from e
        except Exception as e:
            raise RagasEvaluationError(
                "Failed to get OpenShift token. Please run `oc login` and try again."
            ) from e

        return token

    async def run_eval(
        self,
        request: RunEvalRequest,
    ) -> Job:
        """Submit a Ragas evaluation job to Kubeflow Pipelines."""
        try:
            benchmark_id = request.benchmark_id
            benchmark_config = request.benchmark_config

            # Use base class validation
            self._validate_eval_candidate(benchmark_config)
            task_def = self._get_benchmark(benchmark_id)

            job_id = str(uuid.uuid4())
            job = RagasEvaluationJob(
                job_id=job_id,
                status=JobStatus.in_progress,
                result=None,
                kubeflow_run_id=None,
                pipeline_status="submitted",
                runtime_config=RagasEvaluationJobRuntimeConfig(
                    benchmark=task_def,
                    benchmark_config=benchmark_config,
                    embedding_model=self.config.embedding_model,
                    ragas_config=self.config.ragas_config,
                    kubeflow_config=self.config.kubeflow_config,
                ),
            )

            kubeflow_run_id = await self._submit_to_kubeflow(job)
            job.kubeflow_run_id = kubeflow_run_id
            self.evaluation_jobs[job_id] = job

            logger.info(
                f"Submitted Ragas evaluation job {job_id} to Kubeflow with run ID {kubeflow_run_id}"
            )

            return job

        except Exception as e:
            logger.error(f"Failed to submit evaluation job: {str(e)}")
            raise RagasEvaluationError(f"Failed to submit evaluation: {str(e)}") from e

    async def _submit_to_kubeflow(self, job: RagasEvaluationJob) -> str:
        from .kubeflow.pipeline import ragas_evaluation_pipeline

        pipeline_args = {
            "dataset_id": job.runtime_config.benchmark.dataset_id,
            "llama_stack_base_url": job.runtime_config.kubeflow_config.llama_stack_url,
            "num_examples": (
                job.runtime_config.benchmark_config.num_examples
                if job.runtime_config.benchmark_config.num_examples is not None
                else -1
            ),
            "model": job.runtime_config.benchmark_config.eval_candidate.model,
            "sampling_params": job.runtime_config.benchmark_config.eval_candidate.sampling_params.model_dump(
                exclude_none=True
            ),
            "embedding_model": self.config.embedding_model,
            "metrics": job.runtime_config.benchmark.scoring_functions,
            "result_s3_location": job.result_s3_location,
            "results_s3_storage_options": job.runtime_config.kubeflow_config.results_s3_storage_options,
            "s3_credentials_secret_name": job.runtime_config.kubeflow_config.s3_credentials_secret_name,
        }

        run_result = self.kfp_client.create_run_from_pipeline_func(
            pipeline_func=ragas_evaluation_pipeline,
            arguments=pipeline_args,
            run_name=f"ragas-eval-{job.runtime_config.benchmark.benchmark_id}-{job.job_id[:8]}",
            namespace=job.runtime_config.kubeflow_config.namespace,
            experiment_name="lls-provider-ragas-runs",
        )

        return run_result.run_id  # type: ignore

    async def job_status(self, request: JobStatusRequest) -> Job:
        # TODO: replace inmem dict with kubeflow client
        if (job := self.evaluation_jobs.get(request.job_id)) is None:
            raise RagasEvaluationError(f"Job {request.job_id} not found")

        try:
            run_detail = self.kfp_client.get_run(job.kubeflow_run_id)
        except Exception as e:
            # TODO: handle expired token issues
            logger.error(f"Failed to get job status: {str(e)}")
            raise RagasEvaluationError(f"Failed to get job status: {str(e)}") from e
        else:
            if run_detail.state == "FAILED":
                # TODO: add error message
                job.status = JobStatus.failed
            elif run_detail.state == "RUNNING" or run_detail.state == "PENDING":
                job.status = JobStatus.in_progress
            elif run_detail.state == "SUCCEEDED":
                job.status = JobStatus.completed
                try:
                    await self._fetch_kubeflow_results(job)
                except Exception as e:
                    raise RagasEvaluationError(
                        f"Run was successful, but failed to fetch results: {str(e)}"
                    ) from e

            else:
                raise RagasEvaluationError(
                    f"Unknown Kubeflow run state: {run_detail.state}"
                )

        return job

    async def _fetch_kubeflow_results(self, job: RagasEvaluationJob) -> None:
        """Fetch results directly from S3."""
        try:
            result_df = pd.read_json(
                job.result_s3_location,
                lines=True,
                storage_options=job.runtime_config.kubeflow_config.results_s3_storage_options,
            )
            logger.info(f"Successfully fetched results from {job.result_s3_location}")
        except Exception as e:
            raise RagasEvaluationError(
                f"Failed to fetch results from {job.result_s3_location}: {str(e)}"
            ) from e

        table_output = render_dataframe_as_table(
            result_df, "Fetched Evaluation Results"
        )
        logger.info(f"Fetched Evaluation Results:\n{table_output}")

        metric_columns = [
            col
            for col in result_df.columns
            if col in job.runtime_config.benchmark.scoring_functions
        ]
        generation_columns = result_df.columns.difference(metric_columns)
        generations = result_df[generation_columns].to_dict("records")
        scores = {}

        for metric_name in metric_columns:
            metric_scores = result_df[metric_name].dropna().tolist()
            score_rows = [{"score": score} for score in metric_scores]

            scores[metric_name] = ScoringResult(
                score_rows=score_rows,
                aggregated_results={
                    "average": sum(metric_scores) / len(metric_scores)
                    if metric_scores
                    else 0.0,
                    "count": len(metric_scores),
                    "min": min(metric_scores) if metric_scores else 0.0,
                    "max": max(metric_scores) if metric_scores else 0.0,
                },
            )

        job.result = EvaluateResponse(generations=generations, scores=scores)

        logger.info(
            f"Successfully fetched results for job {job.job_id}: {len(generations)} generations, {len(scores)} metrics"
        )

    async def job_cancel(self, request: JobCancelRequest) -> None:
        if (job := self.evaluation_jobs.get(request.job_id)) is None:
            raise RagasEvaluationError(f"Job {request.job_id} not found")

        try:
            self.kfp_client.runs.terminate_run(job.kubeflow_run_id)
            job.status = JobStatus.cancelled
            logger.info(
                f"Cancelled Kubeflow run {job.kubeflow_run_id} for job {request.job_id}"
            )
        except Exception as e:
            logger.error(f"Failed to cancel job: {str(e)}")
            raise RagasEvaluationError(f"Failed to cancel job: {str(e)}") from e

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
        """Register a benchmark for evaluation."""
        if not all(
            metric in AVAILABLE_METRICS for metric in benchmark.scoring_functions
        ):
            raise RagasEvaluationError(
                f"Invalid metrics: {benchmark.scoring_functions}. "
                f"Available metrics: {AVAILABLE_METRICS}"
            )
        # Call parent implementation
        await super().register_benchmark(benchmark)
