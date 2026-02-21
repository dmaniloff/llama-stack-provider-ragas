"""End-to-end tests for the llama-stack-provider-ragas distribution on OpenShift.

Prerequisites:
    - OpenShift cluster with the e2e environment deployed (see deploy-e2e.sh)
    - Port-forward active:
        oc port-forward -n ragas-test svc/lsd-ragas-test-service 8321:8321

Environment variables:
    LLAMA_STACK_BASE_URL  - Llama Stack server URL (default: http://localhost:8321)
    INFERENCE_MODEL       - Model ID for eval candidate (default: Mistral-Small-24B-W8A8)
"""

import os
import time

import pytest
from llama_stack_client import LlamaStackClient
from rich import print as pprint

# Pre-registered resource IDs (must match llama-stack-distribution.yaml)
INLINE_BENCHMARK_ID = "hf-doc-qa-ragas-inline-benchmark"
REMOTE_BENCHMARK_ID = "hf-doc-qa-ragas-remote-benchmark"
DATASET_ID = "hf_doc_qa_ragas_eval"

POLL_INTERVAL = 5  # seconds
POLL_TIMEOUT = 300  # seconds
REMOTE_POLL_TIMEOUT = 600  # seconds – pipeline pods need to pull images and install packages

RAW_EVALUATION_DATA = [
    {
        "user_input": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "retrieved_contexts": [
            "Paris is the capital and most populous city of France."
        ],
        "reference": "Paris",
    },
    {
        "user_input": "Who invented the telephone?",
        "response": "Alexander Graham Bell invented the telephone in 1876.",
        "retrieved_contexts": [
            "Alexander Graham Bell was a Scottish-American inventor who patented the first practical telephone."
        ],
        "reference": "Alexander Graham Bell",
    },
    {
        "user_input": "What is photosynthesis?",
        "response": "Photosynthesis is the process by which plants convert sunlight into energy.",
        "retrieved_contexts": [
            "Photosynthesis is a process used by plants to convert light energy into chemical energy."
        ],
        "reference": "Photosynthesis is the process by which plants and other organisms convert light energy into chemical energy.",
    },
]


@pytest.fixture(scope="module")
def client():
    base_url = os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")
    return LlamaStackClient(base_url=base_url)


@pytest.fixture(scope="module")
def inference_model():
    return os.getenv("INFERENCE_MODEL", "Mistral-Small-24B-W8A8")


@pytest.fixture(scope="module", autouse=True)
def register_dataset(client):
    """Register the evaluation dataset with inline rows."""
    client.beta.datasets.register(
        dataset_id=DATASET_ID,
        purpose="eval/messages-answer",
        source={"type": "rows", "rows": RAW_EVALUATION_DATA},
    )
    yield
    client.beta.datasets.unregister(dataset_id=DATASET_ID)


def _wait_for_job(client, benchmark_id, job_id, timeout=POLL_TIMEOUT):
    """Poll until the eval job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.alpha.eval.jobs.status(benchmark_id=benchmark_id, job_id=job_id)
        pprint("Job details:", job)
        if job.status in ("completed", "failed"):
            return job
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(
        f"Job {job_id} for benchmark {benchmark_id} did not complete within {timeout}s"
    )


class TestClusterSmoke:
    """Verify the cluster has the expected resources registered."""

    def test_models_registered(self, client):
        models = client.models.list()
        assert len(models) > 0, "No models registered"

    def test_datasets_registered(self, client):
        datasets = client.beta.datasets.list()
        dataset_ids = [d.identifier for d in datasets]
        assert DATASET_ID in dataset_ids, (
            f"Dataset '{DATASET_ID}' not found. Available: {dataset_ids}"
        )

    def test_benchmarks_registered(self, client):
        benchmarks = client.alpha.benchmarks.list()
        benchmark_ids = [b.identifier for b in benchmarks]
        assert INLINE_BENCHMARK_ID in benchmark_ids, (
            f"Benchmark '{INLINE_BENCHMARK_ID}' not found. Available: {benchmark_ids}"
        )


class TestInlineEval:
    """Run evaluation using the inline ragas provider."""

    def test_run_eval(self, client, inference_model):
        job = client.alpha.eval.run_eval(
            benchmark_id=INLINE_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": inference_model,
                    "sampling_params": {
                        "temperature": 0.1,
                        "max_tokens": 100,
                    },
                },
                "scoring_params": {},
                "num_examples": 3,
            },
        )
        assert job.job_id is not None
        assert job.status == "in_progress"

        completed = _wait_for_job(client, INLINE_BENCHMARK_ID, job.job_id)
        assert completed.status == "completed", (
            f"Job finished with status '{completed.status}'"
        )

        results = client.alpha.eval.jobs.retrieve(
            benchmark_id=INLINE_BENCHMARK_ID, job_id=job.job_id
        )
        assert results.scores, "Expected non-empty scores"


class TestRemoteEval:
    """Run evaluation using the remote ragas provider (KFP + MinIO)."""

    def test_run_eval(self, client, inference_model):
        job = client.alpha.eval.run_eval(
            benchmark_id=REMOTE_BENCHMARK_ID,
            benchmark_config={
                "eval_candidate": {
                    "type": "model",
                    "model": inference_model,
                    "sampling_params": {
                        "temperature": 0.1,
                        "max_tokens": 100,
                    },
                },
                "scoring_params": {},
                "num_examples": 3,
            },
        )
        assert job.job_id is not None
        assert job.status == "in_progress"

        completed = _wait_for_job(
            client, REMOTE_BENCHMARK_ID, job.job_id, timeout=REMOTE_POLL_TIMEOUT
        )
        assert completed.status == "completed", (
            f"Job finished with status '{completed.status}'"
        )

        results = client.alpha.eval.jobs.retrieve(
            benchmark_id=REMOTE_BENCHMARK_ID, job_id=job.job_id
        )
        assert results.scores, "Expected non-empty scores"
