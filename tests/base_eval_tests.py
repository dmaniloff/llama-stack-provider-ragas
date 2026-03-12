"""Shared test helpers for Llama Stack eval provider tests.

Provides ``SmokeTester`` and ``EvalTester``, plain helper classes that
encapsulate common assertions (model/dataset/benchmark registration) and
eval-job execution logic (run, poll, verify scores).  Test modules
instantiate them via fixtures, supplying the appropriate client and
configuration for each environment (in-process library client or remote
``LlamaStackClient``).
"""

import time

from rich import print as pprint


class SmokeTester:
    def __init__(self, client, dataset_id, inline_benchmark_id, remote_benchmark_id):
        self.client = client
        self.dataset_id = dataset_id
        self.inline_benchmark_id = inline_benchmark_id
        self.remote_benchmark_id = remote_benchmark_id

    def test_providers_registered(self):
        providers = self.client.providers.list()
        assert len(providers) > 0
        assert any(p.api == "eval" for p in providers)
        pprint("Providers:", providers)

    def test_models_registered(self):
        models = self.client.models.list()
        pprint("Models:", models)
        assert len(models) > 0, "No models registered"

    def test_datasets_registered(self):
        datasets = self.client.beta.datasets.list()
        pprint("Datasets:", datasets)
        dataset_ids = [d.identifier for d in datasets]
        assert self.dataset_id in dataset_ids, (
            f"Dataset '{self.dataset_id}' not found. Available: {dataset_ids}"
        )

    def test_benchmarks_registered(self):
        benchmarks = self.client.alpha.benchmarks.list()
        pprint("Benchmarks:", benchmarks)
        benchmark_ids = [b.identifier for b in benchmarks]
        assert self.inline_benchmark_id in benchmark_ids, (
            f"Benchmark '{self.inline_benchmark_id}' not found. Available: {benchmark_ids}"
        )
        assert self.remote_benchmark_id in benchmark_ids, (
            f"Benchmark '{self.remote_benchmark_id}' not found. Available: {benchmark_ids}"
        )


class EvalTester:
    """Base evaluation test class."""

    def __init__(
        self,
        client,
        inference_model,
        dataset_id,
        inline_benchmark_id,
        remote_benchmark_id,
        poll_interval: int = 5,
        poll_timeout: int = 300,
    ):
        self.client = client
        self.inference_model = inference_model
        self.dataset_id = dataset_id
        self.inline_benchmark_id = inline_benchmark_id
        self.remote_benchmark_id = remote_benchmark_id
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout

    def run_eval(
        self,
        benchmark_id: str,
        inference_model: str,
        num_examples: int | None = None,
    ):
        """Run an evaluation job and verify it completes with scores."""
        benchmark_config = self._build_benchmark_config(
            inference_model, num_examples=num_examples
        )
        job = self.client.alpha.eval.run_eval(
            benchmark_id=benchmark_id,
            benchmark_config=benchmark_config,
        )
        assert job.job_id is not None
        assert job.status == "in_progress"

        completed = self._wait_for_job(self.client, benchmark_id, job.job_id)
        assert completed.status == "completed", (
            f"Job finished with status '{completed.status}'"
        )

        results = self.client.alpha.eval.jobs.retrieve(
            benchmark_id=benchmark_id, job_id=job.job_id
        )
        pprint(f"[{self.__class__.__name__}] Results:", results)
        assert results.scores, "Expected non-empty scores"

    # -- helpers --------------------------------------------------------

    def _build_benchmark_config(
        self, inference_model: str, num_examples: int | None = None
    ) -> dict:
        """Build the ``benchmark_config`` dict for ``run_eval``."""
        config: dict = {
            "eval_candidate": {
                "type": "model",
                "model": inference_model,
                "sampling_params": {
                    "temperature": 0.1,
                    "max_tokens": 100,
                },
            },
            "scoring_params": {},
        }
        if num_examples is not None:
            config["num_examples"] = num_examples
        return config

    def _wait_for_job(
        self, client, benchmark_id: str, job_id: str, timeout: int | None = None
    ):
        """Poll until the eval job reaches a terminal state."""
        timeout = timeout if timeout is not None else self.poll_timeout
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = client.alpha.eval.jobs.status(
                benchmark_id=benchmark_id, job_id=job_id
            )
            pprint(f"[{self.__class__.__name__}] Job status:", job)
            if job.status in ("completed", "failed"):
                return job
            time.sleep(self.poll_interval)
        raise TimeoutError(
            f"Job {job_id} for benchmark {benchmark_id} "
            f"did not complete within {timeout}s"
        )
