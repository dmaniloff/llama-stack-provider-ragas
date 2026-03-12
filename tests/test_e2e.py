"""End-to-end tests for the llama-stack-provider-ragas distribution on OpenShift.

Prerequisites:
    - OpenShift cluster with the e2e environment deployed (see cluster-deployment directory)
    - Port-forward active:
        oc port-forward -n ragas-test svc/lsd-ragas-test-service 8321:8321

Environment variables:
    LLAMA_STACK_BASE_URL  - Llama Stack server URL (default: http://localhost:8321)
    INFERENCE_MODEL       - Model ID for eval candidate (default: Mistral-Small-24B-W8A8)
"""

import os

import pytest
from base_eval_tests import EvalTester, SmokeTester
from llama_stack_client import LlamaStackClient

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def client(llama_stack_base_url):
    return LlamaStackClient(base_url=llama_stack_base_url)


@pytest.fixture(scope="module")
def inference_model():
    # Default must match cluster-deployment/manifests/configmap-and-secrets.yaml
    return os.getenv("INFERENCE_MODEL", "Mistral-Small-24B-W8A8")


@pytest.fixture(scope="module")
def embedding_model():
    # Default must match cluster-deployment/manifests/configmap-and-secrets.yaml
    return os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")


@pytest.fixture(scope="module")
def smoke_tester(client, dataset_id, inline_benchmark_id, remote_benchmark_id):
    return SmokeTester(
        client,
        dataset_id,
        inline_benchmark_id,
        remote_benchmark_id,
    )


@pytest.fixture(scope="module")
def eval_tester(
    client, inference_model, dataset_id, inline_benchmark_id, remote_benchmark_id
):
    return EvalTester(
        client,
        inference_model,
        dataset_id,
        inline_benchmark_id,
        remote_benchmark_id,
    )


@pytest.mark.usefixtures("register_benchmarks")
def test_cluster_smoke(smoke_tester):
    smoke_tester.test_providers_registered()
    smoke_tester.test_models_registered()
    smoke_tester.test_datasets_registered()
    smoke_tester.test_benchmarks_registered()


@pytest.mark.usefixtures("register_benchmarks")
def test_inline_eval(eval_tester, inline_benchmark_id, inference_model):
    eval_tester.poll_interval = 3
    eval_tester.poll_timeout = 30
    eval_tester.run_eval(inline_benchmark_id, inference_model, num_examples=3)


@pytest.mark.usefixtures("register_benchmarks")
def test_remote_eval(eval_tester, remote_benchmark_id, inference_model):
    eval_tester.poll_interval = 10
    eval_tester.poll_timeout = 300
    eval_tester.run_eval(remote_benchmark_id, inference_model, num_examples=3)
