"""Llama Stack integration tests using an in-process server.

These tests use ``LlamaStackAsLibraryClient`` to spin up a Llama Stack
server in-process with the configuration defined in the
``library_stack_config`` fixture.  By default, LLM inference (embeddings
and completions) is mocked so no external services are required.

To run against a real inference provider (e.g. a local Ollama instance),
pass ``--no-mock-inference``::

    pytest tests/test_inline_evaluation.py --no-mock-inference

You can also override the models via environment variables::

    INFERENCE_MODEL=ollama/granite3.3:2b \\
    EMBEDDING_MODEL=ollama/all-minilm:latest \\
        pytest tests/test_inline_evaluation.py --no-mock-inference
"""

import os
import random
from types import SimpleNamespace

import pytest
import yaml
from base_eval_tests import EvalTester, SmokeTester
from llama_stack.core.library_client import LlamaStackAsLibraryClient

from llama_stack_provider_ragas.compat import Api
from llama_stack_provider_ragas.constants import PROVIDER_ID_INLINE, PROVIDER_ID_REMOTE

pytestmark = pytest.mark.lls_integration


@pytest.fixture
def library_stack_config(
    tmp_path, embedding_dimension, embedding_model, inference_model
):
    """Stack configuration for library client testing."""
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir()

    return {
        "version": 2,
        "distro_name": "test_ragas_inline",
        "apis": ["eval", "inference", "files", "datasetio"],
        "providers": {
            "inference": [
                {
                    "provider_id": "ollama",
                    "provider_type": "remote::ollama",
                    "config": {"url": "http://localhost:11434"},
                }
            ],
            "eval": [
                {
                    "provider_id": PROVIDER_ID_INLINE,
                    "provider_type": "inline::trustyai_ragas",
                    "module": "llama_stack_provider_ragas.inline",
                    "config": {
                        "embedding_model": embedding_model,
                        "kvstore": {"namespace": "ragas", "backend": "kv_default"},
                    },
                },
                {
                    "provider_id": PROVIDER_ID_REMOTE,
                    "provider_type": "remote::trustyai_ragas",
                    "module": "llama_stack_provider_ragas.remote",
                    "config": {
                        "embedding_model": embedding_model,
                        "kubeflow_config": {
                            "pipelines_endpoint": "http://localhost:8888",
                            "namespace": "default",
                            "llama_stack_url": "http://localhost:8321",
                            "base_image": "python:3.12-slim",
                            "results_s3_prefix": "s3://ragas-results",
                            "s3_credentials_secret_name": "aws-credentials",
                        },
                    },
                },
            ],
            "datasetio": [
                {
                    "provider_id": "localfs",
                    "provider_type": "inline::localfs",
                    "config": {
                        "kvstore": {
                            "namespace": "datasetio::localfs",
                            "backend": "kv_default",
                        }
                    },
                }
            ],
            "files": [
                {
                    "provider_id": "meta-reference-files",
                    "provider_type": "inline::localfs",
                    "config": {
                        "storage_dir": str(storage_dir / "files"),
                        "metadata_store": {
                            "table_name": "files_metadata",
                            "backend": "sql_default",
                        },
                    },
                }
            ],
        },
        "storage": {
            "backends": {
                "kv_default": {
                    "type": "kv_sqlite",
                    "db_path": str(storage_dir / "kv.db"),
                },
                "sql_default": {
                    "type": "sql_sqlite",
                    "db_path": str(storage_dir / "sql.db"),
                },
            },
            "stores": {
                "metadata": {"namespace": "registry", "backend": "kv_default"},
                "inference": {
                    "table_name": "inference_store",
                    "backend": "sql_default",
                    "max_write_queue_size": 10000,
                    "num_writers": 4,
                },
                "conversations": {
                    "table_name": "conversations",
                    "backend": "sql_default",
                },
            },
        },
        "registered_resources": {
            "models": [
                {
                    "metadata": {"embedding_dimension": embedding_dimension},
                    "model_id": embedding_model,
                    "provider_id": "ollama",
                    "provider_model_id": embedding_model.removeprefix("ollama/"),
                    "model_type": "embedding",
                },
                {
                    "metadata": {},
                    "model_id": inference_model,
                    "provider_id": "ollama",
                    "provider_model_id": inference_model.removeprefix("ollama/"),
                    "model_type": "llm",
                },
            ],
            "shields": [],
            "vector_dbs": [],
            "datasets": [],
            "scoring_fns": [],
            "benchmarks": [],
            "tool_groups": [],
        },
    }


@pytest.fixture(autouse=True)
def mocked_inference_response(request):
    """Fake completion text returned by the mocked Ollama inference adapter.

    The in-process library client's ``openai_completion`` and
    ``openai_embeddings`` methods are monkey-patched in
    ``mocked_library_client``; this fixture controls the text that the
    mocked completion endpoint returns.  Use indirect parametrization to
    override the default value per test.
    """
    return getattr(request, "param", "Hello, world!")


@pytest.fixture
def library_stack_config_file(library_stack_config, tmp_path):
    """Write the stack config dict to a temp YAML file and return its path."""

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.safe_dump(library_stack_config), encoding="utf-8")
    return config_file


@pytest.fixture
def library_client(request):
    """Return a library client, with or without mocked inference.

    By default, Ollama inference is mocked so no external services are
    needed.  Pass ``--no-mock-inference`` to use a real Ollama instance.

    The completion text used by the mock can be overridden via indirect
    parametrization of ``mocked_inference_response``::

        @pytest.mark.parametrize(
            "mocked_inference_response",
            ["Hello from mock!"],
            indirect=True,
        )
    """
    if request.config.getoption("--no-mock-inference") is True:
        return request.getfixturevalue("real_library_client")
    else:
        return request.getfixturevalue("mocked_library_client")


@pytest.fixture()
def real_library_client(library_stack_config_file):
    return LlamaStackAsLibraryClient(str(library_stack_config_file))


@pytest.fixture()
def mocked_library_client(
    monkeypatch,
    mocked_inference_response,
    library_stack_config_file,
    embedding_dimension,
    embedding_model,
    inference_model,
):
    completion_text = mocked_inference_response

    # Mock Ollama connectivity check & model listing
    async def _fake_check_model_availability(*args, **kwargs):
        return True

    async def _fake_list_provider_model_ids(*args, **kwargs):
        return [embedding_model, inference_model]

    monkeypatch.setattr("ollama.Client", lambda *args, **kwargs: SimpleNamespace())

    monkeypatch.setattr(
        "llama_stack.providers.remote.inference.ollama.ollama.OllamaInferenceAdapter.check_model_availability",
        _fake_check_model_availability,
    )
    monkeypatch.setattr(
        "llama_stack.providers.remote.inference.ollama.ollama.OllamaInferenceAdapter.list_provider_model_ids",
        _fake_list_provider_model_ids,
    )

    # Create the client after mocking
    real_library_client = LlamaStackAsLibraryClient(str(library_stack_config_file))

    async def _fake_openai_embeddings(req):  # noqa: ANN001
        embedding_input = getattr(req, "input", None)
        n = len(embedding_input) if isinstance(embedding_input, list) else 1
        data = [
            SimpleNamespace(
                embedding=[random.random() for _ in range(embedding_dimension)],
                index=i,
                object="embedding",
            )
            for i in range(n)
        ]
        return SimpleNamespace(
            data=data,
            model=getattr(req, "model", "mocked/model"),
            object="list",
            usage=SimpleNamespace(prompt_tokens=10, total_tokens=10),
        )

    async def _fake_openai_completion(req):  # noqa: ANN001
        choice = SimpleNamespace(
            index=0, text=completion_text, finish_reason="stop", logprobs=None
        )
        return SimpleNamespace(
            id="cmpl-123",
            created=1717000000,
            choices=[choice],
            model=getattr(req, "model", "mocked/model"),
            object="text_completion",
        )

    inference_impl = real_library_client.async_client.impls[Api.inference]
    monkeypatch.setattr(inference_impl, "openai_embeddings", _fake_openai_embeddings)
    monkeypatch.setattr(inference_impl, "openai_completion", _fake_openai_completion)

    return real_library_client


@pytest.fixture
def client(library_client):
    return library_client


@pytest.fixture
def inference_model():
    return os.getenv("INFERENCE_MODEL", "ollama/granite3.3:2b")


@pytest.fixture
def embedding_model():
    return os.getenv("EMBEDDING_MODEL", "ollama/all-minilm:latest")


@pytest.fixture
def smoke_tester(client, dataset_id, inline_benchmark_id, remote_benchmark_id):
    return SmokeTester(
        client,
        dataset_id,
        inline_benchmark_id,
        remote_benchmark_id,
    )


@pytest.fixture
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
def test_library_client_smoke(smoke_tester):
    smoke_tester.test_providers_registered()
    smoke_tester.test_models_registered()
    smoke_tester.test_datasets_registered()
    smoke_tester.test_benchmarks_registered()


@pytest.mark.usefixtures("register_benchmarks")
def test_inline_eval(eval_tester, inline_benchmark_id, inference_model):
    eval_tester.poll_interval = 1
    eval_tester.poll_timeout = 10
    eval_tester.run_eval(inline_benchmark_id, inference_model, num_examples=3)
