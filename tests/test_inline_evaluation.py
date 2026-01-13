"""Test inline evaluation."""

import json
import random
from types import SimpleNamespace

import pytest
import yaml
from llama_stack.core.library_client import LlamaStackAsLibraryClient
from ragas.metrics import answer_relevancy
from rich import print as rprint
from rich.pretty import Pretty

from llama_stack_provider_ragas.compat import Api
from llama_stack_provider_ragas.constants import PROVIDER_ID_INLINE

pytestmark = pytest.mark.lls_integration


@pytest.fixture
def library_stack_config(tmp_path, embedding_dimension):
    """Stack configuration for library client testing."""
    storage_dir = tmp_path / "test_storage"
    storage_dir.mkdir()

    return {
        "version": 2,
        "image_name": "test_ragas_inline",
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
                        "embedding_model": "ollama/all-minilm:latest",
                        "kvstore": {"namespace": "ragas", "backend": "kv_default"},
                    },
                }
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
                    "model_id": "all-MiniLM-L6-v2",
                    "provider_id": "ollama",
                    "provider_model_id": "all-minilm:latest",
                    "model_type": "embedding",
                },
                {
                    "metadata": {},
                    "model_id": "granite3.3:2b",
                    "provider_id": "ollama",
                    "provider_model_id": "granite3.3:2b",
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


@pytest.fixture
def library_stack_config_file(library_stack_config, tmp_path):
    """Write the stack config dict to a temp YAML file and return its path."""

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml.safe_dump(library_stack_config), encoding="utf-8")
    return config_file


@pytest.fixture
def library_client(request):
    """
    Mock LLM inference (embeddings and completions) by default,
    unless --mock-inference=False is passed in the command line.

    You can indirectly parametrize this fixture to customize the completion text:

        @pytest.mark.parametrize(
            "mocked_llm_response",
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
    mocked_llm_response,
    library_stack_config_file,
    embedding_dimension,
):
    completion_text = mocked_llm_response

    # Mock Ollama connectivity check & model listing
    async def _fake_check_model_availability(*args, **kwargs):
        return True

    async def _fake_list_provider_model_ids(*args, **kwargs):
        return ["all-minilm:latest", "granite3.3:2b"]

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


def test_library_client_health(library_client):
    assert library_client is not None
    assert hasattr(library_client, "alpha")
    assert hasattr(library_client.alpha, "eval")

    models = library_client.models.list()
    assert len(models) > 0
    print("Available models:")
    rprint(Pretty(models, max_depth=6, expand_all=True))

    providers = library_client.providers.list()
    assert len(providers) > 0
    assert any(p.api == "eval" for p in providers)
    print("Available providers:")
    rprint(Pretty(providers, max_depth=6, expand_all=True))


@pytest.mark.parametrize(
    "metric_to_test,mocked_llm_response",
    [
        # `answer_relevancy` expects the LLM to output a JSON payload with:
        # - question: a question implied by the given answer
        # - noncommittal: 0/1
        pytest.param(
            answer_relevancy,
            json.dumps(
                {"question": "What is the capital of France?", "noncommittal": 0}
            ),
            id="answer_relevancy",
        ),
    ],
    indirect=["mocked_llm_response"],
)
def test_full_evaluation_with_library_client(
    library_client,
    model,
    sampling_params,
    unique_timestamp,
    raw_evaluation_data,
    metric_to_test,
):
    dataset_id = f"library_full_test_dataset_{unique_timestamp}"
    library_client.beta.datasets.register(
        dataset_id=dataset_id,
        purpose="eval/question-answer",
        source={"type": "rows", "rows": raw_evaluation_data[:1]},
        metadata={"provider_id": "localfs"},
    )
    datasets = library_client.beta.datasets.list()
    print(f"Available datasets: {[d.identifier for d in datasets]}")
    assert any(d.identifier == dataset_id for d in datasets)

    benchmark_id = f"library_full_test_benchmark_{unique_timestamp}"
    library_client.alpha.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset_id,
        scoring_functions=[metric_to_test.name],
        provider_id=PROVIDER_ID_INLINE,
    )
    benchmarks = library_client.alpha.benchmarks.list()
    print(f"Available benchmarks: {[b.identifier for b in benchmarks]}")
    assert any(b.identifier == benchmark_id for b in benchmarks)

    job = library_client.alpha.eval.run_eval(
        benchmark_id=benchmark_id,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": model,
                "sampling_params": sampling_params.model_dump(exclude_none=True),
            },
            "scoring_params": {},
        },
    )

    assert job.job_id is not None
    assert job.status == "in_progress"

    job = library_client.alpha.eval.jobs.status(
        benchmark_id=benchmark_id, job_id=job.job_id
    )
    assert job.status == "completed"
