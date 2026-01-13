import os
import random
from datetime import datetime

import pytest
from dotenv import load_dotenv
from llama_stack_client import AsyncLlamaStackClient, LlamaStackClient
from llama_stack_client.types.completion_create_response import (
    Choice,
    CompletionCreateResponse,
)
from llama_stack_client.types.create_embeddings_response import (
    CreateEmbeddingsResponse,
    Data,
    Usage,
)
from ragas import EvaluationDataset

from llama_stack_provider_ragas.compat import SamplingParams, TopPSamplingStrategy
from llama_stack_provider_ragas.config import (
    KubeflowConfig,
    RagasProviderInlineConfig,
    RagasProviderRemoteConfig,
)

load_dotenv()


def pytest_addoption(parser):
    parser.addoption(
        "--no-mock-inference",
        action="store_true",
        help="Don't mock LLM inference (embeddings and completions)",
    )


@pytest.fixture
def unique_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.fixture
def embedding_dimension():
    """Embedding dimension used for testing."""
    return 384


@pytest.fixture
def lls_client(request):
    if request.config.getoption("--no-mock-inference") is True:
        return request.getfixturevalue("real_lls_client")
    else:
        return request.getfixturevalue("mocked_lls_client")


@pytest.fixture
def real_lls_client():
    return LlamaStackClient(base_url=os.environ.get("KUBEFLOW_LLAMA_STACK_URL"))


@pytest.fixture(autouse=True)
def mocked_llm_response(request):
    return getattr(request, "param", "Hello, world!")


@pytest.fixture()
def mocked_lls_client(mocked_lls_clients):
    sync_client, _ = mocked_lls_clients
    return sync_client


@pytest.fixture()
def mocked_lls_clients(monkeypatch, request, embedding_dimension):
    """
    Mock LLM inference (embeddings and completions) by default,
    unless --mock-inference=False is passed in the command line.

    You can indirectly parametrize this fixture to customize completion text:

        @pytest.mark.parametrize(
            "mocked_llm_response",
            ["Hello from mock!"],
            indirect=True,
        )
    """
    # Create real clients, but patch only the `.create()` methods we need.
    base_url = os.environ.get("KUBEFLOW_LLAMA_STACK_URL")
    sync_client = LlamaStackClient(base_url=base_url)
    async_client = AsyncLlamaStackClient(base_url=base_url)

    completion_text = request.getfixturevalue("mocked_llm_response")

    def _make_embeddings_response(n: int) -> CreateEmbeddingsResponse:
        # return one embedding vector per input string
        return CreateEmbeddingsResponse(
            data=[
                Data(
                    embedding=[random.random() for _ in range(embedding_dimension)],
                    index=i,
                    object="embedding",
                )
                for i in range(n)
            ],
            model="mocked/model",
            object="list",
            usage=Usage(prompt_tokens=10, total_tokens=10),
        )

    def _make_completions_response(text: str) -> CompletionCreateResponse:
        return CompletionCreateResponse(
            id="cmpl-123",
            created=1717000000,
            choices=[Choice(index=0, text=text, finish_reason="stop")],
            model="mocked/model",
            object="text_completion",
        )

    def _embeddings_create(*args, **kwargs):
        embedding_input = kwargs.get("input")
        if isinstance(embedding_input, list):
            return _make_embeddings_response(len(embedding_input))
        return _make_embeddings_response(1)

    async def _async_embeddings_create(*args, **kwargs):
        embedding_input = kwargs.get("input")
        if isinstance(embedding_input, list):
            return _make_embeddings_response(len(embedding_input))
        return _make_embeddings_response(1)

    def _completions_create(*args, **kwargs):
        return _make_completions_response(completion_text)

    async def _async_completions_create(*args, **kwargs):
        return _make_completions_response(completion_text)

    # Patch nested methods (avoids dotted-attribute monkeypatch issues on classes).
    monkeypatch.setattr(sync_client.embeddings, "create", _embeddings_create)
    monkeypatch.setattr(sync_client.completions, "create", _completions_create)
    monkeypatch.setattr(async_client.embeddings, "create", _async_embeddings_create)
    monkeypatch.setattr(async_client.completions, "create", _async_completions_create)

    return sync_client, async_client


@pytest.fixture(autouse=True)
def patch_remote_wrappers(monkeypatch, mocked_lls_clients, request):
    sync_client, async_client = mocked_lls_clients
    if request.config.getoption("--no-mock-inference") is not True:
        from llama_stack_provider_ragas.remote import wrappers_remote

        monkeypatch.setattr(
            wrappers_remote, "LlamaStackClient", lambda *a, **k: sync_client
        )
        monkeypatch.setattr(
            wrappers_remote, "AsyncLlamaStackClient", lambda *a, **k: async_client
        )


@pytest.fixture
def model():
    return "ollama/granite3.3:2b"  # TODO : read from env


@pytest.fixture
def embedding_model():
    return "ollama/all-minilm:latest"


@pytest.fixture
def sampling_params():
    return SamplingParams(
        strategy=TopPSamplingStrategy(temperature=0.1, top_p=0.95),
        max_tokens=100,
        stop=None,
    )


@pytest.fixture
def inline_eval_config(embedding_model):
    return RagasProviderInlineConfig(embedding_model=embedding_model)


@pytest.fixture
def kubeflow_config():
    return KubeflowConfig(
        pipelines_endpoint=os.environ["KUBEFLOW_PIPELINES_ENDPOINT"],
        namespace=os.environ["KUBEFLOW_NAMESPACE"],
        llama_stack_url=os.environ["KUBEFLOW_LLAMA_STACK_URL"],
        base_image=os.environ["KUBEFLOW_BASE_IMAGE"],
        results_s3_prefix=os.environ["KUBEFLOW_RESULTS_S3_PREFIX"],
        s3_credentials_secret_name=os.environ["KUBEFLOW_S3_CREDENTIALS_SECRET_NAME"],
    )


@pytest.fixture
def remote_eval_config(embedding_model, kubeflow_config):
    return RagasProviderRemoteConfig(
        embedding_model=embedding_model,
        kubeflow_config=kubeflow_config,
    )


@pytest.fixture
def raw_evaluation_data():
    """Sample data for Ragas evaluation."""
    return [
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


@pytest.fixture
def evaluation_dataset(raw_evaluation_data):
    """Create EvaluationDataset from sample data."""
    return EvaluationDataset.from_list(raw_evaluation_data)
