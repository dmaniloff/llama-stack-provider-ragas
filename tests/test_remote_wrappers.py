"""Tests for the LangChain-compatible remote wrappers (LLM and Embeddings).

These tests exercise ``LlamaStackRemoteLLM`` and
``LlamaStackRemoteEmbeddings``, which wrap the OpenAI-compatible
completions and embeddings endpoints exposed by a Llama Stack server.

By default, the client is **mocked**: the ``LlamaStackClient`` and
``AsyncLlamaStackClient`` constructors are monkey-patched so that
completions and embeddings calls return deterministic fake responses.
No running server is required in this mode::

    pytest tests/test_remote_wrappers.py

To run against a real Llama Stack server, pass ``--no-mock-client``.
The server URL is read from ``LLAMA_STACK_BASE_URL`` (default
``http://localhost:8321``).  Model IDs can be overridden with
``INFERENCE_MODEL`` and ``EMBEDDING_MODEL``::

    pytest tests/test_remote_wrappers.py --no-mock-client

    INFERENCE_MODEL=ollama/granite3.3:2b \\
    EMBEDDING_MODEL=ollama/all-minilm:latest \\
        pytest tests/test_remote_wrappers.py --no-mock-client
"""

import json
import logging
import os
import random

import pytest
from langchain_core.prompt_values import StringPromptValue
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
from ragas import evaluate
from ragas.evaluation import EvaluationResult
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

from llama_stack_provider_ragas.compat import SamplingParams, TopPSamplingStrategy
from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table
from llama_stack_provider_ragas.remote.wrappers_remote import (
    LlamaStackRemoteEmbeddings,
    LlamaStackRemoteLLM,
)

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.unit


@pytest.fixture
def inference_model():
    return os.getenv("INFERENCE_MODEL", "litellm/Mistral-Small-24B-W8A8")


@pytest.fixture
def embedding_model():
    return os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")


@pytest.fixture
def sampling_params():
    return SamplingParams(
        strategy=TopPSamplingStrategy(temperature=0.1, top_p=0.95),
        max_tokens=100,
        stop=None,
    )


@pytest.fixture
def lls_client(request):
    if request.config.getoption("--no-mock-client") is True:
        return request.getfixturevalue("real_lls_client")
    else:
        return request.getfixturevalue("mocked_lls_client")


@pytest.fixture
def real_lls_client(llama_stack_base_url):
    return LlamaStackClient(base_url=llama_stack_base_url)


@pytest.fixture(autouse=True)
def mocked_client_response(request):
    """Fake completion text returned by the mocked ``LlamaStackClient``.

    The client's ``completions.create`` and ``embeddings.create`` methods
    are monkey-patched in ``mocked_lls_clients``; this fixture controls the
    text that the mocked completions endpoint returns.  Use indirect
    parametrization to override the default value per test.
    """
    return getattr(request, "param", "Hello, world!")


@pytest.fixture()
def mocked_lls_client(mocked_lls_clients):
    sync_client, _ = mocked_lls_clients
    return sync_client


@pytest.fixture()
def mocked_lls_clients(monkeypatch, request, embedding_dimension, llama_stack_base_url):
    """Build mocked sync and async ``LlamaStackClient`` instances.

    Completions and embeddings ``.create()`` methods are replaced with
    fakes that return deterministic responses.  The completion text comes
    from the ``mocked_client_response`` fixture, which can be overridden
    via indirect parametrization::

        @pytest.mark.parametrize(
            "mocked_client_response",
            ["Hello from mock!"],
            indirect=True,
        )
    """
    # Create real clients, but patch only the `.create()` methods we need.
    sync_client = LlamaStackClient(base_url=llama_stack_base_url)
    async_client = AsyncLlamaStackClient(base_url=llama_stack_base_url)

    completion_text = request.getfixturevalue("mocked_client_response")

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
    if request.config.getoption("--no-mock-client") is not True:
        from llama_stack_provider_ragas.remote import wrappers_remote

        monkeypatch.setattr(
            wrappers_remote, "LlamaStackClient", lambda *a, **k: sync_client
        )
        monkeypatch.setattr(
            wrappers_remote, "AsyncLlamaStackClient", lambda *a, **k: async_client
        )


@pytest.fixture
def lls_remote_embeddings(embedding_model, llama_stack_base_url):
    return LlamaStackRemoteEmbeddings(
        base_url=llama_stack_base_url,
        embedding_model_id=embedding_model,
    )


@pytest.fixture
def lls_remote_llm(inference_model, sampling_params, llama_stack_base_url):
    """Remote LLM wrapper for evaluation."""
    return LlamaStackRemoteLLM(
        base_url=llama_stack_base_url,
        model_id=inference_model,
        sampling_params=sampling_params,
    )


def test_remote_embeddings_sync(lls_remote_embeddings):
    embeddings = lls_remote_embeddings.embed_query("Hello, world!")
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], float)

    embeddings = lls_remote_embeddings.embed_documents(["Hello, world!"])
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)


@pytest.mark.asyncio
async def test_remote_embeddings_async(lls_remote_embeddings):
    embeddings = await lls_remote_embeddings.aembed_query("Hello, world!")
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], float)

    embeddings = await lls_remote_embeddings.aembed_documents(
        ["Hello, world!", "How are you?"]
    )
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0], list)
    assert isinstance(embeddings[0][0], float)
    assert len(embeddings) == 2  # One embedding per input text


def test_remote_llm_sync(lls_remote_llm):
    prompt = StringPromptValue(text="What is the capital of France?")
    result = lls_remote_llm.generate_text(prompt, n=1)

    assert hasattr(result, "generations")
    assert len(result.generations) == 1
    assert len(result.generations[0]) == 1
    assert isinstance(result.generations[0][0].text, str)
    assert len(result.generations[0][0].text) > 0

    assert hasattr(result, "llm_output")
    assert result.llm_output["provider"] == "llama_stack_remote"
    assert len(result.llm_output["llama_stack_responses"]) == 1


@pytest.mark.asyncio
async def test_remote_llm_async(lls_remote_llm):
    prompt = StringPromptValue(text="What is the capital of France?")
    result = await lls_remote_llm.agenerate_text(prompt, n=1)

    assert hasattr(result, "generations")
    assert len(result.generations) == 1
    assert len(result.generations[0]) == 1
    assert isinstance(result.generations[0][0].text, str)
    assert len(result.generations[0][0].text) > 0

    assert hasattr(result, "llm_output")
    assert result.llm_output["provider"] == "llama_stack_remote"
    assert len(result.llm_output["llama_stack_responses"]) == 1


@pytest.mark.parametrize(
    "metric_to_test,mocked_client_response",
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
    indirect=["mocked_client_response"],
)
def test_direct_evaluation(
    evaluation_dataset,
    lls_remote_llm,
    lls_remote_embeddings,
    metric_to_test,
) -> None:
    # For this test we only evaluate the first sample to keep it fast/deterministic.
    evaluation_dataset = evaluation_dataset[:1]

    result: EvaluationResult = evaluate(
        dataset=evaluation_dataset,
        metrics=[metric_to_test],
        llm=lls_remote_llm,
        embeddings=lls_remote_embeddings,
        run_config=RunConfig(max_workers=1),
        show_progress=True,
    )

    assert isinstance(result, EvaluationResult)
    pandas_result = result.to_pandas()
    logger.info(render_dataframe_as_table(pandas_result))
    assert metric_to_test.name in pandas_result.columns
    assert len(pandas_result) == len(evaluation_dataset)
    assert pandas_result[metric_to_test.name].dtype == float

    # Use small tolerance for floating point comparisons
    tolerance = 1e-10
    assert pandas_result[metric_to_test.name].min() >= -tolerance
    assert pandas_result[metric_to_test.name].max() <= 1 + tolerance
