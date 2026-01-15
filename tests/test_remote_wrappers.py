"""Test the remote wrappers for the Llama Stack client."""

import json
import logging
import os

import pytest
from langchain_core.prompt_values import StringPromptValue
from ragas import evaluate
from ragas.evaluation import EvaluationResult
from ragas.metrics import answer_relevancy
from ragas.run_config import RunConfig

from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table
from llama_stack_provider_ragas.remote.wrappers_remote import (
    LlamaStackRemoteEmbeddings,
    LlamaStackRemoteLLM,
)

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.lls_integration


@pytest.fixture
def lls_remote_embeddings(embedding_model):
    return LlamaStackRemoteEmbeddings(
        base_url=os.environ.get("KUBEFLOW_LLAMA_STACK_URL"),
        embedding_model_id=embedding_model,
    )


@pytest.fixture
def lls_remote_llm(model, sampling_params):
    """Remote LLM wrapper for evaluation."""
    return LlamaStackRemoteLLM(
        base_url=os.environ.get("KUBEFLOW_LLAMA_STACK_URL"),
        model_id=model,
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
