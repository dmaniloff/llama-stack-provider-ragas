"""Integration tests for Ragas evaluation using remote Llama Stack wrappers."""

import json
import logging

import pytest
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
# mark as integration, see tool.pytest.ini_options in pyproject.toml
pytestmark = pytest.mark.integration_test


@pytest.fixture
def remote_llm(kubeflow_config, model, sampling_params):
    """Remote LLM wrapper for evaluation."""
    return LlamaStackRemoteLLM(
        base_url=kubeflow_config.llama_stack_url,
        model_id=model,
        sampling_params=sampling_params,
    )


@pytest.fixture
def remote_embeddings(kubeflow_config, embedding_model):
    """Remote embeddings wrapper for evaluation."""
    return LlamaStackRemoteEmbeddings(
        base_url=kubeflow_config.llama_stack_url,
        embedding_model_id=embedding_model,
    )


def test_client_connection(lls_client):
    identifiers = [model.identifier for model in lls_client.models.list()]
    assert "vllm/nomic-ai/nomic-embed-text-v1.5" in identifiers
    assert "vllm/meta-llama/Llama-3.1-8B-Instruct" in identifiers


@pytest.mark.parametrize(
    "metric_to_test",
    [
        pytest.param(m, id=m.name) for m in [answer_relevancy]
    ],  # , context_precision, faithfulness, context_recall]
)
@pytest.mark.parametrize(
    "llm_payload",
    [
        # `answer_relevancy` expects the LLM to output a JSON payload with:
        # - question: a question implied by the given answer
        # - noncommittal: 0/1
        json.dumps({"question": "What is the capital of France?", "noncommittal": 0})
    ],
    indirect=True,
)
def test_single_metric_evaluation(
    evaluation_dataset,
    mocked_lls_client,
    llm_payload,
    remote_llm,
    remote_embeddings,
    metric_to_test,
) -> None:
    # For this test we only evaluate the first sample to keep it fast/deterministic.
    evaluation_dataset = evaluation_dataset[:1]

    result: EvaluationResult = evaluate(
        dataset=evaluation_dataset,
        metrics=[metric_to_test],
        llm=remote_llm,
        embeddings=remote_embeddings,
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
