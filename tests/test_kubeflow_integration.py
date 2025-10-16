"""Integration tests for Kubeflow pipeline components against live cluster."""

import os
from textwrap import dedent
from typing import List  # noqa

import kfp
import pytest
from kfp import dsl
from ragas.metrics import answer_relevancy

from llama_stack_provider_ragas.remote.kubeflow.pipeline import (
    ragas_evaluation_pipeline,
)

# Mark all tests as integration tests
pytestmark = pytest.mark.integration_test


@pytest.fixture
def kf_client():
    token = os.popen("oc whoami -t").read().strip()
    return kfp.Client(
        host=os.environ["KUBEFLOW_PIPELINES_ENDPOINT"], existing_token=token
    )


@dsl.component(base_image=os.environ["KUBEFLOW_BASE_IMAGE"])
def retrieve_data_for_testing(output_dataset: dsl.Output[dsl.Dataset]):
    import pandas as pd

    dataset = pd.DataFrame(
        [
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
        ]
    )
    dataset.to_json(output_dataset.path, orient="records", lines=True)


@dsl.component(base_image=os.environ["KUBEFLOW_BASE_IMAGE"])
def run_fake_ragas_evaluation(
    model: str,
    sampling_params: dict,
    embedding_model: str,
    metrics: List[str],  # noqa
    llama_stack_base_url: str,
    input_dataset: dsl.Input[dsl.Dataset],
    output_results: dsl.Output[dsl.Dataset],
):
    import logging
    from unittest.mock import AsyncMock, patch

    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import EvaluationResult
    from ragas.run_config import RunConfig

    from llama_stack_provider_ragas.constants import METRIC_MAPPING
    from llama_stack_provider_ragas.logging_utils import render_dataframe_as_table
    from llama_stack_provider_ragas.remote.wrappers_remote import (
        LlamaStackRemoteEmbeddings,
        LlamaStackRemoteLLM,
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Mock the agenerate_text method to return realistic structured responses
    from langchain_core.language_models.llms import Generation, LLMResult

    def mock_answer_relevancy_side_effect(
        prompt, n=1, temperature=None, stop=None, callbacks=None, **kwargs
    ):
        # Return answer relevancy specific JSON format for all prompts
        # Expected format: {"question": "...", "noncommittal": 0|1}
        json_text = dedent("""```json
{
    "question": "When was the telephone invented?",
    "noncommittal": 0
}
```""")

        generations = [Generation(text=json_text) for _ in range(n)]
        return LLMResult(generations=[generations])

    mock_agenerate = AsyncMock()
    mock_agenerate.side_effect = mock_answer_relevancy_side_effect

    mock_embed_documents = AsyncMock()
    mock_embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4, 0.5] for _ in range(10)]

    with (
        patch.object(LlamaStackRemoteLLM, "agenerate_text", mock_agenerate),
        patch.object(
            LlamaStackRemoteEmbeddings, "embed_documents", mock_embed_documents
        ),
    ):
        llm = LlamaStackRemoteLLM(
            base_url=llama_stack_base_url,
            model_id=model,
            sampling_params=sampling_params,
        )
        embeddings = LlamaStackRemoteEmbeddings(
            base_url=llama_stack_base_url,
            embedding_model_id=embedding_model,
        )

        metrics = [METRIC_MAPPING[m] for m in metrics]

        with open(input_dataset.path) as f:
            df_input = pd.read_json(f, lines=True)
            eval_dataset = EvaluationDataset.from_list(
                df_input.to_dict(orient="records")
            )

        ragas_output: EvaluationResult = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            run_config=RunConfig(max_workers=1),
        )

    df_output = ragas_output.to_pandas()
    table_output = render_dataframe_as_table(df_output, "Ragas Evaluation Results")
    logger.info(f"Ragas evaluation completed:\n{table_output}")
    df_output.to_json(output_results.path, orient="records", lines=True)


def test_pipeline_dummy_dataset_retrieval(kf_client, remote_eval_config):
    @dsl.pipeline()
    def pipline_dataset_retrieval():
        retrieve_data_for_testing()

    run_result = kf_client.create_run_from_pipeline_func(
        pipeline_func=pipline_dataset_retrieval,
        namespace=remote_eval_config.kubeflow_config.namespace,
        run_name="test-pipeline-dummy-dataset-retrieval",
        experiment_name="ragas-provider-kf-tests",
    )

    assert run_result.run_id is not None


@pytest.mark.parametrize(
    "metric_to_test",
    [
        pytest.param(m, id=m.name) for m in [answer_relevancy]
    ],  # , context_precision, faithfulness, context_recall]
)
def test_pipeline_dummy_ragas_evaluation(
    kf_client, remote_eval_config, model, sampling_params, metric_to_test
):
    @dsl.pipeline()
    def pipeline_ragas_evaluation():
        test_dataset = retrieve_data_for_testing()
        run_fake_ragas_evaluation(
            input_dataset=test_dataset.output,
            model=model,
            sampling_params=sampling_params,
            embedding_model=remote_eval_config.embedding_model,
            metrics=[metric_to_test.name],
            llama_stack_base_url=remote_eval_config.kubeflow_config.llama_stack_url,
        )

    run_result = kf_client.create_run_from_pipeline_func(
        pipeline_func=pipeline_ragas_evaluation,
        namespace=remote_eval_config.kubeflow_config.namespace,
        run_name="test-pipeline-dummy-ragas-evaluation",
        experiment_name="ragas-provider-kf-tests",
    )

    assert run_result.run_id is not None


@pytest.mark.parametrize(
    "metric_to_test",
    [
        pytest.param(m, id=m.name) for m in [answer_relevancy]
    ],  # , context_precision, faithfulness, context_recall]
)
def test_full_pipeline(
    kf_client, remote_eval_config, metric_to_test, model, sampling_params
):
    embedding_model = remote_eval_config.embedding_model

    run_result = kf_client.create_run_from_pipeline_func(
        pipeline_func=ragas_evaluation_pipeline,
        namespace=remote_eval_config.kubeflow_config.namespace,
        arguments={
            "model": model,
            "dataset_id": "ragas_demo_dataset_remote",  # TODO: this will fail if the dataset does not exist
            "sampling_params": sampling_params,
            "embedding_model": embedding_model,
            "metrics": [metric_to_test.name],
            "llama_stack_base_url": remote_eval_config.kubeflow_config.llama_stack_url,
            "s3_credentials_secret_name": remote_eval_config.kubeflow_config.s3_credentials_secret_name,
            "result_s3_location": remote_eval_config.kubeflow_config.results_s3_prefix,
        },
        run_name="test-full-pipeline",
        experiment_name="ragas-provider-kf-tests",
    )

    assert run_result.run_id is not None
