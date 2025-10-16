"""Integration tests for Ragas evaluation using Llama Stack eval API (inline)."""

from datetime import datetime

import pytest
from ragas.metrics import answer_relevancy

# mark as integration, see tool.pytest.ini_options in pyproject.toml
pytestmark = pytest.mark.integration_test


@pytest.fixture
def unique_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@pytest.mark.parametrize(
    "metric_to_test",
    [
        pytest.param(m, id=m.name) for m in [answer_relevancy]
    ],  # , context_precision, faithfulness, context_recall]
)
def test_single_metric_evaluation(
    model,
    sampling_params,
    lls_client,
    unique_timestamp,
    raw_evaluation_data,
    metric_to_test,
):
    dataset_id = f"test_ragas_dataset_{unique_timestamp}"
    lls_client.datasets.register(
        dataset_id=dataset_id,
        purpose="eval/question-answer",  # TODO: this works, but check if there is a required data format for this purpose
        source={"type": "rows", "rows": raw_evaluation_data},
        metadata={"provider_id": "localfs"},
    )

    benchmark_id = f"test_ragas_benchmark_{unique_timestamp}"
    lls_client.benchmarks.register(
        benchmark_id=benchmark_id,
        dataset_id=dataset_id,
        scoring_functions=[metric_to_test.name],
        provider_id="trustyai_ragas",
    )

    job = lls_client.eval.run_eval(
        benchmark_id=benchmark_id,
        benchmark_config={
            "eval_candidate": {
                "type": "model",
                "model": model,
                "sampling_params": sampling_params,
            },
            "scoring_params": {},
        },
    )

    assert hasattr(job, "job_id")
    assert hasattr(job, "status")
    assert job.job_id is not None
