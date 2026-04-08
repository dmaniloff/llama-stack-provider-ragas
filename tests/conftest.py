import os

import pytest
from ragas import EvaluationDataset


def pytest_addoption(parser):
    parser.addoption(
        "--no-mock-inference",
        action="store_true",
        help="Don't mock LLM inference (embeddings and completions)",
    )
    parser.addoption(
        "--no-mock-client",
        action="store_true",
        help="Don't mock the LlamaStackClient; use a real server for wrapper tests",
    )


@pytest.fixture(scope="session")
def llama_stack_base_url():
    return os.getenv("LLAMA_STACK_BASE_URL", "http://localhost:8321")


@pytest.fixture
def embedding_dimension():
    """Embedding dimension used for testing."""
    return 384


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def dataset_id():
    return "ragas_test_dataset"


@pytest.fixture(scope="session")
def inline_benchmark_id():
    return "hf-doc-qa-ragas-inline-benchmark"


@pytest.fixture(scope="session")
def remote_benchmark_id():
    return "hf-doc-qa-ragas-remote-benchmark"


@pytest.fixture
def register_dataset(client, raw_evaluation_data, dataset_id):
    """Register the evaluation dataset with inline rows."""
    client.beta.datasets.register(
        dataset_id=dataset_id,
        purpose="eval/messages-answer",
        source={"type": "rows", "rows": raw_evaluation_data},
    )
    yield
    try:
        client.beta.datasets.unregister(dataset_id=dataset_id)
    except Exception:
        pass


@pytest.fixture
def register_benchmarks(
    client, register_dataset, dataset_id, inline_benchmark_id, remote_benchmark_id
):
    """Register evaluation benchmarks for inline and remote providers."""
    client.alpha.benchmarks.register(
        benchmark_id=inline_benchmark_id,
        dataset_id=dataset_id,
        scoring_functions=["answer_similarity"],
        provider_id="trustyai_ragas_inline",
    )
    client.alpha.benchmarks.register(
        benchmark_id=remote_benchmark_id,
        dataset_id=dataset_id,
        scoring_functions=["answer_similarity"],
        provider_id="trustyai_ragas_remote",
    )
    yield
    for bid in (inline_benchmark_id, remote_benchmark_id):
        try:
            client.alpha.benchmarks.unregister(benchmark_id=bid)
        except Exception:
            pass
