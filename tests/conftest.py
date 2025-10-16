import os

import pytest
from dotenv import load_dotenv
from llama_stack_client import LlamaStackClient
from ragas import EvaluationDataset

from llama_stack_provider_ragas.config import (
    KubeflowConfig,
    RagasProviderInlineConfig,
    RagasProviderRemoteConfig,
)

load_dotenv()


@pytest.fixture
def lls_client():
    return LlamaStackClient(
        base_url=os.environ.get("KUBEFLOW_LLAMA_STACK_URL", "http://localhost:8321")
    )


@pytest.fixture
def model():
    return "ollama/granite3.3:2b"  # TODO : read from env


@pytest.fixture
def embedding_model():
    return "all-MiniLM-L6-v2"


@pytest.fixture
def sampling_params():
    return {"temperature": 0.1, "max_tokens": 100}


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
