from llama_stack.schema_utils import json_schema_type
from pydantic import BaseModel, Field


class RagasConfig(BaseModel):
    """Additional configuration parameters for Ragas evaluation."""

    batch_size: int | None = Field(
        default=None,
        description="Batch size for evaluation. If None, no batching is done.",
    )

    show_progress: bool = Field(
        default=True, description="Whether to show progress bar during evaluation"
    )

    raise_exceptions: bool = Field(
        default=True,
        description="Whether to raise exceptions or return NaN for failed evaluations",
    )

    experiment_name: str | None = Field(
        default=None, description="Name for experiment tracking"
    )

    column_map: dict[str, str] | None = Field(
        default=None, description="Mapping of dataset column names to expected names"
    )


class RagasProviderBaseConfig(BaseModel):
    """Base configuration shared by inline and remote providers."""

    # Looking for the model?
    # It's in the benchmark config's eval_candidate.
    # You set it as part of the call to `client.eval.run_eval`.

    # Looking for the sampling params?
    # It's in the benchmark config's eval_candidate.
    # You set them as part of the call to `client.eval.run_eval`.

    # Looking for the dataset?
    # It's in the benchmark config's dataset_id.
    # You set it as part of the call to `client.benchmarks.register` and
    # `client.datasets.register`.

    # Looking for the metrics?
    # They're in the benchmark config's scoring_functions.
    # You set them as part of the call to `client.benchmarks.register`.

    embedding_model: str = Field(
        description=(
            "Embedding model for Ragas evaluation. "
            "At the moment, this cannot be set in Llama Stack's benchmark config. "
            "It must match the identifier of the embedding model in Llama Stack."
        ),
    )

    ragas_config: RagasConfig = Field(
        default=RagasConfig(),
        description="Additional configuration parameters for Ragas.",
    )


@json_schema_type
class RagasProviderInlineConfig(RagasProviderBaseConfig):
    """Configuration for Ragas evaluation provider (inline execution)."""


@json_schema_type
class RagasProviderRemoteConfig(RagasProviderBaseConfig):
    """Configuration for Ragas evaluation provider (remote execution)."""

    kubeflow_config: "KubeflowConfig" = Field(
        description="Additional configuration parameters for remote execution.",
    )


class KubeflowConfig(BaseModel):
    """Configuration for Kubeflow remote execution."""

    results_s3_prefix: str = Field(
        description="S3 prefix (folder) where the evaluation results will be written.",
    )

    s3_credentials_secret_name: str = Field(
        description=(
            "Name of the AWS credentials secret. "
            "Must have write access to the results S3 prefix. "
            "These credentials will be loaded as environment variables in the Kubeflow pipeline components."
        ),
    )

    pipelines_endpoint: str = Field(
        description="Kubeflow Pipelines API endpoint URL.",
    )

    namespace: str = Field(
        description="Kubeflow namespace for pipeline execution.",
    )

    llama_stack_url: str = Field(
        description="Base URL for Llama Stack API (accessible from Kubeflow pods).",
    )

    base_image: str | None = Field(
        description=(
            "Base image for Kubeflow pipeline components. "
            "If not provided via env var, the image name will be read from a ConfigMap specified in constants.py."
            "NOTE: this field is accessed via env var, but is here for completeness."
        ),
        default=None,
    )

    pipelines_token: str | None = Field(
        description=(
            "Kubeflow Pipelines token with access to submit pipelines. "
            "If not provided via env var, the token will be read from the local kubeconfig file."
        ),
        default=None,
    )
