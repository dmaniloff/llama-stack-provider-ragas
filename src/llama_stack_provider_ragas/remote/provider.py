from llama_stack.providers.datatypes import (
    Api,
    ProviderSpec,
    RemoteProviderSpec,
)


def get_provider_spec() -> ProviderSpec:
    return RemoteProviderSpec(
        api=Api.eval,
        provider_type="remote::trustyai_ragas",
        adapter_type="trustyai_ragas",
        module="llama_stack_provider_ragas.remote",
        pip_packages=[
            "ragas==0.3.0",
            "kfp>=2.5.0",
            "kfp-kubernetes>=2.0.0",
            "s3fs>=2024.12.0",
            "kubernetes>=30.0.0",
        ],
        config_class="llama_stack_provider_ragas.config.RagasProviderRemoteConfig",
        api_dependencies=[
            Api.inference,
            Api.files,
            Api.benchmarks,
            Api.datasetio,
            Api.telemetry,
        ],
    )
