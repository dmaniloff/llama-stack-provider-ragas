from ..compat import Api, InlineProviderSpec, ProviderSpec
from ..constants import PROVIDER_TYPE


def get_provider_spec() -> ProviderSpec:
    return InlineProviderSpec(
        api=Api.eval,
        provider_type=f"inline::{PROVIDER_TYPE}",
        pip_packages=["ragas>=0.4.0,<0.5.0"],
        config_class="llama_stack_provider_ragas.config.RagasProviderInlineConfig",
        module="llama_stack_provider_ragas.inline",
        api_dependencies=[
            Api.inference,
            Api.files,
            Api.benchmarks,
            Api.datasetio,
        ],
    )
