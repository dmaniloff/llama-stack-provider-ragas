from typing import Any

from llama_stack.apis.datatypes import Api

from ..config import RagasProviderInlineConfig
from .ragas_inline_eval import RagasEvaluatorInline


async def get_provider_impl(
    config: RagasProviderInlineConfig,
    deps: dict[Api, Any],
) -> RagasEvaluatorInline:
    return RagasEvaluatorInline(config, deps[Api.datasetio], deps[Api.inference])


__all__ = ["RagasEvaluatorInline", "get_provider_impl"]
