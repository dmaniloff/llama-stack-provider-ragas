from typing import Any

from llama_stack.apis.datatypes import Api

from ..config import RagasProviderRemoteConfig
from .ragas_remote_eval import RagasEvaluatorRemote


async def get_adapter_impl(
    config: RagasProviderRemoteConfig,
    deps: dict[Api, Any],
) -> RagasEvaluatorRemote:
    return RagasEvaluatorRemote(config)


__all__ = ["RagasEvaluatorRemote", "get_adapter_impl"]
