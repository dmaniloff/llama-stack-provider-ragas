# remote is the default provider
from .remote import get_adapter_impl
from .remote.provider import get_provider_spec

__all__ = ["get_provider_spec", "get_adapter_impl"]
