import logging

from langchain_core.language_models.llms import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from llama_stack_client import AsyncLlamaStackClient, LlamaStackClient
from llama_stack_client.types import CompletionResponse
from llama_stack_client.types.create_embeddings_response import CreateEmbeddingsResponse
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig

logger = logging.getLogger(__name__)


class LlamaStackRemoteEmbeddings(BaseRagasEmbeddings):
    """Wrapper that makes Llama Stack client embeddings compatible with Ragas."""

    def __init__(
        self,
        base_url: str,
        embedding_model_id: str,
        run_config: RunConfig | None = None,
    ):
        super().__init__()
        self.sync_client = LlamaStackClient(base_url=base_url)
        self.async_client = AsyncLlamaStackClient(base_url=base_url)
        self.embedding_model_id = embedding_model_id
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def _validate_embedding(self, embedding: list[float] | str) -> list[float]:
        """Validate that embedding is in float format, not base64."""
        if isinstance(embedding, str):
            raise ValueError("Expected float embeddings, got base64 string")
        return embedding

    def embed_query(self, text: str) -> list[float]:
        """Synchronous embed query using Llama Stack client."""
        try:
            response: CreateEmbeddingsResponse = self.sync_client.embeddings.create(
                input=text,
                model=self.embedding_model_id,
            )
            return self._validate_embedding(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Query embedding failed: {str(e)}")
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous embed documents using Llama Stack client."""
        try:
            response: CreateEmbeddingsResponse = self.sync_client.embeddings.create(
                input=texts,
                model=self.embedding_model_id,
            )
            return [self._validate_embedding(data.embedding) for data in response.data]
        except Exception as e:
            logger.error(f"Document embedding failed: {str(e)}")
            raise

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async embed documents using Llama Stack client."""
        try:
            response: CreateEmbeddingsResponse = (
                await self.async_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model_id,
                )
            )
            return [self._validate_embedding(data.embedding) for data in response.data]
        except Exception as e:
            logger.error(f"Async document embedding failed: {str(e)}")
            raise

    async def aembed_query(self, text: str) -> list[float]:
        """Async embed query using Llama Stack client."""
        try:
            response: CreateEmbeddingsResponse = (
                await self.async_client.embeddings.create(
                    input=text,
                    model=self.embedding_model_id,
                )
            )
            return self._validate_embedding(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Async query embedding failed: {str(e)}")
            raise


class LlamaStackRemoteLLM(BaseRagasLLM):
    """Wrapper that makes Llama Stack client LLM compatible with Ragas."""

    def __init__(
        self,
        base_url: str,
        model_id: str,
        sampling_params: dict | None = None,
        run_config: RunConfig | None = None,
        multiple_completion_supported: bool = True,
    ):
        if run_config is None:
            run_config = RunConfig()
        super().__init__(run_config, multiple_completion_supported)

        self.sync_client = LlamaStackClient(base_url=base_url)
        self.async_client = AsyncLlamaStackClient(base_url=base_url)
        self.model_id = model_id
        self.sampling_params = sampling_params or {}
        self.enable_prompt_logging = True
        self.prompt_counter = 0

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text."""
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4

    def _log_prompt(self, prompt_text: str, prompt_type: str = "evaluation") -> None:
        """Log prompt details if enabled."""
        if not self.enable_prompt_logging:
            return

        self.prompt_counter += 1
        estimated_tokens = self._estimate_tokens(prompt_text)

        logger.info(f"=== RAGAS PROMPT #{self.prompt_counter} ({prompt_type}) ===")
        logger.info(f"Estimated tokens: {estimated_tokens}")
        logger.info(f"Character count: {len(prompt_text)}")
        logger.info(f"Prompt preview: {prompt_text[:200]}...")
        logger.info(f"Full prompt:\n{prompt_text}")
        logger.info("=" * 50)

    def _prepare_generation_params(
        self, prompt: PromptValue, temperature: float | None = None
    ) -> tuple[str, dict]:
        """Prepare prompt text and sampling parameters for generation."""
        prompt_text = prompt.to_string()
        self._log_prompt(prompt_text)

        sampling_params = self.sampling_params.copy()
        if temperature is not None:
            sampling_params["temperature"] = temperature

        return prompt_text, sampling_params

    def _initialize_llm_output(self) -> dict:
        """Create initial LLM output structure."""
        return {
            "llama_stack_responses": [],
            "model_id": self.model_id,
            "provider": "llama_stack_remote",
        }

    def _update_llm_output(
        self, response: CompletionResponse, llm_output: dict
    ) -> None:
        """Process completion response and update llm_output."""
        llama_stack_info = {
            "stop_reason": response.stop_reason,
            "content_length": len(response.content),
        }
        llm_output["llama_stack_responses"].append(llama_stack_info)

    def generate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        """Synchronous text generation using Llama Stack client."""
        try:
            prompt_text, sampling_params = self._prepare_generation_params(
                prompt, temperature
            )
            generations = []
            llm_output = self._initialize_llm_output()

            for _ in range(n):
                response: CompletionResponse = self.sync_client.inference.completion(
                    content=prompt_text,
                    model_id=self.model_id,
                    sampling_params=sampling_params if sampling_params else None,
                )

                self._update_llm_output(response, llm_output)
                generations.append(Generation(text=response.content))

            return LLMResult(generations=[generations], llm_output=llm_output)

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    async def agenerate_text(
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = None,
        stop: list[str] | None = None,
        callbacks=None,
    ) -> LLMResult:
        """Asynchronous text generation using Llama Stack client."""
        try:
            prompt_text, sampling_params = self._prepare_generation_params(
                prompt, temperature
            )
            generations = []
            llm_output = self._initialize_llm_output()

            for _ in range(n):
                response: CompletionResponse = (
                    await self.async_client.inference.completion(
                        content=prompt_text,
                        model_id=self.model_id,
                        sampling_params=sampling_params if sampling_params else None,
                    )
                )

                self._update_llm_output(response, llm_output)
                generations.append(Generation(text=response.content))

            return LLMResult(generations=[generations], llm_output=llm_output)

        except Exception as e:
            logger.error(f"Async LLM generation failed: {str(e)}")
            raise

    def get_temperature(self, n: int) -> float:
        """Get temperature based on number of completions."""
        return 0.3 if n > 1 else 1e-8
