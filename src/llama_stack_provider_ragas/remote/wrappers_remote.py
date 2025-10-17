import logging

from langchain_core.language_models.llms import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from llama_stack.apis.inference import SamplingParams, TopPSamplingStrategy
from llama_stack_client import AsyncLlamaStackClient, LlamaStackClient, omit
from llama_stack_client.types.completion_create_response import CompletionCreateResponse
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
        sampling_params: SamplingParams | None = None,
        run_config: RunConfig | None = None,
        multiple_completion_supported: bool = True,
    ):
        if run_config is None:
            run_config = RunConfig()
        super().__init__(run_config, multiple_completion_supported)

        self.sync_client = LlamaStackClient(base_url=base_url)
        self.async_client = AsyncLlamaStackClient(base_url=base_url)
        self.model_id = model_id
        self.sampling_params = sampling_params

    def _initialize_llm_output(self) -> dict:
        """Create initial LLM output structure."""
        return {
            "llama_stack_responses": [],
            "model_id": self.model_id,
            "provider": "llama_stack_remote",
        }

    def _update_llm_output(
        self, response: CompletionCreateResponse, llm_output: dict
    ) -> None:
        """Process completion response and update llm_output."""
        choice = response.choices[0] if response.choices else None
        llama_stack_info = {
            "stop_reason": choice.finish_reason if choice else None,
            "content_length": len(choice.text) if choice else 0,
            "has_logprobs": choice.logprobs is not None if choice else False,
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
            generations = []
            llm_output = self._initialize_llm_output()

            # sampling params for this generation should be set via the benchmark config
            # we will ignore the temperature and stop params passed in here
            for _ in range(n):
                response: CompletionCreateResponse = (
                    self.sync_client.completions.create(
                        model=self.model_id,
                        prompt=prompt.to_string(),
                        max_tokens=self.sampling_params.max_tokens
                        if self.sampling_params
                        else omit,
                        temperature=self.sampling_params.strategy.temperature
                        if self.sampling_params
                        and isinstance(
                            self.sampling_params.strategy, TopPSamplingStrategy
                        )
                        else omit,
                        top_p=self.sampling_params.strategy.top_p
                        if self.sampling_params
                        and isinstance(
                            self.sampling_params.strategy, TopPSamplingStrategy
                        )
                        else omit,
                        stop=self.sampling_params.stop
                        if self.sampling_params
                        else omit,
                    )
                )

                if not response.choices:
                    logger.warning("Completion response returned no choices")

                self._update_llm_output(response, llm_output)
                choice = response.choices[0] if response.choices else None
                text = choice.text if choice else ""
                generations.append(Generation(text=text))

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
            generations = []
            llm_output = self._initialize_llm_output()

            # sampling params for this generation should be set via the benchmark config
            # we will ignore the temperature and stop params passed in here
            for _ in range(n):
                response: CompletionCreateResponse = (
                    await self.async_client.completions.create(
                        model=self.model_id,
                        prompt=prompt.to_string(),
                        max_tokens=self.sampling_params.max_tokens
                        if self.sampling_params
                        else omit,
                        temperature=self.sampling_params.strategy.temperature
                        if self.sampling_params
                        and isinstance(
                            self.sampling_params.strategy, TopPSamplingStrategy
                        )
                        else omit,
                        top_p=self.sampling_params.strategy.top_p
                        if self.sampling_params
                        and isinstance(
                            self.sampling_params.strategy, TopPSamplingStrategy
                        )
                        else omit,
                        stop=self.sampling_params.stop
                        if self.sampling_params
                        else omit,
                    )
                )

                if not response.choices:
                    logger.warning("Completion response returned no choices")

                self._update_llm_output(response, llm_output)
                choice = response.choices[0] if response.choices else None
                text = choice.text if choice else ""
                generations.append(Generation(text=text))

            return LLMResult(generations=[generations], llm_output=llm_output)

        except Exception as e:
            logger.error(f"Async LLM generation failed: {str(e)}")
            raise

    def get_temperature(self, n: int) -> float:
        """Get temperature based on number of completions."""
        return 0.3 if n > 1 else 1e-8
