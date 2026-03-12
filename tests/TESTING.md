# Testing

All test files live under `tests/`. Shared evaluation logic (smoke checks, eval job polling) is factored into `base_eval_tests.py`, which is not collected by pytest directly.

## Unit tests (`test_remote_wrappers.py`, pytest marker `unit`)

Tests the LangChain-compatible wrapper classes (`LlamaStackRemoteLLM` and `LlamaStackRemoteEmbeddings`) that the remote provider uses for inference. By default, the `LlamaStackClient` is mocked — no running server is required.

```bash
uv run pytest tests/test_remote_wrappers.py
```

Pass `--no-mock-client` to use a real `LlamaStackClient` against a running Llama Stack server (defaults to `http://localhost:8321`). Model IDs can be overridden with `INFERENCE_MODEL` and `EMBEDDING_MODEL`.

```bash
uv run pytest tests/test_remote_wrappers.py --no-mock-client
```

## Integration tests (`test_inline_evaluation.py`, pytest marker `lls_integration`)

Tests the eval providers through an in-process Llama Stack server using `LlamaStackAsLibraryClient`. The stack configuration (providers, models, storage) is built entirely in fixtures. By default, Ollama connectivity and inference are mocked.

```bash
uv run pytest tests/test_inline_evaluation.py
```

Pass `--no-mock-inference` to use a real Ollama instance for inference:

```bash
INFERENCE_MODEL=ollama/granite3.3:2b \
EMBEDDING_MODEL=ollama/all-minilm:latest \
    uv run pytest tests/test_inline_evaluation.py --no-mock-inference
```

## End-to-end tests (`test_e2e.py`, pytest marker `e2e`)

Tests against a fully deployed Llama Stack distribution on an OpenShift cluster. Requires the cluster environment from `cluster-deployment/` to be set up and a port-forward to the Llama Stack service:

```bash
oc port-forward -n ragas-test svc/lsd-ragas-test-service 8321:8321
uv run pytest tests/test_e2e.py
```

These tests exercise both the inline and remote eval providers through the Llama Stack eval API, including dataset registration, benchmark creation, and eval job execution with result polling.

## Model configuration

Each test module defines its own `inference_model` and `embedding_model` fixtures with defaults appropriate to its backend:

| Module | Inference default | Embedding default | Backend |
|--------|-------------------|-------------------|---------|
| `test_inline_evaluation.py` | `ollama/granite3.3:2b` | `ollama/all-minilm:latest` | In-process Ollama (library client) |
| `test_remote_wrappers.py` | `litellm/Mistral-Small-24B-W8A8` | `nomic-ai/nomic-embed-text-v1.5` | Mocked `LlamaStackClient` |
| `test_e2e.py` | `Mistral-Small-24B-W8A8` | `nomic-ai/nomic-embed-text-v1.5` | OpenShift cluster (see `cluster-deployment/manifests/configmap-and-secrets.yaml`) |

The `INFERENCE_MODEL` and `EMBEDDING_MODEL` environment variables override these defaults across all suites. When overriding, ensure the values match the models registered in the target environment — e.g. e2e defaults must match the OpenShift configmap, and inline defaults must use the `ollama/` prefix expected by the library client config.

## Cluster deployment (`cluster-deployment/`)

Contains the Containerfile, deployment/teardown scripts, and Kubernetes manifests needed to stand up the e2e test environment on OpenShift. See `cluster-deployment/deploy-e2e.sh` to deploy.
