<p align="center">
  <img src="https://raw.githubusercontent.com/trustyai-explainability/llama-stack-provider-ragas/main/docs/_static/provider-logo.png" alt="Llama Stack Provider" height="120">
</p>

# Ragas as an External Provider for Llama Stack

[![PyPI version](https://img.shields.io/pypi/v/llama_stack_provider_ragas.svg)](https://pypi.org/project/llama-stack-provider-ragas/)


## About
This repository implements [Ragas](https://github.com/explodinggradients/ragas) as an out-of-tree [Llama Stack](https://github.com/meta-llama/llama-stack) evaluation provider.

## Features
The goal is to provide all of Ragas' evaluation functionality over Llama Stack's eval API, while leveraging the Llama Stack's built-in APIs for inference (llms and embeddings), datasets, and benchmarks.

There are two versions of the provider:
- `inline`: runs the Ragas evaluation in the same process as the Llama Stack server.
- `remote`: runs the Ragas evaluation in a remote process, using Kubeflow Pipelines.

## Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- The remote provider requires a running [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines) server.

## Setup
- Clone this repository
    ```bash
    git clone <repository-url>
    cd llama-stack-provider-ragas
    ```

- Create and activate a virtual environment
    ```bash
    uv venv
    source .venv/bin/activate
    ```

- Install (optionally as an editable package). There's `distro`, `remote` and `dev` optional dependencies to run the sample LS distribution and the KFP-enabled remote provider. Installing the `dev` dependencies will also install the `distro` and `remote` dependencies.
    ```bash
    uv pip install -e ".[dev]"
    ```
- The sample LS distributions (one for inline and one for remote provider) is a simple LS distribution that uses Ollama for inference and embeddings. See the provider-specific sections below for setup and run commands.

### Remote provider (default)

Create a `.env` file with the following:
```bash
# Required for both inline and remote
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Required for remote provider
KUBEFLOW_LLAMA_STACK_URL=<your-llama-stack-url>
KUBEFLOW_PIPELINES_ENDPOINT=<your-kfp-endpoint>
KUBEFLOW_NAMESPACE=<your-namespace>
KUBEFLOW_BASE_IMAGE=quay.io/diegosquayorg/my-ragas-provider-image:latest
KUBEFLOW_PIPELINES_TOKEN=<your-pipelines-token>
KUBEFLOW_RESULTS_S3_PREFIX=s3://my-bucket/ragas-results
KUBEFLOW_S3_CREDENTIALS_SECRET_NAME=<secret-name>
```

Where:
- `KUBEFLOW_LLAMA_STACK_URL`: The URL of the llama stack server that the remote provider will use to run the evaluation (LLM generations and embeddings, etc.). If you are running Llama Stack locally, you can use [ngrok](https://ngrok.com/) to expose it to the remote provider.
- `KUBEFLOW_PIPELINES_ENDPOINT`: You can get this via `kubectl get routes -A | grep -i pipeline` on your Kubernetes cluster.
- `KUBEFLOW_NAMESPACE`: The name of the data science project where the Kubeflow Pipelines server is running.
- `KUBEFLOW_PIPELINES_TOKEN`: Kubeflow Pipelines token with access to submit pipelines. If not provided, the token will be read from the local kubeconfig file.
- `KUBEFLOW_BASE_IMAGE`: The image used to run the Ragas evaluation in the remote provider. See `Containerfile` for details. There is a public version of this image at `quay.io/diegosquayorg/my-ragas-provider-image:latest`.
- `KUBEFLOW_RESULTS_S3_PREFIX`: S3 location (bucket and prefix folder) where evaluation results will be stored, e.g., `s3://my-bucket/ragas-results`.
- `KUBEFLOW_S3_CREDENTIALS_SECRET_NAME`: Name of the Kubernetes secret containing AWS credentials with write access to the S3 bucket. Create with:
  ```bash
  oc create secret generic <secret-name> \
    --from-literal=AWS_ACCESS_KEY_ID=your-access-key \
    --from-literal=AWS_SECRET_ACCESS_KEY=your-secret-key \
    --from-literal=AWS_DEFAULT_REGION=us-east-1
  ```

Run the server:
```bash
dotenv run uv run llama stack run distribution/run-remote.yaml
```

### Inline provider (need to specify `.inline` in the module name)

Create a `.env` file with the required environment variable:
```bash
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

Run the server:
```bash
dotenv run uv run llama stack run distribution/run-inline.yaml
```

You will notice that `run-inline.yaml` file has the module name as `llama_stack_provider_ragas.inline`, in order to specify the inline provider.

## Usage
See the demos in the `demos` directory.
