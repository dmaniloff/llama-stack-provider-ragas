# Deploying Llama Stack on OpenShift AI with the remote Ragas eval provider

## Prerequisites
* OpenShift AI or Open Data Hub installed on your OpenShift Cluster
* Data Science Pipeline Server configured
* Llama Stack Operator installed
* A VLLM hosted Model either through Kserve or MaaS. You can follow these [docs](https://docs.redhat.com/en/documentation/red_hat_openshift_ai_cloud_service/1/html/working_with_rag/deploying-a-rag-stack-in-a-data-science-project_rag#Deploying-a-llama-model-with-kserve_rag) until step 3.4

## Setup
Create a secret for storing your model's information.
```
export INFERENCE_MODEL="llama-3-2-3b"
export VLLM_URL="https://llama-32-3b-instruct-predictor:8443/v1"
export VLLM_TLS_VERIFY="false" # Use "true" in production!
export VLLM_API_TOKEN="<token identifier>"

oc create secret generic llama-stack-inference-model-secret \
  --from-literal INFERENCE_MODEL="$INFERENCE_MODEL" \
  --from-literal VLLM_URL="$VLLM_URL" \
  --from-literal VLLM_TLS_VERIFY="$VLLM_TLS_VERIFY" \
  --from-literal VLLM_API_TOKEN="$VLLM_API_TOKEN"
```

## Setup Deployment files
### Configuring the `kubeflow-ragas-config` ConfigMap
Update the [kubeflow-ragas-config](deployment/kubeflow-ragas-config.yaml) with the following data:
``` bash
# See project README for more details
EMBEDDING_MODEL=all-MiniLM-L6-v2
KUBEFLOW_LLAMA_STACK_URL=<your-llama-stack-url>
KUBEFLOW_PIPELINES_ENDPOINT=<your-kfp-endpoint>
KUBEFLOW_NAMESPACE=<your-namespace>
KUBEFLOW_BASE_IMAGE=quay.io/diegosquayorg/my-ragas-provider-image:latest
KUBEFLOW_RESULTS_S3_PREFIX=s3://my-bucket/ragas-results
KUBEFLOW_S3_CREDENTIALS_SECRET_NAME=<secret-name>
```

> [!NOTE]
> The `KUBEFLOW_LLAMA_STACK_URL` must be an external route.

### Configuring the `pipelines_token` Secret
Unfortunately the Llama Stack distribution service account does not have privilages to create pipeline runs. In order to work around this we must provide a user token as a secret to the Llama Stack Distribution.

Create the secret with:
``` bash
# Gather your token with `oc whoami -t`
kubectl create secret generic kubeflow-pipelines-token \
  --from-literal=KUBEFLOW_PIPELINES_TOKEN=<your-pipelines-token>
```

## Deploy Llama Stack on OpenShift
You can now deploy the configuration files and the Llama Stack distribution with `oc apply -f deployment/kubeflow-ragas-config.yaml` and `oc apply -f deployment/llama-stack-distribution.yaml`

You should now have a Llama Stack server on OpenShift with the remote ragas eval provider configured.
You can now follow the [remote_demo.ipynb](../../demos/remote_demo.ipynb) demo but ensure you are running it in a Data Science workbench and use the `LLAMA_STACK_URL` defined earlier. Alternatively you can run it locally if you create a Route.
