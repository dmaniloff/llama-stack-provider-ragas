from ragas.metrics import (
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)

PROVIDER_TYPE = "trustyai_ragas"
PROVIDER_ID_INLINE = "trustyai_ragas_inline"
PROVIDER_ID_REMOTE = "trustyai_ragas_remote"

METRIC_MAPPING = {
    metric_func.name: metric_func
    for metric_func in [
        answer_relevancy,
        answer_similarity,
        context_precision,
        faithfulness,
        context_recall,
        # Can add other metrics here, e.g.:
        # "rouge_score": RougeScore(),
    ]
}
AVAILABLE_METRICS = list(METRIC_MAPPING.keys())

# Kubeflow ConfigMap keys and defaults for base image resolution
RAGAS_PROVIDER_IMAGE_CONFIGMAP_NAME = "trustyai-service-operator-config"
RAGAS_PROVIDER_IMAGE_CONFIGMAP_KEY = "ragas-provider-image"
DEFAULT_RAGAS_PROVIDER_IMAGE = "registry.access.redhat.com/ubi9/python-312:latest"
KUBEFLOW_CANDIDATE_NAMESPACES = ["redhat-ods-applications", "opendatahub"]
