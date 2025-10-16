from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRIC_MAPPING = {
    metric_func.name: metric_func
    for metric_func in [
        answer_relevancy,
        context_precision,
        faithfulness,
        context_recall,
        # TODO: add these later
        # "answer_correctness": AnswerCorrectness(),
        # "factual_correctness": FactualCorrectness(),
        # "summarization_score": SummarizationScore(),
        # "bleu_score": BleuScore(),
        # "rouge_score": RougeScore(),
    ]
}

AVAILABLE_METRICS = list(METRIC_MAPPING.keys())

# Kubeflow ConfigMap keys and defaults for base image resolution
RAGAS_PROVIDER_IMAGE_CONFIGMAP_NAME = "trustyai-service-operator-config"
RAGAS_PROVIDER_IMAGE_CONFIGMAP_KEY = "ragas-provider-image"
DEFAULT_RAGAS_PROVIDER_IMAGE = "quay.io/trustyai/llama-stack-provider-ragas:latest"
KUBEFLOW_CANDIDATE_NAMESPACES = ["redhat-ods-applications", "opendatahub"]
