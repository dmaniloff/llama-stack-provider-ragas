import warnings

# Ragas v0.4.x emits deprecation warnings for module-level metric instances
# imported from ragas.metrics. These still work with evaluate() and will be
# migrated to the collections API in a future release.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
    from ragas.metrics import (
        AnswerAccuracy,
        ContextRelevance,
        FactualCorrectness,
        NoiseSensitivity,
        ResponseGroundedness,
        answer_relevancy,
        answer_similarity,
        context_entity_recall,
        context_precision,
        context_recall,
        faithfulness,
    )

PROVIDER_TYPE = "trustyai_ragas"
PROVIDER_ID_INLINE = "trustyai_ragas_inline"
PROVIDER_ID_REMOTE = "trustyai_ragas_remote"

# Pre-instantiated metric singletons (from ragas)
_SINGLETON_METRICS = [
    answer_relevancy,
    answer_similarity,
    context_precision,
    faithfulness,
    context_recall,
    context_entity_recall,
]

# Class-based metrics (new in ragas v0.4.x) that need instantiation.
# Note: BleuScore, ChrfScore, and RougeScore are omitted because they
# require optional dependencies (sacrebleu, rouge_score).
_CLASS_METRICS = [
    AnswerAccuracy(),
    ContextRelevance(),
    FactualCorrectness(),
    NoiseSensitivity(),
    ResponseGroundedness(),
]

METRIC_MAPPING = {m.name: m for m in _SINGLETON_METRICS + _CLASS_METRICS}
AVAILABLE_METRICS = list(METRIC_MAPPING.keys())

# Kubeflow ConfigMap keys and defaults for base image resolution
RAGAS_PROVIDER_IMAGE_CONFIGMAP_NAME = "trustyai-service-operator-config"
RAGAS_PROVIDER_IMAGE_CONFIGMAP_KEY = "ragas-provider-image"
DEFAULT_RAGAS_PROVIDER_IMAGE = "registry.access.redhat.com/ubi9/python-312:latest"
KUBEFLOW_CANDIDATE_NAMESPACES = ["redhat-ods-applications", "opendatahub"]
