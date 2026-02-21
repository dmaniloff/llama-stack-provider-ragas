#!/usr/bin/env bash
#
# Deploy the llama-stack-provider-ragas e2e test environment on an OpenShift cluster.
#
# Usage:
#   ./deploy-e2e.sh --build
#   ./deploy-e2e.sh --image <image-ref>
#
# Reads credentials from ../../.env (repo root) and creates a single
# 'ragas-env' k8s secret from it.
#
# Prerequisites:
#   - oc CLI installed and logged into an OpenShift cluster
#   - podman (only required for --build mode)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}/../.."
IMAGE_NAME="llama-stack-provider-ragas-distro-image"
NAMESPACE="ragas-test"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
MODE=""
IMAGE_REF=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            MODE="build"
            shift
            ;;
        --image)
            MODE="image"
            IMAGE_REF="$2"
            if [[ -z "${IMAGE_REF}" ]]; then
                echo "Error: --image requires an image reference argument."
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --build | --image <image-ref>"
            exit 1
            ;;
    esac
done

if [[ -z "${MODE}" ]]; then
    echo "Usage: $0 --build | --image <image-ref>"
    exit 1
fi

# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------
echo "Checking prerequisites..."

if ! command -v oc &> /dev/null; then
    echo "Error: oc is not installed."
    exit 1
fi

if ! oc whoami &> /dev/null; then
    echo "Error: Not logged into an OpenShift cluster. Run 'oc login' first."
    exit 1
fi

echo "  Logged in as: $(oc whoami)"
echo "  Cluster: $(oc whoami --show-server)"

# ---------------------------------------------------------------------------
# Resolve image
# ---------------------------------------------------------------------------
if [[ "${MODE}" == "build" ]]; then
    if ! command -v podman &> /dev/null; then
        echo "Error: podman is not installed (required for --build)."
        exit 1
    fi

    echo ""
    echo "=== Building image from Containerfile ==="

    # Detect cluster node architecture (not local host arch)
    NODE_ARCH=$(oc get nodes -o jsonpath='{.items[0].status.nodeInfo.architecture}' 2>/dev/null || echo "amd64")
    case "${NODE_ARCH}" in
        amd64)  PLATFORM="linux/amd64" ;;
        arm64)  PLATFORM="linux/arm64" ;;
        *)      echo "Warning: unknown cluster architecture ${NODE_ARCH}, defaulting to linux/amd64"; PLATFORM="linux/amd64" ;;
    esac
    echo "  Cluster node architecture: ${NODE_ARCH} -> ${PLATFORM}"

    # Build the image
    LOCAL_TAG="${IMAGE_NAME}:latest"
    echo "  Building ${LOCAL_TAG}..."
    podman build --no-cache --platform "${PLATFORM}" \
        -t "${LOCAL_TAG}" \
        -f "${SCRIPT_DIR}/Containerfile" "${REPO_ROOT}"

    # Expose the OpenShift internal registry route (idempotent)
    echo "  Exposing OpenShift internal registry..."
    oc patch configs.imageregistry.operator.openshift.io/cluster \
        --type=merge --patch '{"spec":{"defaultRoute":true}}' 2>/dev/null || true

    # Wait briefly for the route to appear
    for i in $(seq 1 12); do
        REGISTRY_ROUTE=$(oc get route default-route -n openshift-image-registry \
            --template='{{ .spec.host }}' 2>/dev/null) && break
        sleep 5
    done

    if [[ -z "${REGISTRY_ROUTE}" ]]; then
        echo "Error: Could not determine the OpenShift internal registry route."
        exit 1
    fi
    echo "  Registry route: ${REGISTRY_ROUTE}"

    # Login to the registry
    echo "  Logging into registry..."
    podman login --tls-verify=false -u "$(oc whoami)" -p "$(oc whoami -t)" "${REGISTRY_ROUTE}"

    # Ensure the namespace exists before pushing (registry needs the namespace/project)
    oc create namespace "${NAMESPACE}" 2>/dev/null || true

    # Tag and push
    REMOTE_TAG="${REGISTRY_ROUTE}/${NAMESPACE}/${IMAGE_NAME}:latest"
    echo "  Tagging ${LOCAL_TAG} -> ${REMOTE_TAG}"
    podman tag "${LOCAL_TAG}" "${REMOTE_TAG}"

    echo "  Pushing to internal registry..."
    podman push --tls-verify=false "${REMOTE_TAG}"

    # The in-cluster image reference uses the internal service address
    IMAGE_REF="image-registry.openshift-image-registry.svc:5000/${NAMESPACE}/${IMAGE_NAME}:latest"
    echo "  In-cluster image ref: ${IMAGE_REF}"

elif [[ "${MODE}" == "image" ]]; then
    echo ""
    echo "=== Using pre-built image ==="
    echo "  Image: ${IMAGE_REF}"
fi

# ---------------------------------------------------------------------------
# Install LlamaStack operator
# ---------------------------------------------------------------------------
echo ""
echo "=== Installing LlamaStack operator ==="
oc apply -f https://raw.githubusercontent.com/llamastack/llama-stack-k8s-operator/main/release/operator.yaml

echo "Waiting for LlamaStack operator to be ready..."
oc wait --for=condition=available deployment/llama-stack-k8s-operator-controller-manager \
    -n llama-stack-k8s-operator-system --timeout=120s

# ---------------------------------------------------------------------------
# Create namespace and apply manifests
# ---------------------------------------------------------------------------
echo ""
echo "=== Setting up ${NAMESPACE} namespace ==="
oc create namespace "${NAMESPACE}" 2>/dev/null || true

echo "Applying configmaps and secrets..."
oc apply -f "${SCRIPT_DIR}/manifests/configmap-and-secrets.yaml"

echo "Creating ragas-env secret from .env..."
ENV_FILE="${REPO_ROOT}/.env"
if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Error: ${ENV_FILE} not found."
    exit 1
fi
oc create secret generic ragas-env -n "${NAMESPACE}" \
    --from-env-file="${ENV_FILE}" \
    --dry-run=client -o yaml | oc apply -f -

echo "Applying MinIO..."
oc apply -f "${SCRIPT_DIR}/manifests/minio.yaml"

echo "Applying LlamaStackDistribution CR (image: ${IMAGE_REF})..."
sed "s|__LLAMA_STACK_IMAGE__|${IMAGE_REF}|g" \
    "${SCRIPT_DIR}/manifests/llama-stack-distribution.yaml" | oc apply -f -

# ---------------------------------------------------------------------------
# Wait for MinIO
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for MinIO ==="
echo "Waiting for MinIO deployment..."
oc wait --for=condition=available deployment/minio -n "${NAMESPACE}" --timeout=120s

echo "Waiting for MinIO bucket creation job..."
oc wait --for=condition=complete job/minio-create-bucket -n "${NAMESPACE}" --timeout=120s

# ---------------------------------------------------------------------------
# Kubeflow pipeline resources (aws-credentials in ragas-test namespace)
# ---------------------------------------------------------------------------
echo ""
echo "=== Applying Kubeflow pipeline resources ==="
oc apply -f "${SCRIPT_DIR}/manifests/kubeflow-pipeline-resources.yaml"

# ---------------------------------------------------------------------------
# Wait for operator reconciliation and deployments
# ---------------------------------------------------------------------------
echo ""
echo "=== Waiting for deployments ==="

echo "Waiting for operator to reconcile LlamaStackDistribution..."
for i in $(seq 1 30); do
    if oc get deployment/lsd-ragas-test -n "${NAMESPACE}" &>/dev/null; then
        echo "  Deployment created."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Error: Timed out waiting for deployment/lsd-ragas-test to be created by the operator."
        exit 1
    fi
    sleep 5
done

echo "Waiting for llama-stack deployment..."
oc wait --for=condition=available deployment/lsd-ragas-test -n "${NAMESPACE}" --timeout=300s

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================="
echo " E2E deployment complete!"
echo "========================================="
echo ""
echo "  Namespace: ${NAMESPACE}"
echo "  Image:     ${IMAGE_REF}"
echo "  Env file:  ${ENV_FILE}"
echo ""
echo "Next steps:"
echo "  1. Verify pods:    oc get pods -n ${NAMESPACE}"
echo "  2. Port forward:   oc port-forward -n ${NAMESPACE} svc/lsd-ragas-test-service 8321:8321 &"
echo "  3. Test API:       curl http://localhost:8321/v1/models"
echo ""
echo "To tear down:"
echo "  ./teardown-e2e.sh"
