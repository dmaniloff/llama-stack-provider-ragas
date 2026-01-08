#!/usr/bin/env bash

set -e
shopt -s expand_aliases
alias docker="podman"

# Configuration
CLUSTER_NAME="ragas-test"
REGISTRY_NAME="kind-registry"
REGISTRY_PORT="5001"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KIND_CONFIG="${SCRIPT_DIR}/kind-config.yaml"

echo "Starting Kind cluster setup for Ragas testing..."

# Check if Kind is installed
if ! command -v kind &> /dev/null; then
    echo "Error: kind is not installed. Please install it first:"
    echo "  brew install kind  # on macOS"
    echo "  # or visit: https://kind.sigs.k8s.io/docs/user/quick-start/#installation"
    exit 1
fi

# Check if kubectl is installed
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed. Please install it first:"
    echo "  brew install kubectl  # on macOS"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if cluster already exists
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "Kind cluster '${CLUSTER_NAME}' already exists."
    read -p "Do you want to delete and recreate it? (y/N): " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting existing cluster..."
        kind delete cluster --name="${CLUSTER_NAME}"
        # Also remove registry if it exists
        docker rm -f "${REGISTRY_NAME}" 2>/dev/null || true
    else
        echo "Using existing cluster. Setting up kubectl context..."
        kind export kubeconfig --name="${CLUSTER_NAME}"
        echo "Cluster '${CLUSTER_NAME}' is ready!"
        echo "Registry should be available at localhost:${REGISTRY_PORT}"
        exit 0
    fi
fi

# Check if registry container exists and remove if needed
if docker ps -a --format 'table {{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
    echo "Removing existing registry container..."
    docker rm -f "${REGISTRY_NAME}"
fi

# Create registry container
echo "Creating local Docker registry..."
docker run -d --restart=always -p "${REGISTRY_PORT}:5000" --name "${REGISTRY_NAME}" registry:2

# Create Kind cluster
echo "Creating Kind cluster with config: ${KIND_CONFIG}"
kind create cluster --config="${KIND_CONFIG}" --name="${CLUSTER_NAME}"

# Connect the registry to the cluster network
echo "Connecting registry to Kind network..."
docker network connect "kind" "${REGISTRY_NAME}" || true

# Configure cluster to use local registry
echo "Configuring cluster to use local registry..."
kubectl apply -f - <<EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: local-registry-hosting
  namespace: kube-public
data:
  localRegistryHosting.v1: |
    host: "localhost:${REGISTRY_PORT}"
    help: "https://kind.sigs.k8s.io/docs/user/local-registry/"
EOF

# Wait for cluster to be ready
echo "Waiting for cluster to be ready..."
kubectl wait --for=condition=ready node --all --timeout=60s

# Build and push custom ragas provider image
echo "Building ragas provider image..."
cd ../..
docker build -t localhost:${REGISTRY_PORT}/llama-stack-provider-ragas-distro-image:latest -f Containerfile .
docker push localhost:${REGISTRY_PORT}/llama-stack-provider-ragas-distro-image:latest --tls-verify=false
cd tests/e2e

# Install LlamaStack operator (includes CRDs)
echo "Installing LlamaStack operator..."
kubectl apply -f https://raw.githubusercontent.com/llamastack/llama-stack-k8s-operator/main/release/operator.yaml

# Create test namespace
echo "Creating ragas-test namespace..."
kubectl create namespace ragas-test

# Apply manifests
echo "Applying ragas test manifests..."
kubectl apply -f manifests/

# Wait for llama-stack service to be ready
echo "Waiting for llama-stack service to be ready..."
sleep 5
kubectl wait --for=condition=available deployment/lsd-ragas-test -n ragas-test

echo ""
echo "✅ Kind cluster setup complete!"
echo ""
echo "Cluster info:"
echo "  Name: ${CLUSTER_NAME}"
echo "  Registry: localhost:${REGISTRY_PORT}"
echo "  Kubectl context: kind-${CLUSTER_NAME}"
echo "  llama-stack API: http://localhost:8321"
echo ""
echo "Port forwarding is running in background (PID: ${PORTFORWARD_PID})"
echo "To stop port forwarding: kill ${PORTFORWARD_PID}"
echo ""
echo "Next steps:"
echo "  1. Verify cluster: kubectl get nodes"
echo "  2. Check deployments: kubectl get pods -n ragas-test"
echo "  3. Check LlamaStackDistribution: kubectl get llamastackdistributions -n ragas-test"
echo "  4. Setup port fwd: kubectl port-forward -n ragas-test service/lsd-ragas-test-service 8321:8321 &"
echo "  5. Test API: curl http://localhost:8321/v1/models"
echo "  6. Run tests: pytest tests/test_e2e_k3s.py -m e2e_test"
echo ""
echo "To delete the cluster later:"
echo "  kind delete cluster --name=${CLUSTER_NAME}"
