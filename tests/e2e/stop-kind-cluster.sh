#!/usr/bin/env bash

set -e

# Configuration
CLUSTER_NAME="ragas-test"
REGISTRY_NAME="kind-registry"

echo "Stopping Kind cluster setup for Ragas testing..."

# Delete Kind cluster
if kind get clusters | grep -q "^${CLUSTER_NAME}$"; then
    echo "Deleting Kind cluster '${CLUSTER_NAME}'..."
    kind delete cluster --name="${CLUSTER_NAME}"
    echo "✅ Kind cluster deleted"
else
    echo "ℹ️  Kind cluster '${CLUSTER_NAME}' does not exist"
fi

# Remove registry container
if docker ps -a --format 'table {{.Names}}' | grep -q "^${REGISTRY_NAME}$"; then
    echo "Removing registry container '${REGISTRY_NAME}'..."
    docker rm -f "${REGISTRY_NAME}"
    echo "✅ Registry container removed"
else
    echo "ℹ️  Registry container '${REGISTRY_NAME}' does not exist"
fi

echo ""
echo "✅ Cleanup complete!"
