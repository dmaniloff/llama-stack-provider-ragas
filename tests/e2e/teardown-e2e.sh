#!/usr/bin/env bash
#
# Tear down the llama-stack-provider-ragas e2e test environment.
#

set -e

echo "Tearing down e2e test environment..."

oc delete namespace ragas-test --ignore-not-found

echo ""
echo "Teardown complete."
