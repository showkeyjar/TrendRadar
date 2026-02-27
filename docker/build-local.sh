#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[build] building trendradar local image..."
docker build -f docker/Dockerfile -t trendradar:local .

echo "[build] building trendradar mcp local image..."
docker build -f docker/Dockerfile.mcp -t trendradar-mcp:local .

echo "[done] images:"
docker images | grep -E "trendradar(:local)?|trendradar-mcp(:local)?" || true

