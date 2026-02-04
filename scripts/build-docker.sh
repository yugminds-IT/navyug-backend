#!/usr/bin/env bash
# Build Docker image for Face Recognition API.
# Usage:
#   ./scripts/build-docker.sh                    # build only, tag: face-recognition-api:latest
#   ./scripts/build-docker.sh -t myreg/myimg:v1  # build and tag for registry
#   ./scripts/build-docker.sh -t myreg/myimg:v1 --push  # build, tag, and push

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-face-recognition-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
DEFAULT_TAG="${IMAGE_NAME}:${IMAGE_TAG}"
PUSH=false
CUSTOM_TAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--tag)
      CUSTOM_TAG="$2"
      shift 2
      ;;
    --push)
      PUSH=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [-t|--tag REGISTRY/IMAGE:TAG] [--push]"
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"
echo "Building Docker image in $ROOT_DIR ..."
docker build -t "$DEFAULT_TAG" .

if [[ -n "$CUSTOM_TAG" ]]; then
  echo "Tagging as $CUSTOM_TAG"
  docker tag "$DEFAULT_TAG" "$CUSTOM_TAG"
fi

if [[ "$PUSH" == true ]]; then
  if [[ -z "$CUSTOM_TAG" ]]; then
    echo "Use -t REGISTRY/IMAGE:TAG with --push to push to a registry."
    exit 1
  fi
  echo "Pushing $CUSTOM_TAG ..."
  docker push "$CUSTOM_TAG"
fi

echo "Done. Image: $DEFAULT_TAG"
[[ -n "$CUSTOM_TAG" ]] && echo "Also: $CUSTOM_TAG"
