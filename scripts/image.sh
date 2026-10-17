#!/usr/bin/env bash
# Build, test, tag and push the desc-mcp-server container image.
#
#   bash scripts/image.sh                 # build linux/amd64 -> ghcr.io/nesar/desc-mcp:<locksha>-u24-cpu (+ latest)
#   bash scripts/image.sh --relock        # re-solve tools/env/environment-image.yml -> conda-lock-image.yml first
#   bash scripts/image.sh --test          # run tests/smoke_env.py + pytest inside the built image
#   bash scripts/image.sh --push          # push both tags (needs `docker login ghcr.io`; see README "Hosting")
#   bash scripts/image.sh --save DIR      # docker save to DIR/desc-mcp-<locksha>-u24-cpu.tar (registry-less fallback:
#                                         #   Polaris: apptainer build x.sif docker-archive://x.tar)
#   bash scripts/image.sh --platform linux/arm64   # when the lock carries linux-aarch64
#
# The tag IS the environment identity: first 12 hex of sha1(conda-lock-image.yml), the same
# hash the hep-genesis engine uses for lock envs, plus the base OS (-u24 = ubuntu 24.04,
# glibc 2.39: Polaris' injected Cray MPICH 9.1 needs GLIBC_2.38). A code-only change rebuilds
# the same tag; a base bump or a relock makes a new one.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/nesar/desc-mcp}"
LOCK="$REPO_DIR/tools/env/conda-lock-image.yml"
SRC="$REPO_DIR/tools/env/environment-image.yml"
PLATFORM="linux/amd64"
RELOCK=0; TEST=0; PUSH=0; SAVE=""; BUILD=1
while [ $# -gt 0 ]; do
    case "$1" in
        --relock) RELOCK=1 ;;
        --test) TEST=1 ;;
        --push) PUSH=1 ;;
        --save) SAVE="$2"; shift ;;
        --no-build) BUILD=0 ;;
        --platform) PLATFORM="$2"; shift ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$RELOCK" = 1 ]; then
    # micromamba is an accepted single-platform solver for conda-lock 4 (fast: ~30 s);
    # the laptop env.sh keeps using conda/mamba because it solves two platforms at once.
    MM="${MM:-$REPO_DIR/.micromamba/bin/micromamba}"
    [ -x "$MM" ] || MM="$(command -v micromamba || true)"
    [ -n "$MM" ] || { echo "--relock needs micromamba (bash scripts/env.sh fetches one)" >&2; exit 2; }
    command -v uvx >/dev/null || { echo "--relock needs uv (https://docs.astral.sh/uv/)" >&2; exit 2; }
    plats="-p linux-64"
    case "$PLATFORM" in *arm64*) plats="-p linux-64 -p linux-aarch64" ;; esac
    echo "Re-solving $SRC ($plats) with $MM ..."
    # shellcheck disable=SC2086
    uvx conda-lock lock --micromamba --conda "$MM" -f "$SRC" $plats --lockfile "$LOCK"
fi
[ -f "$LOCK" ] || { echo "no lock at $LOCK (run with --relock)" >&2; exit 2; }

SHA="$(shasum -a 1 "$LOCK" | cut -c1-12)"
BASE_TAG="${BASE_TAG:-u24}"   # must match the Dockerfile FROM line
case "$PLATFORM" in *arm64*) ARCH_TAG="-arm64" ;; *) ARCH_TAG="" ;; esac
TAG="$IMAGE:$SHA-$BASE_TAG-cpu$ARCH_TAG"
echo "image: $TAG (platform $PLATFORM)"

if [ "$BUILD" = 1 ]; then
    docker buildx build --platform "$PLATFORM" --load \
        --build-arg "LOCK_SHA=$SHA" \
        -t "$TAG" -t "$IMAGE:latest$ARCH_TAG" "$REPO_DIR"
fi

if [ "$TEST" = 1 ]; then
    echo "--- smoke_env.py inside $TAG"
    docker run --rm --platform "$PLATFORM" "$TAG" python tests/smoke_env.py
    echo "--- pytest inside $TAG"
    docker run --rm --platform "$PLATFORM" "$TAG" python -m pytest tests/ -q -x --ignore=tests/smoke_chain_mpi.py
fi

if [ -n "$SAVE" ]; then
    mkdir -p "$SAVE"
    out="$SAVE/desc-mcp-$SHA-$BASE_TAG-cpu$ARCH_TAG.tar"
    echo "--- docker save -> $out"
    docker save -o "$out" "$TAG"
    ls -lh "$out"
fi

if [ "$PUSH" = 1 ]; then
    docker push "$TAG"
    docker push "$IMAGE:latest$ARCH_TAG"
    docker buildx imagetools inspect "$TAG" | sed -n '1,6p'
fi
