# desc-mcp-server container image (linux/amd64; linux/arm64 when the lock resolves).
#
# One image, two uses:
#   * hosted server:   docker run -p 8008:8008 ghcr.io/nesar/desc-mcp
#   * compute kernels: the hep-genesis engine runs `python runner.py` inside it
#     (podman-hpc on Perlmutter, apptainer on Polaris) with the job dir bind-mounted.
#
# The environment is the image lock (tools/env/conda-lock-image.yml, MPICH variant of
# the laptop lock); the server package is pip-installed on top. No vendored clones,
# no example data, no facility identity.
#
# Build (see scripts/image.sh):
#   docker buildx build --platform linux/amd64 --build-arg LOCK_SHA=$(shasum -a 1 tools/env/conda-lock-image.yml | cut -c1-12) -t ghcr.io/nesar/desc-mcp:<sha>-cpu .
FROM ubuntu:22.04
ARG LOCK_SHA=unknown
# release tag in mamba-org/micromamba-releases (version-build); 2.9.0 is what scripts/env.sh fetched
ARG MICROMAMBA_VERSION=2.9.0-0
ARG TARGETARCH

LABEL org.opencontainers.image.source="https://github.com/nesar/desc_mcp_server" \
      org.opencontainers.image.description="LSST DESC cosmology stack (CCL, sacc, firecrown, augur, TJPCov, NaMaster, Smokescreen) as MCP tools" \
      org.desc-mcp.lock-sha="${LOCK_SHA}" \
      org.desc-mcp.mpi="mpich-abi (bundled MPICH in /opt/env/lib/mpich-bundled; host MPI injected via LD_LIBRARY_PATH wins)"

ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/micromamba \
    PATH=/opt/env/bin:/opt/micromamba/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    XDG_CACHE_HOME=/tmp/cache \
    XDG_CONFIG_HOME=/tmp/config \
    OMP_STACKSIZE=512M

# git + ca-certificates: the lock's pip section installs augur/Smokescreen from git.
# libgomp/libgfortran come from conda-forge; nothing else from apt.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git bzip2 \
 && rm -rf /var/lib/apt/lists/*

# micromamba, pinned, straight from the GitHub release (same source scripts/env.sh uses)
RUN set -eu; \
    case "${TARGETARCH}" in amd64) plat=linux-64 ;; arm64) plat=linux-aarch64 ;; *) echo "unsupported arch ${TARGETARCH}" >&2; exit 2 ;; esac; \
    mkdir -p /opt/micromamba/bin; \
    curl -fsSL --retry 3 -o /opt/micromamba/bin/micromamba \
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-${plat}"; \
    chmod +x /opt/micromamba/bin/micromamba; \
    micromamba --version

# the environment, exactly the lock (no solve); caches dropped in the same layer.
# micromamba recognises a conda-lock file ONLY by the name `conda-lock.yml` (any other
# name silently yields "Empty environment created"), hence the rename on COPY.
COPY tools/env/conda-lock-image.yml /opt/env-lock/conda-lock.yml
RUN micromamba create -y -p /opt/env -f /opt/env-lock/conda-lock.yml \
 && micromamba clean --all --yes \
 && rm -rf /opt/micromamba/pkgs /root/.cache \
 && find /opt/env -name '*.pyc' -delete \
 && echo "${LOCK_SHA}" > /opt/env/.lock-sha

# MPI shim. conda-forge binaries carry an RPATH to /opt/env/lib, which the loader
# searches BEFORE LD_LIBRARY_PATH; a Cray MPICH injected by podman-hpc --mpi or
# APPTAINERENV_LD_LIBRARY_PATH would lose to the bundled one. So the bundled MPICH
# libraries move to a side directory found only through ld.so.cache (searched AFTER
# LD_LIBRARY_PATH): `docker run` on a laptop/VM still gets a working MPI, a facility
# gets its own. Everything else in /opt/env/lib stays put.
RUN set -eu; \
    mkdir -p /opt/env/lib/mpich-bundled; \
    for f in /opt/env/lib/libmpi.so* /opt/env/lib/libmpich.so* /opt/env/lib/libmpicxx.so* /opt/env/lib/libmpifort.so*; do \
        [ -e "$f" ] && mv "$f" /opt/env/lib/mpich-bundled/ || true; \
    done; \
    echo /opt/env/lib/mpich-bundled > /etc/ld.so.conf.d/mpich-bundled.conf; \
    ldconfig; \
    ls /opt/env/lib/mpich-bundled; \
    python -c "from mpi4py import MPI; print(MPI.Get_library_version().splitlines()[0])"

# the server itself (not a dependency: never in the lock)
WORKDIR /opt/desc-mcp-server
COPY . /opt/desc-mcp-server
RUN python -m pip install --no-deps --no-cache-dir . \
 && rm -rf build *.egg-info

# writable scratch for caches/artifacts whoever the uid is (podman-hpc --userns keep-id,
# apptainer run-as-user). Hosted deployments mount /srv/artifacts.
RUN mkdir -p /tmp/mpl /tmp/cache /tmp/config /srv/artifacts && chmod 1777 /tmp/mpl /tmp/cache /tmp/config /srv/artifacts
ENV MCP_OUTPUT_ROOT=/srv/artifacts

EXPOSE 8008
# No ENTRYPOINT on purpose: podman-hpc's --mpi/--gpu modules fight ENTRYPOINTs and the
# engine runs kernels as `<image> python runner.py`. `docker run <image>` starts the server.
# --host 0.0.0.0: the default 127.0.0.1 is unreachable through the container's port mapping;
# bind the HOST side to localhost instead (-p 127.0.0.1:8008:8008) and front it with TLS.
CMD ["python", "-m", "mcp_server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8008"]
