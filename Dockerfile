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
#   docker buildx build --platform linux/amd64 --build-arg LOCK_SHA=$(shasum -a 1 tools/env/conda-lock-image.yml | cut -c1-12) -t ghcr.io/nesar/desc-mcp:<sha>-u24-cpu .
#
# Base OS: ubuntu 24.04 (glibc 2.39). Polaris injects Cray MPICH 9.1 via cray-mpich-abi, whose
# libraries need GLIBC_2.38; on ubuntu 22.04 (glibc 2.35) mpi4py failed to import on Polaris
# (Perlmutter's injected MPICH is fine with either). The tag carries the base OS (-u24) so a
# base bump is a new image even with the same lock.
FROM ubuntu:24.04
ARG LOCK_SHA=unknown

LABEL org.opencontainers.image.source="https://github.com/nesar/desc_mcp_server" \
      org.opencontainers.image.description="LSST DESC cosmology stack (CCL, sacc, firecrown, augur, TJPCov, NaMaster, Smokescreen) as MCP tools" \
      org.desc-mcp.lock-sha="${LOCK_SHA}" \
      org.desc-mcp.base="ubuntu:24.04 glibc 2.39 (Polaris cray-mpich-abi needs >= 2.38)" \
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

# No apt at all: ubuntu 24.04's apt verifies signatures with sqv, which fails under
# qemu user-mode emulation ("At least one invalid signature was encountered" when building
# linux/amd64 on an Apple-silicon laptop). Everything the build needs comes from elsewhere:
# the static micromamba binary from the official image (same pinned version as the GitHub
# release scripts/env.sh fetches), git + ca-certificates from conda-forge (both in the lock,
# linked before the lock's pip section clones augur/Smokescreen; /opt/env/bin is on PATH).
COPY --from=mambaorg/micromamba:2.9.0 /bin/micromamba /opt/micromamba/bin/micromamba
# CA bundle for micromamba's downloads ("No CA certificates found on system" otherwise)
COPY --from=mambaorg/micromamba:2.9.0 /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
RUN micromamba --version

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
