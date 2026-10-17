# Environment notes — `desc-mcp`

One environment, defined once, used everywhere: `tools/env/environment.yml`
is the source, `tools/env/conda-lock.yml` the exact solve for `osx-arm64`
(laptop) and `linux-64` (hosted VMs and the ALCF/NERSC compute nodes).
`bash scripts/env.sh` fetches micromamba into `.micromamba/` and creates
`.mcp-env/` from the lock (no conda on the machine); the same lock ships inside
the dispatch pack and the hep-genesis engine builds it on the compute node
(`<WORKDIR>/envs/cl-<hash>`, once per lock). Validate with
`.mcp-env/bin/python tests/smoke_env.py` and `.mcp-env/bin/python -m pytest tests/ -q`,
from the repo root, never from inside a clone directory (the CCL source tree
shadows the installed `pyccl`). To change a version: edit `environment.yml`,
`bash scripts/env.sh --relock` (needs conda or mamba on PATH for the solve, and
`uv`), commit both files.

## What's installed and verified working (2026-10-07, macOS arm64)

| Package | Version | Source | Role |
|---|---|---|---|
| pyccl | 3.3.6 | conda-forge | all `ccl_*` theory tools; also used by sacc covariance + firecrown |
| camb | 1.6.5 | conda-forge | CCL's default transfer function (`boltzmann_camb`), HMcode-2020 |
| sacc | 2.4 | conda-forge | the data-vector format exchanged between every family |
| firecrown | 1.16.0 | conda-forge (`firecrown-deps` metapackage) | likelihoods; needs numcosmo + crow at import |
| numcosmo | 0.28 | conda-forge | pulled by firecrown (n(z) generators) |
| lsstdesc-crow | 1.0.12 | conda-forge | pulled by firecrown (cluster statistics) |
| augur | 1.2.4 | **built from the clone** (`augur/`) via a temporary copy | Fisher forecasts |
| tjpcov | 0.5.1 | conda-forge | `tjpcov_*` covariances (Gaussian f_sky, SSC, cNG, real space); augur `cov_type: tjpcov` |
| pymaster (NaMaster) | 3.0 | conda-forge (`namaster`, pulled by tjpcov) | `namaster_*` bandpowers, masks, simulated maps |
| smokescreen | 1.5.6 | **built from the clone** (`external/Smokescreen`) via a temporary copy | `smokescreen_*` data-vector concealment, encryption |
| jsonargparse, cryptography | 4.x, 46.x | pip | Smokescreen's CLI parser and Fernet encryption |
| lsstdesc-ceci | 2.5.1 | conda-forge | pipeline YAML handling for the TXPipe family (TXPipe itself is NOT installed) |
| cosmosis | 3.26 | conda-forge (via firecrown-deps) | present; CSL not built; not used by first-release tools |
| cobaya, getdist | 3.6.2 | pip | `firecrown_run_chain` (pure-CCL mode, no CAMB theory block) |
| isitgr | 1.6.5 | pip | firecrown's mu-Sigma MG mode (optional) |
| derivkit, numdifftools | 1.4.0, 0.11.1 | pip / conda-forge | augur derivative engines |
| mcp, pydantic, matplotlib | 1.x, 2.13, 3.11 | pip / conda-forge | the server itself |
| hep_genesis | 0.3.12 | `pip install -e <hep-genesis-agent>/backend[iri]` | optional: server-side HPC dispatch |

Build time on this machine: ~3 min with the libmamba solver, conda-forge
only (`--override-channels`). Mixing in the `defaults` channel is the one
sure way to break the solve.

## Version pins and why

- **python 3.12**: intersection of CCL (>=3.10), firecrown (>=3.12), augur (>=3.11).
- **numpy >=2.0,<2.4**: firecrown's cap; everything else is numpy-2 clean.
- **camb <2.0**: firecrown's cap (CAMB 2 changed the API).
- **sacc >=2.4**: firecrown 1.16 requires it. augur's own `environment.yml`
  says `sacc<2.2` (a stale workaround from 2026-08); DESC's `desc-cosmology`
  lock ships augur 1.2.3 with sacc 2.4, and the test suite here runs augur
  1.2.4 against sacc 2.4 — treat the `<2.2` pin as obsolete.
- **firecrown >=1.16** locally; inner scripts that may run on a facility
  restrict themselves to APIs present since **1.14** (`TwoPointExperiment`,
  `CCLFactory`, `ParamsMap`, `get_default_params_map`) because the NERSC
  `desc-python` stack is at 1.15.2.

## The container image (same environment, MPICH flavour)

`tools/env/environment-image.yml` is `environment.yml` plus `mpich` + `mpi4py`
pinned to MPICH; `tools/env/conda-lock-image.yml` is its linux-64 solve
(`bash scripts/image.sh --relock`, micromamba as the conda-lock solver, ~30 s)
and the `Dockerfile` builds `ghcr.io/nesar/desc-mcp:<sha12>-cpu` from it
(ubuntu 22.04, micromamba 2.9, `/opt/env`, server pip-installed on top;
`tools/env/image.yml` records ref/tag and travels in the dispatch pack as
`env/image.yml`). Why MPICH: Perlmutter (`podman-hpc --mpi`) and Polaris
(`cray-mpich-abi` + bind mounts) inject the host Cray MPICH into a running
container, which only works against an MPICH-ABI `libmpi.so.12`; the laptop
lock's Open MPI (via cosmosis/tjpcov) cannot be replaced and drags a
libfabric without Slingshot support into the solve.

Two build-time facts, both verified on 2026-10-16:

- **conda-forge binaries use DT_RPATH, not RUNPATH** (`readelf -d` on
  `mpi4py/MPI.so`: `RPATH [$ORIGIN/../../..]`), and the loader searches RPATH
  *before* `LD_LIBRARY_PATH`, so an injected Cray MPICH would lose to the
  bundled one. The Dockerfile therefore moves the bundled `libmpi*`,
  `libmpich*`, `libmpicxx*`, `libmpifort*` into `/opt/env/lib/mpich-bundled/`
  and registers that directory with `ldconfig` (searched *after*
  `LD_LIBRARY_PATH`). `docker run` on a laptop still gets MPICH 4.3.2; on a
  facility whatever the host puts on `LD_LIBRARY_PATH` wins (checked with a
  decoy `libmpi.so.12`: `ImportError: file too short`, i.e. the decoy was
  picked).
- **micromamba only recognises a conda-lock file named `conda-lock.yml`**;
  any other name (e.g. `conda-lock-image.yml`) silently gives
  `Empty environment created`. The Dockerfile copies the image lock to
  `/opt/env-lock/conda-lock.yml` before `micromamba create`.

Inside the image (`docker run --rm <image> python tests/smoke_env.py`):
9/10 checks pass; the one failure is "TXPipe clone not found", expected
because no upstream clones are shipped. Image size: ~0.9 GB (`docker save`
tar 0.86 GB). Not in the image: isitgr, TXPipe, the clones, example data.

## Why TXPipe is not in this environment

TXPipe pins `firecrown=1.7.*` (its theory and blinding stages import
`firecrown.likelihood.gauss_family`, removed in firecrown 1.10) and needs a
compiled stack (NaMaster, TreeCorr, MPI-enabled h5py + mpi4py, healpy,
RAIL, dask-mpi). augur needs firecrown >= 1.14. No single environment can
hold both, and TXPipe's README does not support Apple-silicon installs.

So the `txpipe_*` tools never import TXPipe. They read its source for stage
metadata (AST), compose ceci `pipeline.yml`/`config.yml`, and run the real
pipeline elsewhere:

- **Locally (optional)**: build TXPipe's own env in a separate prefix and
  point `DESC_TXPIPE_ENV` at a shell snippet that activates it and enters the
  checkout, e.g. `DESC_TXPIPE_ENV="source ~/envs/txpipe/bin/activate; cd ~/TXPipe"`.
  Recipe: `conda env create -p ~/envs/txpipe -f TXPipe/bin/environment-local.yml`
  (Linux/x86 is the supported target; see TXPipe's `bin/install.sh`).
- **On a facility**: the user's TXPipe environment is named per call in
  `env_setup` (see below). The example data (347 MB) is never shipped with
  the server; `txpipe_fetch_example_data` downloads it on demand.

## Server-side settings (the complete list)

| Variable | Meaning | Default |
|---|---|---|
| `DESC_CCL_DIR`, `DESC_CCLX_DIR`, `DESC_TXPIPE_DIR`, `DESC_FIRECROWN_DIR`, `DESC_AUGUR_DIR`, `DESC_EXTERNAL_DIR` | read-only reference clones (example data, stage metadata, n(z) tables) | sibling directories of this repo |
| `DESC_TXPIPE_ENV` | shell snippet activating a LOCAL TXPipe env (optional) | unset → local TXPipe runs refused with instructions |
| `MCP_PUBLIC`, `MCP_OUTPUT_ROOT`, `MCP_ARTIFACT_URL` | hosting (same as the other HEP-KE servers) | unset |

Nothing facility-related is configured on the server: no project, user,
workdir, token, Globus endpoint or environment path. Those belong to the
client's hep-genesis harness.

## Facility environments (optional override; the default is the pack's own lock)

By default no facility environment is involved: the lock above is built on
the node and every inner script runs in it. Heavy tools still accept an
`env_setup` argument — a shell snippet run on the node before the inner
script — to use a facility-resident environment instead (TXPipe pipelines
need one). Public candidates the server documents (`export_dispatch_pack` →
`env_kernel.env_setup_candidates`; `desc-python` was broken on 2026-10-09):

| Facility | Candidate | Contains |
|---|---|---|
| NERSC Perlmutter | `source /global/common/software/lsst/common/miniconda/setup_current_python.sh` (DESC `desc-python`) | firecrown 1.15.2, pyccl 3.3.6, sacc 2.4, tjpcov 0.5.1, numcosmo 0.27 (lock 2026-09-01) |
| NERSC Perlmutter | `source $CFS/lsst/groups/MCP/setup-cosmology.sh` (DESC `desc-cosmology`) | firecrown, augur, pyccl, sacc, tjpcov, cosmosis, namaster, mpi4py |
| NERSC Perlmutter | user's TXPipe env: `module load python; conda activate <env>; cd <TXPipe checkout>` | TXPipe stack from `TXPipe/bin/perlmutter-install.sh` |
| ALCF Polaris | user-built: `source <prefix>/bin/activate <env>` | no DESC-maintained stack exists on ALCF; build once from `external/desc-cosmology-env/slac/env-nobuild-*.yml` (Miniforge in your own area; see that repo's README) |

Pure-pyccl kernels (`ccl_*`) need none of this: they ship as pip-kernels and
the engine installs `pyccl` + `camb` wheels into a persistent venv on the node.

## Gotchas met while building

- `pip install` of augur from the clone writes `augur.egg-info` into the
  source tree; the setup script builds from a temporary copy so the clone
  stays pristine (the instructions forbid touching the clones).
- Importing `firecrown` or `augur` with the cwd inside this repo looks like
  it works even when they are not installed (the clone directories act as
  namespace packages) — always validate from the repo root with the
  installed packages, as the smoke test does.
- firecrown's `ParamsMap.update()` raises on existing keys; overrides use
  item assignment. Values must be `float` (ints and numpy scalars raise).
- CCL is h-free (Mpc, 1/Mpc, Mpc^3) and takes scale factor `a`; the tool
  layer converts to z and h/Mpc. `P_nl(k=1 h/Mpc, z=0) ≈ 400 (Mpc/h)^3`
  for the vanilla cosmology is the sanity number used in the smoke test.
- The hep-genesis engine runs the facility's module python when a kernel
  declares no pip deps; the env-kernel (`tools/envkernel.py`) relies on that
  to launch inner scripts under `env_setup`.
- **NaMaster 3 vs TJPCov 0.5.1**: TJPCov's `FourierGaussianNmt` (and the
  mask-based SSC/cNG types) calls `NmtCovarianceWorkspace()` without
  fields, the NaMaster 2 API; pymaster 3.0 (what conda-forge resolves today)
  raises `TypeError`. `tjpcov_compute_covariance` reports this plainly; the
  f_sky types work. Run NaMaster-coupled covariances under a NaMaster 2.x
  environment on a facility (`env_setup`) until TJPCov catches up.
- **NaMaster 3 requires fields and bins to share lmax**: `namaster_compute_cls`
  builds the fields at the binning's lmax (= last bandpower edge - 1), and
  the inner TJPCov script sets `NaMaster: {f: {lmax: ...}}` for TJPCov's own
  mask fields.
- **Dense windows vs TJPCov's f_sky binning**: TJPCov reads bandpower edges
  from the sacc windows as "where the weight is nonzero" - meaningless for
  mode-coupled NaMaster windows. `namaster_compute_cls` stores the edges in
  the sacc metadata (`binning/ell_edges`) and `tjpcov_compute_covariance`
  hands TJPCov a copy with top-hat windows on those edges (the covariance is
  attached to the original file, dense windows kept). The Knox tool
  (`sacc_attach_gaussian_covariance`) now takes the number of modes from each
  point's own window column, which is exact for either kind.
- **TJPCov real space**: the `CovarianceCalculator` path assumes xi_+/xi_-
  interleaved per theta inside a tracer pair and never fills the xi_- auto
  block (an index slip in its `auto` shortcut) - singular matrices for
  TXPipe/firecrown-ordered files. The inner script assembles
  `RealGaussianFsky` block by block through TJPCov's public per-block method
  instead. Small-angle xi_- needs `lmax >= 3000` (the projection truncates
  the Fourier covariance); lower lmax gives a matrix spanning 20 decades that
  rounds to "not positive definite".
- `pip install` of Smokescreen from the clone would write build metadata into
  the clone; the setup script builds from a temporary copy (as for augur).
  Smokescreen's own CLI deletes the original sacc after encrypting it; the
  server's tools never delete anything.
