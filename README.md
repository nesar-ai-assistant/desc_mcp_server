# desc-mcp-server

MCP server exposing the public **LSST DESC** cosmology stack as agent tools:
[CCL](https://github.com/LSSTDESC/CCL) theory predictions,
[sacc](https://github.com/LSSTDESC/sacc) data vectors,
[firecrown](https://github.com/LSSTDESC/firecrown) likelihoods,
[augur](https://github.com/LSSTDESC/augur) Fisher forecasts,
[TXPipe](https://github.com/LSSTDESC/TXPipe) measurement pipelines,
[TJPCov](https://github.com/LSSTDESC/TJPCov) covariances,
[NaMaster](https://github.com/LSSTDESC/NaMaster) pseudo-C_ell measurements and
[Smokescreen](https://github.com/LSSTDESC/Smokescreen) data-vector concealment — 54
tools in ten families, with server-side skills for the multi-tool pipelines
and optional execution of the heavy steps on ALCF Polaris / NERSC Perlmutter
through the [hep-genesis](https://github.com/HEP-KE/hep-genesis-agent)
dispatch engine.

The upstream packages are used **unmodified** (they are read-only reference
clones next to this repo); every line of glue lives here, so the DESC codes
can keep evolving underneath.

Companion documents:
- [`ENVIRONMENT.md`](ENVIRONMENT.md) — how the single `desc-mcp` Python
  environment is assembled, the version pins and why, why TXPipe lives
  elsewhere, and the facility environments a user can choose
- [`notes/DESIGN.md`](notes/DESIGN.md) — the design decisions, with the
  package surveys that back them in `notes/survey_*.md`

## Quick start

```bash
bash scripts/env.sh                       # micromamba + .mcp-env/ from tools/env/conda-lock.yml (~5 min), then 10 env checks
.mcp-env/bin/python -m pytest tests/ -q       # in-process tool tests
.mcp-env/bin/python tests/smoke_server.py     # every family through a live MCP session
.mcp-env/bin/python -m mcp_server             # stdio transport
.mcp-env/bin/python -m mcp_server --transport streamable-http --port 8000   # HTTP
```

No conda needed: the environment is the exact set of versions in
`tools/env/conda-lock.yml` (source: `tools/env/environment.yml`; change it and
run `bash scripts/env.sh --relock`). The same lock ships inside the dispatch
pack and is built on the compute node, so what runs here runs there.

Register with Claude Code / Claude Desktop (stdio, local):

```bash
claude mcp add desc -- $(pwd)/.mcp-env/bin/python -m mcp_server
```

(Claude Desktop: add the same command and args to `claude_desktop_config.json`,
with `"cwd"` set to this directory.) Or point any MCP client — including the
hep-genesis-agent desktop app's server list — at the HTTP endpoint
(`http://host:8000/mcp`).

## Tool families

| Family | Tools | Backed by |
|---|---|---|
| meta | `list_desc_packages`, `describe_desc_tool_family`, `list_desc_skills`, `load_desc_skill`, `convert_cosmology_names` | registry + skills + name maps |
| ccl | `ccl_describe_cosmology`, `ccl_background`, `ccl_matter_pk`, `ccl_lsst_srd_nz`, `ccl_angular_cls`, `ccl_correlation_functions`, `ccl_correlation_3d`, `ccl_halo_mass_function`, `ccl_halo_model_pk`, `ccl_baryon_boost` | pyccl 3.3.6 (+ CAMB, CosmicEmu, bacco, HMcode) |
| sacc | `sacc_inspect`, `sacc_to_csv`, `sacc_prepare_for_firecrown`, `sacc_attach_gaussian_covariance` | sacc 2.4 (+ pyccl for the Gaussian covariance) |
| firecrown | `firecrown_list_examples`, `firecrown_build_likelihood`, `firecrown_compute_loglike`, `firecrown_theory_data_vector`, `firecrown_scan_loglike`, `firecrown_run_chain`, `firecrown_chain_status`, `firecrown_plot_chain`, `firecrown_chain_cancel`, `firecrown_collect_chain` | firecrown 1.16 factory API, Cobaya (pure-CCL mode; background runs, facility no-wait submit + collect, resume, auto-PPF for w0-wa), getdist corner plots |
| augur | `augur_list_examples`, `augur_generate_forecast_config`, `augur_validate_forecast_config`, `augur_generate_synthetic_datavector`, `augur_compute_fisher`, `augur_plot_fisher_contours` | augur 1.2.4 (+ TJPCov, numdifftools / derivkit) |
| txpipe | `txpipe_list_stages`, `txpipe_describe_stage`, `txpipe_list_examples`, `txpipe_generate_pipeline`, `txpipe_validate_pipeline`, `txpipe_run_pipeline`, `txpipe_run_status`, `txpipe_fetch_example_data` | TXPipe source (read as metadata) + ceci; runs in a separate TXPipe env or on a facility |
| tjpcov | `tjpcov_list_covariance_types`, `tjpcov_generate_config`, `tjpcov_compute_covariance`, `tjpcov_compare_covariances` | tjpcov 0.5.1: Gaussian f_sky (harmonic and real space), halo-model SSC and cNG terms; NaMaster-coupled types need NaMaster 2.x (facility env) |
| namaster | `namaster_mask_properties`, `namaster_simulate_maps`, `namaster_compute_cls` | pymaster 3.0 + healpy: masks, simulated Gaussian maps from CCL theory, pseudo-C_ell bandpowers -> sacc with windows and coupled noise |
| smokescreen | `smokescreen_conceal_datavector`, `smokescreen_inspect`, `smokescreen_encrypt_file`, `smokescreen_decrypt_file` | smokescreen 1.5.6 over the firecrown family's experiment YAML; never deletes originals, never records the hidden shift |
| dispatch | `set_dispatch`, `get_dispatch`, `auth_status`, `export_dispatch_pack` | hep-genesis engine (IRI + Globus) |

Conventions every tool follows: redshift `z` at the boundary (never scale
factor); `k` in h/Mpc and P(k) in (Mpc/h)^3 unless a tool's `k_units` says
otherwise; `theta` in arcmin; parameter names are firecrown's (`Omega_c`,
`Omega_b`, `h`, `n_s`, `sigma8` or `A_s`, `w0`, `wa`, `m_nu`; nuisance
`src0_delta_z`, `lens0_bias`, `ia_bias`, ...). One `cosmology` object (with
presets `vanilla`, `planck18`, `desc_srd`) is accepted by every family. Files,
not arrays, flow between tools; every file-writing tool takes `output_dir`
and returns `{status, files, message, metadata}`.

## Skills (server-side, client-agnostic)

The server carries its own skills — named multi-tool recipes in
[`skills/`](skills/) (markdown with a small frontmatter header), served both
through `list_desc_skills` / `load_desc_skill` and as native MCP prompts.

| Skill | What it does |
|---|---|
| `desc-tour` | one representative call per family; the five-minute demo path |
| `ccl-datavector-to-sacc` | SRD n(z) → angular C_ell (Limber / non-Limber) → sacc file ready for firecrown |
| `pk-ccl-vs-emulators` | CCL nonlinear P(k) across prescriptions vs the companion emulator server, one k grid |
| `likelihood-sanity-checks` | inspect → build → loglike at fiducial → 1-D scan, with the chi2/dof and parameter-completeness checks that must pass before any chain |
| `txpipe-sacc-to-likelihood` | take a TXPipe sacc (tracers `source_i/lens_i`, maybe no covariance) to a firecrown likelihood |
| `txpipe-measurement` | compose, validate and run a TXPipe 3x2pt pipeline (local env or facility) |
| `lsst-3x2pt-forecast` | augur Y1/Y10 Fisher forecast end to end, plausible numbers, pitfalls |
| `maps-to-likelihood` | masked maps (or a simulated test bed) -> NaMaster bandpowers with windows -> TJPCov covariance -> firecrown chi2, with the binning/noise cross-checks |
| `conceal-datavector` | Smokescreen concealment protocol: validate the likelihood, choose ranges and a seed, conceal, record, encrypt the original, analyse the concealed file |
| `hpc-dispatch-handoff` | run the heavy kernels on ALCF/NERSC through the client's hep-genesis facility tools |

## Example queries

Try these from the hep-genesis-agent desktop app or Claude Desktop with the
server registered. Each maps to one family or one skill; the last ones need a
facility sign-in on the client.

1. *"Using CCL, compute the comoving distance, angular diameter distance and growth factor from z=0 to 3 for Planck 2018 and plot them."*
2. *"Compare the nonlinear matter power spectrum at z=0.5 from CCL halofit, CCL HMcode-2020 and CosmicEmu on the same k grid in h/Mpc, and report the maximum fractional difference below k=1 h/Mpc."*
3. *"Generate the LSST Y1 SRD source and lens n(z) bins, compute all 3x2pt angular power spectra for ell 20–2000 with non-Limber clustering, and write them to a sacc file."*
4. *"Inspect this sacc file: tracers, data types, covariance. Then attach a Gaussian covariance with f_sky=0.4, sigma_e=0.26, n_gal=10 per arcmin² and compute the firecrown log-likelihood at the vanilla cosmology."*
5. *"Scan the log-likelihood in sigma8 between 0.7 and 0.9 for the DES Y1 3x2pt test data vector with fiducial nuisance parameters and report the 1-sigma interval."*
6. *"Run an LSST Y1 3x2pt Fisher forecast with augur using the SRD covariance, varying Omega_c, sigma8, w0, wa, n_s, h and the lens biases; give the marginalised 1-sigma errors and the w0–wa figure of merit, and plot the contours."*
7. *"Repeat the forecast for Y10 and tell me by how much the w0–wa FoM improves."*
8. *"Describe the TXTwoPoint stage: its inputs, outputs and configuration options with defaults."*
9. *"Build a TXPipe pipeline for the metadetect example data that produces real-space 3x2pt measurements, show me the stage list and the dry-run commands."*
10. *"Export this server's dispatch pack and run the Y10 Fisher forecast on Perlmutter with the desc-python environment; confirm the host it ran on."* (hosted server: the client's hep-genesis facility tools run the pack) — or, with the server on your own machine and the hep-genesis backend installed: *"Set dispatch to Perlmutter and run the Y10 Fisher forecast there with env_setup for desc-python."*
11. *"Walk me through the desc-tour skill."*
12. *"Simulate shear and clustering maps at nside 128 with f_sky 0.3, measure the 3x2pt bandpowers with NaMaster in linear bins of width 20, and show me the measured spectra against the input theory."* (`namaster_simulate_maps` -> `namaster_compute_cls`)
13. *"Give that measured sacc a TJPCov covariance (Gaussian plus super-sample, f_sky from the mask), compare it with the plain Knox covariance, and report the chi2 at the input cosmology."* (`maps-to-likelihood` skill)
14. *"Here is my HEALPix shear catalog map and mask: what is the effective f_sky with a 1-degree apodization, and what are the EE and BB bandpowers between ell 30 and 1500?"* (`namaster_mask_properties`, `namaster_compute_cls`)
15. *"Compute a real-space Gaussian covariance with TJPCov for this xi_+/xi_-/gamma_t/w(theta) sacc (f_sky 0.1, lmax 3000) and check it is positive definite."* (`tjpcov_generate_config` with RealGaussianFsky)
16. *"Conceal the DES Y1 3x2pt data vector with Smokescreen - hide Omega_c in [0.2, 0.32] and sigma8 in [0.72, 0.9] with a seed I give you, encrypt the original, and set up the likelihood on the concealed file."* (`conceal-datavector` skill)

### Heavy jobs for NERSC Perlmutter / ALCF Polaris

These run for tens of minutes to hours on a compute node and need a facility
sign-in on the client plus a facility environment for the DESC stack
(`env_setup`; on NERSC the `desc-python` or `desc-cosmology` stacks, on ALCF an
environment you built from `desc-cosmology-env` — see `ENVIRONMENT.md`).
Every chain, forecast and pipeline below goes through `export_dispatch_pack`
(hosted server) or `set_dispatch` (server on your own machine); the tools
report the host they ran on. Heavy tools take an explicit walltime
(`walltime_s` on chains, `duration_s` on forecasts, `walltime_s` on TXPipe
runs); the hep-genesis engine routes jobs over the debug cap (Perlmutter
30 min, Polaris 60 min) to the long queue by itself and rejects durations no
queue admits before staging. A chain that stops at its walltime is continued
with `resume=True` and the same arguments. `firecrown_run_chain(nodes=N)`
runs 4 N Cobaya chains in parallel (one MPI process per chain, the kernel's
`DISPATCH_MPI` shape; the engine builds mpi4py into the lock environment)
and reports the cross-chain R-1; the charge is N x walltime, the facility's
node cap (`NERSC_MAX_NODES` / `ALCF_MAX_NODES` in the client's settings)
applies, and a resume needs the same N. While a heavy call waits for a
facility job it sends MCP progress heartbeats (every `MCP_HEARTBEAT_S`,
default 30 s), so clients that reset their request timeout on progress keep
the call alive; a long LOCAL chain should use `background=True` and be
polled with `firecrown_chain_status`, then summarised with
`firecrown_plot_chain`.

12. **MCMC** — *"Build the firecrown likelihood for the DES Y1 3x2pt test data vector, check chi2/dof at the fiducial point, then run a Cobaya chain on Perlmutter with the desc-python environment sampling Omega_c, sigma8, w0, wa, h, n_s and the five lens biases with uniform priors, max_samples 20000 and R-1 < 0.02. Report whether it converged, the means and 68% intervals, and the trace plot."* (`firecrown_run_chain`; a `max_samples` of 200 is the local smoke run)
13. **Same chain on ALCF** — *"Export the dispatch pack and run the same chain on Polaris under my environment at `/eagle/<project>/<user>/envs/desc-cosmology`; compare the posterior means and R-1 with the Perlmutter run and confirm both hosts."*
14. **Chain on a TXPipe measurement** — *"Take the `twopoint_data_real.sacc` from the TXPipe run, prepare it for firecrown, attach a Gaussian covariance with f_sky=0.05, sigma_e=0.26, n_gal=10 per arcmin², verify chi2/dof at Planck 2018, then run a 10000-sample chain in Omega_c, sigma8 and w0 on Perlmutter."* (`txpipe-sacc-to-likelihood` skill, then `firecrown_run_chain`)
15. **Big Fisher forecast** — *"Generate an LSST Y10 3x2pt forecast config with the TJPCov covariance and the CAMB transfer function, varying Omega_c, Omega_b, sigma8, h, n_s, w0, wa, m_nu, all ten lens biases, the IA amplitude and the five source delta_z shifts with the SRD photo-z priors; validate it, run augur_compute_fisher on Perlmutter with a 4-hour walltime, and report the marginalised errors and the w0–wa FoM with and without the priors."* (41 full theory-vector evaluations with CAMB)
16. **Derivative-step study** — *"Repeat the Y10 Fisher on Polaris with derivative steps of 0.5%, 1%, 2% and 5% and tell me how stable the w0–wa FoM is; flag any non-positive-definite result."*
17. **Full TXPipe pipeline** — *"Compose a TXPipe 3x2pt_fourier pipeline for the 20 deg² DC2 metadetect catalog at `$CFS/<project>/<user>/txpipe/data`, nside 2048, five source and five lens bins; validate it, run it on Perlmutter with 128 threads and a 6-hour walltime under my TXPipe environment, then inspect the resulting sacc and give me the per-stage status."* (`txpipe_run_pipeline` keeps the HDF5 catalogs and maps on the facility and returns the sacc, PNGs and log tails)
18. **Likelihood profiles in batch** — *"For w0 = -1.2, -1.0 and -0.8, scan the DES Y1 3x2pt log-likelihood in sigma8 between 0.65 and 0.95 with 200 points each on Perlmutter, and report how the 1-sigma interval in sigma8 shifts with w0."* (three dispatched `firecrown_scan_loglike` jobs)
19. **Large CCL grid (pip-kernel, no DESC environment needed)** — *"On Polaris, compute the non-Limber 3x2pt angular power spectra for the LSST Y10 SRD bins (5 source + 10 lens, ell 2–5000) with the CAMB transfer function and HMcode-2020 baryons on a 5×5 grid of (w0, wa), and report the largest Limber-vs-non-Limber fractional difference per bin pair at ell < 100."*
20. **Cross-facility check** — *"Run the Y1 Fisher forecast with the SRD covariance on both Perlmutter and Polaris, confirm the hosts, and show that the marginalised errors agree to better than 1%."*

## Design principles

- **Packages untouched**: nothing under `CCL/`, `CCLX/`, `TXPipe/`,
  `firecrown/`, `augur/` or `external/` (TJPCov, NaMaster, Smokescreen, sacc,
  ceci, ...) is modified; the server reads them (example data, n(z) tables,
  stage metadata) and installs them from copies. Upstream quirks are worked
  around in the server (e.g. TJPCov's real-space block ordering, NaMaster
  windows vs TJPCov's f_sky binning), never patched in the clones.
- **One cosmology object, firecrown's names**: the same `cosmology` JSON
  feeds CCL, firecrown and augur; `convert_cosmology_names` translates to
  Cobaya, CosmoSIS and the emulator server.
- **sacc is the contract between families**: TXPipe writes it, CCL and augur
  generate it, firecrown reads it. `sacc_prepare_for_firecrown` bridges the
  one naming mismatch (TXPipe `source_i/lens_i` vs firecrown `src{i}/lens{i}`).
- **Compose, validate, then run**: every heavy step is preceded by
  millisecond-scale validation (YAML, parameter completeness, DAG checks) so a
  bad request never costs queue time.
- **Kernel/wrapper split**: heavy computations live in kernels that are
  JSON-safe and runnable on a facility node; wrappers handle files, labels,
  plots and metadata wherever the server runs.
- **Files, not arrays**: CSV / sacc / PNG artifacts with provenance headers;
  only paths and quotable summaries pass through the LLM context.

## Hosting

Same recipe as the other HEP-KE servers (a Linux box behind a reverse proxy):

1. Clone, run `bash scripts/env.sh` (micromamba + `.mcp-env/` from the lock; no
   conda on the box), keep the reference clones next to the repo (or point
   `DESC_*_DIR` at them).
2. Run `.mcp-env/bin/python -m mcp_server --transport streamable-http` under
   systemd bound to localhost — `deploy/desc-mcp.service.example` is a
   complete unit — with `MCP_PUBLIC=1`, `MCP_OUTPUT_ROOT=/srv/artifacts`,
   `MCP_ARTIFACT_URL=https://files.example.org`.
3. TLS reverse proxy in front: one route to the server port, one static file
   server on `MCP_OUTPUT_ROOT`.
4. After every deploy: `python tests/smoke_server.py <url>`.

The hosted server needs **no** facility sign-in, project, or environment
settings and ships no example data; see the next section.

**Container alternative (same environment, one command).** The image
`ghcr.io/nesar/desc-mcp` carries the MPICH variant of the lock
(`tools/env/conda-lock-image.yml`) plus the server; its tag is the lock's hash
(`<sha12>-u24-cpu`: lock hash + base OS ubuntu 24.04, also `latest`). It is the only way to run this server on
Windows (via Docker Desktop/WSL2) and the same image runs the dispatched
kernels on Perlmutter (podman-hpc) and Polaris (apptainer) when the client's
hep-genesis engine has the container carrier enabled.

```bash
docker run -d --name desc-mcp -p 127.0.0.1:8008:8008 \
  -e MCP_PUBLIC=1 -e MCP_ARTIFACT_URL=https://files.example.org/desc \
  -v /srv/artifacts/desc:/srv/artifacts ghcr.io/nesar/desc-mcp:latest
python tests/smoke_server.py http://127.0.0.1:8008/mcp
```

The image ships no upstream clones, so the tools that read example data or
stage metadata from them (`txpipe_*`, `augur_generate_forecast_config`,
`firecrown_list_examples`) answer "clone not available" until you mount one:
`-v /srv/clones/TXPipe:/clones/TXPipe -e DESC_TXPIPE_DIR=/clones/TXPipe`
(same for `DESC_AUGUR_DIR`, `DESC_FIRECROWN_DIR`, ...). Everything else
(`tests/smoke_server.py`: 34/38 calls, the 4 failures are exactly those) works
without them.

Build/test/push it yourself with `bash scripts/image.sh [--relock] [--test] [--push]`
(`--save DIR` writes a `docker save` tar for registry-less facilities:
`apptainer build x.sif docker-archive://x.tar`). The image has no
ENTRYPOINT on purpose (podman-hpc's `--mpi` module and the engine's
`<image> python runner.py` both need a bare command); `docker run <image>`
starts the server through CMD.

## Run on HPC

All facility access goes through the hep-genesis dispatch engine (IRI for
job submission, Globus Transfer or the IRI filesystem API for staging and
results). This server writes no batch scripts, opens no SSH sessions, and
holds no facility identity: project, workdir, tokens, Globus endpoint and
the facility-side software environment are the **user's**, supplied by the
client per call.

**Client-side dispatch (hosted deployments — recommended).** `export_dispatch_pack`
returns the `tools/` kernels plus a manifest. The client's hep-genesis
facility server runs them with `run_pack_kernel` under the user's
credentials. Two kernel kinds:

| Kind | Used by | Node needs |
|---|---|---|
| lock-kernel (default) | every `inner.*` script: firecrown loglike/scan/chain, augur forecast; `ccl_*` grids | nothing: the pack carries `env/conda-lock.yml` and the engine builds that environment on the node with micromamba (first job per lock: minutes; reused afterwards) |
| env-kernel (override) | the same scripts when the user passes `env_setup`; TXPipe pipelines always | a facility-resident environment named by the user in `env_setup` |

The manifest carries `env_lock` (the path of the lock inside the pack) and
still lists public `env_setup` candidates for the override — on NERSC the
DESC `desc-cosmology` stack (`source $CFS/lsst/groups/MCP/setup-cosmology.sh`)
or `desc-python`; on ALCF a user-built environment from `desc-cosmology-env`.
The `hpc-dispatch-handoff` skill walks an agent through it.

**Server-side dispatch (server on your own machine).** With
`pip install -e <hep-genesis-agent>/backend[iri]` in the env and a facility
sign-in, `set_dispatch("perlmutter")` makes the heavy tools dispatch
themselves; env-kernel tools take `env_setup` as an argument. Do not use this
mode on shared deployments.
