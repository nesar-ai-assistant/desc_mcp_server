"""Optional HPC dispatch: run this server's compute kernels on DOE facilities.

Local execution is the default and is untouched by this module. Two modes
(identical to the cosmic-emulator / spectra / gaia servers):

CLIENT-SIDE (hosted deployments - recommended): this server holds NO
credentials, project, workdir or facility environment. export_dispatch_pack
hands the tools/ kernels to the client, and the client's hep-genesis harness
- which owns the facility tokens, Globus endpoint, project and workdir -
stages and submits them itself over IRI + Globus (facility-server
run_pack_kernel tool / sidecar POST /jobs pack=). A VM deployment needs
nothing beyond this server.

SERVER-SIDE (running the server on your own machine): set_dispatch("polaris"
| "perlmutter") routes the heavy tools through the hep-genesis dispatch
engine in THIS process: the tools/ package is staged to the facility over
Globus (or the IRI filesystem API on NERSC), the kernel runs on a compute
node, and results come back. Facility settings (project, workdir, tokens)
come from the USER's hep-genesis .env and sign-in - never from this server.
Do not use server-side mode on shared/hosted deployments.

The pack carries its own environment (hep-genesis dispatch contract R8):
``tools/env/conda-lock.yml`` is the exact environment this server runs in,
and the engine builds it on the compute node with micromamba (once per lock,
then reused), so every kernel - pyccl grids, firecrown chains, augur
forecasts - runs under the same package versions as the server. Nothing
needs to pre-exist on the facility. Two ways a kernel can still run under a
facility-resident environment instead: the client passes `env_setup` to a
tool (the env-kernel path, tools/envkernel.py + tools/inner/*), or
run_pack_kernel(env_lock="none"). TXPipe tools are env_setup-only (TXPipe
pins an older firecrown and cannot share the lock).

No SLURM/PBS scripts and no SSH anywhere: the engine generates the job
wrapper and submits it through the IRI compute API.

Tool names (set_dispatch, get_dispatch, auth_status, export_dispatch_pack)
are a cross-server convention matched by name in the hep-genesis harness -
do not rename them.
"""

import importlib
import os
from contextvars import ContextVar
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"
# The environment the pack carries (relative to tools/); the engine ships and
# builds it by default (hep_genesis.iri.dispatch.envsetup.ENV_LOCK_RELPATH).
ENV_LOCK = "env/conda-lock.yml"
# env/image.yml names the container image built from the MPICH variant of the same
# environment (scripts/image.sh); an engine with a container carrier may run the
# kernels in it instead of building the lock env on the node. The image lock itself
# does not travel: the image is the artefact, the tag is its hash.
ENV_IMAGE = "env/image.yml"
ENV_FILES = ("env/conda-lock.yml", "env/environment.yml", ENV_IMAGE)

_state = {"site": "local"}

# The output_dir of the tool call being served (set by the server's tool
# wrapper; asyncio.to_thread carries it into the worker). A dispatched job's
# files and its dispatch_<job>.json record go to <output_dir>/<job>/ - the
# CALL's own run folder. The process-wide DISPATCH_ARTIFACT_DIR is shared by
# every conversation on this server: one chat's set_dispatch(artifact_dir=)
# pulled other chats' jobs into its run folder (2026-10-14).
CALL_OUTPUT_DIR: ContextVar[str | None] = ContextVar("desc_call_output_dir", default=None)


def _job_placement(function: str, args: dict) -> dict:
    """job_name + artifact_dir for this call's job ({} = engine defaults)."""
    out = CALL_OUTPUT_DIR.get()
    if not out:
        return {}
    try:
        from hep_genesis.iri.dispatch.staging import default_job_name
    except ImportError:  # older engine: keep its defaults
        return {}
    named = f"inner.{args['inner']}.main" if function == "envkernel.run_in_env" and args.get("inner") else function
    job_name = default_job_name(named)
    return {"job_name": job_name, "artifact_dir": str(Path(out).expanduser() / job_name)}

__all__ = ["set_dispatch", "get_dispatch", "auth_status", "export_dispatch_pack"]

# Mirror the hep-genesis engine's pack limits so an exported pack is always
# stageable by the client that receives it.
_PACK_MAX_FILES = 200
_PACK_MAX_BYTES = 10 * 1024 * 1024


def _collect_kernel_manifest() -> dict:
    """Gather DISPATCH_KERNELS from every tool family that declares one."""
    from .server import configured_tool_module_names

    kernels: dict = {}
    for name in configured_tool_module_names():
        if name == __name__:
            continue
        try:
            module = importlib.import_module(name)
        except Exception:  # noqa: BLE001 - manifest stays useful without a family
            continue
        kernels.update(getattr(module, "DISPATCH_KERNELS", {}) or {})
    return kernels


def _image_manifest(text: str | None) -> dict | None:
    """The pack's container image, from tools/env/image.yml: ``{ref, tag, digest,
    platforms, mpi, perlmutter, polaris}``; ``None`` when the file is absent or the
    tag is empty. ``digest`` is empty until the image is pushed."""
    if not text:
        return None
    import yaml

    meta = yaml.safe_load(text) or {}
    if not meta.get("ref") or not meta.get("tag"):
        return None
    return {k: meta.get(k) for k in ("ref", "tag", "digest", "platforms", "mpi", "perlmutter", "polaris")}


def export_dispatch_pack() -> dict:
    """Hand THIS DESC server's compute kernels to the client for HPC runs.

    Returns the ``tools/`` package as in-band text files plus a manifest:
    kernel entry points, per-kernel node-side pip requirements (pip-kernels)
    or the ``env_setup`` requirement with public environment candidates
    (env-kernels, e.g. the NERSC desc-python stack), walltime hints. The
    CLIENT - which holds the facility tokens, Globus endpoint, project and
    workdir - stages and submits the kernels with ITS OWN credentials (e.g.
    the hep-genesis facility servers' run_pack_kernel tool). Nothing
    credential-shaped or project-shaped is needed on, or returned by, this
    server.

    Clients that recognise the ``dispatch_pack`` key should save the files
    locally and NOT echo their contents into the conversation.
    """
    files: dict[str, str] = {}
    total = 0
    env_paths = [TOOLS_DIR / rel for rel in ENV_FILES if (TOOLS_DIR / rel).is_file()]
    for path in sorted(TOOLS_DIR.rglob("*.py")) + env_paths:
        # the image lock is not shipped (see ENV_IMAGE); rglob never matches it,
        # but keep the pack honest if someone adds it to ENV_FILES
        if path.name == "conda-lock-image.yml":
            continue
        rel = path.relative_to(TOOLS_DIR)
        if any(part.startswith((".", "__pycache__")) for part in rel.parts):
            continue
        text = path.read_text()
        total += len(text.encode())
        if len(files) >= _PACK_MAX_FILES or total > _PACK_MAX_BYTES:
            raise RuntimeError(
                f"tools/ package exceeds pack limits ({_PACK_MAX_FILES} files / "
                f"{_PACK_MAX_BYTES} bytes) - prune before exporting."
            )
        files[f"tools/{rel}"] = text

    from tools.envkernel import ENV_SETUP_CANDIDATES

    return {
        "dispatch_pack": {
            "name": "tools",
            "server": "desc-mcp-server",
            "file_count": len(files),
            "bytes": total,
            "files": files,
            "kernels": _collect_kernel_manifest(),
            # contract R8: the lock IS the node environment; the engine detects
            # it at this path and builds it (micromamba) before the kernel runs
            "env_lock": ENV_LOCK if f"tools/{ENV_LOCK}" in files else None,
            # container carrier (optional for the engine): {ref, tag, digest, mpi, ...}
            "image": _image_manifest(files.get(f"tools/{ENV_IMAGE}")),
            "env_lock_note": "Pass no env_setup and no pip_deps: run_pack_kernel builds the "
                             "environment from this lock on the facility (first job with a new "
                             "lock: minutes; later jobs reuse it). env_setup remains an override "
                             "for a facility-resident environment (TXPipe needs it).",
            "env_kernel": {
                "function": "envkernel.run_in_env",
                "args": {"env_setup": "<shell snippet activating a facility env with the DESC stack>",
                         "inner": "<inner script name, see each kernel's 'inner'>",
                         "params": "<the kernel's params object>",
                         "timeout_s": "<job walltime minus ~60 s; default 3000 - ALWAYS pass it, "
                                      "the inner script is killed at this many seconds>"},
                "base_pip_deps": [],
                "env_setup_candidates": ENV_SETUP_CANDIDATES,
                "walltime": "each kernel's duration_hint_s is a FLOOR for the job's duration=; size the "
                            "real request to the work (chains: ~5 s per sample) within the facility caps "
                            "(Perlmutter debug 30 min / regular 12 h; Polaris debug 60 min) and pass "
                            "timeout_s = duration - 60 in args.",
                "note": "env_setup is chosen by the user/client; this server ships only "
                        "documented public candidates and never a default identity.",
            },
            "usage": (
                "Save these files on the CLIENT machine, then dispatch with the "
                "hep-genesis facility server: run_pack_kernel(pack=<saved dir>/tools, "
                "function=<kernel function>, args={...}, duration=<walltime s>) - no pip_deps, "
                "no env_setup: the pack's env/conda-lock.yml is built on the node. Inner scripts "
                "run directly as function='inner.<name>.main' with args={params: {...}}. Only to use a "
                "facility-resident environment instead: function='envkernel.run_in_env' with "
                "args={env_setup, inner, params, timeout_s}, env_lock='none'. Multi-node: a kernel whose "
                "manifest entry has 'mpi' (inner.firecrown_chain.main) takes nodes=N and runs N x "
                "tasks_per_node MPI processes; other kernels are single-process and refuse nodes > 1."
            ),
        }
    }


def _engine():
    """Import the dispatch engine, with install instructions on failure."""
    try:
        from hep_genesis.iri.alcf.dispatch import run_codes_on_polaris
        from hep_genesis.iri.nersc.dispatch import run_codes_on_perlmutter
    except ImportError as exc:
        raise RuntimeError(
            "HPC dispatch needs the hep-genesis backend in this server's "
            "environment. Install it with: pip install -e "
            "<hep-genesis-agent>/backend[iri]  (import failed: " + str(exc) + ")"
        ) from exc
    return {"polaris": run_codes_on_polaris, "perlmutter": run_codes_on_perlmutter}


def remote_site() -> str | None:
    """The active remote site ('polaris'/'perlmutter'), or None for local.

    Not an MCP tool - compute tools call this to branch at call time.
    """
    site = _state["site"]
    return None if site == "local" else site


def run_kernel(function: str, args: dict, pip_deps: list[str] | None = None,
               duration: int = 600, nodes: int = 1, env_lock: str | bool | None = None,
               tasks_per_node: int | None = None, cpus_per_task: int | None = None,
               gpus_per_task: int | None = None, wait: bool = True,
               image: str | bool | None = None) -> dict:
    """Run a kernel from this server's tools/ package on the active site.

    ``image`` (hep-genesis contract R11, engines >= the container carrier):
    None = the pack's own ``tools/env/image.yml`` when the image has been
    built and pushed (the engine then runs the kernel inside that container
    instead of building the lock env on the node); False = never use a
    container (``run_pack_kernel(image='none')``); a fully qualified
    reference = that image. Older engines without the parameter ignore it.

    ``wait=False`` (hep-genesis contract R10) returns as soon as the
    scheduler accepts the job: the engine's ``{"status": "submitted",
    "job_id", "record", ...}`` handle. :func:`collect_kernel` fetches the
    result later from that record.

    ``nodes`` > 1 is an MPI job (contract R9): the kernel's module must
    declare ``DISPATCH_MPI`` or the engine refuses before staging;
    ``tasks_per_node`` / ``cpus_per_task`` / ``gpus_per_task`` override that
    declaration. The engine dict then carries an ``mpi`` block (size, nodes)
    and ``result`` is rank 0's return.

    Not an MCP tool. ``function`` is a dotted path within tools/ (e.g.
    "ccl.kernels.compute_matter_pk" or "envkernel.run_in_env"); ``args`` must
    be JSON-safe. Returns the engine dict: result, host, and artifact_files
    (local paths of files the kernel wrote in its job directory, fetched
    back automatically where the staging path supports it).

    ``env_lock``: None = the pack's own environment (tools/env/conda-lock.yml,
    built on the node); False = do NOT build/activate it (env-kernels that
    activate a facility environment of their own - otherwise the lock env's
    python stays first on PATH and the override never takes effect, seen on
    Perlmutter 2026-10-10).

    Raises RuntimeError with remediation text on any dispatch failure -
    callers should surface it, and agents must NOT blind-retry.
    """
    site = _state["site"]
    if site == "local":
        raise RuntimeError("run_kernel called with local dispatch - use the kernel directly.")
    run = _engine()[site]
    try:
        # codes=tools/ ships env/conda-lock.yml with the kernels; the engine
        # builds that environment on the node (env_lock auto-detected).
        shape = {k: v for k, v in (("tasks_per_node", tasks_per_node), ("cpus_per_task", cpus_per_task),
                                   ("gpus_per_task", gpus_per_task)) if v}
        # image=None is the engine's own default (auto-detect env/image.yml);
        # only an explicit choice is passed, so engines predating R11 still work.
        result = run(function=function, args=args, codes=str(TOOLS_DIR),
                     pip_deps=pip_deps, duration=duration, nodes=nodes, env_lock=env_lock,
                     **_job_placement(function, args), **shape, **({} if wait else {"wait": False}),
                     **({} if image is None else {"image": image}))
    except Exception as exc:
        body = ""
        resp = getattr(exc, "response", None)
        if resp is not None:
            try:
                body = (resp.text or "")[:600]
            except Exception:
                pass
        from hep_genesis.iri.dispatch.hints import dispatch_auth_hint
        hint = dispatch_auth_hint(site, body or str(exc))
        walltime = any(k in (body + str(exc)).lower()
                       for k in ("qosmaxwalldurationperjoblimit", "walltime", "qos cap", "queue cap"))
        advice = (
            "This is a WALLTIME rejection, not auth: resubmit ONCE with a duration the queue admits "
            "(shorter walltime_s / fewer samples, or a longer queue) - never the same arguments."
            if walltime else
            "Do NOT resubmit the same arguments - the failure is in the dispatch layer and will recur. "
            "Surface this error to the user."
        )
        raise RuntimeError(
            f"Remote dispatch to {site} failed: {exc}\n"
            f"{('Response body: ' + body) if body else ''}{hint}\n{advice}"
        ) from exc
    if not wait and result.get("status") == "submitted":
        return result
    return _check_result(result, site)


def _check_result(result: dict, site: str) -> dict:
    """Raise the agent-facing error for a failed engine/kernel result."""
    if result.get("status") != "success":
        raise RuntimeError(
            f"Remote job on {site} failed: {result.get('error', 'unknown error')}\n"
            f"{('Stderr: ' + result['stderr']) if result.get('stderr') else ''}\n"
            "Do NOT retry this call - surface this error to the user."
        )
    inner = result.get("result")
    if isinstance(inner, str) and inner.startswith("Error"):
        raise RuntimeError(f"Kernel on {site} reported: {inner}\nDo NOT retry blindly - "
                           "fix the inputs or the env_setup and call again.")
    return result


def collect_kernel(record: str, wait_s: int = 0) -> dict:
    """One look at a job submitted with ``run_kernel(wait=False)`` (or one
    whose waiting call timed out), from its ``dispatch_<job>.json`` record
    path. Returns the engine dict like :func:`run_kernel` once the job has
    finished, or ``{"status": "pending" | "unknown", "state", "message"}``
    while it has not; raises RuntimeError when it ended without results.
    ``wait_s`` > 0 keeps looking (every 30 s) up to that long while pending -
    engines without the parameter get one look.

    Not an MCP tool. Works whatever the active site is: the record names it.
    """
    try:
        from hep_genesis.iri.dispatch.collect import collect_dispatched, load_record
    except ImportError as exc:
        raise RuntimeError("Collecting a no-wait job needs a hep-genesis backend with "
                           "hep_genesis.iri.dispatch.collect (update the engine install).") from exc
    rec = load_record(record)
    try:
        result = collect_dispatched(rec, wait_s=max(0, int(wait_s))) if wait_s else collect_dispatched(rec)
    except TypeError:  # older engine: no wait_s
        result = collect_dispatched(rec)
    if isinstance(result, dict) and result.get("status") in ("pending", "unknown"):
        return result
    return _check_result(result, rec["site"])


def run_env_kernel(env_setup: str, inner: str, params: dict, duration: int = 1800,
                   nodes: int = 1) -> dict:
    """Run tools/inner/<inner>.main(params) on the active site under env_setup.

    Not an MCP tool. Convenience over run_kernel for env-kernels; returns the
    engine dict whose ["result"] is {"result": <inner result>, "env_check": ...}.
    """
    # env_lock=False: the client's env_setup names the environment; the
    # pack's lock env must not be built or put on PATH for this job.
    return run_kernel("envkernel.run_in_env",
                      {"env_setup": env_setup, "inner": inner, "params": params,
                       "timeout_s": max(60, duration - 60)},
                      pip_deps=None, duration=duration, nodes=nodes, env_lock=False)


def set_dispatch(site: str, artifact_dir: str | None = None) -> str:
    """Set where THIS DESC server's heavy tools execute: 'local' (this
    machine), 'polaris' (ALCF) or 'perlmutter' (NERSC).

    Affects only this server's heavy tools (Fisher forecasts, likelihood
    scans/chains, TXPipe pipelines, large CCL grids) - never other servers'
    dispatch state. Remote sites run each heavy call as one facility job
    (staging + queue + walltime: minutes, not seconds) and need facility
    sign-in - check with auth_status. Light tools always run locally. Tools
    that need the conda-only DESC stack on the node take an env_setup
    argument naming the facility environment (see export_dispatch_pack's
    candidates). Optional artifact_dir sets where produced files are fetched
    back to. Returns the active configuration.
    """
    if artifact_dir:
        os.environ["DISPATCH_ARTIFACT_DIR"] = str(Path(artifact_dir).expanduser())
    site = (site or "").strip().lower()
    if site in ("local", "off", "none", ""):
        _state["site"] = "local"
        return "Dispatch: local execution."
    if site in ("polaris", "alcf"):
        site = "polaris"
    elif site in ("perlmutter", "nersc"):
        site = "perlmutter"
    else:
        return f"Unknown site {site!r}. Use 'local', 'polaris', or 'perlmutter'."
    # Import the engine NOW: a missing install fails here with instructions,
    # and hep_genesis's .env load (override=True at import) happens before we
    # touch the environment below, so it cannot clobber what we set.
    _engine()
    os.environ["DISPATCH_TARGET"] = site
    os.environ.setdefault(
        "DISPATCH_ARTIFACT_DIR", str(Path.home() / "hep-genesis" / "artifacts")
    )
    _state["site"] = site
    facility = "ALCF" if site == "polaris" else "NERSC"
    from tools.envkernel import ENV_SETUP_CANDIDATES

    candidates = "; ".join(f"{c['name']}: {c['env_setup']}" for c in ENV_SETUP_CANDIDATES.get(site, []))
    try:
        from hep_genesis.iri.dispatch.policy import queue_summary

        caps = queue_summary(site)
    except ImportError:  # engine predating the policy module
        caps = ("Perlmutter walltime caps: debug 30 min, regular 48 h (jobs over 30 min route to regular)"
                if site == "perlmutter" else
                "Polaris walltime caps: debug/debug-scaling 60 min below 10 nodes, prod 3 h at 10-24 nodes")
    return (
        f"Dispatch: remote on {site} ({facility}) - heavy tools will stage this "
        "server's kernels, submit via IRI, and fetch results back. Each call is "
        "one facility job (minutes to hours; the call stays alive with progress "
        f"heartbeats). {caps}. The node runs the pack's own environment "
        "(tools/env/conda-lock.yml, built once on the facility with micromamba - the "
        "first job after a lock change takes minutes longer), so no env_setup is "
        f"needed; env_setup is an optional override to a facility environment "
        f"(TXPipe needs one) - candidates: {candidates}."
    )


def get_dispatch() -> str:
    """Report where THIS DESC server's heavy tools currently execute."""
    site = _state["site"]
    if site == "local":
        return "Dispatch: local execution."
    return f"Dispatch: remote on {site}."


def auth_status() -> str:
    """Report the facility sign-in state THIS DESC server sees for its own
    HPC dispatch (ALCF and NERSC).

    Call this before set_dispatch to a remote site, or when one of this
    server's remote calls fails with an auth error. It says nothing about
    the facility servers' own tokens - ask those servers directly. Sign-in
    happens outside this server (hep-genesis auth CLIs or the desktop app's
    HPC panel); tokens are re-read on every call.
    """
    try:
        from hep_genesis.iri.alcf.client import (
            _live_iri_token as alcf_iri, _live_transfer_token as alcf_tx,
        )
        from hep_genesis.iri.nersc.client import (
            _live_iri_token as nersc_iri, _live_transfer_token as nersc_tx,
        )
    except ImportError:
        return (
            "hep-genesis backend not installed in this server's environment - "
            "server-side remote dispatch unavailable (the client-side "
            "export_dispatch_pack handoff still works). Install: pip install -e "
            "<hep-genesis-agent>/backend[iri]"
        )
    lines = []
    for name, iri, tx in (("ALCF", alcf_iri, alcf_tx), ("NERSC", nersc_iri, nersc_tx)):
        lines.append(
            f"{name}: IRI token {'available' if iri() else 'MISSING'}, "
            f"transfer token {'available' if tx() else 'MISSING'}"
        )
    lines.append(f"{get_dispatch()} Sign in via the hep-genesis auth CLIs or the app's HPC panel.")
    return "\n".join(lines)
