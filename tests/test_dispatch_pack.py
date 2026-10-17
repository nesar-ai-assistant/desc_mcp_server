"""export_dispatch_pack: the client-side HPC handoff contract.

The pack must be stageable by any hep-genesis client that receives it:
sanitized relative paths, engine size caps, a manifest naming kernel entry
points, node-side pip requirements (pip-kernels) or the env_setup requirement
with public candidates (env-kernels) - and nothing identity-shaped.
"""

import json

from mcp_server.dispatch import _PACK_MAX_BYTES, _PACK_MAX_FILES, export_dispatch_pack


def test_pack_shape_and_caps():
    pack = export_dispatch_pack()["dispatch_pack"]
    assert pack["name"] == "tools"
    assert pack["server"] == "desc-mcp-server"
    assert 0 < pack["file_count"] <= _PACK_MAX_FILES
    assert 0 < pack["bytes"] <= _PACK_MAX_BYTES
    assert pack["file_count"] == len(pack["files"])
    json.dumps(pack)  # one JSON tool result


def test_paths_are_safe_and_rooted():
    files = export_dispatch_pack()["dispatch_pack"]["files"]
    for rel in files:
        assert rel.startswith("tools/"), rel
        parts = rel.split("/")
        assert all(p and p != ".." and not p.startswith(".") for p in parts), rel
        assert "__pycache__" not in rel, rel
    assert "tools/common.py" in files
    assert "tools/envkernel.py" in files
    assert "tools/inner/__init__.py" in files
    assert "tools/inner/noop.py" in files


def test_inner_scripts_are_facility_safe():
    """Inner scripts run under a facility env that only guarantees the DESC
    stack: no server-only imports allowed there."""
    import ast

    banned_roots = {"pydantic", "matplotlib", "mcp", "tools"}
    files = export_dispatch_pack()["dispatch_pack"]["files"]
    for rel, text in files.items():
        if not rel.startswith("tools/inner/"):
            continue
        tree = ast.parse(text)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level >= 2:  # reaches back into the server package
                    raise AssertionError(f"{rel}: relative import '{'.' * node.level}{node.module}'")
                if node.level == 1:  # a sibling inner script: fine
                    continue
                names = [node.module or ""]
            for name in names:
                root = name.split(".")[0]
                assert root not in banned_roots, f"{rel} imports '{name}'"


def test_manifest_has_no_identity_and_env_candidates():
    pack = export_dispatch_pack()["dispatch_pack"]
    # "image" is the public registry path of the container (ghcr.io/<owner>/...):
    # an identity-free, credential-free pointer, not a facility identity
    blob = json.dumps({k: v for k, v in pack.items() if k not in ("files", "image")}).lower()
    for forbidden in ("nesar", "m1727", "nersc_project", "pscratch/sd"):
        assert forbidden not in blob
    env = pack["env_kernel"]
    assert env["function"] == "envkernel.run_in_env"
    assert "perlmutter" in env["env_setup_candidates"]
    for name, k in pack["kernels"].items():
        assert "function" in k, name
        if k["function"].startswith("inner."):
            # lock-kernel (contract R8): the pack's environment supplies the stack
            assert k.get("env_setup_required") is False, name
            assert "inner" in k and f"tools/inner/{k['inner']}.py" in pack["files"], name
            assert pack["env_lock"] == "env/conda-lock.yml", name
        elif k["function"] == "envkernel.run_in_env":
            assert k.get("env_setup_required") is True, name
            assert "inner" in k, name
            assert f"tools/inner/{k['inner']}.py" in pack["files"], name
        else:
            assert "pip_deps" in k, name


def test_pack_names_its_container_image():
    pack = export_dispatch_pack()["dispatch_pack"]
    assert "tools/env/image.yml" in pack["files"]
    assert "tools/env/conda-lock-image.yml" not in pack["files"]  # the image is the artefact
    img = pack["image"]
    assert img["ref"] == "ghcr.io/nesar/desc-mcp" and img["tag"].endswith("-cpu")
    assert img["mpi"] == "mpich-abi"
    assert img["perlmutter"]["runtime"] == "podman-hpc" and img["polaris"]["runtime"] == "apptainer"
    from mcp_server.dispatch import _image_manifest
    assert _image_manifest(None) is None and _image_manifest("ref: x\n") is None


def test_pack_carries_its_environment():
    pack = export_dispatch_pack()["dispatch_pack"]
    assert pack["env_lock"] == "env/conda-lock.yml"
    assert "tools/env/conda-lock.yml" in pack["files"] and "tools/env/environment.yml" in pack["files"]
    lock = pack["files"]["tools/env/conda-lock.yml"]
    assert "version: 1" in lock and "linux-64" in lock
    assert pack["bytes"] <= 10 * 1024 * 1024
    assert "env_lock" in pack["usage"] or "conda-lock" in pack["usage"]


def test_manifest_documents_walltime_and_timeout():
    pack = export_dispatch_pack()["dispatch_pack"]
    env = pack["env_kernel"]
    assert "timeout_s" in env["args"] and "walltime" in env
    assert "timeout_s" in pack["usage"] and "duration" in pack["usage"]
    assert "duration_note" in pack["kernels"]["firecrown_chain"]


def test_env_kernel_opts_out_of_the_lock_env(monkeypatch):
    """env_setup names the environment: the engine must not build/activate the lock env."""
    from mcp_server import dispatch as d

    calls = []
    monkeypatch.setitem(d._state, "site", "perlmutter")
    monkeypatch.setattr(d, "_engine", lambda: {"perlmutter": lambda **kw: (calls.append(kw) or {"status": "success", "result": {"ok": 1}})})
    d.run_env_kernel("source /x.sh", "noop", {"a": 1}, duration=600)
    assert calls[-1]["env_lock"] is False and calls[-1]["function"] == "envkernel.run_in_env"
    d.run_kernel("inner.noop.main", {"params": {}}, duration=600)
    assert calls[-1]["env_lock"] is None  # lock env by default


def test_job_lands_in_the_calls_output_dir(monkeypatch, tmp_path):
    """A dispatched job's files + dispatch record go to <output_dir>/<kernel>_<ts>/
    of the CALL, not the process-wide DISPATCH_ARTIFACT_DIR another chat set
    (2026-10-14: four jobs from other chats landed in one chat's run folder)."""
    import pytest

    pytest.importorskip("hep_genesis")
    from mcp_server import dispatch as d

    calls = []
    monkeypatch.setitem(d._state, "site", "perlmutter")
    monkeypatch.setattr(d, "_engine", lambda: {"perlmutter": lambda **kw: (calls.append(kw) or {"status": "success", "result": {}})})
    reset = d.CALL_OUTPUT_DIR.set(str(tmp_path))
    try:
        d.run_kernel("inner.firecrown_chain.main", {"params": {}}, duration=600)
        d.run_env_kernel("source /x.sh", "noop", {}, duration=600)
    finally:
        d.CALL_OUTPUT_DIR.reset(reset)
    chain, env = calls
    assert chain["job_name"].startswith("firecrown_chain_") and chain["artifact_dir"] == str(tmp_path / chain["job_name"])
    assert env["job_name"].startswith("noop_")
    d.run_kernel("inner.noop.main", {"params": {}}, duration=600)  # no call context: engine defaults
    assert "artifact_dir" not in calls[-1] and "job_name" not in calls[-1]


def test_run_kernel_no_wait_returns_the_handle_and_collect_kernel_checks(monkeypatch, tmp_path):
    """wait=False (contract R10): the engine's 'submitted' handle passes through
    untouched; collect_kernel reads the record, passes pending through and
    raises the usual error for a failed job."""
    import pytest

    pytest.importorskip("hep_genesis")
    from hep_genesis.iri.dispatch import collect as c

    from mcp_server import dispatch as d

    calls = []
    handle = {"status": "submitted", "job_id": "42", "site": "perlmutter", "record": str(tmp_path / "r.json")}
    monkeypatch.setitem(d._state, "site", "perlmutter")
    monkeypatch.setattr(d, "_engine", lambda: {"perlmutter": lambda **kw: (calls.append(kw) or dict(handle))})
    assert d.run_kernel("inner.noop.main", {"params": {}}, duration=600, wait=False) == handle
    assert calls[-1]["wait"] is False
    d.run_kernel("inner.noop.main", {"params": {}}, duration=600, wait=False)
    calls.clear()
    monkeypatch.setattr(d, "_engine", lambda: {"perlmutter": lambda **kw: (calls.append(kw) or {"status": "success", "result": {}})})
    d.run_kernel("inner.noop.main", {"params": {}}, duration=600)
    assert "wait" not in calls[-1]  # older engines without the arg keep working

    rec = tmp_path / "dispatch_noop_1.json"
    rec.write_text(json.dumps({"job_name": "noop_1", "job_id": "42", "site": "perlmutter"}))
    monkeypatch.setattr(c, "collect_dispatched", lambda r: {"status": "pending", "state": "QUEUED"})
    assert d.collect_kernel(str(rec))["status"] == "pending"
    monkeypatch.setattr(c, "collect_dispatched", lambda r: {"status": "error", "error": "boom", "stderr": "x"})
    with pytest.raises(RuntimeError, match="boom"):
        d.collect_kernel(str(rec))
