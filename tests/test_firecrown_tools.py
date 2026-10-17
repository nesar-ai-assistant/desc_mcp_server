"""firecrown_tools: build / loglike / theory / scan / chain on the offline
DES-Y1-like test data vector (firecrown/tests/sacc_data.hdf5, copied to tmp).

Expected numbers (firecrown 1.16, pure_ccl_mode, halofit): at the example
fiducials chi2 ~ 552 for 457 points (chi2/n ~ 1.2); a noiseless realization
re-evaluated at the same point gives chi2 = 0.
"""

import base64
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pydantic
import pytest

pytest.importorskip("sacc")
pytest.importorskip("firecrown")

from tools.common import clone_dir  # noqa: E402
from tools.cosmology import CosmologyParams  # noqa: E402
from tools.firecrown_tools import (DISPATCH_KERNELS, firecrown_build_likelihood,  # noqa: E402
                                   firecrown_compute_loglike, firecrown_list_examples,
                                   firecrown_run_chain, firecrown_scan_loglike,
                                   firecrown_theory_data_vector)

_FC = clone_dir("firecrown")
_DES = _FC / "tests" / "sacc_data.hdf5" if _FC else None
if _DES is None or not _DES.is_file():
    pytest.skip("firecrown clone with tests/sacc_data.hdf5 not available", allow_module_level=True)

FIDUCIAL = firecrown_list_examples().metadata["des_y1_fiducial"]


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    d = tmp_path_factory.mktemp("firecrown")
    shutil.copy(_DES, d / "sacc_data.hdf5")
    return d


@pytest.fixture(scope="module")
def experiment(work):
    r = firecrown_build_likelihood(output_dir=str(work), sacc_path=str(work / "sacc_data.hdf5"))
    return r


def test_list_examples():
    r = firecrown_list_examples()
    assert set(r.metadata["examples"]) == {"cosmic_shear", "cmb_cross", "sn_srd", "des_y1_3x2pt"}
    assert r.metadata["offline_test_sacc"] == str(_DES)
    assert FIDUCIAL["lens0_bias"] == 1.4 and FIDUCIAL["src1_delta_z"] == -0.019


def test_build_likelihood(experiment, work):
    m = experiment.metadata
    assert Path(m["experiment_yaml"]).is_file() and m["correlation_space"] == "real"
    assert m["n_data"] == 457 and len(m["statistics"]) == 45
    req = m["required_parameters"]
    for name in ("Omega_c", "sigma8", "h", "ia_bias", "alphaz", "z_piv", "lens0_bias", "lens4_delta_z",
                 "src0_delta_z", "src3_mult_bias", "m_nu"):
        assert name in req, name
    assert "A_s" not in req and req["lens0_bias"] == 1.5 and req["src0_mult_bias"] == 1.0
    assert req["m_nu"] == 0.0
    # scale cuts reduce n_data; unknown factories / wrong space are rejected cheaply
    cut = firecrown_build_likelihood(
        output_dir=str(work), sacc_path=str(work / "sacc_data.hdf5"), name="cut",
        scale_cuts=[{"tracer": "src0", "measurement": "shear", "lower": 10, "upper": 100},
                    {"tracer": "lens0", "measurement": "density", "lower": 20, "upper": 250}])
    assert 0 < cut.metadata["n_data"] < 457 and cut.metadata["n_data_total"] == 457
    with pytest.raises(ValueError, match="unsupported systematics"):
        firecrown_build_likelihood(output_dir=str(work), sacc_path=str(work / "sacc_data.hdf5"),
                                   wl_per_bin=["TattAlignmentSystematicFactory"])
    with pytest.raises(ValueError, match="holds real-space"):
        firecrown_build_likelihood(output_dir=str(work), sacc_path=str(work / "sacc_data.hdf5"),
                                   correlation_space="harmonic")


def test_loglike_at_fiducial(experiment, work):
    out = work / "ll"
    r = firecrown_compute_loglike(output_dir=str(out), experiment_yaml=experiment.metadata["experiment_yaml"],
                                  nuisance=dict(FIDUCIAL, typo_param=1.0))
    m = r.metadata
    assert math.isfinite(m["loglike"]) and math.isfinite(m["chi2"])
    assert m["n_data"] == 457 and 0.5 < m["chi2_per_dof"] < 2.5
    assert abs(m["loglike"] + 0.5 * m["chi2"]) < 1e-6
    assert any("typo_param" in w for w in m["warnings"])
    per = m["per_statistic"]
    assert len(per) == 45 and abs(sum(p["chi2_share"] for p in per) - m["chi2"]) < 1e-6
    assert all(p["chi2_block"] > 0 for p in per)
    names = {Path(f).name for f in r.files}
    assert any(n.startswith("theory_vs_data_") and n.endswith(".csv") for n in names)
    assert any(n.endswith(".png") for n in names)
    json.dumps(m)
    # second call: cached likelihood, different cosmology changes chi2
    r2 = firecrown_compute_loglike(output_dir=str(out), experiment_yaml=experiment.metadata["experiment_yaml"],
                                   nuisance=FIDUCIAL, cosmology=CosmologyParams(sigma8=0.7), plot=False)
    assert r2.metadata["chi2"] != m["chi2"] and math.isfinite(r2.metadata["chi2"])
    # amplitude mismatch is a clear error, not a silent default
    with pytest.raises(ValueError, match="amplitude_parameter"):
        firecrown_compute_loglike(output_dir=str(out), experiment_yaml=experiment.metadata["experiment_yaml"],
                                  cosmology=CosmologyParams(A_s=2.1e-9, sigma8=None), plot=False)


def test_scan_loglike(experiment, work):
    r = firecrown_scan_loglike(output_dir=str(work / "scan"), experiment_yaml=experiment.metadata["experiment_yaml"],
                               parameter="sigma8", min_value=0.74, max_value=0.90, n_points=5, nuisance=FIDUCIAL)
    m = r.metadata
    assert len(m["grid"]) == 5 and all(math.isfinite(c) for c in m["chi2"])
    assert min(m["delta_chi2"]) == 0.0 and 0.74 <= m["grid_minimum"] <= 0.90
    assert firecrown_scan_loglike.weight == "dispatchable"
    assert len(r.files) == 2
    with pytest.raises(ValueError, match="not a parameter"):
        firecrown_scan_loglike(output_dir=str(work), experiment_yaml=experiment.metadata["experiment_yaml"],
                               parameter="bogus", min_value=0, max_value=1)


def test_theory_vector_and_closure(experiment, work):
    out = work / "theory"
    r = firecrown_theory_data_vector(output_dir=str(out), experiment_yaml=experiment.metadata["experiment_yaml"],
                                     nuisance=FIDUCIAL, add_noise=False)
    m = r.metadata
    assert m["n_data"] == 457 and m["sacc_file"] and Path(m["sacc_file"]).is_file()
    # closure: likelihood of the noiseless realization at the same point is exactly zero chi2
    b = firecrown_build_likelihood(output_dir=str(out), sacc_path=m["sacc_file"], name="closure")
    l = firecrown_compute_loglike(output_dir=str(out), experiment_yaml=b.metadata["experiment_yaml"],
                                  nuisance=FIDUCIAL, plot=False)
    assert abs(l.metadata["chi2"]) < 1e-6
    noisy = firecrown_theory_data_vector(output_dir=str(out), experiment_yaml=experiment.metadata["experiment_yaml"],
                                         nuisance=FIDUCIAL, add_noise=True, seed=3, plot=False)
    assert noisy.metadata["sacc_file"] != m["sacc_file"]


def test_inner_kernel_json_roundtrip(experiment, work, tmp_path, monkeypatch):
    """The env-kernel path: inline YAML + base64 sacc materialised in a job dir,
    result JSON-serialisable."""
    from tools.inner import firecrown_loglike as inner

    yaml_text = Path(experiment.metadata["experiment_yaml"]).read_text()
    sacc_b64 = base64.b64encode((work / "sacc_data.hdf5").read_bytes()).decode("ascii")
    params = {"experiment_yaml_text": yaml_text, "sacc_b64": sacc_b64, "sacc_name": "shipped.hdf5",
              "points": [dict(FIDUCIAL, Omega_c=0.26)], "return_vectors": False, "per_statistic": False}
    params = json.loads(json.dumps(params))
    monkeypatch.chdir(tmp_path)
    res = json.loads(json.dumps(inner.main(params)))
    assert (tmp_path / "shipped.hdf5").is_file() and (tmp_path / "experiment.yaml").is_file()
    assert res["n_data"] == 457 and math.isfinite(res["results"][0]["loglike"])
    assert "lens0_bias" in res["required"]
    for k in ("firecrown_loglike", "firecrown_theory", "firecrown_chain"):
        assert DISPATCH_KERNELS[k]["env_setup_required"] is False and DISPATCH_KERNELS[k]["inner"] == k
        assert DISPATCH_KERNELS[k]["function"] == f"inner.{k}.main"
        assert "desc-python" in DISPATCH_KERNELS[k]["suitable_envs"]


def test_run_chain_smoke(experiment, work):
    pytest.importorskip("cobaya")
    r = firecrown_run_chain(output_dir=str(work / "chain"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"sigma8": {"min": 0.6, "max": 1.0}}, nuisance=FIDUCIAL,
                            max_samples=4, seed=1)
    m = r.metadata
    assert m["sampled"] == ["sigma8"] and m["n_samples"] >= 4
    assert 0.6 <= m["summary"]["sigma8"]["mean"] <= 1.0 and m["summary"]["sigma8"]["std"] >= 0
    assert firecrown_run_chain.weight == "heavy"
    assert any(f.endswith(".1.txt") for f in m["chain_files"])
    assert any(Path(f).name.startswith("chain_summary_") for f in r.files)
    json.dumps(m)
    assert "smoke test" in r.message  # not converged in 4 samples


def test_chain_walltime_policy():
    from tools.firecrown_tools import _chain_walltime, _needs_ppf

    wt, est, warns = _chain_walltime(20000, None)
    assert wt == 12 * 3600 and est == 100000 and any("resume=True" in w for w in warns)
    wt, est, warns = _chain_walltime(200, None)
    assert wt == 1800 and not warns
    wt, est, warns = _chain_walltime(20000, 3500)
    assert wt == 3500 and any("may stop" in w for w in warns)
    assert _needs_ppf({"wa": {"min": -3, "max": 1}}, {})
    assert _needs_ppf({"w0": {"min": -2, "max": -0.3}}, {})
    assert not _needs_ppf({"w0": {"min": -1, "max": -0.3}}, {"wa": 0.0})
    assert _needs_ppf({"sigma8": {"min": 0.6, "max": 1}}, {"wa": 0.3})


def test_run_chain_auto_ppf_w0wa(experiment, work):
    pytest.importorskip("cobaya")
    r = firecrown_run_chain(output_dir=str(work / "chain_ppf"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"w0": {"min": -1.6, "max": -0.6}, "wa": {"min": -1.0, "max": 0.5}},
                            nuisance=FIDUCIAL, max_samples=4, seed=2, plot=False)
    m = r.metadata
    assert m["cobaya_settings"]["dark_energy_model"] == "ppf"
    assert m["experiment_yaml"].endswith("_ppf.yaml") and Path(m["experiment_yaml"]).is_file()
    assert m["n_samples"] >= 4 and set(m["sampled"]) == {"w0", "wa"}
    assert m["cobaya_settings"]["walltime_s"] == 1800


def test_chain_status_and_plot(experiment, work):
    pytest.importorskip("cobaya")
    from tools.firecrown_tools import firecrown_chain_status, firecrown_plot_chain

    r = firecrown_run_chain(output_dir=str(work / "chain"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"sigma8": {"min": 0.6, "max": 1.0}}, nuisance=FIDUCIAL,
                            max_samples=4, seed=1)
    chain_txt = r.metadata["chain_txt"]
    assert chain_txt and chain_txt.endswith(".1.txt")
    st = firecrown_chain_status(chain_txt=chain_txt)
    assert st.metadata["status"] == "finished" and st.metadata["n_samples"] >= 4
    assert st.metadata["progress"].get("last") is not None or st.metadata["n_samples"] < 320
    p = firecrown_plot_chain(output_dir=str(work / "chain"), chain_txt=chain_txt, burn_in_frac=0.25)
    assert p.metadata["parameters"] == ["sigma8"]
    assert 0.6 <= p.metadata["summary"]["sigma8"]["mean"] <= 1.0
    assert any(f.endswith(".png") and "corner" in f for f in p.files)
    assert any(f.endswith(".png") and "trace" in f for f in p.files)
    assert any(Path(f).name.startswith("chain_posterior_") for f in p.files)
    json.dumps(p.metadata)
    with pytest.raises(ValueError):
        firecrown_plot_chain(output_dir=str(work / "chain"), chain_txt=chain_txt, params=["nope"])


def test_run_chain_background_then_status(experiment, work):
    pytest.importorskip("cobaya")
    import time

    from tools.firecrown_tools import firecrown_chain_status, firecrown_plot_chain

    r = firecrown_run_chain(output_dir=str(work / "chain_bg"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"sigma8": {"min": 0.6, "max": 1.0}}, nuisance=FIDUCIAL,
                            max_samples=4, seed=3, background=True)
    m = r.metadata
    assert m["background"]["pid"] > 0 and m["computed_on"] == "local (background)"
    chain_txt = m["chain_txt"]
    deadline = time.time() + 420
    status = None
    while time.time() < deadline:
        status = firecrown_chain_status(chain_txt=chain_txt).metadata
        if status["status"] in ("finished", "failed"):
            break
        time.sleep(3)
    assert status is not None and status["status"] == "finished", status
    p = firecrown_plot_chain(output_dir=str(work / "chain_bg"), chain_txt=chain_txt, burn_in_frac=0.0, corner=False)
    assert p.metadata["n_total"] >= 4


def test_run_chain_resume_requires_existing_chain(experiment, work):
    pytest.importorskip("cobaya")
    with pytest.raises(ValueError, match="resume=True"):
        firecrown_run_chain(output_dir=str(work / "chain_none"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"sigma8": {"min": 0.6, "max": 1.0}}, nuisance=FIDUCIAL,
                            max_samples=4, seed=9, resume=True)


def test_run_chain_rejects_w0_wa_corner(experiment, work):
    with pytest.raises(ValueError, match="w0 \\+ wa < 0"):
        firecrown_run_chain(output_dir=str(work / "chain_bad"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"w0": {"min": -2, "max": -0.3}, "wa": {"min": -3, "max": 1}},
                            nuisance=FIDUCIAL, max_samples=4)


def test_background_chain_cancel(experiment, work):
    pytest.importorskip("cobaya")
    import time

    from tools.firecrown_tools import firecrown_chain_cancel, firecrown_chain_status

    r = firecrown_run_chain(output_dir=str(work / "chain_cancel"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            priors={"sigma8": {"min": 0.6, "max": 1.0}}, nuisance=FIDUCIAL,
                            max_samples=5000, seed=4, background=True)
    chain_txt = r.metadata["chain_txt"]
    time.sleep(2)
    assert firecrown_chain_status(chain_txt=chain_txt).metadata["status"] == "running"
    c = firecrown_chain_cancel(chain_txt=chain_txt)
    assert c.metadata["was_running"] is True
    st = firecrown_chain_status(chain_txt=chain_txt).metadata
    assert st["status"] in ("stopped", "failed")
    assert firecrown_chain_cancel(chain_txt=chain_txt).metadata["was_running"] is False
    with pytest.raises(ValueError):
        firecrown_chain_cancel(chain_txt=str(work / "nope.1.txt"))


def test_inner_chain_file_locking_switch(experiment, work, monkeypatch):
    pytest.importorskip("cobaya")
    import os

    from tools.inner import firecrown_chain as inner

    monkeypatch.delenv("COBAYA_USE_FILE_LOCKING", raising=False)
    inner.main({"experiment_yaml": experiment.metadata["experiment_yaml"], "fixed": FIDUCIAL,
                "priors": {"sigma8": {"min": 0.6, "max": 1.0}}, "max_samples": 4, "seed": 5,
                "work_dir": str(work / "chain_nolock"), "chain_prefix": "nolock", "file_locking": False})
    assert os.environ.get("COBAYA_USE_FILE_LOCKING") == "False"
    assert (work / "chain_nolock" / "nolock.1.txt").is_file()


def test_remote_routing_lock_kernel_vs_env_setup(experiment, monkeypatch):
    """Remote: no env_setup -> the inner script runs as a lock-kernel; env_setup -> env-kernel."""
    from mcp_server import dispatch as d
    from tools.firecrown_tools import _run_inner

    calls = []
    monkeypatch.setattr(d, "remote_site", lambda: "perlmutter")
    monkeypatch.setattr(d, "run_kernel", lambda fn, args, **kw: (calls.append(("lock", fn, kw)) or
                        {"result": {"ok": 1}, "host": "nid001"}))
    monkeypatch.setattr(d, "run_env_kernel", lambda env, inner, params, **kw: (calls.append(("env", inner, env)) or
                        {"result": {"result": {"ok": 2}}, "host": "nid002"}))
    exp = Path(experiment.metadata["experiment_yaml"])
    res, host = _run_inner("firecrown_loglike", {"points": []}, None, 600, exp, None)
    assert res == {"ok": 1} and host == "nid001" and calls[-1][:2] == ("lock", "inner.firecrown_loglike.main")
    assert calls[-1][2]["pip_deps"] is None and calls[-1][2]["duration"] == 600
    res, host = _run_inner("firecrown_loglike", {"points": []}, "source /x/env.sh", 600, exp, None)
    assert res == {"ok": 2} and host == "nid002" and calls[-1][:2] == ("env", "firecrown_loglike")


def test_inner_chain_progress_line(tmp_path):
    from tools.inner.firecrown_chain import _progress_line, _start_progress_writer

    p = tmp_path / "c.progress"
    assert _progress_line(str(p), 20) is None
    p.write_text("#  N  timestamp  acceptance_rate  Rminus1  Rminus1_cl\n 320.0 2026-10-11T08:40:45 0.243 0.0342 NaN\n")
    assert _progress_line(str(p), 20000) == "320 accepted samples of 20000, acceptance 0.24, R-1 0.0342"
    out = tmp_path / "progress.txt"
    stop = _start_progress_writer(str(p), str(out), 20000, period_s=0.1)
    stop()
    assert out.read_text().strip() == "320 accepted samples of 20000, acceptance 0.24, R-1 0.0342"


# ---- multi-node chains (contract R9: one Cobaya chain per MPI process) ----

def test_chain_declares_its_mpi_shape_and_the_manifest_mirrors_it():
    from tools.inner.firecrown_chain import DISPATCH_MPI

    shape = DISPATCH_MPI["main"]
    assert set(shape) <= {"tasks_per_node", "cpus_per_task", "gpus_per_task", "gather"}
    assert shape["tasks_per_node"] >= 1 and shape["cpus_per_task"] >= 1
    mirror = DISPATCH_KERNELS["firecrown_chain"]["mpi"]
    assert {k: mirror[k] for k in shape} == shape and "nodes" in mirror["note"]
    engine = pytest.importorskip("hep_genesis.iri.dispatch.mpi")
    from mcp_server.dispatch import TOOLS_DIR
    from hep_genesis.iri.dispatch.pack import collect_pack_files, resolve_pack

    pack = resolve_pack(str(TOOLS_DIR))
    files = {k.removeprefix(f"{pack.remote_name}/"): v for k, v in collect_pack_files(pack).items()}
    # the engine reads the declaration from the shipped pack text, no import
    assert engine.kernel_mpi_decl(files, "inner.firecrown_chain.main") == shape
    assert engine.kernel_mpi_decl(files, "inner.firecrown_loglike.main") is None


def test_inner_chain_mpi_helpers(tmp_path):
    import sys

    from tools.inner.firecrown_chain import _chain_count, _mpi_world

    if "mpi4py.MPI" in sys.modules:
        pytest.skip("mpi4py already initialised in this process")
    assert _mpi_world() == (None, 0, 1)  # never imports mpi4py itself
    assert "mpi4py.MPI" not in sys.modules
    for name in ("chain.1.txt", "chain.2.txt", "chain.progress", "chain.1.txt.bak", "other.1.txt"):
        (tmp_path / name).write_text("")
    assert _chain_count(str(tmp_path), "chain") == 2
    assert _chain_count(str(tmp_path / "missing"), "chain") == 0


def test_inner_chain_refuses_resume_with_a_different_chain_count(experiment, work, monkeypatch):
    """Cobaya checkpoints one chain per process: a 1-chain run cannot be resumed as 2."""
    import sys
    import types

    from tools.inner import firecrown_chain as fc

    class FakeComm:
        def Get_rank(self): return 0
        def Get_size(self): return 2
        def bcast(self, obj, root=0): return obj
        def Barrier(self): pass
        def gather(self, obj, root=0): return [obj]

    fake = types.ModuleType("mpi4py.MPI")
    fake.COMM_WORLD = FakeComm()
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", fake)
    assert fc._mpi_world()[1:] == (0, 2)
    wd = work / "chain_resume_mismatch"
    wd.mkdir(exist_ok=True)
    (wd / "chain.1.txt").write_text("")  # one chain from a previous single-process run
    with pytest.raises(ValueError, match="resume=True with 2 chain"):
        fc.main({"experiment_yaml": experiment.metadata["experiment_yaml"], "fixed": FIDUCIAL,
                 "priors": {"sigma8": {"min": 0.6, "max": 1.0}}, "max_samples": 4, "work_dir": str(wd),
                 "chain_prefix": "chain", "resume": True})


def test_run_inner_forwards_nodes_and_reports_the_mpi_shape(experiment, monkeypatch):
    from mcp_server import dispatch as d
    from tools.firecrown_tools import _run_inner

    calls = []
    monkeypatch.setattr(d, "remote_site", lambda: "perlmutter")
    monkeypatch.setattr(d, "run_kernel", lambda fn, args, **kw: (calls.append(kw) or {
        "result": {"ok": 1}, "host": "nid001", "mpi": {"size": 8, "nodes": ["nid001", "nid002"]}}))
    exp = Path(experiment.metadata["experiment_yaml"])
    res, host = _run_inner("firecrown_chain", {"priors": {}}, None, 600, exp, None, nodes=2)
    assert res == {"ok": 1} and calls[-1]["nodes"] == 2
    assert host == "nid001 (8 MPI ranks on 2 nodes)"
    res, host = _run_inner("firecrown_loglike", {"points": []}, None, 600, exp, None)
    assert calls[-1]["nodes"] == 1 and host == "nid001 (8 MPI ranks on 2 nodes)"
    with pytest.raises(ValueError, match="env_setup"):
        _run_inner("firecrown_chain", {"priors": {}}, "source /x/env.sh", 600, exp, None, nodes=2)


def test_run_chain_nodes_validation(experiment, work, monkeypatch):
    from mcp_server import dispatch as d

    exp = experiment.metadata["experiment_yaml"]
    common = dict(output_dir=str(work / "chain_nodes"), experiment_yaml=exp, nuisance=FIDUCIAL,
                  priors={"sigma8": {"min": 0.6, "max": 1.0}}, max_samples=4)
    with pytest.raises(ValueError, match="set_dispatch"):
        firecrown_run_chain(nodes=2, **common)  # local: one chain only
    monkeypatch.setattr(d, "remote_site", lambda: "perlmutter")
    with pytest.raises(ValueError, match="env_setup"):
        firecrown_run_chain(nodes=2, env_setup="source /x/env.sh", **common)
    with pytest.raises(ValueError, match="background"):
        firecrown_run_chain(nodes=2, background=True, **common)
    with pytest.raises(pydantic.ValidationError):
        firecrown_run_chain(nodes=0, **common)


def test_run_chain_passes_nodes_to_the_kernel_and_combines_chains(experiment, work, monkeypatch):
    """Remote 2-node chain: the wrapper sends nodes=2 and shapes the combined result."""
    from mcp_server import dispatch as d

    seen = {}

    def fake_run_kernel(fn, args, **kw):
        seen.update(kw, fn=fn, params=args["params"])
        return {"result": {"chain_files": ["chain.1.txt", "chain.2.txt"], "n_samples": 48, "n_weighted": 60.0,
                           "n_chains": 8, "chain_samples": [9] * 8, "skip_samples": 0.33,
                           "acceptance_rate": 0.25, "chain_acceptance": [0.25] * 8,
                           "summary": {"sigma8": {"mean": 0.8, "std": 0.05, "lower68": 0.75, "upper68": 0.85}},
                           "sampled": ["sigma8"], "converged": True, "Rminus1": 0.01},
                "host": "nid001", "mpi": {"size": 8, "nodes": ["nid001", "nid002"]}}

    monkeypatch.setattr(d, "remote_site", lambda: "perlmutter")
    monkeypatch.setattr(d, "run_kernel", fake_run_kernel)
    r = firecrown_run_chain(output_dir=str(work / "chain_mpi"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            nuisance=FIDUCIAL, priors={"sigma8": {"min": 0.6, "max": 1.0}}, max_samples=100,
                            nodes=2, walltime_s=1800, plot=False)
    assert seen["fn"] == "inner.firecrown_chain.main" and seen["nodes"] == 2 and seen["duration"] == 1800
    assert seen["params"]["file_locking"] is False and seen["params"]["max_samples"] == 100
    m = r.metadata
    assert m["n_chains"] == 8 and m["cobaya_settings"]["nodes"] == 2 and m["cobaya_settings"]["n_chains"] == 8
    assert m["computed_on"] == "nid001 (8 MPI ranks on 2 nodes)"
    assert "8 chains combined" in r.message and "cross-chain" in r.message and "smoke test" not in r.message
    json.dumps(m)


def test_run_chain_no_wait_then_collect(experiment, work, monkeypatch, tmp_path):
    """wait=False submits and returns a pending file at once (contract R10);
    firecrown_collect_chain reports 'pending' until the job is done, then
    shapes the chain like a waiting run - with the facility's chain file paths
    mapped to the local copies the engine fetched back (trace plot works)."""
    from mcp_server import dispatch as d
    from tools.firecrown_tools import firecrown_collect_chain

    seen = {}

    def fake_run_kernel(fn, args, **kw):
        seen.update(kw)
        return {"status": "submitted", "job_id": "59800001", "site": "perlmutter", "node_hours": 2.0,
                "record": str(tmp_path / "dispatch_x.json"), "artifact_dir": str(tmp_path)}

    monkeypatch.setattr(d, "remote_site", lambda: "perlmutter")
    monkeypatch.setattr(d, "run_kernel", fake_run_kernel)
    out = work / "chain_nowait"
    r = firecrown_run_chain(output_dir=str(out), experiment_yaml=experiment.metadata["experiment_yaml"],
                            nuisance=FIDUCIAL, priors={"sigma8": {"min": 0.6, "max": 1.0}}, max_samples=100,
                            walltime_s=1800, wait=False)
    assert seen["wait"] is False and r.metadata["submitted"] and "59800001" in r.message
    pending = r.metadata["pending"]
    assert Path(pending).is_file() and "firecrown_collect_chain" in r.message

    monkeypatch.setattr(d, "collect_kernel", lambda rec, wait_s=0: {"status": "pending", "state": "QUEUED", "message": "later", "wait_s": wait_s})
    p = firecrown_collect_chain(output_dir=str(out), pending=pending)
    assert p.metadata["pending"] and "QUEUED" in p.message

    chain = tmp_path / "fetched" / "chain_abc.1.txt"
    chain.parent.mkdir()
    chain.write_text("# weight minuslogpost sigma8\n" + "\n".join(f"1 {i} {0.8 + 0.001 * i}" for i in range(20)))
    done = {"status": "success", "host": "nid001", "artifact_files": [str(chain)],
            "result": {"chain_files": ["./chain_abc.1.txt"], "n_samples": 20, "n_weighted": 20.0,
                       "acceptance_rate": 0.3, "converged": True, "Rminus1": 0.01, "sampled": ["sigma8"],
                       "summary": {"sigma8": {"mean": 0.81, "std": 0.01, "lower68": 0.80, "upper68": 0.82}}}}
    monkeypatch.setattr(d, "collect_kernel", lambda rec, wait_s=0: done)
    f = firecrown_collect_chain(output_dir=str(out), pending=pending)
    assert f.metadata["chain_txt"] == str(chain) and f.metadata["converged"] is True
    assert any(x.endswith(".png") for x in f.files)  # trace plot drawn from the local copy


def test_run_chain_no_wait_is_facility_only(experiment, work):
    with pytest.raises(ValueError, match="facility"):
        firecrown_run_chain(output_dir=str(work / "nw_local"), experiment_yaml=experiment.metadata["experiment_yaml"],
                            nuisance=FIDUCIAL, priors={"sigma8": {"min": 0.6, "max": 1.0}}, wait=False)
