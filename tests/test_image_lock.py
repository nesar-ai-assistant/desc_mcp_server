"""The container image's environment: environment-image.yml <-> conda-lock-image.yml
consistency (linux-64), MPICH-ABI only (no Open MPI, no bundled MPI in the solve that a
facility could not inject over), same package set as the laptop lock apart from MPI, and
the Dockerfile/image metadata that reference it."""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "tools" / "env"


@pytest.fixture(scope="module")
def lock():
    path = ENV / "conda-lock-image.yml"
    if not path.is_file():
        pytest.fail("tools/env/conda-lock-image.yml missing: run `bash scripts/image.sh --relock`")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _spec_name(spec: str) -> str:
    return re.split(r"[<>=!~ @\[]", spec.strip(), 1)[0].lower()


def _specs(path):
    src = yaml.safe_load(path.read_text(encoding="utf-8"))
    conda, pip = [], []
    for dep in src["dependencies"]:
        if isinstance(dep, dict):
            pip += [_spec_name(s) for s in dep.get("pip", [])]
        else:
            conda.append(_spec_name(dep))
    return conda, pip


def test_image_lock_is_linux_64(lock):
    assert "linux-64" in lock["metadata"]["platforms"]


def test_every_image_spec_is_locked(lock):
    conda_specs, pip_specs = _specs(ENV / "environment-image.yml")
    names = {p["name"].lower() for p in lock["package"] if p["platform"] == "linux-64"}
    missing = [n for n in conda_specs + pip_specs if n not in names]
    assert not missing, f"not in the image lock: {missing}"
    python = [p for p in lock["package"] if p["platform"] == "linux-64" and p["name"] == "python"]
    assert python and python[0]["version"].startswith("3.12.")


def test_image_lock_is_mpich_abi(lock):
    pkgs = {p["name"].lower(): p for p in lock["package"] if p["platform"] == "linux-64"}
    assert "mpich" in pkgs and "mpi4py" in pkgs
    assert "openmpi" not in pkgs, "Open MPI cannot be replaced by the facilities' Cray MPICH"
    assert pkgs["mpi"]["url"].endswith("-mpich.conda"), pkgs["mpi"]["url"]
    assert "mpich" in pkgs["mpi4py"]["dependencies"], "mpi4py must be the mpich build"


def test_image_env_matches_laptop_env_apart_from_mpi():
    conda_a, pip_a = _specs(ENV / "environment.yml")
    conda_b, pip_b = _specs(ENV / "environment-image.yml")
    assert set(conda_b) - set(conda_a) == {"mpich", "mpi4py"}
    assert set(conda_a) - set(conda_b) == set()
    assert pip_a == pip_b


def test_dockerfile_uses_the_image_lock():
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "tools/env/conda-lock-image.yml" in text
    # micromamba only recognises a lock named conda-lock.yml (2026-10-16, micromamba 2.9)
    assert "/conda-lock.yml" in text
    assert "ENTRYPOINT" not in [ln.split()[0] for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    assert "mpich-bundled" in text
    assert "FROM ubuntu:24.04" in text, "Polaris MPI injection needs GLIBC_2.38 (ubuntu 22.04 has 2.35)"


def test_image_metadata_file():
    meta = yaml.safe_load((ENV / "image.yml").read_text(encoding="utf-8"))
    assert meta["ref"].startswith("ghcr.io/")
    assert meta["lock"] == "conda-lock-image.yml"
    import hashlib
    sha = hashlib.sha1((ENV / "conda-lock-image.yml").read_bytes()).hexdigest()[:12]
    assert meta["tag"] == f"{sha}-u24-cpu", f"image.yml tag {meta['tag']} != lock sha {sha}: rerun scripts/image.sh"
    assert meta.get("base") == "ubuntu:24.04"  # Polaris cray-mpich-abi needs glibc >= 2.38
