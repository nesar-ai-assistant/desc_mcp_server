"""Facility smoke test: the multi-node Cobaya chain INSIDE the pack's container image (contract R11).

    .mcp-env/bin/python tests/smoke_chain_image.py perlmutter|polaris [NODES] [MAX_SAMPLES]

Same science as smoke_chain_mpi.py (DES Y1 3x2pt, sigma8 + Omega_c, NODES x 4
chains), but submitted with wait=False and collected with
firecrown_collect_chain, and the engine runs the kernel in the image named by
tools/env/image.yml (the lock env is only the fallback). Prints the dispatch
record's carrier, the queue/run timings and the chain checks. Costs
allocation: NODES x ~30 min debug (+ the first image fetch).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(site: str, nodes: int, max_samples: int) -> int:
    from mcp_server.dispatch import set_dispatch
    from tools.firecrown_tools import (firecrown_build_likelihood, firecrown_collect_chain,
                                       firecrown_list_examples, firecrown_run_chain)
    from tools.inner.firecrown_chain import DISPATCH_MPI

    sacc = Path(__import__("os").environ.get("DESC_FIRECROWN_DIR", ROOT / "firecrown")) / "tests" / "sacc_data.hdf5"
    if not sacc.is_file():
        print(f"missing {sacc} (firecrown clone with tests; set DESC_FIRECROWN_DIR)"); return 2
    out = Path.home() / "hep-genesis" / "artifacts" / f"chain_image_{site}_{nodes}n_{time.strftime('%y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    fiducial = firecrown_list_examples().metadata["des_y1_fiducial"]
    exp = firecrown_build_likelihood(output_dir=str(out), sacc_path=str(sacc)).metadata["experiment_yaml"]
    print(set_dispatch(site, artifact_dir=str(out)))
    per_node = int(DISPATCH_MPI["main"]["tasks_per_node"])
    expected = nodes * per_node
    print(f"[1] {site}: {nodes} node(s) x {per_node} chains = {expected} chains of {max_samples} samples (image); artifacts -> {out}")
    t0 = time.time()
    sub = firecrown_run_chain(output_dir=str(out), experiment_yaml=exp, nuisance=fiducial,
                              priors={"sigma8": {"min": 0.6, "max": 1.0, "ref": 0.8},
                                      "Omega_c": {"min": 0.15, "max": 0.4, "ref": 0.26}},
                              max_samples=max_samples, nodes=nodes, walltime_s=1800, seed=1, plot=True, wait=False)
    t_submit = time.time() - t0
    print(f"[2] submitted after {t_submit:.0f}s: {sub.message[:500]}")
    pending = sub.metadata.get("pending")
    if not pending:
        print("FAIL: no pending file (wait=False did not submit)"); return 1
    rec = Path(sub.metadata["record"])
    r = json.loads(rec.read_text())
    print(f"    record {rec.name}: job {r.get('job_id')} carrier={r.get('carrier')} image={r.get('image')} "
          f"image_job={r.get('image_job_id')} pull_allowance={r.get('image_pull_allowance_s')} duration={r.get('duration_s')}")
    while True:
        time.sleep(60)
        c = firecrown_collect_chain(output_dir=str(out), pending=pending)
        if not c.metadata.get("pending"):
            break
        print(f"    {time.strftime('%H:%M:%S')} {c.metadata.get('state')}: {c.message[:120]}")
    dt = time.time() - t0
    m = c.metadata
    print(f"[3] collected after {dt:.0f}s on {m.get('computed_on')}: {c.message[:600]}")
    chains = sorted(f for f in m.get("chain_files", []) if f.rsplit(".", 2)[-1] == "txt" and f.rsplit(".", 2)[-2].isdigit())
    r1 = m.get("Rminus1")
    ok = True
    for label, cond in (
        (f"n_chains == {expected}", m.get("n_chains") == expected),
        (f"{expected} per-chain sample counts", len(m.get("chain_samples") or []) == expected),
        (f"{expected} chain files <prefix>.k.txt", len(chains) == expected),
        ("R-1 is a finite number", isinstance(r1, (int, float)) and math.isfinite(r1)),
        ("summary has both parameters", set(m.get("sampled") or []) == {"sigma8", "Omega_c"}),
        ("trace plot made", any(str(f).endswith(".png") for f in (c.files or []))),
    ):
        print(f"    {'ok' if cond else 'FAIL'}: {label}")
        ok = ok and bool(cond)
    r = json.loads(rec.read_text())
    print(f"[4] record: {json.dumps({k: r.get(k) for k in ('job_id', 'carrier', 'image', 'image_job_id', 'image_pull_allowance_s', 'duration_s', 'nodes', 'state', 'runtime_s', 'queued_s')})}")
    print("[5] summary:", json.dumps(m.get("summary"), indent=1))
    print("PASS" if ok else "FAIL - see the checks above")
    return 0 if ok else 1


if __name__ == "__main__":
    site = sys.argv[1] if len(sys.argv) > 1 else "perlmutter"
    nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    sys.exit(main(site, nodes, max_samples))
