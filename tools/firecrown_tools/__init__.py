"""firecrown likelihood tools: build a TwoPointExperiment from a sacc file,
evaluate log-likelihoods, theory vectors, 1-D profiles and Cobaya chains.

Computation goes through env-kernel inner scripts (tools/inner/
firecrown_loglike.py, firecrown_theory.py, firecrown_chain.py) so the SAME
code runs in-process locally and under a facility environment remotely
(hep-genesis dispatch, env_setup chosen by the client). Wrappers here only
validate, write CSV/PNG/YAML and shape metadata.

Parameter names are firecrown's: cosmology Omega_c Omega_b h n_s sigma8|A_s
Omega_k Neff m_nu w0 wa T_CMB; per-bin nuisance {tracer}_{param}
(src0_delta_z, src0_mult_bias, lens0_bias, lens0_delta_z, ...); global IA
ia_bias alphaz z_piv.
"""

import contextlib
import sys
import pathlib
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, validate_call

from ..common import (ArtifactResult, as_float_dict, clone_dir, get_cached,
                      param_slug, resolve_outdir, write_csv, write_json)
from ..cosmology import CosmologyParams
from ..inner.firecrown_chain import DISPATCH_MPI as _CHAIN_MPI

__all__ = ["firecrown_list_examples", "firecrown_build_likelihood", "firecrown_compute_loglike",
           "firecrown_theory_data_vector", "firecrown_scan_loglike", "firecrown_run_chain",
           "firecrown_chain_status", "firecrown_plot_chain", "firecrown_chain_cancel",
           "firecrown_collect_chain"]

CONVENTIONS = {
    "parameter_names": "firecrown == pyccl: Omega_c Omega_b h n_s sigma8|A_s Omega_k Neff m_nu[eV] w0 wa T_CMB; "
                       "nuisance {tracer}_{param}: src0_delta_z src0_mult_bias lens0_bias lens0_delta_z "
                       "lens0_sigma_z lens0_alphaz lens0_mag_bias; global IA: ia_bias alphaz z_piv",
    "experiment_yaml": "firecrown TwoPointExperiment YAML (data_source + two_point_factory + ccl_factory); "
                       "sacc path stored absolute; produced by firecrown_build_likelihood",
    "ccl_mode": "pure_ccl_mode: CCL computes background and P(k) itself (no CAMB/CLASS sampler block)",
    "lifecycle": "every evaluation is tools.update -> tools.prepare -> like.update -> compute -> like.reset; tools.reset",
    "loglike": "ConstGaussian: loglike = -chi2/2 with the sacc covariance (dense, must be positive definite)",
    "units": "theta arcmin, ell integer, C_ell dimensionless, xi dimensionless",
    "cache": "the (likelihood, tools) pair per experiment YAML is cached in-process (reloaded if the YAML changes)",
}
CAVEATS = [
    "Every sampler parameter must be present (ParamsMap): the tools fill firecrown defaults for anything you "
    "do not pass, so check metadata.required_parameters and metadata.unused_keys (keys you passed that the "
    "likelihood does not use - typos are silent otherwise).",
    "MultiplicativeShearBias defaults to 1.0 (!) in firecrown; DES-like analyses use ~0.01. "
    "NumberCounts bias defaults to 1.5. Pass nuisance values explicitly.",
    "sigma8 vs A_s is fixed at build time (amplitude_parameter); passing the other name is reported as unused.",
    "m_nu is the summed neutrino mass in eV (0 = massless); firecrown 1.16 shows its default as [] internally.",
    "firecrown_run_chain is a Cobaya MCMC with CCL in the loop (~0.4 s per DES-Y1 3x2pt evaluation): "
    "hundreds of samples locally in the foreground, background=True for longer local runs, dispatch "
    "(env_setup) for real chains. Its walltime_s (default ~5 s/sample, capped at the site's prompt-queue "
    "walltime: Polaris 60 min below 10 nodes / 3 h at 10-24 nodes, Perlmutter 12 h) is what the facility "
    "job requests; continue a chain that stopped at its walltime with resume=True.",
    "w0-wa chains: CAMB's fluid dark energy cannot cross w = -1; firecrown_run_chain switches to the PPF "
    "model automatically when wa is sampled (dark_energy_model='auto'), writing a *_ppf.yaml experiment copy.",
    "TATT / PT bias / halo-model IA factories need a pt_calculator/hm_calculator in ModelingTools and are not "
    "exposed here (they are not expressible in a plain TwoPointExperiment YAML).",
    "Remote runs: the experiment YAML and its sacc file are shipped inline (sacc <= 8 MB) unless you give "
    "sacc_remote_path (a facility path).",
]

_EXAMPLE_FIDUCIAL_DES_Y1 = {
    "lens0_bias": 1.4, "lens1_bias": 1.6, "lens2_bias": 1.6, "lens3_bias": 1.9, "lens4_bias": 2.0,
    "lens0_delta_z": 0.001, "lens1_delta_z": 0.002, "lens2_delta_z": 0.001, "lens3_delta_z": 0.003,
    "lens4_delta_z": 0.0,
    "src0_delta_z": -0.001, "src1_delta_z": -0.019, "src2_delta_z": 0.009, "src3_delta_z": -0.018,
    "src0_mult_bias": 0.012, "src1_mult_bias": 0.012, "src2_mult_bias": 0.012, "src3_mult_bias": 0.012,
    "ia_bias": 0.5, "alphaz": 0.0, "z_piv": 0.62,
}

WL_PER_BIN = ("MultiplicativeShearBiasFactory", "PhotoZShiftFactory", "PhotoZShiftandStretchFactory")
WL_GLOBAL = ("LinearAlignmentSystematicFactory", "PhotoZShiftFactory", "PhotoZShiftandStretchFactory")
NC_PER_BIN = ("PhotoZShiftFactory", "PhotoZShiftandStretchFactory", "LinearBiasSystematicFactory",
              "MagnificationBiasSystematicFactory", "ConstantMagnificationBiasSystematicFactory")
NC_GLOBAL = ("PhotoZShiftFactory", "PhotoZShiftandStretchFactory")
_PARAMS_BY_FACTORY = {
    "MultiplicativeShearBiasFactory": "{tracer}_mult_bias (default 1.0; DES ~0.012)",
    "PhotoZShiftFactory": "{tracer}_delta_z (0.0)",
    "PhotoZShiftandStretchFactory": "{tracer}_delta_z (0.0), {tracer}_sigma_z (1.0)",
    "LinearAlignmentSystematicFactory": "ia_bias (0.5), alphaz (0.0), z_piv (0.5); alphag fixed by ia_alphag",
    "LinearBiasSystematicFactory": "{tracer}_alphaz (0.0), {tracer}_alphag (0.0), {tracer}_z_piv (0.5) on top of {tracer}_bias",
    "MagnificationBiasSystematicFactory": "{tracer}_r_lim, _sig_c, _eta, _z_c, _z_m",
    "ConstantMagnificationBiasSystematicFactory": "{tracer}_mag_bias (1.0)",
}

DISPATCH_KERNELS = {
    "firecrown_loglike": {
        "function": "inner.firecrown_loglike.main", "inner": "firecrown_loglike",
        "args_shape": "{'params': <the params object below>}; the pack's env/conda-lock.yml supplies "
                      "firecrown - no env_setup, no pip_deps (override: function='envkernel.run_in_env')",
        "params": {"experiment_yaml": "<path or inline via experiment_yaml_text + sacc_b64/sacc_remote_path>",
                   "points": [{"Omega_c": 0.25, "sigma8": 0.81, "lens0_bias": 1.4}],
                   "return_vectors": True, "per_statistic": True},
        "env_setup_required": False, "suitable_envs": ["desc-cosmology", "desc-python"],
        "duration_hint_s": 1800,
        "returns": "required params with defaults, n_data, per-point loglike/chi2/per-statistic chi2, theory/data vectors",
    },
    "firecrown_theory": {
        "function": "inner.firecrown_theory.main", "inner": "firecrown_theory",
        "args_shape": "{'params': <the params object below>}; the pack's env/conda-lock.yml supplies "
                      "firecrown - no env_setup, no pip_deps (override: function='envkernel.run_in_env')",
        "params": {"experiment_yaml": "<path>", "params": {}, "write_sacc": True, "add_noise": False,
                   "seed": None, "sacc_output": "theory_realization.hdf5"},
        "env_setup_required": False, "suitable_envs": ["desc-cosmology", "desc-python"],
        "duration_hint_s": 1800,
        "returns": "theory/data/sigma per statistic and the realization sacc path (job CWD)",
    },
    "firecrown_chain": {
        "function": "inner.firecrown_chain.main", "inner": "firecrown_chain",
        # mirror of tools/inner/firecrown_chain.DISPATCH_MPI (contract R9)
        "mpi": {**_CHAIN_MPI["main"],
                "note": "nodes=N launches N x tasks_per_node MPI processes, one Cobaya chain each "
                        "(same max_samples per chain); R-1 is then the cross-chain statistic and the "
                        "result combines all chains. Walltime is per chain, charge is nodes x walltime; "
                        "resume needs the same chain count."},
        "args_shape": "{'params': <the params object below>}; the pack's env/conda-lock.yml supplies "
                      "firecrown - no env_setup, no pip_deps (override: function='envkernel.run_in_env')",
        "params": {"experiment_yaml": "<path>", "fixed": {}, "priors": {"sigma8": {"min": 0.6, "max": 1.0}},
                   "max_samples": 2000, "rminus1_stop": 0.05, "work_dir": ".", "chain_prefix": "chain",
                   "resume": False, "file_locking": False},
        "env_setup_required": False, "suitable_envs": ["desc-cosmology", "desc-python"],
        "duration_hint_s": 7200,
        "duration_note": "request ~5 s per sample as the job walltime (duration=), at most the facility's "
                         "long-queue cap (Perlmutter regular 12 h); pass the same value minus 60 s as "
                         "timeout_s; continue a chain that hit its walltime with resume=True (and the same "
                         "nodes)",
        "returns": "Cobaya chain files (job CWD) + weighted means/stds/68% limits per sampled parameter",
    },
}

_INLINE_SACC_MAX = 8 * 1024 * 1024


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _exp_path(experiment_yaml: str) -> Path:
    p = Path(experiment_yaml).expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"experiment YAML not found: {p} (create one with firecrown_build_likelihood)")
    return p


def _sacc_path_from_yaml(exp_path: Path) -> Path:
    import yaml

    with exp_path.open(encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    try:
        sp = doc["data_source"]["sacc_data_file"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{exp_path} is not a TwoPointExperiment YAML (no data_source.sacc_data_file)") from exc
    p = Path(sp)
    if not p.is_absolute():
        p = exp_path.parent / p
    return p


def _load_local(exp_path: Path):
    """(likelihood, tools, sacc) cached per YAML path + mtime via common.get_cached;
    the inner module's own cache is kept in sync."""
    from ..inner import firecrown_loglike as inner

    key = f"firecrown_exp_{exp_path}_{exp_path.stat().st_mtime_ns}"

    def _factory():
        inner._CACHE.pop(str(exp_path), None)
        return inner.load_experiment(str(exp_path))

    like_tools = get_cached(key, _factory)
    inner._CACHE[str(exp_path)] = like_tools
    return like_tools


def _remote_params(exp_path: Path, sacc_remote_path: str | None) -> dict:
    """Ship the YAML (and, unless sacc_remote_path is given, the sacc file) inline."""
    import base64

    text = exp_path.read_text(encoding="utf-8")
    out = {"experiment_yaml_text": text}
    if sacc_remote_path:
        out["sacc_remote_path"] = sacc_remote_path
    else:
        sp = _sacc_path_from_yaml(exp_path)
        if not sp.is_file():
            raise ValueError(f"sacc file referenced by {exp_path} not found: {sp}")
        size = sp.stat().st_size
        if size > _INLINE_SACC_MAX:
            raise ValueError(f"{sp.name} is {size / 1e6:.1f} MB (> 8 MB): stage it on the facility yourself "
                             "and pass sacc_remote_path.")
        out["sacc_b64"] = base64.b64encode(sp.read_bytes()).decode("ascii")
        out["sacc_name"] = sp.name
    return out


def _localize_files(result: dict, artifact_files: list[str] | None) -> dict:
    """Map the remote paths a facility kernel reports (chain_files, e.g.
    './chain.1.txt' in the job dir) to the local copies the engine fetched
    back (artifact_files, by file name). Without this a facility chain had no
    local chain_txt, so no trace plot and nothing for firecrown_plot_chain."""
    if not isinstance(result, dict) or not artifact_files:
        return result
    local = {Path(f).name: f for f in artifact_files}
    if isinstance(result.get("chain_files"), list):
        result["chain_files"] = [local.get(Path(f).name, f) for f in result["chain_files"]]
    return result


def _remote_result(res: dict, site: str, nodes: int = 1) -> tuple[dict, str]:
    """(kernel result with local file paths, computed_on) from an engine dict."""
    host = res.get("host", site)
    mpi = res.get("mpi") or {}
    if mpi.get("size"):
        host = f"{host} ({mpi['size']} MPI ranks on {len(mpi.get('nodes') or []) or nodes} nodes)"
    return _localize_files(res["result"], res.get("artifact_files")), host


def _run_inner(inner_name: str, params: dict, env_setup: str | None, duration: int,
               exp_path: Path, sacc_remote_path: str | None = None, nodes: int = 1,
               wait: bool = True) -> tuple[dict, str]:
    """Local in-process call or remote env-kernel; returns (result, computed_on).

    ``nodes`` > 1 (facility only) runs the kernel as an MPI job over that many
    nodes; the kernel must declare DISPATCH_MPI (firecrown_chain does) or the
    engine refuses before staging. ``wait=False`` (facility, lock env only)
    returns the engine's submission handle ``{"status": "submitted", ...}``
    instead of the result.
    """
    from mcp_server.dispatch import remote_site, run_env_kernel, run_kernel  # lazy: server-only

    site = remote_site()
    if site:
        p = dict(params)
        p.pop("experiment_yaml", None)
        p.update(_remote_params(exp_path, sacc_remote_path))
        if env_setup and env_setup.strip():
            if nodes > 1:
                raise ValueError("nodes > 1 runs the kernel as MPI processes inside the pack's own lock "
                                 "environment (mpi4py is built there); it cannot be combined with env_setup.")
            if not wait:
                raise ValueError("wait=False runs in the pack's own lock environment; drop env_setup.")
            # override: a facility-resident environment named by the client
            res = run_env_kernel(env_setup, inner_name, p, duration=duration)
            return _localize_files(res["result"]["result"], res.get("artifact_files")), res.get("host", site)
        # default: the pack's own environment (tools/env/conda-lock.yml), built
        # on the node by the engine; the inner script runs as a plain kernel
        res = run_kernel(f"inner.{inner_name}.main", {"params": p}, pip_deps=None, duration=duration,
                         nodes=nodes, wait=wait)
        if res.get("status") == "submitted":
            return res, site
        return _remote_result(res, site, nodes)
    if not wait:
        raise ValueError("wait=False applies to facility runs (set_dispatch first); local long runs use "
                         "background=True.")
    if nodes > 1:
        raise ValueError("nodes > 1 applies to facility runs: call set_dispatch('perlmutter' or 'polaris') "
                         "first. Local runs are one chain (use background=True for long ones).")
    import importlib

    _load_local(exp_path)  # warm/refresh the in-process cache
    mod = importlib.import_module(f"tools.inner.{inner_name}")
    # stdout -> stderr: keeps an MCP stdio transport clean without hiding
    # Cobaya's logging (redirecting into a StringIO makes Cobaya exit silently)
    with contextlib.redirect_stdout(sys.stderr):
        return mod.main(dict(params, experiment_yaml=str(exp_path))), "local"


# firecrown's internal defaults for some nuisance parameters are NOT "no
# systematic" (MultiplicativeShearBias defaults to 1.0, i.e. shear x2; the NLA
# amplitude ia_bias to 0.5). Unspecified parameters matching these suffixes
# therefore take physically neutral values instead, so "evaluate at the
# cosmology" means what it says. Galaxy bias ({tracer}_bias) keeps firecrown's
# 1.5 - zero would be unphysical - and is reported as a firecrown default.
NEUTRAL_NUISANCE = {"_mult_bias": 0.0, "ia_bias": 0.0, "_delta_z": 0.0, "_sigma_z": 1.0}


def _neutral_value(name: str):
    if name == "ia_bias":
        return 0.0
    for suffix, value in NEUTRAL_NUISANCE.items():
        if suffix.startswith("_") and name.endswith(suffix):
            return value
    return None


def _experiment_modelling(exp_path) -> dict:
    try:
        import yaml
        doc = yaml.safe_load(pathlib.Path(exp_path).read_text(encoding="utf-8")) or {}
        return dict((doc.get("ccl_factory") or {}))
    except Exception:  # noqa: BLE001
        return {}


def _check_point(required: dict, cosmology: CosmologyParams | None, nuisance: dict,
                 exp_path=None) -> tuple[dict, list[str]]:
    """Merge cosmology + nuisance into one override dict; return (point, warnings)."""
    spec = cosmology or CosmologyParams()
    point = dict(spec.firecrown_params())
    warnings = []
    if exp_path is not None:
        fac = _experiment_modelling(exp_path)
        exp_tf = fac.get("pure_ccl_transfer_function", "boltzmann_camb")
        if spec.transfer_function != exp_tf:
            warnings.append(
                f"transfer_function: the experiment was built with '{exp_tf}' (fixed in the "
                f"experiment YAML) and that is what firecrown uses; the cosmology's "
                f"'{spec.transfer_function}' is ignored. Rebuild with "
                f"firecrown_build_likelihood(transfer_function='{spec.transfer_function}') to match.")
        if spec.matter_power_spectrum != "halofit" or spec.baryons != "none" or spec.mg_mu0 or spec.mg_sigma0:
            warnings.append("firecrown (pure-CCL mode) uses halofit without baryons/MG here; the "
                            "cosmology's matter_power_spectrum/baryons/MG switches are ignored.")
    if "sigma8" in point and "sigma8" not in required and "A_s" in required:
        raise ValueError("this experiment uses amplitude_parameter A_s but the cosmology gives sigma8; "
                         "pass cosmology={'A_s': ..., 'sigma8': None} or rebuild with amplitude_parameter='sigma8'.")
    if "A_s" in point and "A_s" not in required and "sigma8" in required:
        raise ValueError("this experiment uses amplitude_parameter sigma8 but the cosmology gives A_s; "
                         "pass sigma8 or rebuild with amplitude_parameter='A_s'.")
    for k, v in as_float_dict(nuisance or {}).items():
        point[k] = v
    unknown = sorted(k for k in point if k not in required)
    if unknown:
        warnings.append(f"ignored parameters not used by this likelihood: {unknown}")
        for k in unknown:
            point.pop(k)
    missing = sorted(k for k in required if k not in point)
    neutral = {}
    for k in missing:
        v = _neutral_value(k)
        if v is not None:
            point[k] = v
            neutral[k] = v
    if neutral:
        warnings.append(f"neutral defaults applied (no systematic): {neutral}")
    rest = [k for k in missing if k not in neutral]
    if rest:
        warnings.append(f"using firecrown defaults for: {rest}")
    return point, warnings


def _plot_theory_vs_data(path: Path, stats: list[dict], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import PALETTE, rc_params

    by_type: dict[str, list[dict]] = {}
    for st in stats:
        by_type.setdefault(st["data_type"], []).append(st)
    n_types = len(by_type)
    max_panels = max(len(v) for v in by_type.values())
    ncol = min(max_panels, 5)
    rows_per_type = [int(np.ceil(len(v) / ncol)) for v in by_type.values()]
    nrow = sum(rows_per_type)
    with plt.rc_context(rc_params(10)):
        fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol + 1, 2.2 * nrow + 0.8),
                                 squeeze=False, constrained_layout=True)
        for ax in axes.ravel():
            ax.set_visible(False)
        r0 = 0
        for (dtype, group), nr in zip(by_type.items(), rows_per_type):
            for j, st in enumerate(group):
                ax = axes[r0 + j // ncol, j % ncol]
                ax.set_visible(True)
                x = np.asarray(st["x"], float)
                th = np.asarray(st["theory"], float)
                d = np.asarray(st["data"], float)
                sg = np.asarray(st["sigma"], float)
                ax.errorbar(x, d, yerr=sg, fmt="o", ms=2.5, color=PALETTE[0], lw=0.8, label="data")
                ax.plot(x, th, "-", color=PALETTE[1], label="theory")
                ax.set_xscale("log")
                if np.all(th > 0) and np.all(d[np.isfinite(d)] > 0):
                    ax.set_yscale("log")
                ax.set_title(f"{dtype.replace('galaxy_', '')}: {st['tracer1']}-{st['tracer2']}", fontsize=8)
                ax.set_xlabel(r"$\theta$ [arcmin]" if st["x_name"] == "theta_arcmin" else r"$\ell$", fontsize=8)
                ax.tick_params(labelsize=7)
            r0 += nr
        axes[0, 0].legend(fontsize=7, loc="best")
        fig.suptitle(title, fontsize=10)
        fig.savefig(path)
        plt.close(fig)
    _ = n_types


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------
@validate_call
def firecrown_list_examples() -> ArtifactResult:
    """List firecrown's maintained examples (`firecrown examples` CLI), the offline test data vector, and the parameter-name conventions.

    Use the offline file (firecrown/tests/sacc_data.hdf5 in the firecrown
    clone: DES-Y1-like 3x2pt, real space, tracers src0-3 / lens0-4, 457
    points, dense covariance) for any demo: it needs no download. Example
    nuisance fiducials for it are in metadata.des_y1_fiducial (biases,
    delta_z, mult_bias, IA). Then: firecrown_build_likelihood ->
    firecrown_compute_loglike -> firecrown_scan_loglike.
    """
    examples = {
        "cosmic_shear": {"data": "synthetic (generated with pyccl, offline)", "space": "harmonic",
                         "statistics": "galaxy_shear_cl_ee for all pairs of trc{i} Gaussian n(z) bins",
                         "systematics": "PhotoZShift per bin", "params": "trc{i}_delta_z",
                         "cli": "firecrown examples cosmic_shear OUT --n-bins 2 --ell-min 10 --ell-max 10000"},
        "cmb_cross": {"data": "synthetic NZ + CMB-lensing Map tracer (z_lss 1100), offline", "space": "harmonic",
                      "statistics": "galaxy_shear_cl_ee + cmbGalaxy_convergenceShear_cl_e ...",
                      "systematics": "PhotoZShift per bin", "params": "trc{i}_delta_z"},
        "sn_srd": {"data": "download srd-y1.sacc (firecrown GitHub release files-v1.0.0)", "space": "n/a",
                   "statistics": "Supernova distance moduli (tracer sn_ddf_sample)", "params": "sn_ddf_sample_M (~-19.4)",
                   "note": "uses amplitude_parameter A_s, no P(k) needed"},
        "des_y1_3x2pt": {"data": "download des_y1_3x2pt.sacc (release files-v1.1.0): real DES Y1 3x2pt, 457 points",
                         "space": "real", "statistics": "xi+ (10), xi- (10), gamma_t (20), w(theta) (5) over src0-3 / lens0-4",
                         "systematics": "WL: MultiplicativeShearBias + PhotoZShift per bin, LinearAlignment global; "
                                        "NC: PhotoZShift per bin (+ bias)",
                         "params": "ia_bias alphaz z_piv lens{i}_bias lens{i}_delta_z src{i}_delta_z src{i}_mult_bias",
                         "factory_types": "standard|pt|tatt|hmia|pk_modifier|yaml_default|yaml_pure_ccl|yaml_mu_sigma"},
    }
    fc = clone_dir("firecrown")
    test_file = str(fc / "tests" / "sacc_data.hdf5") if fc and (fc / "tests" / "sacc_data.hdf5").is_file() else None
    try:
        import firecrown
        version = firecrown.__version__
    except Exception:  # noqa: BLE001
        version = None
    return ArtifactResult(
        status="success", files=[],
        message=(f"4 firecrown examples; offline test data vector: {test_file or 'not available (no firecrown clone)'}. "
                 f"firecrown {version} installed."),
        metadata={"examples": examples, "offline_test_sacc": test_file,
                  "offline_test_description": "DES-Y1-like 3x2pt real space; NZ tracers lens0-4, src0-3 (400 z points); "
                                              "galaxy_shear_xi_plus 167, xi_minus 60, galaxy_shearDensity_xi_t 176, "
                                              "galaxy_density_xi 54; dense 457x457 covariance",
                  "des_y1_fiducial": _EXAMPLE_FIDUCIAL_DES_Y1,
                  "parameter_conventions": CONVENTIONS["parameter_names"],
                  "systematics_factories": {"weak_lensing_per_bin": list(WL_PER_BIN), "weak_lensing_global": list(WL_GLOBAL),
                                            "number_counts_per_bin": list(NC_PER_BIN), "number_counts_global": list(NC_GLOBAL),
                                            "parameters": _PARAMS_BY_FACTORY},
                  "firecrown_version": version},
    )


_MEAS_ALIASES = {
    "shear_e": ["SHEAR_E"], "shear_t": ["SHEAR_T"], "xi_plus": ["PART_OF_XI_PLUS"],
    "xi_minus": ["PART_OF_XI_MINUS"], "counts": ["COUNTS"], "density": ["COUNTS"],
}


def _measurements(alias: str, space: str) -> list[str]:
    a = alias.strip()
    if a.lower() == "shear":
        return ["SHEAR_E"] if space == "harmonic" else ["PART_OF_XI_PLUS", "PART_OF_XI_MINUS", "SHEAR_T"]
    if a.lower() in _MEAS_ALIASES:
        return _MEAS_ALIASES[a.lower()]
    if a.upper() in ("SHEAR_E", "SHEAR_T", "PART_OF_XI_PLUS", "PART_OF_XI_MINUS", "COUNTS"):
        return [a.upper()]
    raise ValueError(f"unknown measurement {alias!r}; use shear|density|shear_e|shear_t|xi_plus|xi_minus|counts")


@validate_call
def firecrown_build_likelihood(
    output_dir: Annotated[str, Field(min_length=1)],
    sacc_path: Annotated[str, Field(min_length=1, description="sacc file with src{i}/lens{i} NZ tracers and a covariance (sacc_prepare_for_firecrown / sacc_attach_gaussian_covariance first if needed).")],
    correlation_space: Annotated[Literal["auto", "real", "harmonic"], Field(description="'auto' detects from the data types (xi -> real, cl -> harmonic).")] = "auto",
    wl_per_bin: Annotated[list[str], Field(description=f"Per-bin weak-lensing systematics factories: {WL_PER_BIN}.")] = ["MultiplicativeShearBiasFactory", "PhotoZShiftFactory"],
    wl_global: Annotated[list[str], Field(description=f"Global weak-lensing systematics: {WL_GLOBAL} (LinearAlignment = NLA IA with ia_bias, alphaz, z_piv).")] = ["LinearAlignmentSystematicFactory"],
    nc_per_bin: Annotated[list[str], Field(description=f"Per-bin number-counts systematics: {NC_PER_BIN} ({{tracer}}_bias is always present).")] = ["PhotoZShiftFactory"],
    nc_global: Annotated[list[str], Field(description=f"Global number-counts systematics: {NC_GLOBAL}.")] = [],
    include_rsd: Annotated[bool, Field(description="Redshift-space distortions in the number-counts tracer.")] = False,
    ia_alphag: Annotated[float | None, Field(description="Fixed growth exponent alphag of the linear-alignment model (1.0 = DES/firecrown default); None makes it a free parameter 'alphag'.")] = 1.0,
    require_nonlinear_pk: Annotated[bool, Field(description="Use the nonlinear P(k) (halofit); required for 3x2pt.")] = True,
    amplitude_parameter: Annotated[Literal["sigma8", "A_s"], Field(description="Which amplitude parameter the likelihood samples.")] = "sigma8",
    transfer_function: Literal["boltzmann_camb", "bbks", "eisenstein_hu", "eisenstein_hu_nowiggles", "boltzmann_class"] = "boltzmann_camb",
    scale_cuts: Annotated[list[dict] | None, Field(description="Keep-ranges per tracer: [{'tracer': 'src0', 'measurement': 'shear', 'lower': 10, 'upper': 250}] (theta arcmin or ell). measurement: shear|density|shear_e|shear_t|xi_plus|xi_minus|counts; optional 'tracer2'/'measurement2' restrict to one pair. Points outside [lower, upper] are dropped.")] = None,
    allow_empty_bins: Annotated[bool, Field(description="Allow a scale cut to remove an entire tracer pair (default True).")] = True,
    dark_energy_model: Annotated[Literal["fluid", "ppf"], Field(description="CAMB dark-energy model written into ccl_factory.camb_extra_params: 'fluid' (CAMB default; w(a) must not cross -1) or 'ppf' (needed for w0-wa chains whose w(a) can cross -1). firecrown_run_chain switches to ppf by itself when needed.")] = "fluid",
    name: Annotated[str, Field(min_length=1, description="Stem of the YAML file.")] = "experiment",
) -> ArtifactResult:
    """Turn a sacc file + systematics choices into a validated firecrown experiment YAML and list every parameter the likelihood needs.

    Writes <name>_<slug>.yaml (firecrown TwoPointExperiment: data_source
    with absolute sacc path + optional scale-cut filters, two_point_factory
    with the systematics factories you name, ccl_factory in pure_ccl_mode)
    and loads it once to report: n_data (after cuts), the statistics
    (data type, tracer pair, n points) and metadata.required_parameters -
    every sampler parameter with its firecrown default. Those names are
    what firecrown_compute_loglike / scan / chain accept as `nuisance`
    (cosmology comes from CosmologyParams). Check the defaults: mult_bias
    defaults to 1.0 and lens bias to 1.5 unless you set them.

    Systematic factories are given by firecrown type name (see
    firecrown_list_examples metadata.systematics_factories). Not exposed:
    TATT / PT bias / halo-model IA (need PT or halo-model calculators).
    """
    from ..sacc_tools import _load as _load_sacc

    outdir = resolve_outdir(output_dir)
    spath = Path(sacc_path).expanduser().resolve()
    s = _load_sacc(str(spath))
    if s.covariance is None:
        raise ValueError(f"{spath.name} has no covariance: firecrown's ConstGaussian needs one. Harmonic files: "
                         "sacc_attach_gaussian_covariance; real-space: TJPCov/augur.")
    dtypes = s.get_data_types()
    if not dtypes:
        raise ValueError("the sacc file has no data points.")
    detected = "harmonic" if all("_cl" in d for d in dtypes) else "real" if all("_xi" in d for d in dtypes) else "mixed"
    if correlation_space == "auto":
        if detected == "mixed":
            raise ValueError(f"mixed harmonic/real data types {dtypes}: drop one family with "
                             "sacc_prepare_for_firecrown(keep_data_types=...) or set correlation_space.")
        correlation_space = detected
    elif detected not in ("mixed", correlation_space):
        raise ValueError(f"correlation_space={correlation_space} but the file holds {detected}-space data {dtypes}.")
    bad = [(f, WL_PER_BIN) for f in wl_per_bin if f not in WL_PER_BIN] + \
          [(f, WL_GLOBAL) for f in wl_global if f not in WL_GLOBAL] + \
          [(f, NC_PER_BIN) for f in nc_per_bin if f not in NC_PER_BIN] + \
          [(f, NC_GLOBAL) for f in nc_global if f not in NC_GLOBAL]
    if bad:
        raise ValueError("unknown/unsupported systematics factory " +
                         "; ".join(f"{f!r} (allowed: {list(ok)})" for f, ok in bad))
    tracer_names = list(s.tracers)
    has_shear = any(d.startswith("galaxy_shear") for d in dtypes)
    has_density = any(d.startswith("galaxy_density") or "Density" in d for d in dtypes)

    def _sys(f):
        if f == "LinearAlignmentSystematicFactory":
            return {"type": f, "alphag": ia_alphag}
        return {"type": f}

    exp: dict = {
        "data_source": {"sacc_data_file": str(spath)},
        "two_point_factory": {
            "correlation_space": correlation_space,
            "weak_lensing_factories": ([{"type_source": "default", "per_bin_systematics": [_sys(f) for f in wl_per_bin],
                                         "global_systematics": [_sys(f) for f in wl_global]}] if has_shear else []),
            "number_counts_factories": ([{"type_source": "default", "per_bin_systematics": [_sys(f) for f in nc_per_bin],
                                          "global_systematics": [_sys(f) for f in nc_global],
                                          "include_rsd": include_rsd}] if has_density else []),
        },
        "ccl_factory": {"creation_mode": "pure_ccl_mode", "require_nonlinear_pk": require_nonlinear_pk,
                        "amplitude_parameter": amplitude_parameter,
                        "pure_ccl_transfer_function": transfer_function},
    }
    if dark_energy_model != "fluid":
        exp["ccl_factory"]["camb_extra_params"] = {"dark_energy_model": dark_energy_model}
    cuts_applied = []
    if scale_cuts:
        filters = []
        for c in scale_cuts:
            for key in ("tracer", "measurement", "lower", "upper"):
                if key not in c:
                    raise ValueError(f"scale cut {c} needs keys tracer, measurement, lower, upper")
            if c["tracer"] not in tracer_names:
                raise ValueError(f"scale cut tracer {c['tracer']!r} not in file; tracers: {tracer_names}")
            lo, hi = float(c["lower"]), float(c["upper"])
            if not lo < hi:
                raise ValueError(f"scale cut {c}: lower must be < upper")
            m1 = _measurements(str(c["measurement"]), correlation_space)
            if "tracer2" in c:
                if c["tracer2"] not in tracer_names:
                    raise ValueError(f"scale cut tracer2 {c['tracer2']!r} not in file")
                m2 = _measurements(str(c.get("measurement2", "density")), correlation_space)
                for a in m1:
                    for b in m2:
                        filters.append({"spec": [{"name": c["tracer"], "measurement": {"subject": "Galaxies", "property": a}},
                                                 {"name": c["tracer2"], "measurement": {"subject": "Galaxies", "property": b}}],
                                        "interval": [lo, hi], "method": "support"})
            else:
                for a in m1:
                    filters.append({"spec": [{"name": c["tracer"], "measurement": {"subject": "Galaxies", "property": a}}],
                                    "interval": [lo, hi], "method": "support"})
            cuts_applied.append({"tracer": c["tracer"], "measurements": m1, "tracer2": c.get("tracer2"),
                                 "keep": [lo, hi]})
        exp["data_source"]["filters"] = {"require_filter_for_all": False, "allow_empty": allow_empty_bins,
                                         "filters": filters}

    import yaml

    slug = param_slug({"sacc": str(spath), "exp": repr(exp)})
    ypath = outdir / f"{name}_{slug}.yaml"
    ypath.write_text(yaml.safe_dump(exp, sort_keys=False), encoding="utf-8")

    # validate by loading (pydantic strict) and building the likelihood once
    try:
        like, tools, _ = _load_local(ypath)
    except Exception as exc:
        ypath.unlink(missing_ok=True)
        raise ValueError(f"firecrown rejected the experiment: {type(exc).__name__}: {exc}") from exc
    from ..inner import firecrown_loglike as inner

    required = inner.required_defaults(like, tools)
    layout = inner.statistic_layout(like, _load_local(ypath)[2])
    n_data = int(sum(st["n"] for st in layout))
    n_total = len(s.data)
    stats = [{"data_type": st["data_type"], "tracer1": st["tracer1"], "tracer2": st["tracer2"], "n": st["n"]}
             for st in layout]
    cosmo_keys = [k for k in required if k in ("Omega_c", "Omega_b", "h", "n_s", "sigma8", "A_s", "Omega_k",
                                                "Neff", "m_nu", "w0", "wa", "T_CMB")]
    nuis = {k: v for k, v in required.items() if k not in cosmo_keys}
    jpath = outdir / f"{name}_{slug}_params.json"
    write_json(jpath, {"experiment_yaml": str(ypath), "required_parameters": required,
                       "cosmology_parameters": cosmo_keys, "nuisance_parameters": nuis,
                       "n_data": n_data, "statistics": stats})
    return ArtifactResult(
        status="success", files=[str(ypath), str(jpath)],
        message=(f"Experiment YAML {ypath.name}: {correlation_space} space, {len(stats)} statistics, n_data={n_data}"
                 + (f" (of {n_total} after scale cuts)" if n_data != n_total else "")
                 + f"; {len(required)} parameters required ({len(cosmo_keys)} cosmology + {len(nuis)} nuisance). "
                 "Pass nuisance values to firecrown_compute_loglike; defaults listed in metadata.required_parameters."),
        metadata={"experiment_yaml": str(ypath), "sacc_path": str(spath), "correlation_space": correlation_space,
                  "n_data": n_data, "n_data_total": n_total, "statistics": stats,
                  "required_parameters": required, "cosmology_parameters": cosmo_keys,
                  "nuisance_parameters": nuis, "scale_cuts": cuts_applied,
                  "systematics": {"wl_per_bin": wl_per_bin, "wl_global": wl_global, "nc_per_bin": nc_per_bin,
                                  "nc_global": nc_global, "parameters_by_factory": _PARAMS_BY_FACTORY},
                  "ccl_factory": exp["ccl_factory"]},
    )


@validate_call
def firecrown_compute_loglike(
    output_dir: Annotated[str, Field(min_length=1)],
    experiment_yaml: Annotated[str, Field(min_length=1, description="From firecrown_build_likelihood.")],
    cosmology: Annotated[CosmologyParams | None, Field(description="Cosmology (None = vanilla LCDM: Omega_c 0.25, Omega_b 0.05, h 0.67, n_s 0.96, sigma8 0.81).")] = None,
    nuisance: Annotated[dict[str, float], Field(description="Nuisance values by firecrown name (lens0_bias, src0_delta_z, src0_mult_bias, ia_bias, ...); unspecified mult_bias/ia_bias/delta_z take NEUTRAL values (0; no systematic), other unspecified ones firecrown's defaults (e.g. {tracer}_bias 1.5) - see metadata.warnings.")] = {},
    plot: Annotated[bool, Field(description="Write a theory-vs-data PNG per statistic.")] = True,
    env_setup: Annotated[str | None, Field(description="Remote runs only, OPTIONAL override: a shell snippet activating a facility-resident environment to use INSTEAD of the pack's own lock environment (which the engine builds on the node by default). Leave empty.")] = None,
    sacc_remote_path: Annotated[str | None, Field(description="Remote runs only: facility path of the sacc file (else it is shipped inline).")] = None,
) -> ArtifactResult:
    """Evaluate the firecrown log-likelihood, chi2 and per-statistic chi2 of an experiment at one cosmology + nuisance point.

    Returns loglike (= -chi2/2 for ConstGaussian), chi2, n_data, chi2/n_data
    (expect ~1 at a good fit; the DES-Y1-like test file at the example
    fiducials gives chi2/n ~ 1.2), per-statistic contributions (chi2_block:
    chi2 of that statistic alone with its own covariance block; chi2_share:
    r_i (C^-1 r)_i summed over the block, sums to the total), the list of
    parameters that fell back to defaults, and unused_keys (names you
    passed that the likelihood does not have - a typo detector). Files:
    theory_vs_data_<slug>.csv (statistic, tracers, x, data, theory, sigma,
    pull) and the PNG. Evaluation ~0.4 s for DES-Y1 3x2pt; always local
    unless dispatch is set. Next: firecrown_scan_loglike to profile one
    parameter, firecrown_run_chain for posteriors.
    """
    exp_path = _exp_path(experiment_yaml)
    outdir = resolve_outdir(output_dir)
    like, tools, _ = _load_local(exp_path)
    from ..inner import firecrown_loglike as inner

    required = inner.required_defaults(like, tools)
    point, warns = _check_point(required, cosmology, nuisance, exp_path)
    res, computed_on = _run_inner("firecrown_loglike",
                                  {"points": [point], "return_vectors": True, "per_statistic": True},
                                  env_setup, 1800, exp_path, sacc_remote_path)
    r = res["results"][0]
    if r.get("error") and r.get("loglike") is None:
        raise RuntimeError(f"likelihood evaluation failed: {r['error']}")
    n_data = res["n_data"]
    spec = cosmology or CosmologyParams()
    slug = param_slug({"cosmo": spec.slug(), "nuis": repr(sorted(nuisance.items())), "exp": str(exp_path)})

    # per-point CSV
    stats = res["statistics"]
    pos = 0
    cols = {"statistic_index": [], "x": [], "data": [], "theory": [], "sigma": [], "pull": []}
    rows_meta = []
    for i, st in enumerate(stats):
        n = st["n"]
        th = np.asarray(r["theory"][pos:pos + n])
        d = np.asarray(r["data"][pos:pos + n])
        sg = np.asarray(r["sigma"][pos:pos + n])
        cols["statistic_index"] += [i] * n
        cols["x"] += list(st["x"])
        cols["data"] += d.tolist()
        cols["theory"] += th.tolist()
        cols["sigma"] += sg.tolist()
        cols["pull"] += ((d - th) / np.where(sg > 0, sg, np.nan)).tolist()
        rows_meta.append(f"{i}: {st['data_type']} {st['tracer1']}-{st['tracer2']} ({st['x_name']}, n={n})")
        st["theory"], st["data"], st["sigma"] = th.tolist(), d.tolist(), sg.tolist()
        pos += n
    cpath = outdir / f"theory_vs_data_{slug}.csv"
    write_csv(cpath, {k: np.asarray(v, float) for k, v in cols.items()},
              [f"label: theory vs data ({spec.label()})", "quantity: theory_vs_data",
               f"experiment: {exp_path}", f"loglike: {r['loglike']}", f"chi2: {r['chi2']}", f"n_data: {n_data}",
               "x: theta in arcmin (real) or ell (harmonic); pull = (data - theory)/sigma",
               "statistics: " + " | ".join(rows_meta)])
    files = [str(cpath)]
    if plot:
        ppath = outdir / f"theory_vs_data_{slug}.png"
        _plot_theory_vs_data(ppath, stats, f"{exp_path.stem}: chi2={r['chi2']:.1f}/{n_data} ({spec.label()})")
        files.append(str(ppath))
    per_stat = r.get("per_statistic", [])
    worst = sorted(per_stat, key=lambda p: -(p["chi2_block"] or 0) / max(p["n"], 1))[:5]
    pulls = np.asarray(cols["pull"], float)
    pulls = pulls[np.isfinite(pulls)]
    msg = (f"loglike = {r['loglike']:.3f}, chi2 = {r['chi2']:.2f} for n_data = {n_data} "
           f"(chi2/n = {r['chi2'] / n_data:.3f}) at {spec.label()}"
           + (f", on {computed_on}" if computed_on != "local" else "") + ". "
           + (f"Worst statistics by chi2_block/n: " + ", ".join(
               f"{w['data_type'].replace('galaxy_', '')} {w['tracer1']}-{w['tracer2']} ({w['chi2_block']:.1f}/{w['n']})"
               for w in worst) + ". " if worst else "")
           + (" ".join(warns) if warns else ""))
    return ArtifactResult(
        status="success", files=files, message=msg,
        metadata={"loglike": r["loglike"], "chi2": r["chi2"], "n_data": n_data, "chi2_per_dof": r["chi2"] / n_data,
                  "per_statistic": per_stat, "point": point, "warnings": warns,
                  "unused_keys": r.get("unused_keys", []), "unknown_keys": r.get("unknown_keys", []),
                  "required_parameters": required, "max_abs_pull": float(np.abs(pulls).max()) if pulls.size else None,
                  "rms_pull": float(np.sqrt(np.mean(pulls ** 2))) if pulls.size else None,
                  "computed_on": computed_on, "firecrown_version": res.get("firecrown_version"),
                  "experiment_yaml": str(exp_path), "cosmology": spec.firecrown_params()},
    )


@validate_call
def firecrown_theory_data_vector(
    output_dir: Annotated[str, Field(min_length=1)],
    experiment_yaml: Annotated[str, Field(min_length=1)],
    cosmology: CosmologyParams | None = None,
    nuisance: Annotated[dict[str, float], Field(description="Nuisance values by firecrown name.")] = {},
    write_sacc: Annotated[bool, Field(description="Write a sacc copy whose data vector is the theory (or a noisy draw).")] = True,
    add_noise: Annotated[bool, Field(description="Draw the realization from the covariance (Cholesky) instead of the noiseless theory.")] = False,
    seed: Annotated[int | None, Field(ge=0, description="Seed for the noisy draw.")] = None,
    output_format: Literal["hdf5", "fits"] = "hdf5",
    plot: bool = True,
    env_setup: Annotated[str | None, Field(description="Remote runs only, OPTIONAL override: a facility-resident environment instead of the pack's lock environment. Leave empty.")] = None,
    sacc_remote_path: str | None = None,
) -> ArtifactResult:
    """Compute the theory data vector of an experiment and optionally write it as a new (noiseless or noisy) sacc realization.

    Uses firecrown's make_realization: a copy of the experiment's sacc with
    the modelled points replaced by theory (add_noise=False: a synthetic
    noiseless data vector - the standard forecast input) or by theory +
    Cholesky(C) N(0,1) (add_noise=True; set seed for reproducibility).
    Points removed by scale cuts keep their original data (strict=False).
    Files: theory_vector_<slug>.csv (statistic_index, x, theory, data,
    sigma), the sacc (<name>_theory|noisy_<slug>.hdf5|fits) and a PNG.
    Feed the sacc back to firecrown_build_likelihood to check that the
    likelihood recovers chi2 = 0 (noiseless) or ~n_data (noisy) - the
    standard closure test.
    """
    exp_path = _exp_path(experiment_yaml)
    outdir = resolve_outdir(output_dir)
    like, tools, _ = _load_local(exp_path)
    from ..inner import firecrown_loglike as inner

    required = inner.required_defaults(like, tools)
    point, warns = _check_point(required, cosmology, nuisance, exp_path)
    spec = cosmology or CosmologyParams()
    slug = param_slug({"cosmo": spec.slug(), "nuis": repr(sorted(nuisance.items())), "exp": str(exp_path),
                       "noise": add_noise, "seed": seed})
    tag = "noisy" if add_noise else "theory"
    sacc_out = outdir / f"{exp_path.stem}_{tag}_{slug}.{'fits' if output_format == 'fits' else 'hdf5'}"
    params = {"params": point, "write_sacc": write_sacc, "add_noise": add_noise, "seed": seed,
              "sacc_output": str(sacc_out) if write_sacc else None, "strict": False}
    from mcp_server.dispatch import remote_site

    if remote_site() and write_sacc:
        params["sacc_output"] = sacc_out.name  # job CWD; fetched back as an artifact
    res, computed_on = _run_inner("firecrown_theory", params, env_setup, 1800, exp_path, sacc_remote_path)
    if res.get("error"):
        raise RuntimeError(f"theory evaluation failed: {res['error']}")
    stats = res["statistics"]
    cols = {"statistic_index": [], "x": [], "theory": [], "data": [], "sigma": []}
    rows_meta = []
    for i, st in enumerate(stats):
        cols["statistic_index"] += [i] * st["n"]
        cols["x"] += list(st["x"])
        cols["theory"] += list(st["theory"])
        cols["data"] += list(st["data"])
        cols["sigma"] += list(st["sigma"])
        rows_meta.append(f"{i}: {st['data_type']} {st['tracer1']}-{st['tracer2']} ({st['x_name']}, n={st['n']})")
    cpath = outdir / f"theory_vector_{slug}.csv"
    write_csv(cpath, {k: np.asarray(v, float) for k, v in cols.items()},
              [f"label: theory vector ({spec.label()})", "quantity: theory_vector", f"experiment: {exp_path}",
               f"loglike_at_point: {res['loglike']}", f"chi2_at_point: {res['chi2']}",
               "statistics: " + " | ".join(rows_meta)])
    files = [str(cpath)]
    sacc_file = res.get("sacc_file")
    if write_sacc and sacc_file and Path(sacc_file).is_file():
        files.append(str(Path(sacc_file).resolve()))
    elif write_sacc and sacc_out.is_file():
        files.append(str(sacc_out))
        sacc_file = str(sacc_out)
    if plot:
        ppath = outdir / f"theory_vector_{slug}.png"
        _plot_theory_vs_data(ppath, stats, f"{exp_path.stem}: theory ({spec.label()})")
        files.append(str(ppath))
    th = np.asarray(cols["theory"], float)
    return ArtifactResult(
        status="success", files=files,
        message=(f"Theory vector of {res['n_data']} points at {spec.label()} (chi2 vs the file's data = {res['chi2']:.1f})"
                 + (f"; wrote {tag} realization sacc {Path(sacc_file).name}" if sacc_file else "")
                 + (f", on {computed_on}" if computed_on != "local" else "") + ". " + " ".join(warns)),
        metadata={"n_data": res["n_data"], "loglike_at_point": res["loglike"], "chi2_at_point": res["chi2"],
                  "sacc_file": sacc_file, "add_noise": add_noise, "seed": seed, "point": point, "warnings": warns,
                  "theory_min": float(th.min()), "theory_max": float(th.max()),
                  "statistics": [{k: st[k] for k in ("data_type", "tracer1", "tracer2", "n")} for st in stats],
                  "computed_on": computed_on, "experiment_yaml": str(exp_path)},
    )


def _one_sigma_crossings(x: np.ndarray, dchi2: np.ndarray) -> tuple[float | None, float | None]:
    """Where delta chi2 crosses 1 on each side of the minimum (linear interpolation)."""
    i0 = int(np.nanargmin(dchi2))
    lo = hi = None
    for i in range(i0, 0, -1):
        if dchi2[i - 1] >= 1.0 > dchi2[i]:
            lo = float(x[i] + (1.0 - dchi2[i]) * (x[i - 1] - x[i]) / (dchi2[i - 1] - dchi2[i]))
            break
    for i in range(i0, len(x) - 1):
        if dchi2[i + 1] >= 1.0 > dchi2[i]:
            hi = float(x[i] + (1.0 - dchi2[i]) * (x[i + 1] - x[i]) / (dchi2[i + 1] - dchi2[i]))
            break
    return lo, hi


@validate_call
def firecrown_scan_loglike(
    output_dir: Annotated[str, Field(min_length=1)],
    experiment_yaml: Annotated[str, Field(min_length=1)],
    parameter: Annotated[str, Field(min_length=1, description="Parameter to profile (any required name: sigma8, Omega_c, w0, lens0_bias, ia_bias, ...).")],
    values: Annotated[list[float] | None, Field(description="Explicit grid; or give min/max/n_points.")] = None,
    min_value: float | None = None,
    max_value: float | None = None,
    n_points: Annotated[int, Field(ge=3, le=200)] = 11,
    cosmology: Annotated[CosmologyParams | None, Field(description="Fixed cosmology for the other parameters.")] = None,
    nuisance: Annotated[dict[str, float], Field(description="Fixed nuisance values.")] = {},
    plot: bool = True,
    env_setup: Annotated[str | None, Field(description="Remote runs only, OPTIONAL override: a facility-resident environment instead of the pack's lock environment. Leave empty.")] = None,
    sacc_remote_path: str | None = None,
) -> ArtifactResult:
    """Profile the log-likelihood along one parameter (all others fixed) and report the chi2 minimum and 1-sigma crossing.

    Writes scan_<param>_<slug>.csv (value, loglike, chi2, delta_chi2) and a
    PNG of delta chi2 with the delta chi2 = 1 line. metadata gives the
    grid minimum, a parabolic refinement of the best value and sigma, and
    the delta chi2 = 1 crossings (None when the grid does not bracket
    them: widen the range). This is a conditional (not marginalised)
    profile - the sanity check to run BEFORE any chain: the fiducial
    should sit near the minimum and the curvature should be reasonable.
    Each point is one likelihood evaluation (~0.4 s for DES-Y1 3x2pt);
    dispatchable with env_setup when a remote site is set.
    """
    exp_path = _exp_path(experiment_yaml)
    outdir = resolve_outdir(output_dir)
    like, tools, _ = _load_local(exp_path)
    from ..inner import firecrown_loglike as inner

    required = inner.required_defaults(like, tools)
    if parameter not in required:
        raise ValueError(f"{parameter!r} is not a parameter of this likelihood; required: {sorted(required)}")
    if values is None:
        if min_value is None or max_value is None:
            raise ValueError("give either values=[...] or min_value/max_value (+ n_points).")
        if not min_value < max_value:
            raise ValueError("min_value must be < max_value.")
        grid = np.linspace(min_value, max_value, n_points)
    else:
        grid = np.asarray(sorted(set(float(v) for v in values)), float)
        if len(grid) < 3:
            raise ValueError("values needs at least 3 distinct entries.")
    point, warns = _check_point(required, cosmology, nuisance, exp_path)
    points = [dict(point, **{parameter: float(v)}) for v in grid]
    res, computed_on = _run_inner("firecrown_loglike",
                                  {"points": points, "return_vectors": False, "per_statistic": False},
                                  env_setup, 3600, exp_path, sacc_remote_path)
    ll = np.array([r["loglike"] if r["loglike"] is not None else np.nan for r in res["results"]], float)
    chi2 = np.array([r["chi2"] if r["chi2"] is not None else np.nan for r in res["results"]], float)
    errors = [(float(v), r["error"]) for v, r in zip(grid, res["results"]) if r.get("error")]
    if not np.isfinite(chi2).any():
        raise RuntimeError(f"no finite likelihood on the grid; first error: {errors[0] if errors else 'unknown'}")
    dchi2 = chi2 - np.nanmin(chi2)
    i0 = int(np.nanargmin(chi2))
    lo, hi = _one_sigma_crossings(grid, dchi2)
    # parabolic refinement around the minimum
    best_fit = {"value": float(grid[i0]), "sigma": None}
    if 0 < i0 < len(grid) - 1 and np.isfinite(chi2[i0 - 1:i0 + 2]).all():
        a, b, _c = np.polyfit(grid[i0 - 1:i0 + 2], chi2[i0 - 1:i0 + 2], 2)
        if a > 0:
            best_fit = {"value": float(-b / (2 * a)), "sigma": float(1.0 / np.sqrt(a))}
    spec = cosmology or CosmologyParams()
    slug = param_slug({"p": parameter, "grid": grid.tolist(), "cosmo": spec.slug(),
                       "nuis": repr(sorted(nuisance.items())), "exp": str(exp_path)})
    cpath = outdir / f"scan_{parameter}_{slug}.csv"
    write_csv(cpath, {parameter: grid, "loglike": ll, "chi2": chi2, "delta_chi2": dchi2},
              [f"label: profile of {parameter} ({spec.label()})", "quantity: loglike_scan",
               f"experiment: {exp_path}", f"n_data: {res['n_data']}",
               f"min_chi2: {np.nanmin(chi2)} at {parameter}={grid[i0]}",
               f"one_sigma_crossings: {lo}, {hi}"])
    files = [str(cpath)]
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from ..plotting import PALETTE, param_symbol, rc_params

        with plt.rc_context(rc_params()):
            fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
            ax.plot(grid, dchi2, "-o", color=PALETTE[0], ms=4, label=r"$\Delta\chi^2$")
            ax.axhline(1.0, color=PALETTE[1], ls="--", lw=1.2, label=r"$\Delta\chi^2 = 1$")
            for v in (lo, hi):
                if v is not None:
                    ax.axvline(v, color=PALETTE[2], ls=":", lw=1.0)
            ax.set_xlabel(param_symbol(parameter))
            ax.set_ylabel(r"$\Delta\chi^2$")
            ax.set_ylim(bottom=-0.05 * max(np.nanmax(dchi2), 1.0))
            ax.set_title(f"{exp_path.stem}: profile of {parameter}", fontsize=11)
            ax.legend()
            ppath = outdir / f"scan_{parameter}_{slug}.png"
            fig.savefig(ppath)
            plt.close(fig)
        files.append(str(ppath))
    sig_txt = (f"1-sigma interval [{lo:.4g}, {hi:.4g}]" if lo is not None and hi is not None else
               "delta chi2 = 1 not bracketed on both sides (widen the grid)")
    return ArtifactResult(
        status="success", files=files,
        message=(f"Profiled {parameter} over {len(grid)} points [{grid[0]:g}, {grid[-1]:g}]: min chi2 = "
                 f"{np.nanmin(chi2):.2f} (n_data {res['n_data']}) at {parameter} = {grid[i0]:g}"
                 + (f" (parabolic best {best_fit['value']:.4g} +/- {best_fit['sigma']:.3g})" if best_fit["sigma"] else "")
                 + f"; {sig_txt}" + (f"; {len(errors)} failed points" if errors else "")
                 + (f", on {computed_on}" if computed_on != "local" else "") + ". " + " ".join(warns)),
        metadata={"parameter": parameter, "grid": grid.tolist(), "loglike": ll.tolist(), "chi2": chi2.tolist(),
                  "delta_chi2": dchi2.tolist(), "min_chi2": float(np.nanmin(chi2)), "n_data": res["n_data"],
                  "grid_minimum": float(grid[i0]), "parabolic_best_fit": best_fit,
                  "one_sigma_lower": lo, "one_sigma_upper": hi, "errors": errors, "warnings": warns,
                  "fixed_point": point, "computed_on": computed_on, "experiment_yaml": str(exp_path)},
    )


firecrown_scan_loglike.weight = "dispatchable"


# Cap on a DEFAULT walltime when no site is known (local runs; Perlmutter's
# value in the engine). On a facility the engine's queue policy decides: the
# prompt-queue cap on Polaris (60 min below 10 nodes, 3 h at 10-24 nodes - the
# long small-node queues sat for hours in 2026-10), 12 h on Perlmutter. A
# chain longer than its walltime is resumed in a second job (resume=True).
_WALLTIME_CAP_S = 12 * 3600
_SECONDS_PER_SAMPLE = 5.0  # DES-Y1 3x2pt, ~0.4 s/evaluation x acceptance ~ 1/10


def _default_walltime_cap_s(site: str | None, nodes: int) -> int:
    if not site:
        return _WALLTIME_CAP_S
    try:
        from hep_genesis.iri.dispatch.policy import recommended_walltime_cap_s
    except ImportError:  # engine predating the policy module
        return _WALLTIME_CAP_S
    return int(recommended_walltime_cap_s(site, nodes))


def _chain_walltime(max_samples: int, walltime_s: int | None, site: str | None = None,
                    nodes: int = 1) -> tuple[int, int, list[str]]:
    """(walltime_s, estimated_s, warnings): the walltime a chain job requests.

    Default = the ~5 s/sample estimate clamped to [30 min, the site's cap for a
    default walltime] (see _default_walltime_cap_s); an explicit walltime_s is
    honoured. Either way the estimate is reported, and a chain that cannot
    finish inside the walltime gets a warning pointing at resume.
    """
    est = int(max(1800, max_samples * _SECONDS_PER_SAMPLE))
    warns = []
    cap = _default_walltime_cap_s(site, nodes)
    if walltime_s is None:
        walltime_s = min(est, cap)
        if est > cap:
            where = f" on {site} at {nodes} node(s)" if site else ""
            more = (" (10-24 nodes get 3 h in prod; the longer small-node queues wait for hours and may be "
                    "preempted)" if (site or "").lower() == "polaris" and nodes < 10 else "")
            warns.append(f"estimated runtime {est / 3600:.1f} h exceeds the {cap / 3600:g} h default walltime "
                         f"cap{where}{more}: the chain is capped at walltime_s={walltime_s}; it will stop at the "
                         "walltime and can be continued with resume=True (same arguments).")
    elif est > walltime_s:
        warns.append(f"estimated runtime {est / 3600:.1f} h exceeds walltime_s={walltime_s}: the chain may stop "
                     "before max_samples; continue it with resume=True (same arguments).")
    return int(walltime_s), est, warns


def _needs_ppf(priors: dict, fixed: dict) -> bool:
    """CAMB's fluid dark energy cannot cross w = -1; PPF can. Needed when wa is
    sampled or fixed non-zero, or when the w0 prior straddles -1."""
    if "wa" in priors or float(fixed.get("wa", 0.0) or 0.0) != 0.0:
        return True
    w0 = priors.get("w0")
    return bool(w0) and float(w0["min"]) < -1.0 < float(w0["max"])


def _experiment_with_dark_energy_model(exp_path: Path, model: str) -> Path:
    """A copy of the experiment YAML whose ccl_factory carries
    camb_extra_params.dark_energy_model=<model> (no-op when already set)."""
    import yaml

    doc = yaml.safe_load(exp_path.read_text(encoding="utf-8")) or {}
    extra = dict((doc.get("ccl_factory") or {}).get("camb_extra_params") or {})
    if extra.get("dark_energy_model") == model:
        return exp_path
    extra["dark_energy_model"] = model
    doc.setdefault("ccl_factory", {})["camb_extra_params"] = extra
    out = exp_path.with_name(f"{exp_path.stem}_{model}.yaml")
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return out


def _pid_alive(pid: int) -> bool:
    """True while *pid* runs. A background chain is this server's child: once
    it exits it lingers as a zombie that still answers ``kill(pid, 0)``, so
    reap it first (waitpid raises for processes that are not our children)."""
    import os as _os

    try:
        done, _ = _os.waitpid(pid, _os.WNOHANG)
        if done == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        _os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_chain(chain_txt: Path) -> tuple[list[str], np.ndarray]:
    with chain_txt.open(encoding="utf-8") as fh:
        header = fh.readline().lstrip("#").split()
    arr = np.loadtxt(chain_txt, ndmin=2)
    if arr.shape[1] != len(header):
        raise ValueError(f"{chain_txt.name}: header has {len(header)} columns, data {arr.shape[1]}")
    return header, arr


def _read_progress(progress: Path) -> dict:
    """Cobaya's <prefix>.progress table: N, acceptance_rate, Rminus1 rows."""
    if not progress.is_file():
        return {}
    rows = []
    for line in progress.read_text(encoding="utf-8").splitlines():
        parts = line.lstrip("#").split()
        if not parts or parts[0] == "N":
            continue
        try:
            rows.append({"N": int(float(parts[0])), "acceptance_rate": float(parts[2]),
                         "Rminus1": float(parts[3])})
        except (IndexError, ValueError):
            continue
    if not rows:
        return {}
    return {"checkpoints": rows, "last": rows[-1]}


def _trace_plot(chain_txt: Path, names: list[str], means: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import PALETTE, param_symbol, rc_params

    header, arr = _read_chain(chain_txt)
    colidx = {n: header.index(n) for n in names if n in header}
    with plt.rc_context(rc_params(10)):
        fig, axes = plt.subplots(len(colidx), 1, figsize=(6.4, 1.8 * len(colidx) + 0.6),
                                 squeeze=False, constrained_layout=True, sharex=True)
        for ax, (n, j) in zip(axes[:, 0], colidx.items()):
            ax.plot(np.arange(len(arr)), arr[:, j], "-", color=PALETTE[0], lw=0.9)
            if n in means:
                ax.axhline(means[n], color=PALETTE[1], ls="--", lw=1.0)
            ax.set_ylabel(param_symbol(n))
        axes[-1, 0].set_xlabel("accepted step")
        fig.suptitle(title, fontsize=10)
        fig.savefig(path)
        plt.close(fig)


def _corner_plot(names: list[str], values: np.ndarray, weights: np.ndarray, path: Path, title: str) -> str:
    """Triangle plot; getdist when importable (KDE contours), else a
    matplotlib histogram triangle. Returns which engine drew it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..plotting import PALETTE, param_symbol, rc_params

    try:
        from getdist import MCSamples, plots

        with contextlib.redirect_stdout(sys.stderr):
            mc = MCSamples(samples=values, weights=weights, names=names,
                           labels=[param_symbol(n).strip("$") for n in names], label=title)
            g = plots.get_subplot_plotter(width_inch=min(2.0 * len(names), 10))
            g.triangle_plot([mc], filled=True, contour_colors=[PALETTE[0]])
            g.export(str(path))
        plt.close("all")
        return "getdist"
    except Exception:  # noqa: BLE001 - fall back to a plain histogram triangle
        pass
    n = len(names)
    with plt.rc_context(rc_params(9)):
        fig, axes = plt.subplots(n, n, figsize=(1.9 * n + 0.8, 1.9 * n + 0.8), squeeze=False)
        for i in range(n):
            for j in range(n):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    ax.hist(values[:, i], bins=40, weights=weights, color=PALETTE[0], histtype="stepfilled", alpha=0.6)
                else:
                    ax.hist2d(values[:, j], values[:, i], bins=35, weights=weights, cmap="Blues")
                if i == n - 1:
                    ax.set_xlabel(param_symbol(names[j]))
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(param_symbol(names[i]))
                else:
                    ax.set_yticklabels([])
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
    return "matplotlib"


def _finish_chain(res: dict, outdir: Path, slug: str, exp_path: Path, plot: bool, fixed: dict,
                  warns: list[str], computed_on: str, settings: dict) -> ArtifactResult:
    """Shape the inner firecrown_chain result (summary CSV, trace PNG, metadata)."""
    summary = res["summary"]
    names = list(summary)
    spath = outdir / f"chain_summary_{slug}.csv"
    write_csv(spath, {"parameter_index": np.arange(len(names), dtype=float),
                      "mean": np.array([summary[n]["mean"] for n in names]),
                      "std": np.array([summary[n]["std"] for n in names]),
                      "lower68": np.array([summary[n]["lower68"] for n in names]),
                      "upper68": np.array([summary[n]["upper68"] for n in names])},
              [f"label: Cobaya mcmc summary ({', '.join(names)})", "quantity: chain_summary",
               f"parameters: {names}", f"experiment: {exp_path}", f"n_samples: {res['n_samples']}",
               f"acceptance_rate: {res.get('acceptance_rate')}", f"converged: {res.get('converged')}",
               f"Rminus1: {res.get('Rminus1')}", f"fixed: {fixed}"])
    files = [str(spath)] + [f for f in res.get("chain_files", []) if Path(f).is_file()]
    chain_txt = next((f for f in res.get("chain_files", []) if f.endswith(".1.txt") and Path(f).is_file()), None)
    if plot and chain_txt:
        try:
            ppath = outdir / f"chain_trace_{slug}.png"
            _trace_plot(Path(chain_txt), names, {n: summary[n]["mean"] for n in names}, ppath,
                        f"{exp_path.stem}: Cobaya mcmc trace ({res['n_samples']} samples)")
            files.append(str(ppath))
        except Exception as exc:  # noqa: BLE001 - plotting is best effort
            warns.append(f"trace plot skipped: {exc}")
    txt = ", ".join(f"{n} = {summary[n]['mean']:.4g} +/- {summary[n]['std']:.3g}" for n in names)
    n_chains = int(res.get("n_chains") or 1)
    chains = (f"{n_chains} chains combined (first {res.get('skip_samples', 0):.0%} of each dropped), "
              f"per-chain samples {res.get('chain_samples')}; R-1 is the cross-chain statistic. "
              if n_chains > 1 else "")
    return ArtifactResult(
        status="success", files=files,
        message=(f"Cobaya mcmc: {res['n_samples']} accepted samples ({res.get('n_weighted', 0):.0f} weighted), "
                 f"{chains}"
                 f"acceptance {res.get('acceptance_rate') or float('nan'):.2f}, converged={res.get('converged')} "
                 f"(R-1 = {res.get('Rminus1')}). Posterior means: {txt}."
                 + (f" Computed on {computed_on}." if computed_on != "local" else "")
                 + (" Short run: treat as a smoke test, not a converged posterior." if not res.get("converged") else "")
                 + " Corner/posterior plot: firecrown_plot_chain on the .1.txt chain file."
                 + " " + " ".join(warns)),
        metadata={"summary": summary, "sampled": names, "fixed": fixed, "n_samples": res["n_samples"],
                  "n_weighted": res.get("n_weighted"), "acceptance_rate": res.get("acceptance_rate"),
                  "converged": res.get("converged"), "Rminus1": res.get("Rminus1"), "best_fit": res.get("best_fit"),
                  "used_getdist": res.get("used_getdist"), "chain_files": res.get("chain_files", []),
                  "n_chains": n_chains, "chain_samples": res.get("chain_samples"),
                  "chain_acceptance": res.get("chain_acceptance"),
                  "chain_txt": chain_txt, "warnings": warns, "computed_on": computed_on,
                  "experiment_yaml": str(exp_path), "cobaya_settings": settings},
    )


@validate_call
def firecrown_run_chain(
    output_dir: Annotated[str, Field(min_length=1)],
    experiment_yaml: Annotated[str, Field(min_length=1)],
    priors: Annotated[dict[str, dict[str, float]], Field(description="Sampled parameters with uniform priors: {'sigma8': {'min': 0.6, 'max': 1.0, 'ref': 0.81, 'proposal': 0.02}, 'Omega_c': {...}} (ref/proposal optional).")],
    cosmology: Annotated[CosmologyParams | None, Field(description="Fixed values for cosmology parameters that are not sampled.")] = None,
    nuisance: Annotated[dict[str, float], Field(description="Fixed nuisance values (unsampled ones take firecrown defaults).")] = {},
    max_samples: Annotated[int, Field(ge=4, le=200000, description="Cobaya mcmc max_samples (accepted steps). Small by default: ~0.4 s per step locally for DES-Y1 3x2pt.")] = 200,
    rminus1_stop: Annotated[float, Field(gt=0.0, le=1.0, description="Gelman-Rubin R-1 stopping criterion.")] = 0.05,
    seed: Annotated[int | None, Field(ge=0)] = None,
    chain_prefix: Annotated[str, Field(min_length=1)] = "chain",
    plot: Annotated[bool, Field(description="Trace plot of the sampled parameters (needs the chain file locally).")] = True,
    env_setup: Annotated[str | None, Field(description="Remote runs only, OPTIONAL override: a shell snippet activating a facility-resident environment to use INSTEAD of the pack's own lock environment (which the engine builds on the node by default). Leave empty.")] = None,
    sacc_remote_path: str | None = None,
    walltime_s: Annotated[int | None, Field(ge=300, le=86400, description="Walltime of the facility job / timeout of a local run, seconds. Default: ~5 s per sample, clamped to [30 min, the site's prompt-queue cap]: Polaris 60 min below 10 nodes and 3 h at 10-24 nodes (longer small-node jobs are refused: they wait for hours or get preempted), Perlmutter 12 h (regular admits 48 h; shorter requests backfill sooner). Size max_samples to it, or continue a stopped chain with resume=True. Explicit values up to the QOS cap are honoured.")] = None,
    resume: Annotated[bool, Field(description="Continue a chain with the same prefix/arguments from its Cobaya checkpoint (a run that hit its walltime or max_samples) instead of starting over.")] = False,
    nodes: Annotated[int, Field(ge=1, description="Facility runs only: nodes for the chain job. Each node runs 4 Cobaya chains in parallel (one MPI process per chain, 8 cores each: the kernel's DISPATCH_MPI shape), so nodes=2 is 8 chains and R-1 becomes the real cross-chain Gelman-Rubin statistic; the result combines all chains. Walltime is per chain (each still runs up to max_samples), the allocation charge is nodes x walltime, and the facility's node cap (NERSC_MAX_NODES / ALCF_MAX_NODES) applies. resume needs the same nodes. Not combinable with env_setup.")] = 1,
    background: Annotated[bool, Field(description="LOCAL runs only: start the chain as a detached subprocess and return at once with the chain paths; poll firecrown_chain_status, then firecrown_plot_chain for the posterior. Use for anything beyond a few hundred samples - a foreground local chain blocks this call for its whole runtime.")] = False,
    dark_energy_model: Annotated[Literal["auto", "fluid", "ppf"], Field(description="CAMB dark-energy model. 'auto' (default) switches to 'ppf' when wa is sampled or non-zero, or the w0 prior straddles -1 - CAMB's fluid model cannot cross w = -1 and the chain would die with 'set the dark_energy_model to ppf'.")] = "auto",
    wait: Annotated[bool, Field(description="Facility runs only. True (default): this call blocks until the job finishes (queue + walltime: possibly hours). False: return as soon as the scheduler accepts the job, with metadata.pending (a chain_pending_<slug>.json file) - then call firecrown_collect_chain(pending=...) later for the summary, trace plot and chain files. Use False for long chains and whenever several jobs must run at the same time (e.g. one chain on Perlmutter and one on Polaris: set_dispatch + submit, set_dispatch + submit, then collect both).")] = True,
) -> ArtifactResult:
    """Run a Cobaya MCMC over a firecrown experiment in pure-CCL mode (no theory block) and summarise the posterior.

    Heavy: each accepted step costs several likelihood evaluations (~0.4 s
    each for DES-Y1 3x2pt), so the default max_samples=200 is a smoke run
    (~minutes); real chains need thousands of samples and dispatch to a
    facility (set_dispatch + env_setup) or a local background run
    (background=True). The job's walltime is walltime_s (default ~5 s per
    sample, capped at 12 h): a chain that needs more is continued with
    resume=True in a second call with the SAME arguments. Cobaya 'mcmc' with
    uniform priors, proposal = (max-min)/20 unless given, burn_in 0. Writes
    the Cobaya chain (<prefix>.1.txt, .updated.yaml, .covmat, .progress),
    chain_summary_<slug>.csv (parameter, mean, std, lower68, upper68 from
    the weighted samples; getdist means/stds when getdist is present) and
    a trace PNG; firecrown_plot_chain draws the corner plot. metadata.converged
    / Rminus1 say whether the run reached rminus1_stop (short runs will not:
    treat them as smoke tests). Run firecrown_scan_loglike first to choose
    sensible prior ranges. On a facility, nodes=N runs 4 N chains in parallel
    (one per MPI process) and reports the cross-chain R-1.
    """
    exp_path = _exp_path(experiment_yaml)
    outdir = resolve_outdir(output_dir)
    like, tools, _ = _load_local(exp_path)
    from ..inner import firecrown_loglike as inner

    required = inner.required_defaults(like, tools)
    if not priors:
        raise ValueError("priors must name at least one parameter to sample.")
    bad = sorted(k for k in priors if k not in required)
    if bad:
        raise ValueError(f"priors for parameters the likelihood does not have: {bad}; required: {sorted(required)}")
    for k, spec_ in priors.items():
        if "min" not in spec_ or "max" not in spec_ or not spec_["min"] < spec_["max"]:
            raise ValueError(f"prior for {k} needs min < max: {spec_}")
    point, warns = _check_point(required, cosmology, nuisance, exp_path)
    fixed = {k: v for k, v in point.items() if k not in priors}
    # CAMB (fluid AND ppf) requires w0 + wa < 0 (w(a) < 0 at early times):
    # a prior box reaching w0 + wa >= 0 kills the chain hours in (229 samples
    # then "w0+wa > 0", 2026-10-09). Check the box corner up front.
    w0_max = float(priors["w0"]["max"]) if "w0" in priors else float(fixed.get("w0", -1.0))
    wa_max = float(priors["wa"]["max"]) if "wa" in priors else float(fixed.get("wa", 0.0))
    if w0_max + wa_max >= 0.0:
        raise ValueError(f"CAMB requires w0 + wa < 0 everywhere in the prior box, but max(w0) + max(wa) = "
                         f"{w0_max:g} + {wa_max:g} = {w0_max + wa_max:g}. Tighten the priors (e.g. w0 in "
                         "[-1.5, -0.5], wa in [-1.5, 0.5]) so the corner stays below 0.")
    if dark_energy_model == "auto":
        dark_energy_model = "ppf" if _needs_ppf(priors, fixed) else "fluid"
    if dark_energy_model == "ppf":
        exp_path = _experiment_with_dark_energy_model(exp_path, "ppf")
        warns.append("CAMB dark_energy_model=ppf (w(a) may cross -1); experiment copy " + exp_path.name + ".")
    from mcp_server.dispatch import remote_site

    site = remote_site()
    walltime, est_s, wt_warns = _chain_walltime(max_samples, walltime_s, site, nodes)
    warns.extend(wt_warns)
    n_chains = nodes * int(_CHAIN_MPI["main"]["tasks_per_node"]) if site else 1
    if nodes > 1:
        if not site:
            raise ValueError("nodes > 1 applies to facility runs: call set_dispatch('perlmutter' or 'polaris') "
                             "first. Local runs are one chain (use background=True for long ones).")
        if env_setup and env_setup.strip():
            raise ValueError("nodes > 1 runs the chains in the pack's own lock environment (mpi4py is built "
                             "there); drop env_setup.")
        if background:
            raise ValueError("background=True is for local runs; a facility job is already asynchronous.")
    # slug excludes walltime/resume/background so a resumed chain finds its files
    slug = param_slug({"priors": repr(sorted(priors.items())), "fixed": repr(sorted(fixed.items())),
                       "n": max_samples, "seed": seed, "exp": str(exp_path)})
    prefix = f"{chain_prefix}_{slug}"
    work_dir = str(outdir) if not site else "."
    params = {"fixed": fixed, "priors": {k: as_float_dict(v) for k, v in priors.items()},
              "max_samples": max_samples, "rminus1_stop": rminus1_stop, "work_dir": work_dir,
              "chain_prefix": prefix, "seed": seed, "resume": resume,
              # facility job dirs (NERSC CFS) have no POSIX locks: errno 524
              "file_locking": not site}
    settings = {"sampler": "mcmc", "max_samples": max_samples, "rminus1_stop": rminus1_stop, "seed": seed,
                "creation_mode": "pure_ccl_mode", "dark_energy_model": dark_energy_model,
                "walltime_s": walltime, "estimated_seconds": est_s, "resume": resume,
                "nodes": nodes, "n_chains": n_chains}
    if resume and not site and not (outdir / f"{prefix}.1.txt").is_file():
        raise ValueError(f"resume=True but no chain {prefix}.1.txt in {outdir}: start without resume, or pass "
                         "the same arguments as the run to continue.")

    if background and not site:
        from ..envkernel import start_in_background

        job = start_in_background(env_setup, "firecrown_chain", dict(params, experiment_yaml=str(exp_path)),
                                  job_dir=str(outdir), tag=prefix)
        chain_txt = str(outdir / f"{prefix}.1.txt")
        write_json(outdir / f"chain_background_{slug}.json",
                   {**job, "chain_txt": chain_txt, "progress": str(outdir / f"{prefix}.progress"),
                    "chain_prefix": prefix, "experiment_yaml": str(exp_path), "settings": settings})
        return ArtifactResult(
            status="success", files=[str(outdir / f"chain_background_{slug}.json")],
            message=(f"Chain started in the background (pid {job['pid']}): {max_samples} max samples, "
                     f"est. {est_s / 60:.0f} min. Poll firecrown_chain_status(chain_txt='{chain_txt}') "
                     "(progress file: N, acceptance, R-1); when it reports finished, call "
                     f"firecrown_plot_chain(chain_txt='{chain_txt}') for the posterior summary and plots. "
                     + " ".join(warns)),
            metadata={"background": job, "chain_txt": chain_txt, "chain_prefix": prefix, "sampled": list(priors),
                      "fixed": fixed, "experiment_yaml": str(exp_path), "cobaya_settings": settings,
                      "warnings": warns, "computed_on": "local (background)"},
        )

    if not wait and not site:
        raise ValueError("wait=False applies to facility runs (set_dispatch first); a long local chain uses "
                         "background=True.")
    res, computed_on = _run_inner("firecrown_chain", params, env_setup, walltime, exp_path, sacc_remote_path,
                                  nodes=nodes, wait=wait)
    if isinstance(res, dict) and res.get("status") == "submitted":
        pending = outdir / f"chain_pending_{slug}.json"
        write_json(pending, {"record": res.get("record"), "job_id": res.get("job_id"), "site": res.get("site"),
                             "slug": slug, "experiment_yaml": str(exp_path), "plot": plot, "fixed": fixed,
                             "warnings": warns, "settings": settings, "chain_prefix": prefix,
                             "node_hours": res.get("node_hours")})
        return ArtifactResult(
            status="success", files=[str(pending)],
            message=(f"Chain submitted to {res.get('site')} as job {res.get('job_id')} ({nodes} node(s), "
                     f"walltime {walltime} s, up to {res.get('node_hours')} node-hours); not waiting for it. "
                     f"Poll find_dispatched_jobs(directory='{res.get('artifact_dir')}') on the facility server; "
                     f"when the job has finished, call firecrown_collect_chain(output_dir='{outdir}', "
                     f"pending='{pending}') for the posterior summary, trace plot and chain files. "
                     + (res.get("note") or "") + " " + " ".join(warns)).strip(),
            metadata={"submitted": True, "pending": str(pending), "job_id": res.get("job_id"),
                      "site": res.get("site"), "record": res.get("record"), "artifact_dir": res.get("artifact_dir"),
                      "chain_prefix": prefix, "sampled": list(priors), "fixed": fixed,
                      "experiment_yaml": str(exp_path), "cobaya_settings": settings, "warnings": warns,
                      "computed_on": res.get("site")},
        )
    return _finish_chain(res, outdir, slug, exp_path, plot, fixed, warns, computed_on, settings)


firecrown_run_chain.weight = "heavy"


@validate_call
def firecrown_collect_chain(
    output_dir: Annotated[str, Field(min_length=1)],
    pending: Annotated[str, Field(min_length=1, description="The chain_pending_<slug>.json file firecrown_run_chain(wait=False) returned (metadata.pending).")],
    wait_s: Annotated[int, Field(ge=0, le=3600, description="0 (default): one look. > 0: keep waiting up to this many seconds (polling every 30 s) while the job is still queued/running, then answer. Use e.g. 600 instead of calling this tool in a tight loop - you cannot sleep between calls, and every poll re-sends the whole conversation.")] = 0,
) -> ArtifactResult:
    """Collect a facility chain submitted with firecrown_run_chain(wait=False): one look at the job (or a bounded wait with wait_s).

    Finished: fetches the chain files back and returns the same summary as a
    waiting firecrown_run_chain (means, 68% intervals, R-1 / converged, trace
    plot; then firecrown_plot_chain for the corner plot). Still queued or
    running: status 'pending' with the scheduler state - call again later
    (or pass wait_s), never resubmit. Ended without results: an error with
    the job's stderr verdict. Works whatever set_dispatch currently says:
    the job's record names its facility.
    """
    import json as _json

    from mcp_server.dispatch import collect_kernel  # lazy: server-only

    pfile = Path(pending).expanduser()
    if not pfile.is_file():
        raise ValueError(f"No pending-chain file at {pfile}: pass metadata.pending from firecrown_run_chain(wait=False).")
    info = _json.loads(pfile.read_text(encoding="utf-8"))
    if not info.get("record"):
        raise ValueError(f"{pfile} has no dispatch record (the submission wrote none - no artifact folder?).")
    outdir = resolve_outdir(output_dir)
    res = collect_kernel(info["record"], wait_s=wait_s)
    if res.get("status") in ("pending", "unknown"):
        return ArtifactResult(
            status="success", files=[],
            message=(f"Chain job {info.get('job_id')} on {info.get('site')}: {res.get('state') or res['status']} - "
                     f"no results yet. {res.get('message', '')} Call firecrown_collect_chain again later."),
            metadata={"pending": True, "state": res.get("state"), "job_id": info.get("job_id"),
                      "site": info.get("site"), "record": info["record"]},
        )
    result, computed_on = _remote_result(res, info.get("site") or "facility",
                                         int(info.get("settings", {}).get("nodes") or 1))
    return _finish_chain(result, outdir, info["slug"], Path(info["experiment_yaml"]), bool(info.get("plot", True)),
                         info.get("fixed") or {}, list(info.get("warnings") or []), computed_on,
                         info.get("settings") or {})


firecrown_collect_chain.weight = "dispatchable"


@validate_call
def firecrown_chain_status(
    chain_txt: Annotated[str, Field(min_length=1, description="The chain file <work_dir>/<prefix>.1.txt (from firecrown_run_chain's metadata.chain_txt); its .progress / .locked / background files are found next to it.")],
) -> ArtifactResult:
    """Cheap poll of a running or finished Cobaya chain: accepted samples so far, the latest acceptance rate and R-1 from the .progress file, and whether it is still running.

    Status 'running' (Cobaya's lock file is present or the background
    process is alive), 'finished' (background result written, or a chain
    file with no lock), 'failed' (background driver reported an error - see
    metadata.error / log_tail), or 'not_started'. For facility runs the
    chain lives in the job directory; poll the facility job instead and
    call this on the files fetched back.
    """
    chain = Path(chain_txt).expanduser()
    stem = chain.name[:-len(".1.txt")] if chain.name.endswith(".1.txt") else chain.stem
    work = chain.parent
    progress = _read_progress(work / f"{stem}.progress")
    n_lines = 0
    if chain.is_file():
        with chain.open(encoding="utf-8") as fh:
            n_lines = sum(1 for line in fh if line.strip() and not line.startswith("#"))
    locked = any(p.name.startswith(stem) and p.name.endswith(".locked") for p in work.glob(f"{stem}*.locked"))
    bg = next(iter(work.glob(f"inner_firecrown_chain_{stem}_background.json")), None)
    status, error, log_tail, result_path = "not_started", None, None, None
    if bg is not None:
        import json as _json

        info = _json.loads(bg.read_text(encoding="utf-8"))
        result_path = info.get("result_path")
        log_path = info.get("log_path")
        if log_path and Path(log_path).is_file():
            log_tail = "\n".join(Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()[-15:])
        if result_path and Path(result_path).is_file():
            out = _json.loads(Path(result_path).read_text(encoding="utf-8"))
            status = "finished" if out.get("ok") else "failed"
            error = None if out.get("ok") else (out.get("error") or "")[-1200:]
        else:
            try:
                alive = _pid_alive(int(info["pid"]))
            except (KeyError, ValueError):
                alive = False
            status = "running" if alive else ("failed" if n_lines == 0 else "stopped")
            if not alive and not error:
                error = "background process is gone without writing a result (killed? walltime?); see log_tail"
    elif chain.is_file():
        status = "running" if locked else "finished"
    last = progress.get("last") or {}
    note = ""
    ptxt = work / "progress.txt"
    if ptxt.is_file():
        try:
            note = ptxt.read_text(encoding="utf-8").strip().splitlines()[-1]
        except (OSError, IndexError):
            note = ""
    return ArtifactResult(
        status="success", files=[],
        message=(f"Chain {stem}: {status}; {n_lines} accepted samples in the chain file"
                 + (f" [{note}]" if note else "")
                 + (f"; last checkpoint N={last['N']}, acceptance {last['acceptance_rate']:.2f}, "
                    f"R-1 = {last['Rminus1']:.3g}" if last else "; no checkpoint yet")
                 + (f". Error: {error[:200]}" if error else "")
                 + (". Next: firecrown_plot_chain on this chain file." if status in ("finished", "stopped") else "")
                 + (". Stop it with firecrown_chain_cancel." if status == "running" and bg is not None else "")),
        metadata={"status": status, "n_samples": n_lines, "progress": progress, "locked": locked,
                  "chain_txt": str(chain), "result_path": result_path, "error": error, "log_tail": log_tail},
    )


@validate_call
def firecrown_plot_chain(
    output_dir: Annotated[str, Field(min_length=1)],
    chain_txt: Annotated[str, Field(min_length=1, description="A Cobaya chain file (<prefix>.1.txt) - from firecrown_run_chain here or fetched back from a facility job.")],
    params: Annotated[list[str] | None, Field(description="Parameters to plot (default: every sampled parameter in the file, i.e. all columns except weight/minuslogpost/chi2/prior columns).")] = None,
    burn_in_frac: Annotated[float, Field(ge=0.0, lt=0.9, description="Fraction of the chain discarded before summarising/plotting (0.3 is a common choice for a single chain; firecrown_run_chain's own summary uses the whole chain).")] = 0.3,
    corner: Annotated[bool, Field(description="Triangle (corner) plot: getdist filled contours when getdist is installed, else a histogram triangle.")] = True,
    trace: Annotated[bool, Field(description="Trace plot of each parameter vs accepted step.")] = True,
) -> ArtifactResult:
    """Posterior summary and plots from an existing Cobaya chain file: weighted means, std, 68% intervals after burn-in, a corner plot and a trace plot.

    Works on any Cobaya chain (local runs, background runs once
    firecrown_chain_status says finished, or chain files fetched back from
    a facility job). Reads <prefix>.progress next to the file for the last
    R-1 when present. Files: chain_posterior_<slug>.csv, chain_corner_<slug>.png,
    chain_trace_<slug>.png.
    """
    chain = Path(chain_txt).expanduser()
    if not chain.is_file():
        raise ValueError(f"chain file not found: {chain}")
    outdir = resolve_outdir(output_dir)
    header, arr = _read_chain(chain)
    skip = {"weight", "minuslogpost", "chi2", "minuslogprior"}
    cols = [h for h in header if h not in skip and not h.startswith(("minuslogprior__", "chi2__"))]
    names = list(params) if params else cols
    missing = [n for n in names if n not in header]
    if missing:
        raise ValueError(f"parameters not in the chain: {missing}; available: {cols}")
    n_total = arr.shape[0]
    start = int(n_total * burn_in_frac)
    kept = arr[start:]
    if kept.shape[0] < 3:
        raise ValueError(f"only {kept.shape[0]} samples after burn-in (chain has {n_total}); lower burn_in_frac.")
    weights = kept[:, header.index("weight")] if "weight" in header else np.ones(kept.shape[0])
    values = np.column_stack([kept[:, header.index(n)] for n in names])
    from ..inner.firecrown_chain import _weighted_summary

    summary = _weighted_summary(names, values, weights)
    stem = chain.name[:-len(".1.txt")] if chain.name.endswith(".1.txt") else chain.stem
    progress = _read_progress(chain.parent / f"{stem}.progress")
    slug = param_slug({"chain": str(chain), "burn": burn_in_frac, "params": names})
    spath = outdir / f"chain_posterior_{slug}.csv"
    write_csv(spath, {"parameter_index": np.arange(len(names), dtype=float),
                      "mean": np.array([summary[n]["mean"] for n in names]),
                      "std": np.array([summary[n]["std"] for n in names]),
                      "lower68": np.array([summary[n]["lower68"] for n in names]),
                      "upper68": np.array([summary[n]["upper68"] for n in names])},
              [f"label: posterior summary after {burn_in_frac:.0%} burn-in ({', '.join(names)})",
               "quantity: chain_posterior", f"parameters: {names}", f"chain: {chain}",
               f"n_total: {n_total}", f"n_kept: {kept.shape[0]}",
               f"Rminus1_last: {(progress.get('last') or {}).get('Rminus1')}"])
    files = [str(spath)]
    warns = []
    engine = None
    title = f"{stem} ({kept.shape[0]} samples after {burn_in_frac:.0%} burn-in)"
    if corner:
        try:
            cpath = outdir / f"chain_corner_{slug}.png"
            engine = _corner_plot(names, values, np.asarray(weights, float), cpath, title)
            files.append(str(cpath))
        except Exception as exc:  # noqa: BLE001
            warns.append(f"corner plot skipped: {exc}")
    if trace:
        try:
            tpath = outdir / f"chain_trace_{slug}.png"
            _trace_plot(chain, names, {n: summary[n]["mean"] for n in names}, tpath, f"{stem}: trace")
            files.append(str(tpath))
        except Exception as exc:  # noqa: BLE001
            warns.append(f"trace plot skipped: {exc}")
    txt = ", ".join(f"{n} = {summary[n]['mean']:.4g} [{summary[n]['lower68']:.4g}, {summary[n]['upper68']:.4g}]"
                    for n in names)
    last = progress.get("last") or {}
    return ArtifactResult(
        status="success", files=files,
        message=(f"Posterior from {chain.name}: {kept.shape[0]} of {n_total} samples after {burn_in_frac:.0%} burn-in"
                 + (f", last R-1 = {last['Rminus1']:.3g}" if last else "")
                 + f". Means [68%]: {txt}."
                 + (f" Corner plot drawn with {engine}." if engine else "") + " " + " ".join(warns)),
        metadata={"summary": summary, "parameters": names, "n_total": n_total, "n_kept": int(kept.shape[0]),
                  "burn_in_frac": burn_in_frac, "progress": progress, "corner_engine": engine,
                  "chain_txt": str(chain), "warnings": warns},
    )


@validate_call
def firecrown_chain_cancel(
    chain_txt: Annotated[str, Field(min_length=1, description="The chain file of a BACKGROUND run (firecrown_run_chain background=True); its background record is found next to it.")],
) -> ArtifactResult:
    """Stop a background chain started by firecrown_run_chain(background=True): terminates the detached process (SIGTERM, then SIGKILL after 10 s) and leaves the chain files for firecrown_plot_chain or resume=True.

    The chat's Stop button cannot reach a background chain (it is a
    subprocess of this server) - call this when the user wants it stopped.
    Facility jobs are cancelled through the facility server (cancel_job).
    """
    import json as _json
    import os as _os
    import signal
    import time as _time

    chain = Path(chain_txt).expanduser()
    stem = chain.name[:-len(".1.txt")] if chain.name.endswith(".1.txt") else chain.stem
    bg = next(iter(chain.parent.glob(f"inner_firecrown_chain_{stem}_background.json")), None)
    if bg is None:
        raise ValueError(f"no background record for {stem} in {chain.parent}: this chain was not started with "
                         "background=True (a foreground call ends with its result; facility jobs: cancel_job).")
    info = _json.loads(bg.read_text(encoding="utf-8"))
    pid = int(info["pid"])

    def alive() -> bool:
        return _pid_alive(pid)

    if not alive():
        return ArtifactResult(status="success", files=[], message=f"Chain {stem} (pid {pid}) was not running.",
                              metadata={"pid": pid, "was_running": False, "chain_txt": str(chain)})
    _os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not alive():
            break
        _time.sleep(0.1)
    killed = False
    if alive():
        _os.kill(pid, signal.SIGKILL)
        killed = True
        _time.sleep(0.5)
    n = 0
    if chain.is_file():
        with chain.open(encoding="utf-8") as fh:
            n = sum(1 for line in fh if line.strip() and not line.startswith("#"))
    return ArtifactResult(
        status="success", files=[],
        message=f"Chain {stem} stopped (pid {pid}{', SIGKILL' if killed else ''}); {n} accepted samples kept in "
                f"{chain.name} - firecrown_plot_chain can summarise them, resume=True continues the chain.",
        metadata={"pid": pid, "was_running": True, "killed": killed, "n_samples": n, "chain_txt": str(chain)},
    )
