"""CCL (pyccl 3.3.6) theory predictions: background, P(k), SRD n(z), angular
C_ell, correlation functions, halo model and baryonic boosts.

Boundary conventions (differ from pyccl's native h-free Mpc / scale factor):
redshift z (never a); k in h/Mpc and P(k) in (Mpc/h)^3 by default with a
``k_units`` switch exposing CCL's 1/Mpc and Mpc^3; theta in arcmin; masses in
Msun (NOT Msun/h) and halo number densities in comoving Mpc^-3 (NOT h^3);
distances in Mpc. The cosmology is one ``CosmologyParams`` object (firecrown
names) shared with the other families.

Kernel/wrapper split: ``kernels.py`` holds the pip-dispatchable computations
(pure pyccl, JSON-safe); the wrappers here validate cheaply, branch on
``remote_site()``, then write CSV/PNG/JSON/sacc wherever the server runs.
"""

import time
import warnings
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator, validate_call

from ..common import (ArtifactResult, downsample_columns, get_cached, optional_import,
                      param_slug, read_csv, resolve_outdir, summary_stats, write_csv,
                      write_json, z_to_a)
from ..cosmology import CosmologyParams
from ..plotting import LINESTYLES, PALETTE, rc_params
from .kernels import (BASE_PIP_DEPS, DISPATCH_PIP_DEPS, EMULATOR_BOXES, build_tracer,
                      emulator_box_check)
from .kernels import compute_angular_cls as _cls_kernel
from .kernels import compute_matter_pk as _pk_kernel
from .srd_nz import FORECAST_YEARS, SRD_PARAMS, SRD_REFERENCE, mean_z, srd_sample

__all__ = ["ccl_describe_cosmology", "ccl_background", "ccl_matter_pk", "ccl_lsst_srd_nz",
           "ccl_angular_cls", "ccl_correlation_functions", "ccl_correlation_3d",
           "ccl_halo_mass_function", "ccl_halo_model_pk", "ccl_baryon_boost"]

CONVENTIONS = {
    "redshift": "all tools take z; a=1/(1+z) is internal",
    "k_units": "default 'h/Mpc' with P in (Mpc/h)^3; k_units='1/Mpc' gives CCL-native 1/Mpc and Mpc^3",
    "distances": "Mpc (no h)", "theta": "arcmin", "ell": "integer multipoles >= 2",
    "masses": "Msun (no h); dn/dlog10M in comoving Mpc^-3 (no h^3). Multiply M by h and divide n by h^3 for the emulator server's Msun/h, (Mpc/h)^-3",
    "cosmology": "CosmologyParams (firecrown names Omega_c Omega_b h n_s sigma8|A_s Omega_k Neff m_nu w0 wa); presets vanilla/planck18/desc_srd",
    "sacc": "NZ tracers src{i} (galaxy_shear) / lens{i} (galaxy_density); C_ell types galaxy_shear_cl_ee, galaxy_shearDensity_cl_e (shear tracer first), galaxy_density_cl",
    "srd_bias": "bias='srd' on a number-counts tracer means b = galaxy_bias_prefactor / D(z_mean) (1.05 Y1/Y4, 0.95 Y7/Y10)",
}
CAVEATS = [
    "transfer_function='boltzmann_camb' (default) needs the camb package; 'boltzmann_class' needs classy; "
    "matter_power_spectrum='bacco' and baryon model 'bacco' need baccoemu (+tensorflow). bbks/eisenstein_hu need sigma8, not A_s.",
    "Emulator backends have training boxes (cosmicemu: sigma8 0.7-0.9, z<=2.02; bacco: z<=1.5); ccl_matter_pk reports "
    "'in_training_box' and refuses parameters outside the cosmicemu box (pyccl would raise anyway).",
    "'halofit' is Takahashi 2012; HMcode-2020 is matter_power_spectrum='camb_hmcode' (hmcode_logT_AGN sets feedback).",
    "Non-Limber (FKEM) is applied only to pairs involving a number-counts tracer unless non_limber_pairs='all'; it is 10-30x "
    "slower than Limber and emits harmless CCLWarnings (silenced).",
    "Correlation functions need C_ell to high ell: ccl_correlation_functions recomputes Limber C_ell up to ell_max_internal (20000) internally.",
    "vandaalen19 baryon boost is calibrated for k <= 1 h/Mpc at z=0 only.",
    "pyccl Cosmology objects are cached per parameter set; the first P(k) call pays the spline build (0.05 s bbks, 0.1-0.5 s CAMB).",
]
DISPATCH_KERNELS = {
    "ccl_matter_pk": {"function": "ccl.kernels.compute_matter_pk",
                      "args": {"cosmo_params": "<CosmologyParams dict>", "z": 0.0, "k_list": "[...]", "k_units": "h/Mpc"},
                      "base_pip_deps": BASE_PIP_DEPS, "pip_deps": DISPATCH_PIP_DEPS["ccl_core"],
                      "duration_hint_s": 600, "returns": "{k, pk_lin, pk_nl, sigma8, k_nl, box}"},
    "ccl_angular_cls": {"function": "ccl.kernels.compute_angular_cls",
                        "args": {"cosmo_params": "<CosmologyParams dict>", "tracers": "[{type,name,z,nz,bias_value,...}]",
                                 "pairs": "[[n1,n2],...]", "ell_list": "[...]", "l_limber": -1, "non_limber_pairs": "clustering"},
                        "base_pip_deps": BASE_PIP_DEPS, "pip_deps": DISPATCH_PIP_DEPS["ccl_core"],
                        "duration_hint_s": 900, "returns": "{ell, cls: {'n1|n2': [...]}, l_limber_used, kernel_z_range}"},
}

KUnits = Literal["h/Mpc", "1/Mpc"]
_TRANSFER_DEPS = {"boltzmann_camb": ("camb", "pip install camb (or conda install -c conda-forge camb)"),
                  "boltzmann_class": ("classy", "pip install classy (compiles CLASS)")}
_PK_DEPS = {"camb_hmcode": ("camb", "pip install camb"),
            "bacco": ("baccoemu", "pip install baccoemu tensorflow-cpu")}


# ---------------------------------------------------------------- helpers
def _spec(cosmology) -> CosmologyParams:
    return cosmology if cosmology is not None else CosmologyParams()


def _check_backends(spec: CosmologyParams, needs_pk: bool = True) -> None:
    """Cheap optional-dependency checks with actionable errors (before any dispatch)."""
    if not needs_pk:
        return
    if spec.transfer_function in _TRANSFER_DEPS:
        mod, hint = _TRANSFER_DEPS[spec.transfer_function]
        optional_import(mod, f"transfer_function='{spec.transfer_function}'",
                        f"{hint}, or use transfer_function='eisenstein_hu'/'bbks' (analytic, sigma8 only).")
    if spec.matter_power_spectrum in _PK_DEPS:
        mod, hint = _PK_DEPS[spec.matter_power_spectrum]
        optional_import(mod, f"matter_power_spectrum='{spec.matter_power_spectrum}'",
                        f"{hint}, or use 'halofit' / 'cosmicemu_mt4' (no extra dependency).")


def _build(spec: CosmologyParams, needs_pk: bool = True):
    _check_backends(spec, needs_pk)
    try:
        return spec.build()
    except ImportError as exc:  # pragma: no cover - backends checked above
        raise RuntimeError(f"pyccl backend import failed: {exc}. Install the package or switch backend.") from exc


def _k_grid(k_min, k_max, n_k):
    return np.logspace(np.log10(k_min), np.log10(k_max), int(n_k))


def _k_cols(k_units: str) -> tuple[str, str, float]:
    """(k column name, P column suffix) for the chosen units."""
    if k_units == "h/Mpc":
        return "k_h_per_Mpc", "Mpc3_over_h3"
    return "k_per_Mpc", "Mpc3"


def _to_ccl_k(k, k_units, h):
    return np.asarray(k, dtype=float) * (h if k_units == "h/Mpc" else 1.0)


def _plot_curves(path: Path, x, curves: list[tuple[str, np.ndarray]], *, xlabel: str, ylabel: str,
                 title: str, logx: bool = True, logy: bool = True, unity: bool = False) -> str:
    """One-panel line plot, plain-text labels (no LaTeX), categorical palette up to
    8 curves, viridis ramp + no legend beyond that."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray(x, dtype=float)
    n = len(curves)
    with plt.rc_context(rc_params()):
        fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
        many = n > len(PALETTE)
        cmap = plt.get_cmap("viridis")
        for i, (label, y) in enumerate(curves):
            y = np.asarray(y, dtype=float)
            color = cmap(i / max(n - 1, 1)) if many else PALETTE[i]
            style = "-" if many else LINESTYLES[i]
            if logy:
                neg = y < 0
                ax.plot(x, np.abs(y), color=color, linestyle=style, label=None if many else label)
                if neg.any():
                    ax.plot(x[neg], np.abs(y[neg]), color=color, linestyle="none", marker=".", markersize=3)
            else:
                ax.plot(x, y, color=color, linestyle=style, label=None if many else label)
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if unity:
            ax.axhline(1.0, color="#6b6b6b", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel + (" (|value|, dots = negative)" if logy and any(np.any(np.asarray(y) < 0) for _, y in curves) else ""))
        ax.set_title(title + (f" ({n} curves)" if many else ""), loc="left", fontsize=11)
        if not many and n > 1:
            ax.legend(fontsize=8, ncol=1 if n <= 4 else 2)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
        plt.close(fig)
    return str(path)


def _read_nz(nz_file: str, column: str) -> tuple[np.ndarray, np.ndarray, dict]:
    header, cols = read_csv(nz_file)
    if "z" not in cols:
        raise ValueError(f"{nz_file} has no 'z' column (columns: {list(cols)}).")
    if column not in cols:
        raise ValueError(f"{nz_file} has no column {column!r}; available: {list(cols)}.")
    return np.asarray(cols["z"], float), np.asarray(cols[column], float), header


# ---------------------------------------------------------------- tracer specs
class TracerSpec(BaseModel):
    """One tracer for ccl_angular_cls / ccl_correlation_functions.

    type: 'wl' (weak-lensing shear, spin 2, needs nz_file), 'nc' (galaxy number
    counts, spin 0, needs nz_file), 'cmb_lensing' (CMB convergence at z_source),
    'tsz', 'cib', 'isw' (spin-0 kernels to z_max, unit transfer; combine with a
    halo-model P(k) for physical amplitudes). nz_file is any CSV with a 'z'
    column (e.g. from ccl_lsst_srd_nz, then nz_column='bin_0'...). bias: a
    number, or 'srd' for b = bias_prefactor / D(z_mean) with the prefactor read
    from the n(z) CSV header (galaxy_bias_prefactor) or bias_prefactor (1.05).
    A_ia/eta_ia/z_piv_ia: NLA intrinsic alignments A_IA(z) = A_ia ((1+z)/(1+z_piv))^eta_ia.
    """
    model_config = {"extra": "forbid"}
    type: Literal["wl", "nc", "cmb_lensing", "tsz", "cib", "isw"]
    name: Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$", description="Tracer name (sacc tracer name; src0/lens0 for firecrown).")]
    nz_file: Annotated[str | None, Field(description="CSV with 'z' + n(z) columns (wl/nc only).")] = None
    nz_column: Annotated[str, Field(description="Column of nz_file to use, e.g. 'nz' or 'bin_2'.")] = "nz"
    bias: Annotated[float | Literal["srd"] | None, Field(description="nc: constant linear bias, or 'srd'. None -> 1.0")] = None
    bias_prefactor: Annotated[float | None, Field(ge=0.1, le=5.0, description="For bias='srd' when the CSV header lacks galaxy_bias_prefactor.")] = None
    mag_bias: Annotated[float | None, Field(description="nc: magnification bias s (e.g. 0.1); None = off")] = None
    has_rsd: bool = False
    A_ia: Annotated[float | None, Field(description="wl: NLA amplitude; None = no IA")] = None
    eta_ia: float = 0.0
    z_piv_ia: Annotated[float, Field(ge=0.0)] = 0.62
    z_source: Annotated[float, Field(ge=1.0, le=1500.0)] = 1100.0
    z_max: Annotated[float, Field(ge=0.5, le=20.0, description="tsz/cib/isw kernel extent")] = 6.0

    @model_validator(mode="after")
    def _check(self):
        if self.type in ("wl", "nc") and not self.nz_file:
            raise ValueError(f"tracer {self.name!r} of type {self.type} needs nz_file.")
        return self

    def spin(self) -> int:
        return 2 if self.type == "wl" else 0


def _resolve_tracers(specs: list[TracerSpec], cosmo) -> tuple[list[dict], dict]:
    """TracerSpec -> JSON kernel dicts (n(z) inline, 'srd' bias resolved)."""
    import pyccl as ccl
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"tracer names must be unique: {names}")
    out, notes = [], {}
    for s in specs:
        d = {"type": s.type, "name": s.name, "z_source": s.z_source, "z_max": s.z_max}
        if s.type in ("wl", "nc"):
            z, nz, header = _read_nz(s.nz_file, s.nz_column)
            d["z"], d["nz"] = z.tolist(), nz.tolist()
            d["z_mean"] = mean_z(z, nz)
        if s.type == "wl":
            d.update(A_ia=s.A_ia, eta_ia=s.eta_ia, z_piv_ia=s.z_piv_ia)
        if s.type == "nc":
            if s.bias == "srd":
                pref = s.bias_prefactor
                if pref is None:
                    pref = float(header.get("galaxy_bias_prefactor", 1.05))
                D = float(ccl.growth_factor(cosmo, z_to_a(d["z_mean"])))
                d["bias_value"] = pref / D
                notes[s.name] = f"srd bias {pref:g}/D(z_mean={d['z_mean']:.3f}) = {d['bias_value']:.4f}"
            else:
                d["bias_value"] = 1.0 if s.bias is None else float(s.bias)
            d["mag_bias"], d["has_rsd"] = s.mag_bias, s.has_rsd
        out.append(d)
    return out, notes


def _ell_grid(ell_min, ell_max, n_ell, spacing, ell_list):
    if ell_list:
        ell = np.unique(np.asarray(ell_list, dtype=int))
    elif spacing == "log":
        ell = np.unique(np.geomspace(ell_min, ell_max, n_ell).astype(int))
    else:
        ell = np.unique(np.linspace(ell_min, ell_max, n_ell).astype(int))
    if ell.min() < 2:
        raise ValueError("ell must be >= 2.")
    return ell


def _pairs(pairs, specs: list[TracerSpec]) -> list[tuple[str, str]]:
    names = [s.name for s in specs]
    if pairs == "all":
        return [(names[i], names[j]) for i in range(len(names)) for j in range(i, len(names))]
    if pairs == "auto":
        return [(n, n) for n in names]
    out = []
    for p in pairs:
        if len(p) != 2 or p[0] not in names or p[1] not in names:
            raise ValueError(f"pair {p} must name two tracers from {names}.")
        out.append((p[0], p[1]))
    return out


_SACC_TYPES = {("wl", "wl"): "galaxy_shear_cl_ee", ("wl", "nc"): "galaxy_shearDensity_cl_e",
               ("nc", "nc"): "galaxy_density_cl", ("cmb_lensing", "wl"): "cmbGalaxy_convergenceShear_cl_e",
               ("cmb_lensing", "nc"): "cmbGalaxy_convergenceDensity_cl", ("cmb_lensing", "cmb_lensing"): "cmb_convergence_cl"}
_SACC_ORDER = {"cmb_lensing": 0, "wl": 1, "nc": 2}


def _write_sacc_cls(path: Path, specs: list[TracerSpec], tracer_dicts: list[dict], pairs, ell, cls: dict) -> dict:
    """sacc file with NZ/Map tracers and C_ell data points (shear first in cross types)."""
    sacc = optional_import("sacc", "write_sacc", "pip install sacc")
    s = sacc.Sacc()
    types = {t.name: t.type for t in specs}
    for d in tracer_dicts:
        if d["type"] == "wl":
            s.add_tracer("NZ", d["name"], quantity="galaxy_shear", z=np.asarray(d["z"]), nz=np.asarray(d["nz"]))
        elif d["type"] == "nc":
            s.add_tracer("NZ", d["name"], quantity="galaxy_density", z=np.asarray(d["z"]), nz=np.asarray(d["nz"]))
        elif d["type"] == "cmb_lensing":
            lmax = int(np.max(ell)) + 1
            s.add_tracer("Map", d["name"], quantity="cmb_convergence", spin=0,
                         ell=np.arange(lmax + 1), beam=np.ones(lmax + 1))
    written, skipped = [], []
    for n1, n2 in pairs:
        t1, t2 = types[n1], types[n2]
        a, b = ((n1, t1), (n2, t2))
        if _SACC_ORDER.get(t2, 9) < _SACC_ORDER.get(t1, 9):
            a, b = b, a
        dtype = _SACC_TYPES.get((a[1], b[1]))
        if dtype is None:
            skipped.append([n1, n2])
            continue
        s.add_ell_cl(dtype, a[0], b[0], np.asarray(ell, dtype=int), np.asarray(cls[f"{n1}|{n2}"]))
        written.append([dtype, a[0], b[0]])
    s.save_fits(str(path), overwrite=True)
    return {"sacc_file": str(path), "sacc_data_points": len(s), "sacc_entries": written,
            "sacc_skipped_pairs": skipped, "sacc_covariance": False,
            "sacc_note": "no covariance and no bandpower windows (C_ell at the listed ells); "
                         "attach a covariance with the sacc tools before firecrown."}


# ---------------------------------------------------------------- tools
@validate_call
def ccl_describe_cosmology(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: Annotated[CosmologyParams | None, Field(description="Cosmology spec; None = vanilla LCDM (Omega_c 0.25, Omega_b 0.05, h 0.67, n_s 0.96, sigma8 0.81).")] = None,
    compute_sigma8: Annotated[bool, Field(description="Compute sigma8 from the linear P(k) (needs the transfer backend, e.g. camb); set False for a pure-background summary.")] = True,
) -> ArtifactResult:
    """Describe a cosmology: derived parameters (Omega_m, Omega_nu, sigma8 <-> A_s, H0, age) and a pyccl YAML when serialisable.

    Writes cosmology_<slug>.json (params, modelling switches, derived values)
    and cosmology_<slug>.yaml via pyccl's write_yaml when no baryon model,
    modified gravity or emulator is attached (pyccl cannot serialise those;
    the JSON is then the full spec). sigma8 is computed from the linear
    P(k) when compute_sigma8 (costs the transfer function: ~0.1 s camb).
    Call first to confirm what the other ccl_ tools will use.
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    cosmo = _build(spec, needs_pk=compute_sigma8)
    t0 = time.time()
    d = spec.describe()
    derived = d["derived"]
    derived.update({"Neff": float(cosmo["Neff"]), "m_nu_eV": spec.m_nu,
                    "age_Gyr": float(ccl.age_of_universe(cosmo, 1.0)),
                    "chi_z1_Mpc": float(ccl.comoving_radial_distance(cosmo, 0.5))})
    if compute_sigma8:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            derived["sigma8"] = float(ccl.sigma8(cosmo))
        A_s = float(cosmo["A_s"])
        derived["A_s"] = None if not np.isfinite(A_s) else A_s
    outdir = resolve_outdir(output_dir)
    slug = spec.slug()
    files = []
    yaml_path = outdir / f"cosmology_{slug}.yaml"
    yaml_note = "written"
    try:
        cosmo.write_yaml(str(yaml_path))
        files.append(str(yaml_path))
    except Exception as exc:  # noqa: BLE001 - pyccl refuses baryons/MG/emulators
        yaml_note = f"not written: {str(exc)[:160]}"
    payload = {**d, "yaml": yaml_note, "pyccl_version": ccl.__version__, "runtime_s": round(time.time() - t0, 3)}
    json_path = outdir / f"cosmology_{slug}.json"
    write_json(json_path, payload)
    files.insert(0, str(json_path))
    return ArtifactResult(status="success", files=files,
                          message=f"{spec.label()}: Omega_m={derived['Omega_m']:.4f}"
                                  + (f", sigma8={derived['sigma8']:.4f}" if "sigma8" in derived else "")
                                  + f", age={derived['age_Gyr']:.2f} Gyr. YAML {yaml_note}.",
                          metadata=payload)


@validate_call
def ccl_background(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: CosmologyParams | None = None,
    z_min: Annotated[float, Field(ge=0.0, le=20.0)] = 0.0,
    z_max: Annotated[float, Field(gt=0.0, le=1100.0)] = 3.0,
    n_z: Annotated[int, Field(ge=2, le=5000)] = 100,
    log_z: Annotated[bool, Field(description="Log-spaced z (requires z_min > 0).")] = False,
    make_plot: bool = True,
    return_data: Annotated[bool, Field(description="Include downsampled arrays in metadata.data.")] = False,
) -> ArtifactResult:
    """Background quantities vs z: comoving/angular-diameter/luminosity distances, distance modulus, E(z), H(z), growth D(z) and f(z), lookback time.

    Columns: z, a, chi_Mpc (comoving radial), D_A_Mpc, D_L_Mpc, distance_modulus
    (nan at z=0), E_z = H/H0, H_km_s_Mpc, growth_factor (D(z)/D(0)),
    growth_rate (f = dlnD/dlna), t_lookback_Gyr. All distances in Mpc (no h).
    Needs no transfer function (works without camb). Sanity: vanilla LCDM
    chi(z=1) ~ 3400 Mpc, D(z=1) ~ 0.6.
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    if log_z and z_min <= 0:
        raise ValueError("log_z needs z_min > 0.")
    if z_max <= z_min:
        raise ValueError("z_max must exceed z_min.")
    cosmo = _build(spec, needs_pk=False)
    z = np.geomspace(z_min, z_max, n_z) if log_z else np.linspace(z_min, z_max, n_z)
    a = z_to_a(z)
    t0 = time.time()
    chi = ccl.comoving_radial_distance(cosmo, a)
    D_A = ccl.angular_diameter_distance(cosmo, a)
    D_L = ccl.luminosity_distance(cosmo, a)
    mu = np.full_like(z, np.nan)
    pos = z > 0
    if pos.any():
        mu[pos] = ccl.distance_modulus(cosmo, a[pos])
    E = ccl.h_over_h0(cosmo, a)
    growth = ccl.growth_factor(cosmo, a)
    f = ccl.growth_rate(cosmo, a)
    t_lb = ccl.lookback_time(cosmo, a)
    columns = {"z": z, "a": a, "chi_Mpc": chi, "D_A_Mpc": D_A, "D_L_Mpc": D_L, "distance_modulus": mu,
               "E_z": E, "H_km_s_Mpc": 100.0 * spec.h * E, "growth_factor": growth, "growth_rate": f,
               "t_lookback_Gyr": t_lb}
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "z_min": z_min, "z_max": z_max, "n_z": n_z, "log": log_z})
    path = outdir / f"background_{slug}.csv"
    label = f"CCL background ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: background",
                              "units: z, a, chi [Mpc], D_A [Mpc], D_L [Mpc], mu [mag], E=H/H0, H [km/s/Mpc], D(z)/D(0), f, t [Gyr]",
                              f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with plt.rc_context(rc_params()):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
            for i, (lab, y) in enumerate([("chi (comoving)", chi), ("D_A", D_A), ("D_L", D_L)]):
                ax1.plot(z, y, color=PALETTE[i], linestyle=LINESTYLES[i], label=lab)
            ax1.set_xlabel("z"); ax1.set_ylabel("distance [Mpc]"); ax1.legend(fontsize=9)
            if log_z:
                ax1.set_xscale("log"); ax1.set_yscale("log")
            ax2.plot(z, growth, color=PALETTE[0], label="D(z)/D(0)")
            ax2.plot(z, f, color=PALETTE[1], linestyle="--", label="f = dlnD/dlna")
            ax2.plot(z, E, color=PALETTE[2], linestyle="-.", label="E(z) = H/H0")
            ax2.set_xlabel("z"); ax2.set_ylabel("growth / expansion"); ax2.set_yscale("log"); ax2.legend(fontsize=9)
            if log_z:
                ax2.set_xscale("log")
            fig.suptitle(label, fontsize=11)
            png = path.with_suffix(".png")
            fig.savefig(png, bbox_inches="tight", facecolor="white"); plt.close(fig)
        files.append(str(png))
    i1 = int(np.argmin(np.abs(z - 1.0)))
    metadata = {"cosmology": spec.firecrown_params(), "label": spec.label(), "z_range": [float(z_min), float(z_max)],
                "n_z": n_z, "units": {"distances": "Mpc", "H": "km/s/Mpc", "t_lookback": "Gyr"},
                "at_z_nearest_1": {"z": float(z[i1]), "chi_Mpc": float(chi[i1]), "D_A_Mpc": float(D_A[i1]),
                                   "growth_factor": float(growth[i1]), "growth_rate": float(f[i1])},
                "stats_chi": summary_stats(z, chi, "z", "chi_Mpc"), "runtime_s": round(time.time() - t0, 3)}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(status="success", files=files,
                          message=f"Background for {spec.label()} on {n_z} z in [{z_min:g}, {z_max:g}]; "
                                  f"chi(z={z[i1]:.2f}) = {chi[i1]:.1f} Mpc, D(z={z[i1]:.2f}) = {growth[i1]:.3f}.",
                          metadata=metadata)


@validate_call
def ccl_matter_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: Annotated[CosmologyParams | None, Field(description="Sets the backend too: transfer_function (boltzmann_camb|bbks|eisenstein_hu|boltzmann_class) and matter_power_spectrum (halofit|linear|camb_hmcode|cosmicemu_mt4|cosmicemu_mt2|bacco), baryons, mu-Sigma MG.")] = None,
    z: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0,
    k_min: Annotated[float, Field(gt=0.0, description="in k_units")] = 1e-3,
    k_max: Annotated[float, Field(gt=0.0, description="in k_units; CCL's spline covers k <= 50 1/Mpc, beyond it extrapolates")] = 20.0,
    n_k: Annotated[int, Field(ge=4, le=2000)] = 200,
    k_units: Annotated[KUnits, Field(description="'h/Mpc' -> P in (Mpc/h)^3 (emulator-server convention); '1/Mpc' -> P in Mpc^3 (CCL native).")] = "h/Mpc",
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """Linear and nonlinear matter power spectrum at one redshift from any pyccl backend (CAMB, analytic, HMcode, CosmicEmu, bacco).

    Columns (k_units='h/Mpc'): k_h_per_Mpc, Pk_lin_Mpc3_over_h3, Pk_nl_Mpc3_over_h3,
    ratio_nl_lin; with '1/Mpc': k_per_Mpc, Pk_lin_Mpc3, Pk_nl_Mpc3. Backend
    choices live in the cosmology (transfer_function / matter_power_spectrum /
    baryons / hmcode_logT_AGN). Metadata: sigma8 (as realised), k_nl, and
    'box' with in_training_box for emulator backends (cosmicemu: sigma8 0.7-0.9,
    z <= 2.02, w0 -1.3..-0.7; bacco: z <= 1.5): parameters outside the cosmicemu
    box are refused up front, z beyond the box is flagged (CCL extrapolates).
    Sanity (vanilla, halofit): P_nl(k=1 h/Mpc, z=0) ~ 400 (Mpc/h)^3, i.e. 855 Mpc^3
    at k=1 1/Mpc. Dispatchable (pip-kernel ccl.kernels.compute_matter_pk).
    Compare backends with pk-ccl-vs-emulators; the emulator server's
    compute_nonlinear_pk uses the same h/Mpc grid convention.
    """
    spec = _spec(cosmology)
    if k_max <= k_min:
        raise ValueError("k_max must exceed k_min.")
    params = spec.model_dump()
    box = emulator_box_check(params, z)
    if box.get("parameter_violations"):
        raise ValueError(f"{box['backend']} training box violated: {box['violations']}. "
                         "Change the parameters or use matter_power_spectrum='halofit'/'camb_hmcode'.")
    _check_backends(spec)
    k = _k_grid(k_min, k_max, n_k)
    t0 = time.time()
    from mcp_server.dispatch import remote_site, run_kernel  # lazy: server-only
    site = remote_site()
    if site:
        dep_key = "ccl_bacco" if spec.matter_power_spectrum == "bacco" else "ccl_core"
        res = run_kernel("ccl.kernels.compute_matter_pk",
                         {"cosmo_params": params, "z": z, "k_list": k.tolist(), "k_units": k_units},
                         pip_deps=BASE_PIP_DEPS + DISPATCH_PIP_DEPS[dep_key], duration=600)
        out, computed_on = res["result"], res.get("host", site)
    else:
        out, computed_on = _pk_kernel(params, z, k, k_units), "local"
    pk_lin = np.asarray(out["pk_lin"], float)
    pk_nl = np.asarray(out["pk_nl"], float)
    kcol, psuf = _k_cols(k_units)
    columns = {kcol: k, f"Pk_lin_{psuf}": pk_lin, f"Pk_nl_{psuf}": pk_nl, "ratio_nl_lin": pk_nl / pk_lin}
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "z": z, "k_min": k_min, "k_max": k_max, "n_k": n_k, "u": k_units})
    backend = f"{spec.matter_power_spectrum}/{spec.transfer_function}" + (f"+{spec.baryons}" if spec.baryons != "none" else "")
    label = f"CCL P(k) {backend} z={z:g} ({spec.label()})"
    path = outdir / f"matter_pk_{spec.matter_power_spectrum}_z{z:g}_{slug}.csv"
    punit = "(Mpc/h)^3" if k_units == "h/Mpc" else "Mpc^3"
    write_csv(path, columns, [f"label: {label}", "quantity: power_spectrum",
                              f"units: k [{k_units}], P [{punit}]", f"z: {z:g}", f"backend: {backend}",
                              f"k_units: {k_units}", f"cosmology: {spec.firecrown_params()}",
                              f"in_training_box: {box['in_training_box']}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, k, [("linear", pk_lin), (f"nonlinear ({spec.matter_power_spectrum})", pk_nl)],
                     xlabel=f"k [{k_units}]", ylabel=f"P(k) [{punit}]", title=label)
        files.append(str(png))
    i1 = int(np.argmin(np.abs(np.log(k) - np.log(1.0))))
    metadata = {"cosmology": spec.firecrown_params(), "backend": backend, "z": z, "k_units": k_units,
                "units": {"k": k_units, "P": punit}, "sigma8": out["sigma8"], "k_nl": out["k_nl"],
                "box": box, "computed_on": computed_on, "runtime_s": round(time.time() - t0, 3),
                "at_k_nearest_1": {kcol: float(k[i1]), "Pk_lin": float(pk_lin[i1]), "Pk_nl": float(pk_nl[i1])},
                "stats_nl": summary_stats(k, pk_nl, "k", "Pk_nl"), "label": label}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    msg = (f"P(k) {backend} at z={z:g} on {n_k} k in [{k_min:g}, {k_max:g}] {k_units}; "
           f"P_nl(k={k[i1]:.3g})={pk_nl[i1]:.4g} {punit}, sigma8={out['sigma8']:.4f}")
    if box["in_training_box"] is False:
        msg += f". WARNING outside emulator box: {box['violations']}"
    return ArtifactResult(status="success", files=files, message=msg + ".", metadata=metadata)


ccl_matter_pk.weight = "dispatchable"


@validate_call
def ccl_lsst_srd_nz(
    output_dir: Annotated[str, Field(min_length=1)],
    forecast_year: Annotated[Literal[1, 4, 7, 10], Field(description="LSST SRD forecast year (Y1/Y4/Y7/Y10).")] = 1,
    sample: Annotated[Literal["source", "lens"], Field(description="'source' (shear; 5 equal-number bins, sigma_z 0.05(1+z)) or 'lens' (clustering; 5 or 10 equal-width photo-z bins in 0.2<z<1.2, sigma_z 0.03(1+z)).")] = "source",
    z_max: Annotated[float, Field(gt=0.5, le=6.0)] = 3.5,
    n_z: Annotated[int, Field(ge=50, le=5000)] = 500,
    write_sacc: Annotated[bool, Field(description="Also add NZ tracers src{i}/lens{i} (quantity galaxy_shear/galaxy_density) to a sacc file.")] = False,
    sacc_file: Annotated[str | None, Field(description="sacc path for write_sacc; existing file is loaded and extended (so source + lens can share one file). Default <output_dir>/lsst_srd_nz_Y<yr>.sacc")] = None,
    make_plot: bool = True,
) -> ArtifactResult:
    """LSST DESC SRD tomographic n(z) for the source or lens sample of forecast year Y1/Y4/Y7/Y10 (pure numpy, no clone needed).

    Smail n(z) with the SRD Appendix D parameters (embedded from
    CCLX/parameters/lsst_desc_parameters.yaml), Gaussian photo-z convolution
    (Ma-Hu-Huterer erf), equal-number source bins / equal-width lens bins.
    CSV columns: z, nz_total, bin_0..bin_{N-1} (each normalised to unit
    integral; CCL renormalises anyway). Metadata: edges, centers (mean z per
    bin), n_gal_per_bin_arcmin2, sigma_z, f_sky (0.4363), galaxy_bias_prefactor
    (lens: b_i = prefactor/D(z_mean_i), see ccl_angular_cls bias='srd').
    Feed the CSV to ccl_angular_cls as nz_file with nz_column='bin_i', or to
    augur as the SRD n(z). Pure numpy; no pyccl needed.
    """
    res = srd_sample(forecast_year, sample, z_max=z_max, n_z=n_z)
    z = np.asarray(res["z"])
    columns = {"z": z, "nz_total": np.asarray(res["nz_total"])}
    for i, b in enumerate(res["bins"]):
        columns[f"bin_{i}"] = np.asarray(b)
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"yr": forecast_year, "s": sample, "z_max": z_max, "n_z": n_z})
    path = outdir / f"nz_{sample}_Y{forecast_year}_{slug}.csv"
    label = f"LSST SRD Y{forecast_year} {sample} n(z), {res['n_bins']} bins"
    header = [f"label: {label}", "quantity: nz", "units: z, n(z) [unit integral per column]",
              f"forecast_year: {forecast_year}", f"sample: {sample}", f"n_bins: {res['n_bins']}",
              f"edges: {[round(e, 4) for e in res['edges']]}", f"centers: {[round(c, 4) for c in res['centers']]}",
              f"n_gal_total_arcmin2: {res['n_gal_total_arcmin2']}",
              f"n_gal_per_bin_arcmin2: {[round(n, 4) for n in res['n_gal_per_bin_arcmin2']]}",
              f"sigma_z: {res['sigma_z']}", f"f_sky: {res['f_sky']}"]
    if sample == "lens":
        header.append(f"galaxy_bias_prefactor: {res['galaxy_bias_prefactor']}")
    header.append(f"reference: {SRD_REFERENCE}")
    write_csv(path, columns, header)
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, z, [("total", columns["nz_total"])] + [(f"bin {i}", columns[f"bin_{i}"]) for i in range(res["n_bins"])],
                     xlabel="z", ylabel="n(z) (unit integral)", title=label, logx=False, logy=False)
        files.append(str(png))
    metadata = {k: v for k, v in res.items() if k not in ("z", "nz_total", "bins")}
    metadata["columns"] = list(columns)
    metadata["tracer_prefix"] = "src" if sample == "source" else "lens"
    if write_sacc:
        sacc = optional_import("sacc", "write_sacc", "pip install sacc")
        spath = Path(sacc_file).expanduser().resolve() if sacc_file else outdir / f"lsst_srd_nz_Y{forecast_year}.sacc"
        s = sacc.Sacc.load_fits(str(spath)) if spath.exists() else sacc.Sacc()
        prefix, quantity = ("src", "galaxy_shear") if sample == "source" else ("lens", "galaxy_density")
        names = []
        for i, b in enumerate(res["bins"]):
            name = f"{prefix}{i}"
            s.tracers.pop(name, None)
            s.add_tracer("NZ", name, quantity=quantity, z=z, nz=np.asarray(b))
            names.append(name)
        spath.parent.mkdir(parents=True, exist_ok=True)
        s.save_fits(str(spath), overwrite=True)
        files.append(str(spath))
        metadata.update({"sacc_file": str(spath), "sacc_tracers_added": names, "sacc_all_tracers": list(s.tracers)})
    return ArtifactResult(status="success", files=files,
                          message=f"{label}: edges {[round(e, 3) for e in res['edges']]}, "
                                  f"n_gal {res['n_gal_total_arcmin2']:g}/arcmin^2 total, sigma_z={res['sigma_z']}(1+z)"
                                  + (f", sacc tracers {metadata['sacc_tracers_added']}" if write_sacc else "") + ".",
                          metadata=metadata)


@validate_call
def ccl_angular_cls(
    output_dir: Annotated[str, Field(min_length=1)],
    tracers: Annotated[list[TracerSpec], Field(min_length=1, description="Tracer specs: [{type:'wl', name:'src0', nz_file:..., nz_column:'bin_0', A_ia:1.0, eta_ia:-1.0}, {type:'nc', name:'lens0', nz_file:..., nz_column:'bin_0', bias:'srd', mag_bias:0.1}, {type:'cmb_lensing', name:'ck'}]")],
    cosmology: CosmologyParams | None = None,
    pairs: Annotated[Literal["all", "auto"] | list[list[str]], Field(description="'all' = every pair i<=j (3x2pt for src+lens), 'auto' = same-tracer only, or explicit [[name1,name2],...].")] = "all",
    ell_min: Annotated[int, Field(ge=2)] = 2,
    ell_max: Annotated[int, Field(le=50000)] = 3000,
    n_ell: Annotated[int, Field(ge=2, le=2000)] = 60,
    ell_spacing: Literal["log", "linear"] = "log",
    ell_list: Annotated[list[int] | None, Field(description="Explicit multipoles (overrides ell_min/ell_max/n_ell).")] = None,
    l_limber: Annotated[int | Literal["auto"], Field(description="-1: Limber everywhere; N: exact FKEM non-Limber for ell<N (CCLX uses N=ell_max); 'auto': CCL picks the switch (limber_max_error 1%).")] = -1,
    non_limber_pairs: Annotated[Literal["clustering", "all"], Field(description="Apply l_limber to pairs with a number-counts tracer only (default, as CCLX) or to all pairs.")] = "clustering",
    write_sacc: Annotated[bool, Field(description="Write a sacc file with NZ tracers + C_ell (galaxy_shear_cl_ee / galaxy_shearDensity_cl_e / galaxy_density_cl, shear first in cross pairs).")] = False,
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """Angular power spectra C_ell for any pairs of shear / clustering / CMB-lensing / tSZ / CIB / ISW tracers, Limber or FKEM non-Limber, optionally as a sacc file.

    This is the CCLX generate_dv_nonlimber recipe as a tool: n(z) from
    ccl_lsst_srd_nz (nz_column='bin_i'), SRD bias b=prefactor/D(z_mean)
    (bias='srd'), NLA IA (A_ia, eta_ia, z_piv_ia), magnification (mag_bias),
    l_limber=ell_max for exact non-Limber clustering. Output CSV: ell, then
    one column per pair cl_<name1>_<name2> (dimensionless C_ell; shear C_ell
    are E-mode). Metadata: pairs, kernel z ranges, l_limber_used per pair,
    bias values, runtime. write_sacc produces cls_<slug>.sacc with NZ tracers
    (quantity galaxy_shear/galaxy_density), CMB kappa as a Map tracer, and
    no covariance (attach one with the sacc tools; then sacc_inspect /
    firecrown). Shear-shear ~1e-9..1e-7 at ell~100-1000 for LSST Y1 bins;
    clustering ~1e-5..1e-4. Dispatchable (pip-kernel ccl.kernels.compute_angular_cls).
    Weight: light for Limber, ~0.1-0.3 s per non-Limber pair.
    """
    spec = _spec(cosmology)
    _check_backends(spec)
    ell = _ell_grid(ell_min, ell_max, n_ell, ell_spacing, ell_list)
    pair_list = _pairs(pairs, tracers)
    cosmo = _build(spec)  # cheap: splines are built lazily by the kernel
    tracer_dicts, bias_notes = _resolve_tracers(tracers, cosmo)
    params = spec.model_dump()
    t0 = time.time()
    from mcp_server.dispatch import remote_site, run_kernel
    site = remote_site()
    kargs = {"cosmo_params": params, "tracers": tracer_dicts, "pairs": [list(p) for p in pair_list],
             "ell_list": ell.tolist(), "l_limber": l_limber, "non_limber_pairs": non_limber_pairs}
    if site:
        res = run_kernel("ccl.kernels.compute_angular_cls", kargs,
                         pip_deps=BASE_PIP_DEPS + DISPATCH_PIP_DEPS["ccl_core"], duration=900)
        out, computed_on = res["result"], res.get("host", site)
    else:
        out, computed_on = _cls_kernel(**kargs), "local"
    runtime = time.time() - t0
    columns = {"ell": ell.astype(float)}
    for n1, n2 in pair_list:
        columns[f"cl_{n1}_{n2}"] = np.asarray(out["cls"][f"{n1}|{n2}"], float)
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "t": [t.model_dump() for t in tracers], "p": pair_list,
                       "ell": ell.tolist(), "ll": l_limber, "nl": non_limber_pairs})
    path = outdir / f"cls_{slug}.csv"
    label = f"CCL C_ell {len(pair_list)} pairs, l_limber={l_limber} ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: angular_cl", "units: ell, C_ell [dimensionless]",
                              f"pairs: {pair_list}", f"l_limber: {l_limber}", f"tracers: {[t.name + ':' + t.type for t in tracers]}",
                              f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, ell, [(f"{n1} x {n2}", columns[f"cl_{n1}_{n2}"]) for n1, n2 in pair_list],
                     xlabel="multipole ell", ylabel="C_ell", title=label)
        files.append(str(png))
    metadata = {"cosmology": spec.firecrown_params(), "pairs": [list(p) for p in pair_list],
                "tracers": {t.name: t.type for t in tracers}, "kernel_z_range": out["kernel_z_range"],
                "l_limber_requested": l_limber, "l_limber_used": out["l_limber_used"],
                "bias_values": {d["name"]: d.get("bias_value") for d in tracer_dicts if d["type"] == "nc"},
                "bias_notes": bias_notes, "ell_range": [int(ell.min()), int(ell.max())], "n_ell": int(len(ell)),
                "computed_on": computed_on, "runtime_s": round(runtime, 3), "label": label,
                "columns": list(columns)}
    if write_sacc:
        metadata.update(_write_sacc_cls(path.with_suffix(".sacc"), tracers, tracer_dicts, pair_list, ell, out["cls"]))
        files.append(metadata["sacc_file"])
    first = pair_list[0]
    y = columns[f"cl_{first[0]}_{first[1]}"]
    metadata["stats_first_pair"] = summary_stats(ell, y, "ell", f"cl_{first[0]}_{first[1]}")
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(status="success", files=files,
                          message=f"{len(pair_list)} C_ell on {len(ell)} ell in [{ell.min()}, {ell.max()}], "
                                  f"l_limber={l_limber} ({non_limber_pairs} pairs) in {runtime:.1f} s"
                                  + (f"; sacc with {metadata['sacc_data_points']} points" if write_sacc else "") + ".",
                          metadata=metadata)


ccl_angular_cls.weight = "dispatchable"


@validate_call
def ccl_correlation_functions(
    output_dir: Annotated[str, Field(min_length=1)],
    tracers: Annotated[list[TracerSpec], Field(min_length=1, description="Tracer specs as in ccl_angular_cls (wl/nc need nz_file).")],
    cosmology: CosmologyParams | None = None,
    pairs: Annotated[Literal["all", "auto"] | list[list[str]], Field(description="'all', 'auto' or explicit [[name1,name2],...].")] = "all",
    theta_min_arcmin: Annotated[float, Field(gt=0.0)] = 1.0,
    theta_max_arcmin: Annotated[float, Field(le=2000.0)] = 300.0,
    n_theta: Annotated[int, Field(ge=2, le=500)] = 30,
    ell_max_internal: Annotated[int, Field(ge=500, le=100000, description="Internal Limber C_ell grid 2..ell_max_internal (log, n_ell_internal points) fed to fftlog; raise for theta < 1 arcmin.")] = 20000,
    n_ell_internal: Annotated[int, Field(ge=50, le=5000)] = 400,
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """Real-space 2-point functions xi+/xi- (shear-shear), gamma_t (density-shear, kappa-shear) and w(theta) (density-density) from tracer specs, via CCL fftlog.

    C_ell are recomputed internally with Limber on a dense log grid up to
    ell_max_internal (20000) - the resolution needed by fftlog, which the
    ccl_angular_cls output grid does not have. Types follow the spins:
    (2,2) -> xip_<a>_<b>, xim_<a>_<b>; (0,2) -> gammat_<a>_<b>; (0,0) ->
    w_<a>_<b>. Output CSV: theta_arcmin, then one column per pair/type.
    Sanity (LSST-like source bin, vanilla): xi+(1') ~ 1e-4, xi+(300') ~ 1e-7.
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    _check_backends(spec)
    if theta_max_arcmin <= theta_min_arcmin:
        raise ValueError("theta_max_arcmin must exceed theta_min_arcmin.")
    pair_list = _pairs(pairs, tracers)
    cosmo = _build(spec)
    tracer_dicts, bias_notes = _resolve_tracers(tracers, cosmo)
    ell = np.unique(np.geomspace(2, ell_max_internal, n_ell_internal).astype(int))
    t0 = time.time()
    out = _cls_kernel(spec.model_dump(), tracer_dicts, [list(p) for p in pair_list], ell.tolist(), -1)
    theta = np.geomspace(theta_min_arcmin, theta_max_arcmin, n_theta)
    theta_deg = theta / 60.0
    spins = {t.name: t.spin() for t in tracers}
    columns = {"theta_arcmin": theta}
    types_used = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n1, n2 in pair_list:
            cl = np.asarray(out["cls"][f"{n1}|{n2}"], float)
            sp = (spins[n1], spins[n2])
            if sp == (2, 2):
                kinds = [("xip", "GG+"), ("xim", "GG-")]
            elif 2 in sp:
                kinds = [("gammat", "NG")]
            else:
                kinds = [("w", "NN")]
            for prefix, ctype in kinds:
                columns[f"{prefix}_{n1}_{n2}"] = ccl.correlation(cosmo, ell=ell.astype(float), C_ell=cl,
                                                                 theta=theta_deg, type=ctype, method="fftlog")
                types_used[f"{prefix}_{n1}_{n2}"] = ctype
    runtime = time.time() - t0
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "t": [t.model_dump() for t in tracers], "p": pair_list,
                       "th": [theta_min_arcmin, theta_max_arcmin, n_theta], "lm": ell_max_internal})
    path = outdir / f"xi_{slug}.csv"
    label = f"CCL xi(theta) {len(pair_list)} pairs ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: correlation_function", "units: theta [arcmin], xi [dimensionless]",
                              f"pairs: {pair_list}", f"types: {types_used}", f"ell_max_internal: {ell_max_internal}",
                              f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, theta, [(name, y) for name, y in columns.items() if name != "theta_arcmin"],
                     xlabel="theta [arcmin]", ylabel="correlation function", title=label)
        files.append(str(png))
    metadata = {"cosmology": spec.firecrown_params(), "pairs": [list(p) for p in pair_list], "types": types_used,
                "theta_range_arcmin": [theta_min_arcmin, theta_max_arcmin], "n_theta": n_theta,
                "ell_max_internal": ell_max_internal, "kernel_z_range": out["kernel_z_range"],
                "bias_notes": bias_notes, "runtime_s": round(runtime, 3), "label": label, "columns": list(columns)}
    first = next(k for k in columns if k != "theta_arcmin")
    metadata["stats_first"] = summary_stats(theta, columns[first], "theta_arcmin", first)
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(status="success", files=files,
                          message=f"{len(columns) - 1} correlation functions for {len(pair_list)} pairs on {n_theta} theta "
                                  f"in [{theta_min_arcmin:g}, {theta_max_arcmin:g}] arcmin ({runtime:.1f} s).",
                          metadata=metadata)


@validate_call
def ccl_correlation_3d(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: CosmologyParams | None = None,
    z: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0,
    r_min: Annotated[float, Field(gt=0.0, description="in r_units")] = 1.0,
    r_max: Annotated[float, Field(le=1000.0, description="in r_units")] = 200.0,
    n_r: Annotated[int, Field(ge=4, le=1000)] = 50,
    r_units: Annotated[Literal["Mpc/h", "Mpc"], Field(description="Separation units (CCL native is Mpc).")] = "Mpc/h",
    multipoles: Annotated[list[Literal[0, 2, 4]], Field(description="Redshift-space multipoles xi_l to compute (Kaiser, beta = f/b).")] = [0, 2, 4],
    galaxy_bias: Annotated[float, Field(gt=0.0, description="Linear bias b for beta = f(z)/b when beta is None.")] = 1.0,
    beta: Annotated[float | None, Field(description="RSD parameter f/b; None -> f(z)/galaxy_bias from the growth rate.")] = None,
    use_linear_pk: Annotated[bool, Field(description="Use the linear P(k) (BAO-scale theory) instead of the nonlinear one.")] = False,
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """3D matter correlation function xi(r) and Kaiser redshift-space multipoles xi_0/xi_2/xi_4 at one redshift.

    Columns: r_<units>, xi_r (real space, from correlation_3d), xi_0, xi_2,
    xi_4 (correlation_multipole with beta = f/b; b=galaxy_bias is NOT
    applied to the amplitude - multiply by b^2 yourself). r in Mpc/h by
    default (r_units='Mpc' for CCL native). The BAO peak sits near r ~ 105
    Mpc/h; xi(r) crosses zero around 130 Mpc/h. Uses the cosmology's P(k)
    backend (nonlinear unless use_linear_pk).
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    _check_backends(spec)
    if r_max <= r_min:
        raise ValueError("r_max must exceed r_min.")
    cosmo = _build(spec)
    r = np.geomspace(r_min, r_max, n_r)
    r_ccl = r / spec.h if r_units == "Mpc/h" else r
    a = float(z_to_a(z))
    pk = "delta_matter:delta_matter"
    if use_linear_pk:
        pk = cosmo.get_linear_power()
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = float(ccl.growth_rate(cosmo, a))
        beta_val = f / galaxy_bias if beta is None else float(beta)
        xi_r = ccl.correlation_3d(cosmo, r=r_ccl, a=a, p_of_k_a=pk)
        rcol = "r_Mpc_over_h" if r_units == "Mpc/h" else "r_Mpc"
        columns = {rcol: r, "xi_r": xi_r}
        for ell in sorted(set(multipoles)):
            columns[f"xi_{ell}"] = ccl.correlation_multipole(cosmo, r=r_ccl, a=a, beta=beta_val, ell=ell, p_of_k_a=pk)
    runtime = time.time() - t0
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "z": z, "r": [r_min, r_max, n_r], "u": r_units, "m": multipoles, "beta": beta_val, "lin": use_linear_pk})
    path = outdir / f"xi3d_z{z:g}_{slug}.csv"
    label = f"CCL xi(r) z={z:g} {'linear' if use_linear_pk else spec.matter_power_spectrum} ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: correlation_3d", f"units: r [{r_units}], xi [dimensionless]",
                              f"z: {z:g}", f"beta: {beta_val:.5f}", f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, r, [("xi(r) real space", xi_r)] + [(f"xi_{ell} (beta={beta_val:.2f})", columns[f"xi_{ell}"]) for ell in sorted(set(multipoles))],
                     xlabel=f"r [{r_units}]", ylabel="xi", title=label, logx=True, logy=False)
        files.append(str(png))
    metadata = {"cosmology": spec.firecrown_params(), "z": z, "r_units": r_units, "beta": beta_val, "growth_rate_f": f,
                "galaxy_bias": galaxy_bias, "multipoles": sorted(set(multipoles)), "use_linear_pk": use_linear_pk,
                "runtime_s": round(runtime, 3), "stats_xi_r": summary_stats(r, xi_r, "r", "xi_r"), "label": label,
                "columns": list(columns)}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(status="success", files=files,
                          message=f"xi(r) and multipoles {sorted(set(multipoles))} at z={z:g} on {n_r} r in [{r_min:g}, {r_max:g}] {r_units}, beta={beta_val:.3f}.",
                          metadata=metadata)


_HMF_NAMES = ["Press74", "Sheth99", "Jenkins01", "Tinker08", "Tinker10", "Angulo12", "Watson13", "Despali16", "Bocquet16", "Bocquet20", "Nishimichi19"]
_HBIAS_NAMES = ["Sheth99", "Sheth01", "Bhattacharya11", "Tinker10"]
_CONC_NAMES = ["Duffy08", "Bhattacharya13", "Diemer15", "Klypin11", "Prada12", "Ishiyama21", "Constant"]


def _halo_ingredients(mass_def: str, mass_function: str, halo_bias: str, concentration: str):
    import pyccl as ccl
    H = ccl.halos
    try:
        mdef = H.MassDef.from_name(mass_def)
    except Exception as exc:
        raise ValueError(f"mass_def {mass_def!r} not understood (use e.g. '200m', '200c', '500c', 'vir', 'fof'): {exc}") from exc
    try:
        hmf = H.MassFunc.from_name(mass_function)(mass_def=mdef)
        hbf = H.HaloBias.from_name(halo_bias)(mass_def=mdef)
        cM = H.Concentration.from_name(concentration)(mass_def=mdef)
    except Exception as exc:
        raise ValueError(f"halo-model ingredient incompatible with mass_def={mass_def!r}: {exc}. "
                         f"mass functions {_HMF_NAMES}; biases {_HBIAS_NAMES}; concentrations {_CONC_NAMES} "
                         "(Duffy08 supports 200m/200c/vir; Tinker08/10 need 'matter' or 'critical' Delta; "
                         "Bocquet20 needs MiraTitanHMFemulator, Nishimichi19 needs dark_emulator).") from exc
    return mdef, hmf, hbf, cM


@validate_call
def ccl_halo_mass_function(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: CosmologyParams | None = None,
    z_list: Annotated[list[float], Field(min_length=1, max_length=10, description="Redshifts (one column set per z).")] = [0.0],
    log10M_min: Annotated[float, Field(ge=8.0, le=16.0)] = 11.0,
    log10M_max: Annotated[float, Field(ge=8.0, le=17.0)] = 15.5,
    n_M: Annotated[int, Field(ge=4, le=500)] = 50,
    mass_def: Annotated[str, Field(description="'200m', '200c', '500c', 'vir', 'fof' or '<Delta>m'/'<Delta>c'.")] = "200m",
    mass_function: Annotated[Literal["Press74", "Sheth99", "Jenkins01", "Tinker08", "Tinker10", "Angulo12", "Watson13", "Despali16", "Bocquet16", "Bocquet20", "Nishimichi19"], Field(description="pyccl MassFunc name.")] = "Tinker08",
    halo_bias: Annotated[Literal["Sheth99", "Sheth01", "Bhattacharya11", "Tinker10"], Field(description="pyccl HaloBias name.")] = "Tinker10",
    concentration: Annotated[Literal["Duffy08", "Bhattacharya13", "Diemer15", "Klypin11", "Prada12", "Ishiyama21", "Constant"], Field(description="pyccl Concentration name.")] = "Duffy08",
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """Halo mass function dn/dlog10M, linear halo bias b(M), concentration c(M) and sigma(M) for one mass definition at one or more redshifts.

    Units: M in Msun (NOT Msun/h), dn/dlog10M in comoving Mpc^-3 (NOT
    (Mpc/h)^-3). To compare with the emulator server's compute_hmf
    (M in Msun/h, dn/dlnM in (Mpc/h)^-3): M_h = M*h, dn/dlnM = dn/dlog10M /
    ln(10), n_h = n / h^3. Columns: M_Msun, log10M, then per z:
    dndlog10M_Mpc3_z<z>, bias_z<z>, conc_z<z>, sigmaM_z<z>. Sanity: Tinker08,
    200m, vanilla, z=0: dn/dlog10M(1e14 Msun) ~ 5e-5 Mpc^-3. Uses the
    cosmology's linear P(k) (camb by default; bbks/eisenstein_hu are ~2x faster).
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    _check_backends(spec)
    if log10M_max <= log10M_min:
        raise ValueError("log10M_max must exceed log10M_min.")
    mdef, hmf, hbf, cM = _halo_ingredients(mass_def, mass_function, halo_bias, concentration)
    cosmo = _build(spec)
    M = np.logspace(log10M_min, log10M_max, n_M)
    columns = {"M_Msun": M, "log10M": np.log10(M)}
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for z in z_list:
            a = float(z_to_a(z))
            tag = f"z{z:g}"
            columns[f"dndlog10M_Mpc3_{tag}"] = hmf(cosmo, M, a)
            columns[f"bias_{tag}"] = hbf(cosmo, M, a)
            columns[f"conc_{tag}"] = cM(cosmo, M, a)
            columns[f"sigmaM_{tag}"] = ccl.sigmaM(cosmo, M, a)
    runtime = time.time() - t0
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "z": z_list, "M": [log10M_min, log10M_max, n_M], "md": mass_def,
                       "hmf": mass_function, "hb": halo_bias, "cm": concentration})
    path = outdir / f"hmf_{mass_function}_{mass_def}_{slug}.csv"
    label = f"CCL {mass_function} HMF ({mass_def}) z={z_list} ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: hmf_ccl",
                              "units: M [Msun, no h], dn/dlog10M [comoving Mpc^-3, no h^3], bias, concentration, sigma(M)",
                              f"mass_def: {mass_def}", f"mass_function: {mass_function}", f"halo_bias: {halo_bias}",
                              f"concentration: {concentration}", f"z_list: {z_list}", f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, M, [(f"z={z:g}", columns[f"dndlog10M_Mpc3_z{z:g}"]) for z in z_list],
                     xlabel=f"M_{mass_def} [Msun]", ylabel="dn/dlog10M [Mpc^-3]", title=label)
        files.append(str(png))
    i14 = int(np.argmin(np.abs(np.log10(M) - 14.0)))
    metadata = {"cosmology": spec.firecrown_params(), "z_list": z_list, "mass_def": mass_def, "mass_function": mass_function,
                "halo_bias": halo_bias, "concentration": concentration,
                "units": {"M": "Msun (no h)", "dn/dlog10M": "comoving Mpc^-3 (no h^3)"},
                "h_conversion": "M[Msun/h] = M*h; n[(Mpc/h)^-3] = n/h^3; dn/dlnM = dn/dlog10M/ln(10)",
                "at_M_nearest_1e14": {f"z{z:g}": {"M_Msun": float(M[i14]), "dndlog10M_Mpc3": float(columns[f"dndlog10M_Mpc3_z{z:g}"][i14]),
                                                  "bias": float(columns[f"bias_z{z:g}"][i14]), "conc": float(columns[f"conc_z{z:g}"][i14])} for z in z_list},
                "valid_names": {"mass_function": _HMF_NAMES, "halo_bias": _HBIAS_NAMES, "concentration": _CONC_NAMES},
                "runtime_s": round(runtime, 3), "label": label, "columns": list(columns)}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    z0 = z_list[0]
    return ArtifactResult(status="success", files=files,
                          message=f"{mass_function} dn/dlog10M ({mass_def}) for {n_M} masses 1e{log10M_min:g}-1e{log10M_max:g} Msun at z={z_list}; "
                                  f"at M={M[i14]:.2e} Msun, z={z0:g}: {columns[f'dndlog10M_Mpc3_z{z0:g}'][i14]:.3e} Mpc^-3, b={columns[f'bias_z{z0:g}'][i14]:.2f}.",
                          metadata=metadata)


@validate_call
def ccl_halo_model_pk(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: CosmologyParams | None = None,
    z: Annotated[float, Field(ge=0.0, le=6.0)] = 0.0,
    k_min: Annotated[float, Field(gt=0.0)] = 1e-3,
    k_max: Annotated[float, Field(gt=0.0)] = 20.0,
    n_k: Annotated[int, Field(ge=4, le=1000)] = 100,
    k_units: KUnits = "h/Mpc",
    profile: Annotated[Literal["nfw", "hod"], Field(description="'nfw': matter P(k) (NFW, truncated at R_Delta); 'hod': galaxy P(k) with an HOD (Profile2ptHOD for the 1-halo central-satellite term).")] = "nfw",
    hod_params: Annotated[dict[str, float], Field(description="HaloProfileHOD kwargs: log10Mmin_0=12, siglnM_0=0.4, log10M0_0=7, log10M1_0=13.3, alpha_0=1, fc_0=1, bg_0=1, bmax_0=1 (+ _p redshift slopes, a_pivot).")] = {},
    mass_def: str = "200m",
    mass_function: Literal["Press74", "Sheth99", "Jenkins01", "Tinker08", "Tinker10", "Angulo12", "Watson13", "Despali16", "Bocquet16"] = "Tinker08",
    halo_bias: Literal["Sheth99", "Sheth01", "Bhattacharya11", "Tinker10"] = "Tinker10",
    concentration: Literal["Duffy08", "Bhattacharya13", "Diemer15", "Klypin11", "Prada12", "Ishiyama21", "Constant"] = "Duffy08",
    log10M_min: Annotated[float, Field(ge=6.0, le=14.0)] = 8.0,
    log10M_max: Annotated[float, Field(ge=12.0, le=17.0)] = 16.0,
    n_M: Annotated[int, Field(ge=16, le=512)] = 128,
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """Halo-model power spectrum: 1-halo, 2-halo and total for matter (NFW) or galaxies (HOD), plus the cosmology's nonlinear P(k) for reference.

    Columns: k, Pk_1h, Pk_2h, Pk_hm (= 1h + 2h), Pk_nl_ref (nonlin_matter_power
    of the cosmology; for 'hod' it is the MATTER reference, so the ratio
    Pk_hm/Pk_nl_ref ~ b^2 on large scales). Units follow k_units (h/Mpc and
    (Mpc/h)^3 by default). The halo model under-predicts the 1h/2h transition
    by ~10-30% - that is a known feature, not a bug. ~0.5 s.
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    _check_backends(spec)
    if k_max <= k_min:
        raise ValueError("k_max must exceed k_min.")
    mdef, hmf, hbf, cM = _halo_ingredients(mass_def, mass_function, halo_bias, concentration)
    H = ccl.halos
    if profile == "nfw":
        prof, prof2pt = H.HaloProfileNFW(mass_def=mdef, concentration=cM), None
    else:
        try:
            prof = H.HaloProfileHOD(mass_def=mdef, concentration=cM, **hod_params)
        except TypeError as exc:
            raise ValueError(f"bad hod_params: {exc}") from exc
        prof2pt = H.Profile2ptHOD()
    cosmo = _build(spec)
    k = _k_grid(k_min, k_max, n_k)
    k_ccl = _to_ccl_k(k, k_units, spec.h)
    pfac = spec.h**3 if k_units == "h/Mpc" else 1.0
    a = float(z_to_a(z))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hmc = H.HMCalculator(mass_function=hmf, halo_bias=hbf, mass_def=mdef,
                             log10M_min=log10M_min, log10M_max=log10M_max, nM=n_M)
        p1h = H.halomod_power_spectrum(cosmo, hmc, k_ccl, a, prof, prof_2pt=prof2pt, get_1h=True, get_2h=False) * pfac
        p2h = H.halomod_power_spectrum(cosmo, hmc, k_ccl, a, prof, prof_2pt=prof2pt, get_1h=False, get_2h=True) * pfac
        pnl = ccl.nonlin_matter_power(cosmo, k_ccl, a) * pfac
    runtime = time.time() - t0
    kcol, psuf = _k_cols(k_units)
    columns = {kcol: k, f"Pk_1h_{psuf}": p1h, f"Pk_2h_{psuf}": p2h, f"Pk_hm_{psuf}": p1h + p2h, f"Pk_nl_ref_{psuf}": pnl}
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "z": z, "k": [k_min, k_max, n_k], "u": k_units, "prof": profile, "hod": hod_params,
                       "md": mass_def, "hmf": mass_function, "hb": halo_bias, "cm": concentration, "M": [log10M_min, log10M_max, n_M]})
    path = outdir / f"halomodel_pk_{profile}_z{z:g}_{slug}.csv"
    punit = "(Mpc/h)^3" if k_units == "h/Mpc" else "Mpc^3"
    label = f"CCL halo model P(k) {profile} {mass_function}/{mass_def} z={z:g} ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: halo_model_pk", f"units: k [{k_units}], P [{punit}]",
                              f"z: {z:g}", f"profile: {profile}", f"hod_params: {hod_params}", f"mass_def: {mass_def}",
                              f"mass_function: {mass_function}", f"halo_bias: {halo_bias}", f"concentration: {concentration}",
                              f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, k, [("1-halo", p1h), ("2-halo", p2h), ("halo model total", p1h + p2h), (f"reference {spec.matter_power_spectrum}", pnl)],
                     xlabel=f"k [{k_units}]", ylabel=f"P(k) [{punit}]", title=label)
        files.append(str(png))
    i1 = int(np.argmin(np.abs(np.log(k))))
    metadata = {"cosmology": spec.firecrown_params(), "z": z, "k_units": k_units, "profile": profile, "hod_params": hod_params,
                "mass_def": mass_def, "mass_function": mass_function, "halo_bias": halo_bias, "concentration": concentration,
                "mass_integration": {"log10M_min": log10M_min, "log10M_max": log10M_max, "n_M": n_M},
                "at_k_nearest_1": {kcol: float(k[i1]), "Pk_hm": float((p1h + p2h)[i1]), "Pk_nl_ref": float(pnl[i1]),
                                   "ratio_hm_over_ref": float((p1h + p2h)[i1] / pnl[i1])},
                "stats_hm": summary_stats(k, p1h + p2h, "k", "Pk_hm"), "runtime_s": round(runtime, 3), "label": label,
                "columns": list(columns)}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(status="success", files=files,
                          message=f"Halo-model P(k) ({profile}, {mass_function}, {mass_def}) at z={z:g}; at k={k[i1]:.3g} {k_units}: "
                                  f"P_hm={(p1h + p2h)[i1]:.4g} vs reference {pnl[i1]:.4g} {punit}.",
                          metadata=metadata)


BaryonModel = Literal["schneider15", "vandaalen19", "hmcode2020", "bacco"]


@validate_call
def ccl_baryon_boost(
    output_dir: Annotated[str, Field(min_length=1)],
    cosmology: CosmologyParams | None = None,
    z: Annotated[float, Field(ge=0.0, le=5.0)] = 0.0,
    k_min: Annotated[float, Field(gt=0.0)] = 0.01,
    k_max: Annotated[float, Field(gt=0.0)] = 20.0,
    n_k: Annotated[int, Field(ge=4, le=1000)] = 100,
    k_units: KUnits = "h/Mpc",
    models: Annotated[list[BaryonModel], Field(min_length=1, description="'schneider15' (BCM), 'vandaalen19' (fbar fit, k<=1 h/Mpc, z=0), 'hmcode2020' (CAMB HMcode-2020 feedback/no-feedback ratio, needs camb), 'bacco' (needs baccoemu, z<=1.5).")] = ["schneider15", "vandaalen19", "hmcode2020"],
    schneider15_params: Annotated[dict[str, float], Field(description="log10Mc=14.079, eta_b=0.5, k_s=55.0 [h/Mpc]")] = {},
    vandaalen19_fbar: Annotated[float, Field(ge=0.0, le=1.0, description="Baryon fraction in 500c halos relative to cosmic, vanDaalen19.")] = 0.7,
    hmcode_logT_AGN: Annotated[float, Field(ge=7.0, le=8.5)] = 7.8,
    bacco_params: Annotated[dict[str, float], Field(description="BaccoemuBaryons kwargs: log10_M_c, log10_eta, log10_beta, log10_M1_z0_cen, log10_theta_out, log10_theta_inn, log10_M_inn")] = {},
    make_plot: bool = True,
    return_data: bool = False,
) -> ArtifactResult:
    """Baryonic suppression S(k) = P_baryons / P_DMO per model (Schneider15 BCM, vanDaalen19, HMcode-2020 feedback, bacco), directly comparable with the emulator server's compute_baryon_suppression.

    Columns: k, boost_<model> (dimensionless; <1 where feedback removes
    power, >1 at k >~ 10 h/Mpc from stellar cooling). The CSV `quantity` is
    'suppression' like the emulator server's S(k) files so they overlay.
    hmcode2020 = P(mead2020_feedback, logT_AGN) / P(mead2020) from two CAMB
    runs. vandaalen19 is only calibrated for k <= 1 h/Mpc at z=0; bacco for
    z <= 1.5, k <= 4.9 h/Mpc. Sanity (defaults, z=0): schneider15 ~0.85 at
    k=5 h/Mpc; hmcode2020 logT_AGN=7.8 ~0.9 at k=5 h/Mpc.
    """
    import pyccl as ccl
    spec = _spec(cosmology)
    if k_max <= k_min:
        raise ValueError("k_max must exceed k_min.")
    if "hmcode2020" in models:
        optional_import("camb", "baryon model 'hmcode2020'", "pip install camb, or drop 'hmcode2020' from models.")
    if "bacco" in models:
        optional_import("baccoemu", "baryon model 'bacco'", "pip install baccoemu tensorflow-cpu, or drop 'bacco' from models.")
        if z > 1.5:
            raise ValueError("bacco baryons are only valid for z <= 1.5.")
    base = spec.model_copy(update={"baryons": "none"})
    _check_backends(base, needs_pk=("hmcode2020" in models))
    cosmo = base.build()
    k = _k_grid(k_min, k_max, n_k)
    k_ccl = _to_ccl_k(k, k_units, spec.h)
    a = float(z_to_a(z))
    t0 = time.time()
    columns = {_k_cols(k_units)[0]: k}
    notes = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for m in models:
            if m == "schneider15":
                p = {"log10Mc": 14.079, "eta_b": 0.5, "k_s": 55.0, **schneider15_params}
                columns["boost_schneider15"] = ccl.BaryonsSchneider15(**p).boost_factor(cosmo, k_ccl, a)
                notes[m] = f"BCM Schneider+15 {p} (k_s in h/Mpc)"
            elif m == "vandaalen19":
                columns["boost_vandaalen19"] = ccl.BaryonsvanDaalen19(fbar=vandaalen19_fbar).boost_factor(cosmo, k_ccl, a)
                notes[m] = f"van Daalen+19 fbar={vandaalen19_fbar} (calibrated k<=1 h/Mpc, z=0" + ("; z>0 extrapolated" if z > 0 else "") + ")"
            elif m == "hmcode2020":
                kw = base.ccl_kwargs()
                kw["matter_power_spectrum"] = "camb"
                kw["transfer_function"] = "boltzmann_camb"  # CCL needs CAMB's transfer for its nonlinear P(k)
                kw.pop("extra_parameters", None)
                c_fb = get_cached(f"ccl_hmcode_fb_{base.slug()}_{hmcode_logT_AGN}", lambda: ccl.Cosmology(
                    **kw, extra_parameters={"camb": {"halofit_version": "mead2020_feedback", "HMCode_logT_AGN": hmcode_logT_AGN, "kmax": 20.0}}))
                c_dmo = get_cached(f"ccl_hmcode_dmo_{base.slug()}", lambda: ccl.Cosmology(
                    **kw, extra_parameters={"camb": {"halofit_version": "mead2020", "kmax": 20.0}}))
                columns["boost_hmcode2020"] = ccl.nonlin_matter_power(c_fb, k_ccl, a) / ccl.nonlin_matter_power(c_dmo, k_ccl, a)
                notes[m] = f"CAMB HMcode-2020 feedback (logT_AGN={hmcode_logT_AGN}) / HMcode-2020 no feedback"
            elif m == "bacco":
                columns["boost_bacco"] = ccl.BaccoemuBaryons(**bacco_params).boost_factor(cosmo, k_ccl, a)
                notes[m] = f"baccoemu baryons {bacco_params or 'defaults'} (z<=1.5, k<=4.9 h/Mpc)"
    runtime = time.time() - t0
    outdir = resolve_outdir(output_dir)
    slug = param_slug({"c": spec.slug(), "z": z, "k": [k_min, k_max, n_k], "u": k_units, "m": models, "s15": schneider15_params,
                       "fbar": vandaalen19_fbar, "T": hmcode_logT_AGN, "bacco": bacco_params})
    path = outdir / f"baryon_boost_z{z:g}_{slug}.csv"
    label = f"CCL baryon S(k) {'+'.join(models)} z={z:g} ({spec.label()})"
    write_csv(path, columns, [f"label: {label}", "quantity: suppression", f"units: k [{k_units}], S(k)=P_baryons/P_DMO",
                              f"z: {z:g}", f"models: {notes}", f"cosmology: {spec.firecrown_params()}"])
    files = [str(path)]
    if make_plot:
        png = path.with_suffix(".png")
        _plot_curves(png, k, [(m, columns[f"boost_{m}"]) for m in models], xlabel=f"k [{k_units}]",
                     ylabel="S(k) = P_baryons / P_DMO", title=label, logx=True, logy=False, unity=True)
        files.append(str(png))
    kh = k if k_units == "h/Mpc" else k / spec.h
    i5 = int(np.argmin(np.abs(np.log(kh) - np.log(5.0))))
    at5 = {m: float(columns[f"boost_{m}"][i5]) for m in models}
    metadata = {"cosmology": spec.firecrown_params(), "z": z, "k_units": k_units, "models": notes,
                "at_k_nearest_5_h_per_Mpc": {"k_h_per_Mpc": float(kh[i5]), **at5},
                "min_boost": {m: {"value": float(np.min(columns[f"boost_{m}"])), "k": float(k[int(np.argmin(columns[f"boost_{m}"]))])} for m in models},
                "runtime_s": round(runtime, 3), "label": label, "columns": list(columns)}
    if return_data:
        metadata["data"] = downsample_columns(columns)
    return ArtifactResult(status="success", files=files,
                          message=f"S(k) for {models} at z={z:g}; at k~{kh[i5]:.2g} h/Mpc: " + ", ".join(f"{m}={v:.3f}" for m, v in at5.items()) + ".",
                          metadata=metadata)
