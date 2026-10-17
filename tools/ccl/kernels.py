"""HPC-dispatch kernels for the ccl family (dispatch contract R1).

JSON-safe in and out: the cosmology arrives as the CosmologyParams dict,
arrays as lists; the wrappers in ``__init__.py`` do files, plots, labels
and metadata wherever the server runs. pyccl/camb are imported inside the
functions so the package copies cleanly onto a compute node, where the
engine installs ``DISPATCH_PIP_DEPS["ccl_core"]`` (+ the base deps numpy,
scipy, pydantic that the tools package itself imports).
"""

import warnings

import numpy as np

DISPATCH_PIP_DEPS = {
    "ccl_core": ["pyccl", "camb"],
    "ccl_bacco": ["pyccl", "camb", "baccoemu", "tensorflow-cpu"],
}
BASE_PIP_DEPS = ["numpy", "scipy", "pydantic", "pyyaml"]

# Emulator training boxes enforced by pyccl's EmulatorPk classes; z limits
# are where CCL would otherwise extrapolate silently.
EMULATOR_BOXES = {
    "cosmicemu_mt4": {"sigma8": (0.7, 0.9), "h": (0.55, 0.85), "n_s": (0.85, 1.05),
                      "w0": (-1.3, -0.7), "omega_m_h2": (0.12, 0.155), "omega_b_h2": (0.0215, 0.0235),
                      "z_max": 2.02, "k_max_1_per_Mpc": 5.0,
                      "reference": "Mira-Titan IV (Moran et al. 2022), pyccl.CosmicemuMTIVPk"},
    "cosmicemu_mt2": {"sigma8": (0.7, 0.9), "h": (0.55, 0.85), "n_s": (0.85, 1.05),
                      "w0": (-1.3, -0.7), "omega_m_h2": (0.12, 0.155), "omega_b_h2": (0.0215, 0.0235),
                      "z_max": 2.02, "k_max_1_per_Mpc": 5.0,
                      "reference": "Mira-Titan II (Lawrence et al. 2017), pyccl.CosmicemuMTIIPk"},
    "bacco": {"sigma8": (0.73, 0.9), "h": (0.6, 0.8), "n_s": (0.92, 1.01),
              "w0": (-1.15, -0.85), "omega_m": (0.23, 0.4), "omega_b": (0.04, 0.06),
              "z_max": 1.5, "k_max_h_per_Mpc": 4.9,
              "reference": "baccoemu nonlinear (Angulo et al. 2021), pyccl.BaccoemuNonlinear"},
}


def emulator_box_check(cosmo_params: dict, z: float) -> dict:
    """{'backend', 'in_training_box', 'violations': [...]} for emulator P(k)
    backends; in_training_box is None for non-emulator backends."""
    mps = cosmo_params.get("matter_power_spectrum", "halofit")
    box = EMULATOR_BOXES.get(mps)
    if box is None:
        return {"backend": mps, "in_training_box": None, "violations": []}
    p = cosmo_params
    h = p.get("h", 0.67)
    omega_m = p.get("Omega_c", 0.25) + p.get("Omega_b", 0.05) + p.get("m_nu", 0.0) / (93.14 * h**2)
    values = {"sigma8": p.get("sigma8"), "h": h, "n_s": p.get("n_s", 0.96), "w0": p.get("w0", -1.0),
              "omega_m_h2": omega_m * h**2, "omega_b_h2": p.get("Omega_b", 0.05) * h**2,
              "omega_m": omega_m, "omega_b": p.get("Omega_b", 0.05)}
    violations = []
    for key, val in values.items():
        rng = box.get(key)
        if rng is None or val is None:
            continue
        if not (rng[0] <= val <= rng[1]):
            violations.append(f"{key}={val:.4g} outside [{rng[0]}, {rng[1]}]")
    if values["sigma8"] is None:
        violations.append("emulator backends need sigma8 (A_s given)")
    param_violations = list(violations)
    if z > box["z_max"]:
        violations.append(f"z={z:g} > z_max={box['z_max']} (CCL would extrapolate the emulator)")
    return {"backend": mps, "in_training_box": not violations, "violations": violations,
            "parameter_violations": param_violations,
            "box": {k: v for k, v in box.items() if k != "reference"}, "reference": box["reference"]}


def _build(cosmo_params: dict):
    from ..cosmology import CosmologyParams
    return CosmologyParams.model_validate(cosmo_params).build()


def compute_matter_pk(cosmo_params: dict, z: float, k_list, k_units: str = "h/Mpc") -> dict:
    """Linear and nonlinear P(k) at redshift z on the given k grid.

    k_list is in k_units ('h/Mpc' or '1/Mpc'); P is returned in the matching
    units ((Mpc/h)^3 or Mpc^3). Returns {"k", "pk_lin", "pk_nl", "sigma8",
    "k_nl", "box"} with lists, JSON-safe.
    """
    import pyccl as ccl
    k_in = np.asarray(k_list, dtype=float)
    h = float(cosmo_params.get("h", 0.67))
    k_ccl = k_in * h if k_units == "h/Mpc" else k_in
    pfac = h**3 if k_units == "h/Mpc" else 1.0
    a = 1.0 / (1.0 + float(z))
    box = emulator_box_check(cosmo_params, float(z))
    cosmo = _build(cosmo_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pk_lin = ccl.linear_matter_power(cosmo, k_ccl, a) * pfac
        pk_nl = ccl.nonlin_matter_power(cosmo, k_ccl, a) * pfac
        sigma8 = float(ccl.sigma8(cosmo))
        try:
            k_nl = float(ccl.kNL(cosmo, a))
        except Exception:  # noqa: BLE001 - diagnostic only
            k_nl = float("nan")
    k_nl_out = k_nl / h if k_units == "h/Mpc" else k_nl
    return {"k": k_in.tolist(), "pk_lin": pk_lin.tolist(), "pk_nl": pk_nl.tolist(),
            "sigma8": sigma8, "k_nl": k_nl_out, "k_units": k_units, "box": box}


def build_tracer(cosmo, spec: dict):
    """pyccl Tracer from a JSON tracer spec (z/nz arrays inline)."""
    import pyccl as ccl
    t = spec["type"]
    if t in ("wl", "nc"):
        z = np.asarray(spec["z"], dtype=float)
        nz = np.asarray(spec["nz"], dtype=float)
    if t == "wl":
        ia = None
        if spec.get("A_ia") is not None:
            A = float(spec["A_ia"]) * ((1.0 + z) / (1.0 + float(spec.get("z_piv_ia", 0.62)))) ** float(spec.get("eta_ia", 0.0))
            ia = (z, A)
        return ccl.WeakLensingTracer(cosmo, dndz=(z, nz), has_shear=True, ia_bias=ia, use_A_ia=True)
    if t == "nc":
        bias = spec.get("bias_value")
        b = None if bias is None else (z, np.full_like(z, float(bias)))
        mag = spec.get("mag_bias")
        s = None if mag is None else (z, np.full_like(z, float(mag)))
        return ccl.NumberCountsTracer(cosmo, dndz=(z, nz), bias=b, mag_bias=s,
                                      has_rsd=bool(spec.get("has_rsd", False)))
    if t == "cmb_lensing":
        return ccl.CMBLensingTracer(cosmo, z_source=float(spec.get("z_source", 1100.0)))
    if t == "tsz":
        return ccl.tSZTracer(cosmo, z_max=float(spec.get("z_max", 6.0)))
    if t == "cib":
        return ccl.CIBTracer(cosmo, z_max=float(spec.get("z_max", 6.0)))
    if t == "isw":
        return ccl.ISWTracer(cosmo, z_max=float(spec.get("z_max", 6.0)))
    raise ValueError(f"unknown tracer type {t!r}")


def compute_angular_cls(cosmo_params: dict, tracers: list, pairs: list, ell_list,
                        l_limber=-1, non_limber_pairs: str = "clustering") -> dict:
    """C_ell for each (name1, name2) pair.

    tracers: list of dicts {type, name, z, nz, bias_value, mag_bias, has_rsd,
    A_ia, eta_ia, z_piv_ia, z_source, z_max}. l_limber: -1 (Limber), int
    (FKEM non-Limber below it) or 'auto'; applied to pairs containing a
    number-counts tracer ('clustering') or to every pair ('all').
    Returns {"ell", "cls": {"n1|n2": [...]}, "l_limber_used": {...},
    "kernel_z_range": {name: [z_min, z_max]}}.
    """
    import pyccl as ccl
    cosmo = _build(cosmo_params)
    ell = np.asarray(ell_list, dtype=float)
    built = {}
    zrange = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for spec in tracers:
            tr = build_tracer(cosmo, spec)
            built[spec["name"]] = (spec["type"], tr)
            chi_min, chi_max = float(tr.chi_min), float(tr.chi_max)
            zrange[spec["name"]] = [
                float(1.0 / ccl.scale_factor_of_chi(cosmo, chi_min) - 1.0) if chi_min > 0 else 0.0,
                float(1.0 / ccl.scale_factor_of_chi(cosmo, chi_max) - 1.0)]
        cls, used = {}, {}
        for n1, n2 in pairs:
            t1, tr1 = built[n1]
            t2, tr2 = built[n2]
            nonlimber = l_limber != -1 and (non_limber_pairs == "all" or "nc" in (t1, t2))
            ll = l_limber if nonlimber else -1
            cl, meta = ccl.angular_cl(cosmo, tr1, tr2, ell, l_limber=ll,
                                      non_limber_integration_method="FKEM", return_meta=True)
            key = f"{n1}|{n2}"
            cls[key] = np.asarray(cl, dtype=float).tolist()
            used[key] = int(meta.get("l_limber", ll)) if meta.get("l_limber") is not None else int(ll)
    return {"ell": ell.tolist(), "cls": cls, "l_limber_used": used, "kernel_z_range": zrange}
