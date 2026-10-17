"""LSST DESC SRD redshift distributions and tomographic binning, pure numpy.

Re-implementation of ``CCLX/lsst_galaxy_sample.py`` (N. Sarcevic) with the
per-year parameter table of ``CCLX/parameters/lsst_desc_parameters.yaml``
embedded below, so it needs no clone path, no yaml, no pandas and no
``np.trapz``. The physics is the LSST DESC Science Requirements Document
(arXiv:1809.01669, Appendix D):

* overall sample  n(z) ~ (z/z0)^beta exp[-(z/z0)^alpha]   (SRD eq. 5)
* photo-z bins    Ma, Hu & Huterer 2006 (astro-ph/0506614, eq. 6):
                  n_i(z) = n(z)/2 [erf((z_hi - z + dz)/(sqrt2 s)) - erf((z_lo - z + dz)/(sqrt2 s))],
                  s = sigma_z (1+z), dz = z_bias
* lens bins       equal width in photo-z between 0.2 and 1.2
                  (5 bins of 0.2 for Y1/Y4, 10 bins of 0.1 for Y7/Y10)
* source bins     5 bins with equal galaxy numbers (edges from the CDF)

Everything here is JSON-safe and importable without pyccl, so it can run
inside a dispatch kernel or an inner script.
"""

import numpy as np
from scipy.special import erf

# From CCLX/parameters/lsst_desc_parameters.yaml (LSST DESC SRD, App. D).
# n_gal in galaxies/arcmin^2; sigma_z is the photo-z scatter prefactor
# (scatter = sigma_z (1+z)); z_bias = 0 everywhere.
SRD_PARAMS: dict[str, dict[int, dict]] = {
    "lens": {
        1: {"z_0": 0.26, "alpha": 0.94, "beta": 2.0, "bin_start": 0.2, "bin_stop": 1.2,
            "bin_spacing": 0.2, "n_tomo_bins": 5, "sigma_z": 0.03, "z_bias": 0.0,
            "n_gal": 18.0, "galaxy_bias_prefactor": 1.05, "limiting_magnitude": 24.1},
        4: {"z_0": 0.268, "alpha": 0.92, "beta": 2.0, "bin_start": 0.2, "bin_stop": 1.2,
            "bin_spacing": 0.2, "n_tomo_bins": 5, "sigma_z": 0.03, "z_bias": 0.0,
            "n_gal": 28.0, "galaxy_bias_prefactor": 1.05, "limiting_magnitude": 24.5},
        7: {"z_0": 0.272, "alpha": 0.91, "beta": 2.0, "bin_start": 0.2, "bin_stop": 1.2,
            "bin_spacing": 0.1, "n_tomo_bins": 10, "sigma_z": 0.03, "z_bias": 0.0,
            "n_gal": 38.0, "galaxy_bias_prefactor": 0.95, "limiting_magnitude": 24.9},
        10: {"z_0": 0.28, "alpha": 0.9, "beta": 2.0, "bin_start": 0.2, "bin_stop": 1.2,
             "bin_spacing": 0.1, "n_tomo_bins": 10, "sigma_z": 0.03, "z_bias": 0.0,
             "n_gal": 48.0, "galaxy_bias_prefactor": 0.95, "limiting_magnitude": 25.3},
    },
    "source": {
        1: {"z_0": 0.13, "alpha": 0.78, "beta": 2.0, "n_tomo_bins": 5, "sigma_z": 0.05,
            "z_bias": 0.0, "n_gal": 10.0, "mean_eff_z": 0.85, "limiting_magnitude": 24.1},
        4: {"z_0": 0.12, "alpha": 0.75, "beta": 2.0, "n_tomo_bins": 5, "sigma_z": 0.05,
            "z_bias": 0.0, "n_gal": 15.7, "mean_eff_z": 0.85, "limiting_magnitude": 24.5},
        7: {"z_0": 0.12, "alpha": 0.71, "beta": 2.0, "n_tomo_bins": 5, "sigma_z": 0.05,
            "z_bias": 0.0, "n_gal": 21.3, "mean_eff_z": 1.05, "limiting_magnitude": 24.9},
        10: {"z_0": 0.11, "alpha": 0.68, "beta": 2.0, "n_tomo_bins": 5, "sigma_z": 0.05,
             "z_bias": 0.0, "n_gal": 27.0, "mean_eff_z": 1.05, "limiting_magnitude": 25.3},
    },
}
SRD_SKY = {"frac_sky": 0.4363, "full_sky_deg2": 41252.9612, "lsst_sky_deg2": 18000.0}
SRD_REFERENCE = ("LSST DESC Science Requirements Document, arXiv:1809.01669, Appendix D; "
                 "parameters from CCLX/parameters/lsst_desc_parameters.yaml")
FORECAST_YEARS = (1, 4, 7, 10)


def smail(z, z0: float, alpha: float, beta: float = 2.0) -> np.ndarray:
    """Un-normalised Smail-type n(z) = (z/z0)^beta exp[-(z/z0)^alpha]."""
    z = np.asarray(z, dtype=float)
    return (z / z0) ** beta * np.exp(-((z / z0) ** alpha))


def photoz_bin(z, nz, z_lo: float, z_hi: float, sigma_z: float, z_bias: float = 0.0) -> np.ndarray:
    """n(z) of galaxies whose photo-z falls in [z_lo, z_hi] (Gaussian photo-z,
    scatter sigma_z (1+z), bias z_bias). Un-normalised: its integral over z is
    the fraction of n(z) landing in the bin."""
    z = np.asarray(z, dtype=float)
    scatter = np.maximum(sigma_z * (1.0 + z), 1e-10)
    upper = (z_hi - z + z_bias) / (np.sqrt(2.0) * scatter)
    lower = (z_lo - z + z_bias) / (np.sqrt(2.0) * scatter)
    return 0.5 * np.asarray(nz, dtype=float) * (erf(upper) - erf(lower))


def equal_number_edges(z, nz, n_bins: int) -> list[float]:
    """Redshift edges splitting n(z) into n_bins bins of equal integral."""
    z = np.asarray(z, dtype=float)
    nz = np.asarray(nz, dtype=float)
    dz = np.diff(z)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (nz[1:] + nz[:-1]) * dz)])
    total = cdf[-1]
    inner = [float(np.interp(i / n_bins * total, cdf, z)) for i in range(1, n_bins)]
    return [float(z[0])] + inner + [float(z[-1])]


def normalize(z, nz) -> np.ndarray:
    nz = np.asarray(nz, dtype=float)
    norm = np.trapezoid(nz, x=z)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("n(z) integrates to zero; check the redshift grid.")
    return nz / norm


def mean_z(z, nz) -> float:
    nz = np.asarray(nz, dtype=float)
    return float(np.trapezoid(nz * z, x=z) / np.trapezoid(nz, x=z))


def srd_sample(forecast_year: int, sample: str, z_max: float = 3.5, n_z: int = 500,
               normalize_bins: bool = True) -> dict:
    """SRD n(z) and tomographic bins for one sample and forecast year.

    Returns a JSON-safe dict: z, nz_total (normalised), bins (list of arrays,
    normalised to unit integral when normalize_bins), edges, centers (mean z
    per bin), n_gal_per_bin [arcmin^-2] (total n_gal split by the fraction of
    galaxies each photo-z bin captures), sigma_z, z_bias, n_gal_total, f_sky,
    galaxy_bias_prefactor (lens only), and the SRD parameters used.
    """
    forecast_year = int(forecast_year)
    if forecast_year not in FORECAST_YEARS:
        raise ValueError(f"forecast_year must be one of {FORECAST_YEARS}, got {forecast_year}.")
    if sample not in ("lens", "source"):
        raise ValueError("sample must be 'lens' or 'source'.")
    p = SRD_PARAMS[sample][forecast_year]
    z = np.linspace(0.0, float(z_max), int(n_z))
    nz_total = normalize(z, smail(z, p["z_0"], p["alpha"], p["beta"]))
    n_bins = p["n_tomo_bins"]
    if sample == "lens":
        # equal-width photo-z bins; linspace avoids the floating-point
        # over-run of np.arange(start, stop + step, step) for step=0.1
        edges = np.round(np.linspace(p["bin_start"], p["bin_stop"], n_bins + 1), 6).tolist()
    else:
        edges = equal_number_edges(z, nz_total, n_bins)
    raw = [photoz_bin(z, nz_total, lo, hi, p["sigma_z"], p["z_bias"])
           for lo, hi in zip(edges[:-1], edges[1:])]
    fractions = np.array([np.trapezoid(b, x=z) for b in raw])
    bins = [normalize(z, b) if normalize_bins else b for b in raw]
    centers = [mean_z(z, b) for b in bins]
    n_gal_per_bin = (p["n_gal"] * fractions / fractions.sum()).tolist()
    out = {
        "forecast_year": forecast_year, "sample": sample,
        "z": z.tolist(), "nz_total": nz_total.tolist(),
        "bins": [b.tolist() for b in bins],
        "edges": [float(e) for e in edges], "centers": centers,
        "n_bins": n_bins, "binning": "equal_width" if sample == "lens" else "equal_number",
        "n_gal_total_arcmin2": p["n_gal"], "n_gal_per_bin_arcmin2": n_gal_per_bin,
        "bin_fraction_of_sample": fractions.tolist(),
        "sigma_z": p["sigma_z"], "z_bias": p["z_bias"],
        "f_sky": SRD_SKY["frac_sky"], "sky_area_deg2": SRD_SKY["lsst_sky_deg2"],
        "smail": {"z_0": p["z_0"], "alpha": p["alpha"], "beta": p["beta"]},
        "reference": SRD_REFERENCE,
    }
    if sample == "lens":
        out["galaxy_bias_prefactor"] = p["galaxy_bias_prefactor"]
        out["galaxy_bias_note"] = "SRD linear bias b_i = galaxy_bias_prefactor / D(z_mean_i)"
    else:
        out["mean_eff_z"] = p["mean_eff_z"]
    return out
