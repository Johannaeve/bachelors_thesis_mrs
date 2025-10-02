
from joblib import Parallel, delayed
from tqdm import tqdm
from joblib import parallel_backend
import numpy as np
from .fitting import (fit_1_voigt_NAA_Cr_Ch, fit_2_voigt_NAA_Cr_Ch, fit_1_lorentz_NAA_Cr_Ch, fit_2_lorentz_NAA_Cr_Ch, fit_1_voigt_GABA, fit_2_voigt_GABA, apply_phase)
### helpers

def failed_fit_output(spec, p0, label):
    # NaN output if the fit fails
    return {
        "fit_1": np.full_like(spec, np.nan),
        "baseline_1": np.full_like(spec, np.nan),
        "fit_2": np.full_like(spec, np.nan),
        "baseline_2": np.full_like(spec, np.nan),
        "residual_1": np.full_like(spec, np.nan),
        "residual_2": np.full_like(spec, np.nan),
        "spec_cor": spec,
        label: (np.nan, np.nan, np.nan, np.nan),
        "popt": np.full_like(p0, np.nan),
        "crlbs": np.full_like(p0, np.nan)
    }

def failed_fit_output_nobaseline(spec, p0, label):
    # NaN output if the fit fails
    return {
        "fit_1": np.full_like(spec, np.nan),
        "fit_2": np.full_like(spec, np.nan),
        "residual_1": np.full_like(spec, np.nan),
        "residual_2": np.full_like(spec, np.nan),
        "spec_cor": spec,
        label: np.nan,
        "popt": np.full_like(p0, np.nan),
        "crlbs": np.full_like(p0, np.nan)
    }

### NAA, Cr, Ch Voigt

def parallel_voigt_fit_NAA_Cr_Ch(records, base='spline', n_jobs=4):
    # parallel function, input and output a list of dictionaries
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_voigt_helper_NAA_Cr_Ch)(rec, base)
            for rec in tqdm(records))
    return results
    
def fit_voigt_helper_NAA_Cr_Ch(rec, base):
    freq    = rec["freq"]
    spec    = rec["spec"]
    exclude = rec["exclude"]
    p0      = rec["p0_v"]
    bounds  = rec["bounds_v"]

    best_result = None
    lowest_residual_norm = np.inf

    for delta in [-0.9,-0.3, 0.0, 0.3, 0.9]:
        p0_try = p0.copy()
        p0_try[16] = delta  # index 16 = ph0
        try:
            f1, b1, r1, p1 = fit_1_voigt_NAA_Cr_Ch(freq, spec, p0_try, bounds, exclude, base)
            norm = np.linalg.norm(r1)
            if norm < lowest_residual_norm:
                best_result = (f1, b1, r1, p1)
                lowest_residual_norm = norm
        except Exception:
            continue

    if best_result is None:
        return failed_fit_output(spec, p0, label="areas")
    
    # Use best first-step result for second fit
    f1, b1, r1, p1 = best_result
    ph0 = p1[16]
    ph1 = p1[17]
    spec_cor = apply_phase(spec, freq, ph0, ph1)
    try:
        # Trim baseline and fit second step
        f2, b2, r2, popt, crlbs = fit_2_voigt_NAA_Cr_Ch(freq,spec_cor,p1,bounds,exclude,base)
        return {
            "fit_1": f1,
            "baseline_1": b1,
            "fit_2": f2,
            "baseline_2": b2,
            "residual_1": r1,
            "residual_2": r2,
            "spec_cor": spec_cor,
            "areas": (popt[3], popt[7], popt[11], popt[14]),
            "phase": (ph0,ph1),
            "popt": popt,
            "crlbs": crlbs
        }
    except Exception:
        return failed_fit_output(spec, p0, label="areas")
    
### NAA, Cr, Ch Lorentz

def parallel_lorentz_fit_NAA_Cr_Ch(records, base='spline', n_jobs=4):
    # parallel function, input and output a list of dictionaries
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_lorentz_helper_NAA_Cr_Ch)(rec, base)
            for rec in tqdm(records))
    return results

def fit_lorentz_helper_NAA_Cr_Ch(rec, base):
    freq    = rec["freq"]
    spec    = rec["spec"]
    exclude = rec["exclude"]
    p0      = rec["p0_l"]
    bounds  = rec["bounds_l"]

    best_result = None
    lowest_residual_norm = np.inf

    for delta in [-0.9,-0.3, 0.0, 0.3, 0.9]:
        p0_try = p0.copy()
        p0_try[12] = delta  # Lorentz: ph0 is index 12

        try:
            f1, b1, r1, p1 = fit_1_lorentz_NAA_Cr_Ch(freq, spec, p0_try, bounds, exclude, base)
            norm = np.linalg.norm(r1)
            if norm < lowest_residual_norm:
                best_result = (f1, b1, r1, p1)
                lowest_residual_norm = norm
        except Exception:
            continue

    if best_result is None:
        return failed_fit_output(spec, p0, label="amps")

    f1, b1, r1, p1 = best_result
    ph0 = p1[12]
    ph1 = p1[13]
    spec_cor = apply_phase(spec, freq, ph0, ph1)

    try:
        f2, b2, r2, popt, crlbs = fit_2_lorentz_NAA_Cr_Ch(freq, spec_cor, p1, bounds, exclude, base)
        return {
            "fit_1": f1,
            "baseline_1": b1,
            "fit_2": f2,
            "baseline_2": b2,
            "residual_1": r1,
            "residual_2": r2,
            "spec_cor": spec_cor,
            "amps": (popt[2], popt[5], popt[8], popt[11]),  # Lorentz amplitudes
            "phase": (ph0,ph1),
            "popt": popt,
            "crlbs": crlbs
        }

    except Exception:
        return failed_fit_output(spec, p0, label="amps")
    
#____________________________GABA____________________________________________________________________________

### Voigt

def parallel_voigt_fit_GABA(records, n_jobs=4, backend="loky"):
    with parallel_backend(backend, inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_voigt_helper_GABA)(rec) for rec in records)
    return results

def fit_voigt_helper_GABA(rec):
    # rec must contain: freq, spec, peak (mask), center_freq, shift
    freq = rec["freq"]
    spec = rec["spec"]
    peak_mask = rec["peak"]
    center_freq = rec["center_freq"]
    shift = rec.get("shift", 0.0)

    # step 1: phase fit
    f1, r1, area1, p1 = fit_1_voigt_GABA(freq, spec, shift, center_freq, peak_mask)
    ph0, ph1 = p1[6], p1[7]
    spec_cor = apply_phase(spec, freq, ph0, ph1)

    # step 2: refit without phase (no baseline)
    f2, r2, area2, popt, crlbs = fit_2_voigt_GABA(freq, spec_cor, shift, center_freq, peak_mask)

    return {
        "fit_1": f1,
        "residual_1": r1,
        "fit_2": f2,
        "residual_2": r2,
        "spec_cor": spec_cor,
        "areas": area2,                 
        "popt": popt,                   # [w0, delta, sigma, gamma, A, dA]
        "crlbs": crlbs
    }


