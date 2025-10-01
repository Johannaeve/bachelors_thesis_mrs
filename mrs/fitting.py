import numpy as np
import scipy.optimize as opt
from numpy.polynomial.chebyshev import chebval
import scipy.special as spsp
from scipy.interpolate import CubicSpline

### Helpers

def ppm2freq(ppm,center_freq, ref_ppm=4.7):
    ### Converts ppm axis to Hz
    return (ref_ppm - ppm) * center_freq

def freq2ppm(freq, center_freq, ref_ppm=4.7):
    ### Converts Hz axis to ppm
    return ref_ppm - (freq / (center_freq))

def apply_phase(spectrum, freqs, p0, p1):
    ### Applys phase shift to the spectrum; freq must be in Hz
    phase = p0 + p1 * freqs * 2 * np.pi # In radians and radians per Hz
    return spectrum * np.exp(1j * phase)

def new_base(popt, bounds, freq, n_base_2, model_type='voigt'):
    ### New baseline start params for second fitting step
    lower, upper = bounds
    if model_type == 'voigt':
        n_peaks = 16  # 4 Voigt peaks × 4
        n_phase = 2   # ph0, ph1
    elif model_type == 'lorentz':
        n_peaks = 12  # 4 Lorentz peaks × 3
        n_phase = 2   # ph0, ph1    
    baseline_start = n_peaks + n_phase
    if n_base_2 is not None:
        n_knots_first = len(popt[baseline_start:])
        n_knots_second = n_base_2 + 1

        old_spline_values = popt[baseline_start:]
        old_knots = np.linspace(freq.min(), freq.max(), n_knots_first)
        new_knots = np.linspace(freq.min(), freq.max(), n_knots_second)

        # Interpolate old spline to new knot positions
        interp = CubicSpline(old_knots, old_spline_values, bc_type='natural')
        new_spline_values = interp(new_knots)

        # Build new p0 and bounds
        p0_trimmed = np.concatenate([popt[:n_peaks], new_spline_values])
        lower_spline = [-1e5] * n_knots_second
        upper_spline = [1e5] * n_knots_second
        lower_bounds = np.concatenate([lower[:n_peaks], lower_spline])
        upper_bounds = np.concatenate([upper[:n_peaks], upper_spline])

    else:
        # Keep same number of baseline coeffs
        p0_trimmed = np.concatenate([popt[:n_peaks], popt[baseline_start:]])
        lower_bounds = np.concatenate([lower[:n_peaks], lower[baseline_start:]])
        upper_bounds = np.concatenate([upper[:n_peaks], upper[baseline_start:]])

    bounds_trimmed = (lower_bounds, upper_bounds)
    return p0_trimmed, bounds_trimmed


def residuals(x, y, model_func, params):
    ### Residual function for least squares
    model = model_func(x, *params)
    return model - y

def covariance_least_squares(result):
    ### Estimates pcov for least squares as done in curvefit
    residuals = result.fun
    J = result.jac
    n = len(residuals)
    p = len(result.x)
    if n > p:
        sigma2 = np.sum(residuals**2) / (n - p)
    else:
        sigma2 = np.sum(residuals**2)  # fallbac
    try:
        JTJ_inv = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        JTJ_inv = np.linalg.pinv(J.T @ J)
    pcov = sigma2 * JTJ_inv
    return pcov

def crlb(popt, pcov):
    ### Computes relative CRLBs from pcov
    perr = np.sqrt(np.diag(pcov))  # absolute CRLBs (standard deviations)
    crlb_percent = np.abs(perr / popt) * 100
    return crlb_percent.tolist()

def compute_baseline(w, coeffs, base):
    ### helper do compute the baseline for the model function
    if base == 'cheb':
        # Normalize w to [-1, 1]
        w_norm = 2 * (w - w.min()) / (w.max() - w.min()) - 1
        return chebval(w_norm, coeffs)
    elif base == 'spline':
        n_knots = len(coeffs)
        w_knots = np.linspace(w.min(), w.max(), n_knots)
        spline = CubicSpline(w_knots, coeffs, bc_type='natural')
        return spline(w)

#--------------- Peak models --------------------------------------------------------------------

def model_lorentz(freq, gamma, amp):
    ### Lorentz peak normalized to amplitude
    return amp * (gamma - 1j * freq) / (freq**2 + gamma**2)

def lorentz_area(amp, gamma):
    ### Calculates the area of a lorentz peak
    return amp * np.pi * gamma

def model_voigt(freq, sigma, gamma, area):
    ### Voigt profile normalized to area
    z = (freq + 1j * gamma) / (sigma * np.sqrt(2))
    return area * spsp.wofz(z) / (sigma * np.sqrt(2 * np.pi))

def model_double_voigt(freq, delta, sigma, gamma, A, dA):
    ### Symmetric in-phase doublet for GABA and 2HG; right lobe amplitude = A + dA.
    left  = model_voigt(freq - delta/2, sigma, gamma, A)
    right = model_voigt(freq + delta/2, sigma, gamma, A + dA)
    return left + right


#---------------- NAA Cr Ch ----------------------------------------------------------------------

### Voigt Peaks
# Chebychev baseline is slightly faster, but sometimes exhibits roll-off at the edges
        
def sum_voigt_peaks(w, peak_params):
    # Builds peak model
    return sum(model_voigt(w - w0, sigma, gamma, area) for w0, sigma, gamma, area in peak_params)

def model_voigt_NAA_Cr_Ch(w, params, base='spline', with_phase=True):
    ### Voigt model with spline and cheb option for baseline, with and without phase
    voigt_params = np.array(params[:16]).reshape(4, 4)
    remaining = params[16:]
    if with_phase:
        ph0, ph1 = remaining[:2]
        baseline_params = remaining[2:]
    else:
        baseline_params = remaining
    baseline = compute_baseline(w, baseline_params, base)
    model = sum_voigt_peaks(w, voigt_params) + baseline
    if with_phase:
        phase = np.exp(1j * (ph0 + ph1 * w * 2 * np.pi))
        model = model * phase
        baseline = baseline * phase

    return model, baseline

def get_wrapped_voigt_model(base='cheb', with_phase=True):
    ### Removes the baseline output for fitting
    def model(w, *params):
        full_model, _ = model_voigt_NAA_Cr_Ch(w, params, base=base, with_phase=with_phase)
        return np.real(full_model)
    return model

def fit_voigt_NAA_Cr_Ch(freq, spec, p0, bounds, exclude=None, base='cheb', with_phase=True, max_nfev=10000, compute_crlb=False):
    ### Fit with baseline and phase option; phase = True for 1st step and False for 2nd
    x_fit = freq
    y_fit = np.real(spec) 
    if exclude is not None:
        mask = ~np.array(exclude, dtype=bool)
        x_fit = freq[mask]
        y_fit = y_fit[mask]
    wrapped_model = get_wrapped_voigt_model(base=base, with_phase=with_phase)
    if exclude is not None:
        result = opt.least_squares(lambda p: wrapped_model(x_fit, *p) - y_fit,x0=p0, bounds=bounds, max_nfev=max_nfev)
        popt = result.x
        pcov = covariance_least_squares(result) if compute_crlb else None
    else:
        popt, pcov = opt.curve_fit(wrapped_model, x_fit, y_fit,p0=p0, bounds=bounds, maxfev=max_nfev)
    # Full model reconstruction
    full_model, baseline = model_voigt_NAA_Cr_Ch(freq, popt, base=base, with_phase=with_phase)
    fit = np.real(full_model)
    residual = np.real(spec) - fit
    if compute_crlb:
        crlbs = crlb(popt, pcov)
        return fit, np.real(baseline), residual, popt, crlbs
    else:
        return fit, np.real(baseline), residual, popt
    


def fit_1_voigt_NAA_Cr_Ch(freq, spec, p0, bounds, exclude, base):
    ### 1st Fitting step, with phase
    return fit_voigt_NAA_Cr_Ch(freq=freq,spec=spec,p0=p0,bounds=bounds,exclude=exclude,base=base,with_phase=True,compute_crlb=False)

def fit_2_voigt_NAA_Cr_Ch(freq, spec_cor, popt, bounds, exclude, base, n_base_2=None):
    ### 2nd Fitting step, without phase
    p0_trimmed, bounds_trimmed = new_base(popt, bounds, freq, n_base_2,'voigt')
    return fit_voigt_NAA_Cr_Ch(freq=freq,spec=spec_cor,p0=p0_trimmed,bounds=bounds_trimmed,exclude=exclude,base=base,with_phase=False,compute_crlb=True)

def fit_NAA_Cr_Ch_2steps_voigt(freq, spec, p0, bounds, exclude, base, n_base_2=None):
    ### 2 fitting steps for single fits, parallel helper function does not use this
    # Step 1: fit with phase 
    f1, b1, r1, popt = fit_1_voigt_NAA_Cr_Ch(freq=freq,spec=spec,p0=p0,bounds=bounds,exclude=exclude,base=base)

    # Extract phase and apply phase correction
    ph0 = popt[16]
    ph1 = popt[17]
    s2 = apply_phase(spec, freq, ph0, ph1)
    # Step 2: fit without phase
    f2, b2, r2, popt_2, crlbs = fit_2_voigt_NAA_Cr_Ch(freq=freq,spec_cor=s2,popt=popt,bounds=bounds,exclude=exclude,base=base,n_base_2=n_base_2)

    return f1, b1, r1, s2, f2, b2, r2, popt_2, crlbs

def params_NAA_Cr_Ch_voigt(shift, center_freq, n_baseline, base):
    ### normal start paramters for inverted peaks, parallel call tries several ph0
    w_ppm = np.linspace(1,4.5,2000)
    w = ppm2freq(w_ppm, center_freq)
    
    w0_NAA      = ppm2freq(2 + shift, center_freq)
    sigma_NAA   = 1e-10
    gamma_NAA   = 2
    area_NAA    = 10e4
    
    w0_Cr1      = ppm2freq(3.0 + shift, center_freq)
    sigma_Cr1   = 1e-10
    gamma_Cr1   = 2
    area_Cr1    = 4e4
    
    w0_Cr2      = ppm2freq(3.9 + shift, center_freq)
    sigma_Cr2   = 1e-10
    gamma_Cr2   = 2
    area_Cr2    = 3e4
    
    w0_Ch       = ppm2freq(3.18 + shift, center_freq)
    sigma_Ch    = 1e-10
    gamma_Ch    = 2
    area_Ch     = 3e4

    ph0         = -np.pi
    ph1         = 0
    
    p0 = (w0_NAA,sigma_NAA,gamma_NAA,area_NAA,
          w0_Cr1,sigma_Cr1, gamma_Cr1,area_Cr1,
          w0_Cr2,sigma_Cr2,gamma_Cr2,area_Cr2,
          w0_Ch,sigma_Ch,gamma_Ch,area_Ch,
          ph0,ph1)
    
    bounds = (
        [    w0_NAA-20, 0, 0,  0.0,        w0_Cr1-20, 0,  0,  0.0,      w0_Cr2-20, 0,  0, 0.0,        w0_Ch-20, 0,  0,  0.0,     -np.pi,     -1e-2*np.pi    ],  # lower
        [    w0_NAA+20,  10, 10, np.inf,   w0_Cr1+20, 10, 10, np.inf,   w0_Cr2+20, 10, 10,np.inf,     w0_Ch+20, 10, 10, np.inf,   np.pi ,     1e-2*np.pi     ]   # upper
    )
    
    if base == 'cheb':
        cheb_p0 = [1e3] * (n_baseline + 1)
        p0 = list(p0) + cheb_p0
        lower_cheb = [-1e4] * (n_baseline + 1)
        upper_cheb = [1e4] * (n_baseline + 1)
        lower_bounds = bounds[0] + lower_cheb
        upper_bounds = bounds[1] + upper_cheb
        
    if base == 'spline':
        n_spline_knots = n_baseline + 1
        spline_p0 = [-1.5e3] * n_spline_knots
        p0 = list(p0) + spline_p0
        lower_spline = [-1e5] * n_spline_knots
        upper_spline = [ 1e5] * n_spline_knots
        lower_bounds = bounds[0] + lower_spline
        upper_bounds = bounds[1] + upper_spline
    

    bounds = (lower_bounds, upper_bounds)
    
    return w, p0, bounds

### Lorentz Peaks

def sum_lorentz_peaks(w, peak_params):
    # Builds peak model
    return sum(model_lorentz(w - w0, gamma, area) for w0, gamma, area in peak_params)

def model_lorentz_NAA_Cr_Ch(w, params, base='cheb', with_phase=True):
    # Lorentz model with spline and cheb option for baseline, with and without phase
    lorentz_params = np.array(params[:12]).reshape(4, 3)  # 4 peaks: (w0, gamma, amp)
    remaining = params[12:]

    if with_phase:
        ph0, ph1 = remaining[:2]
        baseline_params = remaining[2:]
    else:
        baseline_params = remaining

    baseline = compute_baseline(w, baseline_params, base)
    model = sum_lorentz_peaks(w, lorentz_params) + baseline

    if with_phase:
        phase = np.exp(1j * (ph0 + ph1 * w * 2 * np.pi))
        model = model * phase
        baseline = baseline * phase

    return model, baseline

def get_wrapped_lorentz_model(base='cheb', with_phase=True):
    # Removes baseline output for fitting
    def model(w, *params):
        full_model, _ = model_lorentz_NAA_Cr_Ch(w, params, base=base, with_phase=with_phase)
        return np.real(full_model)
    return model

def fit_lorentz_NAA_Cr_Ch(freq, spec, p0, bounds, exclude=None, base='spline',with_phase=True, max_nfev=10000, compute_crlb=False):
    # Fit function for Lorentz peaks, with baseline and phase options
    x_fit = freq
    y_fit = np.real(spec)
    if exclude is not None:
        mask = ~np.array(exclude, dtype=bool)
        x_fit = freq[mask]
        y_fit = y_fit[mask]
    model = get_wrapped_lorentz_model(base=base, with_phase=with_phase)
    if exclude is not None:
        result = opt.least_squares(lambda p: model(x_fit, *p) - y_fit, x0=p0, bounds=bounds, max_nfev=max_nfev)
        popt = result.x
        pcov = covariance_least_squares(result) if compute_crlb else None
    else:
        popt, pcov = opt.curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=max_nfev)
    full_model, baseline = model_lorentz_NAA_Cr_Ch(freq, popt, base=base, with_phase=with_phase)
    fit = np.real(full_model)
    residual = np.real(spec) - fit
    if compute_crlb:
        crlbs = crlb(popt, pcov)
        return fit, np.real(baseline), residual, popt, crlbs
    else:
        return fit, np.real(baseline), residual, popt
    
def fit_1_lorentz_NAA_Cr_Ch(freq, spec, p0, bounds, exclude, base):
    ### Lorentz Step 1, with phase
    return fit_lorentz_NAA_Cr_Ch(freq=freq,spec=spec,p0=p0,bounds=bounds,exclude=exclude,base=base,with_phase=True,compute_crlb=False)

def fit_2_lorentz_NAA_Cr_Ch(freq, spec_cor, popt, bounds, exclude, base, n_base_2=None):
    # Lorentz Step 2, without phase 
    p0_trimmed, bounds_trimmed = new_base(popt, bounds, freq, n_base_2, 'lorentz')

    return fit_lorentz_NAA_Cr_Ch(freq=freq,spec=spec_cor,p0=p0_trimmed,bounds=bounds_trimmed,exclude=exclude,base=base,with_phase=False,compute_crlb=True)

def fit_NAA_Cr_Ch_2steps_lorentz(freq, spec, p0, bounds, exclude, base, n_base_2=None):
    # Lorentz 2 Step fitting model for single fits, parallel helper function does not use this
    # Step 1: Fit with phase
    f1, b1, r1, popt = fit_1_lorentz_NAA_Cr_Ch(freq=freq,spec=spec,p0=p0,bounds=bounds,exclude=exclude,base=base)
    # Extract phase and apply correction
    ph0 = popt[12]
    ph1 = popt[13]
    s2 = apply_phase(spec, freq, ph0, ph1)
    # Step 2: Fit without phase
    f2, b2, r2, popt_2, crlbs = fit_2_lorentz_NAA_Cr_Ch(freq=freq,spec_cor=s2,popt=popt,bounds=bounds,exclude=exclude,base=base,n_base_2=n_base_2)
    return f1, b1, r1, s2, f2, b2, r2, popt_2, crlbs

def params_NAA_Cr_Ch_lorentz(shift, center_freq, n_baseline, base):
    # good strat parameters for inverted peaks, parallel call tries several ph0
    w_ppm = np.linspace(1,4.5,2000)
    w = ppm2freq(w_ppm, center_freq)
    
    w0_NAA      = ppm2freq(2 + shift, center_freq)
    gamma_NAA   = 2
    amp_NAA     = 3e4
    
    w0_Cr1      = ppm2freq(3.0 + shift, center_freq)
    gamma_Cr1   = 2
    amp_Cr1     = 1.5e4
    
    w0_Cr2      = ppm2freq(3.9 + shift, center_freq)
    gamma_Cr2   = 2
    amp_Cr2     = 1e4
    
    w0_Ch       = ppm2freq(3.18 + shift, center_freq)
    gamma_Ch    = 2
    amp_Ch      = 1e4

    ph0         = -np.pi
    ph1         = 0

    p0 = (w0_NAA,gamma_NAA,amp_NAA,
          w0_Cr1,gamma_Cr1, amp_Cr1,
          w0_Cr2,gamma_Cr2,amp_Cr2,
          w0_Ch,gamma_Ch,amp_Ch,
          ph0,ph1)
    
    bounds = (
        [    w0_NAA-20, 1, 0.0,         w0_Cr1-20, 1, 0.0,       w0_Cr2-20, 1, 0.0,         w0_Ch-20, 1, 0.0,       -np.pi,     -1e-3*np.pi    ],  # lower
        [    w0_NAA+20, 10, np.inf,     w0_Cr1+20, 10, np.inf,   w0_Cr2+20, 10, np.inf,     w0_Ch+20, 10, np.inf,   np.pi ,     1e-3*np.pi     ]   # upper
    )
    
    if base == 'cheb':
        cheb_p0 = [1e3] * (n_baseline + 1)
        p0 = list(p0) + cheb_p0
        lower_cheb = [-1e4] * (n_baseline + 1)
        upper_cheb = [1e4] * (n_baseline + 1)
        lower_bounds = bounds[0] + lower_cheb
        upper_bounds = bounds[1] + upper_cheb
        
    if base == 'spline':
        n_spline_knots = n_baseline + 1
        spline_p0 = [-1.5e3] * n_spline_knots
        p0 = list(p0) + spline_p0
        lower_spline = [-1e4] * n_spline_knots
        upper_spline = [ 1e4] * n_spline_knots
        lower_bounds = bounds[0] + lower_spline
        upper_bounds = bounds[1] + upper_spline
        
    bounds = (lower_bounds, upper_bounds)
    
    return w, p0, bounds

#---------------- GABA --------------------------------------------------------------------------------

def model_voigt_GABA(w, params, with_phase=True):
    # In-phase gaba doublet
    if with_phase:
        w0, delta, sigma, gamma, A, dA, ph0, ph1 = params
    else:
        w0, delta, sigma, gamma, A, dA = params

    x = w - w0
    model = model_double_voigt(x, delta, sigma, gamma, A, dA)

    if with_phase:
        phase = np.exp(1j * (ph0 + ph1 * w * 2 * np.pi))
        model = model * phase

    return model

def get_wrapped_voigt_GABA_model(with_phase=True):
    # Removes baseline output for fitting
    def model(w, *params):
        return np.real(model_voigt_GABA(w, params, with_phase=with_phase))
    return model

def params_GABA(shift, center_freq):
    # GABA start params
    w0 = ppm2freq(3.0+shift, center_freq)
    delta_ppm = 0.05
    delta0 = abs(ppm2freq(0, center_freq) - ppm2freq(delta_ppm, center_freq))
    sigma0, gamma0 = 2.0, 6.0
    A0  = 2.5e5
    dA0 = 0.0
    ph0, ph1 = 0.0, 0.0
    # Step 1: with phase
    p0_1 = (w0, delta0, sigma0, gamma0, A0, dA0, ph0, ph1)
    lo1  = [w0-20, delta0*0.6, 1,  2,  0, -5e5, -np.pi, -1e-4*np.pi]
    hi1  = [w0+20, delta0*1.6, 8.0, 12.0,   1e7,  5e5,  np.pi,  1e-4*np.pi]
    # Step 2: no phase (same bounds minus ph0/ph1)
    p0_2 = (w0, delta0, sigma0, gamma0, A0, dA0)
    lo2  = [w0-20, delta0*0.6, 1,  2,  -1e6, -5e5]
    hi2  = [w0+20, delta0*1.6, 8.0, 12.0,   1e6,  5e5]

    return (tuple(map(float, p0_1)), (lo1, hi1)), (tuple(map(float, p0_2)), (lo2, hi2))

def fit_voigt_GABA(freq, spec, p0, bounds, with_phase, compute_crlb, peak_mask, max_nfev=10000):
    # Fits GABA, no baseline in difference spectrum
    y = np.real(spec)
    wrapped = get_wrapped_voigt_GABA_model(with_phase=with_phase)

    lo, hi = np.array(bounds[0], float), np.array(bounds[1], float)
    p0 = np.array(p0, float)

    popt, pcov = opt.curve_fit(wrapped, freq, y, p0=tuple(p0), bounds=(lo, hi), maxfev=max_nfev)
    fit = wrapped(freq, *popt)
    residual = y - fit
    area = 2.0 * popt[4] + popt[5]   # 2*A + dA
    crlbs = crlb(popt, pcov) if compute_crlb else None
    return (fit, residual, area, popt, crlbs) if compute_crlb else (fit, residual, area, popt)

def fit_1_voigt_GABA(freq, spec, shift, center_freq, peak_mask):
    # 1st fitting step, tries several ph0
    (p0_1, b1), _ = params_GABA(shift, center_freq)
    best = None; best_norm = np.inf
    for dph0 in (-0.6, -0.3, 0.0, 0.3, 0.6):
        pscan = list(p0_1); pscan[6] += dph0 
        lo1, hi1 = b1
        pscan = np.minimum(np.maximum(pscan, np.array(lo1)+1e-12), np.array(hi1)-1e-12)
        f1, r1, a1, popt1 = fit_voigt_GABA(freq, spec, tuple(pscan), b1, with_phase=True, compute_crlb=False,peak_mask=peak_mask)
        nrm = np.linalg.norm(r1)
        if nrm < best_norm:
            best = (f1, r1, a1, popt1); best_norm = nrm
    return best

def fit_2_voigt_GABA(freq, spec_cor, shift, center_freq, peak_mask, popt1=None):
    # 2nd fitting step, no phase
    _, (p0_2, b2) = params_GABA(shift, center_freq)
    return fit_voigt_GABA(freq, spec_cor, p0_2, b2, with_phase=False, compute_crlb=True,peak_mask=peak_mask)

def fit_GABA_2steps_voigt(freq, spec, shift, center_freq, peak_mask):
    # For single fits, paralles helper function does not use this
    f1, r1, area1, popt1 = fit_1_voigt_GABA(freq, spec, shift, center_freq, peak_mask)
    ph0, ph1 = popt1[6], popt1[7]
    spec_cor = apply_phase(spec, freq, ph0, ph1)
    f2, r2, area2, popt2, crlbs = fit_2_voigt_GABA(freq, spec_cor, shift, center_freq, peak_mask, popt1=popt1)
    return f1, r1, spec_cor, f2, r2, area2, popt2, crlbs


