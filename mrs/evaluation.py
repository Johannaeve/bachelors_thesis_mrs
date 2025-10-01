from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

### Temperature Shift Correction
    
def calc_shift(spec, freq, center_freq):
    # Returns shift of NAA, add shift to positions to correct 
    mask = freq > 500
    i_local = np.argmax(np.abs(spec[mask]))
    i_NAA = np.where(mask)[0][i_local]
    freq_NAA = freq[i_NAA]
    del_NAA = freq_NAA / center_freq
    del_exp = 4.70 - 2.02
    shift = del_exp - del_NAA
    return shift, i_NAA

def open_nii(path):
    nii = nib.load(path)
    data = nii.dataobj[:]
    signal = data[0,0,0,:]

    return signal
def compute_spectrum(signal, dwell_time):
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    freq     = np.fft.fftshift(np.fft.fftfreq(len(signal), d=dwell_time))
    return freq, spectrum

### T1 and T2 correction
def ernst_factor(TR_ms, flip_deg, T1_s):
    TR_s = TR_ms / 1e3
    a = np.deg2rad(flip_deg)
    E1 = np.exp(-TR_s / T1_s)
    return np.sin(a) * (1 - E1) / (1 - E1 * np.cos(a) + 1e-15)

def relax_factor(TR_ms, TE_ms, flip_deg, T1_s, T2_s):
    return ernst_factor(TR_ms, flip_deg, T1_s) * np.exp(-(TE_ms/1e3) / T2_s)

### SNR
def calculate_snr(ppm, spec, shift=0.0,peak_window=(2.7, 3.2),noise_window=(1.3,1.7)):
    # SNR = peak_height (Cr-CH3) / noise_std computed on the spectrum
    ppm = np.asarray(ppm); re = np.real(np.asarray(spec))
    lo, hi = peak_window[0] + shift, peak_window[1] + shift
    peak_mask = (ppm >= min(lo, hi)) & (ppm <= max(lo, hi))
    peak_height = float(np.nanmax(re[peak_mask]))
    noise_std = np.nan
    lo, hi = noise_window[0] + shift, noise_window[1] + shift
    nm = (ppm >= min(lo, hi)) & (ppm <= max(lo, hi))
    s = np.nanstd(re[nm], ddof=1)
    noise_std = s
    return float(peak_height/noise_std)

### Plotting

def plot_fit(ppm,spec,fit,baseline,residual,offset,peak1,peak2,peak3,title,save,ax=None,legend=True):
    # Plots spectrum in mask with fit, baseline and residual in 3 peak regions

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(ppm, spec, label="Data", linewidth=1.5, color='black')
    ax.plot(ppm, baseline, '--', linewidth=1.5, label="Baseline", color='grey')
    ax.plot(ppm, fit, '--', linewidth=1.5, label="Fit", color='mediumblue')

    ax.plot(ppm[peak1], residual[peak1]+offset, linewidth=1.5, label="Residual", color='deepskyblue')
    ax.plot(ppm[peak2], residual[peak2]+offset, linewidth=1.5, color='deepskyblue')
    ax.plot(ppm[peak3], residual[peak3]+offset, linewidth=1.5, color='deepskyblue')

    ax.axhspan(offset + np.min([np.min(residual[peak1]),np.min(residual[peak2]), np.min(residual[peak3])]),
               offset + np.max([np.max(residual[peak1]),np.max(residual[peak2]),np.max(residual[peak3])]),color='grey', alpha=0.3)
    ax.axhline(offset, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlim(ppm.max(), ppm.min())
    ax.grid(True)
    ax.set_xlabel('Chemical Shift (ppm)')
    ax.set_ylabel('Re (a.u.)')
    if title is not None:
        ax.set_title(title, fontsize=10)
        
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=1, loc='best')

    if save is not None and ax is None:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
        
def plot_gaba_fit_step2(ppm, spec, fit, resid, peak, offset=None, title=None, save=None, ax=None, legend=True):
    # Plots fit in difference spectrum, no baseline

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    if offset is None:
        offset = 1.3 * np.nanmax(np.abs(spec))

    ax.plot(ppm, spec, label="Data", color='black', linewidth=1.5)
    ax.plot(ppm, fit, '--', linewidth=1.5, label="Fit", color='mediumblue')
    ax.plot(ppm[peak], resid[peak] + offset, linewidth=1.5, label="Residual", color='deepskyblue')
    ax.axhspan(offset + np.nanmin(resid[peak]), offset + np.nanmax(resid[peak]), color='grey', alpha=0.3)
    ax.axhline(offset, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlim(max(ppm),min(ppm))
    ax.grid(True)

    ax.set_xlabel('Chemical Shift (ppm)')
    ax.set_ylabel(r'Re (a.u.)')

    if title is not None:
        ax.set_title(title, fontsize=10)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, ncol=1, loc='best')

    if save is not None and ax is None:
        plt.savefig(save, bbox_inches='tight')
        plt.close()
        
        
        
        
        
        
        