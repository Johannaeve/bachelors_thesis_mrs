from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mrs.evaluation import  plot_fit, open_nii, compute_spectrum ,calculate_snr_diff, plot_gaba_fit_step2, calculate_snr, relax_factor
from mrs.fitting import freq2ppm,params_NAA_Cr_Ch_voigt
from mrs.multi import  parallel_voigt_fit_GABA, parallel_voigt_fit_NAA_Cr_Ch
INK = r"C:\Users\linkja\InkscapePortable\App\Inkscape\bin\inkscape.exe"
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # remove this when moving to github
#%% 
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
nii1 = DATA / "example_1.nii.gz"
nii2 = DATA / "example_2.nii.gz"
files1 = [nii1] # edit-OFF
files2 = [nii2] # edit-ON

shifts = [-0.07]*len(files1)
tes = [86]*len(files1)
trs = [3000]*len(files1)
flips = [81]*len(files1)
n_base = [4]
records=[]
records_gaba = []
for file_off, file_on, tr, flip, te, shift, n in zip(files1, files2, trs, flips, tes, shifts, n_base):
    # Load spectra
    sig_on,  dwell_on,  cf  = open_nii(file_on)
    sig, dwell, cf = open_nii(file_off)
    freq_on,  spec_on  = compute_spectrum(sig_on,  dwell_on)
    freq, spec = compute_spectrum(sig, dwell)
    spec_diff = - spec + spec_on
    ppm = freq2ppm(freq, cf)
    # Get initial parameters for edit-OFF
    w, p0, bounds = params_NAA_Cr_Ch_voigt(shift, cf, n_baseline=n, base='spline')
    
    # Define masks
    mask = (ppm > 1.5) & (ppm < 4.2)
    mask_diff = (ppm > 2.70 + shift) & (ppm < 3.25 + shift)
    exclude = None
    
    ppm_off = ppm[mask]
    freq_off = freq[mask]
    spec_off = spec[mask]
    ppm_diff = ppm[mask_diff]
    freq_diff = freq[mask_diff]
    spec_diff = spec_diff[mask_diff]
    
    # Define peaks for plotting
    peak1 = (ppm_off < 4.0 + shift) & (ppm_off > 3.9 + shift)
    peak2 = (ppm_off < 3.3 + shift) & (ppm_off > 2.9 + shift)
    peak3 = (ppm_off < 2.15 + shift) & (ppm_off > 1.95 + shift)
    peak_diff = (ppm_diff > 2.9 + shift) & (ppm_diff < 3.1 + shift)
    
    rec = dict(tr=tr, te=te, flip=flip, shift=shift, center_freq=cf, dwell_time=dwell, freq=freq_off, ppm=ppm_off, spec=spec_off, w_v=w, p0_v=p0, 
               bounds_v=bounds, exclude=exclude, peak1=peak1, peak2=peak2, peak3=peak3, mask=mask)
    records.append(rec)
    rec_diff = dict(center_freq=cf, shift=shift, freq=freq_diff, ppm=ppm_diff, spec=spec_diff, peak=peak_diff)
    records_gaba.append(rec_diff)
#%%
if __name__ == "__main__":    
    results   = parallel_voigt_fit_NAA_Cr_Ch(records, base='spline', n_jobs=4)
    results_gaba = parallel_voigt_fit_GABA(records_gaba, n_jobs=4)
    
for rec, rec_g ,off, diff in zip(records,records_gaba,results, results_gaba):
    rec.update(off)
    rec_g.update(diff)

#%% Plot step 2        
rows = []
i=0
for rec, rec_diff in zip(records, records_gaba):
    crlb = rec["crlbs"]
    popt = rec["popt"]
    rows.append({"NAA_CRLB":  crlb[3],"NAA_Area":  popt[3],"NAA_%CRLB": crlb[3]/popt[3]*100,"Cr1_CRLB":  crlb[7],"Cr1_Area":  popt[7],"Cr1_%CRLB": crlb[7]/popt[7]*100,"Cr2_CRLB":  crlb[11],
                 "Cr2_Area":  popt[11],"Cr2_%CRLB": crlb[11]/popt[11]*100,"Ch_CRLB":   crlb[15],"Ch_Area":   popt[15],"Ch_%CRLB":  crlb[15]/popt[15]*100,})
    fig,ax=plt.subplots(figsize=(7, 4.5))
    plot_fit(rec["ppm"],rec["spec_cor"]/1e4, rec["fit_2"]/1e4, rec["baseline_2"]/1e4, rec["residual_2"]/1e4, offset=max(rec["spec_cor"])*1.2/1e4, 
             peak1=rec["peak1"], peak2=rec["peak2"], peak3=rec["peak3"], title=None, save=None, ax=ax)
    plt.show()
   # fig1,ax=plt.subplots(figsize=(7, 4.5))
   # plot_gaba_fit_step2(rec_diff["ppm"], rec_diff["spec"], rec_diff["fit_2"], rec_diff["residual_2"], rec_diff["peak"])

df = pd.DataFrame(rows)
print(df.to_string(index=False))
#%% Print T1/T2-corrected concentration ratios to Cr1 + SNR

TR_ms    = 3000
TE_ms    = 72
flip_deg = 90

T1 = [1.48, 1.82, 1.41, 2.40, 1.6]  # NAA, Cr1, Cr2, Cho, GABA
T2 = [0.893, 0.721, 0.721, 0.599, 0.8]

# Proton counts per resonance
nH = [3, 3, 2, 9, 2]

rows = []
for rec, rec_diff in zip(records, records_gaba):
    ppm, spec = rec["ppm"], rec["spec_cor"]
    popt = rec["popt"]   # indices: NAA=3, Cr1=7, Cr2=11, Ch=15
    baseline=rec["baseline_2"]
    # Relaxation factors
    rf = [relax_factor(rec["tr"], rec["te"], flip_deg, T1[i], T2[i]) for i in range(len(T1))]

    conc_NAA = popt[3] / (rf[0] * nH[0])
    conc_Cr1 = popt[7] / (rf[1] * nH[1])
    conc_Cr2 = popt[11] / (rf[2] * nH[2])
    conc_Ch  = popt[15] / (rf[3]  * nH[3])

    # Ratios to Cr1 
    NAA_over_Cr1 = conc_NAA / conc_Cr1
    Cr2_over_Cr1 = conc_Cr2 / conc_Cr1
    Ch_over_Cr1  = conc_Ch  / conc_Cr1

    # SNR around 3 ppm window
    shift = rec.get("shift", 0.0)
    snr = calculate_snr(ppm, spec-baseline, shift=shift)

    rows.append({"NAA/Cr1": NAA_over_Cr1, "Ch/Cr1":  Ch_over_Cr1, "Cr2/Cr1": Cr2_over_Cr1, "SNR": snr,})

df_ratios = pd.DataFrame(rows)
print(df_ratios.to_string(index=False))

#%%

# Plot each pair
for rec in records_gaba:
    fig1,ax=plt.subplots(figsize=(7, 4.5))
    plot_gaba_fit_step2(rec["ppm"], rec["spec_cor"], rec["fit_2"], rec["residual_2"], rec["peak"])
#%%  
# ---- Summary bar plot of integrated area (model-only) ----
labels = [_short_label(rec.get("file","?")) for rec in records_gaba]
areas  = [float(rec.get("areas", np.nan)) for rec in records_gaba]

plt.figure(figsize=(max(8, 0.9*len(labels)), 4))
pos = np.arange(len(labels))
plt.bar(pos, areas)
plt.xticks(pos, labels, rotation=45, ha="right")
plt.ylabel("Area (window, a.u.)")
plt.title("GABA integrated area (model-only, step 2)")
plt.grid(axis="y", linestyle=":", alpha=0.6)
plt.tight_layout()
plt.show()

#%%
# choose the field you actually use in records_gaba
GABA_AREA_KEY = "areas"   # or "areas" if that’s what you store

# --- Relaxation times (seconds) — add GABA here (use your sequence/field values) ---
T1_s_map = {
    "GABA": 1.6,  # <-- set your value
    "Cr1":  1.82,
    "Cr2":  1.41,
}
T2_s_map = {
    "GABA": 0.8,  # <-- set your value
    "Cr1":  0.721,
    "Cr2":  0.721,
}

# --- Proton counts per resonance ---
nH = {"GABA": 2, "Cr1": 3, "Cr2": 2}


# --- STRICT RF helper: requires rec['tr'], rec['te'], rec['flip'] and valid T1/T2 ---
def _require_pos(name, val, file_label):
    if val is None or not np.isfinite(val) or float(val) <= 0:
        raise ValueError(f"{name} missing/invalid in record for {file_label}: got {val}")
    return float(val)

def _rf_for_strict(met, rec):
    # acquisition must be present in MAIN record under these exact keys
    file_label = rec.get("file", "?")
    TR_ms   = _require_pos("TR_ms (rec['tr'])",   rec["tr"],   file_label)
    TE_ms   = _require_pos("TE_ms (rec['te'])",   rec["te"],   file_label)
    flipdeg = _require_pos("flip_deg (rec['flip'])", rec["flip"], file_label)

    # T1/T2: use per-record override ONLY if you explicitly set it; otherwise use your map entry
    if met not in T1_s_map or met not in T2_s_map:
        raise KeyError(f"T1/T2 map missing for metabolite '{met}'")
    T1s = rec.get(f"{met}_T1_s", T1_s_map[met])
    T2s = rec.get(f"{met}_T2_s", T2_s_map[met])

    T1s = _require_pos(f"{met}_T1_s", T1s, file_label)
    T2s = _require_pos(f"{met}_T2_s", T2s, file_label)

    return relax_factor(TR_ms, TE_ms, flipdeg, T1_s=T1s, T2_s=T2s)

# --- Nominal ratios per file (fill if you want Δ% computed; leave NaN to skip) ---
nominal_conc = {
    "eja_svs_mslaser_1_1.nii.gz": {"GABA": 2.5, "Cr": 4.75
                                   },
    "eja_svs_mslaser_2_1.nii.gz": {"GABA": 12.5, "Cr": 8.0},
    "eja_svs_mslaser_3_1.nii.gz": {"GABA": 2.5, "Cr": 8.0},
    "eja_svs_mslaser_5_1.nii.gz": {"GABA": 12.5, "Cr": 8.0},
    "eja_svs_mslaser_6_1.nii.gz": {"GABA": 5, "Cr": 8.0},
}
nominal_gaba_ratios_per_file = {
    fn: {
        "GABA_over_Cr1": v["GABA"]/v["Cr"],
        "GABA_over_Cr2": v["GABA"]/v["Cr"],  # same molecule reference
    }
    for fn, v in nominal_conc.items()
}
def _pct_err(meas, nom):
    return np.nan if (not np.isfinite(nom) or nom == 0) else 100.0*(meas - nom)/nom

rows = []
n = min(len(records_gaba), len(records))
for i in range(n):
    rec_g = records_gaba[i]   # GABA diff record (holds the GABA area already)
    rec_m = records[i]        # MAIN record (holds Cr areas and TR/TE/flip)

    file_key = Path(rec_m["file"]).name

    # --- areas (NO integration) ---
    gaba_area = float(rec_g[GABA_AREA_KEY])     # e.g. 'areas' as you set above
    popt      = rec_m["popt"]
    cr1_area  = float(popt[7])                  # Cr1 @ ~3.02 ppm
    cr2_area  = float(popt[11])                 # Cr2 @ ~3.93 ppm

    # --- RFs: all from MAIN record (strict) ---
    rf_gaba = _rf_for_strict("GABA", rec_m)
    rf_cr1  = _rf_for_strict("Cr1",  rec_m)
    rf_cr2  = _rf_for_strict("Cr2",  rec_m)

    # --- convert to concentration-like: conc ∝ area / (RF * nH) ---
    conc_gaba = gaba_area / (rf_gaba * nH["GABA"])
    conc_cr1  = cr1_area  / (rf_cr1  * nH["Cr1"])
    conc_cr2  = cr2_area  / (rf_cr2  * nH["Cr2"])

    # --- ratios ---
    gaba_over_cr1 = conc_gaba / conc_cr1
    gaba_over_cr2 = conc_gaba / conc_cr2

    # --- nominal + deviations (optional) ---
    nom = nominal_gaba_ratios_per_file.get(file_key, {})
    nom_cr1 = float(nom.get("GABA_over_Cr1", np.nan))
    nom_cr2 = float(nom.get("GABA_over_Cr2", np.nan))
    dperc_cr1 = _pct_err(gaba_over_cr1, nom_cr1)
    dperc_cr2 = _pct_err(gaba_over_cr2, nom_cr2)

    # --- SNR of the diff (phase-corrected if present) ---
    shift = rec_m.get("shift", 0.0)
    snr_diff = calculate_snr_diff(rec_g, shift=shift)

    rows.append({
        "File":             file_key,
        "GABA_area":        gaba_area,
        "Cr1_area":         cr1_area,
        "Cr2_area":         cr2_area,
        "RF_GABA":          rf_gaba,
        "RF_Cr1":           rf_cr1,
        "RF_Cr2":           rf_cr2,
        "GABA/Cr1 (corr)":  gaba_over_cr1,
        "GABA/Cr2 (corr)":  gaba_over_cr2,
        "Nom GABA/Cr1":     nom_cr1,
        "Δ% vs Nom Cr1":    dperc_cr1,
        "Nom GABA/Cr2":     nom_cr2,
        "Δ% vs Nom Cr2":    dperc_cr2,
        "SNR_diff":         snr_diff,
    })

df_gaba_cr = pd.DataFrame(rows)
with pd.option_context("display.float_format", lambda v: f"{v:0.4f}"):
    print(df_gaba_cr[
        ["File",
         "GABA/Cr1 (corr)", "Nom GABA/Cr1", "Δ% vs Nom Cr1",
         "GABA/Cr2 (corr)", "Nom GABA/Cr2", "Δ% vs Nom Cr2",
         "SNR_diff"]
    ].to_string(index=False))

