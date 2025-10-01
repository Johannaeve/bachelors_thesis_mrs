import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mrs.evaluation import  plot_fit, open_nii, compute_spectrum, plot_gaba_fit_step2, calculate_snr, relax_factor
from mrs.fitting import freq2ppm, params_NAA_Cr_Ch_voigt
from mrs.multi import  parallel_voigt_fit_GABA, parallel_voigt_fit_NAA_Cr_Ch
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
files1 = [DATA / "example_1.nii.gz"] # edit-OFF
files2 = [DATA / "example_2.nii.gz"] # edit-ON

# Info about measurement
dwell = 500*1e-6
cf = 297.183884
shift = -0.07
te = 68
tr = 3000
flip = 81
n_base = 4

# T1 and T2 for each resonance, order: NAA, Cr1, Cr2, Cho, GABA
T1 = [1.48, 1.82, 1.41, 2.40, 1.6] 
T2 = [0.893, 0.721, 0.721, 0.599, 0.8]

# Proton counts 
nH = [3, 3, 2, 9, 2]

records=[]
records_gaba = []

for file_off, file_on in zip(files1, files2):
    
    # Load spectra
    sig_on = open_nii(file_on)
    sig = open_nii(file_off)
    freq_on,  spec_on  = compute_spectrum(sig_on,  dwell)
    freq, spec = compute_spectrum(sig, dwell)
    spec_diff = - spec + spec_on
    ppm = freq2ppm(freq, cf)

    # Get initial parameters for edit-OFF
    w, p0, bounds = params_NAA_Cr_Ch_voigt(shift, cf, n_baseline=n_base, base='spline')
    
    # Define masks
    mask = (ppm > 1.4) & (ppm < 4.2)
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

results   = parallel_voigt_fit_NAA_Cr_Ch(records, base='spline', n_jobs=4)
results_gaba = parallel_voigt_fit_GABA(records_gaba, n_jobs=4)
    
for rec, rec_g ,off, diff in zip(records,records_gaba,results, results_gaba):
    rec.update(off)
    rec_g.update(diff)

rows = []
rows2 = []
for rec, rec_diff in zip(records, records_gaba):
    
    # Plot fit
    fig,ax=plt.subplots(figsize=(7, 4.5))
    plot_fit(rec["ppm"],np.real(rec["spec_cor"])/1e4, rec["fit_2"]/1e4, rec["baseline_2"]/1e4, rec["residual_2"]/1e4, offset=np.nanmax(np.real(rec["spec_cor"]))*1.2/1e4, 
             peak1=rec["peak1"], peak2=rec["peak2"], peak3=rec["peak3"], title=None, save=None, ax=ax)
    plot_gaba_fit_step2(rec_diff["ppm"], np.real(rec_diff["spec"])/1e4, rec_diff["fit_2"]/1e4, rec_diff["residual_2"]/1e4, rec_diff["peak"])
    plt.show()
 
    # Table with resulting areas and crlb
    popt = rec["popt"]   # indices: NAA=3, Cr1=7, Cr2=11, Ch=15
    crlb = rec["crlbs"]  # indices: Area=4, dArea=5 (deviation in the area of the second peak of the doublet)
    crlb_diff = rec_diff["crlbs"]
    popt_diff = rec_diff["popt"]
    rows.append({"NAA_Area":  popt[3],"NAA_%CRLB": crlb[3]/popt[3]*100,"Cr1_Area": popt[7],"Cr1_%CRLB": crlb[7]/popt[7]*100,
                 "Cr2_Area":  popt[11],"Cr2_%CRLB": crlb[11]/popt[11]*100,"Ch_Area": popt[15],"Ch_%CRLB":  crlb[15]/popt[15]*100, 
                 "GABA_Area": popt_diff[4], "GABA_Area_%CRLB": crlb_diff[4]/popt_diff[4], "GABA_dArea": popt_diff[5], "GABA_dArea_%CRLB": crlb_diff[5]/popt_diff[5],})
    
    # Calculate relaxation corrected concentration ratios to Cr1
    rf = [relax_factor(rec["tr"], rec["te"], rec["flip"], T1[i], T2[i]) for i in range(len(T1))]
    conc_Cr1 = popt[7] / (rf[1] * nH[1])
    NAA_over_Cr1 = popt[3] / (rf[0] * nH[0]) / conc_Cr1
    Cr2_over_Cr1 = popt[11] / (rf[2] * nH[2]) / conc_Cr1
    Ch_over_Cr1  = popt[15] / (rf[3]  * nH[3])  / conc_Cr1
    GABA_over_Cr1  = rec_diff["areas"] / (rf[4]  * nH[4])  / conc_Cr1
    
    # SNR
    snr = calculate_snr(rec["ppm"], rec["spec_cor"]-rec["baseline_2"], shift=rec["shift"])
    snr_diff = calculate_snr(rec_diff["ppm"], rec_diff["spec_cor"], shift=rec["shift"], peak_window=(2.88, 3.05), noise_window=(2.5, 3.0))
    rows2.append({"NAA/Cr1": NAA_over_Cr1, "Ch/Cr1":  Ch_over_Cr1, "Cr2/Cr1": Cr2_over_Cr1, "SNR": snr, "GABA/Cr1": GABA_over_Cr1, "SNR_diff": snr_diff})
    print(min(rec_diff["ppm"]))
print()
df_result = pd.DataFrame(rows)
print(df_result.to_string(index=False))
print()
df_ratios = pd.DataFrame(rows2)
print(df_ratios.to_string(index=False))
