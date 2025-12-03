#On this round we compare L1 and H1 files with modifications from Round_4
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, correlate


filename_H1 = 'H-H1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'
filename_L1 = 'L-L1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'

dataFile_H1 = h5py.File(filename_H1, 'r')
dataFile_L1 = h5py.File(filename_L1, 'r')

gpsStart = dataFile_H1['meta']['GPSstart'][()]  # H1 based

dqInfo   = dataFile_H1['quality']['simple']
dqmask   = dqInfo['DQmask'][()]

data_channel       = 0
burst_cat2_channel = 5

data_mask       = (dqmask >> data_channel) & 1
burst_cat2_mask = (dqmask >> burst_cat2_channel) & 1

goodData_mask_1hz = data_mask & burst_cat2_mask

dummy        = np.zeros(goodData_mask_1hz.shape)
masked_dummy = np.ma.masked_array(dummy, np.logical_not(goodData_mask_1hz))
segments     = np.ma.flatnotmasked_contiguous(masked_dummy)
segList      = [(int(seg.start + gpsStart), int(seg.stop + gpsStart)) for seg in segments]

print("Good segments (GPS):")
for s in segList:
    print(s)

# Preprocessing
def bandpass_sos(low, high, fs, order=4):
    nyq = 0.5 * fs
    wn  = [low/nyq, high/nyq]
    sos = butter(order, wn, btype="band", output="sos")
    return sos

def preprocess_strain(strain, fs, bp=(35, 350), do_notch=True,
                      notch_freqs=(60, 120, 180), notch_Q=30.0):
    """
    strain: 1D array (raw strain)
    fs    : sampling frequency
    bp    : (low, high) bandpass range in Hz
    """
    x = strain.astype(np.float64)

    # 1) DC 제거
    x = x - np.mean(x)

    # 2) 밴드패스 필터
    if bp is not None:
        low, high = bp
        sos = bandpass_sos(low, high, fs, order=4)
        x   = sosfiltfilt(sos, x)

    # 3) 노치 필터(전원선 + 고조파 제거)
    if do_notch:
        for f0 in notch_freqs:
            if f0 < fs * 0.5:
                w0 = f0 / (fs/2.0)
                b, a = iirnotch(w0, notch_Q)
                x = filtfilt(b, a, x)

    return x

# Round 3 based segment analysis functions
def analyze_segment(strain, gpsStart, begin, end, dt, N_plot=10000, label="filtered"):
    """
    strain : Whole strain array (already preprocessed)
    begin/end : GPSStart and GPSEnd on particular segment (int/float)
    """
    idx_start = int((begin - gpsStart) / dt)
    idx_end   = int((end   - gpsStart) / dt)

    seg      = strain[idx_start:idx_end]
    seg_time = np.arange(0, len(seg)) * dt
    fs       = 1.0 / dt

    # ---- Time domain ----
    plt.figure(figsize=(10, 3))
    plt.plot(seg_time[:N_plot], seg[:N_plot])
    plt.xlabel("Time [s] (relative)")
    plt.ylabel("strain h(t)")
    plt.title(f"{filename}\n{label} strain, segment {begin}–{end} (GPS)")
    plt.tight_layout()
    plt.show()

    # ---- FFT (Blackman window) ----
    window   = np.blackman(len(seg))
    seg_win  = seg * window
    freq     = np.fft.rfftfreq(len(seg), d=dt)
    fft_amp  = np.abs(np.fft.rfft(seg_win)) / fs

    plt.figure(figsize=(10, 3))
    plt.loglog(freq, fft_amp)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("|FFT|")
    plt.title(f"FFT (Blackman) — {label}, seg {begin}–{end}")
    plt.tight_layout()
    plt.show()

    # ---- PSD ----
    Pxx, freqs = mlab.psd(seg, Fs=fs, NFFT=int(fs))  # 1초 길이 창
    plt.figure(figsize=(10, 3))
    plt.loglog(freqs, Pxx)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title(f"PSD (Welch/mlab) — {label}, seg {begin}–{end}")
    plt.tight_layout()
    plt.show()

    # ---- ASD (sqrt(PSD)) ----
    plt.figure(figsize=(10, 3))
    plt.loglog(freqs, np.sqrt(Pxx))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("ASD [strain / √Hz]")
    plt.title(f"ASD — {label}, seg {begin}–{end}")
    plt.tight_layout()
    plt.show()

    # ---- Spectrogram ----
    NFFT = 1024
    win  = np.blackman(NFFT)
    spec_power, f_spec, t_spec, im = plt.specgram(
        seg, NFFT=NFFT, Fs=fs, window=win
    )
    plt.xlabel("Time [s] (relative in segment)")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"Spectrogram — {label}, seg {begin}–{end}")
    plt.tight_layout()
    plt.show()

    return seg  # can reuse if needed later


# 4. Calling H1 / L1 strain & preprocessing
strain_raw_H1 = dataFile_H1['strain']['Strain'][:]
dt            = dataFile_H1['strain']['Strain'].attrs['Xspacing']  # 기존 dt 그대로 H1 기준
fs            = 1.0 / dt

strain_raw_L1 = dataFile_L1['strain']['Strain'][:]
dt_L1         = dataFile_L1['strain']['Strain'].attrs['Xspacing']
fs_L1         = 1.0 / dt_L1

assert np.isclose(fs, fs_L1)
assert gpsStart == dataFile_L1['meta']['GPSstart'][()]

# Apply preprocessing
strain_filt_H1 = preprocess_strain(
    strain_raw_H1, fs,
    bp=(35, 350),
    do_notch=True,
    notch_freqs=(60, 120, 180),
    notch_Q=30.0
)

strain_filt_L1 = preprocess_strain(
    strain_raw_L1, fs,
    bp=(35, 350),
    do_notch=True,
    notch_freqs=(60, 120, 180),
    notch_Q=30.0
)

# ===============================
# 5. H1-L1 교차상관 + 시간정렬 + overlay
# ===============================
def align_by_xcorr(xH, xL, fs):
    xH = xH - np.mean(xH)
    xL = xL - np.mean(xL)
    corr = correlate(xL, xH, mode='full')  # L vs H
    lags = np.arange(-len(xH) + 1, len(xL))
    k = np.argmax(corr)
    lag_samp = lags[k]
    lag_sec  = lag_samp / fs
    return lag_samp, lag_sec


# 6. Round 3 style analysis + L1 H1 comparison
for (begin, end) in segList[:2]:  # first two segments
    print(f"\n===== Analyzing segment {begin} – {end} (GPS) =====")

    # --- H1 segment ---
    filename = filename_H1
    seg_H1 = analyze_segment(
        strain_filt_H1,
        gpsStart,
        begin,
        end,
        dt,
        N_plot=10000,
        label="H1 preprocessed (35–350 Hz, notched)"
    )

    # --- L1 segment ---
    filename = filename_L1
    seg_L1 = analyze_segment(
        strain_filt_L1,
        gpsStart,
        begin,
        end,
        dt,
        N_plot=10000,
        label="L1 preprocessed (35–350 Hz, notched)"
    )

    # --- H1-L1 교차상관으로 시간정렬 ---
    lag_samp, lag_sec = align_by_xcorr(seg_H1, seg_L1, fs)
    print(f"[ALIGN] L1 is {lag_sec:+.5f} s "
          f"{'ahead of' if lag_sec < 0 else 'behind'} H1")

    seg_L1_aligned = np.roll(seg_L1, -lag_samp)

    # --- Overlay plot ---
    t_seg = np.arange(0, len(seg_H1)) * dt
    plt.figure(figsize=(10, 3))
    plt.plot(t_seg, seg_H1, label="H1", linewidth=0.8)
    plt.plot(t_seg, seg_L1_aligned,
             label=f"L1 (shift {lag_sec:+.4f}s)", linewidth=0.8, alpha=0.8)
    plt.title(f"H1 vs L1 overlay (filtered & aligned)\nsegment {begin}–{end} (GPS)")
    plt.xlabel("Time [s] (relative in segment)")
    plt.ylabel("strain h(t)")
    plt.legend()
    plt.tight_layout()
    plt.show()