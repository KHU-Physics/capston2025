#On this round, we add preprocessing procedures from Round 3.5
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

filename = 'H-H1_GWOSC_O4a_4KHZ_R1-1376026624-4096.hdf5'
dataFile = h5py.File(filename, 'r')

#Finding good segment based on Data Quality information (Round 2)
gpsStart = dataFile['meta']['GPSstart'][()]

dqInfo = dataFile['quality']['simple']
dqmask = dqInfo['DQmask'][()]

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

#Preprocessing 함수 정의
#밴드패스 + 노치 필터 + DC 제거
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

#defining GW data plotting (Based on Round 3)
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

# 4. Call whole strain, apply preprocessing
strain_raw = dataFile['strain']['Strain'][:]
dt         = dataFile['strain']['Strain'].attrs['Xspacing']
fs         = 1.0 / dt

strain_filt = preprocess_strain(
    strain_raw, fs,
    bp=(35, 350),             # Gravitational event band
    do_notch=True,            # Remove 60,120,180 Hz
    notch_freqs=(60, 120, 180),
    notch_Q=30.0
)


# Round 3 type analyzation
# plotting first two segList
for (begin, end) in segList[:2]:
    print(f"\n===== Analyzing segment {begin} – {end} (GPS) =====")
    analyze_segment(
        strain_filt,
        gpsStart,
        begin,
        end,
        dt,
        N_plot=10000,
        label="preprocessed (35–350 Hz, notched)"
    )