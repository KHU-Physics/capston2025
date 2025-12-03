#Some modifications from Round 5 (file path related optimization)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, correlate


filename_H1 = 'H-H1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'
filename_L1 = 'L-L1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'


# Open file, read the strain and some information
def load_file(path):
    f = h5py.File(path, 'r')
    gpsStart = f['meta']['GPSstart'][()]
    duration = f['meta']['Duration'][()]
    strain = f['strain']['Strain'][:]
    dt = f['strain']['Strain'].attrs['Xspacing']
    fs = 1.0 / dt
    return f, strain, gpsStart, duration, dt, fs


H1, strain_H1, gps_H1, dur_H1, dt_H1, fs_H1 = load_file(filename_H1)
L1, strain_L1, gps_L1, dur_L1, dt_L1, fs_L1 = load_file(filename_L1)

# Sampling/time comparison to minimize errors
assert np.isclose(fs_H1, fs_L1)
assert gps_H1 == gps_L1  # 동일 시작 GPS


#Good segment selection based on round 2
dqInfo = H1['quality']['simple']
dqmask = dqInfo['DQmask'][()]

data_channel = 0
burst_cat2_channel = 5

data_mask = (dqmask >> data_channel) & 1
burst_cat2_mask = (dqmask >> burst_cat2_channel) & 1

goodData_mask_1hz = data_mask & burst_cat2_mask

dummy = np.zeros(goodData_mask_1hz.shape)
masked_dummy = np.ma.masked_array(dummy, np.logical_not(goodData_mask_1hz))
segments = np.ma.flatnotmasked_contiguous(masked_dummy)

segList = [(int(seg.start + gps_H1), int(seg.stop + gps_H1)) for seg in segments]

print("\n Good segments (GPS):")
for s in segList:
    print("  ", s)


#Preprocessing
def bandpass_sos(low, high, fs, order=4):
    nyq = 0.5 * fs
    return butter(order, [low/nyq, high/nyq], btype="band", output="sos")

def preprocess_strain(strain, fs, bp=(35, 350), do_notch=True,
                      notch_freqs=(60, 120, 180), notch_Q=30.0):
    x = strain.astype(np.float64)
    x = x - np.mean(x)

    # bandpass
    if bp is not None:
        sos = bandpass_sos(bp[0], bp[1], fs, order=4)
        x = sosfiltfilt(sos, x)

    # notch
    if do_notch:
        for f0 in notch_freqs:
            if f0 < fs * 0.5:
                w0 = f0 / (fs/2.0)
                b, a = iirnotch(w0, notch_Q)
                x = filtfilt(b, a, x)

    return x


strain_H1_f = preprocess_strain(strain_H1, fs_H1)
strain_L1_f = preprocess_strain(strain_L1, fs_L1)


# Round 3 based segment analysis
def analyze_segment(strain, gpsStart, begin, end, dt, label):
    idx_start = int((begin - gpsStart) / dt)
    idx_end = int((end - gpsStart) / dt)

    seg = strain[idx_start:idx_end]
    seg_time = np.arange(0, len(seg)) * dt
    fs = 1.0 / dt

    # TIME DOMAIN
    plt.figure(figsize=(10, 3))
    plt.plot(seg_time, seg)
    plt.title(f"{label} time series — {begin}–{end}")
    plt.xlabel("Time (s)"); plt.ylabel("strain")
    plt.tight_layout(); plt.show()

    # FFT
    window = np.blackman(len(seg))
    seg_win = seg * window
    freq = np.fft.rfftfreq(len(seg), d=dt)
    fft_amp = np.abs(np.fft.rfft(seg_win)) / fs

    plt.figure(figsize=(10, 3))
    plt.loglog(freq, fft_amp)
    plt.title(f"{label} FFT — {begin}–{end}")
    plt.xlabel("Freq (Hz)"); plt.ylabel("|FFT|")
    plt.tight_layout(); plt.show()

    # PSD/ASD
    Pxx, freqs = mlab.psd(seg, Fs=fs, NFFT=int(fs))

    plt.figure(figsize=(10, 3))
    plt.loglog(freqs, np.sqrt(Pxx))
    plt.title(f"{label} ASD — {begin}–{end}")
    plt.xlabel("Freq (Hz)"); plt.ylabel("ASD")
    plt.tight_layout(); plt.show()

    # Spectrogram
    plt.figure(figsize=(10, 3))
    NFFT = 1024
    plt.specgram(seg, NFFT=NFFT, Fs=fs, window=np.blackman(NFFT))
    plt.title(f"{label} spectrogram — {begin}–{end}")
    plt.xlabel("time (s)"); plt.ylabel("Hz")
    plt.tight_layout(); plt.show()

    return seg


# 6) H1-L1 crossover + time sync + overlay
def align_by_crosscorr(xH, xL, fs):
    H = xH - np.mean(xH)
    L = xL - np.mean(xL)
    corr = correlate(L, H, mode='full')
    lags = np.arange(-len(H)+1, len(L))
    k = np.argmax(corr)
    lag_samples = lags[k]
    lag_seconds = lag_samples / fs
    return lag_samples, lag_seconds


def overlay_plot(t_seg, H_seg, L_seg_aligned, begin, end, lag_sec):
    plt.figure(figsize=(10,3))
    plt.plot(t_seg, H_seg, label="H1", linewidth=0.8)
    plt.plot(t_seg, L_seg_aligned, label=f"L1 (shift {lag_sec:+.4f}s)", linewidth=0.8, alpha=0.8)
    plt.title(f"H1 vs L1 overlay — {begin}–{end}")
    plt.xlabel("Time (s)"); plt.ylabel("strain")
    plt.legend()
    plt.tight_layout()
    plt.show()


# All good seg analysis
for (begin, end) in segList[:2]:  # First two
    print(f"\n\n===== Analyzing segment {begin} – {end} =====")

    # H1 segment
    H_seg = analyze_segment(strain_H1_f, gps_H1, begin, end, dt_H1,
                            label="H1 filtered")

    # L1 segment
    L_seg = analyze_segment(strain_L1_f, gps_L1, begin, end, dt_L1,
                            label="L1 filtered")

    # 교차상관 정렬
    lag_samp, lag_sec = align_by_crosscorr(H_seg, L_seg, fs_H1)
    print(f"⚡ L1 is {lag_sec:+.5f} s {'ahead' if lag_sec<0 else 'behind'} H1")

    # L1 시간 이동
    L_seg_aligned = np.roll(L_seg, -lag_samp)

    # overlay
    t_seg = np.arange(0, len(H_seg)) * dt_H1
    overlay_plot(t_seg, H_seg, L_seg_aligned, begin, end, lag_sec)
