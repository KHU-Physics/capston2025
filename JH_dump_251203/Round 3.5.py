#Implementation of Round_2 (good data segment automation) to Round_3
import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = 'H-H1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'
dataFile = h5py.File(filename, 'r')

# --- Data Quality (DQ) 기반 good data segment 찾기 --- #
gpsStart = dataFile['meta']['GPSstart'][()]
dqInfo = dataFile['quality']['simple']
dqmask = dqInfo['DQmask'][()]

# 원하는 DQ bits
data_channel = 0
burst_cat2_channel = 5

data_mask = (dqmask >> data_channel) & 1
burst_cat2_mask = (dqmask >> burst_cat2_channel) & 1

good_mask = data_mask & burst_cat2_mask  # 1Hz DQ mask

# DQ mask → segment 리스트 변환
dummy = np.zeros(good_mask.shape)
masked_dummy = np.ma.masked_array(dummy, np.logical_not(good_mask))
segments = np.ma.flatnotmasked_contiguous(masked_dummy)

# GPS time의 good segment 리스트
segList = [(int(seg.start + gpsStart), int(seg.stop + gpsStart)) for seg in segments]
print("Good segments detected:\n", segList)

#----------------------------------------------------------------------------

def analyze_segment(strain, gpsStart, begin, end, dt, N_plot=10000):
    # index 변환
    idx_start = int((begin - gpsStart) / dt)
    idx_end   = int((end - gpsStart) / dt)

    seg = strain[idx_start:idx_end]
    seg_time = np.arange(0, len(seg)) * dt
    fs = 1.0/dt

    # --- TIME DOMAIN ---
    plt.figure(figsize=(10,3))
    plt.plot(seg_time[:N_plot], seg[:N_plot])
    plt.title(f"Time series — {filename}\nSegment: {begin}–{end}")
    plt.xlabel("time (s)")
    plt.ylabel("strain")
    plt.tight_layout()
    plt.show()

    # --- FFT ---
    window = np.blackman(len(seg))
    seg_win = seg * window
    freq = np.fft.rfftfreq(len(seg), d=dt)
    fft_amp = np.abs(np.fft.rfft(seg_win))/fs

    plt.figure(figsize=(10,3))
    plt.loglog(freq, fft_amp)
    plt.title(f"FFT (Blackman window)\nSeg {begin}–{end}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")
    plt.tight_layout()
    plt.show()

    # --- PSD ---
    Pxx, freqs = plt.mlab.psd(seg, Fs=fs, NFFT=int(fs))
    plt.figure(figsize=(10,3))
    plt.loglog(freqs, Pxx)
    plt.title(f"PSD (Welch)\nSeg {begin}–{end}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.tight_layout()
    plt.show()

    # --- ASD ---
    plt.figure(figsize=(10,3))
    plt.loglog(freqs, np.sqrt(Pxx))
    plt.title(f"ASD (sqrt PSD)\nSeg {begin}–{end}")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("ASD")
    plt.tight_layout()
    plt.show()

    # --- Spectrogram ---
    NFFT = 1024
    win = np.blackman(NFFT)
    spec_power, f_spec, t_spec, im = plt.specgram(
        seg, NFFT=NFFT, Fs=fs, window=win
    )
    plt.title(f"Spectrogram\nSeg {begin}–{end}")
    plt.xlabel("time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()

#--------------------------------------------------------------
strain = dataFile['strain']['Strain']
dt = dataFile['strain']['Strain'].attrs['Xspacing']

# good segment 리스트 중 처음 1~2개만 돌리는 예시
for (begin, end) in segList[:2]:
    print(f"\n===== Analyzing segment {begin} – {end} =====")
    analyze_segment(strain, gpsStart, begin, end, dt)