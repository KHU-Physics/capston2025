#from Round 3.5, we use CLEAN_AR datas from Alternate_strain_downloader and add bandpass processing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, sosfiltfilt
from gwpy.timeseries import TimeSeries

# ==============================
# 1. 파일 경로 / 파라미터 설정
# ==============================
filename = r"Alternate Strains_O4a\H1_GDS-CALIB_STRAIN_CLEAN_AR_1368350720_1368353721.hdf5"

# 분석할 구간 (GPS 기준)
startTime = 1368350720.0   # 시작 GPS 시간 (원하면 바꿔 쓰기)
length    = 16.0           # 분석할 길이 [sec]

# bandpass 대역 (CLEAN_AR이니 notch는 안 씀)
bp_low  = 35.0   # Hz
bp_high = 350.0  # Hz

# ==============================
# 2. CLEAN_AR HDF5 읽기 (gwpy)
# ==============================
data = TimeSeries.read(filename, format="hdf5")

# gwpy TimeSeries -> numpy
strain   = data.value               # strain 값 (1D numpy array)
dt       = float(data.dt.value)     # 샘플링 간격 [sec]
fs       = 1.0 / dt                 # 샘플링 주파수 [Hz]
gpsStart = float(data.t0.value)     # 시작 GPS 시간

print(f"Loaded CLEAN_AR file: {filename}")
print(f"GPS start = {gpsStart}, dt = {dt}, fs = {fs}, N = {len(strain)}")

# ========================================
# 3. Preprocessing: bandpass only (35–350)
# ========================================
def bandpass_sos(low, high, fs, order=4):
    nyq = 0.5 * fs
    wn  = [low/nyq, high/nyq]
    sos = butter(order, wn, btype="band", output="sos")
    return sos

def preprocess_strain_bandpass(strain, fs, bp=(35, 350)):
    x = strain.astype(np.float64)
    # DC 제거
    x = x - np.mean(x)
    # bandpass
    if bp is not None:
        sos = bandpass_sos(bp[0], bp[1], fs, order=4)
        x   = sosfiltfilt(sos, x)
    return x

strain_filt = preprocess_strain_bandpass(strain, fs, bp=(bp_low, bp_high))

# ========================================
# 4. Round 3 스타일 segment 추출 함수
# ========================================
def extract_segment(strain, gpsStart, startTime, length, dt):
    """gpsStart, startTime, length로 segment 잘라오기"""
    N_seg      = int(length / dt)
    idx_start  = int((startTime - gpsStart) / dt)
    idx_end    = idx_start + N_seg
    seg        = strain[idx_start:idx_end]
    rel_time   = np.arange(0, len(seg)) * dt
    return rel_time, seg

time_seg, strain_seg = extract_segment(
    strain_filt,
    gpsStart,
    startTime,
    length,
    dt
)

# ========================================
# 5. Round 3.5 플롯들 (time / FFT / PSD / ASD / spec)
# ========================================
N      = len(strain_seg)
Nplot  = min(10000, N)  # 너무 길면 앞부분만

# --- Time domain ---
plt.figure(figsize=(10, 4))
plt.plot(time_seg[:Nplot], strain_seg[:Nplot])
plt.xlabel("Time [s] since GPS {:.0f}".format(startTime))
plt.ylabel("strain h(t)")
plt.title(f"{filename}\nCLEAN_AR + bandpass [{bp_low}-{bp_high}] Hz\nsegment: {startTime}–{startTime+length} (GPS)")
plt.tight_layout()
plt.show()

# --- FFT (Blackman window) ---
window   = np.blackman(N)
seg_win  = strain_seg * window
freq     = np.fft.rfftfreq(N, d=dt)
fft_amp  = np.abs(np.fft.rfft(seg_win)) / fs

plt.figure(figsize=(10, 4))
plt.loglog(freq, fft_amp)
plt.xlabel("Frequency [Hz]")
plt.ylabel("|FFT|")
plt.title("FFT (Blackman window)")
plt.tight_layout()
plt.show()

# --- PSD ---
Pxx, freqs = mlab.psd(strain_seg, Fs=fs, NFFT=int(fs))  # 1초 길이 창
plt.figure(figsize=(10, 4))
plt.loglog(freqs, Pxx)
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("PSD (Welch/mlab)")
plt.tight_layout()
plt.show()

# --- ASD (sqrt PSD) ---
plt.figure(figsize=(10, 4))
plt.loglog(freqs, np.sqrt(Pxx))
plt.xlabel("Frequency [Hz]")
plt.ylabel("ASD [strain / √Hz]")
plt.title("ASD")
plt.tight_layout()
plt.show()

# --- Spectrogram ---
NFFT = 1024
win  = np.blackman(NFFT)
plt.figure(figsize=(10, 4))
spec_power, f_spec, t_spec, im = plt.specgram(
    strain_seg, NFFT=NFFT, Fs=fs, window=win
)
plt.xlabel("Time [s] within segment")
plt.ylabel("Frequency [Hz]")
plt.title("Spectrogram")
plt.tight_layout()
plt.show()

# # ----- Round 1 스타일: 상대 시간 축으로 한 번 더 플롯 -----
#
# # segment 길이만큼의 상대 시간 벡터 (0 ~ 약 2.4 s)
# numsamples = 10000
# ts=0.000244140625
# rel_time = np.arange(0, numsamples) * ts
#
# plt.figure(figsize=(15, 3))
# plt.plot(rel_time, strain_seg)
# plt.xlabel('Time since segment start (s)')
# plt.ylabel('H1 Strain')
# plt.title(
#     "Hanford (H1) Alternate Strain zoomed segment\n"
#     f"start GPS = {startTime}, 10000 samples, 4096 Hz"
# )
# plt.show()