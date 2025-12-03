#On this round we save the plotted datas instead of using plt.show() function (5K, 16:9 ratio)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import h5py
import os

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt

filename = 'H-H1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'
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

    x = x - np.mean(x)

    if bp is not None:
        low, high = bp
        sos = bandpass_sos(low, high, fs, order=4)
        x   = sosfiltfilt(sos, x)

    if do_notch:
        for f0 in notch_freqs:
            if f0 < fs * 0.5:
                w0 = f0 / (fs/2.0)
                b, a = iirnotch(w0, notch_Q)
                x = filtfilt(b, a, x)

    return x

#---------------------------------------------------------------------------
def analyze_segment(strain, gpsStart, begin, end, dt, N_plot=10000,
                    label="preprocessed", save=True,
                    base_outdir="plots", filename="data"):

    seg_dir = f"{base_outdir}/{filename}/{begin}_{end}"
    if save:
        os.makedirs(seg_dir, exist_ok=True)

    fs = 1/dt
    idx_start = int((begin - gpsStart) / dt)
    idx_end   = int((end   - gpsStart) / dt)

    seg = strain[idx_start:idx_end]
    seg_time = np.arange(0, len(seg)) * dt

    # TIME DOMAIN
    plt.figure(figsize=(25.6, 14.4), dpi=200)
    plt.plot(seg_time[:N_plot], seg[:N_plot], linewidth=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("strain")
    plt.title(f"{label} — time domain\n{begin}–{end} (GPS)")
    if save:
        plt.tight_layout()
        plt.savefig(f"{seg_dir}/time.png", dpi=200, bbox_inches='tight')
    plt.close()

    # FFT
    window = np.blackman(len(seg))
    seg_win = seg * window
    freq = np.fft.rfftfreq(len(seg), d=dt)
    fft_amp = np.abs(np.fft.rfft(seg_win)) / fs

    plt.figure(figsize=(25.6, 14.4), dpi=200)
    plt.loglog(freq, fft_amp)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|FFT|")
    plt.title(f"{label} — FFT\n{begin}–{end}")
    if save:
        plt.tight_layout()
        plt.savefig(f"{seg_dir}/fft.png", dpi=200, bbox_inches='tight')
    plt.close()

    # PSD
    Pxx, freqs = plt.mlab.psd(seg, Fs=fs, NFFT=int(fs))

    plt.figure(figsize=(25.6, 14.4), dpi=200)
    plt.loglog(freqs, Pxx)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD")
    plt.title(f"{label} — PSD\n{begin}–{end}")
    if save:
        plt.tight_layout()
        plt.savefig(f"{seg_dir}/psd.png", dpi=200, bbox_inches='tight')
    plt.close()

    # ASD
    plt.figure(figsize=(25.6, 14.4), dpi=200)
    plt.loglog(freqs, np.sqrt(Pxx))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("ASD (strain/√Hz)")
    plt.title(f"{label} — ASD\n{begin}–{end}")
    if save:
        plt.tight_layout()
        plt.savefig(f"{seg_dir}/asd.png", dpi=200, bbox_inches='tight')
    plt.close()

    # Spectrogram
    NFFT = 1024
    win = np.blackman(NFFT)
    plt.figure(figsize=(25.6, 14.4), dpi=200)
    plt.specgram(seg, NFFT=NFFT, Fs=fs, window=win)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"{label} — Spectrogram\n{begin}–{end}")
    if save:
        plt.tight_layout()
        plt.savefig(f"{seg_dir}/spectrogram.png", dpi=200, bbox_inches='tight')
    plt.close()

    return seg

#-----------------------------------------------------------------------
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

file_base = os.path.splitext(os.path.basename(filename))[0]

for (begin, end) in segList[:2]:   # 원하는 개수만
    analyze_segment(
        strain_filt,
        gpsStart,
        begin,
        end,
        dt,
        label="filtered 35–350Hz + notch",
        filename=file_base
    )