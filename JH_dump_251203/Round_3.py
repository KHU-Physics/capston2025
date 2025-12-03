#Round 3, we plot the GW data deeper with following procedures.
import numpy as np
import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import h5py

# Open file
fileName = 'H-H1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'
dataFile = h5py.File(fileName, 'r')

# Read the strain and some information
strain   = dataFile['strain']['Strain']
ts       = dataFile['strain']['Strain'].attrs['Xspacing']
meta     = dataFile['meta']
gpsStart = meta['GPSstart'][()]
duration = meta['Duration'][()]
gpsEnd   = gpsStart + duration
time     = np.arange(gpsStart, gpsEnd, ts)

# Extract a part of the signal
length     = 16     # Length in second
startTime  = 1368350720.0
numsamples = int(length / ts)
startIndex = np.min(np.nonzero(startTime < time))
time_seg   = time[startIndex:(startIndex+numsamples)]
strain_seg = strain[startIndex:(startIndex+numsamples)]

# Sampling frequency
fs = int(1.0 / ts)

#Parameters for plotting
window          = np.blackman(strain_seg.size)
windowed_strain = strain_seg * window
freq_domain     = np.fft.rfft(windowed_strain) / fs
freq            = np.fft.rfftfreq(len(windowed_strain)) * fs
Pxx, freqs = mlab.psd(strain_seg, Fs=fs, NFFT=fs)

# Apply a Blackman Window, and plot the FFT(Fast Fourier Transform)
plt.loglog(freq, abs(freq_domain))
plt.axis([10, fs/2.0, 1e-25, 1e-19])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('Strain / Hz')
plt.title("Blackman Window FFT plot")
plt.show()

# Make a PSD (Power Spectral Density)
plt.loglog(freqs, Pxx)
plt.axis([10, 2000, 1e-50, 1e-38])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('PSD')
plt.title("Welch's method, PSD plot")
plt.show()

# Plot the ASD(Amplitude Spectral Density), easier for comparison with FFT (Square root of PSD)
plt.loglog(freqs, np.sqrt(Pxx))
plt.axis([10, 2000, 1e-25, 1e-19])
plt.grid('on')
plt.xlabel('Freq (Hz)')
plt.ylabel('ASD (Strain / Hz$^{1/2})$')
plt.title("Welch's method, ASD plot")
plt.show()

# Make a spectrogram
#spectrogram usually shows what we already knew from the PSD: there is a lot more power at very low frequencies than high frequencies.
NFFT = 1024
short_window = np.blackman(NFFT)
spec_power, freqs, bins, im = plt.specgram(
    strain_seg, NFFT=NFFT, Fs=fs,
    window=short_window
)
plt.xlabel('Time (s)')
plt.ylabel('Freq (Hz)')
plt.title("Spectrogram")
plt.show()