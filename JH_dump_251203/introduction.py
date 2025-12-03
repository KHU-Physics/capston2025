#set desired n bewteen 0 and 4094.5 to plot the 10000 samples graph

import numpy as np
import matplotlib.pylab as plt
import h5py

# Open the File
filename = 'H-H1_GWOSC_O4a_4KHZ_R1-1368350720-4096.hdf5'
dataFile = h5py.File(filename, 'r')

# Explore the file
#for key in dataFile.keys():
#    print(key)

# Read in strain data
strain = dataFile['strain']['Strain']
ts = dataFile['strain']['Strain'].attrs['Xspacing']
print(ts)
#print(f"ts = {ts}s, sample rate = {1/ts}Hz")

# Print out some meta data
#-------------------------
metaKeys = dataFile['meta'].keys()
meta = dataFile['meta']
#for key in metaKeys:
#    print(key, meta[key])

# Create a time vector
gpsStart = meta['GPSstart'][()]
duration = meta['Duration'][()]
gpsEnd   = gpsStart + duration

time = np.arange(gpsStart, gpsEnd, ts)

# Plot the time series
plt.figure(figsize=(15,3))
plt.plot(time, strain[()])
plt.xlabel('GPS Time (s)')
plt.ylabel('H1 Strain')
plt.title("Hanford (H1) 4096 seconds, Timeline 1368350720")
plt.show()

# Zoom in the time series
numsamples = 10000
#startTime  = 1264316116.0
#set starTime between 1368350720-1368354813.5 (1368350720+4096-2.5 / 10000 샘플은 약 2.4초)
#or set n between 0-4094.5
n=200
startTime  = (1368350720.0+n)
startIndex = np.min(np.nonzero(startTime < time))
time_seg   = time[startIndex:(startIndex+numsamples)]
strain_seg = strain[startIndex:(startIndex+numsamples)]
plt.plot(time_seg, strain_seg)
plt.xlabel('GPS Time (s)')
plt.ylabel('H1 Strain')
plt.title("Hanford (H1) Zoomed in data, 10000 samples, 4096 Hz")
plt.show()