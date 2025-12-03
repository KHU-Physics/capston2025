#Round 2, plot the graph with data quality information
#With the information, we can automatically find the 'good data' within the raw strain data and ignore bad data.

import numpy as np
import matplotlib.pyplot as plt
import h5py

filename = 'L-L1_LOSC_4_V1-1126256640-4096.hdf5' #1126259462
dataFile = h5py.File(filename, 'r')
gpsStart = dataFile['meta']['GPSstart'][()]

dqInfo = dataFile['quality']['simple']
bitnameList = dqInfo['DQShortnames'][()]
descriptionList = dqInfo['DQDescriptions'][()]
nbits = len(bitnameList)
dqmask = dqInfo['DQmask'][()]
value = dqmask[2400]

data_channel = 0
data_mask = (dqmask >> data_channel) & 1

burst_cat2_channel = 5
burst_cat2_mask = (dqmask >> burst_cat2_channel) & 1

goodData_mask_1hz = data_mask & burst_cat2_mask

# fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, sharey=True)
# ax0.plot(goodData_mask_1hz)
# ax0.set_title('Good data')
# ax1.plot(burst_cat2_mask)
# ax1.set_title('BURST_CAT2')
# ax2.plot(data_mask)
# ax2.set_title('DATA')
# ax2.axis([0, 4096, -1, 2])
# ax2.set_xlabel('Time (s)')
# plt.show()

dummy = np.zeros(goodData_mask_1hz.shape)
masked_dummy = np.ma.masked_array(dummy, np.logical_not(goodData_mask_1hz) )
segments = np.ma.flatnotmasked_contiguous(masked_dummy)
segList = [(int(seg.start+gpsStart), int(seg.stop+gpsStart)) for seg in segments]
print(segList)

N      = 10000
strain = dataFile['strain']['Strain']
dt     = dataFile['strain']['Strain'].attrs['Xspacing']
for (begin, end) in segList:
    plt.figure()
    rel_time = np.arange(0, end - begin, dt)
    # Convert begin and end to indices
    index_begin = int((begin - gpsStart) / dt)
    index_end   = int((end - gpsStart) / dt)
    seg_strain = strain[index_begin:index_end]
    # Plot N values
    plt.plot(rel_time[0:N], seg_strain[0:N])
    plt.xlabel('Seconds since GPS ' + str(begin) )
    plt.title(f"{filename}, "" Good data\n"f"SegList: ({begin}, {end})")
    plt.show()