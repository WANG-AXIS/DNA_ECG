import numpy as np

# Complementary filter1 and filter2 for Physionet Data
N = 18000
bins = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
bin_length = int(N / 2 / len(bins))
filter1 = bins[0] * np.ones(bin_length, dtype=np.float32)
for i in range(1, len(bins)):
    filter1 = np.concatenate((filter1, bins[i] * np.ones(bin_length)))
filter1 = np.append(filter1, filter1[-1])
filter1 = np.concatenate([filter1, np.flip(filter1[1:-1])])

filter2 = 1 - filter1


# Complementary filter1 and filter2 for CPSC 2018 Data
N = 24000
bins = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
bin_length = int(N / 2 / len(bins))
china_filter1 = bins[0] * np.ones(bin_length, dtype=np.float32)
for i in range(1, len(bins)):
    china_filter1 = np.concatenate((china_filter1, bins[i] * np.ones(bin_length)))
china_filter1 = np.append(china_filter1, china_filter1[-1])
china_filter1 = np.concatenate([china_filter1, np.flip(china_filter1[1:-1])])

china_filter2 = 1 - china_filter1