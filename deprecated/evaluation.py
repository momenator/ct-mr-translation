import numpy as np

# Mean Absolute Error
def computeMAE(real_scan, rec_scan):
    assert(real_scan.shape == rec_scan.shape)
    return np.sum(np.abs(real_scan - rec_scan)).mean()


# Mean Squared error
def computeMSE(real_scan, rec_scan):
    assert(real_scan.shape == rec_scan.shape)
    return np.square((real_scan - rec_scan)).mean()


# Peak Signal to Noise Ratio
def computePSNR(real_scan, rec_scan):
    # scan was normalised to [1, -1], so peak value is 1
    peak = 1.0
    return 20 * np.log10(peak / np.sqrt(computeMSE(real_scan, rec_scan))))


def get_peak_value(real_scan, mask):
    # convert all mask to 1
    mask = np.where(mask == 0, mask, 1)
    # get the maximum value where there is a mask
    return np.multiply(real_scan, mask).max()

