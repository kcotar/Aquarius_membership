import numpy as np


def match_two_1d_vectors_within_uncertainties(val_1, error_1, val_2, error_2):
    # compute limiting values of both
    val_1_min = val_1 - error_1
    val_1_max = val_1 + error_1
    val_2_min = val_2 - error_2
    val_2_max = val_2 + error_2
    # determine matching values
    idx_match = np.logical_and(val_1_max > val_2_min,
                               val_2_max > val_1_min)
    return idx_match
