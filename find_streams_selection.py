import numpy as np
from uncertainties import unumpy


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


def parsec_limits(parallax, parallax_error):
    """

    :param parallax:
    :param parallax_error:
    :return:
    """
    min_parsec = 1e3/np.array(parallax + parallax_error)
    idx_inf = min_parsec < 0
    if np.sum(idx_inf) > 0:
        min_parsec[idx_inf] = np.inf
    max_parsec = 1e3/np.array(parallax - parallax_error)
    idx_inf = max_parsec < 0
    if np.sum(idx_inf) > 0:
        max_parsec[idx_inf] = np.inf
    return min_parsec, max_parsec


def match_proper_motion_values(model, observation, dispersion=0., prob_thr=0.8):
    """

    :param model:
    :param observation:
    :param dispersion:
    :param prob_thr:
    :return:
    """
    final_std = np.sqrt(unumpy.std_devs(model)**2 + unumpy.std_devs(observation)**2 + dispersion**2)
    value_prob = np.exp(-1. * ((unumpy.nominal_values(model) - unumpy.nominal_values(observation)) ** 2) / (2. * final_std ** 2))
    return value_prob > prob_thr


def match_parsec_values(parsec1, parsec2, parsec3, prob_thr=-1.):
    median_val = np.median(np.vstack((unumpy.nominal_values(parsec1), unumpy.nominal_values(parsec2), unumpy.nominal_values(parsec3))), axis=0)
    log_prob = 0.5*np.log(((unumpy.nominal_values(parsec1) - median_val) / unumpy.std_devs(parsec1)) ** 2 +
                          ((unumpy.nominal_values(parsec2) - median_val) / unumpy.std_devs(parsec2)) ** 2 +
                          ((unumpy.nominal_values(parsec3) - median_val) / unumpy.std_devs(parsec3)) ** 2)
    return log_prob < prob_thr


