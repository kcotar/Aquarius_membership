import numpy as np
from velocity_transformations import compute_pmra, compute_pmdec, compute_distance_pmra, compute_distance_pmdec

from scipy.stats import norm
from multiprocessing import Pool

# from joblib import Parallel, delayed
# from functools import partial


def MC_values(parallax, parallax_error, n_MC):
    mc_plx = list([])
    for i_row in range(len(parallax)):
        mc_plx.append(np.random.normal(parallax[i_row], parallax_error[i_row], n_MC))
    return mc_plx


def match_values_within_std(obs, obs_error, model, std=1.):
    difference = np.abs(obs - model)
    return difference < (obs_error * std)


def match_values_probability(obs, obs_error, model, log_prob=-3.):
    model_logpdf = norm.logpdf(model, loc=obs, scale=obs_error)
    log_like = np.sum(model_logpdf) / len(model)
    # TODO: possible solution, determine log_prob from log_like histogram
    return log_like >= log_prob
    # likelihood ration test
    # TODO: if it will be needed
    # ratio = 2*ln(log_observed_model/log_reference_model)


def prob_e(vr, er, v1, e1, v2=None, e2=None):
    if v2 is not None and e2 is not None:
        en1 = np.sqrt(er**2 + e1**2)
        en2 = np.sqrt(er**2 + e2**2)
        return -0.5 * ((vr - v1)**2/en1**2 + (vr - v2)**2/en2**2 + np.log(2*np.pi*en1**2) + np.log(2*np.pi*en2**2))
    else:
        en1 = np.sqrt(er ** 2 + e1 ** 2)
        return -0.5 * ((vr - v1)**2/en1**2 + np.log(2*np.pi*en1**2))


def match3(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = np.deg2rad(data_row['ra'])
    obj_dec = np.deg2rad(data_row['dec'])
    parallax_pmra_predicted = compute_distance_pmra(obj_ra, obj_dec, GLOBAL_pmra[i_row], GLOBAL_stream, parallax=True)
    parallax_pmdec_predicted = compute_distance_pmdec(obj_ra, obj_dec, GLOBAL_pmdec[i_row], GLOBAL_stream, parallax=True)
    pmx_med1 = np.median(parallax_pmra_predicted)
    pmx_med2 = np.median(parallax_pmdec_predicted)
    pmx_std1 = np.std(parallax_pmra_predicted)
    pmx_std2 = np.std(parallax_pmdec_predicted)
    if pmx_std2/pmx_med2 < 0.2 and pmx_std1/pmx_med1 < 0.2:
        return prob_e(data_row['parallax'], data_row['parallax_error'],
                      pmx_med1, pmx_std1, pmx_med2, pmx_std2) > GLOBAL_std
    else:
        return False


def match2(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = np.deg2rad(data_row['ra'])
    obj_dec = np.deg2rad(data_row['dec'])
    parallax_pmra_predicted = compute_distance_pmra(obj_ra, obj_dec, GLOBAL_pmra[i_row], GLOBAL_stream, parallax=True)
    parallax_pmdec_predicted = compute_distance_pmdec(obj_ra, obj_dec, GLOBAL_pmdec[i_row], GLOBAL_stream, parallax=True)
    idx_match = np.logical_and(match_values_within_std(data_row['parallax'], data_row['parallax_error'], parallax_pmra_predicted, std=GLOBAL_std),
                               match_values_within_std(data_row['parallax'], data_row['parallax_error'], parallax_pmdec_predicted, std=GLOBAL_std))
    return np.sum(idx_match)


def match4(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = np.deg2rad(data_row['ra'])
    obj_dec = np.deg2rad(data_row['dec'])
    obj_parsec = 1e3 / GLOBAL_plx[i_row]
    pmra_stream_predicted = compute_pmra(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
    pmdec_stream_predicted = compute_pmdec(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
    pmra_med = np.median(pmra_stream_predicted)
    pmdec_med = np.median(pmdec_stream_predicted)
    pmra_std = np.std(pmra_stream_predicted)
    pmdes_std = np.std(pmdec_stream_predicted)
    pmra_prob = prob_e(data_row['pmra'], data_row['pmra_error'], pmra_med, pmra_std) > GLOBAL_std
    pmdec_prob = prob_e(data_row['pmdec'], data_row['pmdec_error'], pmdec_med, pmdes_std) > GLOBAL_std
    return pmra_prob and pmdec_prob


def match(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = np.deg2rad(data_row['ra'])
    obj_dec = np.deg2rad(data_row['dec'])
    obj_parsec = 1e3 / GLOBAL_plx[i_row]
    pmra_stream_predicted = compute_pmra(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
    pmdec_stream_predicted = compute_pmdec(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
    pmra_prob = match_values_probability(data_row['pmra'], data_row['pmra_error'], pmra_stream_predicted, log_prob=GLOBAL_std)
    pmdec_prob = match_values_probability(data_row['pmdec'], data_row['pmdec_error'], pmdec_stream_predicted, log_prob=GLOBAL_std)
    return pmra_prob and pmdec_prob
    # idx_match = np.logical_and(match_values_within_std(data_row['pmra'], data_row['pmra_error'], pmra_stream_predicted, std=GLOBAL_std),
    #                            match_values_within_std(data_row['pmdec'], data_row['pmdec_error'], pmdec_stream_predicted, std=GLOBAL_std))
    # return np.sum(idx_match)


def observations_match_mc(data, xyz_stream, parallax_mc=None, pmra_mc=None, pmdec_mc=None, std=1., percent=50.):
    global GLOBAL_data
    GLOBAL_data = data
    global GLOBAL_plx
    GLOBAL_plx = parallax_mc
    global GLOBAL_pmra
    GLOBAL_pmra = pmra_mc
    global GLOBAL_pmdec
    GLOBAL_pmdec = pmdec_mc
    global GLOBAL_stream
    GLOBAL_stream = xyz_stream
    global GLOBAL_std
    GLOBAL_std = std
    if parallax_mc is not None:
        n_mc = len(parallax_mc[0])
    else:
        n_mc = len(pmra_mc[0])
    n_rows = len(data)

    # n_matches = np.ndarray(n_rows)
    # for i_row in range(n_rows):
    #     n_matches[i_row] = match(i_row)
    # import matplotlib.pyplot as plt
    # print n_matches
    # plt.hist(n_matches[:, 0], bins=150, range=(-20,10))
    # plt.show()
    # plt.close()
    # plt.hist(n_matches[:, 1], bins=150, range=(-20,10))
    # plt.show()
    # plt.close()

    pool = Pool(processes=25)
    if parallax_mc is not None:
        # n_matches = np.array(pool.map(match, range(n_rows)))
        n_matches = np.array(pool.map(match4, range(n_rows)))
    elif pmra_mc is not None and pmdec_mc is not None:
        # n_matches = np.array(pool.map(match2, range(n_rows)))
        n_matches = np.array(pool.map(match3, range(n_rows)))
    pool.close()
    # n_matches = np.array(Parallel(n_jobs=25)(delayed(match)(i_row) for i_row in range(n_rows)))

    GLOBAL_data = None
    GLOBAL_plx = None
    GLOBAL_pmra = None
    GLOBAL_pmdec = None
    GLOBAL_stream = None
    GLOBAL_std = None
    # return (100.*n_matches/n_mc) > percent
    return n_matches

