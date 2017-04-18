import numpy as np
from velocity_transformations import compute_pmra, compute_pmdec, compute_distance_pmra, compute_distance_pmdec
from joblib import Parallel, delayed

from multiprocessing import Pool
from functools import partial


def MC_values(parallax, parallax_error, n_MC):
    mc_plx = list([])
    for i_row in range(len(parallax)):
        mc_plx.append(np.random.normal(parallax[i_row], parallax_error[i_row], n_MC))
    return mc_plx


def match_values_within_std(obs, obs_error, model, std=1.):
    difference = np.abs(obs - model)
    return difference < (obs_error * std)


def match2(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = np.deg2rad(data_row['ra_gaia'])
    obj_dec = np.deg2rad(data_row['dec_gaia'])
    parallax_pmra_predicted = compute_distance_pmra(obj_ra, obj_dec, GLOBAL_pmra[i_row], GLOBAL_stream, parallax=True)
    parallax_pmdec_predicted = compute_distance_pmdec(obj_ra, obj_dec, GLOBAL_pmdec[i_row], GLOBAL_stream, parallax=True)
    idx_match = np.logical_and(match_values_within_std(data_row['parallax'], data_row['parallax_error'], parallax_pmra_predicted, std=GLOBAL_std),
                               match_values_within_std(data_row['parallax'], data_row['parallax_error'], parallax_pmdec_predicted, std=GLOBAL_std))
    return np.sum(idx_match)

def match(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = np.deg2rad(data_row['ra_gaia'])
    obj_dec = np.deg2rad(data_row['dec_gaia'])
    obj_parsec = 1e3 / GLOBAL_plx[i_row]
    pmra_stream_predicted = compute_pmra(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
    pmdec_stream_predicted = compute_pmdec(obj_ra, obj_dec, obj_parsec, GLOBAL_stream)
    idx_match = np.logical_and(match_values_within_std(data_row['pmra'], data_row['pmra_error'], pmra_stream_predicted, std=GLOBAL_std),
                               match_values_within_std(data_row['pmdec'], data_row['pmdec_error'], pmdec_stream_predicted, std=GLOBAL_std))
    return np.sum(idx_match)


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
    # def match2(i_row):
    #     data_row = data[i_row]
    #     obj_ra = np.deg2rad(data_row['ra_gaia'])
    #     obj_dec = np.deg2rad(data_row['dec_gaia'])
    #     obj_parsec = 1e3 / parallax_mc[i_row]
    #     pmra_stream_predicted = compute_pmra(obj_ra, obj_dec, obj_parsec, xyz_stream)
    #     pmdec_stream_predicted = compute_pmdec(obj_ra, obj_dec, obj_parsec, xyz_stream)
    #     idx_match = np.logical_and(np.abs(pmdec_stream_predicted - data_row['pmdec']) < (std * data_row['pmdec_error']),
    #                                np.abs(pmra_stream_predicted - data_row['pmra']) < (std * data_row['pmra_error']))
    #     n_matches[i_row] = np.sum(idx_match)
    pool = Pool(processes=15)
    if parallax_mc is not None:
        n_matches = np.array(pool.map(match, range(n_rows)))
    elif pmra_mc is not None and pmdec_mc is not None:
        n_matches = np.array(pool.map(match2, range(n_rows)))
    pool.close()
    # n_matches = np.array(Parallel(n_jobs=25)(delayed(match)(i_row) for i_row in range(n_rows)))
    GLOBAL_data = None
    GLOBAL_plx = None
    GLOBAL_pmra = None
    GLOBAL_pmdec = None
    GLOBAL_stream = None
    GLOBAL_std = None
    return (100.*n_matches/n_mc) > percent

