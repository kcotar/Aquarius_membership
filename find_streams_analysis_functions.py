import numpy as np
from velocity_transformations import compute_pmra, compute_pmdec, compute_distance_pmra, compute_distance_pmdec
from joblib import Parallel, delayed

from multiprocessing import Pool
from functools import partial

import astropy.units as un
import astropy.coordinates as coord
import gala.coordinates as gal_coord


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
    # covert computed pmra/pmdec to pml/pmb
    ra_dec_coord = coord.ICRS(ra=np.full_like(GLOBAL_pmra[i_row], data_row['ra_gaia']) * un.deg,
                              dec=np.full_like(GLOBAL_pmdec[i_row], data_row['dec_gaia']) * un.deg)
    ra_dec_pm = np.vstack((GLOBAL_pmra[i_row], GLOBAL_pmdec[i_row])) * un.mas / un.yr
    l_b_pm = gal_coord.pm_icrs_to_gal(ra_dec_coord, ra_dec_pm)
    # compute predicted values
    parallax_pmra_predicted = compute_distance_pmra(np.deg2rad(data_row['l_gaia']), np.deg2rad(data_row['b_gaia']),
                                                    l_b_pm[0].value, GLOBAL_stream,
                                                    parallax=True, lsr_vel=GLOBAL_lsr, to_solar=True)
    parallax_pmdec_predicted = compute_distance_pmdec(np.deg2rad(data_row['l_gaia']), np.deg2rad(data_row['b_gaia']),
                                                      l_b_pm[1].value, GLOBAL_stream,
                                                      parallax=True, lsr_vel=GLOBAL_lsr, to_solar=True)
    # if np.mean(parallax_pmra_predicted)>0 and np.mean(parallax_pmdec_predicted)>0:
    #     print np.mean(parallax_pmra_predicted), np.mean(parallax_pmdec_predicted), np.std(parallax_pmra_predicted), np.std(parallax_pmdec_predicted)
    #     print data_row['parallax']
    #     print ''
    # match computed values with observations
    idx_match = np.logical_and(match_values_within_std(data_row['parallax'], data_row['parallax_error'], parallax_pmra_predicted, std=GLOBAL_std),
                               match_values_within_std(data_row['parallax'], data_row['parallax_error'], parallax_pmdec_predicted, std=GLOBAL_std))
    return np.sum(idx_match)


def match(i_row):
    data_row = GLOBAL_data[i_row]
    obj_ra = data_row['l_gaia']
    obj_dec = data_row['b_gaia']
    obj_parsec = 1e3 / GLOBAL_plx[i_row]
    # compute predicted values
    pml_stream_predicted = compute_pmra(np.deg2rad(obj_ra), np.deg2rad(obj_dec), obj_parsec, GLOBAL_stream,
                                        lsr_vel=GLOBAL_lsr, to_solar=True)
    pmb_stream_predicted = compute_pmdec(np.deg2rad(obj_ra), np.deg2rad(obj_dec), obj_parsec, GLOBAL_stream,
                                         lsr_vel=GLOBAL_lsr, to_solar=True)
    # print np.mean(pml_stream_predicted), np.mean(pmb_stream_predicted), np.std(pml_stream_predicted), np.std(pmb_stream_predicted)
    # covert computed pml/pmb to pmra/pmdec
    l_b_coord = coord.Galactic(l=np.full_like(pml_stream_predicted, obj_ra)*un.deg,
                               b=np.full_like(pmb_stream_predicted, obj_dec)*un.deg)
    l_b_pm = np.vstack((pml_stream_predicted, pmb_stream_predicted)) * un.mas / un.yr  # must be of shape (2,N)
    ra_dec_pm = gal_coord.pm_gal_to_icrs(l_b_coord, l_b_pm)
    # match computed values with observations
    idx_match = np.logical_and(match_values_within_std(data_row['pmra'], data_row['pmra_error'], ra_dec_pm[0].value, std=GLOBAL_std),
                               match_values_within_std(data_row['pmdec'], data_row['pmdec_error'], ra_dec_pm[1].value, std=GLOBAL_std))
    return np.sum(idx_match)


def observations_match_mc(data, xyz_stream, parallax_mc=None, pmra_mc=None, pmdec_mc=None,
                          std=1., percent=50., lsr_vel=None, n_cpu=1):
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
    global GLOBAL_lsr
    GLOBAL_lsr = lsr_vel
    if parallax_mc is not None:
        n_mc = len(parallax_mc[0])
    else:
        n_mc = len(pmra_mc[0])
    n_rows = len(data)
    if n_cpu <= 1:
        n_matches = np.ndarray(n_rows)
        for i_row in range(n_rows):
            # print 'Matching row', i_row, n_rows
            if parallax_mc is not None:
                idx_matches = match(i_row)
            elif pmra_mc is not None and pmdec_mc is not None:
                idx_matches = match2(i_row)
            n_matches[i_row] = np.sum(idx_matches)
    else:
        pool = Pool(processes=n_cpu)
        if parallax_mc is not None:
            n_matches = np.array(pool.map(match, range(n_rows)))
        elif pmra_mc is not None and pmdec_mc is not None:
            # NOTE: not yet adapted to l and b coordinates and lsr correction
            n_matches = np.array(pool.map(match2, range(n_rows)))
        pool.close()
    # n_matches = np.array(Parallel(n_jobs=25)(delayed(match)(i_row) for i_row in range(n_rows)))
    GLOBAL_data = None
    GLOBAL_plx = None
    GLOBAL_pmra = None
    GLOBAL_pmdec = None
    GLOBAL_stream = None
    GLOBAL_std = None
    GLOBAL_lsr = None
    return (100.*n_matches/n_mc) > percent

