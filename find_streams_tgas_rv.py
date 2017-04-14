import imp, os, glob
import uncertainties.umath as umath

from uncertainties import unumpy
from astropy.table import Table, join, Column, unique
from velocity_transformations import *
from find_streams_plots import *
from find_streams_analysis import *
from find_streams_selection import *
from find_streams_analysis_functions import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

imp.load_source('galmove', '../tSNE_test/convert_gal_movement.py')
from galmove import *

imp.load_source('veltrans', '../tSNE_test/velocity_transform.py')
from veltrans import *

imp.load_source('tsne', '../tSNE_test/tsne_functions.py')
from tsne import *

imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *


# --------------------------------------------------------
# ---------------- Functions -----------------------------
# --------------------------------------------------------
def parse_selected_streams(subdir):
    png_files = glob.glob(subdir+'/stream_*_radiant-3D.png')
    result = list([[], [], [], []])
    for filename in png_files:
        filename_split = filename.split('/')[1].split('_')
        result[0].append(float(filename_split[filename_split.index('ra') + 1]))
        result[1].append(float(filename_split[filename_split.index('dec') + 1]))
        result[2].append(float(filename_split[filename_split.index('rv') + 1]))
        if 'dist' in filename_split:
            result[3].append(float(filename_split[filename_split.index('rv') + 1]))
        else:
            result[3].append(None)
    return result


def quick_tnse_plot(tsne_data, path='tsne.png', colorize=None):
    if colorize is None:
        plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=0.5)
    else:
        plt.scatter(tsne_data['tsne_axis_1'], tsne_data['tsne_axis_2'], lw=0, s=0.5, c=colorize, cmap='jet')
        plt.colorbar()
    plt.savefig(path, dpi=300)
    plt.close()


def run_tsne(data, norm=False, out_file='tsn_results.fits', distance='manhattan', perp=80, theta=0.2):
    if os.path.isfile(out_file):
        tsne_final = Table.read(out_file)
    else:
        if norm:
            data_temp = np.array(data)
            norm_param = normalize_data(data_temp, method='standardize')
            print 'Normalization parameters:'
            print norm_param
            tsne_result = bh_tsne(data_temp, no_dims=2, perplexity=perp, theta=theta, randseed=-1,
                                  verbose=True, distance=distance)
        else:
            tsne_result = bh_tsne(data, no_dims=2, perplexity=perp, theta=theta, randseed=-1,
                                  verbose=True, distance=distance)
        tsne_ax1, tsne_ax2 = tsne_results_to_columns(tsne_result)
        tsne_final = tsne_table_with_results(tgas_data, ['sobject_id', 'galah_id', 'RAVE_OBS_ID', 'RAVEID'],
                                             tsne_ax1, tsne_ax2)
        tsne_final.write(out_file, format='fits')
    return tsne_final


# --------------------------------------------------------
# ---------------- Constants and settings ----------------
# --------------------------------------------------------
TSNE_PERFORM = False
TSNE_NORM = True

# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH and RAVE data - used for radial velocity data
print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'  # the same for gigli and local pc
rave_data_dir = '/home/klemen/RAVE_data/'

out_file_fits = 'RAVE_GALAH_TGAS_stack.fits'
if os.path.isfile(out_file_fits):
    tgas_data = Table.read(out_file_fits)
else:
    galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
    galah_tgas_xmatch = Table.read(galah_data_dir+'galah_tgas_xmatch.csv')
    rave_param = Table.read(rave_data_dir+'RAVE_DR5.fits')
    rave_tgas = Table.read(rave_data_dir+'RAVE_TGAS.fits')
    # do some column name housekeeping for consistent naming between the data sets
    galah_param['rv_guess'].name = 'RV'
    galah_param['e_rv_guess'].name = 'RV_error'
    galah_param['feh_guess'].name = 'M_H'
    rave_param['HRV'].name = 'RV'
    rave_param['eHRV'].name = 'RV_error'
    rave_param['parallax'].name = 'parallax_binney'
    rave_param['Met_N_K'].name = 'M_H'
    rave_tgas['ra'].name = 'ra_gaia'
    rave_tgas['dec'].name = 'dec_gaia'

    # join datasets
    print 'Joining RAVE and GALAH sets into one'
    use_columns = ['ra_gaia', 'dec_gaia', 'RV', 'RV_error', 'parallax', 'parallax_error', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error', 'M_H']
    use_columns_galah = ['sobject_id', 'galah_id']
    use_columns_galah.extend(use_columns)
    use_columns_rave = ['RAVE_OBS_ID', 'RAVEID']
    use_columns_rave.extend(use_columns)
    galah_joined = join(galah_param, galah_tgas_xmatch, keys='sobject_id', join_type='inner')
    rave_joined = join(rave_param, rave_tgas, keys='RAVE_OBS_ID', join_type='inner')
    tgas_data = vstack([galah_joined[use_columns_galah], rave_joined[use_columns_rave]], join_type='outer')
    tgas_data.write(out_file_fits, format='fits')

# perform some data cleaning and housekeeping
idx_ok = tgas_data['parallax'] > 0  # remove negative parallaxes - objects far away or problems in data reduction
# idx = np.logical_and(np.abs(tgas_data['pmra']) < tgas_data['pmra_error'],
#                      np.abs(tgas_data['pmdec']) < tgas_data['pmdec_error'])
# print tgas_data['pmra', 'pmra_error', 'pmdec', 'pmdec_error'][idx]
# print np.sum(np.max(tgas_data['pmra']))
# print np.sum(np.max(tgas_data['pmdec']))
# validate data
idx_ok = np.logical_and(idx_ok,
                        np.isfinite(tgas_data['ra_gaia','dec_gaia','pmra','pmdec','RV','parallax'].to_pandas().values).all(axis=1))
idx_ok = np.logical_and(idx_ok,
                        tgas_data['RV_error'] < 20.)
print 'Number of removed observations: '+str(len(tgas_data)-np.sum(idx_ok))
tgas_data = tgas_data[idx_ok]
print 'Number of observations: '+str(len(tgas_data))

# remove problems with masks
tgas_data = tgas_data.filled()

# remove duplicates
tgas_data = unique(tgas_data, keys=['ra_gaia', 'dec_gaia'], keep='first')

print 'Final number of unique objects: '+str(len(tgas_data))

# --------------------------------------------------------
# ---------------- Compute different galactic velocities - cartesian and cylindrical
# --------------------------------------------------------
# add systematic errors as described in
# Gaia DR1 - Lindegren 2016 and Gaia DR1: Open clusters astrometry - Gaia Collaboration 2017

# convert parallax to parsec distance
tgas_data.add_column(Column(1e3/tgas_data['parallax'].data, name='parsec'))
# add systematic error to the parallax uncertainties as suggested for the TGAS dataset
tgas_data['parallax_error'] = np.sqrt(tgas_data['parallax_error'].data**2 + 0.3**2)
# limit data by parsec
tgas_data = tgas_data[np.logical_and(tgas_data['parsec'] < 750, tgas_data['parsec'] > 0)]
print 'Number of points after distance limits: ' + str(len(tgas_data))
# define parallax values with uncertainties using uncertainties library
# parallax_u = unumpy.uarray(tgas_data['parallax'], tgas_data['parallax_error'])
# parsec_u = 1e3/parallax_u
# # define other parameters with uncertainties
# pmra_u = unumpy.uarray(tgas_data['pmra'], tgas_data['pmra_error'])
# pmdec_u = unumpy.uarray(tgas_data['pmdec'], tgas_data['pmdec_error'])
# rv_u = unumpy.uarray(tgas_data['RV'], tgas_data['RV_error'])

# TODO BEGIN - remove those calculations, should be performed on the fly in the STREAM
# cylindrical uvw velocity computation
# u_vel, v_vel, w_vel = gal_uvw(np.array(tgas_data['ra_gaia']), np.array(tgas_data['dec_gaia']),
#                               np.array(tgas_data['pmra']), np.array(tgas_data['pmdec']),
#                               np.array(tgas_data['RV']), np.array(tgas_data['parallax']))
# uvw_vel = np.transpose(np.vstack((u_vel, v_vel, w_vel)))
# u_vel = None; v_vel = None; w_vel = None
# TODO END

if TSNE_PERFORM:
    perp = 80
    theta = 0.2
    suffix = '_perp_{:02.0f}_theta_{:01.1f}'.format(perp, theta)
    if TSNE_NORM:
        suffix += '_norm'
    # run t-sne on newly computed uvw velocities
    file_uvw_tsne_out = 'streams_tsne_uvw'+suffix+'.fits'
    uvw_tsne_final = run_tsne(uvw_vel, norm=TSNE_NORM, out_file='streams_tsne_uvw' + suffix + '.fits', distance='manhattan',
                              perp=perp, theta=theta)
    quick_tnse_plot(uvw_tsne_final, path='streams_tsne_uvw'+suffix+'.png')

    # run t-sne on newly computed cartesian xyz velocities
    file_xyz_tsne_out = 'streams_tsne_xyz'+suffix+'.fits'
    xyz_tsne_final = run_tsne(xyz_vel, norm=TSNE_NORM, out_file='streams_tsne_xyz' + suffix + '.fits', distance='manhattan',
                              perp=perp, theta=theta)
    quick_tnse_plot(xyz_tsne_final, path='streams_tsne_xyz'+suffix+'.png')
    # exit program
    raise SystemExit

# --------------------------------------------------------
# ---------------- Stream search parameters --------------
# --------------------------------------------------------
# # radial velocity of observed stream - values for the Aquarius stream
# rv_stream = 200.  # km/s
# # radiant coordinates for stream
# ra_stream = np.deg2rad(164.)  # alpha - RA
# de_stream = np.deg2rad(13.)  # delta - DEC

# stream search criteria
rv_step = 5.  # km/s, rv in the radiant of the stream
ra_step = 5.  # deg
dec_step = 5.  # deg
dist_step = None  # 200  # pc

# results thresholds in percent from theoretical value
rv_thr = 15.
pmra_thr = 15.
pmdec_thr = 15.
parsec_thr = 10.

# --------------------------------------------------------
# ---------------- Evaluation of possible streams --------
# --------------------------------------------------------
manual_stream_radiants = None  # [[90], [0], [45], [None]]  # list of ra, dec, rv values
# manual_stream_radiants = parse_selected_streams('Streams_investigation_lower-thr_selected')
# iterate trough all possible combinations for the initial conditions of the stream (rv, ra, dec)
if manual_stream_radiants is not None:
    ra_combinations = manual_stream_radiants[0]
    dec_combinations = manual_stream_radiants[1]
    rv_combinations = manual_stream_radiants[2]
    dist_combinations = manual_stream_radiants[3]
else:
    rv_range = np.arange(5, 200, rv_step)
    ra_range = np.arange(0, 360, ra_step)
    dec_range = np.arange(-90, 90, dec_step)
    if dist_step is not None:
        dist_range = np.arange(50, 1600, dist_step)
    else:
        dist_range = [np.nan]
    # create a grid of all possible combination
    stream_mesh = np.meshgrid(ra_range, dec_range, rv_range, dist_range)
    ra_combinations = stream_mesh[0].flatten()
    dec_combinations = stream_mesh[1].flatten()
    rv_combinations = stream_mesh[2].flatten()
    dist_combinations = stream_mesh[3].flatten()
n_combinations = len(ra_combinations)
print 'Total number of stream combinations that will be evaluated: '+str(n_combinations)

parallax_MC = MC_parallax(tgas_data['parallax'], tgas_data['parallax_error'], 100)
move_to_dir('Streams_investigation_MC-sigma_RV-sigma')
for i_stream in range(n_combinations):
    ra_stream = ra_combinations[i_stream]
    dec_stream = dec_combinations[i_stream]
    rv_stream = rv_combinations[i_stream]
    dist_stream = dist_combinations[i_stream]
    # combine results by RV values
    move_to_dir(str(rv_stream))
    # velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
    v_xyz_stream = compute_xyz_vel(np.deg2rad(ra_stream), np.deg2rad(dec_stream), rv_stream)

    # compute predicted stream pmra and pmdec, based on stars ra, dec and parsec distance
    rv_stream_predicted = compute_rv(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
                                     v_xyz_stream)

    # pmra_stream_predicted_u = compute_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                        parsec_u, v_xyz_stream)
    # pmdec_stream_predicted_u = compute_pmdec(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                          parsec_u, v_xyz_stream)
    # parsec_pmra_u = compute_distance_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                       pmra_u, v_xyz_stream)
    # parsec_pmdec_u = compute_distance_pmdec(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                         pmdec_u, v_xyz_stream)

    # option 1 - match proper motion values in the same sense as described in the Gaia open clusters paper
    # idx_match = np.logical_and(match_proper_motion_values(pmra_stream_predicted_u, pmra_u, dispersion=0.,
    #                                                       sigma=3, prob_thr=None),
    #                            match_proper_motion_values(pmdec_stream_predicted_u, pmdec_u, dispersion=0.,
    #                                                       sigma=3, prob_thr=None))

    # option 2 - based on calculated and measured parallax values by Zwitter
    # idx_match = match_parsec_values(parsec_u, parsec_pmra_u, parsec_pmdec_u, prob_thr=2.)

    # selection based on RV observation
    # idx_rv_match = match_rv_values(rv_stream_predicted, rv_u, sigma=1., prob_thr=None)

    # first final selection
    # idx_possible = np.logical_and(idx_match, idx_rv_match)
    # pmra_stream_predicted = unumpy.nominal_values(pmra_stream_predicted_u)
    # pmdec_stream_predicted = unumpy.nominal_values(pmdec_stream_predicted_u)


    pmra_stream_predicted = compute_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
                                         tgas_data['parsec'], v_xyz_stream)
    # pmra_error_stream_predicted = np.abs(compute_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                            tgas_data['parsec_error'], v_xyz_stream))
    #
    pmdec_stream_predicted = compute_pmdec(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
                                           tgas_data['parsec'], v_xyz_stream)
    # pmdec_error_stream_predicted = np.abs(compute_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                             tgas_data['parsec_error'], v_xyz_stream))
    #
    # # compute predicted distances based on measured pmra and pmdec and assumed velocities of observed stream
    # parsec_pmra = compute_distance_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                     tgas_data['pmra'], v_xyz_stream)
    # parsec_error_pmra = np.abs(compute_distance_pmra(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                           tgas_data['pmra_error'], v_xyz_stream))
    # parsec_pmdec = compute_distance_pmdec(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                       tgas_data['pmdec'], v_xyz_stream)
    # parsec_error_pmdec = np.abs(compute_distance_pmdec(np.deg2rad(tgas_data['ra_gaia']), np.deg2rad(tgas_data['dec_gaia']),
    #                                             tgas_data['pmdec_error'], v_xyz_stream))
    #
    # # filter data based on measurement errors
    # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['pmra'], tgas_data['pmra_error'],
    #                                                                         pmra_stream_predicted, pmra_error_stream_predicted),
    #                               match_two_1d_vectors_within_uncertainties(tgas_data['pmdec'], tgas_data['pmdec_error'],
    #                                                                         pmdec_stream_predicted, pmdec_error_stream_predicted))
    # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['RV'], tgas_data['RV_error'],
    #                                                                         rv_stream_predicted, 0.),
    #                               idx_possible)

    # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['parsec'], tgas_data['parsec_error'],
    #                                                                         parsec_pmra, parsec_error_pmra),
    #                               match_two_1d_vectors_within_uncertainties(tgas_data['parsec'], tgas_data['parsec_error'],
    #                                                                         parsec_pmdec, parsec_error_pmdec))
    # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['RV'], tgas_data['RV_error'],
    #                                                                         rv_stream_predicted, 0.),
    #                               idx_possible)

    # # filter data based on predefined search criteria
    # idx_possible = np.logical_and(np.logical_and(np.abs((pmra_stream_predicted - tgas_data['pmra'])/pmra_stream_predicted) <= pmra_thr/100.,
    #                                              np.abs((pmdec_stream_predicted - tgas_data['pmdec'])/pmdec_stream_predicted) <= pmdec_thr/100.),
    #                               np.logical_and(np.abs((rv_stream_predicted - tgas_data['RV'])/rv_stream_predicted) <= rv_thr/100.,
    #                                              tgas_data['parallax'].data > 0.))

    # # filter data based on their observed and calculated distance
    # parsec_mean = (parsec_pmra + parsec_pmdec + tgas_data['parsec'])/3.
    # idx_possible = np.logical_and(np.logical_and(np.abs((parsec_mean - parsec_pmra)/parsec_mean) <= parsec_thr/100.,
    #                                              np.abs((parsec_mean - parsec_pmdec) / parsec_mean) <= parsec_thr/100.),
    #                               np.logical_and(np.abs((parsec_mean - tgas_data['parsec']) / parsec_mean) <= parsec_thr/100.,
    #                                              tgas_data['parsec'].data > 0.))

    # # confirm the star selection based on the radial velocity measurement
    # idx_possible = np.logical_and(idx_possible,
    #                               np.abs((rv_stream_predicted - tgas_data['RV']) / rv_stream_predicted) <= rv_thr / 100)

    idx_pm_match = proper_motion_match_mc(
        tgas_data['ra_gaia', 'dec_gaia', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'],
        parallax_MC, v_xyz_stream, std=3., percent=50.)

    # selection based on RV observation
    idx_rv_match = match_values_within_std(tgas_data['RV'], tgas_data['RV_error'],
                                           rv_stream_predicted, std=1.5)

    idx_possible = np.logical_and(idx_pm_match, idx_rv_match)

    n_possible = np.sum(idx_possible)
    print ' Possible members: ' + str(n_possible) + ' for stream from ra={:3.1f} dec={:2.1f} with rv velocity of {:3.1f}'.format(ra_stream, dec_stream, rv_stream)

    if n_possible < 5:
        continue
    else:
        # plot results for visual and later automatised inspection of the candidates
        suffix = '_ra_{:05.1f}_dec_{:04.1f}_rv_{:05.1f}'.format(ra_stream, dec_stream, rv_stream)
        stream_radiant = [ra_stream, dec_stream]

        # plot_ideal distribution at the distance of 0.5kpx
        ra_sim, dec_sim = np.meshgrid(np.arange(0, 360, 20), np.arange(-90, 91, 20))
        pmra_sim = compute_pmra(np.deg2rad(ra_sim), np.deg2rad(dec_sim), 500, v_xyz_stream)
        pmdec_sim = compute_pmdec(np.deg2rad(ra_sim), np.deg2rad(dec_sim), 500, v_xyz_stream)

        # plot results of the comparison of proper motion
        plot_members_location_motion(tgas_data, pmra_stream_predicted, pmdec_stream_predicted, idx_possible,
                                     path='stream' + suffix + '_proper.png', radiant=stream_radiant, add_errors=False,
                                     title='Observations similar to the predictions at stellar distance.')

        # plot_members_location_motion_theoretical(ra_sim, dec_sim, pmra_sim, pmdec_sim,
        #                                          radiant=stream_radiant, path='stream'+suffix+'_proper_sim.png',
        #                                          title='Theoretical proper motion for stars at 0.5 kpc.')

        # plot radial velocities
        plot_members_location_velocity(tgas_data, rv_ref=rv_stream_predicted, idx=idx_possible, radiant=stream_radiant,
                                       path='stream' + suffix + '_rv_sim.png', title='Radial velocity of possible members.')
        # plot theoretical curves for pmra, pmdec and rv, based on stream location and stream radial velocity
        # plot_theoretical_motion(v_xyz_stream, img_prefix='stream'+suffix, dist=1000)

        # --------------------------------------------------------
        # ---------------- Stream analysis -----------------------
        # --------------------------------------------------------
        # perform possible stream analysis and filter out stars that do not belong in the same cluster
        d = STREAM(tgas_data[idx_possible], radiant=stream_radiant)
        d.monte_carlo_simulation(samples=20, distribution='normal')
        d.estimate_stream_dimensions(path='stream'+suffix+'_radiant.png')
        d.estimate_stream_dimensions(path='stream'+suffix+'_radiant_MC.png', MC=True)
        # d.stream_show(view_pos=stream_radiant, path='stream'+suffix+'_radiant-3D.png')
        # d.stream_show(view_pos=stream_radiant, path='stream'+suffix+'_radiant-3D_MC.png', MC=True)
        # d.find_overdensities(path='stream'+suffix+'_radiant_dbscan.png')
        # analysis_res = d.analyse_overdensities(xyz_stream=v_xyz_stream, path_prefix='stream' + suffix + '_radiant_dbscan')
        # d.plot_velocities(uvw=True, xyz=False, path='stream'+suffix+'_vel_uvw.png')
        d.plot_velocities(uvw=False, xyz=True, xyz_stream=v_xyz_stream, path='stream'+suffix+'_vel_xyz.png')
        d.plot_velocities(uvw=False, xyz=True, xyz_stream=v_xyz_stream, path='stream'+suffix+'_vel_xyz_MC.png', MC=True)
    os.chdir('..')
