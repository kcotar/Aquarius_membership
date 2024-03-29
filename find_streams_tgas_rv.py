import imp, os, glob
import pandas as pd

from astropy.table import Table, join, unique
from find_streams_plots import *
from find_streams_analysis import *
from find_streams_analysis_functions import *
from sys import argv

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


def delete_file(filename):
    if os.path.exists(filename):
        os.remove(filename)


# --------------------------------------------------------
# ---------------- Constants and settings ----------------
# --------------------------------------------------------
TSNE_PERFORM = False
TSNE_NORM = True
CLUSTER_ANALYSIS = False
SKIPP_ANALYSED = True
WORK_WITH_GALACTIC_XYZ = True

# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH and RAVE data - used for radial velocity data
print 'Reading data sets'
data_dir = '/data4/cotar/'

# print 'Reading Galaxia simulations'
# simulations matching GALAH survey
# simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/GALAH/'
# simulation_fits = 'galaxy_galah_complete.fits'  # complete all-sky simulation
# simulation_fits = 'galaxy_galah_fields.fits'  # simulation for the observed fields only
# simulations matching RAVE survey
# simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/RAVE/'
# simulation_fits = 'galaxy_rave_complete.fits'
# simulation_fits = 'galaxy_rave_complete_fields_r3.0.fits'
# read
# galaxia_data = Table.read(simulation_dir + simulation_fits)

galah_data = Table.read(data_dir + 'sobject_iraf_53_gaia.fits')['source_id', 'sobject_id']
tgas_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')
tgas_data = tgas_data[tgas_data['parallax_error']/tgas_data['parallax'] < 0.2]

# perform some data cleaning and housekeeping
idx_ok = tgas_data['parallax'] > 0  # remove negative parallaxes - objects far away or problems in data reduction

idx_ok = np.logical_and(idx_ok,
                        np.isfinite(tgas_data['ra','dec','pmra','pmdec','rv','parallax'].to_pandas().values).all(axis=1))
# idx_ok = np.logical_and(idx_ok,
#                         tgas_data['rv_error'] < 5.)
print 'Number of removed observations: '+str(len(tgas_data)-np.sum(idx_ok))
tgas_data = tgas_data[idx_ok]
print 'Number of observations: '+str(len(tgas_data))

# remove problems with masks
tgas_data = tgas_data.filled()

print 'Final number of unique objects: '+str(len(tgas_data))

# --------------------------------------------------------
# ---------------- Compute different galactic velocities - cartesian and cylindrical
# --------------------------------------------------------
# convert parallax to parsec distance
tgas_data.add_column(Column(1e3/tgas_data['parallax'].data, name='parsec'))
# limit data by parsec
tgas_data = tgas_data[np.logical_and(tgas_data['parsec'] < 5000, tgas_data['parsec'] > 0)]
print 'Number of points after distance limits: ' + str(len(tgas_data))

# --------------------------------------------------------
# ---------------- Known clusters - STREAM class test ----
# --------------------------------------------------------
if CLUSTER_ANALYSIS:
    move_to_dir('Stream_known_clusters_Dias_2014')
    # clusters dataset 1
    # stars_cluster_data = Table.read(galah_data_dir+'sobject_clusterstars_1.0.fits')
    # field_id = 'cluster_name'
    # clusters dataset 2
    # stars_cluster_data = Table.read(galah_data_dir+'galah_clusters_Schmeja_xmatch_2014.csv', format='ascii.csv')
    # idx_probable = np.logical_and(np.logical_and(stars_cluster_data['Pkin'] > 0.0, stars_cluster_data['PJH'] > 0.0), stars_cluster_data['Ps'] == 1)
    # stars_cluster_data = stars_cluster_data[idx_probable]
    # field_id = 'MWSC'
    # clusters dataset 3
    # stars_cluster_data = Table.read(galah_data_dir+'galah_clusters_Kharachenko_xmatch_2005.csv', format='ascii.csv')
    # field_id = 'Cluster'
    # clusters dataset 4
    stars_cluster_data = Table.read(galah_data_dir+'galah_clusters_Dias_xmatch_2014.csv', format='ascii.csv')
    idx_probable = stars_cluster_data['P'] > 50.0
    idx_probable = np.logical_and(idx_probable, stars_cluster_data['db'] == 0)
    idx_probable = np.logical_and(idx_probable, stars_cluster_data['of'] == 0)
    stars_cluster_data = stars_cluster_data[idx_probable]
    field_id = 'Cluster'
    # end of datasets
    star_cluster_ids = set(stars_cluster_data[field_id])
    for cluster in star_cluster_ids:
        cluster_sub = tgas_data[np.in1d(tgas_data['sobject_id'], stars_cluster_data[stars_cluster_data[field_id]==cluster]['sobject_id'])]
        if len(cluster_sub) == 0:
            continue
        print cluster
        print cluster_sub['ra', 'dec', 'rv', 'pmra', 'pmdec', 'parallax', 'parallax_error', 'pmra_error', 'pmdec_error']
        stream_obj = STREAM(cluster_sub)
        stream_obj.plot_intersections(path=str(cluster)+'.png', GUI=False)
        stream_obj.monte_carlo_simulation(samples=100, distribution='normal')
        stream_obj.plot_intersections(path=str(cluster)+'_MC.png', MC=True, GUI=False)
    os.chdir('..')

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
ra_step = 10.  # deg
dec_step = 10.  # deg
dist_step = None  # 200  # pc

# results thresholds in percent from theoretical value
rv_thr = 15.
pmra_thr = 15.
pmdec_thr = 15.
parsec_thr = 10.

# selection from terminal
in_args = argv
if len(in_args) > 1:
    rv_selection = float(in_args[1])
    print ' Input RV selection: '+str(rv_selection)
else:
    rv_selection = 10.

# --------------------------------------------------------
# ---------------- Evaluation of possible streams --------
# --------------------------------------------------------
# manual_stream_radiants = [[20,45,140,240,370], [-10,-30,20,10,50], [45,45,45,45,45], [None]]  # list of ra, dec, rv values
# manual_stream_radiants = [[164.], [13.], [200.], [None]]  # list of ra, dec, rv values
manual_stream_radiants = None
# manual_stream_radiants = parse_selected_streams('Streams_investigation_lower-thr_selected')
# iterate trough all possible combinations for the initial conditions of the stream (rv, ra, dec)
if manual_stream_radiants is not None:
    ra_combinations = manual_stream_radiants[0]
    dec_combinations = manual_stream_radiants[1]
    rv_combinations = manual_stream_radiants[2]
    dist_combinations = manual_stream_radiants[3]
else:
    rv_range = np.arange(rv_selection, rv_selection+rv_step, rv_step)
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

n_MC = 100
# parallax_MC = MC_values(tgas_data['parallax'], tgas_data['parallax_error'], n_MC)
pmra_MC = MC_values(tgas_data['pmra'], tgas_data['pmra_error'], n_MC)
pmdec_MC = MC_values(tgas_data['pmdec'], tgas_data['pmdec_error'], n_MC)

out_dir = 'Streams_investigation_MC_density_analysis_px'
# out_dir = 'Streams_investigation_MC_density_analysis_pm'

move_to_dir(out_dir)
for i_stream in range(n_combinations):
    ra_stream = ra_combinations[i_stream]
    dec_stream = dec_combinations[i_stream]
    rv_stream = rv_combinations[i_stream]
    # dist_stream = dist_combinations[i_stream]

    # check for repeated stream conditions at poles
    if (dec_stream == 90. or dec_stream == -90.) and ra_stream > 0.:
        continue

    move_to_dir(str(rv_stream))

    suffix = 'stream_ra_{:05.1f}_dec_{:04.1f}_rv_{:05.1f}'.format(ra_stream, dec_stream, rv_stream)
    print 'Working on ' + suffix

    # velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
    l_b_stream = coord.ICRS(ra=ra_stream * un.deg, dec=dec_stream * un.deg).transform_to(coord.Galactic)
    v_xyz_stream = compute_xyz_vel(np.deg2rad(ra_stream), np.deg2rad(dec_stream), rv_stream)
    v_xyz_stream_gal = compute_xyz_vel(np.deg2rad(l_b_stream.l.value), np.deg2rad(l_b_stream.b.value), rv_stream)

    # compute predicted stream pmra and pmdec, based on stars ra, dec and parsec distance
    rv_stream_predicted = compute_rv(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
                                     v_xyz_stream)

    selection_file = suffix + '_obj.txt'
    if not os.path.exists(selection_file):
        # idx_pm_match = observations_match_mc(tgas_data['ra', 'dec', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'],
        #                                      v_xyz_stream,
        #                                      parallax_mc=parallax_MC, std=-4.)
        idx_pm_match = observations_match_mc(tgas_data['ra', 'dec',  'parallax', 'parallax_error'],
                                             v_xyz_stream,
                                             pmra_mc=pmra_MC, pmdec_mc=pmdec_MC, std=-9.)
        print idx_pm_match
        print 'PM matched: '+str(np.sum(idx_pm_match))
        idx_rv_match = match_values_within_std(tgas_data['rv'], tgas_data['rv_error'], rv_stream_predicted, std=1.5)
        print 'RV matched: ' + str(np.sum(idx_rv_match))
        idx_possible = np.logical_and(idx_pm_match, idx_rv_match)
        print 'Together  : ' + str(np.sum(idx_possible))
        # OR different approach is comparision of velocity vectors itself
        # idx_possible = tgas_data['rv_error'] > 0.

        txt_out = open(selection_file, 'w')
        txt_out.write(','.join([str(pos) for pos in np.where(idx_possible)[0]]))
        txt_out.close()
    else:
        if SKIPP_ANALYSED:
            print 'Skipping, already analyzed.'
            os.chdir('..')
            continue
        idx_possible = pd.read_csv(selection_file, header=None, sep=',').values[0]

    if np.sum(idx_possible) < 5:
        os.chdir('..')
        continue

    # data subset
    tgas_data_selected = tgas_data[idx_possible]

    pmra_stream_predicted = compute_pmra(np.deg2rad(tgas_data_selected['ra']), np.deg2rad(tgas_data_selected['dec']),
                                         tgas_data_selected['parsec'], v_xyz_stream)

    pmdec_stream_predicted = compute_pmdec(np.deg2rad(tgas_data_selected['ra']), np.deg2rad(tgas_data_selected['dec']),
                                           tgas_data_selected['parsec'], v_xyz_stream)

    pm_fig, pm_ax = plt.subplots(1, 1)
    pm_ax.set(xlim=(0, 360), ylim=(-90, 90))
    pm_ax.scatter(tgas_data_selected['ra'], tgas_data_selected['dec'], lw=0, c='black', s=5)
    pm_ax.scatter(ra_stream, dec_stream, lw=0, s=15, c='black', marker='*')
    pm_ax.quiver(tgas_data_selected['ra'], tgas_data_selected['dec'], tgas_data_selected['pmra'],
                 tgas_data_selected['pmdec'],
                 pivot='tail', scale=QUIVER_SCALE, color='green', width=QUIVER_WIDTH)
    pm_ax.quiver(tgas_data_selected['ra'], tgas_data_selected['dec'],
                 pmra_stream_predicted, pmdec_stream_predicted,
                 pivot='tail', scale=QUIVER_SCALE, color='red', width=QUIVER_WIDTH)
    pm_fig.tight_layout()
    plt.savefig(suffix+'_1.png', dpi=350)
    plt.close()

    # begin analysis
    if WORK_WITH_GALACTIC_XYZ:
        stream_obj = STREAM(tgas_data_selected, to_galactic=True)
        stream_obj.monte_carlo_simulation(samples=75, distribution='normal')
        # stream_obj.plot_intersections(xyz_vel_stream=v_xyz_stream_gal, path=suffix + '_2.png', MC=False, GUI=False)
        stream_obj.plot_intersections(xyz_vel_stream=v_xyz_stream_gal, path=suffix + '_2_MC.png', MC=True, GUI=False)
    else:
        stream_obj = STREAM(tgas_data_selected, to_galactic=False)
        stream_obj.monte_carlo_simulation(samples=75, distribution='normal')
        # stream_obj.plot_intersections(xyz_vel_stream=v_xyz_stream, path=suffix+'_2.png', MC=False, GUI=False)
        stream_obj.plot_intersections(xyz_vel_stream=v_xyz_stream, path=suffix+'_2_MC.png', MC=True, GUI=False)

    peaks_txt = 'analysis_peaks_results.txt'
    txt_o = open(peaks_txt, 'a')
    txt_o.write('\n\n\n')
    txt_o.write(suffix+'\n')
    txt_o.close()
    stream_obj.show_density_field(bandwidth=30., kernel='epanechnikov', MC=True, peaks=True, analyze_peaks=True,
                                  GUI=False, path=suffix+'_3_MC.png', plot_orbits=True,
                                  grid_size=1500, grid_bins=2000, recompute=False,
                                  txt_out=peaks_txt, galah=galah_data)

    # plot orbits

    # stream_obj.phase_intersects_analysis(GUI=False, path=suffix+'_3_MC.png', phase_step=3.)
    # peaks_galaxia = 'analysis_peaks_galaxia_compare.txt'
    # txt_o = open(peaks_galaxia, 'a')
    # txt_o.write('\n\n\n')
    # txt_o.write(suffix + '\n')
    # txt_o.close()
    # stream_obj.compare_with_simulation(galaxia_data, r_vel=10., xyz_stream=v_xyz_stream_gal,
    #                                    txt_out=peaks_galaxia, img_path=suffix + '_4.png')
    os.chdir('..')
    continue

    peaks_galaxia = 'analysis_peaks_galaxia_compare.txt'
    if WORK_WITH_GALACTIC_XYZ:
        if len(stream_obj.meaningful_peaks) > 0:
            txt_o = open(peaks_galaxia, 'a')
            txt_o.write('\n\n\n')
            txt_o.write(suffix + '\n')
            txt_o.close()
            stream_obj.compare_with_simulation(galaxia_data, r_vel=10., xyz_stream=v_xyz_stream_gal,
                                               txt_out=peaks_galaxia, img_path=suffix + '_4.png')
        else:
            # delete png plots without any significant information
            for suf in ['_1', '_2', '_2_MC', '_3_MC']:
                delete_file(suffix + suf + '.png')

    # for samples in list([10, 25, 40]):
    #     for eps in list([10, 14, 18]):
    #         out_name = suffix+'_3_s{:2.0f}_e{:2.0f}'.format(samples, eps)
    #         stream_obj.show_dbscan_field(samples=samples, eps=eps, GUI=False, peaks=True, path=out_name+'.png')
    #         stream_obj.evaluate_dbscan_field(MC=True, path=out_name+'.txt')

    os.chdir('..')



    # # pmra_stream_predicted_u = compute_pmra(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                        parsec_u, v_xyz_stream)
    # # pmdec_stream_predicted_u = compute_pmdec(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                          parsec_u, v_xyz_stream)
    # # parsec_pmra_u = compute_distance_pmra(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                       pmra_u, v_xyz_stream)
    # # parsec_pmdec_u = compute_distance_pmdec(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                         pmdec_u, v_xyz_stream)
    #
    # # option 1 - match proper motion values in the same sense as described in the Gaia open clusters paper
    # # idx_match = np.logical_and(match_proper_motion_values(pmra_stream_predicted_u, pmra_u, dispersion=0.,
    # #                                                       sigma=3, prob_thr=None),
    # #                            match_proper_motion_values(pmdec_stream_predicted_u, pmdec_u, dispersion=0.,
    # #                                                       sigma=3, prob_thr=None))
    #
    # # option 2 - based on calculated and measured parallax values by Zwitter
    # # idx_match = match_parsec_values(parsec_u, parsec_pmra_u, parsec_pmdec_u, prob_thr=2.)
    #
    # # selection based on RV observation
    # # idx_rv_match = match_rv_values(rv_stream_predicted, rv_u, sigma=1., prob_thr=None)
    #
    # # first final selection
    # # idx_possible = np.logical_and(idx_match, idx_rv_match)
    # # pmra_stream_predicted = unumpy.nominal_values(pmra_stream_predicted_u)
    # # pmdec_stream_predicted = unumpy.nominal_values(pmdec_stream_predicted_u)
    #
    #
    #
    # # pmra_error_stream_predicted = np.abs(compute_pmra(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                            tgas_data['parsec_error'], v_xyz_stream))
    # #
    #
    # # pmdec_error_stream_predicted = np.abs(compute_pmra(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                             tgas_data['parsec_error'], v_xyz_stream))
    # #
    # # # compute predicted distances based on measured pmra and pmdec and assumed velocities of observed stream
    # # parsec_pmra = compute_distance_pmra(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                     tgas_data['pmra'], v_xyz_stream)
    # # parsec_error_pmra = np.abs(compute_distance_pmra(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                           tgas_data['pmra_error'], v_xyz_stream))
    # # parsec_pmdec = compute_distance_pmdec(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                       tgas_data['pmdec'], v_xyz_stream)
    # # parsec_error_pmdec = np.abs(compute_distance_pmdec(np.deg2rad(tgas_data['ra']), np.deg2rad(tgas_data['dec']),
    # #                                             tgas_data['pmdec_error'], v_xyz_stream))
    # #
    # # # filter data based on measurement errors
    # # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['pmra'], tgas_data['pmra_error'],
    # #                                                                         pmra_stream_predicted, pmra_error_stream_predicted),
    # #                               match_two_1d_vectors_within_uncertainties(tgas_data['pmdec'], tgas_data['pmdec_error'],
    # #                                                                         pmdec_stream_predicted, pmdec_error_stream_predicted))
    # # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['rv'], tgas_data['rv_error'],
    # #                                                                         rv_stream_predicted, 0.),
    # #                               idx_possible)
    #
    # # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['parsec'], tgas_data['parsec_error'],
    # #                                                                         parsec_pmra, parsec_error_pmra),
    # #                               match_two_1d_vectors_within_uncertainties(tgas_data['parsec'], tgas_data['parsec_error'],
    # #                                                                         parsec_pmdec, parsec_error_pmdec))
    # # idx_possible = np.logical_and(match_two_1d_vectors_within_uncertainties(tgas_data['rv'], tgas_data['rv_error'],
    # #                                                                         rv_stream_predicted, 0.),
    # #                               idx_possible)
    #
    # # # filter data based on predefined search criteria
    # # idx_possible = np.logical_and(np.logical_and(np.abs((pmra_stream_predicted - tgas_data['pmra'])/pmra_stream_predicted) <= pmra_thr/100.,
    # #                                              np.abs((pmdec_stream_predicted - tgas_data['pmdec'])/pmdec_stream_predicted) <= pmdec_thr/100.),
    # #                               np.logical_and(np.abs((rv_stream_predicted - tgas_data['rv'])/rv_stream_predicted) <= rv_thr/100.,
    # #                                              tgas_data['parallax'].data > 0.))
    #
    # # # filter data based on their observed and calculated distance
    # # parsec_mean = (parsec_pmra + parsec_pmdec + tgas_data['parsec'])/3.
    # # idx_possible = np.logical_and(np.logical_and(np.abs((parsec_mean - parsec_pmra)/parsec_mean) <= parsec_thr/100.,
    # #                                              np.abs((parsec_mean - parsec_pmdec) / parsec_mean) <= parsec_thr/100.),
    # #                               np.logical_and(np.abs((parsec_mean - tgas_data['parsec']) / parsec_mean) <= parsec_thr/100.,
    # #                                              tgas_data['parsec'].data > 0.))
    #
    # # # confirm the star selection based on the radial velocity measurement
    # # idx_possible = np.logical_and(idx_possible,
    # #                               np.abs((rv_stream_predicted - tgas_data['rv']) / rv_stream_predicted) <= rv_thr / 100)
    #
    # # --------------------------------------------------------
    # # ---------------- Final stream matching methods ---------
    # # --------------------------------------------------------
    # # METHOD 1
    # # idx_pm_match = observations_match_mc(
    # #     tgas_data['ra', 'dec', 'pmra', 'pmra_error', 'pmdec', 'pmdec_error'],
    # #     v_xyz_stream, parallax_mc=parallax_MC, std=3., percent=50.)
    # # METHOD 2
    # idx_pm_match = observations_match_mc(
    #     tgas_data['ra', 'dec', 'parallax', 'parallax_error'],
    #     v_xyz_stream, pmra_mc=pmra_MC, pmdec_mc=pmdec_MC, std=2., percent=50.)
    #
    # # selection based on RV observation
    # idx_rv_match = match_values_within_std(tgas_data['rv'], tgas_data['rv_error'],
    #                                        rv_stream_predicted, std=2.)
    #
    # idx_possible = np.logical_and(idx_pm_match, idx_rv_match)
    #
    # n_possible = np.sum(idx_possible)
    # print ' Possible members: ' + str(n_possible) + ' for stream from ra={:3.1f} dec={:2.1f} with rv velocity of {:3.1f}'.format(ra_stream, dec_stream, rv_stream)
    #
    # if n_possible < 5:
    #     continue
    # else:
    #     # combine results by RV values
    #     move_to_dir(str(rv_stream))
    #     # plot results for visual and later automatised inspection of the candidates
    #     suffix = '_ra_{:05.1f}_dec_{:04.1f}_rv_{:05.1f}'.format(ra_stream, dec_stream, rv_stream)
    #     stream_radiant = [ra_stream, dec_stream]
    #
    #     # plot_ideal distribution at the distance of 0.5kpx
    #     ra_sim, dec_sim = np.meshgrid(np.arange(0, 360, 20), np.arange(-90, 91, 20))
    #     pmra_sim = compute_pmra(np.deg2rad(ra_sim), np.deg2rad(dec_sim), 500, v_xyz_stream)
    #     pmdec_sim = compute_pmdec(np.deg2rad(ra_sim), np.deg2rad(dec_sim), 500, v_xyz_stream)
    #
    #     # plot results of the comparison of proper motion
    #     # plot_members_location_motion(tgas_data, pmra_stream_predicted, pmdec_stream_predicted, idx_possible,
    #     #                              path='stream' + suffix + '_proper.png', radiant=stream_radiant, add_errors=False,
    #     #                              title='Observations similar to the predictions at stellar distance.')
    #
    #     # plot_members_location_motion_theoretical(ra_sim, dec_sim, pmra_sim, pmdec_sim,
    #     #                                          radiant=stream_radiant, path='stream'+suffix+'_proper_sim.png',
    #     #                                          title='Theoretical proper motion for stars at 0.5 kpc.')
    #
    #     # plot radial velocities
    #     # plot_members_location_velocity(tgas_data, rv_ref=rv_stream_predicted, idx=idx_possible, radiant=stream_radiant,
    #     #                                path='stream' + suffix + '_rv_sim.png', title='Radial velocity of possible members.')
    #     # plot theoretical curves for pmra, pmdec and rv, based on stream location and stream radial velocity
    #     # plot_theoretical_motion(v_xyz_stream, img_prefix='stream'+suffix, dist=1000)
    #
    #     # --------------------------------------------------------
    #     # ---------------- Stream analysis -----------------------
    #     # --------------------------------------------------------
    #     # perform possible stream analysis and filter out stars that do not belong in the same cluster
    #     d = STREAM(tgas_data[idx_possible], radiant=stream_radiant)
    #     d.monte_carlo_simulation(samples=20, distribution='normal')
    #     d.estimate_stream_dimensions(path='stream'+suffix+'_radiant.png')
    #     d.estimate_stream_dimensions(path='stream'+suffix+'_radiant_MC.png', MC=True)
    #     # d.stream_show(view_pos=stream_radiant, path='stream'+suffix+'_radiant-3D.png')
    #     # d.stream_show(view_pos=stream_radiant, path='stream'+suffix+'_radiant-3D_MC.png', MC=True)
    #     # d.find_overdensities(path='stream'+suffix+'_radiant_dbscan.png')
    #     # analysis_res = d.analyse_overdensities(xyz_stream=v_xyz_stream, path_prefix='stream' + suffix + '_radiant_dbscan')
    #     # d.plot_velocities(uvw=True, xyz=False, path='stream'+suffix+'_vel_uvw.png')
    #     # d.plot_velocities(uvw=False, xyz=True, xyz_stream=v_xyz_stream, path='stream'+suffix+'_vel_xyz.png')
    #     # d.plot_velocities(uvw=False, xyz=True, xyz_stream=v_xyz_stream, path='stream'+suffix+'_vel_xyz_MC.png', MC=True)
    #     os.chdir('..')
