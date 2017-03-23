import imp, os
import astropy.units as u
import astropy.coordinates as coord

from astropy.table import Table, join, Column, vstack
from velocity_transformations import *
from find_streams_plots import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH and RAVE data - used for radial velocity data
print 'Reading data sets'
galah_data_dir = '/home/klemen/GALAH_data/'  # the same for gigli and local pc
rave_data_dir = '/home/klemen/RAVE_data/'
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_tgas_xmatch = Table.read(galah_data_dir+'galah_tgas_xmatch.csv')
rave_param = Table.read(rave_data_dir+'RAVE_DR5.fits')
rave_tgas = Table.read(rave_data_dir+'RAVE_TGAS.fits')
# do some column name housekeeping for consistent naming between the data sets
galah_param['rv_guess'].name = 'RV'
rave_param['HRV'].name = 'RV'
rave_param['parallax'].name = 'parallax_photo'
rave_tgas['ra'].name = 'ra_gaia'
rave_tgas['dec'].name = 'dec_gaia'

# join datasets
print 'Joining RAVE and GALAH sets into one'
use_columns = ['ra_gaia', 'dec_gaia', 'RV', 'parallax', 'pmra', 'pmdec', 'pmdec_error', 'pmra_error', 'parallax_error']
use_columns_galah = ['sobject_id', 'galah_id']
use_columns_galah.extend(use_columns)
use_columns_rave = ['RAVE_OBS_ID', 'RAVEID']
use_columns_rave.extend(use_columns)
galah_joined = join(galah_param, galah_tgas_xmatch, keys='sobject_id', join_type='inner')
rave_joined = join(rave_param, rave_tgas, keys='RAVE_OBS_ID', join_type='inner')
tgas_data = vstack([galah_joined[use_columns_galah], rave_joined[use_columns_rave]], join_type='outer')

# convert parallax to parsec distance
star_parsec = (tgas_data['parallax'].data * u.mas).to(u.parsec, equivalencies=u.parallax()) # use of .data to remove units as they are not handled correctly by astropy
tgas_data.add_column(Column(star_parsec, name='parsec'))
# remove problems with masks
tgas_data = tgas_data.filled()

print 'Total number of observations: '+str(len(tgas_data))

# --------------------------------------------------------
# ---------------- Stream search parameters --------------
# --------------------------------------------------------
# # radial velocity of observed stream - values for the Aquarius stream
# rv_stream = 200.  # km/s
# # radiant coordinates for stream
# ra_stream = np.deg2rad(164.)  # alpha - RA
# de_stream = np.deg2rad(13.)  # delta - DEC

# stream search criteria
rv_step = 20.  # km/s, rv in the radiant of the stream
ra_step = 10.  # deg
dec_step = 10.  # deg

# results thresholds in percent from theoretical value
rv_thr = 20.
pmra_thr = 20.
pmdec_thr = 20.

# --------------------------------------------------------
# ---------------- Evaluation of possible streams --------
# --------------------------------------------------------
move_to_dir('Streams_investigation')
# iterate trough all possible combinations for the initial conditions of the stream (rv, ra, dec)
for rv_stream in np.arange(5, 350, rv_step):
    for ra_stream in np.arange(0, 360, ra_step):
        for dec_stream in np.arange(-90, 91, dec_step):

            # velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
            v_xyz_stream = compute_xyz_vel(np.deg2rad(ra_stream), np.deg2rad(dec_stream), rv_stream)

            # compute predicted stream pmra and pmdec, based on stars ra, dec and parsec distance
            rv_stream_predicted = compute_rv(np.deg2rad(tgas_data['ra_gaia']),
                                             np.deg2rad(tgas_data['dec_gaia']),
                                             v_xyz_stream)
            pmra_stream_predicted = compute_pmra(np.deg2rad(tgas_data['ra_gaia']),
                                                 np.deg2rad(tgas_data['dec_gaia']),
                                                 tgas_data['parsec'],
                                                 v_xyz_stream)
            pmdec_stream_predicted = compute_pmdec(np.deg2rad(tgas_data['ra_gaia']),
                                                   np.deg2rad(tgas_data['dec_gaia']),
                                                   tgas_data['parsec'],
                                                   v_xyz_stream)
            # filter data based on predefined search criteria
            idx_possible = np.logical_and(np.logical_and(np.abs((pmra_stream_predicted - tgas_data['pmra'])/pmra_stream_predicted) <= pmra_thr/100.,
                                                         np.abs((pmdec_stream_predicted - tgas_data['pmdec'])/pmdec_stream_predicted) <= pmdec_thr/100.),
                                          np.logical_and(np.abs((rv_stream_predicted - tgas_data['RV'])/rv_stream_predicted) <= rv_thr/100.,
                                                         tgas_data['parsec'].data > 0.))
            n_possible = np.sum(idx_possible)
            print ' Possible members: ' + str(n_possible) + ' for stream from ra={:3.1f} dec={:2.1f} with rv speed of {:3.1f}'.format(ra_stream, dec_stream, rv_stream)
            if n_possible < 5:
                continue
            else:
                # plot results for visual and later automatised inspection of the candidates
                suffix = '_ra_{:05.1f}_dec_{:04.1f}_rv_{:05.1f}'.format(ra_stream, dec_stream, rv_stream)

                # plot_ideal distribution at the distance of 0.5kpx
                ra_sim, dec_sim = np.meshgrid(np.arange(0, 360, 20), np.arange(-90, 91, 20))
                pmra_sim = compute_pmra(np.deg2rad(ra_sim), np.deg2rad(dec_sim), 500, v_xyz_stream)
                pmdec_sim = compute_pmdec(np.deg2rad(ra_sim), np.deg2rad(dec_sim), 500, v_xyz_stream)
                plot_members_location_motion_theoretical(ra_sim, dec_sim, pmra_sim, pmdec_sim,
                                                         radiant=[ra_stream, dec_stream], path='stream'+suffix+'_sim.png',
                                                         title='Theoretical proper motion for stars at 0.5 kpc.')
                # plot results of the comparison
                plot_members_location_motion(tgas_data, pmra_stream_predicted, pmdec_stream_predicted, idx_possible,
                                             path='stream'+suffix+'.png', radiant=[ra_stream, dec_stream],
                                             title='Observations similar to the predictions at stellar distance.')
                plot_members_location_velocit()
                # plot_theoretical_motion(v_xyz_stream, img_prefix='stream'+suffix, dist=1000)

