import os, glob
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.coordinates as coord

from astroquery.simbad import Simbad
from astropy.table import Table, join, Column
from velocity_transformations import *

# --------------------------------------------------------
# ---------------- Read Data -----------------------------
# --------------------------------------------------------
# read GALAH data
galah_data_dir = '/home/klemen/GALAH_data/'  # the same for gigli and local pc
galah_param = Table.read(galah_data_dir+'sobject_iraf_param_1.1.fits')
galah_tgas_xmatch = Table.read(galah_data_dir+'galah_tgas_xmatch.csv')
# join both datasets
tgas_data = join(galah_param, galah_tgas_xmatch, keys='sobject_id', join_type='inner')
tgas_fits_files = glob.glob('GaiaTgas/TgasSource_*.fits')

# radial velocity of observed stream
rv_stream = 200.  # km/s
# radiant coordinates for stream
ra_stream = np.deg2rad(164.)  # alpha - RA
de_stream = np.deg2rad(13.)  # delta - DEC

# velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
v_xyz_stream = compute_xyz_vel(ra_stream, de_stream, rv_stream)

# get theoretical observed rv pmra pmdec, based on streams rv values
ra_range = np.deg2rad(np.arange(0, 360, 0.5))
plt.plot(ra_range, compute_pmra(ra_range, np.deg2rad(10.), 500., v_xyz_stream))
plt.savefig('pmra.png')
plt.close()
for rad_deg in np.arange(-20., 90., 10.):
    plt.plot(ra_range, compute_pmdec(ra_range, np.deg2rad(rad_deg), 500., v_xyz_stream))
plt.savefig('pmdec.png')
plt.close()
for rad_deg in np.arange(-20., 90., 10.):
    plt.plot(ra_range, compute_rv(ra_range, np.deg2rad(rad_deg), v_xyz_stream))
plt.savefig('rv.png')
plt.close()

rv_thr = 20.
pmra_thr = 5.
pmdec_thr = 5.
g_mag_thr = 10.5
parsec_thr = 1000.
Simbad.add_votable_fields('otype', 'bibcodelist(1900-2017)')
txt_file = open('possbile_star_ids.txt', 'w')
output_cols = ['tycho2_id', 'ra', 'dec', 'parallax', 'parallax_error', 'pmra_stream', 'pmra', 'pmra_error', 'pmdec_stream', 'pmdec', 'pmdec_error', 'rv_stream', 'phot_g_mean_mag']
tgas_results = Table(names=output_cols,
                     dtype=['S11', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8'])

for tgas_fits in tgas_fits_files:
    print 'Working on Tgas file: '+tgas_fits.split('/')[1]
    # read file
    tgas_data = Table.read(tgas_fits)
    # remove problems with masks
    tgas_data = tgas_data.filled()
    print ' Number of stars in dataset: '+str(len(tgas_data))
    # convert to parsec distance
    star_parsec = (tgas_data['parallax'].data * u.mas).to(u.parsec, equivalencies=u.parallax()) # use of .data to remove units as they are not handled corectlly by astropy
    tgas_data.add_column(Column(star_parsec, name='parsec'))
    # compute predicted stream pmra and pmdec, based on stars ra, dec and parsec distance
    rv_stream_predicted = compute_rv(np.deg2rad(tgas_data['ra']),
                                     np.deg2rad(tgas_data['dec']),
                                     v_xyz_stream)
    pmra_stream_predicted = compute_pmra(np.deg2rad(tgas_data['ra']),
                                         np.deg2rad(tgas_data['dec']),
                                         tgas_data['parsec'],
                                         v_xyz_stream)
    pmdec_stream_predicted = compute_pmdec(np.deg2rad(tgas_data['ra']),
                                           np.deg2rad(tgas_data['dec']),
                                           tgas_data['parsec'],
                                           v_xyz_stream)
    tgas_data.add_column(Column(rv_stream_predicted, name='rv_stream'))
    tgas_data.add_column(Column(pmra_stream_predicted, name='pmra_stream'))
    tgas_data.add_column(Column(pmdec_stream_predicted, name='pmdec_stream'))
    # filter data based on predefined search criteria
    idx_possible = np.logical_and(np.logical_and(np.abs(pmra_stream_predicted - tgas_data['pmra']) <= pmra_thr,
                                                 np.abs(pmdec_stream_predicted - tgas_data['pmdec']) <= pmdec_thr),
                                  np.logical_and(tgas_data['parsec'].data <= parsec_thr,
                                                 tgas_data['parsec'].data > 0.))
    idx_possible = np.logical_and(,
                                  idx_possible)
    # idx_possible = np.logical_and(np.abs(rv_stream_predicted - tgas_data['rv_guess']) <= rv_thr,
    #                               idx_possible)
    n_possible = np.sum(idx_possible)
    print ' Possible members: '+str(n_possible)
    if n_possible == 0:
        continue

    search_radi = 5 * u.arcsec
    for tgas_star in tgas_data[output_cols][idx_possible]:
        tgas_results.add_row(tgas_star)
        if tgas_star['tycho2_id'][0] != ' ':
            txt_file.write('TYC '+tgas_star['tycho2_id']+'\n')
        # star_pos = coord.SkyCoord(ra=tgas_star['ra']*u.deg, dec=tgas_star['dec']*u.deg, frame='icrs')
        # Simbad.query_region(star_pos, radius=search_radi)
        # q_res = Simbad.query_object('TYC '+tgas_star['tycho2_id'])
        # if q_res is None:
        #     continue
        # if (q_res['OTYPE'] == '**').any():
        #     print tgas_star[0]
txt_file.close()
tgas_results.write('possible_stars.csv', format='ascii.csv')
