import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join

rave_data_dir = '/home/klemen/RAVE_data/'

rave_param = Table.read(rave_data_dir+'RAVE_DR5.fits')
rave_tgas = Table.read(rave_data_dir+'RAVE_TGAS.fits')

rave_tgas = join(rave_param, rave_tgas, keys='RAVE_OBS_ID')

val_ra, num_ra = np.unique(rave_tgas['ra'], return_counts=True)

rows_out = list([])
for ra_gaia in val_ra[num_ra > 1]:
    idx_repeated = rave_tgas['ra'] == ra_gaia
    rave_tgas_rep = rave_tgas[idx_repeated]
    ids = np.unique(rave_tgas_rep['RAVEID'])
    if len(ids) > 1:
        print ids
        # print rave_tgas_rep['RAVEID', 'ra', 'dec', 'RAdeg', 'DEdeg', 'pmra', 'pmdec']
        print
        for i_r in np.where(idx_repeated)[0]:
            rows_out.append(i_r)
rave_tgas[rows_out]['RAVE_OBS_ID', 'RAVEID', 'RAdeg', 'DEdeg', 'HRV', 'tycho2_id', 'ra', 'dec', 'pmra', 'pmdec'].write('RAVE_DR5_TGAS_match_problems.fits')
