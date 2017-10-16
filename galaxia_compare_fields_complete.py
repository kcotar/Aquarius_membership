import os
from astropy.table import Table, unique
import astropy.coordinates as coord
import astropy.units as un
import numpy as np
import matplotlib.pyplot as plt

RAVE = True
GALAH = False

# GALAH
# simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/GALAH/'
# simulation_ebf = 'galaxy_galah_complete.ebf'
# simulation_ebf = 'galaxy_galah_fields.ebf'
# RAVE
simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/RAVE/'
simulation_c = 'galaxy_rave_complete.fits'
simulation_f = 'galaxy_rave_complete_fields_r3.0.fits'

l_center = 230.
b_center = 20.
d = 10.

c_data = Table.read(simulation_dir + simulation_c)
f_data = Table.read(simulation_dir + simulation_f)

# subsets
idx_c = np.logical_and(np.logical_and(c_data['glon'] > l_center - d, c_data['glon'] < l_center + d),
                       np.logical_and(c_data['glat'] > b_center - d, c_data['glat'] < b_center + d))
c_data_sub = c_data[idx_c]
idx_f = np.logical_and(np.logical_and(f_data['glon'] > l_center - d, f_data['glon'] < l_center + d),
                       np.logical_and(f_data['glat'] > b_center - d, f_data['glat'] < b_center + d))
f_data_sub = f_data[idx_f]

fig, ax = plt.subplots(1, 2)
ax[0].scatter(c_data_sub['glon'], c_data_sub['glat'], s=1, color='black')
ax[0].set(xlabel='l', ylabel='b', title='Galaxia complete')
ax[1].scatter(f_data_sub['glon'], f_data_sub['glat'], s=1, color='black')
ax[1].set(xlabel='l', ylabel='b', title='Galaxia fields')
plt.show()
plt.close()
