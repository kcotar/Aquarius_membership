import imp, os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir

imp.load_source('tsne', '../tSNE_test/tsne_functions.py')
from tsne import *

imp.load_source('norm', '../Stellar_parameters_interpolator/data_normalization.py')
from norm import *

action_angle_data = Table.read('RAVE_GALAH_TGAS_stack_old_actions.fits')

norm = False
perp = 40
theta = 0.5
suffix = '_perp_{:02.0f}_theta_{:01.1f}'.format(perp, theta)
if norm:
    suffix += '_norm'

move_to_dir('tSNE_test_actions')

# run t-sne on newly computed uvw velocities
tsne_data = action_angle_data['J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z'].to_pandas().values
# so last minute fast data filtering
print 'All rows: '+str(tsne_data.shape[0])
idx_ok = np.isfinite(tsne_data).all(axis=1)
idx_use = np.logical_and(idx_ok,
                         (np.abs(tsne_data) < 1000.).all(axis=1))
tsne_data = tsne_data[idx_use, :]
print 'Use rows: '+str(tsne_data.shape[0])

# normalization
if norm:
    norm_param = normalize_data(tsne_data, method='standardize')

# run t-sne
tsne_result = bh_tsne(tsne_data, no_dims=2, perplexity=perp, theta=theta, randseed=23, verbose=True,
                      distance='manhattan', path='/home/klemen/tSNE_test/')
tsne_ax1, tsne_ax2 = tsne_results_to_columns(tsne_result)

# plot
plt.scatter(tsne_ax1, tsne_ax2, s=0.5, c='black')
plt.savefig('actions'+suffix+'.png', dpi=350)
plt.close()