# import matplotlib as mpl
# mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from joblib import Parallel, delayed
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path


class PointSelector:
    def __init__(self, ax):
        self.n_selection = 1
        self.lasso = LassoSelector(ax, self.determine_points)

    def determine_points(self, vert):
        self.vertices = vert
        if len(self.vertices) > 0:
            self.path = Path(self.vertices)
            # determine objects in region
            print 'Determining objects in selected region'
            temp = [a_row['source_id'] for a_row in proj_data if self.path.contains_point((a_row['tsne_axis1'], a_row['tsne_axis2']))]
            self.n_selected = len(temp)
            if self.n_selection > 0:
                print ','.join([str(s) for s in temp])
                # idx_ok = np.in1d(proj_data['source_id'], temp)
                # idx_bad = np.logical_and(~idx_ok, np.isfinite(proj_data[self.y]))
                # print proj_data[idx_bad]['source_id']
            else:
                print 'Number of points in region is too small'
        else:
            print 'Number of vertices in selection is too small'


# data_dir = '/data4/cotar/'
# gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')

proj_dir = '/shared/data-camelot/cotar/tSNE_Gaia_orbits_r10_allparams_clustercenters/'
proj_data = Table.read(proj_dir+'gaia_tsne_embeded_ra56.6_dec24.1_rad10_p50_t05_all.fits')

fig, ax = plt.subplots(1, 1)
ax.scatter(proj_data['tsne_axis1'], proj_data['tsne_axis2'], c='black', s=2, lw=0)
selector = PointSelector(ax)
plt.show()
plt.close()
