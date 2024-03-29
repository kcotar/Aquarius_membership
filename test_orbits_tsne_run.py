import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import matplotlib.pyplot as plt
from time import time
from astropy.table import Table, vstack
from MulticoreTSNE import MulticoreTSNE as TSNE_multi
import umap
from scipy.stats import circmean
from os.path import isfile
from os import system, chdir
from glob import glob

data_dir = '/shared/ebla/cotar/'
in_dir = '/shared/data-camelot/cotar/'

print 'Reading derived Gaia DR2 orbital parameters'
gaia_data = list([])
for fits_file in glob(in_dir + 'Gaia_dr2_orbital_derivatives_actions_Bdist_*.fits'):
    print '  -', fits_file
    gaia_data.append(Table.read(fits_file))
gaia_data = vstack(gaia_data)
print 'Number of data point:', len(gaia_data)
print 'Parameters in data:', gaia_data.colnames

# galah_gaia_data = Table.read(data_dir+'')
clusters = Table.read(data_dir+'Open_cluster_members_DR2.fits')
clusters = clusters[clusters['proba'] > 0.75]

gaia_data = gaia_data[np.isfinite(gaia_data.to_pandas().values).all(axis=1)]
print 'Rows left:', len(gaia_data)

use_tsne_cols = ['X', 'Z', 'VX', 'VY', 'VZ', 'e', 'zmax', 'Rper', 'J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z',  'Theta_R', 'Theta_Phi', 'Theta_Z']
# use_tsne_cols = ['J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z']
use_tsne_cols_plot = list(np.hstack((use_tsne_cols, ['ra', 'dec'])))
data_transform = gaia_data[use_tsne_cols].to_pandas().values

radius_deg = 10. * un.deg
i_r_pos = 30
ra_rand = np.random.uniform(0, 360, i_r_pos)
dec_rand = np.random.uniform(-75, 75, i_r_pos)

# replace random field centers with cluster centers
ra_rand = []
dec_rand = []
u_cluster, in_cluster = np.unique(clusters['cluster'], return_counts=True)
for cl in u_cluster[in_cluster >= 100]:
    idx = clusters['cluster'] == cl
    idx_in = np.in1d(gaia_data['source_id'], clusters['source_id'][idx])
    if np.sum(idx_in) >= 20:
        ra_rand.append(circmean(clusters[idx]['ra'], 360., 0.))
        dec_rand.append(np.mean(clusters[idx]['dec']))

# ra_rand = [82.]
# dec_rand = [2.]
# sra_rand = [0.]
# dec_rand = [0.]

coord_all = coord.ICRS(ra=gaia_data['ra']*un.deg, dec=gaia_data['dec']*un.deg)

output_subdir = in_dir+'tSNE_Gaia_orbits_r10_allparams_clustercenters_wtheta'
system('mkdir '+output_subdir)
chdir(output_subdir)

for i_r in range(len(ra_rand)):
    coord_center = coord.ICRS(ra=ra_rand[i_r]*un.deg, dec=dec_rand[i_r]*un.deg)
    idx_use = coord_all.separation(coord_center) < radius_deg
    suffix_coord = '_ra{:03.1f}_dec{:02.1f}_rad{:02.0f}__'.format(ra_rand[i_r], dec_rand[i_r], radius_deg.value)
    print suffix_coord, np.sum(idx_use)
    if np.sum(idx_use) < 1:
        continue

    # print gaia_data[idx_use]
    gaia_data_use = gaia_data[idx_use]
    data_transform_use = data_transform[idx_use]

    for i_c in range(data_transform.shape[1]):
        # plot the trainnig set and its statistics
        # plt.hist(data_transform_use[:, i_c], bins=75, range=np.nanpercentile(data_transform_use[:, i_c], [1, 99]))
        # plt.axvline(np.nanpercentile(data_transform_use[:, i_c], 2), lw=1, ls='--', c='black')
        # plt.axvline(np.nanpercentile(data_transform_use[:, i_c], 98), lw=1, ls='--', c='black')
        # plt.axvline(np.nanmedian(data_transform_use[:, i_c]), lw=1, ls='--', c='blue')
        # plt.axvline(np.nanmedian(data_transform_use[:, i_c]) + np.nanstd(data_transform_use[:, i_c]), lw=1, ls='--', c='blue')
        # plt.axvline(np.nanmedian(data_transform_use[:, i_c]) - np.nanstd(data_transform_use[:, i_c]), lw=1, ls='--', c='blue')
        # plt.savefig('orbit_train_set'+suffix_coord+'_' + use_tsne_cols[i_c] + '.png', dpi=200)
        # plt.close()
        # standardization to std=1, mean=0
        # data_transform_use[:, i_c] = (data_transform_use[:, i_c] - np.nanmedian(data_transform_use[:, i_c])) / np.nanstd(data_transform_use[:, i_c])
        # normalization to range -1 ... 1
        c_lim = np.nanpercentile(data_transform_use[:, i_c], [2, 98])
        data_transform_use[:, i_c] = (data_transform_use[:, i_c] - c_lim[0]) / (c_lim[1] - c_lim[0]) * 2. - 1.

    # -----------------------------
    #             TSNE
    # -----------------------------
    suffix = suffix_coord+'_p50_t05_all'
    fits_out = 'gaia_tsne_embeded'+suffix+'.fits'
    if not isfile(fits_out):
        tsne_class = TSNE_multi(n_components=2, perplexity=50, n_iter=1100, n_iter_without_progress=350, init='random',
                                verbose=1, method='barnes_hut', angle=0.5, n_jobs=32)
        tsne_res = tsne_class.fit_transform(data_transform_use)

        gaia_data_use['tsne_axis1'] = tsne_res[:, 0]
        gaia_data_use['tsne_axis2'] = tsne_res[:, 1]
        gaia_data_use['source_id', 'tsne_axis1', 'tsne_axis2'].write(fits_out, overwrite=True)
    else:
        tsne_res = Table.read(fits_out)['tsne_axis1', 'tsne_axis2'].to_pandas().values
        gaia_data_use['tsne_axis1'] = tsne_res[:, 0]
        gaia_data_use['tsne_axis2'] = tsne_res[:, 1]

    # plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, s=2, c='black')    
    # plt.show()
    # plt.close()
    for c in use_tsne_cols_plot:#gaia_data.colnames:
        if 'error' in c:
            continue
        plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, s=0.5, c=gaia_data_use[c],
                    vmin=np.nanpercentile(gaia_data_use[c], 1), vmax=np.nanpercentile(gaia_data_use[c], 99))
        plt.colorbar()
        # plt.show()
        plt.savefig('tsne_orbits'+suffix+'_'+c+'.png', dpi=250)
        plt.close()

    for cl in np.unique(clusters['cluster']):
        cl_s = clusters[clusters['cluster'] == cl]['source_id']
        idx_mark = np.in1d(gaia_data_use['source_id'], cl_s)
        if np.sum(idx_mark) >= 10:
            plt.scatter(tsne_res[:, 0], tsne_res[:, 1], lw=0, s=1, c='black')
            plt.scatter(tsne_res[:, 0][idx_mark], tsne_res[:, 1][idx_mark], lw=0, s=1, c='red')
            plt.savefig('tsne_orbits'+suffix+'_'+cl+'.png', dpi=300)
            plt.close()


# -----------------------------
#             UMAP
# -----------------------------
# suffix = '_stand'
# neighb = 20
# dist = 0.1
# spread = 1.0
# metric = 'manhattan'
# umap_embed = umap.UMAP(n_neighbors=neighb,
#                        min_dist=dist,
#                        spread=spread,
#                        metric=metric,
#                        init='spectral',  # spectral or random
#                        local_connectivity=1,
#                        set_op_mix_ratio=1.,  # 0. - 1.
#                        n_components=2,
#                        transform_seed=42,
#                        n_epochs=1000,
#                        verbose=True).fit_transform(data_transform_use)
#
# plt.scatter(umap_embed[:, 0], umap_embed[:, 1], lw=0, s=2, c='black')
# plt.show()
# plt.close()
# for c in gaia_data.colnames:
#     if 'error' in c:
#         continue
#     plt.scatter(umap_embed[:, 0], umap_embed[:, 1], lw=0, s=1, c=gaia_data_use[c],
#                 vmin=np.nanpercentile(gaia_data_use[c], 1), vmax=np.nanpercentile(gaia_data_use[c], 99))
#     plt.colorbar()
#     # plt.show()
#     plt.savefig('umap_orbits_'+c+''+suffix+'.png', dpi=300)
#     plt.close()
#
# gaia_data_use['umap_axis1'] = umap_embed[:, 0]
# gaia_data_use['umap_axis2'] = umap_embed[:, 1]
# gaia_data_use['source_id', 'umap_axis1', 'umap_axis2'].write('gaia_umap_embeded'+suffix+'.fits', overwrite=True)
