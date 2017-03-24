import matplotlib.pyplot as plt
from velocity_transformations import *

# --------------------------------------------------------
# ---------------- Constants -----------------------------
# --------------------------------------------------------
QUIVER_SCALE = 200.
QUIVER_WIDTH = 0.001


def plot_theoretical_motion(v_xyz_stream, img_prefix='', dist=1000):
    """

    :param v_xyz_stream:
    :param img_prefix:
    :param dist:
    :return:
    """
    # get theoretical observed rv pmra pmdec, based on streams rv values
    ra_range = np.deg2rad(np.arange(0, 360, 0.5))
    for dec_deg in np.arange(-20., 90., 10.):
        plt.plot(ra_range, compute_pmra(ra_range, np.deg2rad(dec_deg), dist, v_xyz_stream))
    plt.savefig(img_prefix+'_pmra.png')
    plt.close()
    for dec_deg in np.arange(-20., 90., 10.):
        plt.plot(ra_range, compute_pmdec(ra_range, np.deg2rad(dec_deg), dist, v_xyz_stream))
    plt.savefig(img_prefix+'_pmdec.png')
    plt.close()
    for dec_deg in np.arange(-20., 90., 10.):
        plt.plot(ra_range, compute_rv(ra_range, np.deg2rad(dec_deg), v_xyz_stream))
    plt.savefig(img_prefix+'_rv.png')
    plt.close()


def plot_members_location_motion(gaia, pmra_pred=None, pmdec_pred=None, idx=None, radiant=None,
                                 path='members.png', title=''):
    """

    :param gaia:
    :param pmra_pred:
    :param pmdec_pred:
    :param idx:
    :param radiant:
    :param path:
    :param title:
    :return:
    """
    if idx is None:
        # use all datarows
        idx = np.ndarray(len(gaia))
        idx.fill(True)
    use_gaia_data = gaia[idx]
    # plot location of the stars
    if radiant is not None:
        plt.scatter(radiant[0], radiant[1], lw=0, s=15, c='black', marker='*')
    plt.scatter(use_gaia_data['ra_gaia'], use_gaia_data['dec_gaia'], lw=0, c='black', s=1)
    gaia_features = gaia.colnames
    if 'pmra' in gaia_features and 'pmdec' in gaia_features:
        plt.quiver(use_gaia_data['ra_gaia'], use_gaia_data['dec_gaia'], use_gaia_data['pmra'], use_gaia_data['pmdec'],
                   pivot='tail', scale=QUIVER_SCALE, color='green', width=QUIVER_WIDTH)
    if pmra_pred is not None and pmdec_pred is not None:
        plt.quiver(use_gaia_data['ra_gaia'], use_gaia_data['dec_gaia'], pmra_pred[idx], pmdec_pred[idx],
                   pivot='tail', scale=QUIVER_SCALE, color='red', width=QUIVER_WIDTH)
    # annotate graph
    plt.xlabel('RA [deg]')
    plt.ylabel('DEC [deg]')
    plt.title(title)
    plt.xlim((0, 360))
    plt.ylim((-90, 90))
    # save graph
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_members_location_motion_theoretical(ra, dec, pmra_pred, pmdec_pred, radiant=None,
                                             path='members.png', title=''):
    """

    :param ra:
    :param dec:
    :param pmra_pred:
    :param pmdec_pred:
    :param radiant:
    :param path:
    :param title:
    :return:
    """
    if radiant is not None:
        plt.scatter(radiant[0], radiant[1], lw=0, s=15, c='black', marker='*')
    plt.scatter(ra, dec, lw=0, c='black', s=1)
    plt.quiver(ra, dec, pmra_pred, pmdec_pred,
               pivot='tail', scale=QUIVER_SCALE, color='green', width=QUIVER_WIDTH)
    # annotate graph
    plt.xlabel('RA [deg]')
    plt.ylabel('DEC [deg]')
    plt.title(title)
    plt.xlim((0, 360))
    plt.ylim((-90, 90))
    # save graph
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def plot_members_location_velocity(gaia, rv=None, idx=None, radiant=None,
                                   path='members.png', title=''):
    """

    :param gaia:
    :param idx:
    :param radiant:
    :param path:
    :param title:
    :return:
    """
    if idx is None:
        # use all datarows
        idx = np.ndarray(len(gaia))
        idx.fill(True)
    use_gaia_data = gaia[idx]
    # plot location of the stars
    if radiant is not None:
        plt.scatter(radiant[0], radiant[1], lw=0, s=15, c='black', marker='*')
    plt.scatter(use_gaia_data['ra_gaia'], use_gaia_data['dec_gaia'], lw=0, c='black', s=1)
    gaia_features = gaia.colnames
    if rv is None and 'RV' in gaia_features:
        rv_plot = use_gaia_data['RV']
    elif rv is not None:
        rv_plot = rv[idx]
    if 'rv_plot' in locals():
        plt.quiver(use_gaia_data['ra_gaia'], use_gaia_data['dec_gaia'], rv_plot, 0.,
                   pivot='tail', scale=QUIVER_SCALE, color='green', width=QUIVER_WIDTH)
    # annotate graph
    plt.xlabel('RA [deg]')
    plt.ylabel('DEC [deg]')
    plt.title(title)
    plt.xlim((0, 360))
    plt.ylim((-90, 90))
    # save graph
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


