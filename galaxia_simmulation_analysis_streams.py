import imp, os
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import numpy as np
import ebf

from astropy.table import Table, vstack, Column
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from skimage.feature import peak_local_max
from scipy.ndimage import watershed_ift
from skimage.morphology import watershed

from vector_plane_calculations import *
from velocity_transformations import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir


# GALAH
# simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/GALAH/'
# simulation_ebf = 'galaxy_galah_complete.ebf'
# simulation_ebf = 'galaxy_galah_fields.ebf'
# RAVE
simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/RAVE/'
# simulation_ebf = 'galaxy_rave_complete.ebf'
simulation_ebf = 'galaxy_rave_fields.ebf'
# out fits
simulation_fits = simulation_ebf.split('.')[0]+'.fits'
output_dir = ''

# --------------------------------------------------------
# ---------------- FUNCTIONS -----------------------------
# --------------------------------------------------------


# --------------------------------------------------------
# ---------------- CONSTANTS AND SETTINGS ----------------
# --------------------------------------------------------
xyz_vel_neighbourhood = 10  # radius km/s

# --------------------------------------------------------
# ---------------- INPUT DATA HANDLING -------------------
# --------------------------------------------------------
if os.path.isfile(simulation_dir+simulation_fits):
    # read data from reduced fits file
    print 'Reading fits file'
    stars_data = Table.read(simulation_dir+simulation_fits)
else:
    # read original ebf file and reduce data
    get_cols = ['px', 'py', 'pz',  # Position (x,y,z) heliocentric in kpc (galactic coordinate system)
                'vx', 'vy', 'vz',  # Velocity (U,V,W) heliocentric in km/s ??????? (galactic coordinate system)
                'glon', 'glat']#,  # galacitic longitude and latitude in degrees
                #'feh', 'teff', 'grav']  # metallicity, effective temperature, surface gravity
    print 'Reading ebf file'
    sim_data = ebf.read(simulation_dir+simulation_ebf)
    print 'Creating fits file'
    stars_data = Table()
    for col in get_cols:
        stars_data[col] = sim_data[col]
    sim_data = None
    stars_data.write(simulation_dir+simulation_fits)

#ra_coord = coord.Galactic(l=stars_data['glon']*un.deg, b=stars_data['glat']*un.deg).transform_to(coord.ICRS)
plt.scatter(stars_data['glon'], stars_data['glat'], s=1, color='black')
# plt.scatter(ra_coord.ra.value, ra_coord.dec.value, s=1, color='black')
plt.show()
plt.close()
raise SystemExit

# --------------------------------------------------------
# ---------------- Stream search parameters --------------
# --------------------------------------------------------
# stream search criteria
rv_step = 10.  # km/s, rv in the radiant of the stream
ra_step = 20.  # deg
dec_step = 10.  # deg

# --------------------------------------------------------
# ---------------- Evaluation of possible streams --------
# --------------------------------------------------------
manual_stream_radiants = [[20,45,140,240,370,125,20,150], [-10,-30,20,10,50,35,-80,-60], [20,15,35,70,45,55,22,10], [None]]  # list of ra, dec, rv values
manual_stream_radiants = [[90], [0], [45], [None]]  # list of ra, dec, rv values
# manual_stream_radiants = parse_selected_streams('Streams_investigation_lower-thr_selected')
# iterate trough all possible combinations for the initial conditions of the stream (rv, ra, dec)
if manual_stream_radiants is not None:
    ra_combinations = manual_stream_radiants[0]
    dec_combinations = manual_stream_radiants[1]
    rv_combinations = manual_stream_radiants[2]
else:
    rv_range = np.arange(30, 31, rv_step)
    ra_range = np.arange(0, 360, ra_step)
    dec_range = np.arange(-90, 90, dec_step)
    # create a grid of all possible combination
    stream_mesh = np.meshgrid(ra_range, dec_range, rv_range)
    ra_combinations = stream_mesh[0].flatten()
    dec_combinations = stream_mesh[1].flatten()
    rv_combinations = stream_mesh[2].flatten()
n_combinations = len(ra_combinations)
print 'Total number of stream combinations that will be evaluated: '+str(n_combinations)

# # transform galactic uvw coordinates to equatorial xyz coordinates
# coords_new = coord.SkyCoord(u=stars_data['px'], v=stars_data['py'], w=stars_data['pz'], unit='kpc',
#                             frame='galactic', representation='cartesian').transform_to(coord.ICRS).cartesian
# veloci_new = coord.SkyCoord(u=stars_data['vx'], v=stars_data['vy'], w=stars_data['vz'], unit='km',
#                             frame='galactic', representation='cartesian').transform_to(coord.ICRS).cartesian
#
# stars_data['px'] = coords_new.x.value
# stars_data['py'] = coords_new.y.value
# stars_data['pz'] = coords_new.z.value
# stars_data['vx'] = veloci_new.x.value
# stars_data['vy'] = veloci_new.y.value
# stars_data['vz'] = veloci_new.z.value

move_to_dir('Streams_investigation_'+simulation_ebf.split('.')[0])
for i_stream in range(n_combinations):
    ra_stream = ra_combinations[i_stream]
    dec_stream = dec_combinations[i_stream]
    rv_stream = rv_combinations[i_stream]

    suffix = 'stream_ra_{:05.1f}_dec_{:04.1f}_rv_{:05.1f}'.format(ra_stream, dec_stream, rv_stream)
    print 'Working on ' + suffix

    # convert radiant coordinate from ra/dec/rv to l/b/rv system as Galaxia coordinates are in Galactic system
    l_b_stream = coord.ICRS(ra=ra_stream*un.deg, dec=dec_stream*un.deg).transform_to(coord.Galactic)
    l_stream = l_b_stream.l.value
    b_stream = l_b_stream.b.value

    # velocity vector of stream in xyz equatorial coordinate system with Earth in the center of it
    # xyz_vel_stream = compute_xyz_vel(np.deg2rad(ra_stream), np.deg2rad(dec_stream), rv_stream)
    xyz_vel_stream = compute_xyz_vel(np.deg2rad(l_stream), np.deg2rad(b_stream), rv_stream)

    # select objects from simulation with similar velocity components
    vel_diff = np.sqrt((stars_data['vx'] - xyz_vel_stream[0])**2 +
                       (stars_data['vy'] - xyz_vel_stream[1])**2 +
                       (stars_data['vz'] - xyz_vel_stream[2])**2)
    idx_close = vel_diff < xyz_vel_neighbourhood

    print 'Selected objects: '+str(np.sum(idx_close))
    stars_data_subset = stars_data[idx_close]

    xyz_pos_stars = np.vstack((stars_data_subset['px'], stars_data_subset['py'], stars_data_subset['pz'])).T * 1000.  # conversion from kpc to pc
    xyz_vel_stars = np.vstack((stars_data_subset['vx'], stars_data_subset['vy'], stars_data_subset['vz'])).T

    # plot selection
    print ' Outputting xyz velocities scatter plot'
    plot_range = 10
    labels = ['X', 'Y', 'Z']
    plot_comb = [[0, 1], [2, 1], [0, 2]]
    plot_pos = [[0, 0], [0, 1], [1, 0]]
    fig, ax = plt.subplots(2, 2)
    for i_c in range(len(plot_comb)):
        fig_pos = (plot_pos[i_c][0], plot_pos[i_c][1])
        i_x = plot_comb[i_c][0]
        i_y = plot_comb[i_c][1]
        alpha_use = 0.1
        ax[fig_pos].scatter(xyz_vel_stream[i_x], xyz_vel_stream[i_y], lw=0, c='black', s=10, marker='*')
        ax[fig_pos].scatter(xyz_vel_stars[:, i_x], xyz_vel_stars[:, i_y], lw=0, c='blue', s=2, alpha=alpha_use)
        ax[fig_pos].set(xlabel=labels[i_x], ylabel=labels[i_y],
                        xlim=[xyz_vel_stream[i_x] - plot_range, xyz_vel_stream[i_x] + plot_range],
                        ylim=[xyz_vel_stream[i_y] - plot_range, xyz_vel_stream[i_y] + plot_range])
    plt.savefig(suffix+'_1.png', dpi=300)
    plt.close()

    # compute intersection between star vectors and plane defined by the stream vector
    print ' Computing intersections'
    plane_intersects_3D = stream_plane_vector_intersect(xyz_pos_stars, xyz_vel_stars, xyz_vel_stream)
    plane_intersects_2D = intersects_to_2dplane(plane_intersects_3D, xyz_vel_stream)

    print ' Outputting plane intersections plot'
    plot_lim = (-1000, 1000)
    # Create a plot
    fig, ax = plt.subplots(1, 1)
    ax.scatter(plane_intersects_2D[:, 0], plane_intersects_2D[:, 1], lw=0, c='blue', s=2, alpha=1.)
    ax.scatter(0, 0, lw=0, c='black', s=10, marker='*')  # solar position
    ax.set(xlabel='X stream plane', ylabel='Y stream plane', xlim=plot_lim, ylim=plot_lim)
    fig.tight_layout()
    plt.savefig(suffix + '_2.png', dpi=300)
    plt.close()

    stars_density = KernelDensity(bandwidth=30, kernel='epanechnikov').fit(plane_intersects_2D)
    grid_pos = np.linspace(-1000, 1000, 2000)
    _x, _y = np.meshgrid(grid_pos, grid_pos)
    print 'Computing density field'
    density_field = stars_density.score_samples(np.vstack((_x.ravel(), _y.ravel())).T) + np.log(plane_intersects_2D.shape[0])
    density_field = np.exp(density_field).reshape(_x.shape) * 1e3

    fig, ax = plt.subplots(1, 1)
    im_ax = ax.imshow(density_field, interpolation=None, cmap='seismic',
                      origin='lower', vmin=0.)  # , vmax=4.)
    fig.colorbar(im_ax)
    ax.set_axis_off()
    fig.tight_layout()
    # plt.savefig(suffix + '_3.png', dpi=250)
    plt.show()
    plt.close()

    heights, edges = np.histogram(density_field, bins=100, range=(1e-5, np.percentile(density_field,98)))
    width = np.abs(edges[0] - edges[1])
    plt.bar(edges[:-1], heights, width=width, color='green', alpha=0.5)
    plt.show()
    plt.close()

