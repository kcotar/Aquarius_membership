import imp
import astropy.units as un
import astropy.coordinates as coord
import matplotlib.pyplot as plt
import gala.coordinates as gal_coord

from astropy.table import Table

from vector_plane_calculations import *
from velocity_transformations import *

imp.load_source('helper', '../tSNE_test/helper_functions.py')
from helper import move_to_dir
imp.load_source('gal_move', '../tSNE_test/convert_gal_movement.py')
from gal_move import gal_uvw
imp.load_source('veltrans', '../tSNE_test/velocity_transform.py')
from veltrans import *



# --------------------------------------------------------
# ---------------- FUNCTIONS -----------------------------
# --------------------------------------------------------
def _prepare_hist_data(d, bins, range, norm=True):
    heights, edges = np.histogram(d, bins=bins, range=range)
    width = np.abs(edges[0] - edges[1])
    if norm:
        heights = 1.*heights / np.max(heights)
    return edges[:-1], heights, width


def _get_range(data, perc_cut=2.):
    return (np.nanpercentile(data, perc_cut), np.nanpercentile(data, 100-perc_cut))
    # return (np.nanmin(data), np.nanmax(data))


def plot_hist(obs, obs_f, galx, galx_f, path=None, title='', hist_bins = 100):
    hist_range = _get_range(obs[obs_f])
    # zgal_range = _get_range(galaxia_sub['pz'])
    plt.title(title)
    h_edg, h_hei, h_wid = _prepare_hist_data(obs[obs_f], hist_bins, hist_range, norm=True)
    plt.bar(h_edg, h_hei, width=h_wid, color='green', alpha=0.2)
    h_edg, h_hei, h_wid = _prepare_hist_data(galx[galx_f], hist_bins, hist_range, norm=True)
    plt.bar(h_edg, h_hei, width=h_wid, color='blue', alpha=0.2)
    plt.show()
    plt.close()

# --------------------------------------------------------
# ---------------- CONSTANTS AND SETTINGS ----------------
# --------------------------------------------------------
# GALAH
# simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/GALAH/'
# simulation_ebf = 'galaxy_galah_complete.ebf'
# simulation_ebf = 'galaxy_galah_fields.ebf'
# RAVE
simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/RAVE/'
simulation_ebf = 'galaxy_rave_complete.ebf'

simulation_fits = simulation_ebf.split('.')[0]+'.fits'
obs_file_fits = 'RAVE_GALAH_TGAS_stack.fits'

# analysis constants
l_center = 310.
b_center = -70.
r_center = 10.

# --------------------------------------------------------
# ---------------- INPUT DATA HANDLING -------------------
# --------------------------------------------------------
print 'Reading data'
glaxia_data = Table.read(simulation_dir + simulation_fits)
obs_data = Table.read(obs_file_fits)
obs_data = obs_data.filled()

# compute observation galactic coordinates
l_b_obs = coord.ICRS(ra=obs_data['ra_gaia']*un.deg, dec=obs_data['dec_gaia']*un.deg).transform_to(coord.Galactic)
obs_data['l'] = l_b_obs.l.value
obs_data['b'] = l_b_obs.b.value

# create a subset of data
lb_center = coord.Galactic(l=l_center*un.deg, b=b_center*un.deg)
xyz_vel_stream = compute_xyz_vel(np.deg2rad(lb_center.l.value), np.deg2rad(lb_center.b.value), 10)

galaxia_sub = glaxia_data[coord.Galactic(l=glaxia_data['glon']*un.deg, b=glaxia_data['glat']*un.deg).separation(lb_center) < r_center*un.deg]
obs_sub = obs_data[coord.Galactic(l=obs_data['l']*un.deg, b=obs_data['b']*un.deg).separation(lb_center) < r_center*un.deg]

print 'Galaxia stars: '+str(len(galaxia_sub))
print 'Observation stars: '+str(len(obs_sub))

galaxia_sub['px'] *= 1e3  # kpc to pc conversion
galaxia_sub['py'] *= 1e3
galaxia_sub['pz'] *= 1e3
# galaxia_sub['vx'] *= -1.  # it has different orientation than our coordinate system

# compute galactic velocities and positions for the obs stars
obs_gal_coord = coord.Galactic(l=obs_sub['l']*un.deg, b=obs_sub['b']*un.deg, distance=1e3/obs_sub['parallax'].data*un.pc)
obs_gal_xyz = obs_gal_coord.cartesian

obs_sub['x_gal'] = obs_gal_xyz.x.value
obs_sub['y_gal'] = obs_gal_xyz.y.value
obs_sub['z_gal'] = obs_gal_xyz.z.value

plot_hist(obs_sub, 'x_gal', galaxia_sub, 'px', path=None, title='')
plot_hist(obs_sub, 'y_gal', galaxia_sub, 'py', path=None, title='')
plot_hist(obs_sub, 'z_gal', galaxia_sub, 'pz', path=None, title='')

# convert velocities from ra/de/pmra/pmdec to more consisten units
u_gal, v_gal, w_gal = gal_uvw(obs_sub['ra_gaia'], obs_sub['dec_gaia'], obs_sub['pmra'], obs_sub['pmdec'], obs_sub['RV'],
                              plx=obs_sub['parallax'])
obs_sub['u_gal'] = u_gal * -1.
obs_sub['v_gal'] = v_gal
obs_sub['w_gal'] = w_gal

ra_dec_pm = np.vstack((obs_sub['pmra'], obs_sub['pmdec'])) * un.mas/un.yr
l_b_pm = gal_coord.pm_icrs_to_gal(coord.ICRS(ra=obs_sub['ra_gaia']*un.deg, dec=obs_sub['dec_gaia']*un.deg), ra_dec_pm)
obs_sub['pml'] = l_b_pm[0].value
obs_sub['pmb'] = l_b_pm[1].value
xyz_vel = motion_to_cartesic(np.array(obs_sub['l']), np.array(obs_sub['b']),
                             np.array(obs_sub['pml']), np.array(obs_sub['pmb']),
                             np.array(obs_sub['RV']), plx=np.array(obs_sub['parallax']))
obs_sub['vx_gal'] = xyz_vel[0]
obs_sub['vy_gal'] = xyz_vel[1]
obs_sub['vz_gal'] = xyz_vel[2]

# plot_hist(obs_sub, 'u_gal', obs_sub, 'vx_gal', path=None, title='')
# plot_hist(obs_sub, 'v_gal', obs_sub, 'vy_gal', path=None, title='')
# plot_hist(obs_sub, 'w_gal', obs_sub, 'vz_gal', path=None, title='')

plot_hist(obs_sub, 'u_gal', galaxia_sub, 'vx', path=None, title='')
plot_hist(obs_sub, 'v_gal', galaxia_sub, 'vy', path=None, title='')
plot_hist(obs_sub, 'w_gal', galaxia_sub, 'vz', path=None, title='')

xyz_pos_stars = np.vstack((obs_sub['x_gal'],obs_sub['y_gal'],obs_sub['z_gal'])).T
xyz_vel_stars = np.vstack((obs_sub['u_gal'],obs_sub['v_gal'],obs_sub['w_gal'])).T
print xyz_pos_stars
print xyz_vel_stars
print xyz_vel_stream
obs_plane_intersects_3D = stream_plane_vector_intersect(xyz_pos_stars, xyz_vel_stars, xyz_vel_stream)
obs_plane_intersects_2D = intersects_to_2dplane(obs_plane_intersects_3D, xyz_vel_stream)

xyz_pos_stars = np.vstack((galaxia_sub['px'],galaxia_sub['py'],galaxia_sub['pz'])).T
xyz_vel_stars = np.vstack((galaxia_sub['vx'],galaxia_sub['vy'],galaxia_sub['vz'])).T
galaxia_plane_intersects_3D = stream_plane_vector_intersect(xyz_pos_stars, xyz_vel_stars, xyz_vel_stream)
galaxia_plane_intersects_2D = intersects_to_2dplane(galaxia_plane_intersects_3D, xyz_vel_stream)

plot_lim = (-1000, 1000)
# Create a plot
fig, ax = plt.subplots(1, 1)
ax.scatter(obs_plane_intersects_2D[:, 0], obs_plane_intersects_2D[:, 1], lw=0, c='red', s=2, alpha=1.)
ax.scatter(galaxia_plane_intersects_2D[:, 0], galaxia_plane_intersects_2D[:, 1], lw=0, c='blue', s=2, alpha=1.)
ax.scatter(0, 0, lw=0, c='black', s=10, marker='*')  # solar position
ax.set(xlabel='X stream plane', ylabel='Y stream plane', xlim=plot_lim, ylim=plot_lim)
fig.tight_layout()
plt.show()
plt.close()


