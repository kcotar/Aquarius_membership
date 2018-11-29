import numpy as np
from astropy.table import Table
from find_streams_analysis import *
from astropy.modeling import models, fitting
import astropy.modeling.polynomial as poly

simulation_dir = '/home/klemen/GALAH_data/Galaxia_simulation/RAVE/'
# simulation_fits = 'galaxy_rave_complete.fits'
simulation_fits = 'galaxy_rave_complete_fields_r3.0.fits'
# read
galaxia_data = Table.read(simulation_dir + simulation_fits)

xyz_stream = np.array([-25, -20, -20])
idx_close_vel = np.sqrt((galaxia_data['vx'] - xyz_stream[0]) ** 2 +
                        (galaxia_data['vy'] - xyz_stream[1]) ** 2 +
                        (galaxia_data['vz'] - xyz_stream[2]) ** 2) < 10
sim_data_subset = galaxia_data[idx_close_vel]

xyz_pos_stars = np.vstack((sim_data_subset['px'], sim_data_subset['py'], sim_data_subset['pz'])).T * 1000.
xyz_vel_stars = np.vstack((sim_data_subset['vx'], sim_data_subset['vy'], sim_data_subset['vz'])).T

plane_intersects_3D = stream_plane_vector_intersect(xyz_pos_stars, xyz_vel_stars, xyz_stream)
plane_intersects_2D = intersects_to_2dplane(plane_intersects_3D, xyz_stream)

plt.scatter(plane_intersects_2D[:, 0], plane_intersects_2D[:, 1], lw=0, c='blue', s=2, alpha=0.3)
plt.scatter(0, 0, lw=0, c='black', s=10, marker='*')  # solar position
plt.xlim((-1000,1000))
plt.ylim((-1000,1000))
plt.tight_layout()
# plt.show()
plt.close()

n_intesects_2d, pos_intersects_2d = scatter_to_2d(plane_intersects_2D[:, 0], plane_intersects_2D[:, 1],
                                                  range=[-1000, 1000], steps=200)

plt.imshow(np.rot90(n_intesects_2d, 2), cmap='Greys', origin='lower', interpolation='none')
plt.colorbar()
# plt.show()
plt.close()

# model = poly.Polynomial2D(35)
# model = poly.Chebyshev2D(21, 21)

for n_poly in range(2, 40):
    # model = poly.Legendre2D(n_poly, n_poly)
    model = poly.Chebyshev2D(n_poly, n_poly)
    fit_p = fitting.LinearLSQFitter()

    _x, _y = np.meshgrid(pos_intersects_2d, pos_intersects_2d)
    p = fit_p(model, _x, _y, n_intesects_2d)
    fited_intersections = p(_x, _y)
    res_intersections = fited_intersections - n_intesects_2d

    print 'Poly:', n_poly, 'abs residual:', np.mean(np.abs(res_intersections)), 'mean residual:', np.mean((res_intersections))


# plt.imshow(np.rot90(fited_intersections, 2), cmap='Greys', origin='lower', interpolation='none')
# plt.colorbar()
# plt.show()
# plt.close()
#
# res_intersections = fited_intersections - n_intesects_2d
# plt.imshow(np.rot90(res_intersections, 2), cmap='Greys', origin='lower', interpolation='none')
# plt.colorbar()
# plt.show()
# plt.close()

