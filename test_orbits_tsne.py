import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.actionAngle import actionAngleStaeckel, actionAngleAdiabatic, estimateDeltaStaeckel, actionAngleSpherical

from time import time
from datetime import timedelta
from astropy.table import Table
from joblib import Parallel, delayed
from sys import argv
from scipy import interpolate

data_dir = '/shared/ebla/cotar/'
data_dir_out = '/shared/data-camelot/cotar/'
gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV_Bdist.fits')

# remove units as they are used later in the code
for col in gaia_data.colnames:
    gaia_data[col].unit = None

# idx_ok = gaia_data['parallax_error']/gaia_data['parallax'] < 0.2
# idx_ok = np.logical_and(idx_ok, gaia_data['parallax'] > 0)
# idx_ok = np.logical_and(idx_ok, 1e3/gaia_data['parallax'] < 3000)
# gaia_data = gaia_data[idx_ok]

idx_ok = gaia_data['r_est'] < 3000
gaia_data = gaia_data[idx_ok]

ts1 = np.linspace(0., -45., 2e4) * un.Myr
ts2 = np.linspace(0., 45., 2e4) * un.Myr
ts_total = np.array(np.hstack((ts1[:1:-1], ts2)))
n_ts_total = len(ts_total)
n_ts_total_4 = int(n_ts_total/4)

empty_vals = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
empty_vals_actions = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

# selected_objects = [361891062438756992]
# gaia_data = gaia_data[np.in1d(gaia_data['source_id'], selected_objects)]

# get arguments
if len(argv) >= 4:
    o_s = '_'+str(argv[3])
    n_s = long(argv[1])
    n_e = long(argv[2])
else:
    o_s = ''
    n_s = 0
    n_e = 6000000

n_beg = len(gaia_data)
gaia_data = gaia_data[n_s:n_e]

n_all = len(gaia_data)
print 'Stars left:', n_all, 'from initial:', n_beg
print n_s, n_e, o_s

# orbits_all = list([])
# for id_star in range(len(gaia_data)):
def get_orbital_data(id_star):
    star_data = gaia_data[id_star]
    if id_star % 1000 == 0:
        t_n = time()
        t_d = (t_n - t_s) / (id_star+1.)
        str_time = 'speed {:.3f} s/obj, remaining '.format(t_d) + str(timedelta(seconds=t_d*(n_all-id_star-1)))
        print id_star, star_data['source_id'], str_time
    orbit_vals_all = []

    orbit1 = Orbit(vxvv=[np.float64(star_data['ra']) * un.deg,
                         np.float64(star_data['dec']) * un.deg,
                         # 1e3 / np.float64(star_data['parallax']) * un.pc,
                         np.float64(star_data['r_est']) * un.pc,
                         np.float64(star_data['pmra']) * un.mas / un.yr,
                         np.float64(star_data['pmdec']) * un.mas / un.yr,
                         np.float64(star_data['rv']) * un.km / un.s],
                   radec=True,
                   ro=8.2, vo=238., zo=0.025,
                   solarmotion=[-11., 10., 7.25])  # as used by JBH in his paper on forced oscillations and phase mixing

    orbit2 = Orbit(vxvv=[np.float64(star_data['ra']) * un.deg,
                         np.float64(star_data['dec']) * un.deg,
                         # 1e3 / np.float64(star_data['parallax']) * un.pc,
                         np.float64(star_data['r_est']) * un.pc,
                         np.float64(star_data['pmra']) * un.mas / un.yr,
                         np.float64(star_data['pmdec']) * un.mas / un.yr,
                         np.float64(star_data['rv']) * un.km / un.s],
                   radec=True,
                   ro=8.2, vo=238., zo=0.025,
                   solarmotion=[-11., 10., 7.25])

    orbit1.turn_physical_on()
    orbit2.turn_physical_on()

    orbit1.integrate(ts1, MWPotential2014)
    orbit2.integrate(ts2, MWPotential2014)

    orbit_y = np.hstack((orbit1.y(ts1)[:1:-1], orbit2.y(ts2)))

    i_cross = np.nanargmin(np.abs(orbit_y - 0.))
    i_cross_s = max(i_cross - 100, 0)
    i_cross_e = min(i_cross + 100, n_ts_total-1)
    mean_y = np.abs(np.mean(orbit_y[i_cross_s:i_cross_e]))
    # print mean_y, i_cross_s, i_cross_e

    if mean_y < 0.1:
        ts_get = interpolate.interp1d(orbit_y[i_cross_s:i_cross_e],
                                      ts_total[i_cross_s:i_cross_e],
                                      bounds_error=False, fill_value=np.nan, assume_sorted=False)(0.) * un.Myr
    else:
        print 'Unable to determine ts_get (mean_y = '+str(mean_y)+') -> '+str(star_data['source_id'])
        ts_get = np.nan

    # plt.plot(orbit_y, ts_total)
    # plt.plot(orbit_y[i_cross_s:i_cross_e], ts_total[i_cross_s:i_cross_e])
    # plt.axvline(0.)
    # try:
    #     plt.axhline(ts_get.value)
    # except:
    #     pass
    # plt.show()
    # plt.close()
    # print ts_get

    if np.isfinite(ts_get):
        try:
            if ts_get < 0.:
                orbit_xyz = [orbit1.x(ts_get) * 1e3, orbit1.y(ts_get) * 1e3, orbit1.z(ts_get) * 1e3,  # in pc
                             orbit1.vx(ts_get), orbit1.vy(ts_get), orbit1.vz(ts_get),  # in km/s
                             orbit1.U(ts_get), orbit1.V(ts_get), orbit1.W(ts_get)]  # in km/s
            else:
                orbit_xyz = [orbit2.x(ts_get) * 1e3, orbit2.y(ts_get) * 1e3, orbit2.z(ts_get) * 1e3,
                             orbit2.vx(ts_get), orbit2.vy(ts_get), orbit2.vz(ts_get),
                             orbit2.U(ts_get), orbit2.V(ts_get), orbit2.W(ts_get)]
            if np.abs(orbit_xyz[1]) > 1e-5:
                print 'Orbit not crossing Y=0 -> '+str(star_data['source_id'])
                orbit_xyz = empty_vals
            else:
                orbit_xyz.append(orbit2.e(analytic=True, pot=MWPotential2014))
                orbit_xyz.append(orbit2.zmax(analytic=True, pot=MWPotential2014))
                orbit_xyz.append(orbit2.rap(analytic=True, pot=MWPotential2014))
                orbit_xyz.append(orbit2.rperi(analytic=True, pot=MWPotential2014))
        except:
            print 'Error in orbit ts_get -> '+str(star_data['source_id'])
            orbit_xyz = empty_vals

        try:
            orbit2.turn_physical_off()
            # Staeckel model
            delta_s = estimateDeltaStaeckel(MWPotential2014, orbit2.R(ts2), orbit2.z(ts2))
            action_staeckel = actionAngleStaeckel(pot=MWPotential2014, delta=delta_s, c=True)
            # actions, frequencies and angles
            object_actions = action_staeckel.actionsFreqsAngles(orbit2.R(), orbit2.vR(), orbit2.vT(), orbit2.z(),
                                                                orbit2.vz(), orbit2.phi(), fixed_quad=True)
        except:
            print 'Error in orbit action estimation -> '+str(star_data['source_id'])
            object_actions = empty_vals_actions
    else:
        orbit_xyz = empty_vals
        object_actions = empty_vals_actions

    for o_v in orbit_xyz:
        orbit_vals_all.append(o_v)
    for o_v in object_actions:
        orbit_vals_all.append(o_v)

    return orbit_vals_all

t_s = time()
orbits_all = Parallel(n_jobs=18)(delayed(get_orbital_data)(i_row) for i_row in range(len(gaia_data)))

# print orbits_all
orbits_all = np.vstack(orbits_all)
# print orbits_all

d_out = gaia_data['source_id', 'ra', 'dec', 'pmra', 'pmra_error', 'pmdec_error', 'parallax', 'parallax_error', 'rv', 'rv_error', 'r_est', 'r_lo', 'r_hi']

for i_c, c_new in enumerate(['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'U', 'V', 'W', 'e', 'zmax', 'Rap', 'Rper', 'J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z', 'Theta_R', 'Theta_Phi', 'Theta_Z']):
    d_out[c_new] = orbits_all[:, i_c]

d_out.write(data_dir_out+'Gaia_dr2_orbital_derivatives_actions_Bdist'+o_s+'.fits', overwrite=True)
