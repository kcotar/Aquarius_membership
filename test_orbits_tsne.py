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

data_dir = '/data4/cotar/'
gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')
idx_ok = gaia_data['parallax_error']/gaia_data['parallax'] < 0.2
idx_ok = np.logical_and(idx_ok, gaia_data['parallax'] > 0)
idx_ok = np.logical_and(idx_ok, 1e3/gaia_data['parallax'] < 2500)
gaia_data = gaia_data[idx_ok]

n_all = len(gaia_data)
print 'Stars left:', n_all

ts1 = np.linspace(0., -30., 1e4) * un.Myr
ts2 = np.linspace(0., 30., 1e4) * un.Myr
ts_total = np.hstack((ts1[:1:-1], ts2))

empty_vals = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
empty_vals_actions = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]


# selected_objects = [3117420201264574080,3117533833218999552,3117654745137604608,3117293624287774720,3117405327797423488,
#                     3117398180971951872,3117626329634066432,3117264418510436480,3117499091224753536,3117232498313708032,
#                     3117286786699953536,3117303829129913344,3117284102343625216,3117265204484915968,3117208687015334784]
# gaia_data = gaia_data[np.in1d(gaia_data['source_id'], selected_objects)]
gaia_data = gaia_data[:800000]

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
                         1e3 / np.float64(star_data['parallax']) * un.pc,
                         np.float64(star_data['pmra']) * un.mas / un.yr,
                         np.float64(star_data['pmdec']) * un.mas / un.yr,
                         np.float64(star_data['rv']) * un.km / un.s],
                   radec=True,
                   ro=8.2, vo=238., zo=0.025,
                   solarmotion=[-11., 10., 7.25])  # as used by JBH in his paper on forced oscillations and phase mixing

    orbit2 = Orbit(vxvv=[np.float64(star_data['ra']) * un.deg,
                         np.float64(star_data['dec']) * un.deg,
                         1e3 / np.float64(star_data['parallax']) * un.pc,
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

    try:
        ts_get = np.interp(0., orbit_y, ts_total) * un.Myr
    except:
        print 'Unable to determine ts_get'
        orbit_xyz = empty_vals

    try:
        if ts_get < 0.:
            orbit_xyz = [orbit1.x(ts_get) * 1e3, orbit1.y(ts_get) * 1e3, orbit1.z(ts_get) * 1e3,  # in pc
                         orbit1.vx(ts_get), orbit1.vy(ts_get), orbit1.vz(ts_get),
                         orbit1.U(ts_get), orbit1.V(ts_get), orbit1.W(ts_get)]  # in km/s
        else:
            orbit_xyz = [orbit2.x(ts_get) * 1e3, orbit2.y(ts_get) * 1e3, orbit2.z(ts_get) * 1e3,
                         orbit2.vx(ts_get), orbit2.vy(ts_get), orbit2.vz(ts_get),
                         orbit2.U(ts_get), orbit2.V(ts_get), orbit2.W(ts_get)]
        if np.abs(orbit_xyz[1]) > 1e-7:
            print 'Orbit not crossing Y=0'
            orbit_xyz = empty_vals
        else:
            orbit_xyz.append(orbit2.e(analytic=True, pot=MWPotential2014))
            orbit_xyz.append(orbit2.zmax(analytic=True, pot=MWPotential2014))
            orbit_xyz.append(orbit2.rap(analytic=True, pot=MWPotential2014))
            orbit_xyz.append(orbit2.rperi(analytic=True, pot=MWPotential2014))
    except:
        print 'Error in orbit ts_get'
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
        print 'Error in orbit action estimation'
        object_actions = empty_vals_actions

    for o_v in orbit_xyz:
        orbit_vals_all.append(o_v)
    for o_v in object_actions:
        orbit_vals_all.append(o_v)

    return orbit_vals_all


t_s = time()
orbits_all = Parallel(n_jobs=35)(delayed(get_orbital_data)(i_row) for i_row in range(len(gaia_data)))

# print orbits_all
orbits_all = np.vstack(orbits_all)
# print orbits_all

d_out = gaia_data['source_id', 'ra', 'dec', 'pmra', 'pmra_error', 'pmdec_error', 'parallax', 'parallax_error', 'rv', 'rv_error']

for i_c, c_new in enumerate(['X', 'Y', 'Z', 'VX', 'VY', 'VZ', 'U', 'V', 'W', 'e', 'zmax', 'Rap', 'Rper', 'J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z', 'Theta_R', 'Theta_Phi', 'Theta_Z']):
    d_out[c_new] = orbits_all[:, i_c]

out_dir = '/data4/cotar/'
d_out.write(out_dir+'Gaia_dr2_orbital_derivatives_actions_1.fits', overwrite=True)
