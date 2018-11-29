import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from time import time
from astropy.table import Table
from joblib import Parallel, delayed

data_dir = '/data4/cotar/'
gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')
idx_ok = gaia_data['parallax_error']/gaia_data['parallax'] < 0.2
idx_ok = np.logical_and(idx_ok, gaia_data['parallax'] > 0)
idx_ok = np.logical_and(idx_ok, 1e3/gaia_data['parallax'] < 2500)
gaia_data = gaia_data[idx_ok]

print 'Stars left:', len(gaia_data)

ts1 = np.linspace(0., -30., 1e4) * un.Myr
ts2 = np.linspace(0., 30., 1e4) * un.Myr
ts_total = np.hstack((ts1[:1:-1], ts2))

empty_vals = [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]

# selected_objects = [3117420201264574080,3117533833218999552,3117654745137604608,3117293624287774720,3117405327797423488,
#                     3117398180971951872,3117626329634066432,3117264418510436480,3117499091224753536,3117232498313708032,
#                     3117286786699953536,3117303829129913344,3117284102343625216,3117265204484915968,3117208687015334784]
# gaia_data = gaia_data[np.in1d(gaia_data['source_id'], selected_objects)]

# orbits_all = list([])
# for id_star in range(len(gaia_data)):
def get_orbital_data(id_star):
    star_data = gaia_data[id_star]
    print star_data['source_id']

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
        return empty_vals
    try:
        if ts_get < 0.:
            orbit_xyz = [orbit1.x(ts_get) * 1e3, orbit1.y(ts_get) * 1e3, orbit1.z(ts_get) * 1e3,  # in pc
                         orbit1.vx(ts_get), orbit1.vy(ts_get), orbit1.vz(ts_get)]  # in km/s
        else:
            orbit_xyz = [orbit2.x(ts_get) * 1e3, orbit2.y(ts_get) * 1e3, orbit2.z(ts_get) * 1e3,
                         orbit2.vx(ts_get), orbit2.vy(ts_get), orbit2.vz(ts_get)]
        if np.abs(orbit_xyz[1]) > 1e-7:
            print 'Orbit not crossing Y=0'
            return empty_vals
        return orbit_xyz
    except:
        print 'Error in orbit ts_get'
        return empty_vals


orbits_all = Parallel(n_jobs=142)(delayed(get_orbital_data)(i_row) for i_row in range(len(gaia_data)))

# print orbits_all
orbits_all = np.vstack(orbits_all)
# print orbits_all

# vel = np.sqrt(np.sum(orbits_all[:, -3:]**2, axis=1))
# for iv in [3, 4, 5]:
#     vel = orbits_all[:, iv]
#     plt.scatter(orbits_all[:, 0], orbits_all[:, 2], vmin=np.nanpercentile(vel, 2), vmax=np.nanpercentile(vel, 98),
#                 cmap='viridis', lw=0, c=vel, s=1)
#     plt.colorbar()
#     plt.show()
#     plt.close()

d_out = gaia_data['source_id', 'ra', 'dec', 'pmra', 'pmra_error', 'pmdec_error', 'parallax', 'parallax_error', 'rv', 'rv_error']

for i_c, c_new in enumerate(['X', 'Y', 'Z', 'VX', 'VY', 'VZ']):
    d_out[c_new] = orbits_all[:, i_c]

out_dir = '/data4/cotar/'
d_out.write(out_dir+'Gaia_dr2_orbital_derivatives.fits', overwrite=True)
