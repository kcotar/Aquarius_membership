import numpy as np
import astropy.coordinates as coord
import astropy.units as un
import astropy.constants as const
import matplotlib.pyplot as plt
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
from galpy.util import bovy_conversion
from galpy.actionAngle import estimateDeltaStaeckel
from galpy.actionAngle import actionAngleStaeckel
from time import time
from astropy.table import Table, join

data_dir = '/data4/cotar/'
gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV_Bdist.fits')

# clusters = Table.read(data_dir+'Open_cluster_members_DR2.fits')
# selected_objects = clusters[clusters['cluster'] == 'ASCC_105']['source_id']

selected_objects = [429968076498071424,552471646265579648,223202479644072576,302204142444316160,507514986631702656,
                    283866246999592576,429607544063400704,570143134586746496,452619051118116224,429471440832635776,
                    466280689250024064,405595928089960320,421499985836042496,415660956313404800,479765477806636032,
                    420200427506635136,431282787468706560,240491269198173568,330024466686913792,158265834005992960]
# selected_objects = [2007911710694905728,2162765176002359168,2171423009746515840,2202700782658624000]
# selected_objects = [3017250126427057024,3018575033640452736,3121604220965378944,3208961454882028416,3210151053446429056,3210775438611713024,4496352578733523584,4508720160407113984,4518269899935842176,4577886588615102592]
# selected_objects = [5970493171281327872,5970751453436281856,5971299220726443136,5974526190511181162241459989970403968,6020604685025643648,6030906146908479360]
gaia_subset = gaia_data[np.in1d(gaia_data['source_id'], selected_objects)]

# gaia_data = gaia_data[gaia_data['parallax_error']/gaia_data['parallax'] < 0.2]
# gaia_data = gaia_data[np.logical_and(gaia_data['parallax'] > 0, 1e3/gaia_data['parallax'] < 2500)]
# gaia_subset = gaia_data[:250]
# gaia_subset = gaia_subset[np.int64(np.random.choice(np.arange(len(gaia_subset)), 250, replace=False))]

ts0 = 0. * un.Myr
ts1 = np.linspace(0., -30., 5e4) * un.Myr
ts2 = np.linspace(0., 30., 5e4) * un.Myr

orbit_sun = Orbit(vxvv=[0. * un.deg,
                        0. * un.deg,
                        0.00001 * un.pc,
                        0. * un.mas / un.yr,
                        0. * un.mas / un.yr,
                        0. * un.km / un.s],
                  radec=True,
                  ro=8.2, vo=238., zo=0.025,
                  solarmotion=[-11., 10., 7.25])

fig, ax = plt.subplots(2, 2)

for star_data in gaia_subset:
    # print star_data['source_id']

    orbit = Orbit(vxvv=[np.float64(star_data['ra']) * un.deg,
                        np.float64(star_data['dec']) * un.deg,
                        # 1e3 / np.float64(star_data['parallax']) * un.pc,
                        np.float64(star_data['r_est']) * un.pc,
                        np.float64(star_data['pmra']) * un.mas / un.yr,
                        np.float64(star_data['pmdec']) * un.mas / un.yr,
                        np.float64(star_data['rv']) * un.km / un.s],
                  radec=True,
                  ro=8.2, vo=238., zo=0.025,
                  solarmotion=[-11., 10., 7.25])  # as used by JBH in his paper on forced oscillations and phase mixing
    orbit.turn_physical_on()

    orbit.integrate(ts1, MWPotential2014)
    orbit_ts0 = [orbit.x(ts0) * 1e3, orbit.y(ts0) * 1e3, orbit.z(ts0) * 1e3, orbit.R(ts0) * 1e3]
    orbit_xyz1 = [orbit.x(ts1) * 1e3, orbit.y(ts1) * 1e3, orbit.z(ts1) * 1e3, orbit.R(ts1) * 1e3]

    orbit.integrate(ts2, MWPotential2014)
    orbit_xyz2 = [orbit.x(ts2) * 1e3, orbit.y(ts2) * 1e3, orbit.z(ts2) * 1e3, orbit.R(ts2) * 1e3]

    orbit_xyz = [np.hstack((orbit_xyz1[0][:1:-1], orbit_xyz2[0])),
                 np.hstack((orbit_xyz1[1][:1:-1], orbit_xyz2[1])),
                 np.hstack((orbit_xyz1[2][:1:-1], orbit_xyz2[2])),
                 np.hstack((orbit_xyz1[3][:1:-1], orbit_xyz2[3]))]

    ax[0, 0].plot(orbit_xyz[0], orbit_xyz[1], lw=0.8, c='C0')
    ax[0, 0].scatter(orbit_ts0[0], orbit_ts0[1], lw=0, s=4, c='black')
    ax[1, 0].plot(orbit_xyz[0], orbit_xyz[2], lw=0.8, c='C0')
    ax[1, 0].scatter(orbit_ts0[0], orbit_ts0[2], lw=0, s=4, c='black')
    ax[0, 1].plot(orbit_xyz[2], orbit_xyz[1], lw=0.8, c='C0')
    ax[0, 1].scatter(orbit_ts0[2], orbit_ts0[1], lw=0, s=4, c='black')
    ax[1, 1].plot(orbit_xyz[3], orbit_xyz[2], lw=0.8, c='C0')
    ax[1, 1].scatter(orbit_ts0[3], orbit_ts0[2], lw=0, s=4, c='black')


    orbit = Orbit(vxvv=[np.float64(star_data['ra']) * un.deg,
                        np.float64(star_data['dec']) * un.deg,
                        1e3 / np.float64(star_data['parallax']) * un.pc,
                        # np.float64(star_data['r_est']) * un.pc,
                        np.float64(star_data['pmra']) * un.mas / un.yr,
                        np.float64(star_data['pmdec']) * un.mas / un.yr,
                        np.float64(star_data['rv']) * un.km / un.s],
                  radec=True,
                  ro=8.2, vo=238., zo=0.025,
                  solarmotion=[-11., 10., 7.25])  # as used by JBH in his paper on forced oscillations and phase mixing
    orbit.turn_physical_on()

    orbit.integrate(ts1, MWPotential2014)
    orbit_ts0 = [orbit.x(ts0) * 1e3, orbit.y(ts0) * 1e3, orbit.z(ts0) * 1e3, orbit.R(ts0) * 1e3]
    orbit_xyz1 = [orbit.x(ts1) * 1e3, orbit.y(ts1) * 1e3, orbit.z(ts1) * 1e3, orbit.R(ts1) * 1e3]

    orbit.integrate(ts2, MWPotential2014)
    orbit_xyz2 = [orbit.x(ts2) * 1e3, orbit.y(ts2) * 1e3, orbit.z(ts2) * 1e3, orbit.R(ts2) * 1e3]

    orbit_xyz = [np.hstack((orbit_xyz1[0][:1:-1], orbit_xyz2[0])),
                 np.hstack((orbit_xyz1[1][:1:-1], orbit_xyz2[1])),
                 np.hstack((orbit_xyz1[2][:1:-1], orbit_xyz2[2])),
                 np.hstack((orbit_xyz1[3][:1:-1], orbit_xyz2[3]))]

    ax[0, 0].plot(orbit_xyz[0], orbit_xyz[1], lw=0.8, c='C3')
    ax[0, 0].scatter(orbit_ts0[0], orbit_ts0[1], lw=0, s=4, c='black')
    ax[1, 0].plot(orbit_xyz[0], orbit_xyz[2], lw=0.8, c='C3')
    ax[1, 0].scatter(orbit_ts0[0], orbit_ts0[2], lw=0, s=4, c='black')
    ax[0, 1].plot(orbit_xyz[2], orbit_xyz[1], lw=0.8, c='C3')
    ax[0, 1].scatter(orbit_ts0[2], orbit_ts0[1], lw=0, s=4, c='black')
    ax[1, 1].plot(orbit_xyz[3], orbit_xyz[2], lw=0.8, c='C3')
    ax[1, 1].scatter(orbit_ts0[3], orbit_ts0[2], lw=0, s=4, c='black')


ax[0, 0].set(xlabel='X', ylabel='Y')
ax[1, 0].set(xlabel='X', ylabel='Z')
ax[0, 1].set(xlabel='Z', ylabel='Y')
ax[1, 1].set(xlabel='R', ylabel='Z')
plt.show()
# plt.savefig('orbits_-10_10_comb.png', dpi=350)
plt.close()




