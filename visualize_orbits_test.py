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
gaia_data = Table.read(data_dir + 'Gaia_DR2_RV/GaiaSource_combined_RV.fits')

# clusters = Table.read(data_dir+'Open_cluster_members_DR2.fits')
# selected_objects = clusters[clusters['cluster'] == 'ASCC_105']['source_id']

selected_objects = [1584855752702115584,1585161142056865792,1585581013763892608,1599170638182153344,1601407388430279552,1606891889869191296,1608412789327913088,1613431437137809792,1622667270386650880,1626000302447696512,1626626577398964864,1627096996577394816,1627813190963893888,1633652967801335168,1635701980734926336,1636102958881605632,1637192952862070144,1642683840915751936,1647154425129739136,1647550730351839232,1647901783798855680,1648224078143868672,1662578850984726272,1662715503959596032,1677492970300907648,1680114760073513984,1680691419562252160,1681675963505181184,1682129615130992256,1686846588733634304,1690210510199277184,1691646090132093824,1695631961517553280,1695939687333571712,1696437899244881920,1697407844595341568,1703781266825101440,1707783209976268928,1708072317110964864,1709881735293952384,1712976070251310720,1713053860699038080,1718579078426160128]
# selected_objects = [2007911710694905728,2162765176002359168,2171423009746515840,2202700782658624000]
# selected_objects = [3017250126427057024,3018575033640452736,3121604220965378944,3208961454882028416,3210151053446429056,3210775438611713024,4496352578733523584,4508720160407113984,4518269899935842176,4577886588615102592]
# selected_objects = [5970493171281327872,5970751453436281856,5971299220726443136,5971459989970403968,6020604685025643648,6030906146908479360]
gaia_subset = gaia_data[np.in1d(gaia_data['source_id'], selected_objects)]

# gaia_data = gaia_data[gaia_data['parallax_error']/gaia_data['parallax'] < 0.2]
# gaia_data = gaia_data[np.logical_and(gaia_data['parallax'] > 0, 1e3/gaia_data['parallax'] < 2500)]
# gaia_subset = gaia_data[:250]
# gaia_subset = gaia_subset[np.int64(np.random.choice(np.arange(len(gaia_subset)), 250, replace=False))]

ts0 = 0. * un.Myr
ts1 = np.linspace(0., -50., 1e3) * un.Myr
ts2 = np.linspace(0., 50., 1e3) * un.Myr

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
                        1e3 / np.float64(star_data['parallax']) * un.pc,
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

    ax[0, 0].plot(orbit_xyz[0], orbit_xyz[1], lw=0.8)
    ax[0, 0].scatter(orbit_ts0[0], orbit_ts0[1], lw=0, s=4, c='black')
    ax[1, 0].plot(orbit_xyz[0], orbit_xyz[2], lw=0.8)
    ax[1, 0].scatter(orbit_ts0[0], orbit_ts0[2], lw=0, s=4, c='black')
    ax[0, 1].plot(orbit_xyz[2], orbit_xyz[1], lw=0.8)
    ax[0, 1].scatter(orbit_ts0[2], orbit_ts0[1], lw=0, s=4, c='black')
    ax[1, 1].plot(orbit_xyz[3], orbit_xyz[2], lw=0.8)
    ax[1, 1].scatter(orbit_ts0[3], orbit_ts0[2], lw=0, s=4, c='black')


ax[0, 0].set(xlabel='X', ylabel='Y')
ax[1, 0].set(xlabel='X', ylabel='Z')
ax[0, 1].set(xlabel='Z', ylabel='Y')
ax[1, 1].set(xlabel='R', ylabel='Z')
plt.show()
# plt.savefig('orbits_-10_10_comb.png', dpi=350)
plt.close()




