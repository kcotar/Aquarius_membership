import numpy as np
import astropy.units as un

from astropy.table import Table
from galpy.potential import MWPotential2014
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel, estimateDeltaStaeckel

print 'Reading data'
tgas_data = Table.read('RAVE_GALAH_TGAS_stack.fits')

# remove units as they are later assigned to every attribute
tgas_data['RV'].unit = ''
print 'Creating orbits and computing orbital information'
# create new field in galah-tgas set

# add output columns to the dataset
# action angles
tgas_data['J_R'] = np.nan
tgas_data['L_Z'] = np.nan
tgas_data['J_Z'] = np.nan
# frequencies
tgas_data['Omega_R'] = np.nan
tgas_data['Omega_Phi'] = np.nan
tgas_data['Omega_Z'] = np.nan
# angles
tgas_data['Theta_R'] = np.nan
tgas_data['Theta_Phi'] = np.nan
tgas_data['Theta_Z'] = np.nan

ts = np.linspace(5, 13., 50) * un.Gyr
i_o = 0
print 'All objects: '+str(len(tgas_data))
for object in tgas_data:
    if i_o % 100 == 0:
        print i_o
    i_o += 1
    # create orbit object
    try:
        orbit = Orbit(vxvv=[object['ra_gaia'] * un.deg,
                            object['dec_gaia'] * un.deg,
                            1./object['parallax'] * un.kpc,
                            object['pmra'] * np.cos(np.deg2rad(object['dec_gaia'])) * un.mas/un.yr,
                            object['pmdec'] * un.mas/un.yr,
                            object['RV'] * un.km/un.s], radec=True)
        orbit.turn_physical_off()
        orbit.integrate(ts, MWPotential2014)
        delta_s = estimateDeltaStaeckel(MWPotential2014, orbit.R(ts), orbit.z(ts))
        action_staeckel = actionAngleStaeckel(pot=MWPotential2014, delta=delta_s, c=True)
        object_actions = action_staeckel.actionsFreqsAngles(orbit.R(), orbit.vR(), orbit.vT(), orbit.z(), orbit.vz(), orbit.phi(), fixed_quad=True)
        tgas_data['J_R'][i_o] = object_actions[0]
        tgas_data['L_Z'][i_o] = object_actions[1]
        tgas_data['J_Z'][i_o] = object_actions[2]
        tgas_data['Omega_R'][i_o] = object_actions[3]
        tgas_data['Omega_Phi'][i_o] = object_actions[4]
        tgas_data['Omega_Z'][i_o] = object_actions[5]
        tgas_data['Theta_R'][i_o] = object_actions[6]
        tgas_data['Theta_Phi'][i_o] = object_actions[7]
        tgas_data['Theta_Z'][i_o] = object_actions[8]
        # print tgas_data['J_R', 'L_Z', 'J_Z'][i_o]
    except:
        print 'Problem with object: '+str(i_o)

# save results
tgas_data.write('RAVE_GALAH_TGAS_stack_actions.fits')