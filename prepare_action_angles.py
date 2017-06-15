import numpy as np
import astropy.units as un
import os

from astropy.table import Table, join
from galpy.potential import MWPotential2014, LogarithmicHaloPotential
from galpy.orbit import Orbit
from galpy.actionAngle import actionAngleStaeckel, actionAngleAdiabatic, estimateDeltaStaeckel, actionAngleSpherical

print 'Reading data'
# OPTION1: ---------
out_dir = ''
data_file = 'RAVE_GALAH_TGAS_stack.fits'
tgas_data = Table.read(data_file)
rv_string= 'RV'
# -------------------

# OR GALAH only dataset
# OPTION2: ---------
# dir = '/home/klemen/GALAH_data/'
# out_dir = dir
# data_file = 'galah_tgas_xmatch.csv'
# galah_data = Table.read(dir+'sobject_iraf_52_reduced.csv')['sobject_id', 'rv_guess', 'teff_guess', 'feh_guess', 'logg_guess']
# tgas_xmatch_data = Table.read(dir+data_file)
# tgas_data = join(tgas_xmatch_data, galah_data, keys='sobject_id')
# rv_string = 'rv_guess'
# -------------------

# remove units
tgas_data[rv_string].unit = ''

print 'Creating orbits and computing orbital information'
# create new field in galah-tgas set

# add output columns to the dataset
new_cols = ['J_R', 'L_Z', 'J_Z', 'Omega_R', 'Omega_Phi', 'Omega_Z', 'Theta_R', 'Theta_Phi', 'Theta_Z']
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
                            object['pmra'] * un.mas/un.yr,
                            object['pmdec'] * un.mas/un.yr,
                            object[rv_string] * un.km/un.s], radec=True)
        orbit.turn_physical_off()

        # Staeckel model
        orbit.integrate(ts, MWPotential2014)
        delta_s = estimateDeltaStaeckel(MWPotential2014, orbit.R(ts), orbit.z(ts))
        action_staeckel = actionAngleStaeckel(pot=MWPotential2014, delta=delta_s, c=True)
        # actions, frequencies and angles
        object_actions = action_staeckel.actionsFreqsAngles(orbit.R(), orbit.vR(), orbit.vT(), orbit.z(), orbit.vz(), orbit.phi(), fixed_quad=True)
        # print object_actions
        # actions and frequencies
        # object_actions = action_staeckel.actionsFreqs(orbit.R(), orbit.vR(), orbit.vT(), orbit.z(), orbit.vz(), fixed_quad=True)

        # simpler model
        # action_spherical = actionAngleSpherical(pot=MWPotential2014)
        # object_actions = action_spherical.actionsFreqsAngles(orbit.R(), orbit.vR(), orbit.vT(), orbit.z(), orbit.vz(), orbit.phi(), fixed_quad=True)
        # print object_actions
        # action_adiabatic = actionAngleAdiabatic(pot=MWPotential2014, c=True)
        # object_actions = action_adiabatic(orbit.R(), orbit.vR(), orbit.vT(), orbit.z(), orbit.vz())

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
        # print ''
    except:
        print 'Problem with object: '+str(i_o)

# save results
out_fits_file = out_dir + data_file[:-5]+'_actions.fits'
if os.path.isfile(out_fits_file):
    # remove it
    os.remove(out_fits_file)
tgas_data.write(out_fits_file)