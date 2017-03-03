import numpy as np

F = 212.  # (mas/yr)*pc/(km/s)


def compute_xyz_vel(ra, dec, rv):
    vx = np.cos(dec) * np.cos(ra) * rv
    vy = np.cos(dec) * np.sin(ra) * rv
    vz = np.ones(np.size(vx)) * np.sin(dec) * rv
    if np.size(vx) <= 1:
        return np.hstack((vx, vy, vz))
    else:
        return np.transpose(np.vstack((vx, vy, vz)))


def compute_rv(ra, dec, vel):
    unit_vect = compute_xyz_vel(ra, dec, 1.)
    return np.sum(unit_vect * vel, axis=1)


def compute_pmra(ra, dec, dist, vel):
    return F / dist * (-vel[0] * np.sin(ra) + vel[1] * np.cos(ra))


def compute_pmdec(ra, dec, dist, vel):
    return F / dist * (-vel[0] * np.cos(ra) * np.sin(dec) - vel[1] * np.sin(ra) * np.sin(dec) + vel[2] * np.cos(dec))
